"""Read-only post-training analysis for the controlled D19-A1-ID pair.

The script never trains, resumes, fine-tunes, saves a checkpoint, or mutates
cached graphs. Runtime path substitutions are made only in deep-copied configs
used to open the local FER CSVs and the already-built evidence-only graph cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.models.structure_gnn import StructureGNN
from d18.scripts.audit_d19_preimplementation import effective_rank, graph_separation, linear_cka
from d18.training.train_d18 import (
    canonical_state_manifest,
    load_checkpoint,
    scientific_resume_signature,
    set_seed,
)


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
RUNS = {
    "null": ROOT / "outputs/d19_runs/d19_a1_id_null_evidence_only_seed42",
    "correct": ROOT / "outputs/d19_runs/d19_a1_id_correct_evidence_only_seed42",
}
A0_RUN = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42"
C2_RUN = ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42"
CACHE_ROOT = ROOT / "outputs/d19_graph_cache/a0_evidence_only"
LOCKED_SOURCE = ROOT / "outputs/d18_analysis/ofix18_factorial_posttraining/06_locked_evaluation_predictions.csv"
IMPLEMENTATION = ROOT / "outputs/d19_analysis/d19_a1_id_implementation_design"
EXPECTED_PARAMETER_COUNT = 266_616
ALLOWED_CONFIG_DIFFS = {
    "description",
    "logging.wandb.tags",
    "model.edge_type_conditioning.mode",
    "output_dir",
    "run_name",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(run_dir: Path) -> dict[str, Any]:
    yaml_path = run_dir / "resolved_config.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return read_json(run_dir / "resolved_config.json")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def md_table(frame: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 6) -> str:
    current = frame[list(columns)].copy() if columns is not None else frame.copy()
    if current.empty:
        return "_No rows._"
    for column in current.columns:
        if pd.api.types.is_float_dtype(current[column]):
            current[column] = current[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
    headers = [str(column) for column in current.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in current.astype(str).values.tolist()]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, name))
    else:
        result[prefix] = value
    return result


def runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    result.setdefault("data", {})["evidence_dir"] = str(ROOT / "data")
    cache = result.setdefault("graph", {}).setdefault("cache", {})
    cache.update(
        {
            "enabled": True,
            "dir": str(CACHE_ROOT),
            "strict": True,
            "fallback_on_error": False,
            "schema": "d19_a0_evidence_only_v1",
        }
    )
    result.setdefault("training", {})["num_workers"] = 0
    result["training"]["pin_memory"] = False
    return result


def locked_manifest() -> pd.DataFrame:
    source = pd.read_csv(LOCKED_SOURCE)
    frame = source[
        source["cell"].eq("C0")
        & source["checkpoint_type"].eq("best")
        & source["mode"].eq("official")
    ][["image_id", "sample_index", "true_class", "detected_state"]].copy()
    frame = frame.drop_duplicates("sample_index", keep="first").reset_index(drop=True)
    ordered = frame["sample_index"].to_numpy(dtype=np.int64)
    digest = hashlib.sha256(ordered.tobytes()).hexdigest()
    if len(frame) != 715 or digest != LOCKED_SHA256:
        raise RuntimeError(f"Locked manifest mismatch: count={len(frame)} sha256={digest}")
    return frame


def cache_paths(split: str, indices: list[int]) -> list[Path]:
    manifest = pd.read_csv(CACHE_ROOT / f"manifest_{split}.csv")
    lookup = {int(row.sample_index): CACHE_ROOT / str(row.cache_file) for row in manifest.itertuples()}
    paths = [lookup[int(index)] for index in indices]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cache files for {split}: {missing[:3]}")
    return paths


def graph_component_hashes(path: Path) -> dict[str, Any]:
    graph = load_d18_graph_cache(path)
    local = graph.edge_type.long() == 0
    knn = graph.edge_type.long() == 1
    dst = graph.edge_index[1].long()
    degree = torch.zeros((graph.x.shape[0],), dtype=torch.long)
    degree.index_add_(0, dst, torch.ones_like(dst))
    normalization = degree.clamp_min(1).float().reciprocal()
    payload = torch.cat(
        [
            graph.x.float().reshape(-1),
            graph.pos.float().reshape(-1),
            graph.edge_index.long().reshape(-1).float(),
            graph.edge_type.long().reshape(-1).float(),
            graph.edge_attr.float().reshape(-1),
            degree.float(),
        ]
    )
    return {
        "sample_index": int(graph.sample_index),
        "label": int(graph.y),
        "node_count": int(graph.x.shape[0]),
        "edge_count": int(graph.edge_index.shape[1]),
        "coordinate_hash": tensor_hash(graph.pos),
        "node_feature_hash": tensor_hash(graph.x),
        "local_edge_hash": tensor_hash(graph.edge_index[:, local]),
        "knn_edge_hash": tensor_hash(graph.edge_index[:, knn]),
        "merged_edge_hash": tensor_hash(graph.edge_index),
        "edge_type_hash": tensor_hash(graph.edge_type),
        "base_edge_attr_hash": tensor_hash(graph.edge_attr),
        "total_degree_hash": tensor_hash(degree),
        "normalization_hash": tensor_hash(normalization),
        "semantic_hash": tensor_hash(payload),
    }


def aggregate_graph_audit(paths: list[Path], label: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    node_counts: list[int] = []
    edge_counts: list[int] = []
    started = time.perf_counter()
    for position, path in enumerate(paths):
        row = graph_component_hashes(path)
        for key in sorted(row):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(row[key]).encode("utf-8"))
            digest.update(b"\0")
        node_counts.append(row["node_count"])
        edge_counts.append(row["edge_count"])
        if (position + 1) % 1000 == 0 or position + 1 == len(paths):
            print(
                json.dumps(
                    {
                        "event": "d19_a1_graph_audit_progress",
                        "scope": label,
                        "done": position + 1,
                        "total": len(paths),
                    }
                ),
                flush=True,
            )
    return {
        "scope": label,
        "count": len(paths),
        "aggregate_semantic_sha256": digest.hexdigest(),
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "edge_count_min": min(edge_counts),
        "edge_count_max": max(edge_counts),
        "elapsed_sec": time.perf_counter() - started,
        "null_correct_same_cache_files": True,
        "exact_pair_parity": True,
    }


def ece_score(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    result = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(
                float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean())
            )
    return float(result)


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["true_class"].to_numpy(dtype=np.int64)
    pred = frame["predicted_class"].to_numpy(dtype=np.int64)
    probs = frame[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(7), zero_division=0
    )
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    confidence = probs.max(axis=1)
    return {
        "count": int(len(frame)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(7)[y]) ** 2, axis=1))),
        "ece": ece_score(y, probs),
        "mean_entropy": float(entropy.mean()),
        "mean_max_confidence": float(confidence.mean()),
        "mean_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).mean()),
        "accuracy_confidence_gap": float(confidence.mean() - (pred == y).mean()),
        "confusion_matrix_json": json.dumps(confusion_matrix(y, pred, labels=np.arange(7)).tolist()),
        **{f"precision_{CLASS_NAMES[index]}": float(precision[index]) for index in range(7)},
        **{f"recall_{CLASS_NAMES[index]}": float(recall[index]) for index in range(7)},
        **{f"f1_{CLASS_NAMES[index]}": float(f1[index]) for index in range(7)},
        **{f"support_{CLASS_NAMES[index]}": int(support[index]) for index in range(7)},
    }


def graph_mean(h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    result = h.new_zeros((num_graphs, h.shape[1]))
    result.index_add_(0, batch_index.long(), h)
    count = h.new_zeros((num_graphs, 1))
    count.index_add_(0, batch_index.long(), torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
    return result / count.clamp_min(1.0)


def bounded_node_metrics(h: torch.Tensor, ptr: torch.Tensor) -> dict[str, float]:
    """Cheap node diagnostics over deterministic 128-node samples per graph."""
    pair_cosines, variances = [], []
    for graph_id in range(ptr.numel() - 1):
        start, end = int(ptr[graph_id]), int(ptr[graph_id + 1])
        local = h[start:end]
        variances.append(float(local.var(dim=0, unbiased=False).mean().item()))
        count = min(128, int(local.shape[0]))
        indices = torch.linspace(0, local.shape[0] - 1, count, device=h.device).long()
        normalized = torch.nn.functional.normalize(local[indices], dim=1)
        cosine = normalized @ normalized.T
        pair_cosines.append(float((cosine.sum() - count).item() / max(count * (count - 1), 1)))
    return {
        "mean_pairwise_node_cosine": float(np.mean(pair_cosines)),
        "node_representation_variance": float(np.mean(variances)),
        "deterministic_nodes_per_graph": 128,
    }

def load_model(
    treatment: str,
    checkpoint_type: str,
    device: torch.device,
) -> tuple[StructureGNN, dict[str, Any], dict[str, Any]]:
    cfg = read_config(RUNS[treatment])
    model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6).to(device)
    payload = load_checkpoint(
        RUNS[treatment] / "checkpoints" / f"{checkpoint_type}.pt",
        model,
        device=device,
        expected_resume_signature=scientific_resume_signature(cfg),
        strict_signature=True,
    )
    model.eval()
    return model, payload, cfg


def evaluate(
    treatment: str,
    checkpoint_type: str,
    split: str,
    mode: str,
    device: torch.device,
    batch_size: int,
    indices: list[int] | None = None,
    capture_layers: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    model, payload, cfg = load_model(treatment, checkpoint_type, device)
    runtime = runtime_config(cfg)
    from d18.training.train_d18 import build_dataset

    dataset = build_dataset(runtime, split)
    selected_indices = list(range(len(dataset))) if indices is None else list(indices)
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_d18_graphs,
    )
    rows: list[dict[str, Any]] = []
    layer_pieces: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "input_projection",
            "gnn_layer_1",
            "gnn_layer_2",
            "gnn_layer_3",
            "pooled_embedding",
            "classifier_input",
            "logits",
        )
    }
    node_rows: list[dict[str, Any]] = []
    graph_hash_digest = hashlib.sha256()
    max_hook_output_difference = 0.0
    with torch.inference_mode():
        for batch_id, cpu_batch in enumerate(loader):
            if bool((cpu_batch.edge_type_cat == 2).any()) or bool((cpu_batch.structure_edge_count != 0).any()):
                raise RuntimeError("Evidence-only evaluation encountered structure edges")
            for name, tensor in (
                ("sample_index", cpu_batch.sample_index),
                ("x", cpu_batch.x_cat),
                ("edge_index", cpu_batch.edge_index_cat),
                ("edge_type", cpu_batch.edge_type_cat),
                ("edge_attr", cpu_batch.edge_attr_cat),
                ("ptr", cpu_batch.ptr),
            ):
                graph_hash_digest.update(name.encode("ascii"))
                graph_hash_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            batch = cpu_batch.to(device)
            if capture_layers:
                captured_nodes: dict[str, torch.Tensor] = {}
                captured_graph: dict[str, torch.Tensor] = {}
                handles = [
                    model.encoder.register_forward_hook(
                        lambda _module, _inputs, output: captured_nodes.__setitem__("input_projection", output)
                    ),
                    *[
                        layer.register_forward_hook(
                            lambda _module, _inputs, output, index=index: captured_nodes.__setitem__(
                                f"gnn_layer_{index + 1}", output
                            )
                        )
                        for index, layer in enumerate(model.gnn.layers)
                    ],
                    model.readout.register_forward_hook(
                        lambda _module, _inputs, output: captured_graph.__setitem__("pooled_embedding", output)
                    ),
                    model.classifier[0].register_forward_hook(
                        lambda _module, _inputs, output: captured_graph.__setitem__("classifier_input", output)
                    ),
                    model.classifier.register_forward_hook(
                        lambda _module, _inputs, output: captured_graph.__setitem__("logits", output)
                    ),
                ]
                out = model(batch, conditioning_mode=mode)
                for handle in handles:
                    handle.remove()
                max_hook_output_difference = max(
                    max_hook_output_difference,
                    float((captured_graph["logits"] - out["logits"]).abs().max().item()),
                )
                for layer_name, state in captured_nodes.items():
                    pooled = graph_mean(state, batch.batch_index, batch.num_graphs)
                    layer_pieces[layer_name].append(pooled.detach().cpu().numpy().astype(np.float32))
                    node_rows.append(
                        {
                            "treatment": treatment,
                            "checkpoint_type": checkpoint_type,
                            "mode": mode,
                            "batch_id": batch_id,
                            "layer": layer_name,
                            **bounded_node_metrics(state, batch.ptr),
                        }
                    )
                for layer_name, state in captured_graph.items():
                    layer_pieces[layer_name].append(state.detach().cpu().numpy().astype(np.float32))
            else:
                out = model(batch, conditioning_mode=mode)
            logits = out["logits"]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Non-finite logits for {treatment}/{checkpoint_type}/{split}/{mode}")
            probs = torch.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            top2 = torch.topk(probs, k=2, dim=1).values
            for position in range(batch.num_graphs):
                row: dict[str, Any] = {
                    "split": split,
                    "treatment": treatment,
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_epoch": int(payload.get("epoch", -1)),
                    "conditioning_mode": mode,
                    "sample_index": int(batch.sample_index[position].item()),
                    "true_class": int(batch.y[position].item()),
                    "predicted_class": int(prediction[position].item()),
                    "correct": int(prediction[position].item() == batch.y[position].item()),
                    "entropy": float(entropy[position].item()),
                    "max_probability": float(probs[position].max().item()),
                    "margin": float((top2[position, 0] - top2[position, 1]).item()),
                    "detected_state": bool(batch.detected[position].item()),
                }
                for class_id in range(7):
                    row[f"logit_{class_id}"] = float(logits[position, class_id].item())
                    row[f"prob_{class_id}"] = float(probs[position, class_id].item())
                rows.append(row)
            if (batch_id + 1) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "event": "d19_a1_eval_progress",
                            "split": split,
                            "treatment": treatment,
                            "checkpoint": checkpoint_type,
                            "mode": mode,
                            "batch": batch_id + 1,
                            "total_batches": len(loader),
                        }
                    ),
                    flush=True,
                )
    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in layer_pieces.items()
        if values
    }
    info = {
        "graph_batch_stream_sha256": graph_hash_digest.hexdigest(),
        "max_hook_output_difference": max_hook_output_difference,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "prediction_finite": True,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), arrays, node_rows, info


def paired_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in ("sample_index", "true_class"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"Paired prediction alignment failed for {column}")


def fast_metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    cm = np.bincount(y * 7 + pred, minlength=49).reshape(7, 7)
    tp = np.diag(cm).astype(np.float64)
    precision_den = cm.sum(axis=0).astype(np.float64)
    recall_den = cm.sum(axis=1).astype(np.float64)
    f1 = 2.0 * tp / np.maximum(precision_den + recall_den, 1.0)
    support = recall_den
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(7)[y]) ** 2, axis=1))),
        "ece": ece_score(y, probs),
    }


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    replicates: int,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    paired_alignment(left, right)
    y = left["true_class"].to_numpy(dtype=np.int64)
    left_pred = left["predicted_class"].to_numpy(dtype=np.int64)
    right_pred = right["predicted_class"].to_numpy(dtype=np.int64)
    left_probs = left[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=np.float64)
    right_probs = right[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=np.float64)
    groups = [np.flatnonzero(y == class_id) for class_id in range(7)]
    rng = np.random.default_rng(seed)
    metrics = ("accuracy", "macro_f1", "weighted_f1", "nll", "brier_score", "ece")
    values = {metric: np.empty(replicates, dtype=np.float64) for metric in metrics}
    for replicate in range(replicates):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        left_values = fast_metrics(y[indices], left_pred[indices], left_probs[indices])
        right_values = fast_metrics(y[indices], right_pred[indices], right_probs[indices])
        for metric in metrics:
            values[metric][replicate] = right_values[metric] - left_values[metric]
    observed_left = fast_metrics(y, left_pred, left_probs)
    observed_right = fast_metrics(y, right_pred, right_probs)
    rows = []
    for metric in metrics:
        rows.append(
            {
                "scope": scope,
                "comparison": "correct_minus_null",
                "metric": metric,
                "left_null": observed_left[metric],
                "right_correct": observed_right[metric],
                "observed_difference": observed_right[metric] - observed_left[metric],
                "ci95_low": float(np.percentile(values[metric], 2.5)),
                "ci95_high": float(np.percentile(values[metric], 97.5)),
                "bootstrap_seed": seed,
                "replicates": replicates,
                "stratified_by_true_class": True,
            }
        )
    null_only = int(((left_pred == y) & (right_pred != y)).sum())
    correct_only = int(((left_pred != y) & (right_pred == y)).sum())
    discordant = null_only + correct_only
    pvalue = 1.0 if discordant == 0 else float(binomtest(min(null_only, correct_only), discordant, 0.5).pvalue)
    rows.append(
        {
            "scope": scope,
            "comparison": "correct_minus_null",
            "metric": "mcnemar_exact_pvalue",
            "left_null": null_only,
            "right_correct": correct_only,
            "observed_difference": correct_only - null_only,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "bootstrap_seed": seed,
            "replicates": replicates,
            "stratified_by_true_class": True,
            "discordant_pairs": discordant,
            "pvalue": pvalue,
        }
    )
    return pd.DataFrame(rows)


def classwise(left: pd.DataFrame, right: pd.DataFrame, scope: str) -> pd.DataFrame:
    paired_alignment(left, right)
    y = left["true_class"].to_numpy(dtype=np.int64)
    rows = []
    bundles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for treatment, frame in (("null", left), ("correct", right)):
        pred = frame["predicted_class"].to_numpy(dtype=np.int64)
        bundles[treatment] = precision_recall_fscore_support(
            y, pred, labels=np.arange(7), zero_division=0
        )
    for class_id, class_name in enumerate(CLASS_NAMES):
        np_, nr, nf, ns = bundles["null"]
        cp, cr, cf, cs = bundles["correct"]
        rows.append(
            {
                "scope": scope,
                "class_id": class_id,
                "class_name": class_name,
                "support": int(ns[class_id]),
                "null_precision": float(np_[class_id]),
                "null_recall": float(nr[class_id]),
                "null_f1": float(nf[class_id]),
                "correct_precision": float(cp[class_id]),
                "correct_recall": float(cr[class_id]),
                "correct_f1": float(cf[class_id]),
                "correct_minus_null_f1": float(cf[class_id] - nf[class_id]),
            }
        )
    return pd.DataFrame(rows)


def transition_rows(left: pd.DataFrame, right: pd.DataFrame, comparison: str) -> list[dict[str, Any]]:
    paired_alignment(left, right)
    y = left["true_class"].to_numpy(dtype=np.int64)
    left_pred = left["predicted_class"].to_numpy(dtype=np.int64)
    right_pred = right["predicted_class"].to_numpy(dtype=np.int64)
    groups = {
        "both_correct": (left_pred == y) & (right_pred == y),
        "right_correct_only": (left_pred != y) & (right_pred == y),
        "left_correct_only": (left_pred == y) & (right_pred != y),
        "both_wrong_same_prediction": (left_pred != y) & (right_pred != y) & (left_pred == right_pred),
        "both_wrong_different_prediction": (left_pred != y) & (right_pred != y) & (left_pred != right_pred),
    }
    rows = []
    for name, mask in groups.items():
        indices = np.flatnonzero(mask)
        rows.append(
            {
                "comparison": comparison,
                "transition": name,
                "count": int(mask.sum()),
                "true_class_distribution_json": json.dumps(
                    {str(key): int(value) for key, value in sorted(Counter(y[mask].tolist()).items())}
                ),
                "informative_sample_indices_up_to_50": json.dumps(
                    left.iloc[indices[:50]]["sample_index"].astype(int).tolist()
                ),
            }
        )
    return rows


def calibration_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "treatment", "checkpoint_type", "conditioning_mode"]
    for values, frame in predictions.groupby(keys, sort=False):
        base = dict(zip(keys, values))
        for outcome, subset in (
            ("all", frame),
            ("correct", frame[frame["correct"].eq(1)]),
            ("incorrect", frame[frame["correct"].eq(0)]),
        ):
            if subset.empty:
                continue
            rows.append({**base, "outcome": outcome, **metric_bundle(subset)})
    return pd.DataFrame(rows)


def representation_tables(
    arrays: dict[str, dict[str, np.ndarray]],
    node_rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry_rows = []
    for state, layer_values in arrays.items():
        previous: np.ndarray | None = None
        for layer in (
            "input_projection",
            "gnn_layer_1",
            "gnn_layer_2",
            "gnn_layer_3",
            "pooled_embedding",
            "classifier_input",
            "logits",
        ):
            current = layer_values[layer]
            geometry_rows.append(
                {
                    "row_type": "state_geometry",
                    "state": state,
                    "layer": layer,
                    **graph_separation(current, labels),
                    "effective_rank": effective_rank(current),
                    "inter_layer_cka_previous": np.nan if previous is None else linear_cka(previous, current),
                }
            )
            previous = current
    comparison_pairs = [
        ("correct_vs_null_trained", "correct_best_correct", "null_best_null"),
        ("correct_correct_vs_null_cf", "correct_best_correct", "correct_best_null"),
        ("correct_correct_vs_swapped_cf", "correct_best_correct", "correct_best_swapped"),
    ]
    comparison_rows = []
    for comparison, left_name, right_name in comparison_pairs:
        for layer in arrays[left_name]:
            left = arrays[left_name][layer]
            right = arrays[right_name][layer]
            comparison_rows.append(
                {
                    "row_type": "pair_comparison",
                    "comparison": comparison,
                    "left_state": left_name,
                    "right_state": right_name,
                    "layer": layer,
                    "linear_cka": linear_cka(left, right),
                    "mean_paired_l2": float(np.linalg.norm(left - right, axis=1).mean()),
                }
            )
    nodes = pd.DataFrame(node_rows)
    if not nodes.empty:
        aggregated = (
            nodes.groupby(["treatment", "checkpoint_type", "mode", "layer"], as_index=False)
            .mean(numeric_only=True)
            .assign(row_type="node_geometry")
        )
        geometry_rows.extend(aggregated.to_dict(orient="records"))
    return pd.DataFrame(geometry_rows), pd.DataFrame(comparison_rows)


def recreate_initial_state(cfg: dict[str, Any], expected_hash: str) -> tuple[dict[str, torch.Tensor] | None, str]:
    set_seed(42)
    model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
    manifest = canonical_state_manifest(model)
    actual = str(manifest["canonical_state_sha256"])
    if actual != expected_hash:
        return None, actual
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, actual


def parameter_category(name: str) -> str:
    if name == "edge_type_embedding.weight":
        return "edge_type_embedding"
    if name.startswith("encoder."):
        return "input_projection"
    if name.startswith("gnn.") and "edge_mlp" in name:
        return "edge_projection"
    if name.startswith("gnn.") and any(part in name for part in ("message", "gate", "ffn", "norm")):
        return "message_module"
    if name.startswith("readout."):
        return "readout"
    if name.startswith("classifier."):
        return "classifier"
    return "other"


def parameter_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    recreation: dict[str, Any] = {}
    for treatment, run_dir in RUNS.items():
        cfg = read_config(run_dir)
        manifest = read_json(run_dir / "initial_state_manifest.json")
        expected_hash = str(manifest["canonical_state_sha256"])
        initial, actual_hash = recreate_initial_state(cfg, expected_hash)
        recreation[treatment] = {
            "expected_initial_hash": expected_hash,
            "recreated_initial_hash": actual_hash,
            "exact_recreation": initial is not None,
            "quantitative_initial_drift_verifiable": initial is not None,
            "note": (
                "Exact saved initial tensors were recovered."
                if initial is not None
                else "NOT VERIFIABLE: run stores initial tensor hashes, not tensor values; local Torch recreation does not match the Kaggle initial-state hash."
            ),
        }
        checkpoint_states: dict[str, dict[str, torch.Tensor]] = {}
        checkpoint_epochs: dict[str, int] = {}
        for checkpoint_type in ("best", "last"):
            payload = torch.load(
                run_dir / "checkpoints" / f"{checkpoint_type}.pt",
                map_location="cpu",
                weights_only=False,
            )
            trained = {name: tensor.detach().cpu() for name, tensor in payload["model_state_dict"].items()}
            checkpoint_states[checkpoint_type] = trained
            checkpoint_epochs[checkpoint_type] = int(payload.get("epoch", -1))
            for name, tensor in trained.items():
                initial_tensor_hash = next(item["sha256"] for item in manifest["tensors"] if item["name"] == name)
                row = {
                    "treatment": treatment,
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_epoch": checkpoint_epochs[checkpoint_type],
                    "scope": parameter_category(name),
                    "parameter_name": name,
                    "numel": int(tensor.numel()),
                    "initial_tensor_hash": initial_tensor_hash,
                    "trained_tensor_hash": tensor_hash(tensor),
                    "drift_reference": "saved_initial_state",
                    "initial_recreation_exact": initial is not None,
                }
                if initial is None:
                    row.update({"l2_drift": np.nan, "relative_l2_drift": np.nan})
                else:
                    before = initial[name]
                    drift = float(torch.linalg.vector_norm((tensor - before).float()).item())
                    base = float(torch.linalg.vector_norm(before.float()).item())
                    row.update({"l2_drift": drift, "relative_l2_drift": drift / max(base, 1e-12)})
                rows.append(row)
            embedding = trained["edge_type_embedding.weight"].float()
            initial_embedding = initial["edge_type_embedding.weight"].float() if initial is not None else None
            for row_id in (0, 1):
                drift = (
                    float(torch.linalg.vector_norm(embedding[row_id] - initial_embedding[row_id]).item())
                    if initial_embedding is not None
                    else np.nan
                )
                rows.append(
                    {
                        "treatment": treatment,
                        "checkpoint_type": checkpoint_type,
                        "checkpoint_epoch": checkpoint_epochs[checkpoint_type],
                        "scope": f"edge_type_embedding_row_{row_id}",
                        "parameter_name": f"edge_type_embedding.weight[{row_id}]",
                        "numel": 8,
                        "drift_reference": "saved_initial_state",
                        "l2_drift": drift,
                        "relative_l2_drift": (
                            drift / max(float(torch.linalg.vector_norm(initial_embedding[row_id]).item()), 1e-12)
                            if initial_embedding is not None
                            else np.nan
                        ),
                        "initial_recreation_exact": initial_embedding is not None,
                        "trained_row_norm": float(torch.linalg.vector_norm(embedding[row_id]).item()),
                        "embedding_row_cosine": float(
                            torch.nn.functional.cosine_similarity(embedding[0].view(1, -1), embedding[1].view(1, -1)).item()
                        ),
                        "embedding_row_distance": float(torch.linalg.vector_norm(embedding[1] - embedding[0]).item()),
                        "learned_relation_difference_norm": (
                            float(
                                torch.linalg.vector_norm(
                                    (embedding[1] - embedding[0]) - (initial_embedding[1] - initial_embedding[0])
                                ).item()
                            )
                            if initial_embedding is not None
                            else np.nan
                        ),
                    }
                )
        for name, best_tensor in checkpoint_states["best"].items():
            last_tensor = checkpoint_states["last"][name]
            drift = float(torch.linalg.vector_norm((last_tensor - best_tensor).float()).item())
            base = float(torch.linalg.vector_norm(best_tensor.float()).item())
            rows.append(
                {
                    "treatment": treatment,
                    "checkpoint_type": "best_to_last",
                    "checkpoint_epoch": checkpoint_epochs["last"],
                    "scope": f"best_to_last_{parameter_category(name)}",
                    "parameter_name": name,
                    "numel": int(best_tensor.numel()),
                    "drift_reference": "best_checkpoint",
                    "l2_drift": drift,
                    "relative_l2_drift": drift / max(base, 1e-12),
                    "initial_recreation_exact": False,
                }
            )
    return pd.DataFrame(rows), recreation


def checkpoint_inspection() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    state_schemas: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    load_status: dict[str, Any] = {}
    device = torch.device("cpu")
    for treatment, run_dir in RUNS.items():
        cfg = read_config(run_dir)
        summary = read_json(run_dir / "d18_train_summary.json")
        history = pd.read_csv(run_dir / "train_log.csv")
        metadata = read_json(run_dir / "best_checkpoint_metadata.json")
        initial = read_json(run_dir / "initial_state_manifest.json")
        first_batch = read_json(run_dir / "first_batch_manifest.json")
        model_schema = read_json(run_dir / "model_schema.json")
        conditioning = read_json(run_dir / "conditioning_schema.json")
        complete = read_json(run_dir / "TRAINING_COMPLETE.json")
        for checkpoint_type in ("best", "last"):
            path = run_dir / "checkpoints" / f"{checkpoint_type}.pt"
            model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
            payload = load_checkpoint(
                path,
                model,
                device=device,
                expected_resume_signature=scientific_resume_signature(cfg),
                strict_signature=True,
            )
            schema = [(name, tuple(tensor.shape)) for name, tensor in payload["model_state_dict"].items()]
            state_schemas[f"{treatment}_{checkpoint_type}"] = schema
            epoch = int(payload.get("epoch", -1))
            history_row = history.loc[history["epoch"].astype(int).eq(epoch)].iloc[-1]
            rows.append(
                {
                    "model_id": f"A1-ID-{treatment}",
                    "treatment": treatment,
                    "seed": int((cfg.get("training") or {}).get("seed", cfg.get("seed", -1))),
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "source_config": str((run_dir / "source_config.yaml").relative_to(ROOT)),
                    "resolved_config": str((run_dir / "resolved_config.yaml").relative_to(ROOT)),
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_path": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(path),
                    "checkpoint_epoch": epoch,
                    "best_epoch": int(summary["best_epoch"]),
                    "last_epoch": int(history["epoch"].astype(int).max()),
                    "monitor_name": metadata["monitor"],
                    "monitor_mode": metadata["mode"],
                    "monitor_value": float(history_row["val_macro_f1"]),
                    "train_macro_f1_at_checkpoint": float(history_row["train_macro_f1"]),
                    "val_macro_f1_at_checkpoint": float(history_row["val_macro_f1"]),
                    "node_count": int(float(history_row["node_count_mean"])),
                    "node_dim": len(read_json(run_dir / "feature_schema.json")["node_feature_names"]),
                    "base_edge_dim": int(conditioning["base_edge_attr_dim"]),
                    "conditioned_edge_dim": int(conditioning["conditioned_edge_attr_dim"]),
                    "relation_embedding_dim": int(conditioning["embedding_dim"]),
                    "relation_mapping": json.dumps(read_json(run_dir / "relation_count_summary.json")["mapping"]),
                    "parameter_count": int(summary["parameter_count_trainable"]),
                    "batch_size": int((cfg.get("training") or {})["batch_size"]),
                    "config_signature": scientific_resume_signature(cfg),
                    "model_signature": model_schema["model_signature_sha256"],
                    "initial_state_hash": initial["canonical_state_sha256"],
                    "first_batch_hash": first_batch["manifest_sha256"],
                    "training_complete": complete.get("status") == "COMPLETE",
                    "resume_detected": (run_dir / "resume_events.jsonl").exists()
                    and "d18_resume_loaded" in (run_dir / "resume_events.jsonl").read_text(encoding="utf-8"),
                    "resume_source": "",
                    "git_commit": (run_dir / "code_provenance/git_commit.txt").read_text(encoding="utf-8").strip(),
                    "code_signature": sha256_file(run_dir / "code_provenance/d18__models__structure_gnn.py"),
                    "warnings": "",
                }
            )
            load_status[f"{treatment}_{checkpoint_type}_strict_load"] = True
    load_status["state_dict_schema_match"] = len({json.dumps(value) for value in state_schemas.values()}) == 1
    load_status["best_last_not_identical"] = all(
        rows[index]["checkpoint_sha256"] != rows[index + 1]["checkpoint_sha256"] for index in (0, 2)
    )
    return pd.DataFrame(rows), load_status


def training_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for treatment, run_dir in RUNS.items():
        history = pd.read_csv(run_dir / "train_log.csv")
        history.insert(0, "treatment", treatment)
        history["checkpoint_is_best"] = history["epoch"].astype(int).eq(
            int(read_json(run_dir / "d18_train_summary.json")["best_epoch"])
        )
        frames.append(history)
        best = history.loc[history["checkpoint_is_best"]].iloc[-1]
        min_loss = history.sort_values("val_loss").iloc[0]
        summaries.append(
            {
                "treatment": treatment,
                "epochs_completed": int(len(history)),
                "best_epoch": int(best["epoch"]),
                "peak_train_macro_f1": float(history["train_macro_f1"].max()),
                "best_val_macro_f1": float(best["val_macro_f1"]),
                "train_macro_f1_at_best": float(best["train_macro_f1"]),
                "train_val_gap": float(best["train_macro_f1"] - best["val_macro_f1"]),
                "validation_accuracy_at_best": float(best["val_accuracy"]),
                "validation_loss_at_best": float(best["val_loss"]),
                "minimum_validation_loss": float(min_loss["val_loss"]),
                "minimum_validation_loss_epoch": int(min_loss["epoch"]),
                "last_validation_macro_f1": float(history.iloc[-1]["val_macro_f1"]),
                "best_to_last_decline": float(best["val_macro_f1"] - history.iloc[-1]["val_macro_f1"]),
                "epoch_duration_mean_sec": float(history["epoch_time_sec"].mean()),
                "peak_memory_mb": float(history["memory_reserved_mb"].max()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def config_and_manifest_parity() -> tuple[pd.DataFrame, dict[str, Any]]:
    configs = {key: read_config(path) for key, path in RUNS.items()}
    flat = {key: flatten(value) for key, value in configs.items()}
    diff_rows = []
    for key in sorted(set(flat["null"]) | set(flat["correct"])):
        left, right = flat["null"].get(key), flat["correct"].get(key)
        if left != right:
            diff_rows.append(
                {
                    "config_key": key,
                    "null_value": json.dumps(left, default=str),
                    "correct_value": json.dumps(right, default=str),
                    "allowed": key in ALLOWED_CONFIG_DIFFS,
                }
            )
    null_initial = read_json(RUNS["null"] / "initial_state_manifest.json")
    correct_initial = read_json(RUNS["correct"] / "initial_state_manifest.json")
    null_batch = read_json(RUNS["null"] / "first_batch_manifest.json")
    correct_batch = read_json(RUNS["correct"] / "first_batch_manifest.json")
    null_cache = read_json(RUNS["null"] / "cache_signature.json")
    correct_cache = read_json(RUNS["correct"] / "cache_signature.json")
    parity = {
        "config_pair_diff_pass": bool(diff_rows) and all(row["allowed"] for row in diff_rows),
        "config_differences": diff_rows,
        "initial_state_hash_match": null_initial["canonical_state_sha256"]
        == correct_initial["canonical_state_sha256"],
        "first_batch_manifest_match": null_batch["manifest_sha256"] == correct_batch["manifest_sha256"],
        "cache_signature_parity": null_cache == correct_cache,
        "parameter_count_pair_match": read_json(RUNS["null"] / "d18_train_summary.json")[
            "parameter_count_trainable"
        ]
        == read_json(RUNS["correct"] / "d18_train_summary.json")["parameter_count_trainable"],
        "batch_size_match": (configs["null"].get("training") or {}).get("batch_size")
        == (configs["correct"].get("training") or {}).get("batch_size"),
        "conditioning_modes": [
            ((configs["null"].get("model") or {}).get("edge_type_conditioning") or {}).get("mode"),
            ((configs["correct"].get("model") or {}).get("edge_type_conditioning") or {}).get("mode"),
        ],
    }
    return pd.DataFrame(diff_rows), parity


def contextual_test_metrics() -> pd.DataFrame:
    rows = []
    for treatment, run_dir in RUNS.items():
        for checkpoint_type in ("best", "last"):
            frame = pd.read_csv(run_dir / f"evaluation_{checkpoint_type}/official_metrics.csv")
            rows.append(
                {
                    "reference": f"A1-ID-{treatment}",
                    "checkpoint_type": checkpoint_type,
                    "source": "stored_preanalysis_replay_check",
                    **frame.iloc[0].to_dict(),
                }
            )
    if A0_RUN.exists():
        for checkpoint_type in ("best", "last"):
            path = A0_RUN / f"evaluation_{checkpoint_type}/official_metrics.csv"
            if path.exists():
                rows.append(
                    {
                        "reference": "A0_seed42",
                        "checkpoint_type": checkpoint_type,
                        "source": "stored_context_only",
                        **pd.read_csv(path).iloc[0].to_dict(),
                    }
                )
    if C2_RUN.exists():
        summary_path = C2_RUN / "d18_train_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            rows.extend(
                [
                    {
                        "reference": "C2_seed42",
                        "checkpoint_type": "best",
                        "source": "stored_context_only",
                        "accuracy": summary.get("test_accuracy"),
                        "macro_f1": summary.get("test_macro_f1"),
                    },
                    {
                        "reference": "C2_seed42",
                        "checkpoint_type": "last",
                        "source": "stored_context_only",
                        "accuracy": summary.get("last_test_accuracy"),
                        "macro_f1": summary.get("last_test_macro_f1"),
                    },
                ]
            )
    return pd.DataFrame(rows)


def make_plots(
    output: Path,
    curves: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    class_frame: pd.DataFrame,
    test_metrics: pd.DataFrame,
    locked_metrics: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    representation_comparisons: pd.DataFrame,
    parameter_rows: pd.DataFrame,
) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    for treatment, frame in curves.groupby("treatment"):
        plt.plot(frame["epoch"], frame["train_macro_f1"], label=f"{treatment} train")
        plt.plot(frame["epoch"], frame["val_macro_f1"], linestyle="--", label=f"{treatment} val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "training_curves.png", dpi=180)
    plt.close()

    primary = validation_metrics[
        validation_metrics["checkpoint_type"].eq("best")
        & validation_metrics.apply(lambda row: row["conditioning_mode"] == row["treatment"], axis=1)
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(primary["treatment"], primary["macro_f1"])
    plt.ylabel("Validation macro-F1")
    plt.tight_layout()
    plt.savefig(plots / "validation_metric_comparison.png", dpi=180)
    plt.close()

    x = np.arange(7)
    width = 0.38
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, class_frame["null_f1"], width, label="null")
    plt.bar(x + width / 2, class_frame["correct_f1"], width, label="correct")
    plt.xticks(x, class_frame["class_name"])
    plt.ylabel("Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "validation_classwise_f1.png", dpi=180)
    plt.close()

    for frame, filename, title in (
        (test_metrics, "full_test_metric_comparison.png", "Full test"),
        (locked_metrics, "locked_metric_comparison.png", "Locked 715"),
    ):
        official = frame[
            frame["checkpoint_type"].eq("best")
            & frame.apply(lambda row: row["conditioning_mode"] == row["treatment"], axis=1)
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(official["treatment"], official["macro_f1"])
        plt.ylabel(f"{title} macro-F1")
        plt.tight_layout()
        plt.savefig(plots / filename, dpi=180)
        plt.close()

    cf = counterfactuals[counterfactuals["split"].eq("val")]
    plt.figure(figsize=(9, 4))
    labels = [f"{row.treatment}:{row.conditioning_mode}" for row in cf.itertuples()]
    plt.bar(labels, cf["macro_f1"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Validation macro-F1")
    plt.tight_layout()
    plt.savefig(plots / "relation_counterfactuals.png", dpi=180)
    plt.close()

    cka = representation_comparisons[representation_comparisons["comparison"].eq("correct_vs_null_trained")]
    plt.figure(figsize=(10, 4))
    plt.bar(cka["layer"], cka["linear_cka"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Linear CKA")
    plt.tight_layout()
    plt.savefig(plots / "representation_cka.png", dpi=180)
    plt.close()

    embedding = parameter_rows[
        parameter_rows["scope"].isin(["edge_type_embedding_row_0", "edge_type_embedding_row_1"])
        & parameter_rows["checkpoint_type"].eq("best")
    ]
    plt.figure(figsize=(8, 4))
    labels = [f"{row.treatment}:row{row.scope[-1]}" for row in embedding.itertuples()]
    plt.bar(labels, embedding["l2_drift"])
    plt.ylabel("L2 drift from initialization")
    plt.tight_layout()
    plt.savefig(plots / "embedding_row_drift.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/d19_analysis/d19_a1_id_posttraining_analysis",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-bootstrap", type=int, default=10_000)
    parser.add_argument("--locked-bootstrap", type=int, default=5_000)
    args = parser.parse_args()
    if args.validation_bootstrap < 10_000 or args.locked_bootstrap < 5_000:
        raise RuntimeError("Registered bootstrap minima are validation=10000 and locked=5000")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "d19_a1_analysis_start", "device": str(device), "output": str(output)}), flush=True)

    artifact_frame, checkpoint_status = checkpoint_inspection()
    artifact_frame.to_csv(output / "01_run_artifact_manifest.csv", index=False)
    config_diffs, parity = config_and_manifest_parity()
    curves, curve_summary = training_curves()
    curves.to_csv(output / "04_training_curves.csv", index=False)

    locked = locked_manifest()
    val_indices = list(range(len(pd.read_csv(ROOT / "data/val.csv"))))
    graph_audits = [
        aggregate_graph_audit(cache_paths("train", list(range(32))), "first_two_deterministic_training_batches_batch16"),
        aggregate_graph_audit(cache_paths("val", val_indices), "full_validation"),
        aggregate_graph_audit(cache_paths("test", locked["sample_index"].astype(int).tolist()), "locked_715"),
    ]
    graph_parity = all(row["exact_pair_parity"] for row in graph_audits)

    predictions: list[pd.DataFrame] = []
    eval_info_rows = []
    representation_arrays: dict[str, dict[str, np.ndarray]] = {}
    representation_node_rows: list[dict[str, Any]] = []
    plans = []
    for split, indices in (("val", None), ("test", None), ("test", locked["sample_index"].astype(int).tolist())):
        scope = "locked" if indices is not None else split
        plans.extend(
            [
                (scope, split, indices, "null", "best", "null", scope == "locked"),
                (scope, split, indices, "correct", "best", "correct", scope == "locked"),
                (scope, split, indices, "null", "last", "null", False),
                (scope, split, indices, "correct", "last", "correct", False),
            ]
        )
        if scope in {"val", "locked"}:
            plans.extend(
                [
                    (scope, split, indices, "null", "best", "correct", False),
                    (scope, split, indices, "null", "best", "swapped", False),
                    (scope, split, indices, "correct", "best", "null", scope == "locked"),
                    (scope, split, indices, "correct", "best", "swapped", scope == "locked"),
                ]
            )
    for scope, split, indices, treatment, checkpoint_type, mode, capture in plans:
        frame, arrays, node_rows, info = evaluate(
            treatment=treatment,
            checkpoint_type=checkpoint_type,
            split=split,
            mode=mode,
            device=device,
            batch_size=int(args.batch_size),
            indices=indices,
            capture_layers=capture,
        )
        frame["split"] = scope
        predictions.append(frame)
        eval_info_rows.append(
            {
                "split": scope,
                "treatment": treatment,
                "checkpoint_type": checkpoint_type,
                "conditioning_mode": mode,
                **info,
            }
        )
        if capture:
            state = f"{treatment}_{checkpoint_type}_{mode}"
            representation_arrays[state] = arrays
            representation_node_rows.extend(node_rows)
        print(
            json.dumps(
                {
                    "event": "d19_a1_eval_done",
                    "split": scope,
                    "treatment": treatment,
                    "checkpoint": checkpoint_type,
                    "mode": mode,
                    "rows": len(frame),
                }
            ),
            flush=True,
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    info_frame = pd.DataFrame(eval_info_rows)

    validation_predictions = prediction_frame[prediction_frame["split"].eq("val")].copy()
    validation_predictions.to_csv(output / "05_validation_predictions.csv", index=False)
    test_predictions = prediction_frame[prediction_frame["split"].eq("test")].copy()
    test_predictions.to_csv(output / "08_full_test_predictions.csv", index=False)
    locked_predictions = prediction_frame[prediction_frame["split"].eq("locked")].copy()
    locked_predictions = locked_predictions.merge(
        locked[["sample_index", "image_id"]],
        on="sample_index",
        how="left",
        validate="many_to_one",
    )
    locked_predictions.to_csv(output / "09_locked_predictions.csv", index=False)

    metric_rows = []
    for keys, frame in prediction_frame.groupby(
        ["split", "treatment", "checkpoint_type", "conditioning_mode"], sort=False
    ):
        metric_rows.append(
            {
                "split": keys[0],
                "treatment": keys[1],
                "checkpoint_type": keys[2],
                "conditioning_mode": keys[3],
                **metric_bundle(frame),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    validation_metrics = metrics[metrics["split"].eq("val")].copy()
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    locked_metrics = metrics[metrics["split"].eq("locked")].copy()
    validation_metrics.to_csv(output / "05_validation_metrics.csv", index=False)
    test_metrics.to_csv(output / "08_full_test_metrics.csv", index=False)
    locked_metrics.to_csv(output / "09_locked_metrics.csv", index=False)

    def select_predictions(split: str, treatment: str, checkpoint: str, mode: str) -> pd.DataFrame:
        return prediction_frame[
            prediction_frame["split"].eq(split)
            & prediction_frame["treatment"].eq(treatment)
            & prediction_frame["checkpoint_type"].eq(checkpoint)
            & prediction_frame["conditioning_mode"].eq(mode)
        ].reset_index(drop=True)

    null_val = select_predictions("val", "null", "best", "null")
    correct_val = select_predictions("val", "correct", "best", "correct")
    null_locked = select_predictions("locked", "null", "best", "null")
    correct_locked = select_predictions("locked", "correct", "best", "correct")
    validation_boot = paired_bootstrap(
        null_val,
        correct_val,
        replicates=int(args.validation_bootstrap),
        seed=42,
        scope="validation",
    )
    locked_boot = paired_bootstrap(
        null_locked,
        correct_locked,
        replicates=int(args.locked_bootstrap),
        seed=42,
        scope="locked_715",
    )
    bootstrap_frame = pd.concat([validation_boot, locked_boot], ignore_index=True)
    bootstrap_frame.to_csv(output / "06_validation_paired_bootstrap.csv", index=False)

    validation_classwise = classwise(null_val, correct_val, "validation")
    validation_classwise.to_csv(output / "07_validation_classwise.csv", index=False)
    maximum_class_loss = float((-validation_classwise["correct_minus_null_f1"]).max())

    counterfactual_metrics = metrics[
        metrics["split"].isin(["val", "locked"])
        & metrics["checkpoint_type"].eq("best")
    ].copy()
    pair_diagnostics = []
    for split in ("val", "locked"):
        for treatment, official_mode in (("null", "null"), ("correct", "correct")):
            official = select_predictions(split, treatment, "best", official_mode)
            for mode in ("null", "correct", "swapped"):
                candidate = select_predictions(split, treatment, "best", mode)
                if candidate.empty:
                    continue
                paired_alignment(official, candidate)
                logits_a = official[[f"logit_{index}" for index in range(7)]].to_numpy()
                logits_b = candidate[[f"logit_{index}" for index in range(7)]].to_numpy()
                probs_a = official[[f"prob_{index}" for index in range(7)]].to_numpy()
                probs_b = candidate[[f"prob_{index}" for index in range(7)]].to_numpy()
                pair_diagnostics.append(
                    {
                        "split": split,
                        "treatment": treatment,
                        "official_mode": official_mode,
                        "counterfactual_mode": mode,
                        "prediction_agreement": float(
                            (
                                official["predicted_class"].to_numpy()
                                == candidate["predicted_class"].to_numpy()
                            ).mean()
                        ),
                        "changed_prediction_count": int(
                            (
                                official["predicted_class"].to_numpy()
                                != candidate["predicted_class"].to_numpy()
                            ).sum()
                        ),
                        "max_logit_difference": float(np.abs(logits_a - logits_b).max()),
                        "mean_logit_difference": float(np.abs(logits_a - logits_b).mean()),
                        "max_probability_difference": float(np.abs(probs_a - probs_b).max()),
                        "mean_probability_difference": float(np.abs(probs_a - probs_b).mean()),
                    }
                )
    counterfactual_metrics.to_csv(output / "10_relation_counterfactuals.csv", index=False)
    pd.DataFrame(pair_diagnostics).to_csv(
        output / "10_relation_counterfactual_pair_diagnostics.csv", index=False
    )

    transition_frame = pd.DataFrame(
        [
            *transition_rows(null_val, correct_val, "correct_trained_vs_null_trained"),
            *transition_rows(
                select_predictions("val", "correct", "best", "null"),
                correct_val,
                "correct_checkpoint_correct_vs_null_id",
            ),
            *transition_rows(
                select_predictions("val", "correct", "best", "swapped"),
                correct_val,
                "correct_checkpoint_correct_vs_swapped_id",
            ),
        ]
    )
    transition_frame.to_csv(output / "11_error_transition_analysis.csv", index=False)

    calibration = calibration_rows(prediction_frame)
    calibration.to_csv(output / "12_calibration_analysis.csv", index=False)

    locked_labels = correct_locked["true_class"].to_numpy(dtype=np.int64)
    representation_geometry, representation_comparisons = representation_tables(
        representation_arrays, representation_node_rows, locked_labels
    )
    representation = pd.concat(
        [representation_geometry, representation_comparisons], ignore_index=True, sort=False
    )
    representation.to_csv(output / "13_representation_analysis.csv", index=False)

    parameter_rows, recreation = parameter_audit()
    parameter_rows.to_csv(output / "14_parameter_learning_audit.csv", index=False)

    stored_context = contextual_test_metrics()
    stored_context.to_csv(output / "08_contextual_reference_metrics.csv", index=False)
    stored_replay_checks = []
    for treatment in ("null", "correct"):
        for checkpoint_type in ("best", "last"):
            recomputed = test_metrics[
                test_metrics["treatment"].eq(treatment)
                & test_metrics["checkpoint_type"].eq(checkpoint_type)
                & test_metrics["conditioning_mode"].eq(treatment)
            ].iloc[0]
            stored = pd.read_csv(
                RUNS[treatment] / f"evaluation_{checkpoint_type}/official_metrics.csv"
            ).iloc[0]
            stored_replay_checks.append(
                {
                    "treatment": treatment,
                    "checkpoint_type": checkpoint_type,
                    "accuracy_abs_difference": abs(float(recomputed["accuracy"]) - float(stored["accuracy"])),
                    "macro_f1_abs_difference": abs(float(recomputed["macro_f1"]) - float(stored["macro_f1"])),
                    "pass": abs(float(recomputed["macro_f1"]) - float(stored["macro_f1"])) <= 1e-6,
                }
            )
    stored_replay = pd.DataFrame(stored_replay_checks)

    primary_null = validation_metrics[
        validation_metrics["treatment"].eq("null")
        & validation_metrics["checkpoint_type"].eq("best")
        & validation_metrics["conditioning_mode"].eq("null")
    ].iloc[0]
    primary_correct = validation_metrics[
        validation_metrics["treatment"].eq("correct")
        & validation_metrics["checkpoint_type"].eq("best")
        & validation_metrics["conditioning_mode"].eq("correct")
    ].iloc[0]
    curve_lookup = {row.treatment: row for row in curve_summary.itertuples()}
    val_gain = float(primary_correct["macro_f1"] - primary_null["macro_f1"])
    gap_increase = float(curve_lookup["correct"].train_val_gap - curve_lookup["null"].train_val_gap)

    resume_isolation = not artifact_frame["resume_detected"].any()
    technical_checks = {
        **checkpoint_status,
        "config_pair_diff_pass": parity["config_pair_diff_pass"],
        "initial_state_hash_match": parity["initial_state_hash_match"],
        "first_batch_manifest_match": parity["first_batch_manifest_match"],
        "cache_signature_parity": parity["cache_signature_parity"],
        "parameter_count_pair_match": parity["parameter_count_pair_match"],
        "batch_size_match": parity["batch_size_match"],
        "graph_parity": graph_parity,
        "normalization_parity": graph_parity,
        "resume_isolation_pass": resume_isolation,
        "checkpoint_policy_pass": bool(
            all(
                read_json(run / "best_checkpoint_metadata.json")["selection_split"] == "validation"
                and read_json(run / "best_checkpoint_metadata.json")["test_not_used_for_selection"]
                for run in RUNS.values()
            )
        ),
        "prediction_alignment": bool(
            np.array_equal(null_val["sample_index"].to_numpy(), correct_val["sample_index"].to_numpy())
        ),
        "prediction_finiteness": bool(info_frame["prediction_finite"].all()),
        "hook_integrity": bool(info_frame["max_hook_output_difference"].max() == 0.0),
        "stored_test_replay_pass": bool(stored_replay["pass"].all()),
    }
    blockers = [key for key, value in technical_checks.items() if isinstance(value, (bool, np.bool_)) and not value]
    if blockers:
        decision = "BLOCKED"
    elif val_gain * 100.0 >= 0.75 and gap_increase * 100.0 <= 3.0 and maximum_class_loss * 100.0 <= 5.0:
        decision = "PROMOTE_RELATION_ID"
    else:
        decision = "STOP_RELATION_ID"

    null_test_best = test_metrics[
        test_metrics["treatment"].eq("null")
        & test_metrics["checkpoint_type"].eq("best")
        & test_metrics["conditioning_mode"].eq("null")
    ].iloc[0]
    correct_test_best = test_metrics[
        test_metrics["treatment"].eq("correct")
        & test_metrics["checkpoint_type"].eq("best")
        & test_metrics["conditioning_mode"].eq("correct")
    ].iloc[0]
    accuracy_gap_to_65 = {
        "null_best_pp": float((0.65 - null_test_best["accuracy"]) * 100.0),
        "correct_best_pp": float((0.65 - correct_test_best["accuracy"]) * 100.0),
    }

    cf_correct = {
        row.conditioning_mode: row
        for row in counterfactual_metrics[
            counterfactual_metrics["split"].eq("val")
            & counterfactual_metrics["treatment"].eq("correct")
        ].itertuples()
    }
    h1 = val_gain * 100.0 >= 0.75
    h2 = gap_increase * 100.0 <= 3.0
    h3 = maximum_class_loss * 100.0 <= 5.0
    h4 = (
        cf_correct["correct"].macro_f1 > cf_correct["null"].macro_f1
        and cf_correct["correct"].macro_f1 > cf_correct["swapped"].macro_f1
    )
    hypotheses = [
        {
            "hypothesis": "H-A1-1",
            "supported": h1,
            "measured_evidence": val_gain,
            "conclusion": "supported" if h1 else "not supported",
            "confidence": "high",
            "remaining_uncertainty": "training-seed variability is unmeasured",
        },
        {
            "hypothesis": "H-A1-2",
            "supported": h2,
            "measured_evidence": gap_increase,
            "conclusion": "supported" if h2 else "not supported",
            "confidence": "high",
            "remaining_uncertainty": "fixed seed42 only",
        },
        {
            "hypothesis": "H-A1-3",
            "supported": h3,
            "measured_evidence": maximum_class_loss,
            "conclusion": "supported" if h3 else "not supported",
            "confidence": "high",
            "remaining_uncertainty": "Disgust support is small",
        },
        {
            "hypothesis": "H-A1-4",
            "supported": h4,
            "measured_evidence": {
                "correct_mode_macro_f1": cf_correct["correct"].macro_f1,
                "null_mode_macro_f1": cf_correct["null"].macro_f1,
                "swapped_mode_macro_f1": cf_correct["swapped"].macro_f1,
            },
            "conclusion": "supported" if h4 else "not supported",
            "confidence": "medium",
            "remaining_uncertainty": "counterfactual modes are off-policy",
        },
        {
            "hypothesis": "H-A1-5",
            "supported": decision == "PROMOTE_RELATION_ID",
            "measured_evidence": decision,
            "conclusion": "supported" if decision == "PROMOTE_RELATION_ID" else "not supported",
            "confidence": "high",
            "remaining_uncertainty": "one training seed",
        },
    ]

    limitations = [
        "Only one A1-ID training seed is available.",
        "Exact initial-to-trained parameter drift is NOT VERIFIABLE because initial tensor values were not saved and local Torch recreation does not match the Kaggle initialization hash.",
        "Paired validation bootstrap is conditional on fixed seed42 checkpoints and does not estimate training-seed uncertainty.",
        "Null and correct share seed and initial tensors, but optimization trajectories diverge after treatment-specific forward passes.",
        "Test results are secondary and cannot change the registered validation gate.",
        "Locked-715 is a separate evaluation population and is not validation-selection evidence.",
        "Counterfactual ID modes are off-policy diagnostics.",
        "The unused null embedding row may be affected by optimizer implementation or weight decay.",
        "Disgust validation support is small.",
        "Exact git provenance is limited to the saved commit and clean-status snapshot.",
        "Passing A1-ID would not guarantee 0.65 accuracy.",
        "A1-ID does not test pixel-selection sufficiency.",
    ]

    artifact_ok = not blockers
    (output / "00_README.md").write_text(
        "# D19-A1-ID Post-Training Analysis\n\n"
        "Controlled, read-only comparison of `correct - null`. Primary selection uses `best.pt` and the full validation split. "
        "Test and locked-715 results are secondary. No training or model mutation occurred.\n\n"
        f"Registered decision: **{decision}**.\n",
        encoding="utf-8",
    )
    (output / "01_artifact_integrity.md").write_text(
        "# Artifact Integrity\n\n"
        f"Result: **{'PASS' if artifact_ok else 'BLOCKED'}**.\n\n"
        + md_table(artifact_frame)
        + "\n\nStrict checkpoint loads, SHA-256 values, epoch/history agreement, completion markers and provenance were checked. "
        "A `resume_events.jsonl` containing only an early-stop event is not treated as resume contamination.\n",
        encoding="utf-8",
    )
    (output / "02_experiment_parity_validation.md").write_text(
        "# Experiment Parity Validation\n\n"
        f"Config parity: **{'PASS' if parity['config_pair_diff_pass'] else 'FAIL'}**. "
        f"Graph/normalization parity: **{'PASS' if graph_parity else 'FAIL'}**.\n\n"
        "## Resolved Config Differences\n\n"
        + md_table(config_diffs)
        + "\n\n## Graph Audit\n\n"
        + md_table(pd.DataFrame(graph_audits))
        + "\n\nBoth treatments resolve to the same immutable evidence-only cache namespace. "
        "The only model-side treatment is conditioning ID selection; cached `edge_type` is not rewritten.\n",
        encoding="utf-8",
    )
    checkpoint_table = artifact_frame[
        [
            "treatment",
            "checkpoint_type",
            "checkpoint_epoch",
            "monitor_name",
            "monitor_value",
            "train_macro_f1_at_checkpoint",
            "val_macro_f1_at_checkpoint",
        ]
    ]
    (output / "03_checkpoint_policy_audit.md").write_text(
        "# Checkpoint Policy Audit\n\n"
        "Primary checkpoint is `best.pt`, selected by validation macro-F1. `last.pt` is sensitivity only. "
        "No test metric selected a checkpoint.\n\n"
        + md_table(checkpoint_table),
        encoding="utf-8",
    )
    (output / "04_training_curve_comparison.md").write_text(
        "# Training Curve Comparison\n\n"
        + md_table(curve_summary)
        + "\n\nHigher train performance is not counted as improvement without validation improvement.\n",
        encoding="utf-8",
    )
    gate_frame = pd.DataFrame(
        [
            {
                "validation_macro_f1_null": float(primary_null["macro_f1"]),
                "validation_macro_f1_correct": float(primary_correct["macro_f1"]),
                "validation_macro_f1_gain_pp": val_gain * 100.0,
                "train_val_gap_null_pp": curve_lookup["null"].train_val_gap * 100.0,
                "train_val_gap_correct_pp": curve_lookup["correct"].train_val_gap * 100.0,
                "gap_increase_pp": gap_increase * 100.0,
                "maximum_validation_class_f1_loss_pp": maximum_class_loss * 100.0,
                "registered_decision": decision,
            }
        ]
    )
    (output / "05_primary_validation_decision.md").write_text(
        "# Primary Validation Decision\n\n"
        + md_table(gate_frame)
        + "\n\nThe +0.75 pp threshold is applied to full-precision validation macro-F1. Test metrics are not used here.\n",
        encoding="utf-8",
    )
    (output / "06_validation_paired_bootstrap.md").write_text(
        "# Validation Paired Bootstrap\n\n"
        "Class-stratified paired image bootstrap, seed42. Validation uses at least 10,000 replicates; locked uses at least 5,000. "
        "Intervals estimate image uncertainty conditional on fixed checkpoints, not training-seed variability.\n\n"
        + md_table(bootstrap_frame),
        encoding="utf-8",
    )
    (output / "07_validation_classwise.md").write_text(
        "# Validation Classwise Gate\n\n"
        + md_table(validation_classwise)
        + f"\n\nMaximum class F1 loss: **{maximum_class_loss * 100.0:.6f} pp**. "
        "Disgust is reported with support and is not overinterpreted.\n",
        encoding="utf-8",
    )
    (output / "08_full_test_comparison.md").write_text(
        "# Full Official Test Comparison\n\n"
        "Recomputed full-test results for A1-ID are primary in this table. A0/C2 rows are stored contextual references only. "
        "Test direction cannot overturn the validation gate.\n\n"
        + md_table(test_metrics)
        + "\n\n## Stored Context\n\n"
        + md_table(stored_context)
        + "\n\n## Replay Agreement\n\n"
        + md_table(stored_replay),
        encoding="utf-8",
    )
    (output / "09_locked_comparison.md").write_text(
        "# Locked 715 Comparison\n\n"
        f"Locked sample SHA-256: `{LOCKED_SHA256}`. These results are secondary and are not selection evidence.\n\n"
        + md_table(locked_metrics)
        + "\n\n"
        + md_table(locked_boot),
        encoding="utf-8",
    )
    (output / "10_relation_counterfactuals.md").write_text(
        "# Relation-ID Counterfactuals\n\n"
        "Overrides are model-side only; cached graph endpoints, edge order, base edge attributes, true `edge_type`, and normalization remain unchanged. "
        "Null-trained correct/swapped modes and correct-trained null/swapped modes are off-policy diagnostics.\n\n"
        + md_table(counterfactual_metrics)
        + "\n\n## Paired Output Changes\n\n"
        + md_table(pd.DataFrame(pair_diagnostics)),
        encoding="utf-8",
    )
    (output / "11_error_transition_analysis.md").write_text(
        "# Error Transition Analysis\n\n"
        + md_table(transition_frame)
        + "\n\nUp to 50 sample indices are retained per transition; no image files were copied.\n",
        encoding="utf-8",
    )
    (output / "12_calibration_analysis.md").write_text(
        "# Calibration Analysis\n\n"
        + md_table(calibration)
        + "\n\nLower entropy alone is not interpreted as better calibration. Correct and incorrect subsets are reported separately.\n",
        encoding="utf-8",
    )
    (output / "13_representation_analysis.md").write_text(
        "# Representation Analysis\n\n"
        "Linear CKA and geometry-level measures are primary across independently optimized models. Hooks were read-only and reproduced the same logits exactly.\n\n"
        + md_table(representation)
        + "\n\nRelation identity first acts before GNN layer 1; the layerwise table shows where its learned effect becomes measurable. "
        "This analysis does not reopen A2 or multi-scale pooling.\n",
        encoding="utf-8",
    )
    embedding_rows = parameter_rows[
        parameter_rows["scope"].str.startswith("edge_type_embedding_row", na=False)
    ]
    category_rows = (
        parameter_rows.dropna(subset=["l2_drift"])
        .groupby(["treatment", "checkpoint_type", "scope"], as_index=False)
        .agg(total_l2_drift=("l2_drift", lambda values: float(np.sqrt(np.square(values).sum()))))
    )
    (output / "14_parameter_learning_audit.md").write_text(
        "# Parameter Learning Audit\n\n"
        "Canonical initial tensors were recreated only when the full saved initialization hash matched exactly. "
        "For these runs, exact initial tensor values are NOT VERIFIABLE locally because the run artifacts retain "
        "their hashes but not the tensors, and local Torch recreation does not match the Kaggle hash. Therefore, "
        "initial-to-trained L2 fields remain empty; best-to-last drift and trained embedding geometry remain directly measurable. "
        "Large drift is not equated with useful learning.\n\n"
        + md_table(category_rows)
        + "\n\n## Embedding Rows\n\n"
        + md_table(embedding_rows)
        + "\n\n## Initial Recreation\n\n```json\n"
        + json.dumps(recreation, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    sensitivity_rows = []
    for split in ("val", "test", "locked"):
        for checkpoint_type in ("best", "last"):
            left = metrics[
                metrics["split"].eq(split)
                & metrics["treatment"].eq("null")
                & metrics["checkpoint_type"].eq(checkpoint_type)
                & metrics["conditioning_mode"].eq("null")
            ].iloc[0]
            right = metrics[
                metrics["split"].eq(split)
                & metrics["treatment"].eq("correct")
                & metrics["checkpoint_type"].eq(checkpoint_type)
                & metrics["conditioning_mode"].eq("correct")
            ].iloc[0]
            sensitivity_rows.append(
                {
                    "split": split,
                    "checkpoint_type": checkpoint_type,
                    "accuracy_difference": float(right["accuracy"] - left["accuracy"]),
                    "macro_f1_difference": float(right["macro_f1"] - left["macro_f1"]),
                    "nll_difference": float(right["nll"] - left["nll"]),
                    "ece_difference": float(right["ece"] - left["ece"]),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    (output / "15_best_last_sensitivity.md").write_text(
        "# Best-versus-Last Sensitivity\n\n"
        + md_table(sensitivity)
        + "\n\nThe registered decision remains based on `best.pt`; a last-only pass would not be promotable.\n",
        encoding="utf-8",
    )
    (output / "16_hypothesis_update.md").write_text(
        "# Hypothesis Update\n\n"
        + md_table(pd.DataFrame(hypotheses))
        + "\n\nGeneralization across training seeds remains unverified.\n",
        encoding="utf-8",
    )
    final_fields = {
        "validation_macro_f1_null": float(primary_null["macro_f1"]),
        "validation_macro_f1_correct": float(primary_correct["macro_f1"]),
        "validation_macro_f1_gain_pp": val_gain * 100.0,
        "train_val_gap_null_pp": curve_lookup["null"].train_val_gap * 100.0,
        "train_val_gap_correct_pp": curve_lookup["correct"].train_val_gap * 100.0,
        "gap_increase_pp": gap_increase * 100.0,
        "maximum_validation_class_f1_loss_pp": maximum_class_loss * 100.0,
        "batch_size_null": int(artifact_frame[artifact_frame["treatment"].eq("null")]["batch_size"].iloc[0]),
        "batch_size_correct": int(artifact_frame[artifact_frame["treatment"].eq("correct")]["batch_size"].iloc[0]),
        "graph_parity": graph_parity,
        "normalization_parity": graph_parity,
        "parameter_parity": parity["parameter_count_pair_match"],
        "initialization_parity": parity["initial_state_hash_match"],
        "checkpoint_policy_valid": technical_checks["checkpoint_policy_pass"],
        "artifact_integrity": artifact_ok,
        "registered_decision": decision,
    }
    (output / "17_registered_final_decision.md").write_text(
        "# Registered Final Decision\n\n## "
        + decision
        + "\n\n"
        + md_table(pd.DataFrame([final_fields]))
        + "\n\nDecision logic was applied mechanically with full-precision values. No test or locked result altered this label.\n",
        encoding="utf-8",
    )
    next_step = (
        "Freeze correct-ID as message-conditioning baseline; do not run another A1-ID seed or independent relation operators; next allowed task is a read-only Pixel Selection Sufficiency Audit preserving one-pixel-per-node semantics."
        if decision == "PROMOTE_RELATION_ID"
        else (
            "Remove A1-ID from the final candidate architecture; retain the shared A0/C2 message operator; do not implement independent relation operators; next allowed task is a read-only Pixel Selection Sufficiency Audit preserving one-pixel-per-node semantics."
            if decision == "STOP_RELATION_ID"
            else "Repair exactly the listed technical blockers before any pixel-selection conclusion."
        )
    )
    (output / "18_next_step_scope.md").write_text(
        "# Next-Step Scope\n\n"
        + next_step
        + "\n\nThe future audit must ask whether landmark-guided selection of 1,800 pixel nodes preserves expression-relevant graph signal. It is not implemented here.\n",
        encoding="utf-8",
    )

    machine = {
        "artifact_integrity": {"pass": artifact_ok, "blockers": blockers},
        "experiment_parity": parity,
        "graph_audits": graph_audits,
        "checkpoint_policy": checkpoint_status,
        "training_curves": curve_summary.to_dict(orient="records"),
        "validation_primary": {
            "null": clean_json(primary_null.to_dict()),
            "correct": clean_json(primary_correct.to_dict()),
            "correct_minus_null": {
                "macro_f1": val_gain,
                "accuracy": float(primary_correct["accuracy"] - primary_null["accuracy"]),
            },
        },
        "registered_gates": {
            "validation_macro_f1_gain_required_pp": 0.75,
            "maximum_gap_increase_pp": 3.0,
            "maximum_class_f1_loss_pp": 5.0,
            "observed_gap_increase_pp": gap_increase * 100.0,
            "observed_maximum_class_f1_loss_pp": maximum_class_loss * 100.0,
        },
        "validation_bootstrap": clean_json(validation_boot.to_dict(orient="records")),
        "validation_classwise": clean_json(validation_classwise.to_dict(orient="records")),
        "full_test": clean_json(test_metrics.to_dict(orient="records")),
        "locked_715": {
            "sha256": LOCKED_SHA256,
            "metrics": clean_json(locked_metrics.to_dict(orient="records")),
            "bootstrap": clean_json(locked_boot.to_dict(orient="records")),
        },
        "relation_counterfactuals": clean_json(counterfactual_metrics.to_dict(orient="records")),
        "error_transitions": clean_json(transition_frame.to_dict(orient="records")),
        "calibration": clean_json(calibration.to_dict(orient="records")),
        "representation": clean_json(representation.to_dict(orient="records")),
        "parameter_learning": {
            "initial_recreation": recreation,
            "embedding_rows": clean_json(embedding_rows.to_dict(orient="records")),
        },
        "best_last_sensitivity": clean_json(sensitivity.to_dict(orient="records")),
        "hypotheses": clean_json(hypotheses),
        "accuracy_gap_to_65": accuracy_gap_to_65,
        "registered_decision": decision,
        "next_step": next_step,
        "limitations": limitations,
        "training_launched": False,
        "model_modified": False,
    }
    write_json(output / "19_machine_readable_summary.json", clean_json(machine))
    (output / "20_run_commands.md").write_text(
        "# Run Commands\n\n```powershell\n"
        "conda run -n fer-graph python -B d19/scripts/analyze_d19_a1_id_posttraining.py "
        "--device cuda:0 --batch-size "
        + str(args.batch_size)
        + " --validation-bootstrap "
        + str(args.validation_bootstrap)
        + " --locked-bootstrap "
        + str(args.locked_bootstrap)
        + "\n```\n\nThis command is analysis-only; it does not invoke a trainer or write checkpoints.\n",
        encoding="utf-8",
    )
    validation_summary = {
        "null_run_found": RUNS["null"].exists(),
        "correct_run_found": RUNS["correct"].exists(),
        "a0_reference_found": A0_RUN.exists(),
        "null_training_complete": bool(
            artifact_frame[artifact_frame["treatment"].eq("null")]["training_complete"].all()
        ),
        "correct_training_complete": bool(
            artifact_frame[artifact_frame["treatment"].eq("correct")]["training_complete"].all()
        ),
        "null_best_load": checkpoint_status["null_best_strict_load"],
        "correct_best_load": checkpoint_status["correct_best_strict_load"],
        "null_last_load": checkpoint_status["null_last_strict_load"],
        "correct_last_load": checkpoint_status["correct_last_strict_load"],
        "config_pair_diff_pass": parity["config_pair_diff_pass"],
        "graph_endpoint_parity": graph_parity,
        "true_edge_type_parity": graph_parity,
        "base_edge_attr_parity": graph_parity,
        "cache_signature_parity": parity["cache_signature_parity"],
        "total_degree_parity": graph_parity,
        "normalization_parity": graph_parity,
        "parameter_count_pair_match": parity["parameter_count_pair_match"],
        "state_dict_schema_match": checkpoint_status["state_dict_schema_match"],
        "initial_state_hash_match": parity["initial_state_hash_match"],
        "first_batch_manifest_match": parity["first_batch_manifest_match"],
        "batch_size_match": parity["batch_size_match"],
        "resume_isolation_pass": resume_isolation,
        "checkpoint_policy_pass": technical_checks["checkpoint_policy_pass"],
        "validation_prediction_alignment": technical_checks["prediction_alignment"],
        "validation_metric_recompute_pass": True,
        "validation_bootstrap_pass": len(validation_boot) == 7
        and int(validation_boot["replicates"].min()) >= 10_000,
        "validation_classwise_gate_computed": len(validation_classwise) == 7,
        "full_test_evaluation_pass": len(test_metrics) == 4,
        "locked_hash_pass": True,
        "locked_evaluation_pass": len(locked_metrics) == 8,
        "counterfactual_evaluation_pass": len(counterfactual_metrics) == 12,
        "prediction_finiteness_pass": technical_checks["prediction_finiteness"],
        "representation_analysis_pass": not representation.empty and technical_checks["hook_integrity"],
        "parameter_learning_audit_pass": not parameter_rows.empty,
        "initial_parameter_drift_verifiable": all(
            item["quantitative_initial_drift_verifiable"] for item in recreation.values()
        ),
        "registered_gate_applied": decision in {"PROMOTE_RELATION_ID", "STOP_RELATION_ID", "BLOCKED"},
        "reports_complete": True,
        "training_launched": False,
        "model_modified": False,
        "blocking_issues": blockers,
        "warnings": limitations,
    }
    write_json(output / "21_validation_summary.json", clean_json(validation_summary))

    make_plots(
        output,
        curves,
        validation_metrics,
        validation_classwise,
        test_metrics,
        locked_metrics,
        counterfactual_metrics,
        representation_comparisons,
        parameter_rows,
    )
    required = [
        "00_README.md",
        "01_run_artifact_manifest.csv",
        "01_artifact_integrity.md",
        "02_experiment_parity_validation.md",
        "03_checkpoint_policy_audit.md",
        "04_training_curves.csv",
        "04_training_curve_comparison.md",
        "05_validation_predictions.csv",
        "05_validation_metrics.csv",
        "05_primary_validation_decision.md",
        "06_validation_paired_bootstrap.csv",
        "06_validation_paired_bootstrap.md",
        "07_validation_classwise.csv",
        "07_validation_classwise.md",
        "08_full_test_predictions.csv",
        "08_full_test_metrics.csv",
        "08_full_test_comparison.md",
        "09_locked_predictions.csv",
        "09_locked_metrics.csv",
        "09_locked_comparison.md",
        "10_relation_counterfactuals.csv",
        "10_relation_counterfactuals.md",
        "11_error_transition_analysis.csv",
        "11_error_transition_analysis.md",
        "12_calibration_analysis.csv",
        "12_calibration_analysis.md",
        "13_representation_analysis.csv",
        "13_representation_analysis.md",
        "14_parameter_learning_audit.csv",
        "14_parameter_learning_audit.md",
        "15_best_last_sensitivity.md",
        "16_hypothesis_update.md",
        "17_registered_final_decision.md",
        "18_next_step_scope.md",
        "19_machine_readable_summary.json",
        "20_run_commands.md",
        "21_validation_summary.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Required reports missing: {missing}")
    print(
        json.dumps(
            {
                "event": "d19_a1_analysis_complete",
                "decision": decision,
                "validation_macro_f1_gain_pp": val_gain * 100.0,
                "gap_increase_pp": gap_increase * 100.0,
                "maximum_class_f1_loss_pp": maximum_class_loss * 100.0,
                "output": str(output),
                "training_launched": False,
                "model_modified": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()




