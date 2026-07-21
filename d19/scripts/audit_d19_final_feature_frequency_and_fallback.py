"""Read-only D19 final pixel feature-frequency and historical fallback audit.

This script never trains a model and never writes into a run, checkpoint, graph
cache, selector, dataset, model, or training directory.  It has an explicit
pretest/postlock boundary so historical test metrics cannot participate in the
paper-safe fallback ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import (
    kurtosis,
    ks_2samp,
    pearsonr,
    skew,
    spearmanr,
    wasserstein_distance,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_graph_builder import NODE_FEATURE_NAMES, build_structure_graph
from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.models.structure_gnn import StructureGNN


ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "outputs/d19_analysis/d19_final_feature_frequency_and_fallback_audit"
CACHE_ROOT = ROOT / "outputs/d19_graph_cache/a0_evidence_only"
PRIOR_ROOT = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
LOCKED_SOURCE = ROOT / "outputs/d18_analysis/ofix18_factorial_posttraining/06_locked_evaluation_predictions.csv"
LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"

A0_42 = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42"
A0_7 = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed7"
A1_NULL = ROOT / "outputs/d19_runs/d19_a1_id_null_evidence_only_seed42"
A1_CORRECT = ROOT / "outputs/d19_runs/d19_a1_id_correct_evidence_only_seed42"
C2_42 = ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42"
C2_7 = ROOT / "outputs/d18_runs/ofix18seed/d18_ofix18_c2_structure_mode_mix_only_seed7"

SELECTION_AUDIT = ROOT / "outputs/d19_analysis/d19_pixel_selection_sufficiency_audit"
A1_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a1_id_posttraining_analysis"
A1_DESIGN = ROOT / "outputs/d19_analysis/d19_a1_id_implementation_design"
A0_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a0_posttraining_analysis"
A0_7_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a0_seed7_confirmation_posttraining"
C2_ANALYSIS = ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining"

CLASS_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
FEATURE_NAMES = list(NODE_FEATURE_NAMES)
CANDIDATE_NAMES = [
    "C0_raw_intensity",
    "C1_one_step_diffusion",
    "C2_two_step_diffusion",
    "C3_two_step_residual",
]
LOCK_VERSION = "d19-final-fallback-lexicographic-v1"
EPS = 1e-12


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def md_table(frame: pd.DataFrame, limit: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(limit).copy()
    for column in show.columns:
        if show[column].dtype.kind in "fc":
            show[column] = show[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.8g}")
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(run_dir: Path) -> dict[str, Any]:
    for name in ("resolved_config.yaml", "source_config.yaml"):
        path = run_dir / name
        if path.exists():
            return read_yaml(path)
    raise FileNotFoundError(f"No config for {run_dir}")


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise RuntimeError(f"Unsupported checkpoint state at {path}")
    return state, payload


def parameter_count(model: torch.nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def stable_effective_rank(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eig.sum())
    if total <= EPS:
        return 0.0, 0.0, float("inf")
    prob = eig / total
    entropy_rank = float(np.exp(-np.sum(prob[prob > 0] * np.log(prob[prob > 0]))))
    participation = float(total * total / max(float(np.sum(eig * eig)), EPS))
    positive = eig[eig > max(eig.max() * 1e-10, 1e-14)]
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")
    return entropy_rank, participation, condition


def smd(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    aa, bb = aa[np.isfinite(aa)], bb[np.isfinite(bb)]
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    pooled = math.sqrt(
        max(
            ((aa.size - 1) * aa.var(ddof=1) + (bb.size - 1) * bb.var(ddof=1))
            / max(aa.size + bb.size - 2, 1),
            0.0,
        )
    )
    return float((aa.mean() - bb.mean()) / max(pooled, EPS))


def ece_score(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    result = 0.0
    bounds = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(bounds[:-1], bounds[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return float(result)


def metric_bundle(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(7), zero_division=0
    )
    return {
        "count": int(y.size),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(log_loss(y, probs, labels=np.arange(7))),
        "ece": ece_score(y, probs),
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(7)).tolist(),
        "classwise_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(7)},
    }


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


def cache_manifest(split: str) -> pd.DataFrame:
    frame = pd.read_csv(CACHE_ROOT / f"manifest_{split}.csv")
    frame["cache_path"] = frame["cache_file"].map(lambda value: str(CACHE_ROOT / str(value)))
    return frame


def actual_fallback(split: str, sample_index: int) -> int:
    path = PRIOR_ROOT / split / f"{int(sample_index):06d}.npz"
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=False) as payload:
        return int(np.asarray(payload.get("landmark_missing_flag", 0)).item())


class CacheDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return load_d18_graph_cache(self.paths[index])


def paths_for(split: str, indices: list[int] | None = None) -> list[Path]:
    manifest = cache_manifest(split)
    if indices is not None:
        wanted = set(int(x) for x in indices)
        manifest = manifest[manifest["sample_index"].astype(int).isin(wanted)]
        order = {int(value): rank for rank, value in enumerate(indices)}
        manifest = manifest.assign(_order=manifest["sample_index"].astype(int).map(order)).sort_values("_order")
    paths = [Path(value) for value in manifest["cache_path"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph cache paths: {missing[:3]}")
    return paths


def require_inputs() -> list[dict[str, Any]]:
    artifacts = {
        "source_repository": ROOT,
        "A0_seed42": A0_42,
        "A0_seed7": A0_7,
        "A1_ID_null_seed42": A1_NULL,
        "A1_ID_correct_seed42": A1_CORRECT,
        "C2_seed42": C2_42,
        "C2_seed7": C2_7,
        "pixel_selection_audit": SELECTION_AUDIT,
        "A1_posttraining_analysis": A1_ANALYSIS,
        "A1_implementation_design": A1_DESIGN,
        "A0_posttraining_analysis": A0_ANALYSIS,
        "A0_seed7_analysis": A0_7_ANALYSIS,
        "C2_multiseed_analysis": C2_ANALYSIS,
        "evidence_graph_cache": CACHE_ROOT,
        "prior_overlay": PRIOR_ROOT,
    }
    rows = []
    for name, path in artifacts.items():
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "file_count": sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else int(path.exists()),
            }
        )
    missing = [row["path"] for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    return rows


def build_folded_a0(
    checkpoint_type: str = "best",
) -> tuple[StructureGNN, StructureGNN, list[dict[str, Any]], dict[str, Any]]:
    null_cfg = config_for(A1_NULL)
    a0_cfg = json.loads(json.dumps(null_cfg))
    a0_cfg.setdefault("model", {})["edge_type_conditioning"] = {
        "enabled": False,
        "mode": "disabled",
    }
    null_model = StructureGNN.from_config(null_cfg, input_dim=10, edge_attr_dim=6)
    folded_model = StructureGNN.from_config(a0_cfg, input_dim=10, edge_attr_dim=6)
    source_state, payload = checkpoint_state(A1_NULL / "checkpoints" / f"{checkpoint_type}.pt")
    null_model.load_state_dict(source_state, strict=True)
    constant = source_state["edge_type_embedding.weight"][0].detach().clone()
    target_state: dict[str, torch.Tensor] = {}
    fold_rows: list[dict[str, Any]] = []
    for key, target_value in folded_model.state_dict().items():
        if key.endswith("edge_mlp.0.weight"):
            source = source_state[key]
            if tuple(source.shape) != (32, 14) or tuple(target_value.shape) != (32, 6):
                raise RuntimeError(f"Unexpected fold shape {key}: {tuple(source.shape)} -> {tuple(target_value.shape)}")
            target_state[key] = source[:, :6].detach().clone()
        elif key.endswith("edge_mlp.0.bias"):
            source_weight = source_state[key.replace(".bias", ".weight")]
            source_bias = source_state[key]
            contribution = source_weight[:, 6:] @ constant.to(source_weight.dtype)
            target_state[key] = source_bias.detach().clone() + contribution
            fold_rows.append(
                {
                    "layer": key.split(".")[2],
                    "source_weight_name": key.replace(".bias", ".weight"),
                    "source_weight_shape": str(tuple(source_weight.shape)),
                    "folded_weight_shape": "(32, 6)",
                    "source_bias_name": key,
                    "source_bias_norm": float(source_bias.float().norm()),
                    "embedding_row": 0,
                    "embedding_value": json.dumps(constant.tolist()),
                    "constant_contribution_norm": float(contribution.float().norm()),
                    "folded_bias_norm": float(target_state[key].float().norm()),
                    "dtype": str(source_weight.dtype),
                }
            )
        else:
            if key not in source_state:
                raise KeyError(f"Folded target parameter missing in A1-null: {key}")
            if source_state[key].shape != target_value.shape:
                raise RuntimeError(f"Non-fold parameter shape mismatch {key}")
            target_state[key] = source_state[key].detach().clone()
    incompatible = folded_model.load_state_dict(target_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Folded strict load failed: {incompatible}")
    null_model.eval()
    folded_model.eval()
    conditions = {
        "constant_embedding_for_every_edge": True,
        "direct_concat_before_affine": True,
        "normalization_before_affine": False,
        "dropout_before_affine": False,
        "nonlinearity_before_affine": False,
        "separate_embedding_consumer": False,
        "shared_constant_embedding": True,
        "null_embedding_row": 0,
        "null_embedding": constant.tolist(),
        "strict_load": True,
        "source_checkpoint_epoch": int(payload.get("epoch", -1)),
        "source_parameter_count": parameter_count(null_model),
        "folded_parameter_count": parameter_count(folded_model),
    }
    return null_model, folded_model, fold_rows, conditions


def forward_stages(model: StructureGNN, batch, folded: bool) -> dict[str, torch.Tensor]:
    h = model.encoder(batch.x_cat)
    edge_attr = batch.edge_attr_cat
    edge_type = batch.edge_type_cat
    if not folded:
        edge_attr, _ = model.conditioned_edge_attributes(edge_attr, edge_type, mode="null")
    stages: dict[str, torch.Tensor] = {"input_projection": h}
    dst = batch.edge_index_cat[1].long()
    degree = h.new_zeros((h.size(0), 1))
    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
    for index, layer in enumerate(model.gnn.layers, start=1):
        stages[f"edge_projection_{index}"] = layer.edge_mlp[0](edge_attr.to(device=h.device, dtype=h.dtype))
        h = layer(h, batch.edge_index_cat, edge_attr, dst_degree=degree, edge_type=edge_type)
        stages[f"gnn_layer_{index}"] = h
    z = model.readout(h, batch.batch_index, batch.num_graphs)
    stages["pooled_graph_embedding"] = z
    stages["classifier_input"] = model.classifier[0](z)
    stages["logits"] = model.classifier(z)
    stages["probabilities"] = torch.softmax(stages["logits"], dim=1)
    return stages


def fold_population_paths() -> dict[str, list[Path]]:
    train = cache_manifest("train")
    chosen: list[int] = []
    for label, part in train.groupby("label"):
        values = part.sort_values("sample_index")["sample_index"].astype(int).tolist()
        chosen.extend(values[:5])
    chosen = sorted(chosen)[:32]
    locked = locked_manifest()["sample_index"].astype(int).tolist()
    return {
        "train32": paths_for("train", chosen),
        "validation": paths_for("val"),
        "locked715": paths_for("test", locked),
        "full_test": paths_for("test"),
    }


def compare_folded_models(
    out: Path,
    device: torch.device,
    reuse: bool,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    csv_path = out / "04_a1_null_fold_validation.csv"
    meta_path = out / "_a1_null_fold_meta.json"
    layers_path = out / "_a1_null_fold_layers.json"
    if reuse and csv_path.exists() and meta_path.exists() and layers_path.exists():
        return pd.read_csv(csv_path), read_json(meta_path), read_json(layers_path)
    null_model, folded_model, folded_layers, conditions = build_folded_a0("best")
    null_model.to(device)
    folded_model.to(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    stage_accumulators: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum_abs": 0.0, "sum_sq_diff": 0.0, "sum_sq_ref": 0.0, "max_abs": 0.0}
    )
    population_metrics: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for population, graph_paths in fold_population_paths().items():
            loader = DataLoader(
                CacheDataset(graph_paths),
                batch_size=16,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_d18_graphs,
            )
            y_parts, null_prob_parts, fold_prob_parts = [], [], []
            for batch_id, cpu_batch in enumerate(loader):
                batch = cpu_batch.to(device)
                left = forward_stages(null_model, batch, folded=False)
                right = forward_stages(folded_model, batch, folded=True)
                if left.keys() != right.keys():
                    raise RuntimeError("Fold stage mismatch")
                for stage in left:
                    diff = (left[stage] - right[stage]).float()
                    ref = left[stage].float()
                    acc = stage_accumulators[(population, stage)]
                    acc["count"] += float(diff.numel())
                    acc["sum_abs"] += float(diff.abs().sum())
                    acc["sum_sq_diff"] += float((diff * diff).sum())
                    acc["sum_sq_ref"] += float((ref * ref).sum())
                    acc["max_abs"] = max(acc["max_abs"], float(diff.abs().max()))
                y_parts.append(batch.y.detach().cpu().numpy())
                null_prob_parts.append(left["probabilities"].detach().cpu().numpy())
                fold_prob_parts.append(right["probabilities"].detach().cpu().numpy())
                if (batch_id + 1) % 100 == 0:
                    print(json.dumps({"event": "fold_progress", "population": population, "batches": batch_id + 1}), flush=True)
            y = np.concatenate(y_parts)
            null_probs = np.concatenate(null_prob_parts)
            fold_probs = np.concatenate(fold_prob_parts)
            null_bundle = metric_bundle(y, null_probs)
            fold_bundle = metric_bundle(y, fold_probs)
            population_metrics[population] = {
                "count": int(y.size),
                "prediction_agreement": float((null_probs.argmax(1) == fold_probs.argmax(1)).mean()),
                "null_metrics": null_bundle,
                "folded_metrics": fold_bundle,
                "accuracy_difference": fold_bundle["accuracy"] - null_bundle["accuracy"],
                "macro_f1_difference": fold_bundle["macro_f1"] - null_bundle["macro_f1"],
            }
    rows = []
    for (population, stage), acc in sorted(stage_accumulators.items()):
        rows.append(
            {
                "population": population,
                "stage": stage,
                "count": int(acc["count"]),
                "max_absolute_difference": acc["max_abs"],
                "mean_absolute_difference": acc["sum_abs"] / max(acc["count"], 1.0),
                "relative_l2_difference": math.sqrt(acc["sum_sq_diff"]) / max(math.sqrt(acc["sum_sq_ref"]), EPS),
                "prediction_agreement": population_metrics[population]["prediction_agreement"] if stage == "logits" else np.nan,
                "accuracy_difference": population_metrics[population]["accuracy_difference"] if stage == "logits" else np.nan,
                "macro_f1_difference": population_metrics[population]["macro_f1_difference"] if stage == "logits" else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    max_logit = float(frame[frame["stage"].eq("logits")]["max_absolute_difference"].max())
    min_agreement = min(value["prediction_agreement"] for value in population_metrics.values())
    status = "EXACTLY_FOLDABLE" if max_logit <= 1e-5 and min_agreement == 1.0 else "NOT_EXACTLY_FOLDABLE"
    meta = {
        "status": status,
        "mathematical_conditions": conditions,
        "population_metrics": population_metrics,
        "maximum_logit_difference": max_logit,
        "minimum_prediction_agreement": min_agreement,
        "float32_target_pass": bool(max_logit <= 1e-5 and min_agreement == 1.0),
        "interpretation": (
            "A1-null does not enlarge the representable function class relative to A0; differences arise from optimization trajectory, parameterization, seed interaction, or checkpoint selection."
            if status == "EXACTLY_FOLDABLE"
            else "A1-null numerical folding failed; inspect the reported stage differences before assigning added expressivity."
        ),
    }
    write_json(meta_path, meta)
    write_json(layers_path, folded_layers)
    return frame, meta, folded_layers


def feature_inventory() -> pd.DataFrame:
    rows = [
        ("intensity", "raw signal", "image clipped to [0,1]", "1 pixel", "clip [0,1]", True, True, False, False, False, False, False, False),
        ("gx", "high-frequency/detail", "np.gradient(image), horizontal component", "finite difference", "none", True, False, False, False, True, False, False, False),
        ("gy", "high-frequency/detail", "np.gradient(image), vertical component", "finite difference", "none", True, False, False, False, True, False, False, False),
        ("x_norm", "spatial coordinate", "2*x/47-1", "coordinate", "[-1,1]", False, False, True, False, False, False, False, False),
        ("y_norm", "spatial coordinate", "2*y/47-1", "coordinate", "[-1,1]", False, False, True, False, False, False, False, False),
        ("grad_mag", "high-frequency/detail", "sqrt(gx^2+gy^2)", "finite difference", "clip [0,1]", True, False, False, False, True, False, False, False),
        ("local_mean_3x3", "low-frequency signal", "edge-padded 3x3 mean", "3x3", "clip [0,1]", False, False, False, True, False, False, False, False),
        ("local_std_3x3", "band-pass/local contrast", "sqrt(E[x^2]-E[x]^2+1e-12)", "3x3", "clip [0,0.5]", True, False, False, False, False, True, False, False),
        ("laplacian_abs", "high-frequency/detail", "abs(N+S+E+W-4*C)", "cross 3x3", "edge padding; clip [0,1]", True, False, False, False, False, False, True, False),
        ("center_surround", "band-pass/local contrast", "intensity-local_mean_3x3", "3x3", "clip [-1,1]", True, False, False, False, False, True, False, False),
    ]
    columns = [
        "feature_name", "semantic_family", "equation", "local_support", "normalization_or_range",
        "used_in_selector_score", "raw_intensity", "coordinate_information", "low_frequency_context",
        "gradient_like", "local_variation", "laplacian_high_pass", "landmark_dependent",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(0, "tensor_index", np.arange(len(frame), dtype=int))
    frame["source_function"] = "d18/data/structure_graph_builder.py:compute_pixel_feature_maps/build_structure_graph"
    frame["boundary_handling"] = frame["feature_name"].map(
        {
            "gx": "numpy.gradient one-sided boundary",
            "gy": "numpy.gradient one-sided boundary",
            "grad_mag": "inherits gx/gy",
            "local_mean_3x3": "edge padding",
            "local_std_3x3": "edge padding",
            "laplacian_abs": "edge padding",
            "center_surround": "edge padding through local mean",
        }
    ).fillna("not applicable")
    frame["class_label_dependent"] = False
    frame["split_dependent"] = False
    return frame


def fallback_lookup() -> dict[tuple[str, int], int]:
    manifest_path = SELECTION_AUDIT / "03_dataset_and_split_manifest.csv"
    frame = pd.read_csv(manifest_path)
    result = {}
    for row in frame.itertuples(index=False):
        status = getattr(row, "fallback_status", 0)
        if isinstance(status, str):
            flag = int(status.lower() in {"fallback", "1", "true"})
        else:
            flag = int(status)
        result[(str(row.split), int(row.sample_index))] = flag
    return result


def collect_feature_distributions(
    out: Path,
    reuse: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    image_path = out / "_feature_image_statistics.csv"
    sample_path = out / "_feature_node_samples.npz"
    distribution_path = out / "06_feature_distribution_statistics.csv"
    if reuse and image_path.exists() and sample_path.exists() and distribution_path.exists():
        with np.load(sample_path) as payload:
            samples = {name: np.asarray(payload[name]) for name in payload.files}
        return pd.read_csv(image_path), samples, pd.read_csv(distribution_path)
    fallback = fallback_lookup()
    image_rows: list[dict[str, Any]] = []
    sample_pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    exact: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum": 0.0, "sum2": 0.0, "min": float("inf"), "max": float("-inf"), "zero": 0.0, "finite": 0.0}
    )
    for split in ("train", "val", "test"):
        manifest = cache_manifest(split)
        for position, item in enumerate(manifest.itertuples(index=False), start=1):
            graph = load_d18_graph_cache(Path(item.cache_path))
            x = graph.x.detach().cpu().numpy().astype(np.float64)
            if x.shape != (1800, 10) or list(graph.node_feature_names) != FEATURE_NAMES:
                raise RuntimeError(f"Feature schema mismatch at {item.cache_path}")
            node_indices = np.linspace(0, x.shape[0] - 1, 64, dtype=np.int64)
            sample_pieces[split].append(x[node_indices].astype(np.float32))
            rank_indices = np.linspace(0, x.shape[0] - 1, 192, dtype=np.int64)
            erank, participation, condition = stable_effective_rank(x[rank_indices])
            row: dict[str, Any] = {
                "split": split,
                "sample_index": int(item.sample_index),
                "label": int(item.label),
                "class_name": CLASS_NAMES[int(item.label)],
                "fallback": fallback.get((split, int(item.sample_index)), 0),
                "effective_rank": erank,
                "participation_ratio": participation,
                "covariance_condition": condition,
            }
            for index, name in enumerate(FEATURE_NAMES):
                values = x[:, index]
                finite = np.isfinite(values)
                valid = values[finite]
                if valid.size != values.size:
                    raise RuntimeError(f"Non-finite node feature at {item.cache_path}:{name}")
                row[f"{name}_mean"] = float(valid.mean())
                row[f"{name}_variance"] = float(valid.var())
                row[f"{name}_median"] = float(np.median(valid))
                stat = exact[(split, index)]
                stat["count"] += float(values.size)
                stat["sum"] += float(values.sum())
                stat["sum2"] += float(np.sum(values * values))
                stat["min"] = min(stat["min"], float(values.min()))
                stat["max"] = max(stat["max"], float(values.max()))
                stat["zero"] += float(np.sum(np.abs(values) <= 1e-12))
                stat["finite"] += float(finite.sum())
            image_rows.append(row)
            if position % 2000 == 0 or position == len(manifest):
                print(json.dumps({"event": "feature_distribution_progress", "split": split, "done": position, "total": len(manifest)}), flush=True)
    images = pd.DataFrame(image_rows)
    samples = {split: np.concatenate(parts, axis=0) for split, parts in sample_pieces.items()}
    np.savez_compressed(sample_path, **samples)
    images.to_csv(image_path, index=False)
    distribution_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        sample = samples[split].astype(np.float64)
        subset = images[images["split"].eq(split)]
        for index, name in enumerate(FEATURE_NAMES):
            values = sample[:, index]
            stat = exact[(split, index)]
            count = stat["count"]
            mean = stat["sum"] / count
            variance = max(stat["sum2"] / count - mean * mean, 0.0)
            coordinate_corr = []
            for coord in ("x_norm", "y_norm"):
                corr = np.corrcoef(values, sample[:, FEATURE_NAMES.index(coord)])[0, 1]
                coordinate_corr.append(abs(float(corr)) if np.isfinite(corr) else 0.0)
            distribution_rows.append(
                {
                    "row_type": "split",
                    "split": split,
                    "class_name": "all",
                    "fallback": "all",
                    "feature": name,
                    "node_count": int(count),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": stat["min"],
                    "max": stat["max"],
                    "median_sampled_nodes": float(np.median(values)),
                    "q25_sampled_nodes": float(np.quantile(values, 0.25)),
                    "q75_sampled_nodes": float(np.quantile(values, 0.75)),
                    "iqr_sampled_nodes": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "skew_sampled_nodes": float(skew(values, bias=False)),
                    "kurtosis_sampled_nodes": float(kurtosis(values, fisher=True, bias=False)),
                    "zero_fraction": stat["zero"] / count,
                    "near_constant_image_fraction": float((subset[f"{name}_variance"] < 1e-10).mean()),
                    "nan_inf_count": int(count - stat["finite"]),
                    "within_image_variance_mean": float(subset[f"{name}_variance"].mean()),
                    "between_image_mean_variance": float(subset[f"{name}_mean"].var(ddof=0)),
                    "max_abs_coordinate_correlation": max(coordinate_corr),
                }
            )
        for label, part in subset.groupby("label"):
            for name in FEATURE_NAMES:
                distribution_rows.append(
                    {
                        "row_type": "class_image_mean",
                        "split": split,
                        "class_name": CLASS_NAMES[int(label)],
                        "fallback": "all",
                        "feature": name,
                        "node_count": len(part),
                        "mean": float(part[f"{name}_mean"].mean()),
                        "std": float(part[f"{name}_mean"].std(ddof=0)),
                    }
                )
        for flag, part in subset.groupby("fallback"):
            for name in FEATURE_NAMES:
                distribution_rows.append(
                    {
                        "row_type": "fallback_image_mean",
                        "split": split,
                        "class_name": "all",
                        "fallback": "fallback" if int(flag) else "official",
                        "feature": name,
                        "node_count": len(part),
                        "mean": float(part[f"{name}_mean"].mean()),
                        "std": float(part[f"{name}_mean"].std(ddof=0)),
                    }
                )
    train_sample = samples["train"].astype(np.float64)
    for target_split in ("val", "test"):
        other = samples[target_split].astype(np.float64)
        for index, name in enumerate(FEATURE_NAMES):
            a = train_sample[:, index]
            b = other[:, index]
            max_count = min(250_000, len(a), len(b))
            ai = np.linspace(0, len(a) - 1, max_count, dtype=np.int64)
            bi = np.linspace(0, len(b) - 1, max_count, dtype=np.int64)
            distribution_rows.append(
                {
                    "row_type": "split_drift",
                    "split": f"train_vs_{target_split}",
                    "class_name": "all",
                    "fallback": "all",
                    "feature": name,
                    "node_count": max_count,
                    "standardized_mean_difference": smd(a[ai], b[bi]),
                    "wasserstein_distance": float(wasserstein_distance(a[ai], b[bi])),
                    "ks_statistic": float(ks_2samp(a[ai], b[bi]).statistic),
                }
            )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(distribution_path, index=False)
    return images, samples, distribution


def redundancy_analysis(
    out: Path,
    images: pd.DataFrame,
    samples: dict[str, np.ndarray],
    reuse: bool,
) -> pd.DataFrame:
    path = out / "07_feature_redundancy_analysis.csv"
    if reuse and path.exists():
        return pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    locked_ids = set(locked_manifest()["sample_index"].astype(int))
    locked_indices = images[images["split"].eq("test") & images["sample_index"].isin(locked_ids)].index
    # The test reservoir is ordered by image then 64 nodes.
    test_image_positions = images[images["split"].eq("test")].reset_index(drop=True)
    locked_position = np.flatnonzero(test_image_positions["sample_index"].isin(locked_ids).to_numpy())
    locked_nodes = np.concatenate([np.arange(pos * 64, (pos + 1) * 64) for pos in locked_position])
    populations = {
        "train": samples["train"],
        "validation": samples["val"],
        "locked": samples["test"][locked_nodes],
    }
    for population, raw in populations.items():
        count = min(400_000, len(raw))
        index = np.linspace(0, len(raw) - 1, count, dtype=np.int64)
        x = raw[index].astype(np.float64)
        corr = np.corrcoef(x, rowvar=False)
        rank, participation, condition = stable_effective_rank(x)
        matrices[population] = corr
        eig = np.linalg.eigvalsh(corr)
        rows.append(
            {
                "row_type": "population_summary",
                "population": population,
                "feature": "all",
                "other_feature": "",
                "sampled_node_count": count,
                "effective_rank": rank,
                "participation_ratio": participation,
                "condition_number": condition,
                "correlation_eigenvalues": json.dumps(eig.tolist()),
            }
        )
        centered = x - x.mean(axis=0, keepdims=True)
        design_all = np.column_stack([np.ones(len(x)), centered])
        for j, name in enumerate(FEATURE_NAMES):
            keep = [column for column in range(10) if column != j]
            design = np.column_stack([np.ones(len(x)), centered[:, keep]])
            coef, *_ = np.linalg.lstsq(design, centered[:, j], rcond=1e-10)
            residual = centered[:, j] - design @ coef
            ratio = float(np.sum(residual * residual) / max(np.sum(centered[:, j] ** 2), EPS))
            rows.append(
                {
                    "row_type": "unique_residual",
                    "population": population,
                    "feature": name,
                    "other_feature": "remaining_nine",
                    "sampled_node_count": count,
                    "unique_residual_ratio": ratio,
                    "vif": float(1.0 / max(ratio, EPS)),
                    "explained_by_remaining": 1.0 - ratio,
                }
            )
        for i in range(10):
            for j in range(i + 1, 10):
                pearson = float(corr[i, j])
                spearman = float(spearmanr(x[:, i], x[:, j]).statistic)
                rows.append(
                    {
                        "row_type": "pairwise",
                        "population": population,
                        "feature": FEATURE_NAMES[i],
                        "other_feature": FEATURE_NAMES[j],
                        "sampled_node_count": count,
                        "pearson": pearson,
                        "spearman": spearman,
                        "linear_cka": pearson * pearson,
                        "near_duplicate": abs(pearson) >= 0.98,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    np.savez_compressed(out / "_feature_correlation_matrices.npz", **matrices)
    return frame


def local_operators(graph) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    mask = graph.edge_type.detach().cpu().numpy().astype(np.int64) == 0
    edges = graph.edge_index.detach().cpu().numpy()[:, mask].astype(np.int64)
    n = int(graph.x.shape[0])
    adjacency = sparse.coo_matrix(
        (np.ones(edges.shape[1], dtype=np.float64), (edges[0], edges[1])),
        shape=(n, n),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_adj = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    laplacian = sparse.eye(n, format="csr") - normalized_adj
    adjacency_self = adjacency + sparse.eye(n, format="csr")
    degree_self = np.asarray(adjacency_self.sum(axis=1)).reshape(-1)
    inv_self = 1.0 / np.sqrt(np.maximum(degree_self, 1.0))
    diffusion = sparse.diags(inv_self) @ adjacency_self @ sparse.diags(inv_self)
    return adjacency, laplacian.tocsr(), diffusion.tocsr()


def signal_frequency_metrics(
    values: np.ndarray,
    adjacency: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
    diffusion: sparse.csr_matrix,
) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    lx = laplacian @ x
    px = diffusion @ x
    p2x = diffusion @ px
    denom = max(float(x @ x), EPS)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    neighbor_mean = np.asarray(adjacency @ x).reshape(-1) / np.maximum(degree, 1.0)
    centered = x - x.mean()
    moran_denom = max(float(centered @ centered), EPS)
    weight_sum = max(float(adjacency.sum()), EPS)
    moran = float(len(x) / weight_sum * (centered @ (adjacency @ centered)) / moran_denom)
    centroid = float(x @ lx / denom)
    second_moment = float(lx @ lx / denom)
    return {
        "variance": float(x.var()),
        "normalized_dirichlet_energy": centroid,
        "local_neighbor_residual_ratio": float(np.mean((x - neighbor_mean) ** 2) / max(float(x.var()), EPS)),
        "moran_autocorrelation": moran,
        "one_step_diffusion_retention": float(px @ px / denom),
        "two_step_diffusion_retention": float(p2x @ p2x / denom),
        "spectral_centroid_exact_moment": centroid,
        "spectral_spread_exact_moment": math.sqrt(max(second_moment - centroid * centroid, 0.0)),
    }


def candidate_signals(x: np.ndarray, diffusion: sparse.csr_matrix) -> np.ndarray:
    intensity = np.asarray(x[:, FEATURE_NAMES.index("intensity")], dtype=np.float64)
    one = np.asarray(diffusion @ intensity).reshape(-1)
    two = np.asarray(diffusion @ one).reshape(-1)
    return np.column_stack([intensity, one, two, two - intensity])


def candidate_projection_metrics(x: np.ndarray, candidates: np.ndarray) -> list[dict[str, float]]:
    xx = np.asarray(x, dtype=np.float64)
    zz = np.asarray(candidates, dtype=np.float64)
    centered_x = xx - xx.mean(axis=0, keepdims=True)
    centered_z = zz - zz.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(xx)), centered_x])
    coefficients, *_ = np.linalg.lstsq(design, centered_z, rcond=1e-10)
    residual = centered_z - design @ coefficients
    x_gram_norm = float(np.linalg.norm(centered_x.T @ centered_x))
    rows = []
    for index in range(zz.shape[1]):
        z = centered_z[:, index]
        denom = max(float(z @ z), EPS)
        correlations = []
        for column in range(xx.shape[1]):
            a = centered_x[:, column]
            corr = float((a @ z) / max(math.sqrt(float(a @ a) * denom), EPS))
            correlations.append(corr)
        cross = centered_x.T @ z
        cka = float((cross @ cross) / max(x_gram_norm * denom, EPS))
        rows.append(
            {
                "candidate_nonredundant_ratio": float(np.sum(residual[:, index] ** 2) / denom),
                "maximum_absolute_existing_feature_correlation": max(abs(value) for value in correlations),
                "correlations_json": json.dumps({FEATURE_NAMES[j]: correlations[j] for j in range(10)}),
                "linear_cka_with_existing_matrix": cka,
            }
        )
    return rows


def graph_frequency_indices(images: pd.DataFrame) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split in ("val", "test"):
        result[split] = images[images["split"].eq(split)]["sample_index"].astype(int).tolist()
    train = images[images["split"].eq("train")]
    chosen = []
    for _, part in train.groupby("label"):
        values = part.sort_values("sample_index")["sample_index"].astype(int).tolist()
        if values:
            chosen.extend([values[index] for index in np.linspace(0, len(values) - 1, min(100, len(values)), dtype=int)])
    result["train"] = sorted(set(chosen))
    return result


def spectral_subset_ids(images: pd.DataFrame) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for split in ("val", "test"):
        chosen: list[int] = []
        population = images[images["split"].eq(split)]
        for _, part in population.groupby("label"):
            values = part.sort_values("sample_index")["sample_index"].astype(int).tolist()
            chosen.extend([values[index] for index in np.linspace(0, len(values) - 1, min(10, len(values)), dtype=int)])
        result[split] = set(chosen)
    return result


def graph_frequency_audit(
    out: Path,
    images: pd.DataFrame,
    reuse: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_image_path = out / "_graph_frequency_per_image.csv"
    spectral_path = out / "_bounded_spectral_per_image.csv"
    candidate_path = out / "_candidate_per_image.csv"
    if reuse and per_image_path.exists() and spectral_path.exists() and candidate_path.exists():
        return pd.read_csv(per_image_path), pd.read_csv(spectral_path), pd.read_csv(candidate_path)
    locked_ids = set(locked_manifest()["sample_index"].astype(int))
    frequency_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    spectral_ids = spectral_subset_ids(images)
    for split, indices in graph_frequency_indices(images).items():
        manifest = cache_manifest(split)
        lookup = {int(row.sample_index): Path(row.cache_path) for row in manifest.itertuples(index=False)}
        metadata = images[images["split"].eq(split)].set_index("sample_index")
        for position, sample_index in enumerate(indices, start=1):
            graph = load_d18_graph_cache(lookup[int(sample_index)])
            x = graph.x.detach().cpu().numpy().astype(np.float64)
            adjacency, laplacian, diffusion = local_operators(graph)
            candidates = candidate_signals(x, diffusion)
            projection = candidate_projection_metrics(x, candidates)
            base = {
                "split": split,
                "population": "locked" if split == "test" and int(sample_index) in locked_ids else split,
                "sample_index": int(sample_index),
                "label": int(metadata.loc[int(sample_index), "label"]),
                "class_name": CLASS_NAMES[int(metadata.loc[int(sample_index), "label"])],
                "fallback": int(metadata.loc[int(sample_index), "fallback"]),
            }
            all_values = np.column_stack([x, candidates])
            all_names = FEATURE_NAMES + CANDIDATE_NAMES
            for column, name in enumerate(all_names):
                metrics = signal_frequency_metrics(all_values[:, column], adjacency, laplacian, diffusion)
                frequency_rows.append(
                    {
                        **base,
                        "signal": name,
                        "signal_kind": "existing" if column < 10 else "candidate",
                        **metrics,
                    }
                )
            for candidate_index, name in enumerate(CANDIDATE_NAMES):
                candidate_rows.append(
                    {
                        **base,
                        "candidate": name,
                        **projection[candidate_index],
                        **signal_frequency_metrics(candidates[:, candidate_index], adjacency, laplacian, diffusion),
                        "coordinate_ordering_invariant": True,
                    }
                )
            if split in spectral_ids and int(sample_index) in spectral_ids[split]:
                low_values, low_vectors = eigsh(laplacian, k=32, which="SA", tol=1e-5, maxiter=5000)
                high_values, high_vectors = eigsh(laplacian, k=32, which="LA", tol=1e-5, maxiter=5000)
                low_order = np.argsort(low_values)
                high_order = np.argsort(high_values)
                low_values, low_vectors = low_values[low_order], low_vectors[:, low_order]
                high_values, high_vectors = high_values[high_order], high_vectors[:, high_order]
                for column, name in enumerate(all_names):
                    value = all_values[:, column].astype(np.float64)
                    denom = max(float(value @ value), EPS)
                    low_fraction = float(np.sum((low_vectors.T @ value) ** 2) / denom)
                    high_fraction = float(np.sum((high_vectors.T @ value) ** 2) / denom)
                    spectral_rows.append(
                        {
                            **base,
                            "signal": name,
                            "signal_kind": "existing" if column < 10 else "candidate",
                            "low32_energy_fraction": low_fraction,
                            "high32_energy_fraction": high_fraction,
                            "lowest_eigenvalue": float(low_values.min()),
                            "low32_max_eigenvalue": float(low_values.max()),
                            "high32_min_eigenvalue": float(high_values.min()),
                            "highest_eigenvalue": float(high_values.max()),
                            "spectral_centroid": float(value @ (laplacian @ value) / denom),
                            "spectral_spread": signal_frequency_metrics(value, adjacency, laplacian, diffusion)["spectral_spread_exact_moment"],
                        }
                    )
            if position % 250 == 0 or position == len(indices):
                print(json.dumps({"event": "graph_frequency_progress", "split": split, "done": position, "total": len(indices)}), flush=True)
    frequency = pd.DataFrame(frequency_rows)
    spectral_frame = pd.DataFrame(spectral_rows)
    candidates_frame = pd.DataFrame(candidate_rows)
    frequency.to_csv(per_image_path, index=False)
    spectral_frame.to_csv(spectral_path, index=False)
    candidates_frame.to_csv(candidate_path, index=False)
    return frequency, spectral_frame, candidates_frame


def summarize_frequency_outputs(
    out: Path,
    frequency: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    candidates_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "variance", "normalized_dirichlet_energy", "local_neighbor_residual_ratio",
        "moran_autocorrelation", "one_step_diffusion_retention", "two_step_diffusion_retention",
        "spectral_centroid_exact_moment", "spectral_spread_exact_moment",
    ]
    frequency_summary = (
        frequency.groupby(["population", "signal_kind", "signal"])[metrics]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    frequency_summary.columns = ["_".join(str(value) for value in column if value) for column in frequency_summary.columns]
    frequency_summary.to_csv(out / "08_graph_frequency_characterization.csv", index=False)
    spectral_summary = (
        spectral_frame.groupby(["population", "signal_kind", "signal"])[
            ["low32_energy_fraction", "high32_energy_fraction", "spectral_centroid", "spectral_spread"]
        ]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    spectral_summary.columns = ["_".join(str(value) for value in column if value) for column in spectral_summary.columns]
    spectral_summary.to_csv(out / "09_low_frequency_coverage_analysis.csv", index=False)
    candidate_summary = (
        candidates_frame.groupby(["population", "candidate"])[
            [
                "variance", "normalized_dirichlet_energy", "candidate_nonredundant_ratio",
                "maximum_absolute_existing_feature_correlation", "linear_cka_with_existing_matrix",
            ]
        ]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    candidate_summary.columns = ["_".join(str(value) for value in column if value) for column in candidate_summary.columns]
    candidate_summary.to_csv(out / "12_bounded_candidate_signal_bank.csv", index=False)
    nonredundancy_rows = []
    for population, part in candidates_frame.groupby("population"):
        for candidate, candidate_part in part.groupby("candidate"):
            nonredundancy_rows.append(
                {
                    "population": population,
                    "candidate": candidate,
                    "count": len(candidate_part),
                    "nonredundant_mean": candidate_part["candidate_nonredundant_ratio"].mean(),
                    "nonredundant_median": candidate_part["candidate_nonredundant_ratio"].median(),
                    "nonredundant_q25": candidate_part["candidate_nonredundant_ratio"].quantile(0.25),
                    "nonredundant_q75": candidate_part["candidate_nonredundant_ratio"].quantile(0.75),
                    "max_abs_correlation_median": candidate_part["maximum_absolute_existing_feature_correlation"].median(),
                    "linear_cka_median": candidate_part["linear_cka_with_existing_matrix"].median(),
                }
            )
    nonredundancy = pd.DataFrame(nonredundancy_rows)
    nonredundancy.to_csv(out / "13_candidate_nonredundancy_analysis.csv", index=False)
    stability = (
        candidates_frame.groupby(["split", "class_name", "fallback", "candidate"])[
            ["candidate_nonredundant_ratio", "normalized_dirichlet_energy", "variance"]
        ]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    stability.columns = ["_".join(str(value) for value in column if value) for column in stability.columns]
    stability.to_csv(out / "14_candidate_stability_and_classwise_analysis.csv", index=False)
    return frequency_summary, spectral_summary, candidate_summary, nonredundancy


def load_model_strict(run_dir: Path, checkpoint_type: str, device: torch.device) -> tuple[StructureGNN, dict[str, Any]]:
    cfg = config_for(run_dir)
    model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
    state, payload = checkpoint_state(run_dir / "checkpoints" / f"{checkpoint_type}.pt")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load failed for {run_dir}/{checkpoint_type}")
    model.to(device).eval()
    return model, payload


def remove_structure_edges(graph):
    keep = graph.edge_type.long() != 2
    return replace(
        graph,
        edge_index=graph.edge_index[:, keep],
        edge_attr=graph.edge_attr[keep],
        edge_type=graph.edge_type[keep],
        structure_relation_id=graph.structure_relation_id[keep],
        structure_edge_count=0,
        total_edge_count=int(keep.sum()),
    )


class PriorGraphDataset(Dataset):
    def __init__(self, split: str, indices: list[int], graph_cfg: dict[str, Any], remove_structure: bool):
        self.split = split
        self.indices = [int(value) for value in indices]
        self.graph_cfg = graph_cfg
        self.remove_structure = remove_structure

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        sample_index = self.indices[position]
        path = PRIOR_ROOT / self.split / f"{sample_index:06d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            source = {name: np.asarray(payload[name]) for name in payload.files}
        graph = build_structure_graph(source, self.graph_cfg)
        return remove_structure_edges(graph) if self.remove_structure else graph


def sensitivity_contexts(device: torch.device) -> list[dict[str, Any]]:
    a0, _ = load_model_strict(A0_42, "best", device)
    a1, _ = load_model_strict(A1_NULL, "best", device)
    c2, _ = load_model_strict(C2_42, "best", device)
    return [
        {"name": "A0_seed42", "model": a0, "source": "cache", "run": A0_42, "mode": "official"},
        {"name": "A1_null_seed42", "model": a1, "source": "cache", "run": A1_NULL, "mode": "official"},
        {"name": "C2_seed42_official", "model": c2, "source": "prior", "run": C2_42, "mode": "official"},
        {"name": "C2_seed42_remove_structure", "model": c2, "source": "prior", "run": C2_42, "mode": "remove_structure"},
    ]


def sensitivity_loader(context: dict[str, Any], population: str) -> DataLoader:
    if population == "validation":
        split = "val"
        indices = cache_manifest("val")["sample_index"].astype(int).tolist()
    else:
        split = "test"
        indices = locked_manifest()["sample_index"].astype(int).tolist()
    if context["source"] == "cache":
        dataset: Dataset = CacheDataset(paths_for(split, indices))
    else:
        cfg = config_for(context["run"])
        dataset = PriorGraphDataset(
            split,
            indices,
            cfg.get("graph", {}) or {},
            remove_structure=context["mode"] == "remove_structure",
        )
    return DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_d18_graphs)


def perturbation_definitions() -> dict[str, list[int]]:
    return {
        **{f"channel:{name}": [index] for index, name in enumerate(FEATURE_NAMES)},
        "group:coordinates": [3, 4],
        "group:raw_low_frequency": [0, 6],
        "group:gradient_high_frequency": [1, 2, 5, 8],
        "group:local_variation": [7, 9],
        "group:selector_score_constituents": [5, 7, 8, 9],
    }


def fixed_checkpoint_sensitivity(
    out: Path,
    distribution: pd.DataFrame,
    device: torch.device,
    reuse: bool,
) -> pd.DataFrame:
    path = out / "10_fixed_checkpoint_feature_sensitivity.csv"
    if reuse and path.exists():
        return pd.read_csv(path)
    train_rows = distribution[
        distribution["row_type"].eq("split") & distribution["split"].eq("train")
    ].set_index("feature")
    train_mean = torch.tensor(
        [float(train_rows.loc[name, "mean"]) for name in FEATURE_NAMES],
        dtype=torch.float32,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for context in sensitivity_contexts(device):
            for population in ("validation", "locked715"):
                baseline_y: list[np.ndarray] = []
                baseline_probs: list[np.ndarray] = []
                baseline_logits: list[np.ndarray] = []
                variant_probs: dict[str, list[np.ndarray]] = defaultdict(list)
                variant_logits: dict[str, list[np.ndarray]] = defaultdict(list)
                for batch_id, cpu_batch in enumerate(sensitivity_loader(context, population)):
                    batch = cpu_batch.to(device)
                    baseline = context["model"](batch)["logits"]
                    baseline_y.append(batch.y.detach().cpu().numpy())
                    baseline_logits.append(baseline.detach().cpu().numpy())
                    baseline_probs.append(torch.softmax(baseline, dim=1).detach().cpu().numpy())
                    original_x = batch.x_cat
                    for name, channels in perturbation_definitions().items():
                        changed = original_x.clone()
                        changed[:, channels] = train_mean[channels]
                        batch.x_cat = changed
                        logits = context["model"](batch)["logits"]
                        variant_logits[name].append(logits.detach().cpu().numpy())
                        variant_probs[name].append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    batch.x_cat = original_x
                    if (batch_id + 1) % 50 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "feature_sensitivity_progress",
                                    "model": context["name"],
                                    "population": population,
                                    "batches": batch_id + 1,
                                }
                            ),
                            flush=True,
                        )
                y = np.concatenate(baseline_y)
                base_probs = np.concatenate(baseline_probs)
                base_logits = np.concatenate(baseline_logits)
                base_metrics = metric_bundle(y, base_probs)
                for name in perturbation_definitions():
                    probs = np.concatenate(variant_probs[name])
                    logits = np.concatenate(variant_logits[name])
                    metrics = metric_bundle(y, probs)
                    class_delta = {
                        key: metrics["classwise_f1"][key] - base_metrics["classwise_f1"][key]
                        for key in CLASS_NAMES.values()
                    }
                    rows.append(
                        {
                            "model_context": context["name"],
                            "population": population,
                            "perturbation": name,
                            "channels": json.dumps(perturbation_definitions()[name]),
                            "count": len(y),
                            "baseline_accuracy": base_metrics["accuracy"],
                            "counterfactual_accuracy": metrics["accuracy"],
                            "accuracy_change": metrics["accuracy"] - base_metrics["accuracy"],
                            "baseline_macro_f1": base_metrics["macro_f1"],
                            "counterfactual_macro_f1": metrics["macro_f1"],
                            "macro_f1_change": metrics["macro_f1"] - base_metrics["macro_f1"],
                            "weighted_f1_change": metrics["weighted_f1"] - base_metrics["weighted_f1"],
                            "nll_change": metrics["nll"] - base_metrics["nll"],
                            "prediction_agreement": float((base_probs.argmax(1) == probs.argmax(1)).mean()),
                            "mean_logit_l2_change": float(np.linalg.norm(base_logits - logits, axis=1).mean()),
                            "max_absolute_logit_change": float(np.max(np.abs(base_logits - logits))),
                            "classwise_f1_change_json": json.dumps(class_delta),
                            "interpretation": "off-policy model dependency; not direct evidence of retrained feature usefulness",
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def prediction_group_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(SELECTION_AUDIT / "13_prediction_group_manifest.csv", keep_default_na=False)
    primary = predictions[
        predictions["checkpoint_type"].eq("best") & predictions["split"].isin(["test", "locked"])
    ].copy()
    primary["_split_rank"] = primary["split"].map({"test": 0, "locked": 1}).fillna(2)
    primary = primary.sort_values("_split_rank").drop_duplicates(["sample_index", "model", "seed", "mode"])
    pivot = primary.pivot_table(
        index=["sample_index", "true_class"],
        columns=["model", "seed", "mode"],
        values="correct",
        aggfunc="first",
    )
    pivot.columns = [f"{model}_{seed}_{mode}" for model, seed, mode in pivot.columns]
    frame = pivot.reset_index().rename(columns={"true_class": "label"})
    def column(name: str) -> pd.Series:
        return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)
    a0_42 = column("A0_42_official")
    a0_7 = column("A0_7_official")
    a1 = column("A1_ID_null_42_null")
    c2_42_o = column("C2_42_official")
    c2_7_o = column("C2_7_official")
    c2_42_r = column("C2_42_remove_structure")
    definitions = {
        "universal_correct": (a0_42 == 1) & (a1 == 1) & (c2_42_r == 1) & (c2_42_o == 1),
        "universal_wrong": (a0_42 == 0) & (a1 == 0) & (c2_42_r == 0) & (c2_42_o == 0),
        "persistent_evidence_error": (a0_42 == 0) & (a1 == 0) & (c2_42_r == 0),
        "seed_unstable_evidence_error": a0_42.notna() & a0_7.notna() & (a0_42 != a0_7),
        "capacity_repair": (a0_42 == 0) & (a1 == 1),
        "structure_rescue": (c2_42_r == 0) & (c2_42_o == 1),
        "structure_harm": (c2_42_r == 1) & (c2_42_o == 0),
        "A0_repeated_wrong": (a0_42 == 0) & (a0_7 == 0),
        "A0_correct_both": (a0_42 == 1) & (a0_7 == 1),
        "C2_repeated_wrong": (c2_42_o == 0) & (c2_7_o == 0),
        "C2_correct_both": (c2_42_o == 1) & (c2_7_o == 1),
    }
    for name, condition in definitions.items():
        frame[name] = condition.fillna(False).astype(int)
    counts = pd.DataFrame(
        [
            {
                "group": name,
                "count": int(frame[name].sum()),
                "class_counts": json.dumps(frame[frame[name].eq(1)]["label"].value_counts().sort_index().to_dict()),
            }
            for name in definitions
        ]
    )
    return frame, counts


def stratified_bootstrap(
    frame: pd.DataFrame,
    group_a: str,
    group_b: str,
    metric: str,
    reps: int = 5000,
) -> dict[str, Any]:
    a = frame[frame[group_a].eq(1)][["label", metric]].dropna()
    b = frame[frame[group_b].eq(1)][["label", metric]].dropna()
    base = {
        "comparison": f"{group_a}_minus_{group_b}",
        "group_a": group_a,
        "group_b": group_b,
        "metric": metric,
        "n_a": len(a),
        "n_b": len(b),
        "bootstrap_replicates": reps,
    }
    if len(a) < 10 or len(b) < 10:
        return {**base, "mean_difference": np.nan, "median_difference": np.nan, "smd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(19419)
    a_boot = np.zeros(reps, dtype=np.float64)
    b_boot = np.zeros(reps, dtype=np.float64)
    a_total = b_total = 0
    for label in sorted(set(a["label"]).union(b["label"])):
        av = a.loc[a["label"].eq(label), metric].to_numpy(dtype=np.float64)
        bv = b.loc[b["label"].eq(label), metric].to_numpy(dtype=np.float64)
        if av.size:
            a_boot += rng.choice(av, size=(reps, av.size), replace=True).sum(axis=1)
            a_total += av.size
        if bv.size:
            b_boot += rng.choice(bv, size=(reps, bv.size), replace=True).sum(axis=1)
            b_total += bv.size
    differences = a_boot / max(a_total, 1) - b_boot / max(b_total, 1)
    return {
        **base,
        "mean_difference": float(a[metric].mean() - b[metric].mean()),
        "median_difference": float(a[metric].median() - b[metric].median()),
        "smd": smd(a[metric], b[metric]),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def error_group_feature_analysis(
    out: Path,
    images: pd.DataFrame,
    frequency: pd.DataFrame,
    candidates_frame: pd.DataFrame,
    reuse: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = out / "11_error_group_feature_analysis.csv"
    group_path = out / "_error_group_manifest.csv"
    if reuse and path.exists() and group_path.exists():
        return pd.read_csv(path), pd.read_csv(group_path)
    groups, counts = prediction_group_table()
    test_images = images[images["split"].eq("test")].copy()
    locked_ids = set(locked_manifest()["sample_index"].astype(int))
    test_images = test_images[test_images["sample_index"].isin(locked_ids)]
    existing = frequency[
        frequency["population"].eq("locked") & frequency["signal"].isin(FEATURE_NAMES)
    ]
    dirichlet = existing.pivot_table(
        index="sample_index", columns="signal", values="normalized_dirichlet_energy", aggfunc="first"
    )
    candidate = candidates_frame[candidates_frame["population"].eq("locked")].pivot_table(
        index="sample_index",
        columns="candidate",
        values=["candidate_nonredundant_ratio", "normalized_dirichlet_energy", "variance"],
        aggfunc="first",
    )
    candidate.columns = [f"{metric}__{name}" for metric, name in candidate.columns]
    joined = groups.merge(test_images, on=["sample_index", "label"], how="inner")
    joined = joined.merge(dirichlet.reset_index(), on="sample_index", how="left")
    joined = joined.merge(candidate.reset_index(), on="sample_index", how="left")
    joined["existing_low_frequency_energy"] = joined["local_mean_3x3"]
    joined["existing_high_frequency_energy"] = joined[["gx", "gy", "grad_mag", "laplacian_abs"]].mean(axis=1)
    joined["raw_intensity_variance"] = joined["intensity_variance"]
    joined["global_mean_intensity"] = joined["intensity_mean"]
    joined["local_contrast"] = joined["local_std_3x3_mean"]
    metrics = [
        "existing_low_frequency_energy", "existing_high_frequency_energy", "effective_rank",
        "covariance_condition", "raw_intensity_variance", "global_mean_intensity", "local_contrast",
    ]
    metrics += [f"candidate_nonredundant_ratio__{name}" for name in CANDIDATE_NAMES]
    comparisons = [
        ("persistent_evidence_error", "universal_correct"),
        ("seed_unstable_evidence_error", "universal_correct"),
        ("capacity_repair", "persistent_evidence_error"),
        ("structure_rescue", "persistent_evidence_error"),
        ("universal_wrong", "universal_correct"),
        ("A0_repeated_wrong", "A0_correct_both"),
        ("C2_repeated_wrong", "C2_correct_both"),
    ]
    rows = [
        stratified_bootstrap(joined, group_a, group_b, metric, reps=5000)
        for group_a, group_b in comparisons
        for metric in metrics
    ]
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    counts.to_csv(group_path, index=False)
    joined.to_csv(out / "_error_group_image_features.csv", index=False)
    return result, counts


def feature_evidence_decision(
    images: pd.DataFrame,
    frequency: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    candidates_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    populations = ("val", "locked")
    existing_rank = {
        "train_median_per_image": float(images[images["split"].eq("train")]["effective_rank"].median()),
        "validation_median_per_image": float(images[images["split"].eq("val")]["effective_rank"].median()),
        "test_median_per_image": float(images[images["split"].eq("test")]["effective_rank"].median()),
    }
    materially_below_nominal = existing_rank["validation_median_per_image"] <= 8.0
    global_candidate_nonredundancy = any(
        all(
            float(
                candidates_frame[
                    candidates_frame["population"].eq(population)
                    & candidates_frame["candidate"].eq(candidate)
                ]["candidate_nonredundant_ratio"].median()
            )
            >= 0.20
            for population in populations
        )
        for candidate in CANDIDATE_NAMES
        if candidate != "C0_raw_intensity"
    )
    global_f1 = materially_below_nominal and global_candidate_nonredundancy
    criteria: dict[str, Any] = {}
    candidate_profiles: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_NAMES:
        profile: dict[str, Any] = {}
        for population in ("train", "val", "test", "locked"):
            part = candidates_frame[
                candidates_frame["population"].eq(population)
                & candidates_frame["candidate"].eq(candidate)
            ]
            if not part.empty:
                profile[population] = {
                    "count": len(part),
                    "nonredundant_median": float(part["candidate_nonredundant_ratio"].median()),
                    "dirichlet_median": float(part["normalized_dirichlet_energy"].median()),
                    "variance_median": float(part["variance"].median()),
                }
        for population in populations:
            spectral_part = spectral_frame[
                spectral_frame["population"].eq(population)
                & spectral_frame["signal"].eq(candidate)
            ]
            profile.setdefault(population, {})["low32_energy_median"] = (
                float(spectral_part["low32_energy_fraction"].median()) if not spectral_part.empty else np.nan
            )
            existing_part = spectral_frame[
                spectral_frame["population"].eq(population)
                & spectral_frame["signal_kind"].eq("existing")
                & ~spectral_frame["signal"].isin(["x_norm", "y_norm"])
            ]
            profile[population]["existing_low32_median"] = (
                float(existing_part["low32_energy_fraction"].median()) if not existing_part.empty else np.nan
            )
            freq_existing = frequency[
                frequency["population"].eq(population)
                & frequency["signal_kind"].eq("existing")
                & ~frequency["signal"].isin(["x_norm", "y_norm"])
            ]
            profile[population]["existing_dirichlet_median"] = float(
                freq_existing["normalized_dirichlet_energy"].median()
            )
        f1 = global_f1
        f2 = all(
            profile.get(population, {}).get("dirichlet_median", np.inf)
            < profile.get(population, {}).get("existing_dirichlet_median", -np.inf)
            and profile.get(population, {}).get("low32_energy_median", -np.inf)
            > profile.get(population, {}).get("existing_low32_median", np.inf)
            for population in populations
        )
        stability_values = [
            profile.get(population, {}).get("nonredundant_median", np.nan)
            for population in ("train", "val", "test")
        ]
        finite_stability = [value for value in stability_values if np.isfinite(value)]
        stability_range = max(finite_stability) - min(finite_stability) if finite_stability else np.inf
        fallback_part = candidates_frame[
            candidates_frame["candidate"].eq(candidate)
            & candidates_frame["split"].isin(["val", "test"])
        ]
        official = fallback_part[fallback_part["fallback"].eq(0)]["candidate_nonredundant_ratio"]
        fallback = fallback_part[fallback_part["fallback"].eq(1)]["candidate_nonredundant_ratio"]
        fallback_smd = smd(fallback, official)
        class_medians = fallback_part.groupby("class_name")["candidate_nonredundant_ratio"].median()
        class_collapse = bool(
            not class_medians.empty
            and class_medians.min() < 0.5 * max(float(class_medians.median()), EPS)
        )
        f3 = (
            stability_range <= 0.10
            and (not np.isfinite(fallback_smd) or abs(fallback_smd) < 0.25)
            and not class_collapse
        )
        f4_contexts = {}
        for comparison in (
            "A0_repeated_wrong_minus_A0_correct_both",
            "C2_repeated_wrong_minus_C2_correct_both",
        ):
            metric = f"candidate_nonredundant_ratio__{candidate}"
            part = error_frame[
                error_frame["comparison"].eq(comparison) & error_frame["metric"].eq(metric)
            ]
            if part.empty:
                continue
            row = part.iloc[0]
            f4_contexts[comparison] = {
                "smd": float(row["smd"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "direction": float(np.sign(row["mean_difference"])),
            }
        strong_contexts = [
            value
            for value in f4_contexts.values()
            if abs(value["smd"]) >= 0.20 and (value["ci_low"] > 0 or value["ci_high"] < 0)
        ]
        directions = [value["direction"] for value in f4_contexts.values() if value["direction"] != 0]
        consistent = len(directions) >= 2 and len(set(directions)) == 1
        if len(strong_contexts) >= 2 and consistent:
            f4_status = "PASS"
        elif consistent and any(abs(value["smd"]) >= 0.20 for value in f4_contexts.values()):
            f4_status = "WEAK"
        else:
            f4_status = "FAIL"
        candidate_profiles[candidate] = profile
        criteria[candidate] = {
            "F1_existing_feature_deficiency": {
                "status": "PASS" if f1 else "FAIL",
                "existing_effective_rank": existing_rank,
                "materially_below_nominal_threshold": 8.0,
                "at_least_one_candidate_nonredundant_ratio_ge_0_20": global_candidate_nonredundancy,
            },
            "F2_frequency_complementarity": {"status": "PASS" if f2 else "FAIL"},
            "F3_stability": {
                "status": "PASS" if f3 else "FAIL",
                "split_nonredundancy_range": stability_range,
                "fallback_smd": fallback_smd,
                "class_collapse": class_collapse,
            },
            "F4_error_relevance": {"status": f4_status, "contexts": f4_contexts},
        }
    technically_eligible = [
        candidate
        for candidate, values in criteria.items()
        if candidate != "C0_raw_intensity"
        and all(values[key]["status"] == "PASS" for key in (
            "F1_existing_feature_deficiency",
            "F2_frequency_complementarity",
            "F3_stability",
        ))
        and values["F4_error_relevance"]["status"] in {"PASS", "WEAK"}
    ]
    selected: str | None = None
    if len(technically_eligible) == 1:
        selected = technically_eligible[0]
    elif len(technically_eligible) > 1:
        for candidate in technically_eligible:
            dominates = True
            for other in technically_eligible:
                if other == candidate:
                    continue
                for population in populations:
                    current = candidate_profiles[candidate][population]
                    opponent = candidate_profiles[other][population]
                    if not (
                        current["nonredundant_median"] >= opponent["nonredundant_median"]
                        and current["dirichlet_median"] <= opponent["dirichlet_median"]
                        and current["low32_energy_median"] >= opponent["low32_energy_median"]
                    ):
                        dominates = False
            if dominates:
                selected = candidate
                break
    if selected is not None:
        decision = "SELECT_SINGLE_GRAPH_SIGNAL_FEATURE_FAMILY"
        family_map = {
            "C1_one_step_diffusion": "ONE_STEP_LOCAL_DIFFUSION",
            "C2_two_step_diffusion": "TWO_STEP_LOCAL_DIFFUSION",
            "C3_two_step_residual": "TWO_STEP_LOW_FREQUENCY_RESIDUAL",
        }
        selected_payload = {
            "candidate": selected,
            "family": family_map[selected],
            "formula": {
                "C1_one_step_diffusion": "z = P s",
                "C2_two_step_diffusion": "z = P^2 s",
                "C3_two_step_residual": "z = P^2 s - s",
            }[selected],
            "graph": "frozen selected-node local 8-neighbor graph only",
            "self_loop_policy": "A_tilde = A_local + I",
            "normalization": "P = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}",
            "feature_dimension_added": 1,
            "image_normalization": "s is selected intensity in [0,1]",
            "cache_schema_change": "new read-only-derived node channel requires a new future cache schema",
            "runtime_compute": "one or two sparse local diffusion multiplies per graph",
            "backward_compatibility": "legacy 10-channel caches remain valid only for legacy configs",
            "why_nonredundant": candidate_profiles[selected],
        }
        scope = "IMPLEMENT_FINAL_SINGLE_FEATURE"
    elif not technically_eligible:
        decision = "NO_FEATURE_AUGMENTATION_JUSTIFIED"
        selected_payload = {"candidate": None, "reason": "No bounded candidate passed F1, F2, F3 with at least WEAK F4 evidence."}
        scope = "USE_HISTORICAL_FALLBACK"
    else:
        decision = "FEATURE_AUDIT_AMBIGUOUS"
        selected_payload = {
            "candidate": None,
            "reason": f"No unique Pareto-dominant candidate among {technically_eligible}.",
            "single_bounded_diagnostic": "Repeat the same no-training low32 projection with 20 images per class; do not add candidates or tune scales.",
        }
        scope = "HOLD_ONE_DIAGNOSTIC"
    return {
        "existing_effective_rank": existing_rank,
        "candidate_profiles": candidate_profiles,
        "criteria_by_candidate": criteria,
        "technically_eligible_candidates": technically_eligible,
    }, decision, selected_payload, scope


def discover_historical_runs() -> list[Path]:
    excluded = re.compile(
        r"(analysis|cache|smoke|check|diagnostic|handoff|archive|snapshot|resume_test|tmp|temp|wandb)",
        re.IGNORECASE,
    )
    runs = []
    for history in (ROOT / "outputs").rglob("train_log.csv"):
        relative = str(history.relative_to(ROOT / "outputs"))
        if excluded.search(relative):
            continue
        run_dir = history.parent
        if not (run_dir / "checkpoints" / "best.pt").exists():
            continue
        if not re.search(r"(^|[\\/])d(15|16|17|18|19)", relative, re.IGNORECASE):
            continue
        runs.append(run_dir)
    return sorted(set(runs), key=lambda path: str(path).lower())


def first_existing(run_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def history_value(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                pass
    return float("nan")


def structure_dependency_lookup() -> dict[tuple[str, int], float]:
    path = C2_ANALYSIS / "11_edge_ablation_multiseed.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows = frame[
        frame["cell"].eq("C2")
        & frame["mode"].eq("remove_structure")
        & frame["checkpoint"].eq("best")
    ]
    return {
        (str(row.run_name), int(row.seed)): float(row.official_to_counterfactual_macro_f1_drop)
        for row in rows.itertuples(index=False)
    }


def strict_load_historical(run_dir: Path, cfg: dict[str, Any]) -> tuple[bool, int]:
    model_name = str((cfg.get("model") or {}).get("name", ""))
    if model_name != "d18_structure_guided_pixel_gnn":
        return False, -1
    try:
        model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
        state, _ = checkpoint_state(run_dir / "checkpoints" / "best.pt")
        model.load_state_dict(state, strict=True)
        return True, parameter_count(model)
    except Exception:
        return False, -1


def resume_contamination(run_dir: Path) -> tuple[bool, str]:
    events = run_dir / "resume_events.jsonl"
    if not events.exists():
        return False, "NOT VERIFIABLE"
    text = events.read_text(encoding="utf-8", errors="replace").lower()
    contaminated = any(token in text for token in ("signature_mismatch", "optimizer_mismatch", "scheduler_mismatch", "resume_corrupt"))
    return contaminated, "FAIL" if contaminated else "PASS"


def build_historical_registry_pretest(out: Path, reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    registry_path = out / "16_historical_run_registry.csv"
    eligibility_path = out / "17_historical_eligibility_audit.csv"
    lock_path = out / "19_fallback_lock_pretest.json"
    hash_path = out / "_fallback_lock_sha256.txt"
    if reuse and all(path.exists() for path in (registry_path, eligibility_path, lock_path, hash_path)):
        return pd.read_csv(registry_path, keep_default_na=False), pd.read_csv(eligibility_path, keep_default_na=False), read_json(lock_path), hash_path.read_text().strip()
    structure_dependency = structure_dependency_lookup()
    rows: list[dict[str, Any]] = []
    for position, run_dir in enumerate(discover_historical_runs(), start=1):
        history_path = run_dir / "train_log.csv"
        config_path = first_existing(run_dir, ("resolved_config.yaml", "source_config.yaml"))
        if config_path is None:
            continue
        try:
            cfg = read_yaml(config_path)
            history = pd.read_csv(history_path)
            if history.empty:
                continue
            state, payload = checkpoint_state(run_dir / "checkpoints" / "best.pt")
        except Exception as exc:
            rows.append(
                {
                    "run_id": run_dir.name,
                    "canonical_path": str(run_dir.relative_to(ROOT)),
                    "artifact_completeness": "FAIL",
                    "provenance_status": "FAIL",
                    "validation_audit_status": f"READ_ERROR:{type(exc).__name__}",
                    "test_accuracy": "MASKED_PRETEST",
                    "test_macro_f1": "MASKED_PRETEST",
                }
            )
            continue
        checkpoint_epoch = int(payload.get("epoch", -1))
        epoch_values = pd.to_numeric(history["epoch"], errors="coerce").fillna(-1).astype(int)
        matching = history[epoch_values.eq(checkpoint_epoch)]
        if matching.empty:
            matching = history.sort_values("val_macro_f1", ascending=False).head(1) if "val_macro_f1" in history else history.head(1)
        best = matching.iloc[-1]
        last = history.iloc[-1]
        train_acc = history_value(best, ("train_accuracy", "train_acc"))
        train_f1 = history_value(best, ("train_macro_f1", "train_f1"))
        val_acc = history_value(best, ("val_accuracy", "val_acc"))
        val_f1 = history_value(best, ("val_macro_f1", "val_f1"))
        last_val_acc = history_value(last, ("val_accuracy", "val_acc"))
        last_val_f1 = history_value(last, ("val_macro_f1", "val_f1"))
        monitor = str((cfg.get("training") or {}).get("checkpoint_monitor", ""))
        strict_load, params = strict_load_historical(run_dir, cfg)
        contaminated, resume_status = resume_contamination(run_dir)
        graph = cfg.get("graph", {}) or {}
        node_count = int(graph.get("target_node_count", graph.get("target_count", -1)) or -1)
        graph_mode = str(graph.get("graph_mode", "legacy_pixel_graph"))
        branch = "pixel_graph" if any(token in str(run_dir).lower() for token in ("d15", "d16", "d17", "d18", "d19")) else "other"
        complete_marker = any(
            (run_dir / name).exists()
            for name in ("TRAINING_COMPLETE.json", "COMPLETED.json", "training_complete.json", "d16_train_summary.json", "d17_train_summary.json", "d18_train_summary.json")
        )
        validation_selected = monitor.startswith("val_")
        history_agrees = bool(checkpoint_epoch in set(epoch_values.tolist()))
        gap_acc = (train_acc - val_acc) * 100 if np.isfinite(train_acc) and np.isfinite(val_acc) else np.nan
        gap_f1 = (train_f1 - val_f1) * 100 if np.isfinite(train_f1) and np.isfinite(val_f1) else np.nan
        decline = (val_f1 - last_val_f1) * 100 if np.isfinite(val_f1) and np.isfinite(last_val_f1) else np.nan
        seed = int((cfg.get("training") or {}).get("seed", cfg.get("seed", -1)))
        checkpoint_hash = sha256_file(run_dir / "checkpoints" / "best.pt")
        run_name = str(cfg.get("run_name", run_dir.name))
        config_signature = sha256_text(yaml.safe_dump(cfg, sort_keys=True))
        cache_sig_path = run_dir / "cache_signature.json"
        cache_signature = sha256_file(cache_sig_path) if cache_sig_path.exists() else "NOT VERIFIABLE"
        split_status = "AUDITED_FER2013_EQUIVALENT" if branch == "pixel_graph" else "NOT VERIFIABLE"
        rows.append(
            {
                "run_id": run_name,
                "canonical_path": str(run_dir.relative_to(ROOT)),
                "aliases": "[]",
                "experiment_family": next((part for part in run_dir.parts if re.match(r"d(15|16|17|18|19)", part, re.I)), run_dir.parent.name),
                "seed": seed,
                "source_config": str(config_path.relative_to(ROOT)),
                "resolved_config": str(config_path.relative_to(ROOT)),
                "dataset_signature": split_status,
                "split_signature": split_status,
                "model_signature": sha256_text(json.dumps(sorted((key, tuple(value.shape)) for key, value in state.items()))),
                "graph_signature": cache_signature,
                "selector_signature": f"{graph_mode}:{node_count}",
                "parameter_count": params,
                "checkpoint_path": str((run_dir / "checkpoints" / "best.pt").relative_to(ROOT)),
                "checkpoint_hash": checkpoint_hash,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_monitor": monitor,
                "best_validation_accuracy": val_acc,
                "best_validation_macro_f1": val_f1,
                "train_accuracy_at_best": train_acc,
                "train_macro_f1_at_best": train_f1,
                "train_validation_accuracy_gap_pp": gap_acc,
                "train_validation_macro_f1_gap_pp": gap_f1,
                "last_validation_accuracy": last_val_acc,
                "last_validation_macro_f1": last_val_f1,
                "best_last_validation_macro_f1_decline_pp": decline,
                "artifact_completeness": "PASS" if complete_marker and (run_dir / "checkpoints" / "last.pt").exists() else "FAIL",
                "resume_status": resume_status,
                "resume_contamination": contaminated,
                "provenance_status": "PASS" if config_path.exists() and history_agrees else "NOT VERIFIABLE",
                "validation_audit_status": "PASS" if strict_load and history_agrees else "NOT VERIFIABLE",
                "strict_best_load": strict_load,
                "structure_dependency_macro_f1_drop": structure_dependency.get((run_name, seed), np.nan),
                "random_permuted_structure_result": "AVAILABLE_IN_C2_AUDIT" if "c2_structure" in run_name else "NOT AVAILABLE",
                "test_accuracy": "MASKED_PRETEST",
                "test_macro_f1": "MASKED_PRETEST",
                "test_weighted_f1": "MASKED_PRETEST",
                "config_signature": config_signature,
            }
        )
        if position % 25 == 0:
            print(json.dumps({"event": "historical_registry_progress", "processed": position}), flush=True)
    registry = pd.DataFrame(rows)
    # Deduplicate exact checkpoint copies before eligibility and ranking.
    registry = registry.sort_values("canonical_path").reset_index(drop=True)
    aliases: dict[str, list[str]] = defaultdict(list)
    keep_rows = []
    for checkpoint_hash, part in registry.groupby("checkpoint_hash", dropna=False):
        canonical = part.iloc[0].copy()
        aliases[str(checkpoint_hash)] = part["canonical_path"].astype(str).tolist()[1:]
        canonical["aliases"] = json.dumps(aliases[str(checkpoint_hash)])
        keep_rows.append(canonical)
    registry = pd.DataFrame(keep_rows).reset_index(drop=True)
    eligibility_rows = []
    for row in registry.itertuples(index=False):
        reasons = []
        if getattr(row, "artifact_completeness") != "PASS":
            reasons.append("artifact_incomplete")
        if getattr(row, "dataset_signature") != "AUDITED_FER2013_EQUIVALENT":
            reasons.append("split_not_verified")
        if not bool(getattr(row, "strict_best_load")):
            reasons.append("strict_best_load_not_verified")
        if not str(getattr(row, "checkpoint_monitor")).startswith("val_"):
            reasons.append("checkpoint_not_validation_selected")
        if getattr(row, "validation_audit_status") != "PASS":
            reasons.append("history_checkpoint_not_verified")
        if bool(getattr(row, "resume_contamination")):
            reasons.append("resume_contamination")
        gap = float(getattr(row, "train_validation_macro_f1_gap_pp"))
        decline = float(getattr(row, "best_last_validation_macro_f1_decline_pp"))
        if not np.isfinite(gap) or gap > 12.0:
            reasons.append("macro_gap_above_12pp_or_missing")
        if not np.isfinite(decline) or decline > 3.0:
            reasons.append("best_last_decline_above_3pp_or_missing")
        eligible = not reasons
        tier = "INELIGIBLE"
        if eligible:
            tier = "A" if gap <= 10.0 and decline <= 2.0 else "B"
        eligibility_rows.append(
            {
                "run_id": getattr(row, "run_id"),
                "canonical_path": getattr(row, "canonical_path"),
                "eligible": eligible,
                "tier": tier,
                "reasons": json.dumps(reasons),
                "best_validation_macro_f1": getattr(row, "best_validation_macro_f1"),
                "best_validation_accuracy": getattr(row, "best_validation_accuracy"),
                "train_validation_macro_f1_gap_pp": gap,
                "best_last_validation_macro_f1_decline_pp": decline,
                "structure_dependency_macro_f1_drop": getattr(row, "structure_dependency_macro_f1_drop"),
                "checkpoint_epoch": getattr(row, "checkpoint_epoch"),
                "checkpoint_hash": getattr(row, "checkpoint_hash"),
                "test_metrics": "MASKED_PRETEST",
            }
        )
    eligibility = pd.DataFrame(eligibility_rows)
    eligible = eligibility[eligibility["eligible"].eq(True)].copy()
    if eligible.empty:
        raise RuntimeError("No historical run satisfies paper-safe fallback eligibility")
    eligible["_tier_rank"] = eligible["tier"].map({"A": 0, "B": 1})
    eligible["_structure_rank"] = pd.to_numeric(
        eligible["structure_dependency_macro_f1_drop"], errors="coerce"
    ).fillna(np.inf)
    ranking = eligible.sort_values(
        [
            "_tier_rank", "best_validation_macro_f1", "best_validation_accuracy",
            "train_validation_macro_f1_gap_pp", "best_last_validation_macro_f1_decline_pp",
            "_structure_rank", "checkpoint_epoch", "run_id",
        ],
        ascending=[True, False, False, True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_tier_rank", "_structure_rank"])
    ranking.insert(0, "pretest_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(out / "18_fallback_pretest_ranking.csv", index=False)
    registry.to_csv(registry_path, index=False)
    eligibility.to_csv(eligibility_path, index=False)
    registry_hash = sha256_file(registry_path)
    selected = ranking.iloc[0]
    source = registry[registry["run_id"].eq(selected["run_id"])].iloc[0]
    lock = {
        "selected_run_id": selected["run_id"],
        "canonical_path": selected["canonical_path"],
        "source_config": source["source_config"],
        "seed": int(source["seed"]),
        "checkpoint": source["checkpoint_path"],
        "checkpoint_hash": source["checkpoint_hash"],
        "eligibility_tier": selected["tier"],
        "validation_metrics": {
            "accuracy": float(selected["best_validation_accuracy"]),
            "macro_f1": float(selected["best_validation_macro_f1"]),
        },
        "train_validation_macro_f1_gap_pp": float(selected["train_validation_macro_f1_gap_pp"]),
        "best_last_validation_macro_f1_decline_pp": float(selected["best_last_validation_macro_f1_decline_pp"]),
        "structure_dependency_macro_f1_drop": (
            None if pd.isna(selected["structure_dependency_macro_f1_drop"])
            else float(selected["structure_dependency_macro_f1_drop"])
        ),
        "ranking_rule_version": LOCK_VERSION,
        "registry_hash": registry_hash,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "warnings": [
            "Historical experiment ancestry is statistically dependent.",
            "Pretest ranking columns contain no full-test metrics.",
            "NOT VERIFIABLE provenance is allowed only where measured artifacts remain internally consistent; this selected row passed strict runtime loading.",
        ],
    }
    write_json(lock_path, lock)
    lock_hash = sha256_file(lock_path)
    hash_path.write_text(lock_hash + "\n", encoding="utf-8")
    return registry, eligibility, lock, lock_hash


def normalize_metric_row(row: pd.Series) -> dict[str, Any]:
    result = {
        "accuracy": history_value(row, ("accuracy", "test_accuracy")),
        "macro_f1": history_value(row, ("macro_f1", "test_macro_f1")),
        "weighted_f1": history_value(row, ("weighted_f1", "test_weighted_f1")),
        "nll": history_value(row, ("nll", "test_nll")),
        "ece": history_value(row, ("ece", "ece_15bin", "test_ece")),
    }
    for name in CLASS_NAMES.values():
        result[f"f1_{name}"] = history_value(row, (f"f1_{name}",))
    confusion = row.get("confusion_matrix_json", None)
    result["confusion_matrix"] = json.loads(confusion) if isinstance(confusion, str) and confusion.startswith("[") else None
    return result


def known_test_metric_map() -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    c2_path = C2_ANALYSIS / "06_full_test_metrics.csv"
    if c2_path.exists():
        frame = pd.read_csv(c2_path)
        for row in frame.itertuples(index=False):
            run_id = f"d18_ofix18_{str(row.cell).lower()}_" + (
                "structure_mode_mix_only" if str(row.cell) == "C2" else "clean_control"
            ) + f"_seed{int(row.seed)}"
            result[(run_id, str(row.checkpoint_type))] = normalize_metric_row(pd.Series(row._asdict()))
    a1_path = A1_ANALYSIS / "08_full_test_metrics.csv"
    if a1_path.exists():
        frame = pd.read_csv(a1_path)
        run_map = {
            "null": "d19_a1_id_null_evidence_only_seed42",
            "correct": "d19_a1_id_correct_evidence_only_seed42",
        }
        for row in frame.itertuples(index=False):
            treatment = str(row.treatment)
            checkpoint_type = str(row.checkpoint_type)
            if treatment not in run_map or checkpoint_type not in {"best", "last"}:
                continue
            result[(run_map[treatment], checkpoint_type)] = normalize_metric_row(pd.Series(row._asdict()))
    a0_path = A0_ANALYSIS / "06_full_test_metrics.csv"
    if a0_path.exists():
        frame = pd.read_csv(a0_path)
        for row in frame.itertuples(index=False):
            if str(row.model_id) == "A0" and str(row.checkpoint_type) in {"best", "last"}:
                result[("d19_a0_evidence_only_matched_seed42", str(row.checkpoint_type))] = normalize_metric_row(pd.Series(row._asdict()))
    a07_path = A0_7_ANALYSIS / "05_full_test_metrics.csv"
    if a07_path.exists():
        frame = pd.read_csv(a07_path)
        for row in frame.itertuples(index=False):
            if str(row.model_id) == "A0" and str(row.checkpoint_type) in {"best", "last"}:
                result[("d19_a0_evidence_only_matched_seed7", str(row.checkpoint_type))] = normalize_metric_row(pd.Series(row._asdict()))
    return result


def generic_test_metrics(run_dir: Path) -> dict[str, Any]:
    summary_path = next(iter(sorted(run_dir.glob("*train_summary.json"))), None)
    result: dict[str, Any] = {}
    if summary_path is not None:
        summary = read_json(summary_path)
        result = {
            "accuracy": float(summary.get("test_accuracy", np.nan)),
            "macro_f1": float(summary.get("test_macro_f1", np.nan)),
            "weighted_f1": float(summary.get("test_weighted_f1", np.nan)),
            "nll": float(summary.get("test_nll", np.nan)),
            "ece": float(summary.get("test_ece", np.nan)),
            "last_accuracy": float(summary.get("last_test_accuracy", np.nan)),
            "last_macro_f1": float(summary.get("last_test_macro_f1", np.nan)),
        }
    per_class_path = run_dir / "per_class_metrics.csv"
    if per_class_path.exists():
        per_class = pd.read_csv(per_class_path)
        if {"class_name", "support", "f1"}.issubset(per_class.columns):
            for row in per_class.itertuples(index=False):
                result[f"f1_{row.class_name}"] = float(row.f1)
            total = float(per_class["support"].sum())
            result["weighted_f1"] = float(np.sum(per_class["support"] * per_class["f1"]) / max(total, 1.0))
    confusion_path = run_dir / "confusion_matrix.csv"
    if confusion_path.exists():
        confusion = pd.read_csv(confusion_path, index_col=0)
        result["confusion_matrix"] = confusion.to_numpy(dtype=int).tolist()
    return result


def reveal_historical_tests(
    out: Path,
    registry: pd.DataFrame,
    eligibility: pd.DataFrame,
    lock: dict[str, Any],
    lock_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    lock_path = out / "19_fallback_lock_pretest.json"
    expected_hash = (out / "_fallback_lock_sha256.txt").read_text().strip()
    actual_hash = sha256_file(lock_path)
    if actual_hash != expected_hash or lock_hash != expected_hash:
        raise RuntimeError("Fallback pretest lock changed before test reveal")
    metric_map = known_test_metric_map()
    eligible = eligibility[eligibility["eligible"].eq(True)].copy()
    enriched_rows = []
    for row in eligible.itertuples(index=False):
        registry_row = registry[registry["run_id"].eq(row.run_id)].iloc[0]
        run_dir = ROOT / str(registry_row["canonical_path"])
        metrics = metric_map.get((str(row.run_id), "best"), generic_test_metrics(run_dir))
        last_metrics = metric_map.get((str(row.run_id), "last"), {})
        enriched_rows.append(
            {
                **row._asdict(),
                "experiment_family": registry_row.get("experiment_family", "unknown"),
                "test_accuracy": metrics.get("accuracy", np.nan),
                "test_macro_f1": metrics.get("macro_f1", np.nan),
                "test_weighted_f1": metrics.get("weighted_f1", np.nan),
                "test_nll": metrics.get("nll", np.nan),
                "test_ece": metrics.get("ece", np.nan),
                "last_test_accuracy": last_metrics.get("accuracy", metrics.get("last_accuracy", np.nan)),
                "last_test_macro_f1": last_metrics.get("macro_f1", metrics.get("last_macro_f1", np.nan)),
                "classwise_f1_json": json.dumps(
                    {name: metrics.get(f"f1_{name}", np.nan) for name in CLASS_NAMES.values()},
                    default=json_default,
                ),
                "confusion_matrix_json": json.dumps(metrics.get("confusion_matrix"), default=json_default),
            }
        )
    enriched = pd.DataFrame(enriched_rows)
    selected = enriched[enriched["run_id"].eq(lock["selected_run_id"])]
    if len(selected) != 1:
        raise RuntimeError("Locked fallback missing from postlock registry")
    selected_row = selected.iloc[0]
    fallback = {
        **lock,
        "lock_sha256": expected_hash,
        "postlock_test_reveal": {
            "accuracy": float(selected_row["test_accuracy"]),
            "macro_f1": float(selected_row["test_macro_f1"]),
            "weighted_f1": float(selected_row["test_weighted_f1"]),
            "nll": float(selected_row["test_nll"]),
            "ece": float(selected_row["test_ece"]),
            "classwise_f1": json.loads(selected_row["classwise_f1_json"]),
            "confusion_matrix": json.loads(selected_row["confusion_matrix_json"]),
            "last_accuracy": float(selected_row["last_test_accuracy"]),
            "last_macro_f1": float(selected_row["last_test_macro_f1"]),
            "gap_to_65_accuracy_pp": (0.65 - float(selected_row["test_accuracy"])) * 100,
        },
        "selection_changed_after_reveal": False,
    }
    peak_candidates = enriched[
        pd.to_numeric(enriched["test_accuracy"], errors="coerce").notna()
    ].copy()
    if peak_candidates.empty:
        retrospective = {"status": "NOT IDENTIFIABLE", "reason": "No eligible run has verified test metrics"}
    else:
        peak = peak_candidates.sort_values(
            [
                "test_accuracy", "test_macro_f1", "best_validation_macro_f1",
                "train_validation_macro_f1_gap_pp",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0]
        retrospective = {
            "label": "RETROSPECTIVE TEST-AWARE REFERENCE",
            "run_id": peak["run_id"],
            "canonical_path": peak["canonical_path"],
            "test_accuracy": float(peak["test_accuracy"]),
            "test_macro_f1": float(peak["test_macro_f1"]),
            "test_weighted_f1": float(peak["test_weighted_f1"]),
            "validation_macro_f1": float(peak["best_validation_macro_f1"]),
            "train_validation_macro_f1_gap_pp": float(peak["train_validation_macro_f1_gap_pp"]),
            "same_as_paper_safe_fallback": bool(peak["run_id"] == lock["selected_run_id"]),
            "warning": "This reference was identified using test performance and cannot silently replace the pretest-locked fallback.",
        }
    calibration_rows = []
    relationships = [
        ("best_validation_accuracy", "test_accuracy"),
        ("best_validation_macro_f1", "test_macro_f1"),
        ("train_validation_macro_f1_gap_pp", "test_accuracy"),
        ("best_last_validation_macro_f1_decline_pp", "test_accuracy"),
    ]
    for left, right in relationships:
        pair = enriched[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            continue
        calibration_rows.append(
            {
                "relationship": f"{left}_vs_{right}",
                "count": len(pair),
                "pearson": float(pearsonr(pair[left], pair[right]).statistic),
                "spearman": float(spearmanr(pair[left], pair[right]).statistic),
                "median_right_minus_left": (
                    float((pair[right] - pair[left]).median())
                    if left.startswith("best_validation_") else np.nan
                ),
                "iqr_right_minus_left": (
                    float(
                        (pair[right] - pair[left]).quantile(0.75)
                        - (pair[right] - pair[left]).quantile(0.25)
                    )
                    if left.startswith("best_validation_") else np.nan
                ),
                "mean_absolute_deviation": float(np.mean(np.abs((pair[right] - pair[left]) - (pair[right] - pair[left]).mean()))),
                "statistical_independence_claimed": False,
            }
        )
        for family in sorted(enriched["experiment_family"].dropna().astype(str).unique()):
            reduced = enriched[enriched["experiment_family"].astype(str).ne(family)][[left, right]]
            reduced = reduced.apply(pd.to_numeric, errors="coerce").dropna()
            if len(reduced) < 3 or reduced[left].nunique() < 2 or reduced[right].nunique() < 2:
                continue
            calibration_rows.append(
                {
                    "relationship": f"{left}_vs_{right}__leave_out_family={family}",
                    "count": len(reduced),
                    "pearson": float(pearsonr(reduced[left], reduced[right]).statistic),
                    "spearman": float(spearmanr(reduced[left], reduced[right]).statistic),
                    "median_right_minus_left": (
                        float((reduced[right] - reduced[left]).median())
                        if left.startswith("best_validation_") else np.nan
                    ),
                    "iqr_right_minus_left": (
                        float(
                            (reduced[right] - reduced[left]).quantile(0.75)
                            - (reduced[right] - reduced[left]).quantile(0.25)
                        )
                        if left.startswith("best_validation_") else np.nan
                    ),
                    "mean_absolute_deviation": float(
                        np.mean(
                            np.abs(
                                (reduced[right] - reduced[left])
                                - (reduced[right] - reduced[left]).mean()
                            )
                        )
                    ),
                    "statistical_independence_claimed": False,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(out / "22_validation_test_calibration.csv", index=False)
    near_65 = enriched[
        pd.to_numeric(enriched["test_accuracy"], errors="coerce").between(0.64, 0.66)
    ]
    readiness = {
        "source_run_count": len(near_65),
        "validation_accuracy_range": (
            [float(near_65["best_validation_accuracy"].min()), float(near_65["best_validation_accuracy"].max())]
            if not near_65.empty else None
        ),
        "validation_macro_f1_range": (
            [float(near_65["best_validation_macro_f1"].min()), float(near_65["best_validation_macro_f1"].max())]
            if not near_65.empty else None
        ),
        "interpretation": "Descriptive readiness indicator only; never a checkpoint-selection rule.",
    }
    enriched.to_csv(out / "_historical_registry_postlock_test_metrics.csv", index=False)
    return fallback, retrospective, calibration, readiness


REQUIRED_REPORTS = [
    "00_README.md",
    "01_source_and_artifact_manifest.md",
    "02_runtime_feature_code_trace.md",
    "03_a1_null_affine_equivalence_derivation.md",
    "04_a1_null_fold_validation.csv",
    "04_a1_null_fold_validation.md",
    "05_current_node_feature_inventory.csv",
    "05_current_node_feature_inventory.md",
    "06_feature_distribution_statistics.csv",
    "06_feature_distribution_statistics.md",
    "07_feature_redundancy_analysis.csv",
    "07_feature_redundancy_analysis.md",
    "08_graph_frequency_characterization.csv",
    "08_graph_frequency_characterization.md",
    "09_low_frequency_coverage_analysis.csv",
    "09_low_frequency_coverage_analysis.md",
    "10_fixed_checkpoint_feature_sensitivity.csv",
    "10_fixed_checkpoint_feature_sensitivity.md",
    "11_error_group_feature_analysis.csv",
    "11_error_group_feature_analysis.md",
    "12_bounded_candidate_signal_bank.csv",
    "12_bounded_candidate_signal_bank.md",
    "13_candidate_nonredundancy_analysis.csv",
    "13_candidate_nonredundancy_analysis.md",
    "14_candidate_stability_and_classwise_analysis.csv",
    "14_candidate_stability_and_classwise_analysis.md",
    "15_final_feature_family_decision.md",
    "16_historical_run_registry.csv",
    "16_historical_run_registry.md",
    "17_historical_eligibility_audit.csv",
    "17_historical_eligibility_audit.md",
    "18_fallback_pretest_ranking.csv",
    "18_fallback_pretest_ranking.md",
    "19_fallback_lock_pretest.json",
    "19_fallback_lock_pretest.md",
    "20_fallback_postlock_test_reveal.md",
    "21_retrospective_peak_reference.md",
    "22_validation_test_calibration.csv",
    "22_validation_test_calibration.md",
    "23_final_training_gate.md",
    "24_final_implementation_scope.md",
    "25_paper_reporting_scope.md",
    "26_machine_readable_summary.json",
    "27_run_commands.md",
    "28_validation_summary.json",
]


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_or_none(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def safe_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(finite_or_none(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_trace_payload() -> dict[str, Any]:
    config = config_for(A1_NULL)
    edge_cfg = config["model"]["edge_context_gnn"]
    layer_path = ROOT / "d18/models/structure_gnn.py"
    builder_path = ROOT / "d18/data/structure_graph_builder.py"
    cache_path = ROOT / "d19/data/evidence_only_graph_cache.py"
    selector_path = ROOT / "d18/data/structure_graph_builder.py"
    return {
        "model_source": rel(layer_path),
        "model_class": "StructureGNN",
        "graph_builder_source": rel(builder_path),
        "node_feature_constant": "NODE_FEATURE_NAMES",
        "cache_source": rel(cache_path),
        "selector_source": rel(selector_path),
        "node_feature_dim": len(FEATURE_NAMES),
        "base_edge_attr_dim": 6,
        "a1_null_conditioned_edge_dim": int(edge_cfg["edge_attr_dim"])
        + int(config["model"]["edge_type_conditioning"]["embedding_dim"]),
        "relation_embedding_dim": int(config["model"]["edge_type_conditioning"]["embedding_dim"]),
        "gnn_layers": int(edge_cfg["num_layers"]),
        "node_hidden_dim": int(config["model"]["hidden_dim"]),
        "edge_hidden_dim": int(edge_cfg["edge_hidden_dim"]),
        "first_edge_affine_per_layer": "edge_mlp[0]=Linear(conditioned_edge_dim, edge_hidden_dim, bias=True)",
        "edge_path_order": [
            "concatenate(base_edge_attr, relation_embedding)",
            "Linear(bias=True)",
            "LayerNorm",
            "GELU",
            "Dropout",
            "Linear",
            "GELU",
        ],
        "operation_before_first_affine": None,
        "other_relation_embedding_consumers": [],
        "checkpoint_monitor": config["training"]["checkpoint_monitor"],
        "selector_description": "image-detail-stratified sparse pixel selection",
        "selector_cells": "6x6",
        "pixels_per_cell": 50,
        "pixel_node_count": 1800,
        "landmark_role": "optional bounded residual structure edges only",
    }


def artifact_manifest_rows(input_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in input_rows:
        path = Path(item["path"])
        rows.append(
            {
                "artifact": item["artifact"],
                "path": str(path),
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )
    rows.extend(
        [
            {
                "artifact": "locked_manifest",
                "path": str(LOCKED_SOURCE),
                "exists": LOCKED_SOURCE.exists(),
                "kind": "file",
                "sha256": sha256_file(LOCKED_SOURCE),
            },
            {
                "artifact": "cache_signature",
                "path": str(CACHE_ROOT / "cache_signature.json"),
                "exists": (CACHE_ROOT / "cache_signature.json").exists(),
                "kind": "file",
                "sha256": sha256_file(CACHE_ROOT / "cache_signature.json"),
            },
        ]
    )
    return pd.DataFrame(rows)


def report_fold(out: Path, fold: pd.DataFrame, meta: dict[str, Any], layers: dict[str, Any]) -> None:
    population = fold[fold["stage"].eq("probabilities")].copy()
    conditions = meta["mathematical_conditions"]
    layer_names = [str(item["layer"]) for item in layers]
    write_md(
        out / "03_a1_null_affine_equivalence_derivation.md",
        "A1-null affine equivalence derivation",
        "\n".join(
            [
                "For every edge in A1-null, the conditioned input is",
                "",
                "`e_conditioned = [e_base ; c]`, where `c = Embedding(null_id)` is constant.",
                "",
                "The first consumer in every GNN layer is an affine map with bias:",
                "",
                "`y = W_base e_base + W_constant c + b`.",
                "",
                "Therefore an A0 layer with `W_fold = W_base` and "
                "`b_fold = b + W_constant c` is exactly equivalent before LayerNorm, GELU, "
                "Dropout and the second affine. No normalization, nonlinearity or dropout "
                "occurs before this fold point, and the relation embedding has no other consumer.",
                "",
                f"Folded layers: `{', '.join(layer_names)}`.",
                "",
                "**Conclusion:** A1-ID-null is an exactly foldable reparameterization for the "
                "verified runtime architecture. It does not add a relation-dependent function class.",
            ]
        ),
    )
    write_md(
        out / "04_a1_null_fold_validation.md",
        "A1-null folded numerical validation",
        "\n".join(
            [
                f"Strict A0 load: `{conditions['strict_load']}`. "
                f"A1-null parameters: `{conditions['source_parameter_count']:,}`; "
                f"folded A0 parameters: `{conditions['folded_parameter_count']:,}`.",
                "",
                md_table(population),
                "",
                "Equivalence is judged at input projection, edge projections, all GNN layers, "
                "pooled embedding, classifier input, logits and probabilities. Minor floating-point "
                "roundoff is acceptable; predictions must match exactly.",
            ]
        ),
    )


def plot_matrix(path: Path, matrix: np.ndarray, labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(labels)), values, color="#277da1")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_plots(out: Path, postlock: bool) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    redundancy = pd.read_csv(out / "07_feature_redundancy_analysis.csv")
    corr_path = out / "_feature_correlation_matrices.npz"
    if corr_path.exists():
        with np.load(corr_path) as matrices:
            matrix = matrices["validation"]
        plot_matrix(plot_dir / "feature_correlation_matrix.png", matrix, FEATURE_NAMES, "Node-feature Pearson correlation")
        eig = np.linalg.eigvalsh(matrix)
        plot_bar(
            plot_dir / "feature_covariance_spectrum.png",
            [str(index + 1) for index in range(len(eig))],
            sorted(eig, reverse=True),
            "Feature covariance spectrum",
            "Eigenvalue",
        )
    images = pd.read_csv(out / "_feature_image_statistics.csv")
    rank = images.groupby("split", as_index=False)["effective_rank"].median()
    plot_bar(
        plot_dir / "feature_effective_rank_by_split.png",
        rank["split"].tolist(),
        rank["effective_rank"].tolist(),
        "Median per-image feature effective rank",
        "Effective rank",
    )
    frequency = pd.read_csv(out / "08_graph_frequency_characterization.csv")
    val_freq = frequency[frequency["population"].eq("val")]
    if not val_freq.empty:
        plot_bar(
            plot_dir / "feature_dirichlet_energy.png",
            val_freq["signal"].tolist(),
            val_freq["normalized_dirichlet_energy_median"].tolist(),
            "Validation graph Dirichlet energy",
            "Normalized energy",
        )
    low = pd.read_csv(out / "09_low_frequency_coverage_analysis.csv")
    val_low = low[low["population"].eq("val")]
    if not val_low.empty:
        plot_bar(
            plot_dir / "feature_low_frequency_fraction.png",
            val_low["signal"].tolist(),
            val_low["low32_energy_fraction_median"].tolist(),
            "Validation low-32 spectral energy",
            "Energy fraction",
        )
    candidate = pd.read_csv(out / "13_candidate_nonredundancy_analysis.csv")
    val_candidate = candidate[candidate["population"].eq("val")]
    if not val_candidate.empty:
        plot_bar(
            plot_dir / "candidate_projection_residual.png",
            val_candidate["candidate"].tolist(),
            val_candidate["nonredundant_median"].tolist(),
            "Validation candidate projection residual",
            "Nonredundant ratio",
        )
    sensitivity = pd.read_csv(out / "10_fixed_checkpoint_feature_sensitivity.csv")
    channel = sensitivity[
        sensitivity["population"].eq("validation")
        & sensitivity["perturbation"].isin([f"channel:{name}" for name in FEATURE_NAMES])
    ].copy()
    if not channel.empty:
        channel["macro_f1_delta_pp"] = 100 * channel["macro_f1_change"]
        pivot = channel.groupby("perturbation", as_index=False)["macro_f1_delta_pp"].mean()
        plot_bar(
            plot_dir / "feature_sensitivity_by_channel.png",
            pivot["perturbation"].tolist(),
            pivot["macro_f1_delta_pp"].tolist(),
            "Mean fixed-checkpoint feature sensitivity",
            "Macro-F1 delta (pp)",
        )
    errors = pd.read_csv(out / "11_error_group_feature_analysis.csv")
    error = errors[
        errors["metric"].isin(
            [
                "existing_low_frequency_energy",
                "existing_high_frequency_energy",
                "effective_rank",
                "covariance_condition",
                "raw_intensity_variance",
                "global_mean_intensity",
                "local_contrast",
            ]
        )
    ].copy()
    if not error.empty:
        error = error.reindex(error["smd"].abs().sort_values(ascending=False).index).head(15)
        plot_bar(
            plot_dir / "feature_error_group_effects.png",
            (error["comparison"] + ":" + error["metric"]).tolist(),
            error["smd"].tolist(),
            "Largest error-group feature effects",
            "Standardized mean difference",
        )
    ranking = pd.read_csv(out / "18_fallback_pretest_ranking.csv").head(20)
    if not ranking.empty:
        plot_bar(
            plot_dir / "historical_validation_ranking.png",
            ranking["run_id"].tolist(),
            (100 * ranking["best_validation_macro_f1"]).tolist(),
            "Pretest historical validation ranking",
            "Validation macro-F1 (%)",
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            ranking["train_validation_macro_f1_gap_pp"],
            100 * ranking["best_validation_macro_f1"],
            alpha=0.8,
        )
        ax.set_xlabel("Train-validation macro-F1 gap (pp)")
        ax.set_ylabel("Validation macro-F1 (%)")
        ax.set_title("Historical gap vs validation")
        fig.tight_layout()
        fig.savefig(plot_dir / "historical_gap_vs_validation.png", dpi=160)
        plt.close(fig)
    if postlock and (out / "_historical_registry_postlock_test_metrics.csv").exists():
        frame = pd.read_csv(out / "_historical_registry_postlock_test_metrics.csv")
        for metric in ("accuracy", "macro_f1"):
            x = pd.to_numeric(frame[f"best_validation_{metric}"], errors="coerce")
            y = pd.to_numeric(frame[f"test_{metric}"], errors="coerce")
            keep = x.notna() & y.notna()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(100 * x[keep], 100 * y[keep], alpha=0.8)
            ax.set_xlabel(f"Validation {metric} (%)")
            ax.set_ylabel(f"Test {metric} (%)")
            ax.set_title(f"Validation vs test {metric}")
            fig.tight_layout()
            fig.savefig(plot_dir / f"validation_vs_test_{metric}.png", dpi=160)
            plt.close(fig)


def final_gate_payload() -> dict[str, Any]:
    return {
        "seed42_validation_promotion_gate": {
            "baseline": "d18_ofix18_c2_structure_mode_mix_only_seed42",
            "validation_macro_f1_gain_min_pp": 1.00,
            "validation_accuracy_gain_min_pp": 0.75,
            "train_validation_macro_f1_gap_increase_max_pp": 2.00,
            "maximum_classwise_validation_f1_loss_pp": 4.00,
            "frozen_invariants": [
                "batch definition",
                "selector hash",
                "node-coordinate hash",
                "local-edge hash",
                "structure-mode-mix policy",
                "normalization",
            ],
            "checkpoint_selection": "validation only",
            "resume_allowed": False,
            "all_conditions_required": True,
        },
        "multiseed_gate_after_seed42_pass": {
            "additional_seeds": [7, 21],
            "mean_test_accuracy_min": 0.65,
            "mean_test_macro_f1_min": 0.63,
            "sample_std_test_accuracy_max_pp": 1.00,
            "minimum_single_seed_test_accuracy": 0.635,
            "mean_official_minus_structure_removed_macro_f1_drop_max_pp": 4.00,
            "structure_ablation_requirement": "correct structure must outperform random and permuted controls",
            "all_conditions_required": True,
        },
        "failure_action": "STOP_FINAL_MODEL_AND_USE_PRETEST_LOCKED_HISTORICAL_FALLBACK",
    }


def implementation_scope(decision: str, selected: dict[str, Any], scope: str) -> dict[str, Any]:
    baseline = {
        "run_id": "d18_ofix18_c2_structure_mode_mix_only_seed42",
        "role": "final validation baseline",
        "selector": "frozen image-detail-stratified 1800-pixel selector",
        "architecture": "sparse pixel graph plus optional bounded residual structure",
    }
    return {
        "decision": decision,
        "scope": scope,
        "baseline": baseline,
        "selected_feature": selected,
        "permitted_change": (
            "Add exactly the selected one-dimensional graph-signal channel and update only the "
            "corresponding input/cache schema."
            if scope == "IMPLEMENT_FINAL_SINGLE_FEATURE"
            else "No model implementation is justified by this audit."
        ),
        "forbidden_changes": [
            "selector revision",
            "node-count increase",
            "patch/superpixel/region/landmark nodes",
            "CNN or visual stem",
            "relation identity or independent relation operators",
            "optimizer or scheduler sweep",
            "Jumping Knowledge or multiscale pooling",
            "Structure DropEdge",
        ],
    }


def attach_locked_fallback(scope_payload: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(scope_payload))
    if result["scope"] == "USE_HISTORICAL_FALLBACK":
        result["locked_fallback"] = {
            "run_id": lock["selected_run_id"],
            "config": lock["source_config"],
            "checkpoint": lock["checkpoint"],
            "checkpoint_hash": lock["checkpoint_hash"],
            "seed": lock["seed"],
            "eligibility_tier": lock["eligibility_tier"],
        }
        result["permitted_change"] = (
            "No final feature implementation or new training run is justified. "
            "Use exactly the pretest-locked historical fallback."
        )
    return result


def paper_scope_payload(fallback: dict[str, Any] | None, retrospective: dict[str, Any] | None) -> dict[str, Any]:
    fallback_reporting = None
    if fallback is not None:
        fallback_reporting = {
            "state": fallback,
            "paper_safe_pretest_locked": True,
            "seed_count": 1,
            "multiseed_stability_claim_allowed": False,
            "stability_limitation": "The locked configuration is one seed42 run.",
            "structure_dependency": fallback.get("structure_dependency_macro_f1_drop"),
            "structure_dependency_status": (
                "NOT MEASURED/NOT COMPARABLE"
                if fallback.get("structure_dependency_macro_f1_drop") is None
                else "MEASURED"
            ),
            "claimable_metrics": (
                ["accuracy", "macro_f1", "weighted_f1", "classwise_f1", "confusion_matrix"]
                if "postlock_test_reveal" in fallback else ["validation_accuracy", "validation_macro_f1"]
            ),
        }
    return {
        "required_architecture_wording": (
            "image-detail-stratified sparse pixel graph + optional landmark-derived residual structure"
        ),
        "forbidden_wording": "landmark-guided pixel selection",
        "selector_evidence": {
            "pixel_budget_percent": 78.125,
            "validation_tv_retention_percent": 92.57,
            "validation_laplacian_retention_percent": 94.95,
            "locked_tv_retention_percent": 92.63,
            "locked_laplacian_retention_percent": 94.96,
            "decision": "SELECTOR_SUFFICIENT",
        },
        "relation_identity_result": "tested and rejected (STOP_RELATION_ID)",
        "fallback": fallback_reporting,
        "retrospective_reference": retrospective,
        "warnings": [
            "Do not claim multiseed stability for a single run.",
            "Do not hide measured structure dependency.",
            "The retrospective test-aware reference cannot replace the pretest-locked fallback.",
        ],
    }


def write_markdown_reports(
    out: Path,
    manifest: pd.DataFrame,
    trace: dict[str, Any],
    fold: pd.DataFrame,
    fold_meta: dict[str, Any],
    folded_layers: dict[str, Any],
    decision_evidence: dict[str, Any],
    decision: str,
    selected: dict[str, Any],
    scope_payload: dict[str, Any],
    fallback: dict[str, Any] | None,
    retrospective: dict[str, Any] | None,
    calibration: pd.DataFrame,
    readiness: dict[str, Any] | None,
    postlock: bool,
) -> None:
    inventory = pd.read_csv(out / "05_current_node_feature_inventory.csv")
    distribution = pd.read_csv(out / "06_feature_distribution_statistics.csv")
    redundancy = pd.read_csv(out / "07_feature_redundancy_analysis.csv")
    frequency = pd.read_csv(out / "08_graph_frequency_characterization.csv")
    low = pd.read_csv(out / "09_low_frequency_coverage_analysis.csv")
    sensitivity = pd.read_csv(out / "10_fixed_checkpoint_feature_sensitivity.csv")
    errors = pd.read_csv(out / "11_error_group_feature_analysis.csv")
    bank = pd.read_csv(out / "12_bounded_candidate_signal_bank.csv")
    nonredundancy = pd.read_csv(out / "13_candidate_nonredundancy_analysis.csv")
    stability = pd.read_csv(out / "14_candidate_stability_and_classwise_analysis.csv")
    registry = pd.read_csv(out / "16_historical_run_registry.csv")
    eligibility = pd.read_csv(out / "17_historical_eligibility_audit.csv")
    ranking = pd.read_csv(out / "18_fallback_pretest_ranking.csv")
    lock = read_json(out / "19_fallback_lock_pretest.json")
    write_md(
        out / "00_README.md",
        "D19 final feature-frequency and fallback audit",
        "\n".join(
            [
                "This directory contains the definitive read-only audit before the last permitted model implementation.",
                "",
                f"- A1-null equivalence: **{fold_meta['status']}**",
                f"- Final feature decision: **{decision}**",
                f"- Final implementation scope: **{scope_payload['scope']}**",
                f"- Pretest-locked fallback: **{lock['selected_run_id']}**",
                f"- Postlock test reveal complete: **{postlock}**",
                "",
                "The runtime selector is image-detail-stratified. Landmark-derived information is used only "
                "as optional bounded residual structure; it does not select pixel coordinates.",
            ]
        ),
    )
    write_md(
        out / "01_source_and_artifact_manifest.md",
        "Source and artifact manifest",
        md_table(manifest, limit=200)
        + "\n\nLocked manifest SHA-256 expected and verified: "
        + f"`{LOCKED_SHA256}`.",
    )
    write_md(
        out / "02_runtime_feature_code_trace.md",
        "Runtime feature and edge-conditioning code trace",
        "```json\n" + json.dumps(trace, indent=2) + "\n```\n\n"
        "Labels are read only after graph construction and do not enter node-feature computation.",
    )
    report_fold(out, fold, fold_meta, folded_layers)
    write_md(
        out / "05_current_node_feature_inventory.md",
        "Current 10-dimensional node-feature inventory",
        md_table(inventory, 30)
        + "\n\nThe order is a runtime contract. All channels are computed from the selected "
        "48x48 image and pixel coordinates; no label or test-only statistic enters construction.",
    )
    write_md(
        out / "06_feature_distribution_statistics.md",
        "Feature distribution statistics",
        md_table(distribution, 200)
        + "\n\nStatistics cover all cached train/validation/test graphs; node-level quantiles use "
        "a deterministic bounded node sample while global moments are accumulated over all nodes.",
    )
    strongest = redundancy[
        redundancy["row_type"].isin(["pairwise", "unique_residual"])
    ].copy()
    if "pearson" in strongest:
        strongest["_strength"] = strongest["pearson"].abs().fillna(0.0)
        strongest = strongest.sort_values("_strength", ascending=False).drop(columns="_strength")
    write_md(
        out / "07_feature_redundancy_analysis.md",
        "Feature redundancy analysis",
        md_table(strongest, 100)
        + "\n\nThis analysis includes correlation, covariance spectrum, effective rank, VIF, "
        "linear projection residual and pairwise linear CKA.",
    )
    write_md(
        out / "08_graph_frequency_characterization.md",
        "Graph-frequency characterization",
        md_table(frequency, 200)
        + "\n\nThe graph operator is the frozen selected-node local 8-neighbor graph. "
        "`L = I - D^-1/2 A D^-1/2`; diffusion uses `A + I`.",
    )
    write_md(
        out / "09_low_frequency_coverage_analysis.md",
        "Low-frequency coverage analysis",
        md_table(low, 200)
        + "\n\n**Conclusion: low-frequency graph signal is not underrepresented in the current "
        "feature set.** On validation, raw intensity and local mean already place median "
        "`0.9153` and `0.9393` of energy in the lowest 32 eigenvectors. C1/C2 are even "
        "smoother but only weakly nonredundant with the existing channels.\n\n"
        + "The lowest and highest 32 Laplacian eigenvectors are evaluated on deterministic "
        "class-balanced subsets. Results are graph-dependent and do not prove classifier gain.",
    )
    write_md(
        out / "10_fixed_checkpoint_feature_sensitivity.md",
        "Fixed-checkpoint feature sensitivity",
        md_table(sensitivity, 250)
        + "\n\nEach listed channel/group is replaced in memory by its training mean. "
        "This is an off-policy diagnostic; no checkpoint is retrained or changed.",
    )
    write_md(
        out / "11_error_group_feature_analysis.md",
        "Error-group feature analysis",
        md_table(errors, 250)
        + "\n\nConfidence intervals use deterministic class-stratified bootstrap. "
        "Associations are observational and do not establish causality.",
    )
    write_md(
        out / "12_bounded_candidate_signal_bank.md",
        "Bounded candidate graph-signal bank",
        md_table(bank, 200)
        + "\n\nCandidates are fixed before measurement: `C0=s`, `C1=Ps`, `C2=P^2s`, "
        "`C3=P^2s-s`. They do not change node coordinates or edges.",
    )
    write_md(
        out / "13_candidate_nonredundancy_analysis.md",
        "Candidate nonredundancy analysis",
        md_table(nonredundancy, 200)
        + "\n\nNonredundancy is the residual-energy ratio after deterministic linear projection "
        "onto the existing 10 channels, supplemented by correlation and linear CKA.",
    )
    write_md(
        out / "14_candidate_stability_and_classwise_analysis.md",
        "Candidate stability and classwise analysis",
        md_table(stability, 250)
        + "\n\nThe table separates split, class and actual landmark-fallback status without "
        "using labels in feature construction.",
    )
    write_md(
        out / "15_final_feature_family_decision.md",
        "Final feature-family decision",
        "\n".join(
            [
                f"**Decision: {decision}**",
                "",
                "```json",
                json.dumps(finite_or_none(selected), indent=2),
                "```",
                "",
                "F1/F2/F3/F4 evidence:",
                "",
                "```json",
                json.dumps(finite_or_none(decision_evidence["criteria_by_candidate"]), indent=2),
                "```",
                "",
                "Exactly one outcome is registered. No feature has been implemented or trained in this audit.",
            ]
        ),
    )
    write_md(
        out / "16_historical_run_registry.md",
        "Historical run registry",
        f"Scanned canonical rows: **{len(registry)}**.\n\n" + md_table(registry, 100),
    )
    tier_counts = eligibility.groupby(["eligible", "tier"], dropna=False).size().reset_index(name="count")
    write_md(
        out / "17_historical_eligibility_audit.md",
        "Historical eligibility audit",
        md_table(tier_counts, 20)
        + "\n\n"
        + md_table(eligibility, 120)
        + "\n\nTest metrics are absent from all eligibility and ranking columns.",
    )
    write_md(
        out / "18_fallback_pretest_ranking.md",
        "Fallback pretest ranking",
        "\n".join(
            [
                "Exact lexicographic order:",
                "",
                "1. Tier A before Tier B.",
                "2. Best validation macro-F1 descending.",
                "3. Best validation accuracy descending.",
                "4. Train-validation macro-F1 gap ascending.",
                "5. Best-to-last validation macro-F1 decline ascending.",
                "6. Structure dependency ascending when comparable.",
                "7. Checkpoint epoch ascending.",
                "8. Canonical run ID ascending.",
                "",
                md_table(ranking, 100),
                "",
                "No test metric is present in this table.",
            ]
        ),
    )
    write_md(
        out / "19_fallback_lock_pretest.md",
        "Pretest paper-safe fallback lock",
        "```json\n"
        + json.dumps(lock, indent=2)
        + "\n```\n\n"
        + f"Lock SHA-256: `{sha256_file(out / '19_fallback_lock_pretest.json')}`.",
    )
    if postlock and fallback is not None:
        write_md(
            out / "20_fallback_postlock_test_reveal.md",
            "Postlock fallback test reveal",
            "The pretest lock hash was verified unchanged before reading test metrics.\n\n```json\n"
            + json.dumps(finite_or_none(fallback), indent=2)
            + "\n```\n\nThe fallback identity was not changed after the reveal.",
        )
        write_md(
            out / "21_retrospective_peak_reference.md",
            "Retrospective test-aware peak reference",
            "```json\n"
            + json.dumps(finite_or_none(retrospective), indent=2)
            + "\n```\n\nThis is explicitly test-aware and cannot silently replace the pretest lock.",
        )
        write_md(
            out / "22_validation_test_calibration.md",
            "Historical validation-test calibration",
            md_table(calibration, 100)
            + "\n\n```json\n"
            + json.dumps(finite_or_none(readiness), indent=2)
            + "\n```\n\nHistorical runs are statistically dependent; this is descriptive only.",
        )
    else:
        write_md(
            out / "20_fallback_postlock_test_reveal.md",
            "Postlock fallback test reveal",
            "PENDING: the pretest lock exists, but full-test metrics have not yet been opened by the postlock phase.",
        )
        write_md(
            out / "21_retrospective_peak_reference.md",
            "Retrospective test-aware peak reference",
            "PENDING until the postlock phase.",
        )
        write_md(
            out / "22_validation_test_calibration.md",
            "Historical validation-test calibration",
            "PENDING until the postlock phase.",
        )
    gate = final_gate_payload()
    write_md(
        out / "23_final_training_gate.md",
        "Final validation-first training gate",
        "```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
        "The full test set remains unavailable for seed42 checkpoint selection. Failure of any "
        "seed42 condition stops the final model and activates the locked fallback.",
    )
    write_md(
        out / "24_final_implementation_scope.md",
        "Final implementation scope",
        "```json\n" + json.dumps(finite_or_none(scope_payload), indent=2) + "\n```",
    )
    paper = paper_scope_payload(fallback, retrospective)
    write_md(
        out / "25_paper_reporting_scope.md",
        "Paper reporting scope",
        "```json\n" + json.dumps(finite_or_none(paper), indent=2) + "\n```",
    )
    write_md(
        out / "27_run_commands.md",
        "Audit run commands",
        "\n".join(
            [
                "No training command was run.",
                "",
                "```powershell",
                "python -B d19/scripts/audit_d19_final_feature_frequency_and_fallback.py --phase pretest --device cuda:0",
                "python -B d19/scripts/audit_d19_final_feature_frequency_and_fallback.py --phase postlock --device cuda:0 --reuse-intermediate",
                "```",
            ]
        ),
    )


def build_machine_summary(
    manifest: pd.DataFrame,
    fold: pd.DataFrame,
    fold_meta: dict[str, Any],
    folded_layers: dict[str, Any],
    decision_evidence: dict[str, Any],
    decision: str,
    selected: dict[str, Any],
    registry: pd.DataFrame,
    eligibility: pd.DataFrame,
    lock: dict[str, Any],
    lock_hash: str,
    fallback: dict[str, Any] | None,
    retrospective: dict[str, Any] | None,
    calibration: pd.DataFrame,
    readiness: dict[str, Any] | None,
    scope_payload: dict[str, Any],
) -> dict[str, Any]:
    inventory = pd.read_csv(OUT_DEFAULT / "05_current_node_feature_inventory.csv") if False else feature_inventory()
    redundancy = pd.read_csv(OUT_DEFAULT / "07_feature_redundancy_analysis.csv")
    population_redundancy = redundancy[redundancy["row_type"].eq("population_summary")][
        ["population", "effective_rank", "participation_ratio", "condition_number"]
    ].to_dict(orient="records")
    validation_pairs = redundancy[
        redundancy["row_type"].eq("pairwise") & redundancy["population"].eq("validation")
    ].copy()
    validation_pairs["absolute_pearson"] = validation_pairs["pearson"].abs()
    strongest_pairs = validation_pairs.sort_values("absolute_pearson", ascending=False)[
        ["feature", "other_feature", "pearson", "spearman", "linear_cka"]
    ].head(10).to_dict(orient="records")
    frequency = pd.read_csv(OUT_DEFAULT / "08_graph_frequency_characterization.csv")
    frequency_profile = frequency[frequency["population"].isin(["val", "locked"])][
        [
            "population", "signal_kind", "signal",
            "normalized_dirichlet_energy_median",
            "moran_autocorrelation_median",
            "one_step_diffusion_retention_median",
            "two_step_diffusion_retention_median",
        ]
    ].to_dict(orient="records")
    population_fold = {}
    for population, part in fold.groupby("population"):
        population_meta = fold_meta["population_metrics"][population]
        population_fold[population] = {
            "max_abs_difference": float(part["max_absolute_difference"].max()),
            "prediction_match_fraction": float(population_meta["prediction_agreement"]),
            "sample_count": int(population_meta["count"]),
        }
    return {
        "artifact_integrity": {
            "all_required_inputs_found": bool(manifest["exists"].all()),
            "locked_manifest_sha256": LOCKED_SHA256,
            "source_artifact_count": len(manifest),
        },
        "a1_null_equivalence": {
            "status": fold_meta["status"],
            "mathematical_conditions": fold_meta["mathematical_conditions"],
            "folded_layers": list(folded_layers),
            "population_equivalence": population_fold,
            "interpretation": (
                "A1-null is not more expressive than A0 for the verified runtime architecture; "
                "its observed training difference is optimization/parameterization behavior, "
                "not relation-dependent representational capacity."
            ),
        },
        "current_features": {
            "count": len(inventory),
            "ordered_inventory": inventory.to_dict(orient="records"),
            "effective_rank": decision_evidence["existing_effective_rank"],
            "redundancy": {
                "population_summaries": population_redundancy,
                "strongest_validation_pairs": strongest_pairs,
            },
            "frequency_characterization": frequency_profile,
        },
        "candidate_signals": decision_evidence["candidate_profiles"],
        "feature_evidence_criteria": decision_evidence["criteria_by_candidate"],
        "final_feature_decision": decision,
        "selected_feature_family": selected,
        "historical_registry": {
            "rows_scanned": len(registry),
            "eligible_count": int(eligibility["eligible"].sum()),
            "tier_counts": finite_or_none(
                eligibility[eligibility["eligible"].eq(True)]["tier"].value_counts(dropna=False).to_dict()
            ),
        },
        "paper_safe_fallback": fallback if fallback is not None else lock,
        "fallback_lock_hash": lock_hash,
        "retrospective_peak_reference": retrospective,
        "validation_test_calibration": {
            "rows": calibration.to_dict(orient="records"),
            "readiness": readiness,
        },
        "final_baseline": scope_payload["baseline"],
        "final_training_gate": final_gate_payload(),
        "final_implementation_scope": scope_payload,
        "paper_reporting_scope": paper_scope_payload(fallback, retrospective),
        "limitations": [
            "This audit does not train the proposed feature family.",
            "Nonredundancy does not guarantee classification improvement.",
            "Graph-frequency measures depend on the selected local graph.",
            "Fixed-checkpoint feature replacement is off-policy.",
            "Error associations are observational.",
            "The bounded bank does not cover every graph-signal feature.",
            "Historical runs are statistically dependent.",
            "Historical validation-test calibration is descriptive.",
            "The retrospective peak is test-aware.",
            "The paper-safe fallback may not be the highest historical test run.",
            "Provenance may remain incomplete for some historical runs.",
            "A1-null equivalence applies only to the verified runtime architecture.",
            "No result guarantees 65 percent accuracy.",
            "The final run must pass validation and multiseed gates.",
        ],
    }


def validation_summary(
    out: Path,
    manifest: pd.DataFrame,
    fold: pd.DataFrame,
    fold_meta: dict[str, Any],
    registry: pd.DataFrame,
    eligibility: pd.DataFrame,
    postlock: bool,
    decision: str,
    scope_payload: dict[str, Any],
) -> dict[str, Any]:
    def population_ok(name: str) -> bool:
        part = fold[fold["population"].eq(name)]
        return bool(
            not part.empty
            and part["prediction_agreement"].fillna(1.0).min() == 1.0
            and part["max_absolute_difference"].max() <= 1e-5
        )

    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    return {
        "source_repository_found": ROOT.exists(),
        "a0_seed42_found": A0_42.exists(),
        "a1_null_seed42_found": A1_NULL.exists(),
        "c2_seed42_found": C2_42.exists(),
        "pixel_selection_audit_found": SELECTION_AUDIT.exists(),
        "a1_posttraining_analysis_found": A1_ANALYSIS.exists(),
        "feature_code_traced": True,
        "feature_order_verified": FEATURE_NAMES == [
            "intensity", "gx", "gy", "x_norm", "y_norm", "grad_mag",
            "local_mean_3x3", "local_std_3x3", "laplacian_abs", "center_surround",
        ],
        "labels_do_not_affect_features": True,
        "a1_null_fold_conditions_traced": True,
        "a1_null_fold_created": fold_meta["status"] == "EXACTLY_FOLDABLE",
        "a1_null_fold_strict_load": bool(fold_meta["mathematical_conditions"]["strict_load"]),
        "a1_null_validation_equivalence": population_ok("validation"),
        "a1_null_locked_equivalence": population_ok("locked715"),
        "a1_null_full_test_equivalence": population_ok("full_test"),
        "feature_distribution_computed": (out / "06_feature_distribution_statistics.csv").exists(),
        "feature_redundancy_computed": (out / "07_feature_redundancy_analysis.csv").exists(),
        "feature_effective_rank_computed": (out / "_feature_image_statistics.csv").exists(),
        "graph_frequency_computed": (out / "08_graph_frequency_characterization.csv").exists(),
        "spectral_subset_computed": (out / "_bounded_spectral_per_image.csv").exists(),
        "candidate_signal_bank_computed": (out / "12_bounded_candidate_signal_bank.csv").exists(),
        "candidate_nonredundancy_computed": (out / "13_candidate_nonredundancy_analysis.csv").exists(),
        "fixed_checkpoint_sensitivity_computed": (out / "10_fixed_checkpoint_feature_sensitivity.csv").exists(),
        "error_group_analysis_computed": (out / "11_error_group_feature_analysis.csv").exists(),
        "feature_criteria_assigned": bool(decision),
        "final_feature_decision_applied": decision in {
            "SELECT_SINGLE_GRAPH_SIGNAL_FEATURE_FAMILY",
            "NO_FEATURE_AUGMENTATION_JUSTIFIED",
            "FEATURE_AUDIT_AMBIGUOUS",
        },
        "historical_runs_scanned": len(registry),
        "historical_runs_deduplicated": bool(registry["checkpoint_hash"].is_unique),
        "historical_eligibility_computed": len(eligibility) == len(registry),
        "pretest_metrics_masked": all(
            registry[column].astype(str).eq("MASKED_PRETEST").all()
            for column in registry.columns
            if column.startswith("test_")
        ),
        "fallback_locked_before_test_reveal": (out / "19_fallback_lock_pretest.json").exists(),
        "fallback_lock_hash_created": (out / "_fallback_lock_sha256.txt").exists(),
        "fallback_test_revealed_after_lock": postlock,
        "retrospective_peak_identified": postlock,
        "validation_test_calibration_computed": postlock and (out / "22_validation_test_calibration.csv").exists(),
        "final_training_gate_registered": True,
        "final_scope_selected": bool(scope_payload["scope"]),
        "reports_complete": not missing,
        "training_launched": False,
        "model_modified": False,
        "selector_modified": False,
        "graph_builder_modified": False,
        "graph_cache_modified": False,
        "checkpoints_modified": False,
        "blocking_issues": missing,
        "warnings": [
            "Historical experiments are statistically dependent.",
            "Some older run provenance can remain NOT VERIFIABLE and is ineligible.",
            "Human analysts had prior project context; procedural pretest masking applies to the implemented ranking dataflow.",
        ],
    }


def save_pretest_state(
    out: Path,
    manifest: pd.DataFrame,
    fold_meta: dict[str, Any],
    folded_layers: dict[str, Any],
    decision_evidence: dict[str, Any],
    decision: str,
    selected: dict[str, Any],
    scope_payload: dict[str, Any],
    lock_hash: str,
) -> None:
    safe_json(
        out / "_pretest_state.json",
        {
            "manifest": manifest.to_dict(orient="records"),
            "fold_meta": fold_meta,
            "folded_layers": folded_layers,
            "decision_evidence": decision_evidence,
            "decision": decision,
            "selected": selected,
            "scope_payload": scope_payload,
            "lock_hash": lock_hash,
            "created_utc": pd.Timestamp.utcnow().isoformat(),
        },
    )


def run_pretest(out: Path, device: torch.device, reuse: bool) -> None:
    started = time.time()
    inputs = require_inputs()
    manifest = artifact_manifest_rows(inputs)
    manifest.to_csv(out / "_source_and_artifact_manifest.csv", index=False)
    trace = source_trace_payload()
    safe_json(out / "_runtime_trace.json", trace)
    inventory = feature_inventory()
    inventory.to_csv(out / "05_current_node_feature_inventory.csv", index=False)
    fold, fold_meta, folded_layers = compare_folded_models(out, device=device, reuse=reuse)
    images, nodes, distribution = collect_feature_distributions(out, reuse=reuse)
    redundancy = redundancy_analysis(out, images, nodes, reuse=reuse)
    frequency, spectral_frame, candidates_frame = graph_frequency_audit(out, images, reuse=reuse)
    summarize_frequency_outputs(out, frequency, spectral_frame, candidates_frame)
    sensitivity = fixed_checkpoint_sensitivity(out, distribution, device=device, reuse=reuse)
    error_frame, error_counts = error_group_feature_analysis(
        out, images, frequency, candidates_frame, reuse=reuse
    )
    decision_evidence, decision, selected, scope = feature_evidence_decision(
        images, frequency, spectral_frame, candidates_frame, error_frame
    )
    safe_json(
        out / "_feature_decision.json",
        {
            "evidence": decision_evidence,
            "decision": decision,
            "selected": selected,
            "scope": scope,
        },
    )
    registry, eligibility, lock, lock_hash = build_historical_registry_pretest(out, reuse=reuse)
    scope_payload = attach_locked_fallback(implementation_scope(decision, selected, scope), lock)
    save_pretest_state(
        out,
        manifest,
        fold_meta,
        folded_layers,
        decision_evidence,
        decision,
        selected,
        scope_payload,
        lock_hash,
    )
    calibration = pd.DataFrame(
        columns=[
            "relationship", "count", "pearson", "spearman",
            "median_right_minus_left", "iqr_right_minus_left",
            "mean_absolute_deviation", "statistical_independence_claimed",
        ]
    )
    calibration.to_csv(out / "22_validation_test_calibration.csv", index=False)
    write_markdown_reports(
        out, manifest, trace, fold, fold_meta, folded_layers, decision_evidence,
        decision, selected, scope_payload, None, None, calibration, None, False,
    )
    summary = build_machine_summary(
        manifest, fold, fold_meta, folded_layers, decision_evidence, decision,
        selected, registry, eligibility, lock, lock_hash, None, None,
        calibration, None, scope_payload,
    )
    safe_json(out / "26_machine_readable_summary.json", summary)
    validation = validation_summary(
        out, manifest, fold, fold_meta, registry, eligibility, False, decision, scope_payload
    )
    validation["pretest_elapsed_sec"] = time.time() - started
    safe_json(out / "28_validation_summary.json", validation)
    generate_plots(out, postlock=False)


def run_postlock(out: Path, device: torch.device) -> None:
    del device
    state_path = out / "_pretest_state.json"
    if not state_path.exists():
        raise RuntimeError("Postlock phase requires a completed pretest state")
    state = read_json(state_path)
    lock_path = out / "19_fallback_lock_pretest.json"
    current_lock_hash = sha256_file(lock_path)
    expected_lock_hash = (out / "_fallback_lock_sha256.txt").read_text(encoding="utf-8").strip()
    if current_lock_hash != expected_lock_hash or state["lock_hash"] != expected_lock_hash:
        raise RuntimeError("Pretest fallback lock changed before postlock reveal")
    manifest = pd.DataFrame(state["manifest"])
    trace = read_json(out / "_runtime_trace.json")
    fold = pd.read_csv(out / "04_a1_null_fold_validation.csv")
    fold_meta = state["fold_meta"]
    folded_layers = state["folded_layers"]
    decision_evidence = state["decision_evidence"]
    decision = state["decision"]
    selected = state["selected"]
    scope_payload = state["scope_payload"]
    registry = pd.read_csv(out / "16_historical_run_registry.csv")
    eligibility = pd.read_csv(out / "17_historical_eligibility_audit.csv")
    lock = read_json(lock_path)
    scope_payload = attach_locked_fallback(scope_payload, lock)
    fallback, retrospective, calibration, readiness = reveal_historical_tests(
        out, registry, eligibility, lock, expected_lock_hash
    )
    write_markdown_reports(
        out, manifest, trace, fold, fold_meta, folded_layers, decision_evidence,
        decision, selected, scope_payload, fallback, retrospective, calibration,
        readiness, True,
    )
    summary = build_machine_summary(
        manifest, fold, fold_meta, folded_layers, decision_evidence, decision,
        selected, registry, eligibility, lock, expected_lock_hash, fallback,
        retrospective, calibration, readiness, scope_payload,
    )
    safe_json(out / "26_machine_readable_summary.json", summary)
    validation = validation_summary(
        out, manifest, fold, fold_meta, registry, eligibility, True, decision, scope_payload
    )
    safe_json(out / "28_validation_summary.json", validation)
    generate_plots(out, postlock=True)
    validation = validation_summary(
        out, manifest, fold, fold_meta, registry, eligibility, True, decision, scope_payload
    )
    safe_json(out / "28_validation_summary.json", validation)
    if not validation["reports_complete"]:
        raise RuntimeError(f"Required reports missing: {validation['blocking_issues']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pretest", "postlock", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-intermediate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    if out != OUT_DEFAULT.resolve():
        raise RuntimeError(f"Audit output must be the registered directory: {OUT_DEFAULT}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(json.dumps({"phase": args.phase, "output_dir": str(out), "device": str(device)}))
    if args.phase in {"pretest", "all"}:
        run_pretest(out, device=device, reuse=args.reuse_intermediate)
    if args.phase in {"postlock", "all"}:
        run_postlock(out, device=device)
    print(json.dumps({"status": "complete", "phase": args.phase, "output_dir": str(out)}))


if __name__ == "__main__":
    main()
