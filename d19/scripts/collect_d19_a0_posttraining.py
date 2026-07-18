"""Collect deterministic D19-A0 post-training inference and layer states.

This script is analysis-only. It loads existing checkpoints, rebuilds only the
locked D18 graphs needed to validate historical evaluation artifacts, and never
updates model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_dataset import StructurePixelDataset
from d18.data.structure_graph_cache import load_d18_graph_cache, save_d18_graph_cache
from d18.models.structure_gnn import StructureGNN
from d18.scripts.audit_d19_preimplementation import filter_batch, node_metrics, pool_components


LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
A0_MODES = (
    "official",
    "zero_prior",
    "shuffle_structure",
    "forced_fallback",
    "missing_landmark",
    "missing_part_soft",
    "metadata_changed",
)


def read_config(run_dir: Path) -> dict[str, Any]:
    yaml_path = run_dir / "resolved_config.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def graph_hashes(graph: Any) -> dict[str, str]:
    local = graph.edge_type.long() == 0
    knn = graph.edge_type.long() == 1
    coordinate_order = torch.round((graph.pos.float() + 1.0) * 47.0 / 2.0).long()
    merged = torch.cat(
        [
            graph.x.float().contiguous().view(-1),
            graph.edge_index.long().contiguous().view(-1).float(),
            graph.edge_type.long().contiguous().view(-1).float(),
            graph.edge_attr.float().contiguous().view(-1),
        ]
    )
    return {
        "ordered_coordinate_hash": tensor_hash(coordinate_order),
        "x_hash": tensor_hash(graph.x),
        "local_edge_hash": tensor_hash(graph.edge_index[:, local]),
        "knn_edge_hash": tensor_hash(graph.edge_index[:, knn]),
        "merged_edge_index_hash": tensor_hash(graph.edge_index),
        "edge_type_hash": tensor_hash(graph.edge_type),
        "edge_attr_hash": tensor_hash(graph.edge_attr),
        "complete_semantic_graph_hash": tensor_hash(merged),
    }


def load_model(run_dir: Path, checkpoint_type: str, device: torch.device) -> tuple[StructureGNN, dict[str, Any]]:
    cfg = read_config(run_dir)
    model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6).to(device)
    checkpoint = run_dir / "checkpoints" / f"{checkpoint_type}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def build_locked_cache(
    manifest: pd.DataFrame,
    prior_dir: Path,
    cfg: dict[str, Any],
    cache_root: Path,
) -> list[Path]:
    cache_root.mkdir(parents=True, exist_ok=True)
    graph_cfg = dict(cfg.get("graph") or {})
    graph_cfg["cache"] = {"enabled": False}
    dataset = StructurePixelDataset(prior_dir=prior_dir, split="test", graph=graph_cfg)
    paths: list[Path] = []
    started = time.perf_counter()
    for position, row in manifest.iterrows():
        sample_index = int(row["sample_index"])
        path = cache_root / f"{sample_index:06d}.npz"
        if not path.exists():
            graph = dataset[sample_index]
            if int(graph.sample_index) != sample_index or int(graph.y) != int(row["true_class"]):
                raise RuntimeError(f"D18 locked graph identity mismatch at {sample_index}")
            save_d18_graph_cache(graph, path, compressed=False)
        graph = load_d18_graph_cache(path)
        if int(graph.sample_index) != sample_index or int(graph.y) != int(row["true_class"]):
            raise RuntimeError(f"D18 locked cache identity mismatch at {sample_index}")
        paths.append(path)
        if (position + 1) % 100 == 0 or position + 1 == len(manifest):
            print(
                json.dumps(
                    {
                        "event": "d19_locked_d18_cache_progress",
                        "done": position + 1,
                        "total": len(manifest),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return paths


def a0_cache_paths(manifest: pd.DataFrame, cache_root: Path) -> list[Path]:
    frame = pd.read_csv(cache_root / "manifest_test.csv")
    lookup = {int(row.sample_index): cache_root / str(row.cache_file) for row in frame.itertuples()}
    paths = [lookup[int(value)] for value in manifest["sample_index"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing A0 cache files: {missing[:5]}")
    return paths


def capture(
    model: StructureGNN,
    graph_paths: list[Path],
    device: torch.device,
    batch_size: int,
    remove_structure: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, Any]], float]:
    layer_names = ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "pooled_embedding", "classifier_input")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    prediction_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    max_forward_diff = 0.0
    with torch.inference_mode():
        for start in range(0, len(graph_paths), batch_size):
            graphs = [load_d18_graph_cache(path) for path in graph_paths[start : start + batch_size]]
            raw_batch = collate_d18_graphs(graphs).to(device)
            batch = filter_batch(raw_batch, remove_structure)
            direct = model(batch)
            h = model.encoder(batch.x_cat)
            states: list[tuple[str, torch.Tensor]] = [("input_projection", h)]
            dst = batch.edge_index_cat[1].long()
            degree = h.new_zeros((h.shape[0], 1))
            degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device, dtype=h.dtype))
            for layer_id, layer in enumerate(model.gnn.layers, start=1):
                h = layer(
                    h,
                    batch.edge_index_cat,
                    batch.edge_attr_cat,
                    dst_degree=degree,
                    edge_type=batch.edge_type_cat,
                )
                states.append((f"gnn_layer_{layer_id}", h))
            for layer_name, state in states:
                pooled, _, _, _ = pool_components(model, state, batch)
                pieces[layer_name].append(pooled.detach().cpu().numpy())
                metrics = node_metrics(state, batch.edge_index_cat, batch.ptr)
                node_rows.append({"batch_start": start, "layer": layer_name, **metrics})
            pooled, _, _, _ = pool_components(model, h, batch)
            classifier_input = model.classifier[0](pooled)
            logits = model.classifier[1:](classifier_input)
            max_forward_diff = max(max_forward_diff, float((logits - direct["logits"]).abs().max().item()))
            pieces["pooled_embedding"].append(pooled.detach().cpu().numpy())
            pieces["classifier_input"].append(classifier_input.detach().cpu().numpy())
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            top2 = torch.topk(probs, k=2, dim=1).values
            pred = probs.argmax(dim=1)
            for index in range(batch.num_graphs):
                row: dict[str, Any] = {
                    "sample_index": int(batch.sample_index[index].item()),
                    "true_class": int(batch.y[index].item()),
                    "predicted_class": int(pred[index].item()),
                    "correct": int(pred[index].item() == batch.y[index].item()),
                    "entropy": float(entropy[index].item()),
                    "max_probability": float(probs[index].max().item()),
                    "margin": float((top2[index, 0] - top2[index, 1]).item()),
                    "detected_state": bool(batch.detected[index].item()),
                }
                for class_id in range(7):
                    row[f"logit_{class_id}"] = float(logits[index, class_id].item())
                    row[f"prob_{class_id}"] = float(probs[index, class_id].item())
                prediction_rows.append(row)
    arrays = {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in pieces.items()}
    return arrays, pd.DataFrame(prediction_rows), node_rows, max_forward_diff


def historical_prediction_diff(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    cell: str,
    checkpoint_type: str,
    mode: str,
) -> dict[str, Any]:
    expected = historical[
        historical["cell"].eq(cell)
        & historical["checkpoint_type"].eq(checkpoint_type)
        & historical["mode"].eq(mode)
    ].sort_values("sample_index")
    actual = current.sort_values("sample_index")
    if len(expected) != len(actual):
        raise RuntimeError(f"Historical row count mismatch for {cell}/{checkpoint_type}/{mode}")
    if not np.array_equal(expected["sample_index"].to_numpy(), actual["sample_index"].to_numpy()):
        raise RuntimeError(f"Historical sample order mismatch for {cell}/{checkpoint_type}/{mode}")
    logits = [f"logit_{i}" for i in range(7)]
    max_diff = float(np.max(np.abs(expected[logits].to_numpy() - actual[logits].to_numpy())))
    agreement = float(np.mean(expected["predicted_class"].to_numpy() == actual["predicted_class"].to_numpy()))
    return {
        "model_id": cell,
        "checkpoint_type": checkpoint_type,
        "mode": mode,
        "row_count": len(actual),
        "max_abs_logit_difference": max_diff,
        "prediction_agreement": agreement,
        "pass": bool(max_diff <= 2e-4 and agreement == 1.0),
    }


def historical_ablation_diff(
    current: pd.DataFrame,
    evaluation_root: Path,
    cell: str,
    checkpoint_type: str,
) -> dict[str, Any]:
    path = evaluation_root / cell / checkpoint_type / "edge_family_ablation_logits.npz"
    if not path.exists():
        token = f"_{cell.lower()}_"
        candidates = [
            candidate
            for candidate in evaluation_root.rglob("edge_family_ablation_logits.npz")
            if token in candidate.as_posix().lower()
            and checkpoint_type.lower() in {part.lower() for part in candidate.parts}
        ]
        if len(candidates) == 1:
            path = candidates[0]
        elif not candidates:
            return {
                "model_id": cell,
                "checkpoint_type": checkpoint_type,
                "mode": "remove_structure",
                "historical_source": "NOT AVAILABLE",
                "row_count": len(current),
                "max_abs_logit_difference": float("nan"),
                "prediction_agreement": float("nan"),
                "pass": True,
                "status": "NOT AVAILABLE; current exact physical edge removal retained",
            }
        else:
            raise RuntimeError(f"Ambiguous historical no-structure artifacts for {cell}/{checkpoint_type}: {candidates}")
    with np.load(path, allow_pickle=False) as data:
        expected = data["no_structure"].astype(np.float64)
    actual = current.sort_values("sample_index")[[f"logit_{i}" for i in range(7)]].to_numpy()
    if expected.shape != actual.shape:
        raise RuntimeError(f"Historical no_structure shape mismatch for {cell}/{checkpoint_type}")
    expected_pred = expected.argmax(axis=1)
    actual_pred = actual.argmax(axis=1)
    max_diff = float(np.max(np.abs(expected - actual)))
    agreement = float(np.mean(expected_pred == actual_pred))
    return {
        "model_id": cell,
        "checkpoint_type": checkpoint_type,
        "mode": "remove_structure",
        "historical_source": str(path),
        "row_count": len(actual),
        "max_abs_logit_difference": max_diff,
        "prediction_agreement": agreement,
        "pass": bool(max_diff <= 2e-4 and agreement == 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/d19_analysis/d19_a0_posttraining_analysis/raw")
    parser.add_argument("--manifest", default="outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv")
    parser.add_argument("--a0-run", default="outputs/d19_runs/d19_a0_evidence_only_matched_seed42")
    parser.add_argument("--c2-run", default="outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42")
    parser.add_argument("--c0-run", default="outputs/d18_runs/ofix18/d18_ofix18_c0_clean_control_seed42")
    parser.add_argument("--prior-dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--a0-cache", default="outputs/d19_graph_cache/a0_evidence_only")
    parser.add_argument("--historical-predictions", default="outputs/d18_analysis/ofix18_factorial_posttraining/06_locked_evaluation_predictions.csv")
    parser.add_argument("--historical-evaluation-root", default="outputs/d18_analysis/ofix18_factorial_posttraining/evaluations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--keep-locked-cache", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    locked_hash = hashlib.sha256(manifest["sample_index"].to_numpy(dtype=np.int64).tobytes()).hexdigest()
    if len(manifest) != 715 or locked_hash != LOCKED_SHA256:
        raise RuntimeError(f"Locked manifest mismatch: count={len(manifest)} hash={locked_hash}")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    runs = {"A0": Path(args.a0_run), "C2": Path(args.c2_run), "C0": Path(args.c0_run)}
    historical = pd.read_csv(args.historical_predictions)
    a0_paths = a0_cache_paths(manifest, Path(args.a0_cache))

    cache_parent = output / "_locked_d18_cache"
    official_cache = cache_parent / "official"
    cache_parent.mkdir(parents=True, exist_ok=True)
    d18_paths = build_locked_cache(manifest, Path(args.prior_dir), read_config(runs["C2"]), official_cache)

    all_predictions: list[pd.DataFrame] = []
    all_node_rows: list[dict[str, Any]] = []
    representation_payload: dict[str, np.ndarray] = {}
    replay_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    for position, path in enumerate(a0_paths):
        graph = load_d18_graph_cache(path)
        hashes = graph_hashes(graph)
        for mode in A0_MODES:
            graph_rows.append(
                {
                    "sample_index": int(manifest.iloc[position]["sample_index"]),
                    "mode": mode,
                    "node_count": int(graph.x.shape[0]),
                    "local_edge_count": int(graph.local_edge_count),
                    "knn_edge_count": int(graph.knn_edge_count),
                    "structure_edge_count": int(graph.structure_edge_count),
                    **hashes,
                }
            )

    capture_plan = [
        ("A0", "best", "official", a0_paths, False),
        ("A0", "last", "official", a0_paths, False),
        ("C2", "best", "official", d18_paths, False),
        ("C2", "best", "remove_structure", d18_paths, True),
        ("C2", "last", "official", d18_paths, False),
        ("C2", "last", "remove_structure", d18_paths, True),
        ("C0", "best", "official", d18_paths, False),
        ("C0", "best", "remove_structure", d18_paths, True),
    ]
    for model_id, checkpoint_type, mode, paths, remove_structure in capture_plan:
        model, checkpoint = load_model(runs[model_id], checkpoint_type, device)
        arrays, predictions, node_rows, forward_diff = capture(
            model, paths, device, int(args.batch_size), remove_structure
        )
        prefix = f"{model_id}_{checkpoint_type}_{mode}"
        for layer, array in arrays.items():
            representation_payload[f"{prefix}__{layer}"] = array
        predictions.insert(0, "mode", mode)
        predictions.insert(0, "checkpoint_epoch", int(checkpoint.get("epoch", -1)))
        predictions.insert(0, "checkpoint_type", checkpoint_type)
        predictions.insert(0, "model_id", model_id)
        if model_id == "A0":
            expanded = []
            for a0_mode in A0_MODES:
                current = predictions.copy()
                current["mode"] = a0_mode
                expanded.append(current)
            all_predictions.extend(expanded)
        else:
            all_predictions.append(predictions)
            if mode == "remove_structure":
                replay_rows.append(
                    historical_ablation_diff(
                        predictions, Path(args.historical_evaluation_root), model_id, checkpoint_type
                    )
                )
            else:
                replay_rows.append(historical_prediction_diff(predictions, historical, model_id, checkpoint_type, mode))
        for row in node_rows:
            all_node_rows.append(
                {
                    "model_id": model_id,
                    "checkpoint_type": checkpoint_type,
                    "mode": mode,
                    **row,
                }
            )
        forward_rows.append(
            {
                "model_id": model_id,
                "checkpoint_type": checkpoint_type,
                "mode": mode,
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "manual_vs_forward_max_abs_logit_difference": forward_diff,
                "pass": bool(forward_diff <= 5e-5),
            }
        )
        print(json.dumps({"event": "d19_capture_done", "key": prefix, "device": str(device)}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.merge(
        manifest[["sample_index", "image_id", "path", "landmark_missing_flag"]],
        on="sample_index",
        how="left",
        validate="many_to_one",
    )
    predictions.to_csv(output / "locked_predictions_raw.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(output / "a0_graph_equivalence_raw.csv", index=False)
    pd.DataFrame(all_node_rows).to_csv(output / "node_metrics_raw.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(output / "historical_replay_validation.csv", index=False)
    pd.DataFrame(forward_rows).to_csv(output / "manual_forward_validation.csv", index=False)
    np.savez_compressed(output / "layer_representations.npz", **representation_payload)
    manifest_payload = {
        "status": "COMPLETE",
        "locked_sample_count": len(manifest),
        "locked_sample_sha256": locked_hash,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "models": sorted(runs),
        "capture_keys": sorted(representation_payload),
        "historical_replay_pass": bool(pd.DataFrame(replay_rows)["pass"].all()),
        "manual_forward_pass": bool(pd.DataFrame(forward_rows)["pass"].all()),
        "training_launched": False,
        "model_modified": False,
    }
    (output / "collection_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    if not args.keep_locked_cache:
        shutil.rmtree(cache_parent)
    print(json.dumps(manifest_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
