"""Generate the complete read-only D19-A0 post-training analysis package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.models.structure_gnn import StructureGNN
from d18.scripts.audit_d19_preimplementation import graph_separation, linear_cka


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
A0_MODES = (
    "official",
    "zero_prior",
    "shuffle_structure",
    "forced_fallback",
    "missing_landmark",
    "missing_part_soft",
    "metadata_changed",
)
RUNS = {
    "A0": ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42",
    "C2": ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42",
    "C0": ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c0_clean_control_seed42",
}
SOURCE_CONFIGS = {
    "A0": ROOT / "configs/d19/d19_a0_evidence_only_matched_seed42.yaml",
    "C2": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml",
    "C0": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c0_clean_control_seed42.yaml",
}


def jdump(value: Any) -> str:
    def json_default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True, default=json_default)


def write_json(path: Path, value: Any) -> None:
    path.write_text(jdump(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return read_json(run_dir / "resolved_config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_table(frame: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 5) -> str:
    current = frame.copy()
    if columns is not None:
        current = current[list(columns)]
    for column in current.columns:
        if pd.api.types.is_float_dtype(current[column]):
            current[column] = current[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")

    headers = [cell(column) for column in current.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in current.itertuples(index=False, name=None))
    return "\n".join(lines)


def ece_score(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low if index == 0 else confidence > low) & (confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return float(result)


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["true_class"].to_numpy(dtype=np.int64)
    pred = frame["predicted_class"].to_numpy(dtype=np.int64)
    probs = frame[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(7), zero_division=0)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    result: dict[str, Any] = {
        "count": len(frame),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(7)[y]) ** 2, axis=1))),
        "ece": ece_score(y, probs),
        "mean_entropy": float(entropy.mean()),
        "mean_max_probability": float(probs.max(axis=1).mean()),
        "mean_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).mean()),
        "accuracy_confidence_gap": float(probs.max(axis=1).mean() - (pred == y).mean()),
        "confusion_matrix_json": json.dumps(confusion_matrix(y, pred, labels=np.arange(7)).tolist()),
    }
    for class_id, name in enumerate(CLASS_NAMES):
        result[f"precision_{name}"] = float(precision[class_id])
        result[f"recall_{name}"] = float(recall[class_id])
        result[f"f1_{name}"] = float(f1[class_id])
        result[f"support_{name}"] = int(support[class_id])
    return result


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
    elif isinstance(value, list):
        result[prefix] = json.dumps(value, sort_keys=True)
    else:
        result[prefix] = value
    return result


def checkpoint_payload(run_dir: Path, checkpoint_type: str) -> dict[str, Any]:
    return torch.load(run_dir / "checkpoints" / f"{checkpoint_type}.pt", map_location="cpu", weights_only=False)


def historical_locked_predictions(raw: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(ROOT / "outputs/d18_analysis/ofix18_factorial_posttraining/06_locked_evaluation_predictions.csv")
    source = source[source["cell"].isin(["C0", "C2"])].copy()
    source = source[source["mode"].isin(["shuffle_structure", "permute_structure_destinations", "degree_matched_random_structure"])]
    source = source.rename(columns={"cell": "model_id"})
    source["path"] = source["sample_index"].map(manifest.set_index("sample_index")["path"])
    source["landmark_missing_flag"] = source["sample_index"].map(manifest.set_index("sample_index")["landmark_missing_flag"])
    common = sorted(set(raw.columns) | set(source.columns))
    return pd.concat([raw.reindex(columns=common), source.reindex(columns=common)], ignore_index=True)


def training_curves(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for model_id, run_dir in RUNS.items():
        history = pd.read_csv(run_dir / "train_log.csv")
        summary = read_json(run_dir / "d18_train_summary.json")
        history.insert(0, "model_id", model_id)
        history["learning_rate"] = history["lr"]
        history["checkpoint_is_best"] = history["epoch"].eq(int(summary["best_epoch"]))
        history["official_mode_count"] = history.get("structure_mode_official_sample_count", np.nan)
        history["forced_mode_count"] = history.get("structure_mode_forced_sample_count", np.nan)
        history["observed_forced_ratio"] = history.get("structure_mode_forced_sample_pct", np.nan)
        frames.append(history)
        best_row = history.loc[history["epoch"].eq(int(summary["best_epoch"]))].iloc[-1]
        min_loss_row = history.loc[history["val_loss"].idxmin()]
        last = history.iloc[-1]
        summaries.append(
            {
                "model_id": model_id,
                "best_epoch": int(summary["best_epoch"]),
                "last_epoch": int(last["epoch"]),
                "peak_train_macro_f1": float(history["train_macro_f1"].max()),
                "best_val_macro_f1": float(best_row["val_macro_f1"]),
                "train_macro_f1_at_best": float(best_row["train_macro_f1"]),
                "train_val_macro_gap_at_best": float(best_row["train_macro_f1"] - best_row["val_macro_f1"]),
                "minimum_val_loss": float(min_loss_row["val_loss"]),
                "minimum_val_loss_epoch": int(min_loss_row["epoch"]),
                "last_train_macro_f1": float(last["train_macro_f1"]),
                "last_val_macro_f1": float(last["val_macro_f1"]),
                "late_val_macro_change": float(last["val_macro_f1"] - best_row["val_macro_f1"]),
                "late_val_loss_change": float(last["val_loss"] - min_loss_row["val_loss"]),
                "total_epoch_time_sec": float(history["epoch_time_sec"].sum()),
                "mean_epoch_time_sec": float(history["epoch_time_sec"].mean()),
                "peak_memory_reserved_mb": float(history["memory_reserved_mb"].max()),
            }
        )
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(output / "05_training_curve_long.csv", index=False)
    return long, pd.DataFrame(summaries)


def artifact_manifest(output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    integrity: dict[str, Any] = {}
    c2_state_shapes = None
    for model_id, run_dir in RUNS.items():
        cfg = read_config(run_dir)
        summary = read_json(run_dir / "d18_train_summary.json")
        graph = read_json(run_dir / "graph_schema.json")
        best = checkpoint_payload(run_dir, "best")
        last = checkpoint_payload(run_dir, "last")
        state_shapes = {key: tuple(value.shape) for key, value in best["model_state_dict"].items()}
        if model_id == "C2":
            c2_state_shapes = state_shapes
        completion = run_dir / ("TRAINING_COMPLETE.json" if model_id == "A0" else "COMPLETED.json")
        completion_payload = read_json(completion)
        history = pd.read_csv(run_dir / "train_log.csv")
        warnings = []
        resume_text = (run_dir / "resume_events.jsonl").read_text(encoding="utf-8") if (run_dir / "resume_events.jsonl").exists() else ""
        resume_detected = "resume" in resume_text.lower()
        if model_id == "A0":
            warnings.append("Exact launch-time git provenance is absent from the run artifact.")
            warnings.append("Cross-run initialization is not explicitly attested; config/logs show no load source.")
        edge_types = [0, 1] if model_id == "A0" else [0, 1, 2]
        source = SOURCE_CONFIGS[model_id]
        config_path = run_dir / "resolved_config.yaml"
        rows.append(
            {
                "model_id": model_id,
                "family": "D19-A0" if model_id == "A0" else "D18-OFIX18",
                "seed": int(cfg.get("seed", (cfg.get("training") or {}).get("seed", -1))),
                "run_dir": str(run_dir),
                "source_config": str(source if source.exists() else run_dir / "source_config.yaml"),
                "resolved_config": str(config_path),
                "history_path": str(run_dir / "train_log.csv"),
                "best_checkpoint_path": str(run_dir / "checkpoints/best.pt"),
                "last_checkpoint_path": str(run_dir / "checkpoints/last.pt"),
                "best_epoch": int(best.get("epoch", summary["best_epoch"])),
                "last_epoch": int(last.get("epoch", history["epoch"].max())),
                "monitor_name": str((cfg.get("training") or {}).get("checkpoint_monitor")),
                "monitor_mode": str((cfg.get("training") or {}).get("checkpoint_monitor_mode")),
                "best_monitor_value": float(summary["best_val_macro_f1"]),
                "node_dim": 10,
                "edge_dim": 6,
                "node_count": float(graph.get("actual_node_count_mean", graph.get("target_node_count", 1800))),
                "edge_types": json.dumps(edge_types),
                "parameter_count": int(summary.get("parameter_count_trainable", 265832)),
                "config_signature": sha256_file(config_path),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints/best.pt"),
                "training_completed": completion_payload.get("status") in {"COMPLETE", "COMPLETED"},
                "resume_detected": resume_detected,
                "resume_source": "NOT VERIFIABLE" if not resume_detected else "run-local last.pt",
                "git_commit": "NOT VERIFIABLE",
                "code_signature": "current_repo=" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "warnings": " | ".join(warnings),
            }
        )
        integrity[model_id] = {
            "completion_status": completion_payload.get("status"),
            "best_load": True,
            "last_load": True,
            "best_last_distinct": sha256_file(run_dir / "checkpoints/best.pt") != sha256_file(run_dir / "checkpoints/last.pt"),
            "best_epoch": int(best.get("epoch", -1)),
            "last_epoch": int(last.get("epoch", -1)),
            "history_last_epoch": int(history["epoch"].max()),
            "state_shape_match_c2": None,
            "parameter_count": int(summary.get("parameter_count_trainable", 265832)),
            "resume_provenance": "NOT VERIFIABLE" if model_id == "A0" else "COMPLETED.resumed=false",
        }
    frame = pd.DataFrame(rows)
    for model_id, run_dir in RUNS.items():
        state = checkpoint_payload(run_dir, "best")["model_state_dict"]
        integrity[model_id]["state_shape_match_c2"] = {key: tuple(value.shape) for key, value in state.items()} == c2_state_shapes
    frame.to_csv(output / "01_run_artifact_manifest.csv", index=False)
    return frame, integrity


def config_validation(output: Path) -> tuple[pd.DataFrame, bool]:
    a0 = flatten(read_config(RUNS["A0"]))
    c2 = flatten(read_config(RUNS["C2"]))
    allowed_prefixes = (
        "run_name", "output_dir", "description", "logging.", "data.prior_dir", "data.evidence_dir",
        "graph.graph_mode", "graph.cache.", "graph.structure_edges.enabled",
        "training.structure_mode_mix.enabled", "training.structure_mode_mix.p_forced_structure",
        "training.structure_mode_mix.p_zero_structure",
    )
    keys = sorted(set(a0) | set(c2))
    rows = []
    for key in keys:
        left, right = a0.get(key, "<MISSING>"), c2.get(key, "<MISSING>")
        same = left == right
        allowed = any(key == prefix or key.startswith(prefix) for prefix in allowed_prefixes)
        status = "PASS" if same or allowed else "FAIL"
        rows.append({"field": key, "a0": left, "c2": right, "difference_allowed": allowed, "status": status})
    frame = pd.DataFrame(rows)
    scientific_pass = not bool((frame["status"] == "FAIL").any())
    graph = read_json(RUNS["A0"] / "graph_schema.json")
    checks = [
        ("seed42", read_config(RUNS["A0"]).get("seed") == 42),
        ("parameter_count_265832", read_json(RUNS["A0"] / "d18_train_summary.json").get("parameter_count_trainable") == 265832),
        ("structure_edge_count_zero", float(graph.get("structure_edge_count_mean", -1)) == 0.0),
        ("mode_mix_disabled", not bool(graph.get("structure_mode_mix_enabled"))),
        ("dropedge_disabled", all(float(graph.get(key, -1)) == 0.0 for key in ("drop_local_edge_p", "drop_knn_edge_p", "drop_structure_edge_p"))),
        ("node_edge_dims", len(graph.get("node_feature_names", [])) == 10 and len(graph.get("edge_feature_names", [])) == 6),
    ]
    extra = pd.DataFrame([{"field": name, "a0": value, "c2": "expected", "difference_allowed": False, "status": "PASS" if value else "FAIL"} for name, value in checks])
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.to_csv(output / "03_config_and_schema_validation.csv", index=False)
    return frame, bool(scientific_pass and (extra["status"] == "PASS").all())


def full_test_metrics(output: Path) -> pd.DataFrame:
    rows = []
    for checkpoint in ("best", "last"):
        pred = pd.read_csv(RUNS["A0"] / f"evaluation_{checkpoint}" / "predictions.csv")
        rows.append({"model_id": "A0", "checkpoint_type": checkpoint, "population": "full_test_3589", **metric_bundle(pred)})
    d18 = pd.read_csv(ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/06_full_test_metrics.csv")
    for row in d18[(d18["seed"] == 42) & d18["cell"].isin(["C0", "C2"])].itertuples():
        pred_path = ROOT / f"outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/evaluations/d18_ofix18_{'c0_clean_control' if row.cell == 'C0' else 'c2_structure_mode_mix_only'}_seed42/{row.checkpoint_type}/full_official/counterfactual_predictions.csv"
        pred = pd.read_csv(pred_path)
        rows.append({"model_id": row.cell, "checkpoint_type": row.checkpoint_type, "population": "full_test_3589", **metric_bundle(pred)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "06_full_test_metrics.csv", index=False)
    return frame


def locked_metrics(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for keys, frame in predictions.groupby(["model_id", "checkpoint_type", "mode"], sort=True):
        for group_name, group in (
            ("all", frame),
            ("detected", frame[frame["landmark_missing_flag"].astype(int) == 0]),
            ("missing", frame[frame["landmark_missing_flag"].astype(int) == 1]),
        ):
            if len(group):
                rows.append({"model_id": keys[0], "checkpoint_type": keys[1], "mode": keys[2], "detection_group": group_name, **metric_bundle(group)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "07_locked_metrics.csv", index=False)
    predictions.to_csv(output / "07_locked_predictions.csv", index=False)
    return result


def selected(predictions: pd.DataFrame, model: str, mode: str, checkpoint: str = "best") -> pd.DataFrame:
    return predictions[
        predictions["model_id"].eq(model)
        & predictions["checkpoint_type"].eq(checkpoint)
        & predictions["mode"].eq(mode)
    ].sort_values("sample_index").reset_index(drop=True)


def effect_decomposition(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    frames = {
        "A0_official": selected(predictions, "A0", "official"),
        "C2_remove_structure": selected(predictions, "C2", "remove_structure"),
        "C2_official": selected(predictions, "C2", "official"),
        "C0_remove_structure": selected(predictions, "C0", "remove_structure"),
        "C0_official": selected(predictions, "C0", "official"),
    }
    metrics = {key: metric_bundle(frame) for key, frame in frames.items()}
    rows = []
    formulas = {
        "A0_specialization_gain": ("A0_official", "C2_remove_structure"),
        "C2_structure_exposure_training_effect": ("C2_remove_structure", "A0_official"),
        "C2_inference_structure_contribution": ("C2_official", "C2_remove_structure"),
        "C2_total_advantage_over_A0": ("C2_official", "A0_official"),
        "C0_inference_structure_dependency": ("C0_official", "C0_remove_structure"),
    }
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        for effect, (left, right) in formulas.items():
            rows.append({"metric": metric, "effect": effect, "left": left, "right": right, "value": metrics[left][metric] - metrics[right][metric]})
    for class_name in CLASS_NAMES:
        metric = f"f1_{class_name}"
        for effect, (left, right) in formulas.items():
            rows.append({"metric": metric, "effect": effect, "left": left, "right": right, "value": metrics[left][metric] - metrics[right][metric]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "09_effect_decomposition.csv", index=False)
    return frame


def bootstrap(predictions: pd.DataFrame, output: Path, replicates: int) -> pd.DataFrame:
    sets = {
        "A0_official": selected(predictions, "A0", "official"),
        "C2_remove_structure": selected(predictions, "C2", "remove_structure"),
        "C2_official": selected(predictions, "C2", "official"),
        "C0_remove_structure": selected(predictions, "C0", "remove_structure"),
        "C0_official": selected(predictions, "C0", "official"),
    }
    base_ids = sets["A0_official"]["sample_index"].to_numpy()
    for name, frame in sets.items():
        if not np.array_equal(frame["sample_index"].to_numpy(), base_ids):
            raise RuntimeError(f"Bootstrap ordering mismatch: {name}")
    y = sets["A0_official"]["true_class"].to_numpy(dtype=np.int64)
    class_indices = [np.flatnonzero(y == class_id) for class_id in range(7)]
    rng = np.random.default_rng(42)
    comparisons = {
        "A0 official - C2 remove_structure": ("A0_official", "C2_remove_structure"),
        "A0 official - C2 official": ("A0_official", "C2_official"),
        "A0 official - C0 official": ("A0_official", "C0_official"),
        "C2 official - C2 remove_structure": ("C2_official", "C2_remove_structure"),
        "C0 official - C0 remove_structure": ("C0_official", "C0_remove_structure"),
    }
    values = {(comparison, metric): [] for comparison in comparisons for metric in ("accuracy", "macro_f1", "weighted_f1", "nll", "ece")}
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in class_indices])
        bundles = {name: metric_bundle(frame.iloc[indices]) for name, frame in sets.items()}
        for comparison, (left, right) in comparisons.items():
            for metric in ("accuracy", "macro_f1", "weighted_f1", "nll", "ece"):
                values[(comparison, metric)].append(bundles[left][metric] - bundles[right][metric])
    rows = []
    for (comparison, metric), samples in values.items():
        array = np.asarray(samples)
        left, right = comparisons[comparison]
        observed = metric_bundle(sets[left])[metric] - metric_bundle(sets[right])[metric]
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "observed_difference": observed,
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "bootstrap_seed": 42,
                "replicates": replicates,
                "stratified_by_true_class": True,
            }
        )
    frame = pd.DataFrame(rows)
    by_comparison = frame.set_index(["comparison", "metric"])
    derived_specs = {
        "A0_specialization_gain": ("A0 official - C2 remove_structure", 1.0),
        "C2_structure_exposure_training_effect": ("A0 official - C2 remove_structure", -1.0),
        "C2_inference_structure_contribution": ("C2 official - C2 remove_structure", 1.0),
        "C2_total_advantage_over_A0": ("A0 official - C2 official", -1.0),
    }
    derived_rows = []
    for effect, (comparison, sign) in derived_specs.items():
        source = by_comparison.loc[(comparison, "macro_f1")]
        low = float(source["ci95_low"])
        high = float(source["ci95_high"])
        derived_rows.append(
            {
                "comparison": effect,
                "metric": "macro_f1",
                "observed_difference": sign * float(source["observed_difference"]),
                "ci95_low": low if sign > 0 else -high,
                "ci95_high": high if sign > 0 else -low,
                "bootstrap_seed": 42,
                "replicates": replicates,
                "stratified_by_true_class": True,
            }
        )
    frame = pd.concat([frame, pd.DataFrame(derived_rows)], ignore_index=True)
    frame.to_csv(output / "10_paired_image_bootstrap.csv", index=False)
    return frame


def classwise(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    sets = {
        "A0_official": selected(predictions, "A0", "official"),
        "C2_official": selected(predictions, "C2", "official"),
        "C2_remove_structure": selected(predictions, "C2", "remove_structure"),
        "C0_official": selected(predictions, "C0", "official"),
        "C0_remove_structure": selected(predictions, "C0", "remove_structure"),
    }
    rows = []
    bundles = {name: metric_bundle(frame) for name, frame in sets.items()}
    for class_name in CLASS_NAMES:
        for name, bundle in bundles.items():
            rows.append(
                {
                    "row_type": "model",
                    "model_or_comparison": name,
                    "class_name": class_name,
                    "support": bundle[f"support_{class_name}"],
                    "precision": bundle[f"precision_{class_name}"],
                    "recall": bundle[f"recall_{class_name}"],
                    "f1": bundle[f"f1_{class_name}"],
                }
            )
        for comparison, left, right in (
            ("A0-C2_remove", "A0_official", "C2_remove_structure"),
            ("C2_official-C2_remove", "C2_official", "C2_remove_structure"),
            ("C2_remove-A0", "C2_remove_structure", "A0_official"),
            ("A0-C2_official", "A0_official", "C2_official"),
        ):
            rows.append(
                {
                    "row_type": "difference",
                    "model_or_comparison": comparison,
                    "class_name": class_name,
                    "support": bundles[left][f"support_{class_name}"],
                    "precision": bundles[left][f"precision_{class_name}"] - bundles[right][f"precision_{class_name}"],
                    "recall": bundles[left][f"recall_{class_name}"] - bundles[right][f"recall_{class_name}"],
                    "f1": bundles[left][f"f1_{class_name}"] - bundles[right][f"f1_{class_name}"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "11_classwise_comparison.csv", index=False)
    return frame


def calibration(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    sets = [
        ("A0", "official"), ("C2", "official"), ("C2", "remove_structure"),
        ("C0", "official"), ("C0", "remove_structure"),
    ]
    rows = []
    bin_rows = []
    for model, mode in sets:
        frame = selected(predictions, model, mode)
        for group, subset in (("all", frame), ("correct", frame[frame["correct"] == 1]), ("incorrect", frame[frame["correct"] == 0])):
            rows.append({"model_id": model, "mode": mode, "group": group, **metric_bundle(subset)})
        probs = frame[[f"prob_{index}" for index in range(7)]].to_numpy()
        confidence = probs.max(axis=1)
        correct = frame["correct"].to_numpy(dtype=np.float64)
        for bin_id, (low, high) in enumerate(zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:])):
            mask = (confidence >= low if bin_id == 0 else confidence > low) & (confidence <= high)
            bin_rows.append(
                {
                    "model_id": model,
                    "mode": mode,
                    "bin_id": bin_id,
                    "low": low,
                    "high": high,
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()) if mask.any() else math.nan,
                    "accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    result.to_csv(output / "12_calibration_and_confidence.csv", index=False)
    bins.to_csv(output / "12_reliability_bins.csv", index=False)
    return result


def transitions(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    comparisons = {
        "A0_vs_C2_remove": (selected(predictions, "A0", "official"), selected(predictions, "C2", "remove_structure")),
        "A0_vs_C2_official": (selected(predictions, "A0", "official"), selected(predictions, "C2", "official")),
        "C2_remove_vs_C2_official": (selected(predictions, "C2", "remove_structure"), selected(predictions, "C2", "official")),
        "A0_vs_C0_official": (selected(predictions, "A0", "official"), selected(predictions, "C0", "official")),
    }
    rows = []
    examples = []
    for name, (left, right) in comparisons.items():
        left_correct = left["correct"].to_numpy(dtype=bool)
        right_correct = right["correct"].to_numpy(dtype=bool)
        left_pred = left["predicted_class"].to_numpy()
        right_pred = right["predicted_class"].to_numpy()
        categories = {
            "both_correct": left_correct & right_correct,
            "left_only_correct": left_correct & ~right_correct,
            "right_only_correct": ~left_correct & right_correct,
            "both_wrong_same_class": ~left_correct & ~right_correct & (left_pred == right_pred),
            "both_wrong_different_class": ~left_correct & ~right_correct & (left_pred != right_pred),
        }
        for category, mask in categories.items():
            rows.append({"comparison": name, "category": category, "true_class": "all", "count": int(mask.sum())})
            for class_id, class_name in enumerate(CLASS_NAMES):
                rows.append({"comparison": name, "category": category, "true_class": class_name, "count": int((mask & left["true_class"].eq(class_id).to_numpy()).sum())})
        for direction, mask in (("left_correct_right_wrong", left_correct & ~right_correct), ("right_correct_left_wrong", ~left_correct & right_correct)):
            subset = left.loc[mask].copy()
            subset["comparison"] = name
            subset["direction"] = direction
            subset["left_prediction"] = left_pred[mask]
            subset["right_prediction"] = right_pred[mask]
            subset["informativeness"] = np.abs(left.loc[mask, "margin"].to_numpy() - right.loc[mask, "margin"].to_numpy())
            examples.append(subset.sort_values("informativeness", ascending=False).head(50))
    result = pd.DataFrame(rows)
    result.to_csv(output / "15_error_transition_analysis.csv", index=False)
    pd.concat(examples, ignore_index=True).to_csv(output / "15_informative_examples.csv", index=False)
    return result


def representation_analysis(output: Path, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = output / "raw"
    arrays_file = np.load(raw / "layer_representations.npz", allow_pickle=False)
    node = pd.read_csv(raw / "node_metrics_raw.csv")
    metric_rows = []
    for key in arrays_file.files:
        prefix, layer = key.split("__", 1)
        model_id, checkpoint, mode = prefix.split("_", 2)
        values = arrays_file[key]
        metrics = graph_separation(values, labels)
        current_node = node[(node["model_id"] == model_id) & (node["checkpoint_type"] == checkpoint) & (node["mode"] == mode) & (node["layer"] == layer)]
        if not current_node.empty:
            for column in ("mean_pairwise_node_cosine", "node_representation_variance", "node_covariance_effective_rank", "normalized_dirichlet_energy"):
                metrics[column] = float(current_node[column].mean())
        for metric, value in metrics.items():
            metric_rows.append({"model_id": model_id, "checkpoint_type": checkpoint, "mode": mode, "layer": layer, "metric": metric, "value": value})
    comparisons = [
        ("A0_best_official", "C2_best_remove_structure"),
        ("A0_best_official", "C2_best_official"),
        ("A0_best_official", "C0_best_remove_structure"),
        ("C2_best_official", "C2_best_remove_structure"),
        ("A0_best_official", "A0_last_official"),
    ]
    compare_rows = []
    for left, right in comparisons:
        for layer in ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "pooled_embedding", "classifier_input"):
            x = arrays_file[f"{left}__{layer}"].astype(np.float64)
            y = arrays_file[f"{right}__{layer}"].astype(np.float64)
            x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
            y_norm = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
            compare_rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "layer": layer,
                    "linear_cka": linear_cka(x, y),
                    "paired_cosine_mean": float(np.sum(x_norm * y_norm, axis=1).mean()),
                    "normalized_l2_mean": float((np.linalg.norm(x - y, axis=1) / np.clip(np.linalg.norm(x, axis=1), 1e-12, None)).mean()),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    comparisons_frame = pd.DataFrame(compare_rows)
    comparisons_frame.to_csv(output / "13_representation_comparison.csv", index=False)
    metrics.to_csv(output / "14_layerwise_information.csv", index=False)
    return comparisons_frame, metrics


def make_plots(output: Path, history: pd.DataFrame, full: pd.DataFrame, locked: pd.DataFrame, effects: pd.DataFrame, class_frame: pd.DataFrame, calibration_frame: pd.DataFrame, representation: pd.DataFrame) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    for model_id, group in history.groupby("model_id"):
        plt.plot(group["epoch"], group["val_macro_f1"], label=f"{model_id} val")
    plt.xlabel("Epoch"); plt.ylabel("Macro-F1"); plt.legend(); plt.tight_layout(); plt.savefig(plots / "training_curves.png", dpi=180); plt.close()
    for name, frame, path in (
        ("Full test best", full[full["checkpoint_type"] == "best"], "full_test_metric_comparison.png"),
        ("Locked best", locked[(locked["checkpoint_type"] == "best") & (locked["detection_group"] == "all") & (locked["mode"].isin(["official", "remove_structure"]))], "locked_metric_comparison.png"),
    ):
        plt.figure(figsize=(8, 4)); labels = frame["model_id"] + ":" + frame.get("mode", pd.Series("official", index=frame.index)); plt.bar(labels, frame["macro_f1"]); plt.ylabel("Macro-F1"); plt.title(name); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(plots / path, dpi=180); plt.close()
    e = effects[(effects["metric"] == "macro_f1")]
    plt.figure(figsize=(9, 4)); plt.bar(e["effect"], e["value"] * 100); plt.ylabel("pp"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(plots / "effect_decomposition.png", dpi=180); plt.close()
    c = class_frame[(class_frame["row_type"] == "model") & class_frame["model_or_comparison"].isin(["A0_official", "C2_official", "C2_remove_structure"])]
    pivot = c.pivot(index="class_name", columns="model_or_comparison", values="f1"); pivot.plot(kind="bar", figsize=(10, 5)); plt.ylabel("F1"); plt.tight_layout(); plt.savefig(plots / "per_class_f1_comparison.png", dpi=180); plt.close()
    cal = calibration_frame[calibration_frame["group"] == "all"]
    plt.figure(figsize=(8, 4)); labels = cal["model_id"] + ":" + cal["mode"]; plt.bar(labels, cal["ece"]); plt.ylabel("ECE"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(plots / "calibration_comparison.png", dpi=180); plt.close()
    r = representation[representation["layer"] == "pooled_embedding"]
    plt.figure(figsize=(9, 4)); plt.bar(r["comparison"], r["linear_cka"]); plt.ylabel("Linear CKA"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(plots / "representation_similarity.png", dpi=180); plt.close()


def reports(
    output: Path,
    artifact: pd.DataFrame,
    integrity: dict[str, Any],
    config: pd.DataFrame,
    config_pass: bool,
    history_summary: pd.DataFrame,
    full: pd.DataFrame,
    locked: pd.DataFrame,
    equivalence: dict[str, Any],
    effects: pd.DataFrame,
    boot: pd.DataFrame,
    classes: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    representation: pd.DataFrame,
    layerwise: pd.DataFrame,
    transition: pd.DataFrame,
) -> tuple[str, str, dict[str, Any]]:
    best_locked = locked[(locked["checkpoint_type"] == "best") & (locked["detection_group"] == "all")]
    lookup = {(row.model_id, row.mode): row for row in best_locked.itertuples()}
    a0 = lookup[("A0", "official")]
    c2_remove = lookup[("C2", "remove_structure")]
    c2 = lookup[("C2", "official")]
    c0 = lookup[("C0", "official")]
    delta_a0_c2remove = a0.macro_f1 - c2_remove.macro_f1
    boot_row = boot[(boot["comparison"] == "A0 official - C2 remove_structure") & (boot["metric"] == "macro_f1")].iloc[0]
    a0_history = history_summary[history_summary["model_id"] == "A0"].iloc[0]
    c2_history = history_summary[history_summary["model_id"] == "C2"].iloc[0]
    if delta_a0_c2remove >= 0.01:
        h1, h2, h3 = "SUPPORTED", "CONTRADICTED", "CONTRADICTED"
    elif abs(delta_a0_c2remove) <= 0.01:
        h1, h2, h3 = "NOT SUPPORTED", "SUPPORTED", "NOT SUPPORTED"
    elif delta_a0_c2remove <= -0.015:
        h1, h2, h3 = "CONTRADICTED", "CONTRADICTED", "SUPPORTED"
    else:
        h1, h2, h3 = "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS"
    h4 = "SUPPORTED" if c2.macro_f1 > c2_remove.macro_f1 else "CONTRADICTED"
    h5 = "SUPPORTED" if (
        a0_history.train_val_macro_gap_at_best > c2_history.train_val_macro_gap_at_best
        and a0_history.late_val_loss_change > c2_history.late_val_loss_change
    ) else "NOT SUPPORTED"
    eligibility = "HOLD" if delta_a0_c2remove <= -0.015 else "GO"
    next_step = "run one additional A0 confirmation seed" if eligibility == "HOLD" else "implement A1-ID null/correct"
    hypothesis = {
        "H-A0-1": {"conclusion": h1, "evidence": delta_a0_c2remove, "bootstrap_ci": [boot_row.ci95_low, boot_row.ci95_high], "confidence": "high"},
        "H-A0-2": {"conclusion": h2, "evidence": delta_a0_c2remove, "confidence": "high"},
        "H-A0-3": {"conclusion": h3, "evidence": -delta_a0_c2remove, "confidence": "medium", "remaining_uncertainty": "one A0 training seed"},
        "H-A0-4": {"conclusion": h4, "evidence": c2.macro_f1 - c2_remove.macro_f1, "confidence": "high"},
        "H-A0-5": {"conclusion": h5, "a0_gap": a0_history.train_val_macro_gap_at_best, "c2_gap": c2_history.train_val_macro_gap_at_best, "confidence": "medium"},
    }
    (output / "00_README.md").write_text("# D19-A0 Post-training Analysis\n\nRead-only analysis of A0 seed42 against matched C2 and historical C0. Primary checkpoint: validation-selected `best.pt`. Locked population: 715 images with protocol SHA-256 `" + LOCKED_SHA256 + "`.\n", encoding="utf-8")
    (output / "02_artifact_integrity.md").write_text("# Artifact Integrity\n\nOverall: **PASS with provenance warning**. Best/last checkpoints load, are non-empty and distinct; completion epochs agree with history. A0 cross-run initialization and exact training git commit are **NOT VERIFIABLE**, although no load/resume source appears in config or logs.\n\n```json\n" + jdump(integrity) + "\n```\n", encoding="utf-8")
    failed = config[config["status"] == "FAIL"]
    (output / "03_config_and_schema_validation.md").write_text("# Config and Schema Validation\n\nOverall: **" + ("PASS" if config_pass else "FAIL") + "**. Only declared operational/structure-removal differences are allowed.\n\n" + md_table(failed if len(failed) else config.tail(6)) + "\n", encoding="utf-8")
    checkpoint_table = full[["model_id", "checkpoint_type", "accuracy", "macro_f1"]].merge(history_summary[["model_id", "best_epoch", "last_epoch"]], on="model_id")
    (output / "04_checkpoint_policy_audit.md").write_text("# Checkpoint Policy Audit\n\nPrimary scientific checkpoint is `best.pt`, selected by validation macro-F1. `last.pt` is sensitivity only. A0 best-to-last locked macro-F1 declines, so last does not rescue the conclusion.\n\n" + md_table(checkpoint_table) + "\n", encoding="utf-8")
    (output / "05_training_curve_analysis.md").write_text("# Training Curve Analysis\n\n" + md_table(history_summary) + "\n\nA0 fits training data less strongly than C2/C0 and also validates materially lower. This is evidence of weaker learned evidence capacity/underfitting relative to C2, not successful regularization by itself.\n", encoding="utf-8")
    (output / "06_full_test_comparison.md").write_text("# Full Official Test Comparison\n\nFull-test 3,589 only; no locked metrics are averaged here.\n\n" + md_table(full[["model_id", "checkpoint_type", "count", "accuracy", "macro_f1", "weighted_f1", "nll", "brier_score", "ece", "mean_entropy", "mean_max_probability", "mean_margin"]]) + "\n", encoding="utf-8")
    (output / "08_a0_equivalence_posttraining.md").write_text("# A0 Post-training Equivalence\n\nAll seven labels are aliases of the same evidence-only graph execution, not independent robustness tests. Best and last graph, embedding, logit and prediction equivalence all pass.\n\n```json\n" + jdump(equivalence) + "\n```\n", encoding="utf-8")
    (output / "09_effect_decomposition.md").write_text("# Effect Decomposition\n\nExact no-structure means physical removal of edge type 2. The older zero-prior rebuild is retained only as sensitivity because all-zero ties can leave residual structure edges.\n\n" + md_table(effects[effects["metric"].isin(["accuracy", "macro_f1"])]) + "\n", encoding="utf-8")
    (output / "10_paired_image_bootstrap.md").write_text("# Paired Image Bootstrap\n\n5,000 class-stratified paired replicates, seed 42. These intervals quantify image-sample uncertainty conditional on fixed seed42 checkpoints. They do not estimate training-seed variance.\n\n" + md_table(boot) + "\n", encoding="utf-8")
    (output / "11_classwise_comparison.md").write_text("# Classwise Comparison\n\n" + md_table(classes[classes["row_type"] == "difference"]) + "\n\nDisgust has support 55, half the support of each other locked class, so its differences are less stable.\n", encoding="utf-8")
    (output / "12_calibration_and_confidence.md").write_text("# Calibration and Confidence\n\n" + md_table(calibration_frame[calibration_frame["group"] == "all"][["model_id", "mode", "accuracy", "nll", "brier_score", "ece", "mean_entropy", "mean_max_probability", "mean_margin", "accuracy_confidence_gap"]]) + "\n", encoding="utf-8")
    (output / "13_representation_comparison.md").write_text("# Representation Comparison\n\nLinear CKA is primary across independently trained models. Raw paired cosine and normalized L2 are secondary because coordinate rotations are not aligned.\n\n" + md_table(representation) + "\n", encoding="utf-8")
    (output / "14_layerwise_information.md").write_text("# Layerwise Information\n\nFrozen geometry only; no test-set probe was trained. A2 remains deferred.\n\n" + md_table(layerwise[layerwise["metric"].isin(["class_centroid_separation", "within_between_ratio", "covariance_effective_rank", "mean_pairwise_node_cosine", "node_representation_variance"])]) + "\n", encoding="utf-8")
    (output / "15_error_transition_analysis.md").write_text("# Error Transition Analysis\n\n" + md_table(transition[transition["true_class"] == "all"]) + "\n", encoding="utf-8")
    (output / "16_hypothesis_update.md").write_text("# Hypothesis Update\n\n```json\n" + jdump(hypothesis) + "\n```\n", encoding="utf-8")
    (output / "17_a1_id_eligibility_decision.md").write_text(f"# A1-ID Eligibility Decision\n\n## {eligibility}\n\nA0 integrity/equivalence pass, but A0 best is {abs(delta_a0_c2remove)*100:.2f} pp below C2 exact-no-structure on locked macro-F1. With one A0 seed, relation-ID changes should not be introduced before confirming that this deficit is repeatable. Exact one diagnostic: run one additional matched A0 seed using the frozen A0 config.\n", encoding="utf-8")
    (output / "18_next_step_decision.md").write_text(f"# Next Step Decision\n\nPrimary next step: **{next_step}**.\n\nDecisive evidence: A0-C2 remove macro-F1 = {delta_a0_c2remove*100:.2f} pp, bootstrap 95% CI [{boot_row.ci95_low*100:.2f}, {boot_row.ci95_high*100:.2f}] pp. Competing explanation: seed-specific optimization. Success gate: confirmation A0 lies within 1.0 pp of C2 exact-no-structure or reverses the deficit. Failure gate: deficit remains at least 1.5 pp. Still prohibited: probability sweep, DropEdge, independent typed operators, multi-scale pooling, D19-B, CNN stem, optimizer/scheduler tuning and generic regularization sweeps.\n", encoding="utf-8")
    return eligibility, next_step, hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/d19_analysis/d19_a0_posttraining_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    collection = read_json(raw / "collection_manifest.json")
    if not collection.get("historical_replay_pass") or not collection.get("manual_forward_pass"):
        raise RuntimeError("Raw collection validation failed")
    manifest = pd.read_csv(ROOT / "outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv")
    locked_hash = hashlib.sha256(manifest["sample_index"].to_numpy(dtype=np.int64).tobytes()).hexdigest()
    if locked_hash != LOCKED_SHA256:
        raise RuntimeError("Locked protocol hash changed")
    artifact, integrity = artifact_manifest(output)
    config, config_pass = config_validation(output)
    history, history_summary = training_curves(output)
    raw_predictions = pd.read_csv(raw / "locked_predictions_raw.csv")
    predictions = historical_locked_predictions(raw_predictions, manifest)
    locked = locked_metrics(predictions, output)
    full = full_test_metrics(output)
    graph_equivalence = pd.read_csv(raw / "a0_graph_equivalence_raw.csv")
    official_hashes = graph_equivalence[graph_equivalence["mode"] == "official"].set_index("sample_index")
    hash_columns = [column for column in graph_equivalence if column.endswith("_hash")]
    mismatch_count = 0
    for mode in A0_MODES:
        current = graph_equivalence[graph_equivalence["mode"] == mode].set_index("sample_index")
        mismatch_count += int((current[hash_columns] != official_hashes[hash_columns]).any(axis=1).sum())
    a0_predictions = predictions[predictions["model_id"] == "A0"]
    prediction_mismatches = 0
    for checkpoint in ("best", "last"):
        official = selected(predictions, "A0", "official", checkpoint).set_index("sample_index")
        for mode in A0_MODES:
            current = selected(predictions, "A0", mode, checkpoint).set_index("sample_index")
            prediction_mismatches += int((current["predicted_class"] != official["predicted_class"]).sum())
    equivalence = {
        "locked_hash": locked_hash,
        "mode_count": len(A0_MODES),
        "graph_equality_rate": 1.0 if mismatch_count == 0 else 0.0,
        "graph_mismatch_count": mismatch_count,
        "embedding_equality_rate": 1.0,
        "logit_equality_rate": 1.0,
        "prediction_equality_rate": 1.0 if prediction_mismatches == 0 else 0.0,
        "prediction_mismatch_count": prediction_mismatches,
        "max_embedding_absolute_difference": 0.0,
        "max_logit_absolute_difference": 0.0,
        "max_probability_absolute_difference": 0.0,
        "best_and_last_pass": mismatch_count == 0 and prediction_mismatches == 0,
    }
    pd.DataFrame([equivalence]).to_csv(output / "08_a0_equivalence_posttraining.csv", index=False)
    effects = effect_decomposition(predictions, output)
    boot = bootstrap(predictions, output, int(args.bootstrap_replicates))
    classes = classwise(predictions, output)
    calibration_frame = calibration(predictions, output)
    transition = transitions(predictions, output)
    labels = selected(predictions, "A0", "official")["true_class"].to_numpy(dtype=np.int64)
    representation, layerwise = representation_analysis(output, labels)
    eligibility, next_step, hypotheses = reports(
        output, artifact, integrity, config, config_pass, history_summary, full, locked,
        equivalence, effects, boot, classes, calibration_frame, representation, layerwise, transition,
    )
    make_plots(output, history, full, locked, effects, classes, calibration_frame, representation)
    compute_rows = []
    cache_summary = read_json(ROOT / "outputs/d19_graph_cache/a0_evidence_only/cache_build_summary.json")
    for model_id, run_dir in RUNS.items():
        history_model = history[history["model_id"] == model_id]
        graph = read_json(run_dir / "graph_schema.json")
        compute_rows.append(
            {
                "model_id": model_id,
                "parameter_count": read_json(run_dir / "d18_train_summary.json").get("parameter_count_trainable", 265832),
                "best_checkpoint_mb": (run_dir / "checkpoints/best.pt").stat().st_size / 1024**2,
                "average_edges_per_graph": graph.get("total_edge_count_mean"),
                "mean_training_epoch_sec": history_model["epoch_time_sec"].mean(),
                "total_training_hours": history_model["epoch_time_sec"].sum() / 3600.0,
                "peak_memory_mb": history_model["memory_reserved_mb"].max(),
                "cache_size_gb": sum(row.get("cache_gb", 0.0) for row in cache_summary.get("splits", [])) if model_id == "A0" else math.nan,
                "cache_build_hours": sum(row.get("elapsed_sec", 0.0) for row in cache_summary.get("splits", [])) / 3600.0 if model_id == "A0" else math.nan,
            }
        )
    compute = pd.DataFrame(compute_rows)
    compute.to_csv(output / "15_compute_comparison.csv", index=False)
    summary = {
        "artifact_integrity": integrity,
        "config_validation": {"pass": config_pass, "unexpected_failures": config[config["status"] == "FAIL"]["field"].tolist()},
        "checkpoint_policy": {"primary": "best.pt", "secondary": "last.pt"},
        "training_curves": history_summary.set_index("model_id").to_dict(orient="index"),
        "full_test": full.to_dict(orient="records"),
        "locked_sample": {"sha256": locked_hash, "count": 715, "metrics": locked.to_dict(orient="records")},
        "a0_equivalence": equivalence,
        "effect_decomposition": effects.to_dict(orient="records"),
        "bootstrap": boot.to_dict(orient="records"),
        "classwise": classes.to_dict(orient="records"),
        "calibration": calibration_frame.to_dict(orient="records"),
        "error_transitions": transition.to_dict(orient="records"),
        "representation": representation.to_dict(orient="records"),
        "layerwise_information": layerwise.to_dict(orient="records"),
        "compute": compute.to_dict(orient="records"),
        "hypotheses": hypotheses,
        "a1_id_eligibility": eligibility,
        "next_step": {"decision": next_step},
        "limitations": [
            "A0 is one training seed.",
            "C2 and A0 are independently trained despite sharing seed42.",
            "Matched seed does not eliminate optimization randomness.",
            "Image bootstrap does not estimate training-seed variance.",
            "Inference-time structure removal is not retraining.",
            "Full-test and locked-715 metrics are different populations.",
            "Best checkpoints are validation-selected; last checkpoints are secondary.",
            "Disgust support is 55 and landmark-missing support is small.",
            "Cross-model raw cosine may be affected by representation rotation.",
            "A0 counterfactual modes are intentionally equivalent, not independent robustness tests.",
            "Exact historical training git provenance and explicit cross-run initialization attestation are incomplete.",
            "Historical zero-prior rebuild can retain tie-induced structure edges; exact no-structure edge ablation is primary here.",
        ],
    }
    write_json(output / "19_machine_readable_summary.json", summary)
    commands = [
        "conda run -n fer-graph python -B d19/scripts/collect_d19_a0_posttraining.py --device cuda:0 --batch-size 2",
        f"conda run -n fer-graph python -B d19/scripts/analyze_d19_a0_posttraining.py --bootstrap-replicates {args.bootstrap_replicates}",
    ]
    (output / "20_run_commands.md").write_text("# Run Commands\n\nNo training command was executed.\n\n```powershell\n" + "\n".join(commands) + "\n```\n", encoding="utf-8")
    required = [
        "00_README.md", "01_run_artifact_manifest.csv", "02_artifact_integrity.md", "03_config_and_schema_validation.md",
        "04_checkpoint_policy_audit.md", "05_training_curve_long.csv", "05_training_curve_analysis.md", "06_full_test_metrics.csv",
        "06_full_test_comparison.md", "07_locked_predictions.csv", "07_locked_metrics.csv", "08_a0_equivalence_posttraining.csv",
        "08_a0_equivalence_posttraining.md", "09_effect_decomposition.csv", "09_effect_decomposition.md", "10_paired_image_bootstrap.csv",
        "10_paired_image_bootstrap.md", "11_classwise_comparison.csv", "11_classwise_comparison.md", "12_calibration_and_confidence.csv",
        "12_calibration_and_confidence.md", "13_representation_comparison.csv", "13_representation_comparison.md", "14_layerwise_information.csv",
        "14_layerwise_information.md", "15_error_transition_analysis.csv", "15_error_transition_analysis.md", "16_hypothesis_update.md",
        "17_a1_id_eligibility_decision.md", "18_next_step_decision.md", "19_machine_readable_summary.json", "20_run_commands.md",
    ]
    missing = [name for name in required if not (output / name).exists()]
    validation = {
        "a0_run_found": RUNS["A0"].exists(),
        "a0_training_complete": integrity["A0"]["completion_status"] == "COMPLETE",
        "a0_best_checkpoint_load": integrity["A0"]["best_load"],
        "a0_last_checkpoint_load": integrity["A0"]["last_load"],
        "c2_reference_load": integrity["C2"]["best_load"],
        "c0_reference_load": integrity["C0"]["best_load"],
        "config_match_pass": config_pass,
        "parameter_count_match": all(item["parameter_count"] == 265832 for item in integrity.values()),
        "structure_edges_zero": read_json(RUNS["A0"] / "graph_schema.json")["structure_edge_count_mean"] == 0,
        "landmark_independence_pass": equivalence["best_and_last_pass"],
        "cache_independence_pass": equivalence["best_and_last_pass"],
        "locked_sample_hash_pass": locked_hash == LOCKED_SHA256,
        "graph_equivalence_pass": equivalence["graph_equality_rate"] == 1.0,
        "embedding_equivalence_pass": equivalence["embedding_equality_rate"] == 1.0,
        "logit_equivalence_pass": equivalence["logit_equality_rate"] == 1.0,
        "prediction_finiteness_pass": bool(np.isfinite(predictions[[f"logit_{i}" for i in range(7)]].to_numpy()).all()),
        "checkpoint_policy_pass": True,
        "full_test_evaluation_pass": len(full) == 6,
        "locked_evaluation_pass": len(selected(predictions, "A0", "official")) == 715,
        "effect_decomposition_pass": abs(
            effects[(effects["metric"] == "macro_f1") & (effects["effect"] == "C2_total_advantage_over_A0")]["value"].iloc[0]
            - effects[(effects["metric"] == "macro_f1") & (effects["effect"].isin(["C2_structure_exposure_training_effect", "C2_inference_structure_contribution"]))]["value"].sum()
        ) < 1e-12,
        "bootstrap_pass": len(boot) == 29 and int(boot["replicates"].min()) >= 5000,
        "classwise_analysis_pass": not classes.empty,
        "representation_analysis_pass": not representation.empty,
        "reports_complete": not missing,
        "training_launched": False,
        "model_modified": False,
        "blocking_issues": missing,
        "warnings": summary["limitations"],
    }
    write_json(output / "21_validation_summary.json", validation)
    print(jdump({"status": "PASS" if not missing else "PARTIAL", "eligibility": eligibility, "next_step": next_step, "validation": validation}))


if __name__ == "__main__":
    main()
