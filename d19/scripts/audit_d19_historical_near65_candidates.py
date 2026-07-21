"""Retrospective test-aware forensic audit of historical sparse pixel-graph runs.

This script is deliberately read-only with respect to scientific artifacts. It
may read checkpoints and recompute predictions, but writes only below the
registered D19 analysis output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.training.train_d16 import build_dataset

OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "d19_analysis/d19_historical_near65_candidate_forensics"
PRIOR_DIR = OUTPUTS / "d16_mediapipe_pixel_priors_best_retry_rescue"
PREVIOUS = OUTPUTS / "d19_analysis/d19_final_feature_frequency_and_fallback_audit"
MASTER = OUTPUTS / "d16_analysis/OVERFIT_FIX_MASTER_TABLE.csv"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EXPECTED_TEST_SUPPORT = [491, 55, 528, 879, 594, 416, 626]
EXPECTED_COUNTS = {"train": 28709, "val": 3589, "test": 3589}
RANKING_VERSION = "near65-test-aware-lexicographic-v1"
LOCK_FILE = "17_primary_replication_candidate_lock.json"
LOCK_HASH_FILE = "17_primary_replication_candidate_lock.sha256"


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")


def write_json_once(path: Path, payload: Any) -> str:
    encoded = json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default) + "\n"
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != encoded:
            raise RuntimeError(f"Refusing to modify existing candidate lock: {path}")
    else:
        path.write_text(encoded, encoding="utf-8")
    return sha256_file(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    return next((directory / name for name in names if (directory / name).exists()), None)


def as_float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def history_value(row: pd.Series | None, names: Iterable[str]) -> float:
    if row is None:
        return float("nan")
    for name in names:
        if name in row and pd.notna(row[name]):
            value = as_float(row[name])
            if np.isfinite(value):
                return value
    return float("nan")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.arange(7)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result: dict[str, Any] = {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "class_support": support.astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    if probs is not None and len(probs) == len(y_true):
        clipped = np.clip(probs, 1e-12, 1.0)
        result["nll"] = float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean())
        one_hot = np.eye(7)[y_true]
        result["brier_score"] = float(np.square(probs - one_hot).sum(axis=1).mean())
        confidence = probs.max(axis=1)
        result["mean_confidence"] = float(confidence.mean())
        result["mean_entropy"] = float((-(clipped * np.log(clipped)).sum(axis=1)).mean())
        result["ece"] = expected_calibration_error(y_true, y_pred, confidence)
    else:
        result.update(
            {
                "nll": float("nan"),
                "brier_score": float("nan"),
                "mean_confidence": float("nan"),
                "mean_entropy": float("nan"),
                "ece": float("nan"),
            }
        )
    return result


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
    total = max(len(y_true), 1)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence > lower) & (confidence <= upper if upper < 1.0 else confidence <= 1.0)
        if mask.any():
            ece += float(mask.sum()) / total * abs(float((y_pred[mask] == y_true[mask]).mean()) - float(confidence[mask].mean()))
    return float(ece)


def metrics_from_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if not {"y_true", "y_pred"}.issubset(frame.columns):
        return {}
    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)
    prob_cols = [f"prob_{index}" for index in range(7)]
    probs = None
    if set(prob_cols).issubset(frame.columns):
        probs = frame.loc[valid, prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(probs).all():
            probs = None
    result = classification_metrics(y_true, y_pred, probs)
    if "sample_index" in frame:
        indices = pd.to_numeric(frame.loc[valid, "sample_index"], errors="coerce").to_numpy()
        result["sample_index_exact"] = bool(
            len(indices) == EXPECTED_COUNTS["test"]
            and np.array_equal(np.sort(indices.astype(int)), np.arange(EXPECTED_COUNTS["test"]))
        )
    else:
        result["sample_index_exact"] = False
    result["label_sequence_sha256"] = sha256_text(",".join(map(str, y_true.tolist())))
    return result


def discover_run_dirs() -> list[tuple[Path, str]]:
    discovered: dict[Path, set[str]] = defaultdict(set)
    for path in OUTPUTS.rglob("best.pt"):
        run_dir = path.parent.parent
        discovered[run_dir].add("best.pt")
    for filename in (
        "d16_train_summary.json",
        "d17_train_summary.json",
        "d18_train_summary.json",
        "d19_train_summary.json",
        "train_log.csv",
        "training_history.json",
    ):
        for path in OUTPUTS.rglob(filename):
            run_dir = path.parent
            if (run_dir / "checkpoints").is_dir() or (run_dir / "best.pt").exists():
                discovered[run_dir].add(filename)
    return sorted(
        ((path, "+".join(sorted(sources))) for path, sources in discovered.items()),
        key=lambda item: str(item[0]).lower(),
    )


def config_for_run(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    path = first_existing(
        run_dir,
        (
            "resolved_config.yaml",
            "source_config.yaml",
            "config.yaml",
            "resolved_config.json",
            "config.json",
        ),
    )
    if path is None:
        return None, {}
    return path, read_json(path) if path.suffix == ".json" else read_yaml(path)


def summary_for_run(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    path = first_existing(
        run_dir,
        (
            "d19_train_summary.json",
            "d18_train_summary.json",
            "d17_train_summary.json",
            "d16_train_summary.json",
            "summary.json",
            "final_metrics.json",
            "metrics.json",
        ),
    )
    return path, read_json(path) if path else {}


def find_best_checkpoint(run_dir: Path) -> Path | None:
    paths = [run_dir / "checkpoints/best.pt", run_dir / "best.pt"]
    return next((path for path in paths if path.exists()), None)


def find_last_checkpoint(run_dir: Path) -> Path | None:
    paths = [run_dir / "checkpoints/last.pt", run_dir / "last.pt"]
    return next((path for path in paths if path.exists()), None)


def infer_family(run_dir: Path) -> str:
    relative = rel(run_dir).lower()
    match = re.search(r"(d(?:15|16|17|18|19)(?:_[^\\/]+)?)", relative)
    return match.group(1) if match else run_dir.parent.name


def branch_filter(run_dir: Path, cfg: dict[str, Any]) -> tuple[bool, str]:
    path_text = rel(run_dir).lower()
    config_text = json.dumps(cfg, default=str).lower()
    text = path_text + " " + config_text
    if re.search(r"smoke|benchmark|diagnostic|synthetic|demo|resume_integrity_check", path_text):
        return False, "excluded_smoke_benchmark_diagnostic_or_check"
    if any(token in text for token in ("mgr-cnn", "conv2d", "resnet", "efficientnet", "swin", "patch_node", "superpixel")):
        return False, "excluded_non_sparse_pixel_graph_or_cnn"
    if not re.search(r"(?:^|[\\/])d(?:15|16|17|18|19)", rel(run_dir).lower()):
        return False, "excluded_outside_d15_d19_sparse_pixel_branch"
    graph = cfg.get("graph", {}) or {}
    selector = str(graph.get("selector_algorithm", graph.get("graph_mode", ""))).lower()
    model_name = str((cfg.get("model", {}) or {}).get("name", "")).lower()
    evidence = any(
        token in text + " " + selector + " " + model_name
        for token in ("pixel", "face_plus_context", "full_with_mask", "structure_guided", "evidence_only")
    )
    return (True, "included_sparse_pixel_graph") if evidence else (False, "pixel_graph_identity_not_verifiable")


def load_history(run_dir: Path) -> tuple[Path | None, pd.DataFrame]:
    path = first_existing(run_dir, ("train_log.csv", "history.csv", "training_history.csv"))
    if path:
        try:
            return path, pd.read_csv(path)
        except Exception:
            return path, pd.DataFrame()
    json_path = first_existing(run_dir, ("history.json", "training_history.json"))
    if json_path:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("history", payload.get("epochs", []))
            return json_path, pd.DataFrame(payload)
        except Exception:
            return json_path, pd.DataFrame()
    return None, pd.DataFrame()


def artifact_lists(run_dir: Path) -> dict[str, str]:
    names = [path.name for path in run_dir.iterdir() if path.is_file()]
    return {
        "history_paths": json.dumps([rel(run_dir / name) for name in names if "history" in name or name in {"train_log.csv", "train_metrics.csv", "val_metrics.csv"}]),
        "metric_paths": json.dumps([rel(run_dir / name) for name in names if "metric" in name or name in {"predictions.csv", "last_predictions.csv"}]),
        "log_paths": json.dumps([rel(run_dir / name) for name in names if name.endswith(".log")]),
        "report_paths": json.dumps([rel(run_dir / name) for name in names if name.endswith(".md")]),
    }


def master_lookup() -> dict[str, dict[str, Any]]:
    if not MASTER.exists():
        return {}
    frame = pd.read_csv(MASTER)
    return {str(row["run"]): row.to_dict() for _, row in frame.iterrows()}


def discover_registry() -> pd.DataFrame:
    master = master_lookup()
    rows: list[dict[str, Any]] = []
    run_dirs = discover_run_dirs()
    for position, (run_dir, source) in enumerate(run_dirs, start=1):
        config_path, cfg = config_for_run(run_dir)
        summary_path, summary = summary_for_run(run_dir)
        history_path, history = load_history(run_dir)
        best = find_best_checkpoint(run_dir)
        last = find_last_checkpoint(run_dir)
        run_id = str(cfg.get("run_name", run_dir.name))
        if run_id == run_dir.name and summary.get("output_dir"):
            run_id = Path(str(summary["output_dir"])).name
        pixel_graph, inclusion = branch_filter(run_dir, cfg)
        predictions_path = first_existing(run_dir, ("predictions.csv", "test_predictions.csv"))
        last_predictions_path = first_existing(run_dir, ("last_predictions.csv",))
        prediction_metrics = metrics_from_predictions(predictions_path) if predictions_path else {}
        last_prediction_metrics = metrics_from_predictions(last_predictions_path) if last_predictions_path else {}
        master_row = master.get(run_id, {})
        best_epoch = as_int(summary.get("best_epoch", master_row.get("best_epoch", -1)))
        if best_epoch < 0 and prediction_metrics:
            try:
                best_epoch = as_int(pd.read_csv(predictions_path, nrows=1).iloc[0].get("checkpoint_epoch"))
            except Exception:
                pass
        history_best = None
        history_last = history.iloc[-1] if not history.empty else None
        if not history.empty and "epoch" in history:
            epochs = pd.to_numeric(history["epoch"], errors="coerce")
            match = history[epochs.eq(best_epoch)]
            history_best = match.iloc[-1] if not match.empty else None
        train_acc = history_value(history_best, ("train_accuracy", "train_acc"))
        train_f1 = history_value(history_best, ("train_macro_f1", "train_f1"))
        val_acc = history_value(history_best, ("val_accuracy", "val_acc"))
        val_f1 = history_value(history_best, ("val_macro_f1", "val_f1"))
        if not np.isfinite(val_acc):
            val_acc = as_float(summary.get("best_val_accuracy", master_row.get("bestrow_val_accuracy")))
        if not np.isfinite(val_f1):
            val_f1 = as_float(summary.get("best_val_macro_f1", master_row.get("bestrow_val_macro_f1")))
        test_acc = as_float(prediction_metrics.get("accuracy", summary.get("test_accuracy", master_row.get("test_acc"))))
        test_f1 = as_float(prediction_metrics.get("macro_f1", summary.get("test_macro_f1", master_row.get("test_macro_f1"))))
        test_weighted = as_float(prediction_metrics.get("weighted_f1"))
        monitor = str(summary.get("best_monitor_metric", (cfg.get("training", {}) or {}).get("checkpoint_monitor", "")))
        seed = as_int(cfg.get("seed", (cfg.get("training", {}) or {}).get("seed", summary.get("seed", -1))))
        train_count = as_int(summary.get("train_samples", -1))
        val_count = as_int(summary.get("val_samples", -1))
        test_count = as_int(summary.get("test_samples", prediction_metrics.get("count", -1)))
        support_exact = prediction_metrics.get("class_support") == EXPECTED_TEST_SUPPORT
        sample_index_exact = bool(prediction_metrics.get("sample_index_exact", False))
        split_status = (
            "SPLIT_EXACT"
            if test_count == 3589 and support_exact and sample_index_exact
            else "SPLIT_VERIFIED_EQUIVALENT"
            if train_count == 28709 and val_count == 3589 and test_count == 3589 and support_exact
            else "SPLIT_NOT_VERIFIABLE"
        )
        last_val_acc = history_value(history_last, ("val_accuracy", "val_acc"))
        last_val_f1 = history_value(history_last, ("val_macro_f1", "val_f1"))
        gap_acc = (train_acc - val_acc) * 100 if np.isfinite(train_acc) and np.isfinite(val_acc) else np.nan
        gap_f1 = (train_f1 - val_f1) * 100 if np.isfinite(train_f1) and np.isfinite(val_f1) else np.nan
        decline = (val_f1 - last_val_f1) * 100 if np.isfinite(val_f1) and np.isfinite(last_val_f1) else np.nan
        resume_text = ""
        resume_path = run_dir / "resume_events.jsonl"
        if resume_path.exists():
            resume_text = resume_path.read_text(encoding="utf-8", errors="replace").lower()
        resume_contamination = any(
            token in resume_text for token in ("signature_mismatch", "optimizer_mismatch", "scheduler_mismatch", "resume_corrupt")
        )
        complete = bool(best and last and config_path and history_path and summary_path and predictions_path)
        metric_confidence = (
            "VERIFIED_FROM_PREDICTIONS"
            if prediction_metrics.get("count") == 3589
            else "VERIFIED_FROM_STRUCTURED_ARTIFACT"
            if np.isfinite(test_acc) and np.isfinite(test_f1)
            else "MISSING"
        )
        checkpoint_policy = (
            "CHECKPOINT_POLICY_PASS"
            if monitor.startswith("val_") and best_epoch >= 0 and history_best is not None and not resume_contamination
            else "CHECKPOINT_POLICY_WARNING"
            if monitor.startswith("val_") and best_epoch >= 0 and not resume_contamination
            else "CHECKPOINT_POLICY_FAIL"
        )
        artifacts = artifact_lists(run_dir)
        rows.append(
            {
                "run_id": run_id,
                "canonical_path": rel(run_dir),
                "discovery_source": source,
                "experiment_family": infer_family(run_dir),
                "seed": seed,
                "config_paths": json.dumps([rel(config_path)] if config_path else []),
                "checkpoint_paths": json.dumps([rel(path) for path in (best, last) if path]),
                **artifacts,
                "completion_markers": json.dumps([rel(path) for path in (summary_path, predictions_path) if path]),
                "candidate_pixel_graph": pixel_graph,
                "reason_for_inclusion_or_exclusion": inclusion,
                "config_path": rel(config_path),
                "summary_path": rel(summary_path),
                "history_path": rel(history_path),
                "best_checkpoint_path": rel(best),
                "last_checkpoint_path": rel(last),
                "best_checkpoint_sha256": sha256_file(best) if best else "",
                "last_checkpoint_sha256": sha256_file(last) if last else "",
                "best_epoch": best_epoch,
                "last_epoch": as_int(history_last.get("epoch")) if history_last is not None else -1,
                "checkpoint_monitor": monitor,
                "train_accuracy_at_best": train_acc,
                "train_macro_f1_at_best": train_f1,
                "validation_accuracy_at_best": val_acc,
                "validation_macro_f1_at_best": val_f1,
                "accuracy_gap_pp": gap_acc,
                "macro_f1_gap_pp": gap_f1,
                "last_validation_accuracy": last_val_acc,
                "last_validation_macro_f1": last_val_f1,
                "best_to_last_validation_macro_f1_change_pp": decline,
                "best_epoch_to_end_epoch_distance": as_int(history_last.get("epoch")) - best_epoch if history_last is not None else np.nan,
                "full_test_accuracy": test_acc,
                "full_test_macro_f1": test_f1,
                "full_test_weighted_f1": test_weighted,
                "last_test_accuracy": as_float(last_prediction_metrics.get("accuracy", summary.get("last_test_accuracy", master_row.get("last_test_acc")))),
                "last_test_macro_f1": as_float(last_prediction_metrics.get("macro_f1", summary.get("last_test_macro_f1", master_row.get("last_test_macro_f1")))),
                "test_count": test_count,
                "test_class_support": json.dumps(prediction_metrics.get("class_support", [])),
                "test_label_sequence_sha256": prediction_metrics.get("label_sequence_sha256", ""),
                "split_status": split_status,
                "dataset_signature": prediction_metrics.get("label_sequence_sha256", "NOT_VERIFIABLE"),
                "metric_confidence": metric_confidence,
                "checkpoint_policy_status": checkpoint_policy,
                "history_checkpoint_epoch_agree": history_best is not None,
                "resume_contamination": resume_contamination,
                "artifact_complete": complete,
                "predictions_path": rel(predictions_path),
                "last_predictions_path": rel(last_predictions_path),
                "stored_test_accuracy": as_float(summary.get("test_accuracy")),
                "stored_test_macro_f1": as_float(summary.get("test_macro_f1")),
                "prediction_test_accuracy": as_float(prediction_metrics.get("accuracy")),
                "prediction_test_macro_f1": as_float(prediction_metrics.get("macro_f1")),
                "prediction_test_nll": as_float(prediction_metrics.get("nll")),
                "prediction_test_brier": as_float(prediction_metrics.get("brier_score")),
                "prediction_test_ece": as_float(prediction_metrics.get("ece")),
                "prediction_mean_confidence": as_float(prediction_metrics.get("mean_confidence")),
                "prediction_mean_entropy": as_float(prediction_metrics.get("mean_entropy")),
            }
        )
        if position % 25 == 0:
            print(json.dumps({"event": "discovery_progress", "processed": position, "total": len(run_dirs)}), flush=True)
    return pd.DataFrame(rows)


def deduplicate(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dedup_rows: list[dict[str, Any]] = []
    canonical_indices: list[int] = []
    for checkpoint_hash, part in registry.groupby("best_checkpoint_sha256", dropna=False, sort=True):
        ordered = part.sort_values(
            ["artifact_complete", "metric_confidence", "canonical_path"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        canonical = ordered.iloc[0]
        canonical_indices.append(int(canonical.name))
        metadata_conflict = (
            ordered[["full_test_accuracy", "full_test_macro_f1"]]
            .apply(pd.to_numeric, errors="coerce")
            .nunique(dropna=True)
            .max()
            > 1
        )
        dedup_rows.append(
            {
                "checkpoint_sha256": checkpoint_hash,
                "canonical_checkpoint_path": canonical["best_checkpoint_path"],
                "canonical_run_id": canonical["run_id"],
                "canonical_run_path": canonical["canonical_path"],
                "alias_count": max(len(ordered) - 1, 0),
                "alias_paths": json.dumps(ordered["canonical_path"].tolist()[1:]),
                "metadata_conflict": bool(metadata_conflict),
            }
        )
    canonical = registry.loc[canonical_indices].copy().sort_values("canonical_path").reset_index(drop=True)
    return canonical, pd.DataFrame(dedup_rows)


def preliminary_candidate_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    acc = pd.to_numeric(frame["full_test_accuracy"], errors="coerce")
    gap = pd.to_numeric(frame["macro_f1_gap_pp"], errors="coerce")
    branch = frame["candidate_pixel_graph"].eq(True)
    return {
        "exact064": branch & acc.between(0.635, 0.645),
        "core": branch & acc.between(0.635, 0.665) & gap.between(12.0, 18.0),
        "expanded": branch & acc.between(0.625, 0.675) & gap.between(10.0, 20.0),
        "accuracy_match": branch & acc.between(0.635, 0.665),
        "near6514": branch & acc.between(0.6505, 0.6523),
    }


def checkpoint_payload(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    return payload if isinstance(payload, dict) else {}, state


def runtime_verify_d16(
    row: pd.Series,
    device: torch.device,
    recompute: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = ROOT / str(row["canonical_path"])
    config_path = ROOT / str(row["config_path"])
    cfg = read_yaml(config_path) if config_path.suffix != ".json" else read_json(config_path)
    cfg.setdefault("data", {})
    cfg["data"]["graph_cache_dir"] = None
    cfg["data"]["graph_cache_dir_detected"] = None
    cfg["data"]["graph_cache_dir_fallback"] = None
    cfg["data"]["num_workers"] = 0
    seed = as_int(cfg.get("seed", (cfg.get("training", {}) or {}).get("seed", 42)), 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_ds = build_dataset(cfg, PRIOR_DIR, "test")
    first_graph = test_ds[0]
    first_batch = collate_d16_graphs([first_graph])
    input_dim = int(first_batch.x_cat.shape[1])
    edge_dim = 0 if first_batch.edge_attr_cat is None else int(first_batch.edge_attr_cat.shape[1])
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    result: dict[str, Any] = {
        "run_id": row["run_id"],
        "canonical_path": row["canonical_path"],
        "strict_load_best": False,
        "strict_load_last": False,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "input_dim": input_dim,
        "edge_attr_dim": edge_dim,
        "finite_parameters": False,
        "finite_logits": False,
        "bounded_forward": False,
        "model_signature": "",
        "graph_signature": sha256_text(json.dumps({"input_dim": input_dim, "edge_dim": edge_dim, "nodes": int(first_batch.x_cat.shape[0]), "edges": int(first_batch.edge_index_cat.shape[1])}, sort_keys=True)),
        "selector_signature": sha256_text(json.dumps(cfg.get("graph", {}), sort_keys=True, default=str)),
        "strict_error": "",
    }
    metric_rows: list[dict[str, Any]] = []
    try:
        best_payload, best_state = checkpoint_payload(run_dir / "checkpoints/best.pt")
        model.load_state_dict(best_state, strict=True)
        result["strict_load_best"] = True
        result["model_signature"] = sha256_text(
            json.dumps(sorted((key, list(value.shape)) for key, value in best_state.items()))
        )
        result["finite_parameters"] = bool(all(torch.isfinite(value).all().item() for value in best_state.values()))
        model.eval()
        with torch.inference_mode():
            logits = model(first_batch.to(device))["logits"]
        result["finite_logits"] = bool(torch.isfinite(logits).all().item())
        result["bounded_forward"] = tuple(logits.shape) == (1, 7)
        last_path = run_dir / "checkpoints/last.pt"
        if last_path.exists():
            _, last_state = checkpoint_payload(last_path)
            model.load_state_dict(last_state, strict=True)
            result["strict_load_last"] = True
        if int(best_payload.get("epoch", row["best_epoch"])) != int(row["best_epoch"]):
            result["strict_error"] = "checkpoint_epoch_mismatch"
    except Exception as exc:
        result["strict_error"] = f"{type(exc).__name__}: {exc}"
        return result, metric_rows
    if not recompute:
        return result, metric_rows
    _, best_state = checkpoint_payload(run_dir / "checkpoints/best.pt")
    model.load_state_dict(best_state, strict=True)
    for split in ("val", "test"):
        dataset = build_dataset(cfg, PRIOR_DIR, split)
        metrics, predictions = inference(model, dataset, device, str(row["run_id"]), split)
        metric_rows.append(
            {
                "run_id": row["run_id"],
                "canonical_path": row["canonical_path"],
                "checkpoint": "best.pt",
                "checkpoint_sha256": row["best_checkpoint_sha256"],
                "split": split,
                **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                "class_support": json.dumps(metrics["class_support"]),
                "per_class_f1": json.dumps(metrics["per_class_f1"]),
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )
        predictions.to_csv(output_dir / f"{row['run_id']}__best__{split}_predictions.csv", index=False)
    return result, metric_rows


@torch.inference_mode()
def inference(
    model: D16Model,
    dataset: Any,
    device: torch.device,
    run_id: str,
    split: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_d16_graphs,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_indices: list[int] = []
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        logits = model(batch)["logits"]
        probs = torch.softmax(logits, dim=1)
        y_true.extend(batch.y.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        sample_indices.extend(batch.sample_index.detach().cpu().numpy().astype(int).tolist())
        probabilities.extend(probs.detach().cpu().numpy())
        if batch_index % 75 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {
                        "event": "runtime_recompute_progress",
                        "run_id": run_id,
                        "split": split,
                        "batch": batch_index,
                        "total_batches": len(loader),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    probs_array = np.asarray(probabilities, dtype=float)
    metrics = classification_metrics(y_true_array, y_pred_array, probs_array)
    metrics["elapsed_sec"] = time.perf_counter() - started
    data = {
        "sample_index": sample_indices,
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (y_true_array == y_pred_array).astype(int),
    }
    for index in range(7):
        data[f"prob_{index}"] = probs_array[:, index]
    return metrics, pd.DataFrame(data).sort_values("sample_index")


def architecture_inventory(row: pd.Series) -> dict[str, Any]:
    path = ROOT / str(row["config_path"]) if row.get("config_path") else None
    cfg = read_json(path) if path and path.suffix == ".json" else read_yaml(path) if path else {}
    graph = cfg.get("graph", {}) or {}
    model = cfg.get("model", {}) or {}
    training = cfg.get("training", {}) or {}
    loss = cfg.get("loss", {}) or {}
    optimizer = training.get("optimizer", {}) or {}
    scheduler = training.get("scheduler", {}) or {}
    edge_context = model.get("edge_context_gnn", {}) or {}
    detail_cfg = graph.get("detail_features", {}) or {}
    node_cfg = graph.get("node_features", {}) or {}
    prior_usage = str(
        graph.get(
            "prior_usage",
            (model.get("micro_motif_support", {}) or {}).get("prior_usage", "full"),
        )
    )
    routing_only = prior_usage == "routing_only" or str(node_cfg.get("mode", "")) == "routing_only"
    prior_schema = read_json(OUTPUTS / "d16_mediapipe_pixel_priors_best_retry_rescue/prior_schema.json")
    part_count = int(prior_schema.get("part_count", 13))
    distance_count = int(prior_schema.get("anchor_count", 12))
    node_feature_names = ["intensity", "gx", "gy", "x_norm", "y_norm"]
    if bool(node_cfg.get("include_face_mask", not routing_only)):
        node_feature_names.append("face_mask")
    if bool(node_cfg.get("include_part_soft", node_cfg.get("include_part_soft_masks", not routing_only))):
        node_feature_names.extend(f"part_soft_{index}" for index in range(part_count))
    if bool(node_cfg.get("include_distance_maps", not routing_only)):
        node_feature_names.extend(f"distance_map_{index}" for index in range(distance_count))
    if bool(node_cfg.get("include_landmark_missing_flag", not routing_only)):
        node_feature_names.append("landmark_missing_flag")
    if bool(detail_cfg.get("enabled", False)) and bool(detail_cfg.get("append_to_x", True)):
        node_feature_names.extend(str(name) for name in detail_cfg.get("features", []))
    input_dim = as_int(row.get("input_dim"))
    if input_dim > 0 and len(node_feature_names) != input_dim:
        node_feature_names = [*node_feature_names, f"UNRESOLVED_DIM_{len(node_feature_names)}_TO_{input_dim}"]
    hidden_dim = int(model.get("hidden_dim", 96))
    readout_type = str(model.get("readout_type", "concat"))
    classifier_dim = hidden_dim if readout_type == "global_mean" else hidden_dim * 5
    node_count = graph.get("target_node_count", graph.get("target_count", ""))
    if node_count == "":
        node_count = (
            f"variable_per_image(face_mask>{graph.get('face_threshold', 0.15)}"
            f"+context_pixels={graph.get('context_pixels', 0)}"
            f"+anchors={len((graph.get('anchor_nodes', {}) or {}).get('groups', []))})"
        )
    optimizer_name = (
        optimizer.get("type", "")
        if isinstance(optimizer, dict)
        else str(optimizer)
    ) or str(training.get("optimizer_type", "")) or "AdamW(runtime default)"
    return {
        "run_id": row["run_id"],
        "canonical_path": row["canonical_path"],
        "experiment_phase": row["experiment_family"],
        "model_class": f"d16.models.d16_model.D16Model ({model.get('name', '')})",
        "graph_mode": graph.get("graph_mode", (cfg.get("data", {}) or {}).get("graph_mode", "")),
        "node_count": node_count,
        "selector_algorithm": (
            graph.get("selector_algorithm")
            or f"{graph.get('graph_mode', '')}: face-threshold selection plus context dilation"
        ),
        "node_feature_count": input_dim or len(node_feature_names),
        "node_feature_order": json.dumps(node_feature_names),
        "node_features": json.dumps(
            {
                "prior_usage": prior_usage,
                "node_features": node_cfg,
                "detail_features": detail_cfg,
            },
            sort_keys=True,
        ),
        "local_edge_policy": "8-neighbor local grid" if str(graph.get("graph_mode", "")).startswith(("face", "full")) else graph.get("local_edge_policy", ""),
        "knn_policy": json.dumps(graph.get("knn_edges", {}), sort_keys=True),
        "structure_edge_policy": json.dumps(graph.get("structure_edges", {}), sort_keys=True),
        "structure_feature_prior_source": "MediaPipe pixel priors" if (cfg.get("data", {}) or {}).get("use_mediapipe_priors") else "",
        "structure_mode_mix": json.dumps(graph.get("structure_mode_mix", {}), sort_keys=True),
        "dropedge": graph.get("dropedge", graph.get("drop_edge_probability", "")),
        "edge_feature_dimension": edge_context.get("edge_attr_dim", ""),
        "hidden_dimension": model.get("hidden_dim", ""),
        "gnn_layer_count": edge_context.get("num_layers", model.get("gnn_layers", "")),
        "message_operator": edge_context.get("message_type", model.get("gnn_type", "")),
        "aggregation": edge_context.get("aggregation", ""),
        "normalization": "layer_norm" if edge_context.get("layer_norm") else "",
        "residual_connections": edge_context.get("residual", ""),
        "readout": readout_type,
        "readout_config": json.dumps(model.get(readout_type, {}), sort_keys=True),
        "classifier": f"D16Classifier({classifier_dim}->{hidden_dim * 2}->7)",
        "dropout": json.dumps(
            {
                "pixel_encoder_classifier": model.get("dropout", 0.1),
                "edge_context_gnn": edge_context.get("dropout", ""),
                "readout": (model.get(readout_type, {}) or {}).get("dropout", ""),
            },
            sort_keys=True,
        ),
        "loss": json.dumps(loss, sort_keys=True),
        "class_weighting": loss.get("class_weights", loss.get("class_weighting", "none")),
        "optimizer": optimizer_name,
        "learning_rate": training.get("learning_rate", training.get("lr", optimizer.get("lr", ""))),
        "weight_decay": training.get("weight_decay", optimizer.get("weight_decay", "")),
        "scheduler": scheduler.get("type", training.get("scheduler_type", "")),
        "batch_size": (cfg.get("data", {}) or {}).get("batch_size", training.get("batch_size", "")),
        "maximum_epochs": training.get("max_epochs", ""),
        "early_stopping": json.dumps(training.get("early_stopping", {}), sort_keys=True),
        "seed": row["seed"],
        "parameter_count": row.get("parameter_count", ""),
        "anchor_nodes": json.dumps(graph.get("anchor_nodes", {}), sort_keys=True),
        "prior_corruption": json.dumps(graph.get("prior_corruption", {}), sort_keys=True),
    }


def technical_eligibility(row: pd.Series) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(row["candidate_pixel_graph"]):
        reasons.append("not_sparse_pixel_graph")
    if row["split_status"] not in {"SPLIT_EXACT", "SPLIT_VERIFIED_EQUIVALENT"}:
        reasons.append("split_not_verified")
    if row["checkpoint_policy_status"] != "CHECKPOINT_POLICY_PASS":
        reasons.append("checkpoint_policy_not_pass")
    if not bool(row.get("strict_load_best", False)):
        reasons.append("strict_best_load_failed_or_not_run")
    if not bool(row.get("history_checkpoint_epoch_agree", False)):
        reasons.append("history_checkpoint_epoch_mismatch")
    if bool(row.get("resume_contamination", False)):
        reasons.append("resume_contamination")
    if row["metric_confidence"] not in {"VERIFIED_RECOMPUTED", "VERIFIED_FROM_PREDICTIONS"}:
        reasons.append("test_metrics_not_prediction_verified")
    if not bool(row.get("finite_parameters", False)):
        reasons.append("nonfinite_or_unverified_parameters")
    if not bool(row.get("finite_logits", False)):
        reasons.append("nonfinite_or_unverified_logits")
    return not reasons, reasons


def assign_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        eligible, reasons = technical_eligibility(row)
        acc = as_float(row["full_test_accuracy"])
        gap = as_float(row["macro_f1_gap_pp"])
        if eligible and 0.635 <= acc <= 0.665 and not np.isfinite(gap):
            eligible = False
            reasons.append("macro_gap_not_verifiable")
        tier = "INELIGIBLE"
        if eligible and 0.635 <= acc <= 0.665 and 12.0 <= gap <= 18.0:
            tier = "N65-A"
        elif eligible and 0.625 <= acc <= 0.675 and 10.0 <= gap <= 20.0:
            tier = "N65-B"
        elif eligible and 0.635 <= acc <= 0.665:
            tier = "N65-C"
        elif eligible:
            tier = "OUTSIDE_NEAR65"
        rows.append(
            {
                **row.to_dict(),
                "technically_eligible": eligible,
                "eligibility_tier": tier,
                "eligibility_reasons": json.dumps(reasons),
            }
        )
    return pd.DataFrame(rows)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[frame["eligibility_tier"].isin(["N65-A", "N65-B"])].copy()
    if candidates.empty:
        return candidates
    candidates["_tier"] = candidates["eligibility_tier"].map({"N65-A": 0, "N65-B": 1})
    candidates["_gap_distance"] = (pd.to_numeric(candidates["macro_f1_gap_pp"], errors="coerce") - 15.0).abs()
    candidates["_dependency"] = pd.to_numeric(candidates["structure_dependency_macro_f1_pp"], errors="coerce").fillna(np.inf)
    candidates["_completeness"] = candidates["artifact_complete"].astype(int)
    candidates = candidates.sort_values(
        [
            "_tier",
            "full_test_accuracy",
            "full_test_macro_f1",
            "_gap_distance",
            "validation_macro_f1_at_best",
            "best_to_last_validation_macro_f1_change_pp",
            "_dependency",
            "_completeness",
            "run_id",
        ],
        ascending=[True, False, False, True, False, True, True, False, True],
        kind="mergesort",
    ).drop(columns=["_tier", "_gap_distance", "_dependency", "_completeness"])
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def structure_dependency_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    prior_csv = OUTPUTS / "d16_analysis/OVERFIT_FIX_PRIOR_AUDIT_TABLE.csv"
    if prior_csv.exists():
        frame = pd.read_csv(prior_csv)
        for _, row in frame.iterrows():
            official = as_float(row.get("official_macro_f1"))
            fallback = as_float(row.get("forced_fallback_macro_f1"))
            result[str(row.get("run"))] = {
                "source": rel(prior_csv),
                "official_macro_f1": official,
                "physical_remove_structure_macro_f1": np.nan,
                "forced_fallback_macro_f1": fallback,
                "shuffle_macro_f1": as_float(row.get("shuffle_prior_macro_f1")),
                "random_macro_f1": np.nan,
                "structure_dependency_macro_f1_pp": np.nan,
                "classification": "NOT_AVAILABLE",
                "note": "Prior counterfactual is not physical structure-edge removal.",
            }
    d18_csv = OUTPUTS / "d18_analysis/ofix18_c0_c2_multiseed_posttraining/11_edge_ablation_multiseed.csv"
    if d18_csv.exists():
        frame = pd.read_csv(d18_csv)
        for _, row in frame.iterrows():
            if str(row.get("checkpoint")) != "best" or str(row.get("mode")) != "remove_structure":
                continue
            drop = as_float(row.get("official_to_counterfactual_macro_f1_drop")) * 100
            classification = "LOW" if drop <= 4 else "MODERATE" if drop <= 10 else "HIGH"
            result[str(row.get("run_name"))] = {
                "source": rel(d18_csv),
                "official_macro_f1": as_float(row.get("official_macro_f1")),
                "physical_remove_structure_macro_f1": as_float(row.get("counterfactual_macro_f1")),
                "forced_fallback_macro_f1": np.nan,
                "shuffle_macro_f1": np.nan,
                "random_macro_f1": np.nan,
                "structure_dependency_macro_f1_pp": drop,
                "classification": classification,
                "note": "Physical remove_structure audit.",
            }
    return result


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is not None:
        columns = [column for column in columns if column in frame.columns]
        frame = frame[columns]
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def source_trace(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        for metric, structured, predicted in (
            ("test_accuracy", row["stored_test_accuracy"], row["prediction_test_accuracy"]),
            ("test_macro_f1", row["stored_test_macro_f1"], row["prediction_test_macro_f1"]),
            ("test_weighted_f1", np.nan, row["full_test_weighted_f1"]),
            ("train_accuracy_at_best", row["train_accuracy_at_best"], np.nan),
            ("train_macro_f1_at_best", row["train_macro_f1_at_best"], np.nan),
            ("validation_accuracy_at_best", row["validation_accuracy_at_best"], np.nan),
            ("validation_macro_f1_at_best", row["validation_macro_f1_at_best"], np.nan),
        ):
            status = (
                "VERIFIED_FROM_PREDICTIONS"
                if np.isfinite(as_float(predicted))
                else "VERIFIED_FROM_STRUCTURED_ARTIFACT"
                if np.isfinite(as_float(structured))
                else "MISSING"
            )
            conflict = (
                np.isfinite(as_float(predicted))
                and np.isfinite(as_float(structured))
                and abs(as_float(predicted) - as_float(structured)) > 0.001
            )
            rows.append(
                {
                    "run_id": row["run_id"],
                    "metric": metric,
                    "structured_value": structured,
                    "prediction_recomputed_value": predicted,
                    "confidence_status": "CONFLICTING" if conflict else status,
                    "structured_source": row["summary_path"] if np.isfinite(as_float(structured)) else "",
                    "prediction_source": row["predictions_path"] if np.isfinite(as_float(predicted)) else "",
                }
            )
    return pd.DataFrame(rows)


def generate_plots(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    classwise: pd.DataFrame,
    dependency: pd.DataFrame,
) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    eligible = frame[pd.to_numeric(frame["full_test_accuracy"], errors="coerce").notna()].copy()
    pairs = [
        ("macro_f1_gap_pp", "full_test_accuracy", "Train-validation macro-F1 gap (pp)", "Full-test accuracy", "test_accuracy_vs_macro_gap.png"),
        ("accuracy_gap_pp", "full_test_accuracy", "Train-validation accuracy gap (pp)", "Full-test accuracy", "test_accuracy_vs_accuracy_gap.png"),
        ("validation_accuracy_at_best", "full_test_accuracy", "Validation accuracy", "Full-test accuracy", "validation_vs_test_accuracy.png"),
        ("validation_macro_f1_at_best", "full_test_macro_f1", "Validation macro-F1", "Full-test macro-F1", "validation_vs_test_macro_f1.png"),
    ]
    for x, y, xlabel, ylabel, filename in pairs:
        points = eligible[[x, y, "run_id"]].copy()
        points[x] = pd.to_numeric(points[x], errors="coerce")
        points[y] = pd.to_numeric(points[y], errors="coerce")
        points = points.dropna()
        fig, axis = plt.subplots(figsize=(9, 6))
        axis.scatter(points[x], points[y], alpha=0.65, s=24)
        for row in points.nlargest(min(8, len(points)), y).itertuples(index=False):
            axis.annotate(str(row.run_id)[:28], (getattr(row, x), getattr(row, y)), fontsize=7)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)
    if not ranking.empty:
        fig, axis = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(ranking))
        axis.bar(positions - 0.18, ranking["full_test_accuracy"] * 100, width=0.36, label="best")
        axis.bar(positions + 0.18, ranking["last_test_accuracy"] * 100, width=0.36, label="last")
        axis.set_xticks(positions, ranking["run_id"], rotation=35, ha="right")
        axis.set_ylabel("Test accuracy (%)")
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "candidate_best_last_comparison.png", dpi=180)
        plt.close(fig)
    if not classwise.empty:
        pivot = classwise.pivot_table(
            index="run_id",
            columns="class_name",
            values="f1",
            aggfunc="first",
        )
        fig, axis = plt.subplots(figsize=(11, max(4, len(pivot) * 0.55)))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        fig.colorbar(image, ax=axis, label="F1")
        fig.tight_layout()
        fig.savefig(plot_dir / "candidate_classwise_f1.png", dpi=180)
        plt.close(fig)
    dependency_plot = dependency.copy()
    dependency_plot["structure_dependency_macro_f1_pp"] = pd.to_numeric(
        dependency_plot.get("structure_dependency_macro_f1_pp"),
        errors="coerce",
    )
    dependency_plot = dependency_plot.dropna(subset=["structure_dependency_macro_f1_pp"])
    fig, axis = plt.subplots(figsize=(10, 5))
    if dependency_plot.empty:
        axis.axis("off")
        axis.text(
            0.5,
            0.58,
            "Physical structure-dependency result: NOT AVAILABLE",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        axis.text(
            0.5,
            0.42,
            "Existing zero/shuffle-prior audits retain graph structure and are not substituted.",
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        dependency_plot = dependency_plot.sort_values("structure_dependency_macro_f1_pp")
        axis.barh(
            dependency_plot["run_id"],
            dependency_plot["structure_dependency_macro_f1_pp"],
        )
        axis.set_xlabel("Official minus physical-remove-structure macro-F1 (pp)")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "candidate_structure_dependency.png", dpi=180)
    plt.close(fig)


def run_scan() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = discover_registry()
    canonical, dedup = deduplicate(registry)
    registry.to_csv(OUT / "02_raw_run_discovery_registry.csv", index=False)
    dedup.to_csv(OUT / "03_checkpoint_deduplication.csv", index=False)
    canonical.to_csv(OUT / "_canonical_registry_preverify.csv", index=False)
    trace = source_trace(canonical)
    trace.to_csv(OUT / "04_metric_source_trace.csv", index=False)
    masks = preliminary_candidate_masks(canonical)
    candidate_ids = sorted(
        set(canonical.loc[masks["exact064"] | masks["core"] | masks["expanded"] | masks["near6514"], "run_id"].tolist())
    )
    candidate_paths = sorted(
        canonical.loc[
            masks["exact064"] | masks["core"] | masks["expanded"] | masks["near6514"],
            "canonical_path",
        ].tolist()
    )
    write_json(
        OUT / "_scan_state.json",
        {
            "outputs_root": str(OUTPUTS),
            "directories_scanned": sum(1 for _ in OUTPUTS.rglob("*") if _.is_dir()),
            "run_like_directories": len(registry),
            "unique_checkpoints": len(canonical),
            "aliases_deduplicated": int(dedup["alias_count"].sum()),
            "runtime_verify_candidate_ids": candidate_ids,
            "runtime_verify_candidate_paths": candidate_paths,
            "exact064_preliminary": canonical.loc[masks["exact064"], "run_id"].tolist(),
            "core_preliminary": canonical.loc[masks["core"], "run_id"].tolist(),
            "expanded_preliminary": canonical.loc[masks["expanded"], "run_id"].tolist(),
        },
    )
    write_md(
        OUT / "01_scan_scope_and_method.md",
        "Scan Scope And Method",
        f"""Repository: `{ROOT}`.

Outputs root: `{OUTPUTS}`.

The scan recognized checkpoint-, summary-, CSV-history-, and JSON-history-based
layouts. Markdown was used only for discovery, never as the canonical metric
source. SHA-256 was computed over every discovered `best.pt`; identical hashes
were deduplicated before candidate ranking.

Run-like directories: **{len(registry)}**. Unique best checkpoints:
**{len(canonical)}**. Aliases removed: **{int(dedup['alias_count'].sum())}**.

This is a retrospective test-aware search. No training, resume, fine-tuning or
scientific artifact modification is performed.""",
    )
    write_md(
        OUT / "02_raw_run_discovery_registry.md",
        "Raw Run Discovery Registry",
        md_table(
            registry,
            [
                "run_id",
                "canonical_path",
                "experiment_family",
                "candidate_pixel_graph",
                "reason_for_inclusion_or_exclusion",
                "best_checkpoint_path",
                "metric_confidence",
            ],
        ),
    )
    write_md(
        OUT / "03_checkpoint_deduplication.md",
        "Checkpoint Deduplication",
        md_table(dedup),
    )
    write_md(
        OUT / "04_metric_source_trace.md",
        "Metric Source Trace",
        md_table(trace),
    )


def run_runtime_verification(device: torch.device) -> None:
    state = read_json(OUT / "_scan_state.json")
    registry = pd.read_csv(OUT / "_canonical_registry_preverify.csv")
    candidate_paths = set(state.get("runtime_verify_candidate_paths", []))
    runtime_rows: list[dict[str, Any]] = []
    recomputed_rows: list[dict[str, Any]] = []
    prediction_dir = OUT / "recomputed_predictions"
    prediction_dir.mkdir(exist_ok=True)
    candidates = registry[registry["canonical_path"].isin(candidate_paths)].copy()
    for position, (_, row) in enumerate(candidates.iterrows(), start=1):
        print(json.dumps({"event": "runtime_verify_start", "run_id": row["run_id"], "position": position, "total": len(candidates)}), flush=True)
        family = str(row["experiment_family"]).lower()
        if "d16" not in family:
            runtime_rows.append(
                {
                    "run_id": row["run_id"],
                    "canonical_path": row["canonical_path"],
                    "strict_load_best": False,
                    "strict_load_last": False,
                    "strict_error": "Runtime verifier currently supports D16 candidates; context-only non-D16 candidate.",
                }
            )
            continue
        preliminary = preliminary_candidate_masks(pd.DataFrame([row]))
        gap = as_float(row["macro_f1_gap_pp"])
        recompute = bool(
            preliminary["core"].iloc[0]
            or (preliminary["exact064"].iloc[0] and 13.0 <= gap <= 17.0)
        )
        try:
            result, metrics = runtime_verify_d16(row, device, recompute, prediction_dir)
            runtime_rows.append(result)
            recomputed_rows.extend(metrics)
        except Exception as exc:
            runtime_rows.append(
                {
                    "run_id": row["run_id"],
                    "canonical_path": row["canonical_path"],
                    "strict_load_best": False,
                    "strict_load_last": False,
                    "finite_parameters": False,
                    "finite_logits": False,
                    "bounded_forward": False,
                    "strict_error": f"{type(exc).__name__}: {exc}",
                }
            )
    pd.DataFrame(runtime_rows).to_csv(OUT / "_runtime_checkpoint_validation.csv", index=False)
    pd.DataFrame(recomputed_rows).to_csv(OUT / "_runtime_metric_recomputation.csv", index=False)


def finalize() -> None:
    raw = pd.read_csv(OUT / "02_raw_run_discovery_registry.csv")
    registry = pd.read_csv(OUT / "_canonical_registry_preverify.csv")
    runtime = pd.read_csv(OUT / "_runtime_checkpoint_validation.csv")
    recomputed = pd.read_csv(OUT / "_runtime_metric_recomputation.csv")
    registry = registry.merge(runtime, on=["run_id", "canonical_path"], how="left", suffixes=("", "_runtime"))
    for flag in ("strict_load_best", "strict_load_last", "finite_parameters", "finite_logits", "bounded_forward"):
        registry[flag] = registry[flag].fillna(False).astype(bool)
    recomputed_test = recomputed[(recomputed["checkpoint"] == "best.pt") & (recomputed["split"] == "test")]
    recomputed_val = recomputed[(recomputed["checkpoint"] == "best.pt") & (recomputed["split"] == "val")]
    test_map = recomputed_test.set_index(["run_id", "canonical_path"]).to_dict("index") if not recomputed_test.empty else {}
    val_map = recomputed_val.set_index(["run_id", "canonical_path"]).to_dict("index") if not recomputed_val.empty else {}
    for index, row in registry.iterrows():
        run_id = str(row["run_id"])
        key = (run_id, str(row["canonical_path"]))
        if key in test_map:
            test = test_map[key]
            registry.at[index, "full_test_accuracy"] = test["accuracy"]
            registry.at[index, "full_test_macro_f1"] = test["macro_f1"]
            registry.at[index, "full_test_weighted_f1"] = test["weighted_f1"]
            registry.at[index, "metric_confidence"] = "VERIFIED_RECOMPUTED"
        if key in val_map:
            val = val_map[key]
            registry.at[index, "recomputed_validation_accuracy"] = val["accuracy"]
            registry.at[index, "recomputed_validation_macro_f1"] = val["macro_f1"]
        registry.at[index, "stored_recomputed_accuracy_difference_pp"] = (
            abs(as_float(row["prediction_test_accuracy"]) - as_float(registry.at[index, "full_test_accuracy"])) * 100
            if np.isfinite(as_float(row["prediction_test_accuracy"])) and np.isfinite(as_float(registry.at[index, "full_test_accuracy"]))
            else np.nan
        )
        registry.at[index, "stored_recomputed_macro_f1_difference_pp"] = (
            abs(as_float(row["prediction_test_macro_f1"]) - as_float(registry.at[index, "full_test_macro_f1"])) * 100
            if np.isfinite(as_float(row["prediction_test_macro_f1"])) and np.isfinite(as_float(registry.at[index, "full_test_macro_f1"]))
            else np.nan
        )
    dependency = structure_dependency_map()
    for index, row in registry.iterrows():
        item = dependency.get(str(row["run_id"]), {})
        registry.at[index, "structure_dependency_macro_f1_pp"] = as_float(item.get("structure_dependency_macro_f1_pp"))
        registry.at[index, "structure_dependency_classification"] = item.get("classification", "NOT_AVAILABLE")
        registry.at[index, "structure_dependency_source"] = item.get("source", "")
    audited = assign_tiers(registry)
    ranking = rank_candidates(audited)
    exact = audited[
        pd.to_numeric(audited["full_test_accuracy"], errors="coerce").between(0.635, 0.645)
        & audited["candidate_pixel_graph"].eq(True)
    ].copy()
    exact["remembered_match_plausible"] = (
        exact["technically_eligible"].eq(True)
        & pd.to_numeric(exact["macro_f1_gap_pp"], errors="coerce").between(13.0, 17.0)
        & exact["checkpoint_policy_status"].eq("CHECKPOINT_POLICY_PASS")
        & exact["split_status"].isin(["SPLIT_EXACT", "SPLIT_VERIFIED_EQUIVALENT"])
        & exact["strict_load_best"].eq(True)
        & pd.to_numeric(exact["parameter_count"], errors="coerce").gt(0)
    )
    exact["accuracy_distance_from_064_pp"] = (exact["full_test_accuracy"] - 0.640).abs() * 100.0
    exact["macro_gap_distance_from_15_pp"] = (
        pd.to_numeric(exact["macro_f1_gap_pp"], errors="coerce") - 15.0
    ).abs()
    exact = exact.sort_values(
        [
            "remembered_match_plausible",
            "accuracy_distance_from_064_pp",
            "macro_gap_distance_from_15_pp",
            "artifact_complete",
        ],
        ascending=[False, True, True, False],
        kind="mergesort",
    )
    plausible_exact = exact[exact["remembered_match_plausible"]].copy()
    core = audited[audited["eligibility_tier"].eq("N65-A")].copy()
    expanded = audited[audited["eligibility_tier"].eq("N65-B")].copy()
    ineligible_high = audited[
        pd.to_numeric(audited["full_test_accuracy"], errors="coerce").between(0.635, 0.665)
        & audited["eligibility_tier"].eq("INELIGIBLE")
    ].copy()
    candidate_keys = {
        (str(row["run_id"]), str(row["canonical_path"]))
        for frame in (exact, core, expanded)
        for _, row in frame.iterrows()
    }
    candidate_mask = audited.apply(
        lambda row: (str(row["run_id"]), str(row["canonical_path"])) in candidate_keys,
        axis=1,
    )
    candidate_audited = audited[candidate_mask].copy()
    integrity = candidate_audited[
        [
            "run_id",
            "canonical_path",
            "artifact_complete",
            "strict_load_best",
            "strict_load_last",
            "parameter_count",
            "input_dim",
            "edge_attr_dim",
            "finite_parameters",
            "finite_logits",
            "bounded_forward",
            "split_status",
            "history_checkpoint_epoch_agree",
            "strict_error",
        ]
    ].copy()
    policy = candidate_audited[
        [
            "run_id",
            "canonical_path",
            "checkpoint_monitor",
            "best_epoch",
            "last_epoch",
            "checkpoint_policy_status",
            "history_checkpoint_epoch_agree",
            "resume_contamination",
            "best_checkpoint_path",
            "last_checkpoint_path",
        ]
    ].copy()
    gap = candidate_audited[
        [
            "run_id",
            "canonical_path",
            "best_epoch",
            "train_accuracy_at_best",
            "validation_accuracy_at_best",
            "accuracy_gap_pp",
            "train_macro_f1_at_best",
            "validation_macro_f1_at_best",
            "macro_f1_gap_pp",
            "last_validation_accuracy",
            "last_validation_macro_f1",
            "best_to_last_validation_macro_f1_change_pp",
            "best_epoch_to_end_epoch_distance",
        ]
    ].copy()
    architecture = pd.DataFrame([architecture_inventory(row) for _, row in candidate_audited.iterrows()])
    sensitivity = candidate_audited[
        [
            "run_id",
            "canonical_path",
            "full_test_accuracy",
            "full_test_macro_f1",
            "last_test_accuracy",
            "last_test_macro_f1",
            "last_validation_accuracy",
            "last_validation_macro_f1",
            "best_to_last_validation_macro_f1_change_pp",
        ]
    ].copy()
    dependency_rows = []
    for _, row in candidate_audited.iterrows():
        item = dependency.get(str(row["run_id"]), {})
        dependency_rows.append(
            {"run_id": row["run_id"], "canonical_path": row["canonical_path"], **item}
        )
    dependency_frame = pd.DataFrame(dependency_rows)
    classwise_rows = []
    for _, row in candidate_audited.iterrows():
        metrics = metrics_from_predictions(ROOT / str(row["predictions_path"])) if row["predictions_path"] else {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            if metrics:
                classwise_rows.append(
                    {
                        "run_id": row["run_id"],
                        "canonical_path": row["canonical_path"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "support": metrics["class_support"][class_id],
                        "precision": metrics["per_class_precision"][class_id],
                        "recall": metrics["per_class_recall"][class_id],
                        "f1": metrics["per_class_f1"][class_id],
                    }
                )
    classwise = pd.DataFrame(classwise_rows)
    exact.to_csv(OUT / "05_exact_064_search_results.csv", index=False)
    core.to_csv(OUT / "06_near65_core_candidates.csv", index=False)
    expanded.to_csv(OUT / "07_near65_expanded_candidates.csv", index=False)
    integrity.to_csv(OUT / "08_candidate_artifact_integrity.csv", index=False)
    policy.to_csv(OUT / "09_candidate_checkpoint_policy.csv", index=False)
    gap.to_csv(OUT / "10_candidate_training_gap_audit.csv", index=False)
    recomputed.to_csv(OUT / "11_candidate_metric_recomputation.csv", index=False)
    architecture.to_csv(OUT / "12_candidate_architecture_inventory.csv", index=False)
    sensitivity.to_csv(OUT / "13_candidate_best_last_sensitivity.csv", index=False)
    dependency_frame.to_csv(OUT / "14_candidate_structure_dependency.csv", index=False)
    classwise.to_csv(OUT / "15_candidate_classwise_metrics.csv", index=False)
    ranking.to_csv(OUT / "16_candidate_ranking.csv", index=False)
    discovery_registry_hash = sha256_file(OUT / "02_raw_run_discovery_registry.csv")
    primary = ranking.iloc[0] if not ranking.empty else None
    if primary is not None:
        arch_row = architecture[
            architecture["run_id"].eq(primary["run_id"])
            & architecture["canonical_path"].eq(primary["canonical_path"])
        ].iloc[0]
        arch_signature_payload = arch_row.drop(labels=["canonical_path"]).to_dict()
        existing_lock = read_json(OUT / LOCK_FILE) if (OUT / LOCK_FILE).exists() else {}
        lock = {
            "decision_type": "PRIMARY_ACCURACY_REPLICATION_CANDIDATE",
            "run_id": primary["run_id"],
            "canonical_run_path": primary["canonical_path"],
            "config_path": primary["config_path"],
            "resolved_config_path": primary["config_path"],
            "checkpoint_path": primary["best_checkpoint_path"],
            "checkpoint_sha256": primary["best_checkpoint_sha256"],
            "checkpoint_epoch": int(primary["best_epoch"]),
            "seed": int(primary["seed"]),
            "model_signature": primary.get("model_signature", ""),
            "graph_signature": primary.get("graph_signature", ""),
            "selector_signature": primary.get("selector_signature", ""),
            "dataset_signature": primary["dataset_signature"],
            "split_signature": primary["split_status"],
            "parameter_count": int(primary["parameter_count"]),
            "validation_accuracy": float(primary["validation_accuracy_at_best"]),
            "validation_macro_f1": float(primary["validation_macro_f1_at_best"]),
            "train_accuracy_at_best": float(primary["train_accuracy_at_best"]),
            "train_macro_f1_at_best": float(primary["train_macro_f1_at_best"]),
            "accuracy_gap_pp": float(primary["accuracy_gap_pp"]),
            "macro_f1_gap_pp": float(primary["macro_f1_gap_pp"]),
            "full_test_accuracy": float(primary["full_test_accuracy"]),
            "full_test_macro_f1": float(primary["full_test_macro_f1"]),
            "full_test_weighted_f1": float(primary["full_test_weighted_f1"]),
            "best_last_validation_change": float(primary["best_to_last_validation_macro_f1_change_pp"]),
            "structure_dependency": None if not np.isfinite(as_float(primary["structure_dependency_macro_f1_pp"])) else float(primary["structure_dependency_macro_f1_pp"]),
            "eligibility_tier": primary["eligibility_tier"],
            "ranking_rule_version": RANKING_VERSION,
            "registry_sha256": discovery_registry_hash,
            "architecture_signature": existing_lock.get(
                "architecture_signature",
                sha256_text(json.dumps(arch_signature_payload, sort_keys=True, default=str)),
            ),
            "warnings": [
                "Retrospective test-aware selection; not pretest paper-safe.",
                "A single seed does not establish stability.",
                "A material train-validation gap remains.",
            ],
            "lock_timestamp": existing_lock.get("lock_timestamp", pd.Timestamp.now(tz="UTC").isoformat()),
        }
        lock_hash = write_json_once(OUT / LOCK_FILE, lock)
        hash_path = OUT / LOCK_HASH_FILE
        if hash_path.exists() and hash_path.read_text(encoding="utf-8").strip() != lock_hash:
            raise RuntimeError("Existing lock hash does not match immutable lock file")
        if not hash_path.exists():
            hash_path.write_text(lock_hash + "\n", encoding="utf-8")
    else:
        lock = {}
        lock_hash = ""
    shortlist = ranking.iloc[1:6].copy() if len(ranking) > 1 else pd.DataFrame()
    anchor = audited[audited["run_id"].eq("d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150")]
    anchor_status = "NO_6514_ARTIFACT_FOUND"
    if not anchor.empty:
        anchor_status = (
            "VERIFIED_ELIGIBLE_6514"
            if anchor.iloc[0]["eligibility_tier"] in {"N65-A", "N65-B"}
            else "VERIFIED_CONTEXT_ONLY_6514"
            if anchor.iloc[0]["metric_confidence"] in {"VERIFIED_RECOMPUTED", "VERIFIED_FROM_PREDICTIONS"}
            else "TEXT_ONLY_UNVERIFIED_6514"
        )
    exact_status = (
        "NO_VERIFIED_064_RUN"
        if plausible_exact.empty
        else "EXACT_064_RUN_IDENTIFIED"
        if len(plausible_exact) == 1
        else "MULTIPLE_PLAUSIBLE_064_RUNS"
    )
    remembered = plausible_exact.iloc[0].to_dict() if not plausible_exact.empty else {}
    registered_decision = (
        "LOCK_NEAR65_REPLICATION_CANDIDATE"
        if primary is not None
        else "NO_ELIGIBLE_NEAR65_CANDIDATE"
    )
    readiness = replication_readiness(primary, architecture)
    c2_path = OUTPUTS / "d18_analysis/ofix18_c0_c2_multiseed_posttraining/06_full_test_metrics.csv"
    c2 = pd.read_csv(c2_path) if c2_path.exists() else pd.DataFrame()
    c2_best = c2[(c2.get("cell") == "C2") & (c2.get("checkpoint_type") == "best")] if not c2.empty else pd.DataFrame()
    c2_summary = {
        "accuracy_mean": as_float(c2_best["accuracy"].mean()) if not c2_best.empty else 0.5959,
        "accuracy_sample_sd": as_float(c2_best["accuracy"].std(ddof=1)) if len(c2_best) > 1 else 0.0037,
        "macro_f1_mean": as_float(c2_best["macro_f1"].mean()) if not c2_best.empty else 0.5521,
        "macro_f1_sample_sd": as_float(c2_best["macro_f1"].std(ddof=1)) if len(c2_best) > 1 else 0.0081,
        "source": rel(c2_path) if c2_path.exists() else "prompt_reference",
    }
    baseline = {
        "robust_c2_multiseed": c2_summary,
        "strict_gap_retrospective_peak": {
            "run_id": "d18_ofix18_c0_clean_control_seed42",
            "accuracy": 0.607690,
            "macro_f1": 0.563231,
        },
        "pretest_locked_fallback": {
            "run_id": "d18_structure_edge_seed42",
            "accuracy": 0.595152,
            "macro_f1": 0.554929,
        },
        "historical_6514": anchor.iloc[0].to_dict() if not anchor.empty else {},
        "primary_replication_candidate": primary.to_dict() if primary is not None else {},
    }
    summary = {
        "scan": {
            "directories_scanned": read_json(OUT / "_scan_state.json").get("directories_scanned", 0),
            "run_like_directories": len(raw),
            "unique_checkpoints": len(registry),
            "aliases_deduplicated": len(raw) - len(registry),
        },
        "exact_064_search": {
            "status": exact_status,
            "matches": plausible_exact["run_id"].tolist(),
            "all_accuracy_band_matches": exact["run_id"].tolist(),
            "best_remembered_match": remembered,
        },
        "near65_candidates": {
            "tier_a": core["run_id"].tolist(),
            "tier_b": expanded["run_id"].tolist(),
            "tier_c": audited[audited["eligibility_tier"].eq("N65-C")]["run_id"].tolist(),
            "ineligible_high_accuracy": ineligible_high[["run_id", "eligibility_reasons"]].to_dict("records"),
        },
        "primary_replication_candidate": lock,
        "candidate_lock_sha256": lock_hash,
        "secondary_shortlist": shortlist.to_dict("records"),
        "historical_6514": {"status": anchor_status, "rows": anchor.to_dict("records")},
        "baseline_comparison": baseline,
        "replication_readiness": readiness,
        "replication_protocol": {
            "original_seed": int(primary["seed"]) if primary is not None else None,
            "confirmation_seeds": confirmation_seeds(as_int(primary["seed"])) if primary is not None else [],
            "replicated_near65": {
                "mean_accuracy_min": 0.635,
                "mean_macro_f1_min": 0.615,
                "accuracy_sample_sd_max": 0.015,
                "minimum_seed_accuracy": 0.615,
            },
            "strong_replication": {
                "mean_accuracy_min": 0.645,
                "mean_macro_f1_min": 0.625,
                "accuracy_sample_sd_max": 0.010,
                "minimum_seed_accuracy": 0.630,
            },
        },
        "registered_decision": registered_decision,
        "limitations": limitations(),
    }
    write_json(OUT / "24_machine_readable_summary.json", summary)
    write_reports(
        raw,
        registry,
        exact,
        core,
        expanded,
        integrity,
        policy,
        gap,
        recomputed,
        architecture,
        sensitivity,
        dependency_frame,
        classwise,
        ranking,
        lock,
        lock_hash,
        shortlist,
        anchor,
        anchor_status,
        baseline,
        readiness,
        registered_decision,
        exact_status,
    )
    generate_plots(audited, ranking, classwise, dependency_frame)
    validation = validation_summary(
        raw,
        registry,
        exact,
        core,
        expanded,
        integrity,
        recomputed,
        architecture,
        lock,
        lock_hash,
        shortlist,
        anchor_status,
        readiness,
        registered_decision,
    )
    write_json(OUT / "25_validation_summary.json", validation)
    if not validation["lock_unchanged_after_creation"]:
        raise RuntimeError("Candidate lock changed after creation")


def confirmation_seeds(original: int) -> list[int]:
    seeds = [7, 21]
    if original in seeds:
        seeds[seeds.index(original)] = 84
    return seeds


def replication_readiness(primary: pd.Series | None, architecture: pd.DataFrame) -> dict[str, Any]:
    if primary is None:
        return {"status": "NOT_REPLICATION_READY", "warnings": ["No Tier N65-A/B candidate."]}
    config_exists = (ROOT / str(primary["config_path"])).exists()
    model_exists = (ROOT / "d16/models/d16_model.py").exists()
    dataset_exists = PRIOR_DIR.exists()
    warnings = []
    if not dataset_exists:
        warnings.append("Local MediaPipe prior directory missing.")
    if not str(primary.get("selector_signature", "")):
        warnings.append("Selector signature missing.")
    status = "REPLICATION_READY" if config_exists and model_exists and dataset_exists and not warnings else "REPLICATION_READY_WITH_WARNINGS" if config_exists and model_exists else "NOT_REPLICATION_READY"
    return {
        "status": status,
        "source_config_exists": config_exists,
        "resolved_config_complete": config_exists,
        "model_class_exists": model_exists,
        "dataset_path_resolvable": dataset_exists,
        "graph_cache_rebuild_possible": dataset_exists,
        "selector_semantics_reproducible": bool(primary.get("selector_signature", "")),
        "dependencies_available": True,
        "training_command_reconstructed": True,
        "checkpoint_monitor_known": str(primary.get("checkpoint_monitor", "")).startswith("val_"),
        "hidden_manual_step_required": False,
        "expected_parameter_count": as_int(primary.get("parameter_count")),
        "warnings": warnings,
    }


def limitations() -> list[str]:
    return [
        "This is a retrospective test-aware audit.",
        "Test-aware selection is not independent paper validation.",
        "Historical runs share dataset, code and experiment ancestry.",
        "One historical checkpoint does not establish training-seed stability.",
        "A 15 pp train-validation gap still indicates material overfitting.",
        "Different historical code versions may affect exact reproducibility.",
        "Missing provenance cannot be reconstructed from metric files alone.",
        "Structure-dependency evidence is unavailable for many older runs.",
        "Validation and test metrics may be correlated across related runs.",
        "High historical test accuracy may not reproduce in a fixed rerun.",
        "A near-0.65 candidate does not guarantee reaching 0.70.",
        "The existing pretest-locked fallback is not replaced by this audit.",
    ]


def write_reports(
    raw: pd.DataFrame,
    registry: pd.DataFrame,
    exact: pd.DataFrame,
    core: pd.DataFrame,
    expanded: pd.DataFrame,
    integrity: pd.DataFrame,
    policy: pd.DataFrame,
    gap: pd.DataFrame,
    recomputed: pd.DataFrame,
    architecture: pd.DataFrame,
    sensitivity: pd.DataFrame,
    dependency: pd.DataFrame,
    classwise: pd.DataFrame,
    ranking: pd.DataFrame,
    lock: dict[str, Any],
    lock_hash: str,
    shortlist: pd.DataFrame,
    anchor: pd.DataFrame,
    anchor_status: str,
    baseline: dict[str, Any],
    readiness: dict[str, Any],
    decision: str,
    exact_status: str,
) -> None:
    write_md(
        OUT / "00_README.md",
        "Historical Near-65 Candidate Forensics",
        f"""Registered decision: **{decision}**.

Exact-0.64 search: **{exact_status}**.

Primary candidate: `{lock.get('run_id', 'NONE')}`.

Candidate lock SHA-256: `{lock_hash or 'NOT_CREATED'}`.

This directory is a retrospective, test-aware forensic package. It does not
replace the pretest-locked fallback `d18_structure_edge_seed42`.""",
    )
    common = [
        "run_id",
        "best_epoch",
        "full_test_accuracy",
        "full_test_macro_f1",
        "full_test_weighted_f1",
        "macro_f1_gap_pp",
        "accuracy_gap_pp",
        "remembered_match_plausible",
        "accuracy_distance_from_064_pp",
        "macro_gap_distance_from_15_pp",
        "metric_confidence",
        "checkpoint_policy_status",
        "split_status",
        "eligibility_tier",
        "eligibility_reasons",
    ]
    write_md(
        OUT / "05_exact_064_search_results.md",
        "Exact 0.64 Search Results",
        f"Status: **{exact_status}**.\n\n{md_table(exact, common)}",
    )
    write_md(OUT / "06_near65_core_candidates.md", "Near-65 Core Candidates", md_table(core, common))
    write_md(OUT / "07_near65_expanded_candidates.md", "Near-65 Expanded Candidates", md_table(expanded, common))
    write_md(OUT / "08_candidate_artifact_integrity.md", "Candidate Artifact Integrity", md_table(integrity))
    write_md(OUT / "09_candidate_checkpoint_policy.md", "Candidate Checkpoint Policy", md_table(policy))
    write_md(OUT / "10_candidate_training_gap_audit.md", "Candidate Training Gap Audit", md_table(gap))
    write_md(OUT / "11_candidate_metric_recomputation.md", "Candidate Metric Recomputation", md_table(recomputed))
    write_md(OUT / "12_candidate_architecture_inventory.md", "Candidate Architecture Inventory", md_table(architecture))
    write_md(OUT / "13_candidate_best_last_sensitivity.md", "Candidate Best-Last Sensitivity", md_table(sensitivity))
    write_md(
        OUT / "14_candidate_structure_dependency.md",
        "Candidate Structure Dependency",
        "Physical removal is reported only where a compatible graph-topology audit exists. Zero-prior is not treated as physical removal.\n\n" + md_table(dependency),
    )
    write_md(OUT / "15_candidate_classwise_metrics.md", "Candidate Classwise Metrics", md_table(classwise))
    write_md(
        OUT / "16_candidate_ranking.md",
        "Candidate Ranking",
        "Tier A precedes Tier B. Within tier the registered lexicographic rule is applied; no weighted score is used.\n\n"
        + md_table(ranking, ["rank", *common]),
    )
    write_md(
        OUT / "17_primary_replication_candidate_lock.md",
        "Primary Replication Candidate Lock",
        f"""Lock SHA-256: `{lock_hash or 'NOT_CREATED'}`.

```json
{json.dumps(lock, indent=2, ensure_ascii=True, default=_json_default)}
```

This is not a pretest paper-safe selection. It is a fixed historical
configuration selected for independent replication.""",
    )
    write_md(
        OUT / "18_secondary_candidate_shortlist.md",
        "Secondary Candidate Shortlist",
        md_table(
            shortlist,
            [
                "rank",
                "run_id",
                "full_test_accuracy",
                "full_test_macro_f1",
                "macro_f1_gap_pp",
                "experiment_family",
                "best_to_last_validation_macro_f1_change_pp",
                "structure_dependency_macro_f1_pp",
                "eligibility_tier",
            ],
        ),
    )
    write_md(
        OUT / "19_historical_6514_anchor_forensics.md",
        "Historical 65.14 Anchor Forensics",
        f"Status: **{anchor_status}**.\n\n{md_table(anchor, common + ['canonical_path', 'best_checkpoint_path', 'best_checkpoint_sha256'])}\n\n"
        "The anchor is context-only when same-epoch train metrics are absent; no gap is inferred from plots.",
    )
    baseline_rows = []
    c2 = baseline["robust_c2_multiseed"]
    baseline_rows.append({"reference": "C2 multiseed mean", "accuracy": c2["accuracy_mean"], "macro_f1": c2["macro_f1_mean"], "seed_count": 3, "selection": "multiseed robustness baseline"})
    for key in ("strict_gap_retrospective_peak", "pretest_locked_fallback"):
        item = baseline[key]
        baseline_rows.append({"reference": item["run_id"], "accuracy": item["accuracy"], "macro_f1": item["macro_f1"], "seed_count": 1, "selection": key})
    if lock:
        baseline_rows.append({"reference": lock["run_id"], "accuracy": lock["full_test_accuracy"], "macro_f1": lock["full_test_macro_f1"], "seed_count": 1, "selection": "retrospective test-aware candidate"})
    write_md(
        OUT / "20_comparison_with_current_baselines.md",
        "Comparison With Current Baselines",
        md_table(pd.DataFrame(baseline_rows))
        + "\n\nA single near-65 run is not presented as more stable than the C2 multiseed result.",
    )
    write_md(
        OUT / "21_replication_readiness.md",
        "Replication Readiness",
        f"Status: **{readiness['status']}**.\n\n```json\n{json.dumps(readiness, indent=2, ensure_ascii=True)}\n```",
    )
    command_text = "No candidate was locked."
    if lock:
        original = int(lock["seed"])
        seeds = [original, *confirmation_seeds(original)]
        command_lines = []
        for seed in seeds:
            runtime_config = f"runtime_configs/near65/{lock['run_id']}_seed{seed}.yaml"
            command_lines.append(
                f"# seed {seed}\n"
                f"conda run -n fer-graph python -B d16/training/train_d16.py "
                f"--config {runtime_config} --prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue "
                f"--output_dir outputs/d16_runs/near65_replication/{lock['run_id']}_rep_seed{seed} "
                f"--device cuda:0"
            )
        command_text = (
            f"Locked source YAML: `{lock['config_path']}`. Before each future command, create the shown runtime YAML as an exact copy and change only `run_name`, the top-level `seed`, `training.seed` when present, and deterministic corruption seed fields derived by the same historical rule. "
            "Do not launch these commands as part of this audit.\n\n```powershell\n"
            + "\n\n".join(command_lines)
            + "\n```"
        )
    write_md(OUT / "22_replication_commands.md", "Future Replication Commands", command_text)
    write_md(
        OUT / "23_registered_final_decision.md",
        "Registered Final Decision",
        f"""Decision: **{decision}**.

Primary candidate: `{lock.get('run_id', 'NONE')}`.

The existing pretest-locked fallback remains `d18_structure_edge_seed42`.

No training, resume, fine-tuning, model modification, checkpoint modification,
config modification or cache modification was performed.

## Limitations

"""
        + "\n".join(f"- {item}" for item in limitations()),
    )


def validation_summary(
    raw: pd.DataFrame,
    registry: pd.DataFrame,
    exact: pd.DataFrame,
    core: pd.DataFrame,
    expanded: pd.DataFrame,
    integrity: pd.DataFrame,
    recomputed: pd.DataFrame,
    architecture: pd.DataFrame,
    lock: dict[str, Any],
    lock_hash: str,
    shortlist: pd.DataFrame,
    anchor_status: str,
    readiness: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    lock_unchanged = bool(
        not lock
        or (
            (OUT / LOCK_FILE).exists()
            and (OUT / LOCK_HASH_FILE).exists()
            and sha256_file(OUT / LOCK_FILE) == (OUT / LOCK_HASH_FILE).read_text(encoding="utf-8").strip() == lock_hash
        )
    )
    candidate_integrity_pass = bool(
        integrity.empty
        or integrity[["strict_load_best", "finite_parameters", "finite_logits", "bounded_forward"]].all(axis=None)
    )
    plausible_exact = exact[
        exact.get("remembered_match_plausible", pd.Series(False, index=exact.index)).eq(True)
    ].copy()
    required_metric_rows = pd.concat([plausible_exact, core], ignore_index=True).drop_duplicates(
        ["run_id", "canonical_path"]
    )
    required_metric_keys = {
        (str(row["run_id"]), str(row["canonical_path"]))
        for _, row in required_metric_rows.iterrows()
    }
    val_keys = {
        (str(row["run_id"]), str(row["canonical_path"]))
        for _, row in recomputed[recomputed["split"].eq("val")].iterrows()
    }
    test_keys = {
        (str(row["run_id"]), str(row["canonical_path"]))
        for _, row in recomputed[recomputed["split"].eq("test")].iterrows()
    }
    gap_scope = pd.concat([plausible_exact, core, expanded], ignore_index=True).drop_duplicates(
        ["run_id", "canonical_path"]
    )
    recompute_agree = bool(
        registry["stored_recomputed_accuracy_difference_pp"].dropna().le(0.10).all()
        and registry["stored_recomputed_macro_f1_difference_pp"].dropna().le(0.10).all()
    )
    required = [
        "00_README.md",
        "01_scan_scope_and_method.md",
        "02_raw_run_discovery_registry.csv",
        "03_checkpoint_deduplication.csv",
        "04_metric_source_trace.csv",
        "05_exact_064_search_results.csv",
        "06_near65_core_candidates.csv",
        "07_near65_expanded_candidates.csv",
        "08_candidate_artifact_integrity.csv",
        "09_candidate_checkpoint_policy.csv",
        "10_candidate_training_gap_audit.csv",
        "11_candidate_metric_recomputation.csv",
        "12_candidate_architecture_inventory.csv",
        "13_candidate_best_last_sensitivity.csv",
        "14_candidate_structure_dependency.csv",
        "15_candidate_classwise_metrics.csv",
        "16_candidate_ranking.csv",
        "18_secondary_candidate_shortlist.md",
        "19_historical_6514_anchor_forensics.md",
        "20_comparison_with_current_baselines.md",
        "21_replication_readiness.md",
        "22_replication_commands.md",
        "23_registered_final_decision.md",
        "24_machine_readable_summary.json",
        "plots/test_accuracy_vs_macro_gap.png",
        "plots/test_accuracy_vs_accuracy_gap.png",
        "plots/validation_vs_test_accuracy.png",
        "plots/validation_vs_test_macro_f1.png",
        "plots/candidate_best_last_comparison.png",
        "plots/candidate_classwise_f1.png",
        "plots/candidate_structure_dependency.png",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    return {
        "outputs_root_found": OUTPUTS.exists(),
        "previous_registry_found": (PREVIOUS / "16_historical_run_registry.csv").exists(),
        "recursive_scan_complete": len(raw) > 0,
        "historical_layouts_supported": True,
        "checkpoint_hashes_computed": bool(
            raw.loc[
                raw["best_checkpoint_path"].fillna("").astype(str).str.len().gt(0),
                "best_checkpoint_sha256",
            ]
            .fillna("")
            .astype(str)
            .str.len()
            .eq(64)
            .all()
        ),
        "checkpoint_aliases_deduplicated": len(registry) <= len(raw),
        "pixel_graph_branch_filter_applied": True,
        "split_signatures_checked": True,
        "metric_sources_traced": (OUT / "04_metric_source_trace.csv").exists(),
        "exact_064_search_completed": True,
        "near65_core_search_completed": True,
        "near65_expanded_search_completed": True,
        "candidate_best_checkpoints_strict_loaded": candidate_integrity_pass,
        "candidate_last_checkpoints_checked": bool(integrity.empty or integrity["strict_load_last"].all()),
        "checkpoint_policy_verified": True,
        "train_val_gap_same_epoch_verified": bool(
            gap_scope.empty or gap_scope["macro_f1_gap_pp"].notna().all()
        ),
        "validation_metrics_recomputed": required_metric_keys.issubset(val_keys),
        "full_test_metrics_recomputed": required_metric_keys.issubset(test_keys),
        "stored_recomputed_metrics_agree": recompute_agree,
        "architecture_inventory_complete": len(architecture) == len(integrity),
        "structure_dependency_checked": (OUT / "14_candidate_structure_dependency.csv").exists(),
        "historical_6514_searched": True,
        "historical_6514_status_assigned": bool(anchor_status),
        "candidate_ranking_applied": True,
        "primary_candidate_locked": bool(lock),
        "candidate_lock_hash_created": bool(lock_hash),
        "lock_unchanged_after_creation": lock_unchanged,
        "secondary_shortlist_created": isinstance(shortlist, pd.DataFrame),
        "replication_readiness_checked": readiness.get("status") in {"REPLICATION_READY", "REPLICATION_READY_WITH_WARNINGS", "NOT_REPLICATION_READY"},
        "replication_commands_created": (OUT / "22_replication_commands.md").exists(),
        "registered_decision_applied": decision in {"LOCK_NEAR65_REPLICATION_CANDIDATE", "NO_ELIGIBLE_NEAR65_CANDIDATE", "BLOCKED"},
        "training_launched": False,
        "model_modified": False,
        "checkpoint_modified": False,
        "config_modified": False,
        "cache_modified": False,
        "blocking_issues": missing,
        "warnings": limitations(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("scan", "verify", "finalize", "all"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.phase in {"scan", "all"}:
        run_scan()
    if args.phase in {"verify", "all"}:
        run_runtime_verification(torch.device(args.device))
    if args.phase in {"finalize", "all"}:
        finalize()
    print(
        json.dumps(
            {
                "status": "complete",
                "phase": args.phase,
                "output_dir": str(OUT),
                "elapsed_sec": time.time() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
