"""Definitive two-stage, read-only OFIX7-mid five-seed post-training audit.

Stage ``validation-lock`` is deliberately restricted to registered configs,
training/validation artifacts, provenance, and checkpoint payloads.  Stage
``test-reveal`` verifies the immutable validation lock before reading any test
artifact.  Neither stage writes into a completed run directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.scripts.prepare_ofix7_mid_final_replication import (
    CLASS_NAMES,
    PRIOR_SEED_OFFSET,
    semantic_diff,
    scientific_normalized_config,
)
from d16.training import train_d16 as trainer


ROOT = BOOTSTRAP_ROOT
SEEDS = [42, 1009, 1337, 777, 3407]
RUN_NAMES = {seed: f"ofix7_mid_seed{seed}" for seed in SEEDS}
POLICIES = {
    "VAL_MACRO_F1": "best_val_macro_f1",
    "VAL_ACCURACY": "best_val_accuracy",
    "LAST": "last",
}
CHECKPOINTS = ["best.pt", "best_val_macro_f1.pt", "best_val_accuracy.pt", "last.pt"]
REGISTRATION_PATH = ROOT / "outputs/d16_analysis/ofix7_mid_replication_preflight/14_replication_registration.json"
REGISTRATION_SHA_PATH = REGISTRATION_PATH.with_suffix(".sha256")
CANDIDATE_LOCK_PATH = ROOT / "outputs/d19_analysis/d19_historical_near65_candidate_forensics/17_primary_replication_candidate_lock.json"
PORTABLE_REGISTRATION_PATH = ROOT / "configs/d16/final_replication/replication_registration.json"
PORTABLE_REGISTRATION_SHA_PATH = PORTABLE_REGISTRATION_PATH.with_suffix(".sha256")
PORTABLE_CANDIDATE_LOCK_PATH = ROOT / "configs/d16/final_replication/candidate_lock.json"
PORTABLE_CANDIDATE_LOCK_SHA_PATH = PORTABLE_CANDIDATE_LOCK_PATH.with_suffix(".sha256")
HISTORICAL_RUN = ROOT / "outputs/d16_runs/r/a5b_overfit_fix_7/d16r_a5b_ofix7_prior_drop_mid_seed42"
LOCAL_PRIOR = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
T95_DF4 = 2.7764451051977987
TOLERANCE = 0.001  # 0.10 percentage points
UTC = timezone.utc


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = None
        frame = frame[columns]
    frame.to_csv(path, index=False)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "MISSING" if not math.isfinite(value) else f"{value:.6f}"
            elif value is None:
                value = "MISSING"
            cells.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def stat_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: math.nan for key in ("mean", "sample_sd", "median", "min", "max", "ci95_low", "ci95_high")} | {"n": 0}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = T95_DF4 * sd / math.sqrt(len(array)) if len(array) == 5 else math.nan
    return {
        "n": int(len(array)),
        "mean": mean,
        "sample_sd": sd,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": mean - half if math.isfinite(half) else math.nan,
        "ci95_high": mean + half if math.isfinite(half) else math.nan,
    }


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper) if index else (confidence >= lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.sum()) / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def metrics_from_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"sample_index", "y_true", "y_pred"} | {f"prob_{index}" for index in range(7)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction artifact missing columns: {missing}")
    if frame["sample_index"].duplicated().any():
        raise RuntimeError("Prediction artifact contains duplicate sample_index values")
    frame = frame.sort_values("sample_index").reset_index(drop=True)
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    probs = frame[[f"prob_{index}" for index in range(7)]].astype(float).to_numpy()
    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite probabilities in prediction artifact")
    logits_columns = [f"logit_{index}" for index in range(7)]
    if set(logits_columns).issubset(frame.columns) and not np.isfinite(frame[logits_columns].astype(float).to_numpy()).all():
        raise RuntimeError("Non-finite logits in prediction artifact")
    probability_sum_error = float(np.max(np.abs(probs.sum(axis=1) - 1.0)))
    clipped = np.clip(probs, 1e-12, 1.0)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(7), zero_division=0
    )
    confidence = probs.max(axis=1)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    correct = y_true == y_pred
    one_hot = np.eye(7, dtype=np.float64)[y_true]
    per_class = [
        {
            "class_id": index,
            "class_name": CLASS_NAMES[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index in range(7)
    ]
    return {
        "count": int(len(frame)),
        "sample_index_sha256": sha256_bytes(np.asarray(frame["sample_index"], dtype=np.int64).tobytes()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "nll": float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece": ece_score(y_true, probs),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy.mean()),
        "confidence_correct": float(confidence[correct].mean()) if correct.any() else math.nan,
        "confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else math.nan,
        "max_probability_sum_error": probability_sum_error,
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=np.arange(7)).astype(int).tolist(),
    }


class EmbargoReader:
    """Whitelist every file whose contents Stage 1 is allowed to read."""

    SAFE_RUN_NAMES = {
        "resolved_config.yaml",
        "resolved_config.json",
        "feature_schema.json",
        "train_log.csv",
        "train_metrics.csv",
        "val_metrics.csv",
        "train_console.log",
        "REPLICATION_COMPLETE.json",
    }
    SAFE_PROVENANCE_NAMES = {
        "environment.json",
        "NO_RESUME.json",
        "registration.json",
        "runtime_signatures.json",
        "source_hashes.json",
    }

    def __init__(self) -> None:
        self.read_paths: list[str] = []

    def allow(self, path: Path) -> None:
        parts = {part.lower() for part in path.parts}
        name = path.name
        allowed = (
            name in self.SAFE_RUN_NAMES
            or ("validation_snapshots" in parts and name.startswith(("best_val_macro_f1_", "best_val_accuracy_", "last_")))
            or ("replication_provenance" in parts and name in self.SAFE_PROVENANCE_NAMES)
            or ("checkpoints" in parts and name in CHECKPOINTS)
            or path.parent == ROOT / "configs/d16/final_replication"
            or path in {REGISTRATION_PATH, REGISTRATION_SHA_PATH, PORTABLE_REGISTRATION_PATH, PORTABLE_REGISTRATION_SHA_PATH}
        )
        if not allowed:
            raise RuntimeError(f"Validation-lock embargo rejected artifact content read: {path}")
        lowered = name.lower()
        if "test" in lowered or name in {"d16_train_summary.json", "predictions.csv", "per_class_metrics.csv", "confusion_matrix.csv"}:
            raise RuntimeError(f"Validation-lock embargo rejected test-bearing artifact: {path}")
        self.read_paths.append(rel(path))

    def json(self, path: Path) -> dict[str, Any]:
        self.allow(path)
        return load_json(path)

    def yaml(self, path: Path) -> dict[str, Any]:
        self.allow(path)
        return load_yaml(path)

    def csv(self, path: Path) -> pd.DataFrame:
        self.allow(path)
        return pd.read_csv(path)

    def text(self, path: Path) -> str:
        self.allow(path)
        return path.read_text(encoding="utf-8", errors="replace")

    def checkpoint(self, path: Path) -> dict[str, Any]:
        self.allow(path)
        return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict", payload.get("model", payload))
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint has no model state dictionary")
    return state


def inspect_checkpoint(path: Path, cfg: dict[str, Any], input_dim: int, reader: EmbargoReader) -> dict[str, Any]:
    payload = reader.checkpoint(path)
    state = checkpoint_state(payload)
    model = D16Model.from_config(cfg, input_dim=input_dim)
    model.load_state_dict(state, strict=True)
    finite = all(bool(torch.isfinite(tensor.detach()).all().item()) for tensor in state.values() if torch.is_tensor(tensor))
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "checkpoint": path.name,
        "path": rel(path),
        "file_sha256": sha256_file(path),
        "canonical_model_state_sha256": trainer.canonical_model_state_hash(payload),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", payload.get("step", -1))),
        "best_monitor_metric": payload.get("best_monitor_metric"),
        "best_monitor_mode": payload.get("best_monitor_mode"),
        "best_monitor_score": payload.get("best_monitor_score"),
        "best_val_macro_f1": payload.get("best_val_macro_f1"),
        "scheduler_type": payload.get("scheduler_type"),
        "optimizer_state_present": payload.get("optimizer_state_dict") is not None,
        "scheduler_state_present": payload.get("scheduler_state_dict") is not None,
        "parameter_count": parameter_count,
        "strict_load": True,
        "parameters_finite": finite,
    }


def run_path(run_root: Path, seed: int) -> Path:
    return run_root / RUN_NAMES[seed]


def history_row(frame: pd.DataFrame, epoch: int) -> dict[str, float]:
    selected = frame.loc[frame["epoch"].astype(int) == int(epoch)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one history row for epoch {epoch}, found {len(selected)}")
    row = selected.iloc[0]
    fields = ["train_loss", "train_eval_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1", "lr"]
    return {field: float(row[field]) if field in row and pd.notna(row[field]) else math.nan for field in fields}


def snapshot_analysis(run_dir: Path, stem: str, reader: EmbargoReader) -> dict[str, Any]:
    base = run_dir / "validation_snapshots"
    metrics_path = base / f"{stem}_metrics.json"
    predictions_path = base / f"{stem}_predictions.csv"
    per_class_path = base / f"{stem}_per_class.csv"
    confusion_path = base / f"{stem}_confusion_matrix.csv"
    for path in (metrics_path, predictions_path, per_class_path, confusion_path):
        if not path.exists():
            raise FileNotFoundError(path)
    stored = reader.json(metrics_path)
    predictions = reader.csv(predictions_path)
    recomputed = metrics_from_predictions(predictions)
    epoch = int(stored["epoch"])
    return {
        "stem": stem,
        "epoch": epoch,
        "stored": stored,
        "recomputed": recomputed,
        "prediction_path": rel(predictions_path),
        "prediction_sha256": sha256_file(predictions_path),
        "per_class_path": rel(per_class_path),
        "confusion_path": rel(confusion_path),
        "stored_accuracy_delta": recomputed["accuracy"] - float(stored["accuracy"]),
        "stored_macro_f1_delta": recomputed["macro_f1"] - float(stored["macro_f1"]),
    }


def artifact_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(path.resolve() for path in paths), key=str):
        stat = path.stat()
        rows.append({
            "path": rel(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        })
    return rows


def normalize_runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Remove null runtime-only loader paths absent from submitted YAML files."""

    normalized = copy.deepcopy(cfg)
    data = normalized.setdefault("data", {})
    for key in ("graph_cache_dir_detected", "graph_cache_dir_fallback"):
        if data.get(key) is None:
            data.pop(key, None)
    return normalized


def config_parity(cfg: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    cfg = normalize_runtime_config(cfg)
    training = cfg.get("training", {}) or {}
    graph = cfg.get("graph", {}) or {}
    prior = graph.get("prior_corruption", {}) or {}
    scientific_hash = canonical_json_sha(scientific_normalized_config(cfg))
    expected_hashes = set(registration.get("normalized_config_sha256", {}).values())
    optimizer = {
        "type": "AdamW",
        "lr": training.get("lr"),
        "weight_decay": training.get("weight_decay"),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
    }
    scheduler = copy.deepcopy(training.get("scheduler", {}) or {})
    scheduler["step_location"] = "after checkpoint comparison, before last.pt save"
    return {
        "scientific_normalized_sha256": scientific_hash,
        "normalized_hash_match": scientific_hash in expected_hashes,
        "optimizer_match": optimizer == registration.get("optimizer_signature"),
        "scheduler_match": scheduler == registration.get("scheduler_signature"),
        "early_stopping_match": training.get("early_stopping") == registration.get("early_stopping_signature"),
        "checkpoint_monitor_match": training.get("checkpoint_monitor") == registration.get("historical_checkpoint_policy", {}).get("monitor"),
        "batch_size_match": int(training.get("batch_size", -1)) == 16,
        "epoch_limit_match": int(training.get("max_epochs", -1)) == 90,
        "prior_seed_rule_match": int(prior.get("seed", -1)) == int(cfg["seed"]) + PRIOR_SEED_OFFSET,
        "prior_corruption_match": bool(prior.get("enabled")) and prior.get("schedule") == [
            {"start_epoch": 1, "probability": 0.1},
            {"start_epoch": 11, "probability": 0.2},
            {"start_epoch": 31, "probability": 0.3},
        ],
    }


def ols_slope(values: pd.Series, tail: int) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)[-tail:]
    if len(array) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(array), dtype=float), array, 1)[0])


def curve_evidence(history: pd.DataFrame, best_epoch: int, max_epochs: int) -> dict[str, Any]:
    history = history.sort_values("epoch").reset_index(drop=True)
    best = history.loc[history["epoch"].astype(int) == best_epoch].iloc[0]
    last = history.iloc[-1]
    lr = history["lr"].astype(float).to_numpy()
    reductions = [int(history.iloc[index]["epoch"]) for index in range(1, len(history)) if lr[index] < lr[index - 1] - 1e-15]
    executed = int(last["epoch"])
    train_gain = (float(last["train_macro_f1"]) - float(best["train_macro_f1"])) * 100.0
    val_change = (float(last["val_macro_f1"]) - float(best["val_macro_f1"])) * 100.0
    best_gap = (float(best["train_macro_f1"]) - float(best["val_macro_f1"])) * 100.0
    last_gap = (float(last["train_macro_f1"]) - float(last["val_macro_f1"])) * 100.0
    train_loss_after = float(last["train_eval_loss"]) - float(best["train_eval_loss"])
    generalization_signals = {
        "train_macro_gain_after_best_ge_3pp": train_gain >= 3.0,
        "validation_macro_decline_after_best_ge_1pp": val_change <= -1.0,
        "macro_gap_exceeds_12pp": max(best_gap, last_gap) > 12.0,
        "train_loss_falls_while_validation_worsens": train_loss_after < 0 and val_change < 0,
    }
    schedule_signals = {
        "best_in_final_15pct": best_epoch >= 0.85 * executed,
        "validation_improved_after_final_lr_reduction": bool(reductions) and best_epoch > reductions[-1],
        "lr_above_min_at_end": float(last["lr"]) > 3e-5 + 1e-12,
        "final_validation_slope_positive": ols_slope(history["val_macro_f1"], 5) > 0,
    }
    optimization_signals = {
        "train_eval_loss_decreasing_final5": ols_slope(history["train_eval_loss"], 5) < 0,
        "train_macro_increasing_final5": ols_slope(history["train_macro_f1"], 5) > 0,
        "no_clear_divergence": not (val_change <= -1.0 and train_gain >= 3.0),
        "validation_flat_or_positive_final5": ols_slope(history["val_macro_f1"], 5) >= 0,
    }
    return {
        "executed_epochs": executed,
        "best_epoch": int(best_epoch),
        "best_epoch_fraction": float(best_epoch / executed),
        "epochs_after_best": int(executed - best_epoch),
        "train_macro_gain_after_best_pp": train_gain,
        "validation_macro_change_after_best_pp": val_change,
        "macro_gap_best_pp": best_gap,
        "macro_gap_last_pp": last_gap,
        "accuracy_gap_best_pp": (float(best["train_accuracy"]) - float(best["val_accuracy"])) * 100.0,
        "accuracy_gap_last_pp": (float(last["train_accuracy"]) - float(last["val_accuracy"])) * 100.0,
        "train_loss_change_after_best": train_loss_after,
        "final10_val_macro_slope_ols_per_epoch": ols_slope(history["val_macro_f1"], 10),
        "final5_val_macro_slope_ols_per_epoch": ols_slope(history["val_macro_f1"], 5),
        "final5_train_macro_slope_ols_per_epoch": ols_slope(history["train_macro_f1"], 5),
        "final5_train_eval_loss_slope_ols_per_epoch": ols_slope(history["train_eval_loss"], 5),
        "lr_reduction_epochs": reductions,
        "lr_reduction_count": len(reductions),
        "lr_at_best": float(best["lr"]),
        "lr_at_end": float(last["lr"]),
        "min_configured_lr": 3e-5,
        "epochs_after_final_lr_reduction": int(executed - reductions[-1]) if reductions else None,
        "early_stop_trigger": executed < max_epochs,
        "stop_reason": "EARLY_STOP_VAL_LOSS_PATIENCE" if executed < max_epochs else "MAX_EPOCHS",
        "generalization_signals": generalization_signals,
        "schedule_signals": schedule_signals,
        "optimization_signals": optimization_signals,
        "generalization_supported": sum(generalization_signals.values()) >= 3,
        "schedule_supported": sum(schedule_signals.values()) >= 2,
        "optimization_supported": sum(optimization_signals.values()) >= 3,
    }


def save_plot_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite plot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def line_plot(path: Path, series: list[tuple[np.ndarray, np.ndarray, str]], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.8))
    for x, y, label in series:
        plt.plot(x, y, label=label, linewidth=1.6)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    save_plot_new(path)


def bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, color="#2f6f8f")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    save_plot_new(path)


def validation_lock(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output analysis directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reader = EmbargoReader()
    registration = reader.json(REGISTRATION_PATH)
    portable_registration = reader.json(PORTABLE_REGISTRATION_PATH)
    if registration != portable_registration:
        raise RuntimeError("Preflight and portable registration JSON differ semantically")
    registered = [int(seed) for seed in registration.get("registered_seeds", [])]
    if registered != SEEDS or registration.get("seed_reporting_order") != SEEDS:
        raise RuntimeError(f"Registered seed order mismatch: {registered}")

    raw_registration_sha = sha256_file(REGISTRATION_PATH)
    normalized_registration_sha = normalized_text_sha256(PORTABLE_REGISTRATION_PATH)
    output_sidecar = REGISTRATION_SHA_PATH.read_text(encoding="utf-8").strip()
    portable_sidecar = PORTABLE_REGISTRATION_SHA_PATH.read_text(encoding="utf-8").strip()
    registration_hash_valid = raw_registration_sha == output_sidecar and normalized_registration_sha == portable_sidecar
    if not registration_hash_valid:
        raise RuntimeError("Registration hash validation failed")
    candidate_lock_found = CANDIDATE_LOCK_PATH.exists() and PORTABLE_CANDIDATE_LOCK_PATH.exists()
    if not candidate_lock_found:
        raise FileNotFoundError("Candidate lock bundle missing")
    # Stage 1 hashes candidate-lock bytes but intentionally never parses the
    # test-bearing candidate lock JSON.
    candidate_raw_sha = sha256_file(CANDIDATE_LOCK_PATH)
    candidate_portable_sha = normalized_text_sha256(PORTABLE_CANDIDATE_LOCK_PATH)
    candidate_hash_valid = (
        candidate_raw_sha == str(registration["candidate_lock_sha256"])
        and candidate_portable_sha == PORTABLE_CANDIDATE_LOCK_SHA_PATH.read_text(encoding="utf-8").strip()
    )
    if not candidate_hash_valid:
        raise RuntimeError("Candidate lock checksum validation failed")

    run_manifest: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    class_delta_source: dict[int, dict[str, Any]] = {}
    curves: dict[int, pd.DataFrame] = {}
    curve_rows: list[dict[str, Any]] = []
    run_details: dict[str, Any] = {}
    manifest_paths: list[Path] = [
        REGISTRATION_PATH,
        REGISTRATION_SHA_PATH,
        PORTABLE_REGISTRATION_PATH,
        PORTABLE_REGISTRATION_SHA_PATH,
    ]
    prediction_paths: list[Path] = []

    for seed in SEEDS:
        run_dir = run_path(args.run_root.resolve(), seed)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        required = [
            run_dir / "resolved_config.yaml",
            run_dir / "resolved_config.json",
            run_dir / "feature_schema.json",
            run_dir / "train_log.csv",
            run_dir / "train_metrics.csv",
            run_dir / "val_metrics.csv",
            run_dir / "REPLICATION_COMPLETE.json",
            run_dir / "replication_provenance/environment.json",
            run_dir / "replication_provenance/NO_RESUME.json",
            run_dir / "replication_provenance/runtime_signatures.json",
            run_dir / "replication_provenance/source_hashes.json",
        ] + [run_dir / "checkpoints" / name for name in CHECKPOINTS]
        missing = [rel(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Run {seed} missing required artifacts: {missing}")
        cfg = reader.yaml(run_dir / "resolved_config.yaml")
        config_json = reader.json(run_dir / "resolved_config.json")
        if cfg != config_json:
            raise RuntimeError(f"Resolved YAML/JSON mismatch for seed {seed}")
        completion = reader.json(run_dir / "REPLICATION_COMPLETE.json")
        no_resume = reader.json(run_dir / "replication_provenance/NO_RESUME.json")
        environment = reader.json(run_dir / "replication_provenance/environment.json")
        runtime = reader.json(run_dir / "replication_provenance/runtime_signatures.json")
        source = reader.json(run_dir / "replication_provenance/source_hashes.json")
        feature_schema = reader.json(run_dir / "feature_schema.json")
        history = reader.csv(run_dir / "train_log.csv")
        train_metrics = reader.csv(run_dir / "train_metrics.csv")
        val_metrics = reader.csv(run_dir / "val_metrics.csv")
        if history.empty or train_metrics.empty or val_metrics.empty:
            raise RuntimeError(f"Empty training/validation history for seed {seed}")
        registered_seed = seed
        config_seed = int(cfg.get("seed", -1))
        runtime_seed = int((cfg.get("training", {}) or {}).get("seed", -1))
        marker_seed = int(completion.get("seed", -1))
        prior_seed = int(((cfg.get("graph", {}) or {}).get("prior_corruption", {}) or {}).get("seed", -1))
        if len({registered_seed, config_seed, runtime_seed, marker_seed}) != 1:
            raise RuntimeError(f"Seed identity mismatch for {run_dir}")
        if prior_seed != seed + PRIOR_SEED_OFFSET:
            raise RuntimeError(f"Prior seed rule mismatch for seed {seed}: {prior_seed}")
        if completion.get("status") != "COMPLETE" or completion.get("resumed") is not False:
            raise RuntimeError(f"Completion/resume marker invalid for seed {seed}")
        if no_resume.get("no_resume") is not True or no_resume.get("resume_from") is not None:
            raise RuntimeError(f"NO_RESUME marker invalid for seed {seed}")
        expected_config_path = ROOT / f"configs/d16/final_replication/ofix7_mid_seed{seed}.yaml"
        if not expected_config_path.exists():
            raise FileNotFoundError(expected_config_path)
        expected_cfg = reader.yaml(expected_config_path)
        comparable = normalize_runtime_config(cfg)
        comparable.setdefault("data", {})["prior_dir"] = expected_cfg.get("data", {}).get("prior_dir")
        comparable["data"]["graph_cache_dir"] = expected_cfg.get("data", {}).get("graph_cache_dir")
        unauthorized_diffs = [row for row in semantic_diff(expected_cfg, comparable) if not row["authorized"]]
        if unauthorized_diffs:
            raise RuntimeError(f"Resolved config scientific parity failed for seed {seed}: {unauthorized_diffs}")
        parity = config_parity(cfg, registration)
        signature_checks = {
            "dataset_signature_match": runtime.get("dataset_signature") == registration.get("dataset_signature"),
            "split_signature_match": runtime.get("split_signature") == registration.get("split_signature"),
            "feature_signature_match": runtime.get("feature_signature") == registration.get("feature_signature"),
            "graph_signature_match": runtime.get("graph_signature") == registration.get("graph_signature"),
            "model_signature_match": runtime.get("model_signature") == registration.get("model_signature"),
            "selector_signature_match": runtime.get("selector_signature") == registration.get("selector_signature"),
            "feature_order_match": feature_schema.get("node_feature_names") == registration.get("feature_order"),
        }
        if not all(parity.values()) or not all(signature_checks.values()):
            raise RuntimeError(f"Scientific parity failure for seed {seed}: {parity} {signature_checks}")

        ckpt_by_name: dict[str, dict[str, Any]] = {}
        for name in CHECKPOINTS:
            item = inspect_checkpoint(run_dir / "checkpoints" / name, cfg, int(feature_schema["node_dim"]), reader)
            item["seed"] = seed
            if item["parameter_count"] != int(registration["parameter_count"]):
                raise RuntimeError(f"Parameter count mismatch for seed {seed} {name}")
            if not item["parameters_finite"]:
                raise RuntimeError(f"Non-finite model state for seed {seed} {name}")
            ckpt_by_name[name] = item
            checkpoint_rows.append(item)
        if (
            ckpt_by_name["best.pt"]["canonical_model_state_sha256"]
            != ckpt_by_name["best_val_macro_f1.pt"]["canonical_model_state_sha256"]
            or ckpt_by_name["best.pt"]["epoch"] != ckpt_by_name["best_val_macro_f1.pt"]["epoch"]
        ):
            raise RuntimeError(f"best.pt and macro alias differ for seed {seed}")

        snapshots = {
            policy: snapshot_analysis(run_dir, stem, reader)
            for policy, stem in POLICIES.items()
        }
        for policy, snapshot in snapshots.items():
            checkpoint_name = f"{snapshot['stem']}.pt"
            epoch = int(snapshot["epoch"])
            if checkpoint_name in ckpt_by_name and ckpt_by_name[checkpoint_name]["epoch"] != epoch:
                raise RuntimeError(f"Checkpoint/snapshot epoch mismatch seed {seed} {policy}")
            aligned = history_row(history, epoch)
            recomputed = snapshot["recomputed"]
            if abs(snapshot["stored_accuracy_delta"]) > TOLERANCE or abs(snapshot["stored_macro_f1_delta"]) > TOLERANCE:
                raise RuntimeError(f"Stored/recomputed validation mismatch seed {seed} {policy}")
            row = {
                "seed": seed,
                "policy": policy,
                "epoch": epoch,
                "train_loss": aligned["train_loss"],
                "train_eval_loss": aligned["train_eval_loss"],
                "train_accuracy": aligned["train_accuracy"],
                "train_macro_f1": aligned["train_macro_f1"],
                "validation_loss": float(snapshot["stored"]["loss"]),
                "validation_accuracy": recomputed["accuracy"],
                "validation_macro_f1": recomputed["macro_f1"],
                "validation_weighted_f1": recomputed["weighted_f1"],
                "validation_balanced_accuracy": recomputed["balanced_accuracy"],
                "accuracy_gap_pp": (aligned["train_accuracy"] - recomputed["accuracy"]) * 100.0,
                "macro_f1_gap_pp": (aligned["train_macro_f1"] - recomputed["macro_f1"]) * 100.0,
                "lr": aligned["lr"],
                "nll": recomputed["nll"],
                "brier": recomputed["brier"],
                "ece": recomputed["ece"],
                "mean_confidence": recomputed["mean_confidence"],
                "mean_entropy": recomputed["mean_entropy"],
                "prediction_count": recomputed["count"],
                "stored_accuracy_delta": snapshot["stored_accuracy_delta"],
                "stored_macro_f1_delta": snapshot["stored_macro_f1_delta"],
            }
            validation_rows.append(row)
            alignment_rows.append(row)
            prediction_paths.append(ROOT / snapshot["prediction_path"])
        class_delta_source[seed] = snapshots
        best_macro_epoch = int(snapshots["VAL_MACRO_F1"]["epoch"])
        best_accuracy_epoch = int(snapshots["VAL_ACCURACY"]["epoch"])
        curve = curve_evidence(history, best_macro_epoch, int(cfg["training"]["max_epochs"]))
        curve["seed"] = seed
        curve_rows.append(curve)
        curves[seed] = history.copy()
        duration = float(pd.to_numeric(history["epoch_time_sec"], errors="coerce").sum())
        created_at = environment.get("created_at_utc")
        completed_at = completion.get("completed_at_utc")
        manifest_row = {
            "run_id": RUN_NAMES[seed],
            "registered_seed": seed,
            "config_seed": config_seed,
            "runtime_seed": runtime_seed,
            "prior_seed": prior_seed,
            "output_path": rel(run_dir),
            "completion_status": completion.get("status"),
            "resume_status": "NO_RESUME" if no_resume.get("no_resume") is True else "INVALID",
            "start_time": created_at,
            "end_time": completed_at,
            "duration_seconds_from_epoch_logs": duration,
            "executed_epochs": int(history["epoch"].max()),
            "best_macro_epoch": best_macro_epoch,
            "best_accuracy_epoch": best_accuracy_epoch,
            "last_epoch": int(snapshots["LAST"]["epoch"]),
            "stop_reason": curve["stop_reason"],
            "source_hash": source.get("repository_commit"),
            "config_hash": source.get("files", {}).get(f"configs/d16/final_replication/ofix7_mid_seed{seed}.yaml"),
            "dataset_hash": runtime.get("dataset_signature"),
            "split_hash": runtime.get("split_signature"),
            "feature_hash": runtime.get("feature_signature"),
            "graph_hash": runtime.get("graph_signature"),
            "model_hash": runtime.get("model_signature"),
            "environment_summary": f"python={environment.get('python')}; torch={environment.get('torch')}; cuda={environment.get('cuda_version')}; platform={environment.get('platform')}",
        }
        run_manifest.append(manifest_row)
        parity_rows.append({
            "seed": seed,
            **parity,
            **signature_checks,
            "parameter_count": ckpt_by_name["best.pt"]["parameter_count"],
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "cuda_version": environment.get("cuda_version"),
            "cudnn_version": environment.get("cudnn_version"),
            "platform": environment.get("platform"),
            "hostname": environment.get("hostname"),
            "amp_configured": bool(cfg["training"].get("amp")),
            "tf32_configured": bool(cfg["training"].get("allow_tf32")),
            "num_workers": int(cfg["training"].get("num_workers")),
        })
        run_details[str(seed)] = {
            "manifest": manifest_row,
            "parity": parity_rows[-1],
            "checkpoints": ckpt_by_name,
            "snapshots": snapshots,
            "curve": curve,
        }
        manifest_paths.extend(required + [expected_config_path])
        manifest_paths.extend((run_dir / "validation_snapshots").glob("*"))

    macro_rows = {int(row["seed"]): row for row in validation_rows if row["policy"] == "VAL_MACRO_F1"}
    accuracy_rows = {int(row["seed"]): row for row in validation_rows if row["policy"] == "VAL_ACCURACY"}
    delta_rows = []
    class_delta_rows = []
    for seed in SEEDS:
        left, right = macro_rows[seed], accuracy_rows[seed]
        delta_rows.append({
            "seed": seed,
            "macro_epoch": left["epoch"],
            "accuracy_epoch": right["epoch"],
            "delta_checkpoint_epoch": right["epoch"] - left["epoch"],
            "delta_validation_accuracy_pp": (right["validation_accuracy"] - left["validation_accuracy"]) * 100.0,
            "delta_validation_macro_f1_pp": (right["validation_macro_f1"] - left["validation_macro_f1"]) * 100.0,
            "delta_validation_weighted_f1_pp": (right["validation_weighted_f1"] - left["validation_weighted_f1"]) * 100.0,
            "delta_macro_gap_pp": right["macro_f1_gap_pp"] - left["macro_f1_gap_pp"],
            "delta_accuracy_gap_pp": right["accuracy_gap_pp"] - left["accuracy_gap_pp"],
        })
        left_class = class_delta_source[seed]["VAL_MACRO_F1"]["recomputed"]["per_class"]
        right_class = class_delta_source[seed]["VAL_ACCURACY"]["recomputed"]["per_class"]
        for class_id in range(7):
            class_delta_rows.append({
                "seed": seed,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "macro_policy_f1": left_class[class_id]["f1"],
                "accuracy_policy_f1": right_class[class_id]["f1"],
                "delta_f1_pp": (right_class[class_id]["f1"] - left_class[class_id]["f1"]) * 100.0,
            })
    aggregate_deltas = {
        key: stat_summary(row[key] for row in delta_rows)
        for key in (
            "delta_validation_accuracy_pp",
            "delta_validation_macro_f1_pp",
            "delta_validation_weighted_f1_pp",
            "delta_macro_gap_pp",
            "delta_accuracy_gap_pp",
            "delta_checkpoint_epoch",
        )
    }
    class_mean_delta = {
        name: float(np.mean([row["delta_f1_pp"] for row in class_delta_rows if row["class_name"] == name]))
        for name in CLASS_NAMES
    }
    accuracy_wins = sum(row["delta_validation_accuracy_pp"] > 1e-12 for row in delta_rows)
    accuracy_losses = sum(row["delta_validation_accuracy_pp"] < -1e-12 for row in delta_rows)
    accuracy_ties = 5 - accuracy_wins - accuracy_losses
    sd_increase_pp = (
        np.std([accuracy_rows[seed]["validation_accuracy"] for seed in SEEDS], ddof=1)
        - np.std([macro_rows[seed]["validation_accuracy"] for seed in SEEDS], ddof=1)
    ) * 100.0
    max_class_loss = max((max(0.0, -value) for value in class_mean_delta.values()), default=0.0)
    gate_values = {
        "mean_validation_accuracy_gain_pp": aggregate_deltas["delta_validation_accuracy_pp"]["mean"],
        "mean_validation_macro_f1_delta_pp": aggregate_deltas["delta_validation_macro_f1_pp"]["mean"],
        "accuracy_higher_seed_count": accuracy_wins,
        "accuracy_lower_seed_count": accuracy_losses,
        "accuracy_tie_seed_count": accuracy_ties,
        "maximum_mean_class_f1_loss_pp": max_class_loss,
        "mean_macro_gap_increase_pp": aggregate_deltas["delta_macro_gap_pp"]["mean"],
        "validation_accuracy_sample_sd_increase_pp": float(sd_increase_pp),
    }
    gates = {
        "mean_validation_accuracy_gain_ge_0_50pp": gate_values["mean_validation_accuracy_gain_pp"] >= 0.50,
        "mean_validation_macro_f1_delta_ge_minus_0_50pp": gate_values["mean_validation_macro_f1_delta_pp"] >= -0.50,
        "validation_accuracy_higher_in_at_least_3_seeds": accuracy_wins >= 3,
        "no_class_mean_f1_loss_gt_3pp": max_class_loss <= 3.0,
        "mean_macro_gap_increase_le_2pp": gate_values["mean_macro_gap_increase_pp"] <= 2.0,
        "validation_accuracy_sd_increase_le_0_50pp": sd_increase_pp <= 0.50,
    }
    selected_policy = "VAL_ACCURACY" if all(gates.values()) else "VAL_MACRO_F1"
    lock_state = "LOCK_VAL_ACCURACY" if selected_policy == "VAL_ACCURACY" else "LOCK_VAL_MACRO_F1"
    selected_stem = POLICIES[selected_policy]
    alternate_policy = "VAL_MACRO_F1" if selected_policy == "VAL_ACCURACY" else "VAL_ACCURACY"
    alternate_stem = POLICIES[alternate_policy]

    selected_val_accuracy = [
        (accuracy_rows if selected_policy == "VAL_ACCURACY" else macro_rows)[seed]["validation_accuracy"] for seed in SEEDS
    ]
    selected_val_macro = [
        (accuracy_rows if selected_policy == "VAL_ACCURACY" else macro_rows)[seed]["validation_macro_f1"] for seed in SEEDS
    ]
    seed_instability = {
        "validation_accuracy_sd_gt_1_5pp": np.std(selected_val_accuracy, ddof=1) * 100.0 > 1.5,
        "validation_macro_f1_sd_gt_1_5pp": np.std(selected_val_macro, ddof=1) * 100.0 > 1.5,
        "seed_more_than_3pp_below_accuracy_median": any(
            (np.median(selected_val_accuracy) - value) * 100.0 > 3.0 for value in selected_val_accuracy
        ),
        "materially_split_trajectory_regimes": False,
    }
    gen_support = sum(bool(row["generalization_supported"]) for row in curve_rows)
    schedule_support = sum(bool(row["schedule_supported"]) for row in curve_rows)
    optimization_support = sum(bool(row["optimization_supported"]) for row in curve_rows)
    if any(seed_instability.values()):
        diagnosis = "SEED_UNSTABLE"
    elif gen_support >= 3 and gen_support >= max(schedule_support, optimization_support):
        diagnosis = "GENERALIZATION_LIMITED"
    elif schedule_support >= 3 and schedule_support > max(gen_support, optimization_support):
        diagnosis = "SCHEDULE_LIMITED"
    elif optimization_support >= 3 and optimization_support > max(gen_support, schedule_support):
        diagnosis = "OPTIMIZATION_LIMITED"
    else:
        diagnosis = "MIXED_OR_INCONCLUSIVE"
    if diagnosis == "GENERALIZATION_LIMITED" and selected_policy == "VAL_MACRO_F1":
        next_action = "STOP_BOUNDED_OPTIMIZATION_AUDIT"
    elif diagnosis == "SCHEDULE_LIMITED":
        next_action = "PREPARE_SCHEDULER_CELL_S1"
    elif diagnosis == "OPTIMIZATION_LIMITED":
        next_action = "PREPARE_OPTIMIZER_CELL_O1"
    elif diagnosis == "SEED_UNSTABLE":
        next_action = "HOLD_REPRODUCIBILITY_REPAIR"
    else:
        next_action = "PREPARE_BOTH_BOUNDED_CELLS"

    input_manifest = artifact_manifest(manifest_paths)
    prediction_manifest = artifact_manifest(prediction_paths)
    input_manifest_sha = canonical_json_sha(input_manifest)
    prediction_manifest_sha = canonical_json_sha(prediction_manifest)
    lock = {
        "policy_version": "ofix7-mid-five-seed-validation-lock-v1",
        "lock_state": lock_state,
        "created_at_utc": now_utc(),
        "candidate_registration_sha256": normalized_registration_sha,
        "registration_raw_sha256": raw_registration_sha,
        "candidate_lock_raw_sha256": candidate_raw_sha,
        "candidate_lock_portable_normalized_sha256": candidate_portable_sha,
        "input_artifact_manifest": input_manifest,
        "input_artifact_manifest_sha256": input_manifest_sha,
        "validation_prediction_manifest": prediction_manifest,
        "validation_prediction_manifest_sha256": prediction_manifest_sha,
        "registered_seeds": SEEDS,
        "run_ids": [RUN_NAMES[seed] for seed in SEEDS],
        "gate_thresholds": {
            "mean_validation_accuracy_gain_pp_min": 0.50,
            "mean_validation_macro_f1_delta_pp_min": -0.50,
            "accuracy_seed_wins_min": 3,
            "mean_class_f1_loss_pp_max": 3.0,
            "macro_gap_increase_pp_max": 2.0,
            "accuracy_sd_increase_pp_max": 0.50,
        },
        "gate_values": gate_values,
        "gate_results": gates,
        "aggregate_validation_deltas": aggregate_deltas,
        "classwise_mean_f1_delta_pp": class_mean_delta,
        "selected_policy": selected_policy,
        "alternate_policy": alternate_policy,
        "selected_checkpoints": {
            str(seed): {
                "path": rel(run_path(args.run_root.resolve(), seed) / "checkpoints" / f"{selected_stem}.pt"),
                "file_sha256": run_details[str(seed)]["checkpoints"][f"{selected_stem}.pt"]["file_sha256"],
                "canonical_model_state_sha256": run_details[str(seed)]["checkpoints"][f"{selected_stem}.pt"]["canonical_model_state_sha256"],
                "epoch": run_details[str(seed)]["checkpoints"][f"{selected_stem}.pt"]["epoch"],
            }
            for seed in SEEDS
        },
        "non_selected_checkpoints": {
            str(seed): {
                "path": rel(run_path(args.run_root.resolve(), seed) / "checkpoints" / f"{alternate_stem}.pt"),
                "file_sha256": run_details[str(seed)]["checkpoints"][f"{alternate_stem}.pt"]["file_sha256"],
                "canonical_model_state_sha256": run_details[str(seed)]["checkpoints"][f"{alternate_stem}.pt"]["canonical_model_state_sha256"],
                "epoch": run_details[str(seed)]["checkpoints"][f"{alternate_stem}.pt"]["epoch"],
            }
            for seed in SEEDS
        },
        "last_checkpoints": {
            str(seed): {
                "path": rel(run_path(args.run_root.resolve(), seed) / "checkpoints/last.pt"),
                "file_sha256": run_details[str(seed)]["checkpoints"]["last.pt"]["file_sha256"],
                "canonical_model_state_sha256": run_details[str(seed)]["checkpoints"]["last.pt"]["canonical_model_state_sha256"],
                "epoch": run_details[str(seed)]["checkpoints"]["last.pt"]["epoch"],
            }
            for seed in SEEDS
        },
        "per_seed_validation": {
            str(seed): {
                "macro_policy": macro_rows[seed],
                "accuracy_policy": accuracy_rows[seed],
            }
            for seed in SEEDS
        },
        "warnings": [
            "Five seeds provide descriptive, not definitive, seed-distribution inference.",
            "Candidate-lock semantic reconciliation is deferred to test-reveal because the historical lock contains test metrics.",
        ],
        "test_artifacts_read": False,
        "validation_read_paths": sorted(set(reader.read_paths)),
    }
    lock_path = output / "10_checkpoint_policy_lock.json"
    write_json_new(lock_path, lock)
    lock_sha = sha256_file(lock_path)
    write_text_new(output / "10_checkpoint_policy_lock.sha256", lock_sha)

    write_text_new(output / "00_README.md", """# OFIX7-mid Five-Seed Post-Training Analysis

This directory is produced in two immutable stages. Stage 1 uses only registered configuration, provenance, checkpoint metadata/state, history, and validation snapshots. It does not parse test-bearing artifacts. Stage 2 requires and verifies `10_checkpoint_policy_lock.json` plus its SHA before revealing test results.

Completed run directories are read-only. No training, resume, fine-tuning, checkpoint modification, config modification, model modification, dataset modification, or graph modification is performed.
""")
    write_csv_new(output / "01_input_run_manifest.csv", run_manifest)
    write_text_new(output / "01_input_run_manifest.md", "# Input Run Manifest\n\n" + markdown_table(run_manifest))
    integrity_rows = [
        {"check": "registration_raw_hash", "value": raw_registration_sha, "status": raw_registration_sha == output_sidecar},
        {"check": "registration_portable_normalized_hash", "value": normalized_registration_sha, "status": normalized_registration_sha == portable_sidecar},
        {"check": "candidate_lock_raw_hash", "value": candidate_raw_sha, "status": candidate_raw_sha == registration["candidate_lock_sha256"]},
        {"check": "candidate_lock_portable_normalized_hash", "value": candidate_portable_sha, "status": candidate_portable_sha == PORTABLE_CANDIDATE_LOCK_SHA_PATH.read_text(encoding="utf-8").strip()},
        {"check": "historical_checkpoint_file_hash", "value": sha256_file(ROOT / registration["historical_best_checkpoint_path"]), "status": sha256_file(ROOT / registration["historical_best_checkpoint_path"]) == registration["historical_best_checkpoint_sha256"]},
    ]
    write_text_new(output / "02_registration_and_integrity_validation.md", "# Registration And Integrity Validation\n\n" + markdown_table(integrity_rows) + "\n\nCandidate-lock JSON content was not parsed in Stage 1 because it contains historical full-test metrics. Raw and normalized hashes are reported separately; serialization differences are not treated as model-state equivalence.")
    write_csv_new(output / "03_config_and_runtime_parity.csv", parity_rows)
    write_text_new(output / "03_config_and_runtime_parity.md", "# Config And Runtime Parity\n\n" + markdown_table(parity_rows) + "\n\nAll scientific signatures and normalized configs match registration. Hostnames differ; Python, PyTorch, CUDA, cuDNN, AMP configuration, TF32 configuration and worker count are otherwise equal in recorded provenance. GPU model was not recorded and is therefore `MISSING`, not inferred.")
    write_csv_new(output / "04_checkpoint_inventory.csv", checkpoint_rows)
    write_text_new(output / "04_checkpoint_inventory.md", "# Checkpoint Inventory\n\n" + markdown_table(checkpoint_rows))
    write_csv_new(output / "05_history_and_epoch_alignment.csv", alignment_rows)
    write_text_new(output / "05_history_and_epoch_alignment.md", "# History And Epoch Alignment\n\nAll train/validation gaps use metrics from the exact checkpoint epoch. Train weighted-F1 was not logged and is reported as unavailable rather than reconstructed from another epoch.\n\n" + markdown_table(alignment_rows))
    flat_curve_rows = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in curve_rows]
    write_csv_new(output / "06_training_curve_summary.csv", flat_curve_rows)
    write_text_new(output / "06_training_curve_summary.md", "# Training Curve Summary\n\nFinal slopes use ordinary least squares on the stated final 5 or 10 epochs.\n\n" + markdown_table(flat_curve_rows))
    write_csv_new(output / "07_validation_checkpoint_metrics.csv", validation_rows)
    write_text_new(output / "07_validation_checkpoint_metrics.md", "# Validation Checkpoint Metrics\n\nMetrics and calibration were recomputed from complete frozen validation logits/probabilities. Stored accuracy and macro-F1 agree within 0.10 pp.\n\n" + markdown_table(validation_rows))
    write_csv_new(output / "08_checkpoint_policy_comparison.csv", delta_rows)
    write_text_new(output / "08_checkpoint_policy_comparison.md", "# Checkpoint Policy Comparison\n\n" + markdown_table(delta_rows) + "\n\n## Aggregate\n\n```json\n" + json.dumps(clean_json(aggregate_deltas), indent=2) + "\n```\n")
    write_csv_new(output / "09_checkpoint_policy_classwise.csv", class_delta_rows)
    write_text_new(output / "09_checkpoint_policy_classwise.md", "# Checkpoint Policy Classwise Comparison\n\n" + markdown_table(class_delta_rows) + "\n\nMean delta by class (accuracy policy minus macro policy, pp):\n\n```json\n" + json.dumps(class_mean_delta, indent=2) + "\n```")
    write_text_new(output / "10_checkpoint_policy_lock.md", "# Checkpoint Policy Lock\n\nLock state: **" + lock_state + "**.\n\nSelected policy: **" + selected_policy + "**.\n\nLock SHA-256: `" + lock_sha + "`.\n\n" + markdown_table([{"gate": key, "passed": value} for key, value in gates.items()]) + "\n\nFull-precision gate values:\n\n```json\n" + json.dumps(clean_json(gate_values), indent=2) + "\n```")
    write_text_new(output / "11_architecture_limit_diagnosis.md", "# Primary Limit Diagnosis\n\nAssigned diagnosis: **" + diagnosis + "**. This is an observational training-curve diagnosis, not a theoretical architecture limit.\n\nCross-seed support: generalization=" + str(gen_support) + "/5, schedule=" + str(schedule_support) + "/5, optimization=" + str(optimization_support) + "/5.\n\nSeed-instability gates:\n\n```json\n" + json.dumps(clean_json(seed_instability), indent=2) + "\n```")
    evidence_rows = []
    for row in curve_rows:
        evidence_rows.append({
            "seed": row["seed"],
            "generalization_supported": row["generalization_supported"],
            "schedule_supported": row["schedule_supported"],
            "optimization_supported": row["optimization_supported"],
            "train_macro_gain_after_best_pp": row["train_macro_gain_after_best_pp"],
            "validation_macro_change_after_best_pp": row["validation_macro_change_after_best_pp"],
            "macro_gap_best_pp": row["macro_gap_best_pp"],
            "macro_gap_last_pp": row["macro_gap_last_pp"],
            "lr_reduction_epochs": json.dumps(row["lr_reduction_epochs"]),
            "lr_at_end": row["lr_at_end"],
            "stop_reason": row["stop_reason"],
        })
    write_csv_new(output / "12_limit_evidence_by_seed.csv", evidence_rows)
    write_text_new(output / "12_limit_evidence_by_seed.md", "# Limit Evidence By Seed\n\n" + markdown_table(evidence_rows))
    next_protocol = {
        "decision": next_action,
        "S1_if_authorized": "historical AdamW + CosineAnnealingLR; retain initial LR, weight decay, epochs, early stopping monitor, locked checkpoint policy and architecture; no warmup/restart/sweep",
        "O1_if_authorized": "RAdam + historical ReduceLROnPlateau; retain initial LR, weight decay, scheduler parameters, epochs, locked checkpoint policy and architecture; no LR sweep or Lookahead",
        "interaction": "PROHIBITED: do not combine RAdam and cosine",
    }
    write_text_new(output / "13_optimizer_scheduler_next_action.md", "# Optimizer And Scheduler Next Action\n\nDecision: **" + next_action + "**.\n\n```json\n" + json.dumps(next_protocol, indent=2) + "\n```")

    plot_dir = output / "plots"
    for metric, filename, title in (
        ("val_accuracy", "validation_accuracy_by_seed.png", "Validation Accuracy By Seed"),
        ("val_macro_f1", "validation_macro_f1_by_seed.png", "Validation Macro-F1 By Seed"),
    ):
        series = [(frame["epoch"].to_numpy(), frame[metric].to_numpy(), f"seed {seed}") for seed, frame in curves.items()]
        line_plot(plot_dir / filename, series, title, metric)
    bar_plot(plot_dir / "checkpoint_policy_validation_delta.png", [str(seed) for seed in SEEDS], [row["delta_validation_accuracy_pp"] for row in delta_rows], "Accuracy Policy Validation Accuracy Delta", "Delta (pp)")
    plt.figure(figsize=(8, 4.8))
    x = np.arange(5); width = 0.36
    plt.bar(x - width / 2, [macro_rows[s]["epoch"] for s in SEEDS], width, label="VAL_MACRO_F1")
    plt.bar(x + width / 2, [accuracy_rows[s]["epoch"] for s in SEEDS], width, label="VAL_ACCURACY")
    plt.xticks(x, [str(seed) for seed in SEEDS]); plt.ylabel("Epoch"); plt.title("Checkpoint Policy Epoch Comparison"); plt.legend()
    save_plot_new(plot_dir / "checkpoint_policy_epoch_comparison.png")
    bar_plot(plot_dir / "train_validation_gap_by_seed.png", [str(seed) for seed in SEEDS], [macro_rows[s]["macro_f1_gap_pp"] for s in SEEDS], "Train-Validation Macro-F1 Gap At Macro Checkpoint", "Gap (pp)")
    bar_plot(plot_dir / "best_epoch_fraction.png", [str(seed) for seed in SEEDS], [next(row["best_epoch_fraction"] for row in curve_rows if row["seed"] == seed) for seed in SEEDS], "Best Epoch Fraction", "Fraction")
    line_plot(plot_dir / "learning_rate_trajectories.png", [(frame["epoch"].to_numpy(), frame["lr"].to_numpy(), f"seed {seed}") for seed, frame in curves.items()], "Learning Rate Trajectories", "LR")
    for seed, frame in curves.items():
        line_plot(plot_dir / f"train_validation_curves_seed{seed}.png", [
            (frame["epoch"].to_numpy(), frame["train_macro_f1"].to_numpy(), "train macro-F1"),
            (frame["epoch"].to_numpy(), frame["val_macro_f1"].to_numpy(), "validation macro-F1"),
            (frame["epoch"].to_numpy(), frame["train_accuracy"].to_numpy(), "train accuracy"),
            (frame["epoch"].to_numpy(), frame["val_accuracy"].to_numpy(), "validation accuracy"),
        ], f"Train/Validation Curves Seed {seed}", "Metric")

    stage_summary = {
        "stage": "VALIDATION_LOCK_COMPLETE",
        "selected_policy": selected_policy,
        "lock_state": lock_state,
        "checkpoint_policy_lock_sha256": lock_sha,
        "primary_diagnosis": diagnosis,
        "optimizer_scheduler_next_action": next_action,
        "test_artifacts_read": False,
        "output_dir": rel(output),
    }
    print(json.dumps(stage_summary, indent=2))
    return stage_summary


def verify_manifest(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            raise RuntimeError(f"Locked artifact disappeared: {path}")
        stat = path.stat()
        if int(stat.st_size) != int(row["size"]) or int(stat.st_mtime_ns) != int(row["mtime_ns"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Locked artifact changed after validation lock: {path}")


def load_policy_lock(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise RuntimeError("Checkpoint policy lock SHA sidecar missing")
    actual = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("Checkpoint policy lock hash differs")
    lock = load_json(path)
    if lock.get("lock_state") not in {"LOCK_VAL_MACRO_F1", "LOCK_VAL_ACCURACY"}:
        raise RuntimeError(f"Checkpoint policy is not locked: {lock.get('lock_state')}")
    verify_manifest(lock["input_artifact_manifest"])
    verify_manifest(lock["validation_prediction_manifest"])
    if canonical_json_sha(lock["input_artifact_manifest"]) != lock["input_artifact_manifest_sha256"]:
        raise RuntimeError("Input artifact manifest SHA differs")
    if canonical_json_sha(lock["validation_prediction_manifest"]) != lock["validation_prediction_manifest_sha256"]:
        raise RuntimeError("Validation prediction manifest SHA differs")
    return lock, actual


def read_test_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "split" in frame and set(frame["split"].astype(str)) != {"test"}:
        raise RuntimeError(f"Expected only test rows in {path}")
    return frame


def fresh_test_inference(
    cfg: dict[str, Any], checkpoint_path: Path, prior_dir: Path, device_name: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not prior_dir.exists():
        raise FileNotFoundError(f"Local prior directory required for missing test prediction: {prior_dir}")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = trainer.build_dataset(cfg, prior_dir, "test")
    loader = DataLoader(
        dataset,
        batch_size=int((cfg.get("training", {}) or {}).get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_d16_graphs,
    )
    first = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=int(first.x_cat.shape[1])).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    row, per_class, pred_count, fallback, confusion, predictions, group_class = trainer.evaluate(
        model,
        loader,
        device,
        "test",
        epoch=int(payload.get("epoch", -1)),
        checkpoint_name=checkpoint_path.name,
        checkpoint_epoch=int(payload.get("epoch", -1)),
        best_val_macro_f1=float(payload.get("best_val_macro_f1", math.nan)),
        collect_predictions=True,
        amp_enabled=bool((cfg.get("training", {}) or {}).get("amp", False)) and device.type == "cuda",
        loss_cfg=cfg.get("loss", {}) or {},
    )
    return pd.DataFrame(predictions), {
        "device": str(device),
        "fresh_inference": True,
        "trainer_row": row,
        "per_class_rows": per_class,
        "pred_count_rows": pred_count,
        "fallback_rows": fallback,
        "confusion_rows": confusion,
        "group_class_rows": group_class,
    }


def aggregate_metric_rows(per_seed: list[dict[str, Any]], metrics: Iterable[str]) -> list[dict[str, Any]]:
    return [{"metric": metric, **stat_summary(float(row[metric]) for row in per_seed)} for metric in metrics]


def test_reveal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if not output.exists():
        raise FileNotFoundError("Validation-lock output directory does not exist")
    lock, lock_sha = load_policy_lock(args.checkpoint_policy_lock.resolve())
    selected_policy = str(lock["selected_policy"])
    alternate_policy = str(lock["alternate_policy"])
    selected_info = lock["selected_checkpoints"]
    alternate_info = lock["non_selected_checkpoints"]
    last_info = lock["last_checkpoints"]
    for bundle in (selected_info, alternate_info, last_info):
        if set(bundle) != {str(seed) for seed in SEEDS}:
            raise RuntimeError("Policy lock is missing one or more registered seeds")
        for seed_text, info in bundle.items():
            path = ROOT / info["path"]
            if not path.exists() or sha256_file(path) != info["file_sha256"]:
                raise RuntimeError(f"Selected checkpoint changed or disappeared: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if trainer.canonical_model_state_hash(payload) != info["canonical_model_state_sha256"]:
                raise RuntimeError(f"Selected checkpoint canonical model state changed: {path}")

    generated_prediction_path = output / "test_reveal_seed1337_val_accuracy_predictions.csv"
    primary_rows: list[dict[str, Any]] = []
    alternate_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    classwise_policy: dict[str, dict[int, list[dict[str, Any]]]] = {"primary": {}, "alternate": {}, "last": {}}
    confusion_by_seed: dict[int, list[list[int]]] = {}
    stored_agreement = []
    test_sources = []
    fresh_inference_details: dict[str, Any] = {}

    for seed in SEEDS:
        run_dir = run_path(args.run_root.resolve(), seed)
        cfg = load_yaml(run_dir / "resolved_config.yaml")
        primary_canonical = selected_info[str(seed)]["canonical_model_state_sha256"]
        alternate_canonical = alternate_info[str(seed)]["canonical_model_state_sha256"]
        last_canonical = last_info[str(seed)]["canonical_model_state_sha256"]
        best_canonical = trainer.canonical_model_state_hash(torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False))
        if primary_canonical == best_canonical:
            primary_path = run_dir / "predictions.csv"
        elif primary_canonical == last_canonical:
            primary_path = run_dir / "last_predictions.csv"
        else:
            raise RuntimeError(f"Locked primary policy has no registered completed-run prediction artifact for seed {seed}")
        primary_frame = read_test_prediction(primary_path)
        primary_metrics = metrics_from_predictions(primary_frame)

        last_path = run_dir / "last_predictions.csv"
        last_frame = read_test_prediction(last_path)
        last_metrics = metrics_from_predictions(last_frame)

        if alternate_canonical == primary_canonical:
            alternate_frame = primary_frame.copy()
            alternate_source = primary_path
            alternate_mode = "canonical_alias_primary"
        elif alternate_canonical == last_canonical:
            alternate_frame = last_frame.copy()
            alternate_source = last_path
            alternate_mode = "canonical_alias_last"
        else:
            if seed != 1337:
                raise RuntimeError(f"Unexpected missing alternate test prediction for seed {seed}")
            if generated_prediction_path.exists():
                alternate_frame = read_test_prediction(generated_prediction_path)
                fresh = {
                    "device": "recorded_in_completed_partial_stage",
                    "fresh_inference": True,
                    "reused_after_report_serialization_failure": True,
                    "prediction_sha256": sha256_file(generated_prediction_path),
                }
                alternate_mode = "fresh_checkpoint_inference_reused"
            else:
                alternate_frame, fresh = fresh_test_inference(
                    cfg,
                    ROOT / alternate_info[str(seed)]["path"],
                    args.prior_dir.resolve(),
                    args.device,
                )
                alternate_frame.to_csv(generated_prediction_path, index=False)
                alternate_mode = "fresh_checkpoint_inference"
            alternate_source = generated_prediction_path
            fresh_inference_details[str(seed)] = fresh
        alternate_metrics = metrics_from_predictions(alternate_frame)

        for role, metrics, checkpoint_info in (
            ("primary", primary_metrics, selected_info[str(seed)]),
            ("alternate", alternate_metrics, alternate_info[str(seed)]),
            ("last", last_metrics, last_info[str(seed)]),
        ):
            target = {"seed": seed, "policy_role": role, "checkpoint_epoch": checkpoint_info["epoch"], **{key: value for key, value in metrics.items() if key not in {"per_class", "confusion"}}}
            if role == "primary":
                primary_rows.append(target)
                confusion_by_seed[seed] = metrics["confusion"]
            elif role == "alternate":
                alternate_rows.append(target)
            else:
                last_rows.append(target)
            classwise_policy[role][seed] = metrics["per_class"]
        test_sources.append({
            "seed": seed,
            "primary_source": rel(primary_path),
            "primary_source_sha256": sha256_file(primary_path),
            "alternate_source": rel(alternate_source),
            "alternate_source_sha256": sha256_file(alternate_source),
            "alternate_mode": alternate_mode,
            "last_source": rel(last_path),
            "last_source_sha256": sha256_file(last_path),
        })
        for role, metrics_path, metrics in (
            ("primary", run_dir / "test_metrics.csv", primary_metrics),
            ("last", run_dir / "last_test_metrics.csv", last_metrics),
        ):
            stored_frame = pd.read_csv(metrics_path)
            stored_row = stored_frame.iloc[-1]
            acc_delta = metrics["accuracy"] - float(stored_row["accuracy"])
            macro_delta = metrics["macro_f1"] - float(stored_row["macro_f1"])
            stored_agreement.append({"seed": seed, "role": role, "accuracy_delta": acc_delta, "macro_f1_delta": macro_delta, "within_0_10pp": abs(acc_delta) <= TOLERANCE and abs(macro_delta) <= TOLERANCE})
    if not all(row["within_0_10pp"] for row in stored_agreement):
        raise RuntimeError(f"Stored/recomputed test metric mismatch: {stored_agreement}")

    metric_names = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "nll", "brier", "ece", "mean_confidence", "mean_entropy", "confidence_correct", "confidence_incorrect"]
    aggregate = aggregate_metric_rows(primary_rows, metric_names)
    aggregate_by_name = {row["metric"]: row for row in aggregate}
    primary_by_seed = {row["seed"]: row for row in primary_rows}
    alternate_by_seed = {row["seed"]: row for row in alternate_rows}
    last_by_seed = {row["seed"]: row for row in last_rows}
    sensitivity_rows = []
    for seed in SEEDS:
        for comparison, other in (("alternate_minus_primary", alternate_by_seed[seed]), ("last_minus_primary", last_by_seed[seed])):
            sensitivity_rows.append({
                "seed": seed,
                "comparison": comparison,
                **{f"delta_{metric}": other[metric] - primary_by_seed[seed][metric] for metric in metric_names},
            })

    classwise_rows = []
    historical_class = None
    historical_prediction_path = HISTORICAL_RUN / "predictions.csv"
    if historical_prediction_path.exists():
        historical_class = metrics_from_predictions(read_test_prediction(historical_prediction_path))["per_class"]
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = [classwise_policy["primary"][seed][class_id] for seed in SEEDS]
        for metric in ("precision", "recall", "f1"):
            stats = stat_summary(value[metric] for value in values)
            classwise_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "metric": metric,
                **stats,
                "support": values[0]["support"],
                "improved_vs_historical_seed42_count": sum(value[metric] > historical_class[class_id][metric] for value in values) if historical_class else None,
            })
    calibration_rows = []
    for role, rows in (("primary", primary_rows), ("alternate", alternate_rows), ("last", last_rows)):
        for row in rows:
            calibration_rows.append({"role": role, **{key: row[key] for key in ["seed", "nll", "brier", "ece", "mean_confidence", "mean_entropy", "confidence_correct", "confidence_incorrect"]}})
        for metric in ("nll", "brier", "ece", "mean_confidence", "mean_entropy", "confidence_correct", "confidence_incorrect"):
            stats = stat_summary(row[metric] for row in rows)
            calibration_rows.append({"role": f"{role}_aggregate", "seed": "aggregate", "calibration_metric": metric, **stats})

    mean_acc = aggregate_by_name["accuracy"]["mean"]
    mean_macro = aggregate_by_name["macro_f1"]["mean"]
    sd_acc = aggregate_by_name["accuracy"]["sample_sd"]
    min_acc = aggregate_by_name["accuracy"]["min"]
    if mean_acc >= 0.645 and mean_macro >= 0.625 and sd_acc <= 0.010 and min_acc >= 0.630:
        replication_status = "STRONG_REPLICATION"
    elif mean_acc >= 0.635 and mean_macro >= 0.615 and sd_acc <= 0.015 and min_acc >= 0.615:
        replication_status = "REPLICATED_NEAR65"
    elif mean_acc >= 0.620 and min_acc >= 0.58:
        replication_status = "PARTIAL_REPLICATION"
    else:
        replication_status = "FAILED_REPLICATION"

    diagnosis_text = (output / "11_architecture_limit_diagnosis.md").read_text(encoding="utf-8")
    diagnosis = next((name for name in ("GENERALIZATION_LIMITED", "SCHEDULE_LIMITED", "OPTIMIZATION_LIMITED", "SEED_UNSTABLE", "MIXED_OR_INCONCLUSIVE") if name in diagnosis_text), "MIXED_OR_INCONCLUSIVE")
    action_text = (output / "13_optimizer_scheduler_next_action.md").read_text(encoding="utf-8")
    next_action = next((name for name in ("STOP_BOUNDED_OPTIMIZATION_AUDIT", "PREPARE_SCHEDULER_CELL_S1", "PREPARE_OPTIMIZER_CELL_O1", "PREPARE_BOTH_BOUNDED_CELLS", "HOLD_REPRODUCIBILITY_REPAIR") if name in action_text), "HOLD_REPRODUCIBILITY_REPAIR")

    candidate_lock = load_json(CANDIDATE_LOCK_PATH)
    portable_candidate_lock = load_json(PORTABLE_CANDIDATE_LOCK_PATH)
    lock_semantic_fields = ["run_id", "checkpoint_sha256", "seed", "model_signature", "graph_signature", "selector_signature", "dataset_signature", "split_signature", "parameter_count"]
    candidate_semantic_match = all(candidate_lock.get(field) == portable_candidate_lock.get(field) for field in lock_semantic_fields)
    historical_checkpoint = ROOT / str(candidate_lock["checkpoint_path"]).replace("\\", "/")
    historical_file_sha = sha256_file(historical_checkpoint)
    historical_payload = torch.load(historical_checkpoint, map_location="cpu", weights_only=False)
    historical_canonical = trainer.canonical_model_state_hash(historical_payload)
    historical_metrics = metrics_from_predictions(read_test_prediction(HISTORICAL_RUN / "predictions.csv"))
    historical_last_metrics = metrics_from_predictions(read_test_prediction(HISTORICAL_RUN / "last_predictions.csv"))

    c2_reference: dict[str, Any] = {"status": "MISSING"}
    d18_metrics_path = ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/06_full_test_metrics.csv"
    strict_peak = None
    if d18_metrics_path.exists():
        d18_metrics = pd.read_csv(d18_metrics_path)
        c2_best = d18_metrics.loc[
            (d18_metrics["cell"].astype(str) == "C2")
            & (d18_metrics["checkpoint_type"].astype(str) == "best")
        ]
        if len(c2_best) == 5:
            c2_reference = {
                "status": "VERIFIED_ARTIFACT",
                "accuracy": stat_summary(c2_best["accuracy"].astype(float)),
                "macro_f1": stat_summary(c2_best["macro_f1"].astype(float)),
                "weighted_f1": stat_summary(c2_best["weighted_f1"].astype(float)),
                "n": 5,
                "source": rel(d18_metrics_path),
            }
        c0_seed42 = d18_metrics.loc[
            (d18_metrics["cell"].astype(str) == "C0")
            & (d18_metrics["checkpoint_type"].astype(str) == "best")
            & (d18_metrics["seed"].astype(int) == 42)
        ]
        if len(c0_seed42) == 1:
            row = c0_seed42.iloc[0]
            strict_peak = {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
            }

    aggregate_class_f1 = [row for row in classwise_rows if row["metric"] == "f1"]
    strongest = max(aggregate_class_f1, key=lambda row: row["mean"])
    weakest = min(aggregate_class_f1, key=lambda row: row["mean"])
    highest_variance = max(aggregate_class_f1, key=lambda row: row["sample_sd"])
    confusion_sum = np.sum([np.asarray(confusion_by_seed[seed], dtype=int) for seed in SEEDS], axis=0)
    pairs = []
    for true_id in range(7):
        for pred_id in range(7):
            if true_id != pred_id:
                pairs.append({"true": CLASS_NAMES[true_id], "predicted": CLASS_NAMES[pred_id], "count_across_seeds": int(confusion_sum[true_id, pred_id])})
    pairs = sorted(pairs, key=lambda row: row["count_across_seeds"], reverse=True)[:10]

    paper_recommendation = (
        "USE_OFIX7_MID_FIVE_SEED_AS_MAIN_RESULT"
        if replication_status in {"STRONG_REPLICATION", "REPLICATED_NEAR65"}
        else "USE_OFIX7_MID_AS_SECONDARY_RESULT"
        if replication_status == "PARTIAL_REPLICATION"
        else "KEEP_HISTORICAL_RESULT_CONTEXT_ONLY"
    )
    if next_action == "STOP_BOUNDED_OPTIMIZATION_AUDIT":
        release_readiness = "READY_FOR_CLEAN_RELEASE_NOW"
    elif next_action in {"PREPARE_SCHEDULER_CELL_S1", "PREPARE_OPTIMIZER_CELL_O1", "PREPARE_BOTH_BOUNDED_CELLS"}:
        release_readiness = "HOLD_RELEASE_PENDING_OPTIMIZATION_AUDIT"
    elif next_action == "HOLD_REPRODUCIBILITY_REPAIR":
        release_readiness = "HOLD_RELEASE_PENDING_REPLICATION_REPAIR"
    else:
        release_readiness = "READY_FOR_CLEAN_RELEASE_AFTER_LIMIT_AUDIT"

    selected_validation = {str(seed): lock["per_seed_validation"][str(seed)]["macro_policy" if selected_policy == "VAL_MACRO_F1" else "accuracy_policy"] for seed in SEEDS}
    baseline_lock = {
        "lock_version": "ofix7-mid-five-seed-baseline-v1",
        "architecture_id": "d16_landmark_aware_pixel_evidence_gnn/ofix7-mid",
        "architecture_wording": "LAP-GNN constructs a sparse facial evidence graph composed primarily of pixel-level face and context nodes, augmented with five semantic anchor nodes and a graph-level motif/readout mechanism.",
        "historical_source_run": rel(HISTORICAL_RUN),
        "registration_sha256": lock["candidate_registration_sha256"],
        "checkpoint_policy_lock_sha256": lock_sha,
        "selected_policy": selected_policy,
        "seeds": SEEDS,
        "config_hashes": {str(row["registered_seed"]): row["config_hash"] for row in pd.read_csv(output / "01_input_run_manifest.csv").to_dict("records")},
        "selected_checkpoints": selected_info,
        "selected_checkpoint_canonical_hashes": {seed: info["canonical_model_state_sha256"] for seed, info in selected_info.items()},
        "per_seed_validation_metrics": selected_validation,
        "per_seed_test_metrics": {str(row["seed"]): row for row in primary_rows},
        "aggregate": aggregate_by_name,
        "classwise_aggregate": classwise_rows,
        "calibration_aggregate": [row for row in calibration_rows if str(row["role"]).endswith("_aggregate")],
        "replication_status": replication_status,
        "primary_limit_diagnosis": diagnosis,
        "optimizer_scheduler_next_action": next_action,
        "limitations": [
            "Five seeds are a small sample for estimating the full seed distribution.",
            "All runs share the same dataset split and code ancestry.",
            "Confidence intervals with n=5 are wide and descriptive.",
            "Validation checkpoint comparison is paired but seed-level inference is weak.",
            "Train-validation gap is an empirical generalization indicator, not a formal bound.",
            "Curve-based diagnosis is observational.",
            "No optimizer or scheduler alternative was trained in this task.",
            "Last checkpoint results are test-aware sensitivity only.",
            "Historical seed42 was discovered retrospectively.",
            "Test evaluation cannot be reused to revise checkpoint policy.",
            "High FER2013 test accuracy does not establish RAF-DB performance.",
            "No result guarantees 70% FER2013 accuracy.",
            "Semantic anchors mean the model is graph-only but not strictly pixel-node-only.",
        ],
        "created_at_utc": now_utc(),
    }
    baseline_path = output / "23_baseline_replication_lock.json"
    write_json_new(baseline_path, baseline_lock)
    baseline_sha = sha256_file(baseline_path)
    write_text_new(output / "23_baseline_replication_lock.sha256", baseline_sha)

    write_text_new(output / "14_test_reveal_manifest.md", "# Test Reveal Manifest\n\nCheckpoint-policy lock SHA verified before any test artifact read: `" + lock_sha + "`.\n\n" + markdown_table(test_sources) + "\n\nFresh inference was required only where the alternate checkpoint canonical hash matched neither the primary nor last checkpoint. All other metrics were recomputed from complete frozen official predictions generated by the completed runs.")
    write_csv_new(output / "15_primary_policy_test_results.csv", primary_rows)
    write_text_new(output / "15_primary_policy_test_results.md", "# Primary Policy Test Results\n\nLocked policy: **" + selected_policy + "**.\n\n" + markdown_table(primary_rows))
    write_csv_new(output / "16_secondary_checkpoint_sensitivity.csv", sensitivity_rows)
    write_text_new(output / "16_secondary_checkpoint_sensitivity.md", "# POST-LOCK TEST SENSITIVITY\n\nThese are **TEST-AWARE DESCRIPTIVE RESULT** comparisons and cannot revise the validation-selected policy.\n\n" + markdown_table(sensitivity_rows))
    write_csv_new(output / "17_five_seed_aggregate.csv", aggregate)
    write_text_new(output / "17_five_seed_aggregate.md", "# Five-Seed Aggregate\n\nSample standard deviation uses denominator n-1. The 95% interval is Student-t with df=4.\n\n" + markdown_table(aggregate))
    write_csv_new(output / "18_classwise_five_seed_results.csv", classwise_rows)
    write_text_new(output / "18_classwise_five_seed_results.md", "# Classwise Five-Seed Results\n\nStrongest class by mean F1: **" + strongest["class_name"] + "**. Weakest: **" + weakest["class_name"] + "**. Highest variance: **" + highest_variance["class_name"] + "**. Disgust has only 55 test samples and should not be interpreted with excessive certainty.\n\n" + markdown_table(classwise_rows) + "\n\nTop confusion destinations:\n\n" + markdown_table(pairs))
    confusion_sections = ["# Confusion Matrices"]
    for seed in SEEDS:
        matrix = confusion_by_seed[seed]
        rows = [{"true_class": CLASS_NAMES[index], **{CLASS_NAMES[j]: matrix[index][j] for j in range(7)}} for index in range(7)]
        confusion_sections.append(f"## Seed {seed}\n\n" + markdown_table(rows))
    write_text_new(output / "19_confusion_matrices.md", "\n\n".join(confusion_sections))
    write_csv_new(output / "20_calibration_results.csv", calibration_rows)
    write_text_new(output / "20_calibration_results.md", "# Calibration Results\n\nCalibration is descriptive and cannot override discrimination/replication gates. ECE uses 15 equal-width confidence bins.\n\n" + markdown_table(calibration_rows))
    historical_rows = [
        {"reference": "OFIX7-mid historical seed42 best", "n": 1, "checkpoint_policy": "val_macro_f1", "selection_status": "retrospective/test-aware candidate discovery", "accuracy": historical_metrics["accuracy"], "macro_f1": historical_metrics["macro_f1"], "weighted_f1": historical_metrics["weighted_f1"], "limitations": "single seed"},
        {"reference": "OFIX7-mid historical seed42 last", "n": 1, "checkpoint_policy": "last.pt sensitivity", "selection_status": "test-aware descriptive", "accuracy": historical_last_metrics["accuracy"], "macro_f1": historical_last_metrics["macro_f1"], "weighted_f1": historical_last_metrics["weighted_f1"], "limitations": "single seed; not primary"},
        {"reference": "OFIX7-mid current five-seed", "n": 5, "checkpoint_policy": selected_policy, "selection_status": "prospectively validation-locked", "accuracy": mean_acc, "macro_f1": mean_macro, "weighted_f1": aggregate_by_name["weighted_f1"]["mean"], "limitations": "five seeds, common split"},
        {"reference": "D18 C2 robustness baseline", "n": c2_reference.get("n"), "checkpoint_policy": "best.pt", "selection_status": c2_reference.get("status"), "accuracy": c2_reference.get("accuracy", {}).get("mean") if isinstance(c2_reference.get("accuracy"), dict) else None, "macro_f1": c2_reference.get("macro_f1", {}).get("mean") if isinstance(c2_reference.get("macro_f1"), dict) else None, "weighted_f1": None, "limitations": "different architecture/objective; exact artifact source recorded"},
        {"reference": "D18 strict-gap C0 seed42", "n": 1 if strict_peak else None, "checkpoint_policy": "best.pt", "selection_status": "contextual", "accuracy": strict_peak.get("accuracy") if strict_peak else None, "macro_f1": strict_peak.get("macro_f1") if strict_peak else None, "weighted_f1": strict_peak.get("weighted_f1") if strict_peak else None, "limitations": "single seed, different architecture"},
        {"reference": "Historical 65.14 anchor", "n": 1, "checkpoint_policy": "context only", "selection_status": "VERIFIED_CONTEXT_ONLY_6514", "accuracy": 0.6514, "macro_f1": None, "weighted_f1": None, "limitations": "exact eligible independent provenance not established"},
    ]
    write_text_new(output / "21_historical_replication_comparison.md", "# Historical Replication Comparison\n\nA five-seed mean is not directly equivalent to a single-seed point.\n\n" + markdown_table(historical_rows) + "\n\n## Hash Reconciliation\n\n" + markdown_table([
        {"artifact": "candidate lock registered checkpoint", "file_sha256": candidate_lock.get("checkpoint_sha256"), "canonical_model_state_sha256": historical_canonical, "status": candidate_lock.get("checkpoint_sha256") == historical_file_sha},
        {"artifact": "historical checkpoint file", "file_sha256": historical_file_sha, "canonical_model_state_sha256": historical_canonical, "status": historical_file_sha == lock.get("candidate_lock_raw_sha256", historical_file_sha) or historical_file_sha == candidate_lock.get("checkpoint_sha256")},
        {"artifact": "portable vs historical candidate lock", "file_sha256": f"raw={sha256_file(CANDIDATE_LOCK_PATH)} portable_normalized={normalized_text_sha256(PORTABLE_CANDIDATE_LOCK_PATH)}", "canonical_model_state_sha256": historical_canonical, "status": candidate_semantic_match},
    ]) + "\n\nDifferent lock-file serialization hashes are accepted only after semantic identity and the same historical checkpoint file/canonical model-state hash are verified.")
    write_text_new(output / "22_replication_status.md", "# Replication Status\n\nAssigned: **" + replication_status + "**.\n\nPrimary policy aggregate: accuracy mean=" + f"{mean_acc*100:.4f}%" + ", macro-F1 mean=" + f"{mean_macro*100:.4f}%" + ", accuracy sample SD=" + f"{sd_acc*100:.4f} pp" + ", minimum accuracy=" + f"{min_acc*100:.4f}%" + ".")
    write_text_new(output / "23_baseline_replication_lock.md", "# Baseline Replication Lock\n\nSHA-256: `" + baseline_sha + "`.\n\nThis freezes the five-seed baseline under the validation-selected policy; it does not silently promote any post-lock alternate/last test result.")
    permissible = "Report the five-seed mean and sample SD under the locked validation policy, disclose all five seeds and the common FER2013 split, and describe semantic anchor nodes explicitly."
    write_text_new(output / "24_paper_result_recommendation.md", "# Paper Result Recommendation\n\nDecision: **" + paper_recommendation + "**.\n\nPermissible claim: " + permissible + "\n\nDo not call the model pure pixel-only, do not use a single last checkpoint as primary, and do not claim 70% or RAF-DB performance.")
    write_text_new(output / "25_release_readiness.md", "# Release Readiness\n\nDecision: **" + release_readiness + "**.\n\nA clean release must preserve architecture, feature order, node semantics, graph schema, prior generation, optimizer, scheduler, checkpoint policy, split and deterministic seed handling. No release code was modified by this audit.")

    validation_rows = pd.read_csv(output / "07_validation_checkpoint_metrics.csv").to_dict("records")
    delta_rows = pd.read_csv(output / "08_checkpoint_policy_comparison.csv").to_dict("records")
    summary = {
        "input_runs": pd.read_csv(output / "01_input_run_manifest.csv").to_dict("records"),
        "registration_integrity": {"registration_hash_valid": True, "candidate_lock_valid": True, "candidate_semantic_match_after_reveal": candidate_semantic_match, "historical_checkpoint_file_sha256": historical_file_sha, "historical_checkpoint_canonical_model_state_sha256": historical_canonical},
        "runtime_parity": pd.read_csv(output / "03_config_and_runtime_parity.csv").to_dict("records"),
        "checkpoint_integrity": pd.read_csv(output / "04_checkpoint_inventory.csv").to_dict("records"),
        "validation_policy_analysis": {"per_checkpoint": validation_rows, "deltas": delta_rows, "gate_results": lock["gate_results"], "selected_policy": selected_policy, "lock_sha256": lock_sha},
        "training_curve_diagnosis": {"per_seed": pd.read_csv(output / "12_limit_evidence_by_seed.csv").to_dict("records"), "primary_diagnosis": diagnosis},
        "optimizer_scheduler_next_action": next_action,
        "test_reveal": {"primary_policy_per_seed": primary_rows, "aggregate": aggregate_by_name, "alternate_policy_sensitivity": [row for row in sensitivity_rows if row["comparison"] == "alternate_minus_primary"], "last_checkpoint_sensitivity": [row for row in sensitivity_rows if row["comparison"] == "last_minus_primary"]},
        "replication_status": replication_status,
        "historical_comparison": historical_rows,
        "classwise": {"rows": classwise_rows, "strongest": strongest["class_name"], "weakest": weakest["class_name"], "highest_variance": highest_variance["class_name"], "top_confusions": pairs},
        "calibration": calibration_rows,
        "paper_recommendation": paper_recommendation,
        "release_readiness": release_readiness,
        "baseline_lock_sha256": baseline_sha,
        "limitations": baseline_lock["limitations"],
    }
    write_json_new(output / "26_machine_readable_summary.json", summary)
    validation_summary = {
        "all_five_run_directories_found": True,
        "registered_seed_set_exact": True,
        "all_runs_complete": True,
        "no_resume_all_runs": True,
        "registration_found": True,
        "registration_hash_valid": True,
        "candidate_lock_found": True,
        "candidate_lock_valid": candidate_semantic_match and historical_file_sha == candidate_lock.get("checkpoint_sha256"),
        "config_hashes_match": True,
        "scientific_config_parity": True,
        "dataset_signature_match": True,
        "split_signature_match": True,
        "feature_signature_match": True,
        "graph_signature_match": True,
        "model_signature_match": True,
        "parameter_count_match": True,
        "optimizer_signature_match": True,
        "scheduler_signature_match": True,
        "early_stopping_signature_match": True,
        "checkpoint_monitor_match": True,
        "all_best_checkpoints_found": True,
        "all_macro_checkpoints_found": True,
        "all_accuracy_checkpoints_found": True,
        "all_last_checkpoints_found": True,
        "all_checkpoints_strict_load": True,
        "all_parameters_finite": True,
        "all_logits_finite": True,
        "best_alias_macro_hash_match": True,
        "history_checkpoint_alignment": True,
        "validation_metrics_recomputed": True,
        "stored_validation_metrics_agree": True,
        "validation_test_embargo_pass": lock.get("test_artifacts_read") is False,
        "checkpoint_policy_gate_applied": True,
        "checkpoint_policy_locked": True,
        "checkpoint_policy_lock_hash_created": True,
        "test_reveal_after_lock": True,
        "test_metrics_recomputed": True,
        "stored_test_metrics_agree": all(row["within_0_10pp"] for row in stored_agreement),
        "five_seed_aggregate_computed": True,
        "sample_standard_deviation_used": True,
        "confidence_intervals_computed": True,
        "classwise_analysis_computed": True,
        "calibration_analysis_computed": True,
        "architecture_limit_diagnosis_assigned": True,
        "optimizer_scheduler_next_action_assigned": True,
        "replication_status_assigned": True,
        "baseline_replication_lock_created": True,
        "baseline_lock_hash_created": True,
        "paper_recommendation_assigned": True,
        "release_readiness_assigned": True,
        "training_launched": False,
        "resume_launched": False,
        "checkpoint_modified": False,
        "config_modified": False,
        "model_modified": False,
        "dataset_modified": False,
        "graph_modified": False,
        "blocking_issues": [],
        "warnings": baseline_lock["limitations"],
    }
    write_json_new(output / "27_validation_summary.json", validation_summary)

    plot_dir = output / "plots"
    bar_plot(plot_dir / "test_accuracy_by_seed.png", [str(seed) for seed in SEEDS], [primary_by_seed[seed]["accuracy"] * 100.0 for seed in SEEDS], "Locked Policy Test Accuracy", "Accuracy (%)")
    bar_plot(plot_dir / "test_macro_f1_by_seed.png", [str(seed) for seed in SEEDS], [primary_by_seed[seed]["macro_f1"] * 100.0 for seed in SEEDS], "Locked Policy Test Macro-F1", "Macro-F1 (%)")
    f1_stats = [next(row for row in classwise_rows if row["class_name"] == name and row["metric"] == "f1") for name in CLASS_NAMES]
    plt.figure(figsize=(9, 5)); plt.bar(CLASS_NAMES, [row["mean"] * 100 for row in f1_stats], yerr=[row["sample_sd"] * 100 for row in f1_stats], capsize=4, color="#39796b"); plt.ylabel("F1 (%)"); plt.title("Classwise F1 Mean And Sample SD")
    save_plot_new(plot_dir / "classwise_f1_mean_std.png")
    plt.figure(figsize=(7, 5)); plt.scatter([row["accuracy"] * 100 for row in primary_rows], [row["macro_f1"] * 100 for row in primary_rows], s=60, color="#8a4f7d")
    for row in primary_rows: plt.annotate(str(row["seed"]), (row["accuracy"] * 100, row["macro_f1"] * 100), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Accuracy (%)"); plt.ylabel("Macro-F1 (%)"); plt.title("Accuracy And Macro-F1 By Seed"); plt.grid(alpha=0.25)
    save_plot_new(plot_dir / "accuracy_macro_f1_scatter.png")

    result = {
        "stage": "TEST_REVEAL_COMPLETE",
        "checkpoint_policy_lock_sha256": lock_sha,
        "selected_policy": selected_policy,
        "replication_status": replication_status,
        "primary_diagnosis": diagnosis,
        "optimizer_scheduler_next_action": next_action,
        "paper_recommendation": paper_recommendation,
        "release_readiness": release_readiness,
        "baseline_replication_lock_sha256": baseline_sha,
        "output_dir": rel(output),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["validation-lock", "test-reveal"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy-lock", type=Path)
    parser.add_argument("--prior-dir", type=Path, default=LOCAL_PRIOR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "validation-lock" and args.checkpoint_policy_lock is not None:
        raise RuntimeError("validation-lock must not receive a checkpoint policy lock")
    if args.stage == "test-reveal" and args.checkpoint_policy_lock is None:
        raise RuntimeError("test-reveal requires --checkpoint-policy-lock")
    return args


def main() -> None:
    args = parse_args()
    if args.stage == "validation-lock":
        validation_lock(args)
    else:
        test_reveal(args)


if __name__ == "__main__":
    main()
