"""Deterministic protocol helpers for the residual experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
import torch


CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state_or_model: Any) -> str:
    state = state_or_model.state_dict() if hasattr(state_or_model, "state_dict") else state_or_model
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probability = softmax(logits)
    prediction = logits.argmax(axis=1)
    recalls = []
    for class_id in range(7):
        mask = labels == class_id
        recalls.append(None if not mask.any() else float(np.mean(prediction[mask] == class_id)))
    return {
        "loss": float(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-300, 1.0)).mean()),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", labels=np.arange(7), zero_division=0)),
        "per_class_recall": recalls,
        "correct_count": int(np.sum(prediction == labels)),
    }


def disagreement(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first_ok = first == labels
    second_ok = second == labels
    equal = first == second
    return {
        "N_CC": int((first_ok & second_ok).sum()),
        "N_CW": int((first_ok & ~second_ok).sum()),
        "N_WC": int((~first_ok & second_ok).sum()),
        "N_WW": int((~first_ok & ~second_ok).sum()),
        "prediction_agreement_count": int(equal.sum()),
        "prediction_disagreement_count": int((~equal).sum()),
        "prediction_disagreement_rate": float(np.mean(~equal)),
        "correctness_disagreement_count": int((first_ok ^ second_ok).sum()),
        "either_correct_count": int((first_ok | second_ok).sum()),
        "either_correct_accuracy": float(np.mean(first_ok | second_ok)),
    }


def change_counts(labels: np.ndarray, baseline: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    base_ok = baseline == labels
    residual_ok = residual == labels
    changed = baseline != residual
    fixed = ~base_ok & residual_ok
    broken = base_ok & ~residual_ok
    return {
        "argmax_changed_count": int(changed.sum()),
        "argmax_changed_fraction": float(changed.mean()),
        "npf_wrong_to_residual_correct": int(fixed.sum()),
        "npf_correct_to_residual_wrong": int(broken.sum()),
        "net_corrected_count": int(fixed.sum() - broken.sum()),
    }


def is_better(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None) -> bool:
    if incumbent is None:
        return True
    return (candidate["accuracy"], candidate["macro_f1"]) > (
        incumbent["accuracy"], incumbent["macro_f1"]
    )


def learning_rate_for_epoch(epoch: int, *, base_lr: float = 3e-4, epochs: int = 30, warmup_epochs: int = 2) -> float:
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch outside registered horizon")
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def gate_summary(gate: np.ndarray | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    if gate.ndim != 2 or gate.shape[1] != 7 or not np.isfinite(gate).all():
        raise ValueError("gate must be finite [N,7]")
    return {
        "mean_per_class": gate.mean(axis=0).tolist(),
        "median_per_class": np.median(gate, axis=0).tolist(),
        "fraction_gt_0_5_per_class": (gate > 0.5).mean(axis=0).tolist(),
        "fraction_gt_0_75_per_class": (gate > 0.75).mean(axis=0).tolist(),
    }


def finite_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON value")
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
