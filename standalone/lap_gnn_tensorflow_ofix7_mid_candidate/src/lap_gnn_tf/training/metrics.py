"""Seven-class metric semantics shared by validation and test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def classification_metrics(labels, probabilities, num_classes: int = 7) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    classes = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0,
    )
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = -np.log(clipped[np.arange(labels.size), labels]).mean()
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support_per_class": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "nll": float(nll),
        "brier": float(brier),
        "ece": expected_calibration_error(probabilities, labels),
    }
