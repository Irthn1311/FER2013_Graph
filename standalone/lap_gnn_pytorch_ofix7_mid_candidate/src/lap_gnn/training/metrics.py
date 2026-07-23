"""Metric definitions aligned with the historical evaluator."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(labels, predictions, probabilities=None, bins: int = 15) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(7), zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(7)).tolist(),
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        one_hot = np.eye(7, dtype=np.float64)[labels]
        confidence = probs.max(axis=1)
        correct = predictions == labels
        result["nll"] = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
        result["brier"] = float(np.square(probs - one_hot).sum(axis=1).mean())
        ece = 0.0
        boundaries = np.linspace(0.0, 1.0, bins + 1)
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            mask = (confidence > lower) & (confidence <= upper)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        result["ece"] = float(ece)
    return result
