"""Scientific evaluation metrics and independent toy reference implementations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np


@dataclass
class ScientificMetrics:
    accuracy: float
    macro_f1: float
    precision_per_class: List[float]
    recall_per_class: List[float]
    f1_per_class: List[float]
    confusion_matrix: List[List[int]]


def toy_reference_macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 7) -> float:
    """Independent unvectorized reference implementation for Macro-F1 testing."""
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / num_classes)


def compute_scientific_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> ScientificMetrics:
    """Computes full evaluation metrics on validation predictions with strict input validation."""
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()

    # 1. Reject empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # 2. Reject mismatched lengths
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements."
        )

    # 3. Reject labels outside [0, num_classes - 1]
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        invalid_labels = y_true[(y_true < 0) | (y_true >= num_classes)]
        raise ValueError(f"Found y_true labels outside range [0, {num_classes - 1}]: {invalid_labels[:5]}")

    # 4. Reject predictions outside [0, num_classes - 1]
    if np.any(y_pred < 0) or np.any(y_pred >= num_classes):
        invalid_preds = y_pred[(y_pred < 0) | (y_pred >= num_classes)]
        raise ValueError(f"Found y_pred predictions outside range [0, {num_classes - 1}]: {invalid_preds[:5]}")

    accuracy = float(np.mean(y_true == y_pred))

    prec_list = []
    rec_list = []
    f1_list = []
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))

    return ScientificMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision_per_class=prec_list,
        recall_per_class=rec_list,
        f1_per_class=f1_list,
        confusion_matrix=cm.tolist(),
    )
