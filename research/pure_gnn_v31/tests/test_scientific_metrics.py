"""Tests verifying that scientific metrics (Macro-F1, etc.) match toy reference implementations."""

import numpy as np
import pytest
from pure_gnn_v31.scientific.metrics import (
    compute_scientific_metrics,
    toy_reference_macro_f1,
)


def test_macro_f1_matches_independent_toy_reference():
    # Synthetic toy test case with known edge cases (e.g. empty class predictions)
    y_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 0, 0, 3, 4]
    y_pred = [0, 1, 2, 3, 4, 5, 5, 1, 1, 2, 0, 0, 2, 4]  # class 6 never predicted by y_pred

    toy_f1 = toy_reference_macro_f1(y_true, y_pred, num_classes=7)
    metrics = compute_scientific_metrics(np.array(y_true), np.array(y_pred), num_classes=7)

    assert abs(metrics.macro_f1 - toy_f1) < 1e-6
    assert len(metrics.precision_per_class) == 7
    assert len(metrics.recall_per_class) == 7
    assert len(metrics.f1_per_class) == 7
    assert len(metrics.confusion_matrix) == 7
