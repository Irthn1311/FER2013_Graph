import numpy as np

from lap_gnn_tf.training.metrics import classification_metrics, expected_calibration_error


def test_metric_parity():
    labels = np.arange(7)
    probabilities = np.eye(7)
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_ece_uses_pytorch_boundary_convention():
    probabilities = np.asarray([
        [0.8, 0.2, 0, 0, 0, 0, 0],
        [0.2, 0.8, 0, 0, 0, 0, 0],
        [0.6, 0.4, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    expected = 0.0
    boundaries = np.linspace(0.0, 1.0, 6)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            expected += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    assert expected_calibration_error(probabilities, labels, bins=5) == expected
