import csv
import json

import numpy as np

from lap_gnn_tf.training.artifacts import (
    write_artifact_manifest,
    write_confusion_matrix,
    write_per_class_metrics,
    write_predictions,
    write_training_curves,
)


def _metrics():
    return {
        "accuracy": 0.75,
        "macro_f1": 0.70,
        "per_class_precision": [0.7] * 7,
        "per_class_recall": [0.6] * 7,
        "per_class_f1": [0.65] * 7,
        "support_per_class": [2] * 7,
        "confusion_matrix": np.eye(7, dtype=np.int64).tolist(),
    }


def test_training_and_evaluation_artifacts(tmp_path):
    history = [
        {
            "epoch": 1,
            "train_loss": 1.5,
            "train_eval_loss": 1.4,
            "train_accuracy": 0.5,
            "train_macro_f1": 0.45,
            "val_loss": 1.6,
            "val_accuracy": 0.48,
            "val_macro_f1": 0.43,
            "lr": 3e-4,
        },
        {
            "epoch": 2,
            "train_loss": 1.3,
            "train_eval_loss": None,
            "train_accuracy": None,
            "train_macro_f1": None,
            "val_loss": 1.5,
            "val_accuracy": 0.52,
            "val_macro_f1": 0.47,
            "lr": 3e-4,
        },
    ]
    curves = write_training_curves(tmp_path, history)
    metrics = _metrics()
    per_class = write_per_class_metrics(tmp_path, metrics)
    confusion_csv, confusion_png = write_confusion_matrix(
        tmp_path,
        metrics["confusion_matrix"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    details = {
        "labels": np.asarray([0, 1], dtype=np.int64),
        "predictions": np.asarray([0, 2], dtype=np.int64),
        "probabilities": np.asarray([
            [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0],
        ]),
        "sample_ids": np.asarray([10, 11], dtype=np.int64),
        "detected": np.asarray([True, False]),
        "landmark_missing_flag": np.asarray([0, 1], dtype=np.int64),
    }
    predictions = write_predictions(tmp_path, details)
    manifest = write_artifact_manifest(
        tmp_path,
        [curves, per_class, confusion_csv, confusion_png, predictions],
    )

    assert curves.stat().st_size > 0
    assert confusion_png.stat().st_size > 0
    with predictions.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[1]["predicted_class"] == "fear"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["artifacts"]) == 5
