"""Human-readable and machine-readable training artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from lap_gnn_tf.constants import CLASS_NAMES


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _plotting():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def write_training_curves(output_dir: str | Path, history: list[dict[str, Any]]) -> Path:
    output_dir = Path(output_dir)
    path = output_dir / "training_curves.png"
    if not history:
        return path
    epochs = np.asarray([int(row["epoch"]) for row in history], dtype=np.int64)
    train_loss = np.asarray([float(row["train_loss"]) for row in history])
    val_loss = np.asarray([float(row["val_loss"]) for row in history])
    train_eval_loss = np.asarray([
        np.nan if row.get("train_eval_loss") is None else float(row["train_eval_loss"])
        for row in history
    ])
    train_accuracy = np.asarray([
        np.nan if row.get("train_accuracy") is None else float(row["train_accuracy"])
        for row in history
    ])
    train_macro = np.asarray([
        np.nan if row.get("train_macro_f1") is None else float(row["train_macro_f1"])
        for row in history
    ])
    val_accuracy = np.asarray([float(row["val_accuracy"]) for row in history])
    val_macro = np.asarray([float(row["val_macro_f1"]) for row in history])
    lr = np.asarray([float(row["lr"]) for row in history])

    plt = _plotting()
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(epochs, train_loss, label="train optimization loss", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="validation loss", linewidth=1.8)
    if np.isfinite(train_eval_loss).any():
        axes[0].plot(epochs, train_eval_loss, "o-", label="clean train-eval loss", markersize=3)
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, val_accuracy, label="validation accuracy", linewidth=1.8)
    axes[1].plot(epochs, val_macro, label="validation macro-F1", linewidth=1.8)
    if np.isfinite(train_accuracy).any():
        axes[1].plot(epochs, train_accuracy, "o-", label="clean train accuracy", markersize=3)
    if np.isfinite(train_macro).any():
        axes[1].plot(epochs, train_macro, "o-", label="clean train macro-F1", markersize=3)
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, lr, color="#8c564b", linewidth=1.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].grid(alpha=0.25)
    fig.suptitle("TensorFlow OFIX7-mid training history")
    fig.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)
    return path


def write_confusion_matrix(
    output_dir: str | Path,
    confusion: list[list[int]],
    accuracy: float,
    macro_f1: float,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    matrix = np.asarray(confusion, dtype=np.int64)
    csv_path = output_dir / "confusion_matrix.csv"
    rows = [
        {"true_class": CLASS_NAMES[index], **{
            CLASS_NAMES[column]: int(matrix[index, column])
            for column in range(len(CLASS_NAMES))
        }}
        for index in range(len(CLASS_NAMES))
    ]
    _atomic_csv(csv_path, ["true_class", *CLASS_NAMES], rows)

    plt = _plotting()
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Test confusion matrix | acc={accuracy:.4f} macro-F1={macro_f1:.4f}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    png_path = output_dir / "confusion_matrix.png"
    temporary = png_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, png_path)
    return csv_path, png_path


def write_per_class_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(output_dir) / "per_class_metrics.csv"
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append({
            "class_index": index,
            "class_name": name,
            "precision": float(metrics["per_class_precision"][index]),
            "recall": float(metrics["per_class_recall"][index]),
            "f1": float(metrics["per_class_f1"][index]),
            "support": int(metrics["support_per_class"][index]),
        })
    _atomic_csv(
        path,
        ["class_index", "class_name", "precision", "recall", "f1", "support"],
        rows,
    )
    return path


def write_predictions(output_dir: str | Path, details: dict[str, Any]) -> Path:
    path = Path(output_dir) / "predictions.csv"
    labels = np.asarray(details["labels"], dtype=np.int64)
    predictions = np.asarray(details["predictions"], dtype=np.int64)
    probabilities = np.asarray(details["probabilities"], dtype=np.float64)
    sample_ids = np.asarray(details["sample_ids"], dtype=np.int64)
    detected = np.asarray(details["detected"], dtype=np.bool_)
    missing = np.asarray(details["landmark_missing_flag"], dtype=np.int64)
    fields = [
        "sample_id", "true_index", "true_class", "predicted_index",
        "predicted_class", "correct", "confidence", "detected",
        "landmark_missing_flag", *[f"p_{name}" for name in CLASS_NAMES],
    ]
    rows = []
    for index in range(labels.size):
        row = {
            "sample_id": int(sample_ids[index]),
            "true_index": int(labels[index]),
            "true_class": CLASS_NAMES[int(labels[index])],
            "predicted_index": int(predictions[index]),
            "predicted_class": CLASS_NAMES[int(predictions[index])],
            "correct": int(labels[index] == predictions[index]),
            "confidence": float(probabilities[index, predictions[index]]),
            "detected": int(detected[index]),
            "landmark_missing_flag": int(missing[index]),
        }
        row.update({
            f"p_{name}": float(probabilities[index, class_index])
            for class_index, name in enumerate(CLASS_NAMES)
        })
        rows.append(row)
    _atomic_csv(path, fields, rows)
    return path


def write_artifact_manifest(output_dir: str | Path, paths: list[Path]) -> Path:
    output_dir = Path(output_dir)
    records = []
    for path in paths:
        payload = path.read_bytes()
        records.append({
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path
