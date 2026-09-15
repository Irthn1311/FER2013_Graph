"""Pixel predictions without fabricated landmark metadata."""

import os
from pathlib import Path
import numpy as np
from lap_gnn_tf.constants import CLASS_NAMES
from lap_gnn_tf.training.artifacts import _atomic_csv
from lap_gnn_tf.training.artifacts import _plotting


def write_training_curves(output_dir: str | Path, history: list[dict]) -> Path:
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
    fig.suptitle("Pixel-GNN Only training history")
    fig.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)
    return path



def write_predictions(output_dir: str | Path, details: dict) -> Path:
    path = Path(output_dir) / "predictions.csv"
    labels = np.asarray(details["labels"], dtype=np.int64)
    predictions = np.asarray(details["predictions"], dtype=np.int64)
    probabilities = np.asarray(details["probabilities"], dtype=np.float64)
    sample_ids = np.asarray(details["sample_ids"], dtype=np.int64)
    fields = [
        "sample_id", "true_index", "true_class", "predicted_index",
        "predicted_class", "correct", "confidence", *[f"p_{name}" for name in CLASS_NAMES],
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
        }
        row.update({
            f"p_{name}": float(probabilities[index, class_index])
            for class_index, name in enumerate(CLASS_NAMES)
        })
        rows.append(row)
    _atomic_csv(path, fields, rows)
    return path
