"""Future validation-only seed-42 lifecycle for CF-HPG v1.1 Resolution."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from .data import (
    build_clean_evaluation_dataset,
    build_dataset,
    load_fer_csv,
    validate_allowed_csv_path,
)
from .model import build_cf_hpg_v1_1_resolution


STATUS = "CF_HPG_V1_1_RESOLUTION_IMPLEMENTATION_ONLY"
DATASET_SIGNATURE = "fer2013_train28709_val3589_test3589"
SEED = 42
TRAIN_SAMPLES = 28_709
VALIDATION_SAMPLES = 3_589
NUM_CLASSES = 7
EXPECTED_PARAMETER_COUNT = 411_527
V1_0_REFERENCE = {
    "validation_accuracy": 0.5282808581777654,
    "validation_macro_f1": 0.46269581824371386,
    "clean_train_accuracy": 0.5650492876798217,
    "clean_train_macro_f1": 0.503022788224014,
}
TRAINING_CONFIG = {
    "seed": SEED,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 5e-4,
    "global_clipnorm": 1.0,
    "batch_size": 64,
    "max_epochs": 100,
    "warmup_epochs": 5,
    "cosine_final_learning_rate": 1e-6,
    "loss": "categorical_crossentropy_from_logits",
    "label_smoothing": 0.05,
    "checkpoint": "earliest_strict_max_val_accuracy",
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0,
}


class ValidationOnlyHarnessError(RuntimeError):
    """Raised for invalid future validation-only lifecycle state."""


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        steps_per_epoch: int,
        initial_learning_rate: float = 3e-4,
        final_learning_rate: float = 1e-6,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.steps_per_epoch = int(steps_per_epoch)
        self.initial_learning_rate = float(initial_learning_rate)
        self.final_learning_rate = float(final_learning_rate)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = float(self.steps_per_epoch * self.warmup_epochs)
        total_steps = float(self.steps_per_epoch * self.max_epochs)
        warmup = self.initial_learning_rate * step / warmup_steps
        progress = tf.clip_by_value(
            (step - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0
        )
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
        decayed = self.final_learning_rate + (
            self.initial_learning_rate - self.final_learning_rate
        ) * cosine
        return tf.where(step < warmup_steps, warmup, decayed)

    def get_config(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
        }


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
def sparse_smoothed_cross_entropy(labels, logits):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    one_hot = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.float32)
    return tf.keras.losses.categorical_crossentropy(
        one_hot,
        tf.cast(logits, tf.float32),
        from_logits=True,
        label_smoothing=0.05,
    )


def earliest_strict_max_epoch(values: Sequence[float]) -> int:
    if not values:
        raise ValidationOnlyHarnessError("At least one validation accuracy is required")
    best_epoch = 0
    best_value = float(values[0])
    if not math.isfinite(best_value):
        raise ValidationOnlyHarnessError("Validation accuracy must be finite")
    for epoch, value in enumerate(values[1:], start=1):
        value = float(value)
        if not math.isfinite(value):
            raise ValidationOnlyHarnessError("Validation accuracy must be finite")
        if value > best_value:
            best_value = value
            best_epoch = epoch
    return best_epoch


class EarliestStrictMaximumCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, output_root: str | Path):
        super().__init__()
        self.output_root = Path(output_root)
        self.best = -float("inf")
        self.selected_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        value = None if logs is None else logs.get("val_accuracy")
        if value is None or not math.isfinite(float(value)):
            raise ValidationOnlyHarnessError("Finite val_accuracy is required")
        if float(value) > self.best:
            self.best = float(value)
            self.selected_epoch = int(epoch)
            checkpoint = self.output_root / "checkpoints" / "best_val_accuracy.keras"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(checkpoint)
            _atomic_json(
                checkpoint.with_suffix(".metadata.json"),
                {
                    "policy": "earliest_strict_max_val_accuracy",
                    "selected_epoch_zero_based": self.selected_epoch,
                    "val_accuracy": self.best,
                    "seed": SEED,
                    "dataset_signature": DATASET_SIGNATURE,
                },
            )


def classify_outcome(
    *,
    validation_accuracy: float,
    validation_macro_f1: float,
    clean_train_accuracy: float,
    clean_train_macro_f1: float,
) -> str:
    acc_gap = round(100.0 * (clean_train_accuracy - validation_accuracy), 12)
    macro_gap = round(100.0 * (clean_train_macro_f1 - validation_macro_f1), 12)
    deltas = outcome_deltas(
        validation_accuracy=validation_accuracy,
        validation_macro_f1=validation_macro_f1,
        clean_train_accuracy=clean_train_accuracy,
        clean_train_macro_f1=clean_train_macro_f1,
    )
    if (
        validation_accuracy >= 0.7000
        and validation_macro_f1 >= 0.6700
        and acc_gap <= 8.0
        and macro_gap <= 8.0
    ):
        return "CF_HPG_V1_1_STRETCH_PASS"
    if (
        validation_accuracy >= 0.6500
        and validation_macro_f1 >= 0.6200
        and acc_gap <= 8.0
        and macro_gap <= 8.0
    ):
        return "CF_HPG_V1_1_PASS"
    if deltas["delta_clean_train_accuracy_pp"] >= 5.0 and (
        acc_gap > 10.0 or macro_gap > 10.0
    ):
        return "RESOLUTION_OVERFIT_SHIFT"
    if (
        deltas["delta_val_accuracy_pp"] >= 5.0
        and deltas["delta_clean_train_accuracy_pp"] >= 5.0
    ):
        return "RESOLUTION_STRONG_SIGNAL"
    if 3.0 <= deltas["delta_val_accuracy_pp"] < 5.0:
        return "RESOLUTION_PARTIAL_SIGNAL"
    if clean_train_accuracy < 0.6500 and validation_accuracy < 0.6000:
        return "RESOLUTION_UNDERFIT_REMAINS"
    return "RESOLUTION_INCONCLUSIVE"


def outcome_deltas(
    *,
    validation_accuracy: float,
    validation_macro_f1: float,
    clean_train_accuracy: float,
    clean_train_macro_f1: float,
) -> dict[str, float]:
    """Return the four preregistered percentage-point deltas from v1.0."""

    return {
        "delta_val_accuracy_pp": 100.0
        * (validation_accuracy - V1_0_REFERENCE["validation_accuracy"]),
        "delta_val_macro_pp": 100.0
        * (validation_macro_f1 - V1_0_REFERENCE["validation_macro_f1"]),
        "delta_clean_train_accuracy_pp": 100.0
        * (clean_train_accuracy - V1_0_REFERENCE["clean_train_accuracy"]),
        "delta_clean_train_macro_pp": 100.0
        * (clean_train_macro_f1 - V1_0_REFERENCE["clean_train_macro_f1"]),
    }


def build_optimizer(steps_per_epoch: int):
    schedule = WarmupCosine(steps_per_epoch=steps_per_epoch)
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule, weight_decay=5e-4, global_clipnorm=1.0
    )


def _atomic_json(path: Path, payload: Mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _evaluate(model, dataset):
    labels = []
    predictions = []
    losses = []
    for images, batch_labels in dataset:
        logits = model(images, training=False)
        batch_loss = sparse_smoothed_cross_entropy(batch_labels, logits)
        labels.extend(tf.reshape(batch_labels, [-1]).numpy().tolist())
        predictions.extend(tf.argmax(logits, axis=-1).numpy().tolist())
        losses.extend(batch_loss.numpy().tolist())
    return {
        "sample_count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "loss": float(np.mean(losses)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the future registered CF-HPG v1.1 Resolution lifecycle."
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_csv = validate_allowed_csv_path(args.train_csv)
    val_csv = validate_allowed_csv_path(args.val_csv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    train_images, train_labels = load_fer_csv(train_csv, TRAIN_SAMPLES)
    val_images, val_labels = load_fer_csv(val_csv, VALIDATION_SAMPLES)
    train_dataset = build_dataset(train_images, train_labels, training=True)
    clean_train = build_clean_evaluation_dataset(train_images, train_labels)
    validation = build_clean_evaluation_dataset(val_images, val_labels)
    model = build_cf_hpg_v1_1_resolution()
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise ValidationOnlyHarnessError(
            "Registered architecture parameter identity drift: "
            f"{model.count_params()} != {EXPECTED_PARAMETER_COUNT}"
        )
    steps_per_epoch = math.ceil(TRAIN_SAMPLES / TRAINING_CONFIG["batch_size"])
    model.compile(
        optimizer=build_optimizer(steps_per_epoch),
        loss=sparse_smoothed_cross_entropy,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (output_root / "model_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8", newline="\n"
    )
    _atomic_json(
        output_root / "resolved_config.json",
        {"status": STATUS, "dataset_signature": DATASET_SIGNATURE, **TRAINING_CONFIG},
    )
    checkpoint = EarliestStrictMaximumCheckpoint(output_root)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.0,
        restore_best_weights=False,
    )
    history = model.fit(
        train_dataset,
        validation_data=validation,
        epochs=100,
        callbacks=[checkpoint, early_stopping],
    )
    selected_model = tf.keras.models.load_model(
        output_root / "checkpoints" / "best_val_accuracy.keras", compile=False
    )
    clean_metrics = _evaluate(selected_model, clean_train)
    validation_metrics = _evaluate(selected_model, validation)
    decision = classify_outcome(
        validation_accuracy=validation_metrics["accuracy"],
        validation_macro_f1=validation_metrics["macro_f1"],
        clean_train_accuracy=clean_metrics["accuracy"],
        clean_train_macro_f1=clean_metrics["macro_f1"],
    )
    _atomic_json(
        output_root / "validation_only_result.json",
        {
            "status": "COMPLETE",
            "dataset_signature": DATASET_SIGNATURE,
            "selected_epoch_zero_based": checkpoint.selected_epoch,
            "epochs_completed": len(history.history["loss"]),
            "clean_train": clean_metrics,
            "validation": validation_metrics,
            "decision": decision,
            "deltas_vs_v1_0_pp": outcome_deltas(
                validation_accuracy=validation_metrics["accuracy"],
                validation_macro_f1=validation_metrics["macro_f1"],
                clean_train_accuracy=clean_metrics["accuracy"],
                clean_train_macro_f1=clean_metrics["macro_f1"],
            ),
            "training": True,
            "validation_only": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
