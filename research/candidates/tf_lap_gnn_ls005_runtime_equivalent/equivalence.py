"""Deterministic tiny proof that inference-only clean evaluation is state neutral."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.early_stopping import ValidationLossEarlyStopping
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW
from lap_gnn_tf.training.plateau import TorchCompatibleReduceLROnPlateau
from lap_gnn_tf.training.losses import sparse_cross_entropy as hard_cross_entropy
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (
    smoothed_sparse_cross_entropy,
)


SEED = 42
PRIOR_CORRUPTION_SEED = 7741
EPOCH_MIX = 1_000_003
SAMPLE_MIX = 97_531
EPOCHS = 3


class TinyLifecycleModel(tf.keras.Model):
    """Small training/evaluation graph with training-only dropout RNG."""

    def __init__(self) -> None:
        super().__init__(name="ls005_lifecycle_equivalence_model")
        self.hidden = tf.keras.layers.Dense(
            8,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=101),
            bias_initializer="zeros",
        )
        self.dropout = tf.keras.layers.Dropout(0.20, seed=202)
        self.classifier = tf.keras.layers.Dense(
            7,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=303),
            bias_initializer="zeros",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden = self.hidden(inputs)
        hidden = self.dropout(hidden, training=training)
        return self.classifier(hidden)


@dataclass
class CheckpointState:
    best_accuracy: float = -float("inf")
    best_epoch: int = -1
    weights_sha256: str | None = None

    def update(self, model: tf.keras.Model, epoch: int, accuracy: float) -> None:
        if float(accuracy) > self.best_accuracy:
            self.best_accuracy = float(accuracy)
            self.best_epoch = int(epoch)
            self.weights_sha256 = _arrays_sha256(model.variables)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "weights_sha256": self.weights_sha256,
        }


def _arrays_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value.numpy())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _epoch_order(epoch: int, size: int = 12) -> np.ndarray:
    order = np.arange(size, dtype=np.int64)
    np.random.default_rng(SEED + int(epoch) * EPOCH_MIX).shuffle(order)
    return order


def _augmentation_value(epoch: int, sample_index: int) -> np.ndarray:
    mixed = (
        PRIOR_CORRUPTION_SEED
        + int(epoch) * EPOCH_MIX
        + int(sample_index) * SAMPLE_MIX
    ) % (2**32 - 1)
    return np.random.default_rng(mixed).normal(0.0, 0.025, size=(4,)).astype(
        np.float32
    )


def _next_epoch_data_state(epoch: int) -> dict[str, Any]:
    next_epoch = int(epoch) + 1
    order = _epoch_order(next_epoch)
    augmentations = np.stack(
        [_augmentation_value(next_epoch, int(index)) for index in order]
    )
    return {
        "epoch": next_epoch,
        "order": order.tolist(),
        "augmentation_sha256": hashlib.sha256(
            augmentations.tobytes(order="C")
        ).hexdigest(),
    }


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(
        [[(row * 7 + col * 3) % 19 / 9.0 - 1.0 for col in range(4)] for row in range(12)],
        dtype=np.float32,
    )
    train_y = np.asarray([0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 6, 1], dtype=np.int32)
    val_x = np.asarray(
        [[(row * 5 + col * 11) % 23 / 11.0 - 1.0 for col in range(4)] for row in range(8)],
        dtype=np.float32,
    )
    val_y = np.asarray([0, 2, 4, 6, 1, 3, 5, 0], dtype=np.int32)
    return train_x, train_y, val_x, val_y


def _metrics(model, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = model(tf.constant(features), training=False)
    loss = float(hard_cross_entropy(tf.constant(labels), logits).numpy())
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = float(
        tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)).numpy()
    )
    return {"loss": loss, "accuracy": accuracy}


def _run_lifecycle(include_clean_train_evaluation: bool) -> list[dict[str, Any]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    train_x, train_y, val_x, val_y = _dataset()
    model = TinyLifecycleModel()
    model(tf.zeros((1, 4), dtype=tf.float32), training=False)
    optimizer = TorchCompatibleAdamW(
        learning_rate=3e-4,
        weight_decay=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        global_clipnorm=5.0,
        name="adamw",
    )
    optimizer.build(model.trainable_variables)
    scheduler = TorchCompatibleReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=3e-5,
    )
    early = ValidationLossEarlyStopping(min_epochs=30, patience=15)
    checkpoint = CheckpointState()

    @tf.function(reduce_retracing=True)
    def train_step(features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = smoothed_sparse_cross_entropy(labels, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    snapshots: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        order = _epoch_order(epoch)
        for start in range(0, len(order), 4):
            indices = order[start : start + 4]
            augmented = np.stack(
                [train_x[index] + _augmentation_value(epoch, int(index)) for index in indices]
            )
            train_step(tf.constant(augmented), tf.constant(train_y[indices]))
        if include_clean_train_evaluation:
            # This is the only A/B difference. Inference must not consume model/data RNG.
            _metrics(model, train_x, train_y)
        validation = _metrics(model, val_x, val_y)
        stop = early.update(epoch, validation["loss"])
        checkpoint.update(model, epoch, validation["accuracy"])
        scheduler.step(validation["loss"])
        snapshots.append(
            {
                "epoch": epoch,
                "trainable_sha256": _arrays_sha256(model.trainable_variables),
                "non_trainable_sha256": _arrays_sha256(model.non_trainable_variables),
                "optimizer_sha256": _arrays_sha256(optimizer.variables),
                "optimizer_iterations": int(optimizer.iterations.numpy()),
                "learning_rate": float(optimizer.learning_rate.numpy()),
                "scheduler_state": scheduler.get_state(),
                "validation": validation,
                "checkpoint_state": checkpoint.as_dict(),
                "early_stopping_state": early.get_state(),
                "stop_requested": bool(stop),
                "next_epoch_data_state": _next_epoch_data_state(epoch),
            }
        )
    return snapshots


def prove_synthetic_lifecycle_equivalence() -> dict[str, Any]:
    """Return an exact, machine-readable A/B lifecycle equivalence proof."""

    with_clean = _run_lifecycle(include_clean_train_evaluation=True)
    lean = _run_lifecycle(include_clean_train_evaluation=False)
    exact = with_clean == lean
    differing_epochs = [
        index + 1
        for index, (left, right) in enumerate(zip(with_clean, lean))
        if left != right
    ]
    return {
        "schema_version": 1,
        "status": "PASS" if exact else "FAIL",
        "epochs_compared": EPOCHS,
        "exact_equality": exact,
        "tolerance": {"floating": 0.0, "discrete": "exact"},
        "differing_epochs": differing_epochs,
        "compared_state": [
            "trainable_model_variables",
            "non_trainable_model_variables",
            "optimizer_variables_and_slots",
            "optimizer_iteration",
            "current_learning_rate",
            "scheduler_bookkeeping",
            "validation_metrics",
            "checkpoint_selection_state",
            "early_stopping_state",
            "next_epoch_training_batch_order",
            "next_epoch_augmentation_sequence",
        ],
        "state_sha256": hashlib.sha256(
            json.dumps(lean, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "epochs": lean,
    }
