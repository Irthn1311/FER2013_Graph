"""Registered training-only label-smoothing objective for Issue #58."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import tensorflow as tf


LABEL_SMOOTHING = 0.05
NUM_CLASSES = 7


def _targets_for_epsilon(labels: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Build only the registered smooth target or its epsilon-zero control."""

    epsilon = float(epsilon)
    if epsilon not in {0.0, LABEL_SMOOTHING}:
        raise ValueError("Only epsilon 0.0 and the registered epsilon 0.05 are allowed")
    one_hot = tf.one_hot(
        tf.cast(labels, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return (1.0 - epsilon) * one_hot + epsilon / NUM_CLASSES


def registered_smoothed_targets(labels: tf.Tensor) -> tf.Tensor:
    """Return `(1 - 0.05) * one_hot(label, 7) + 0.05 / 7`."""

    return _targets_for_epsilon(labels, LABEL_SMOOTHING)


def _mean_cross_entropy(
    labels: tf.Tensor, logits: tf.Tensor, *, epsilon: float
) -> tf.Tensor:
    logits = tf.cast(logits, tf.float32)
    if logits.shape.rank is not None and logits.shape[-1] not in {None, NUM_CLASSES}:
        raise ValueError(f"Registered logits width must be {NUM_CLASSES}")
    tf.debugging.assert_equal(
        tf.shape(logits)[-1],
        NUM_CLASSES,
        message="Registered logits width must be 7",
    )
    targets = _targets_for_epsilon(labels, epsilon)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=targets, logits=logits
    )
    return tf.reduce_mean(losses)


def smoothed_sparse_cross_entropy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    """Compute the fixed float32 training CE with epsilon exactly 0.05."""

    return _mean_cross_entropy(labels, logits, epsilon=LABEL_SMOOTHING)


def hard_sparse_cross_entropy_compatibility(
    labels: tf.Tensor, logits: tf.Tensor
) -> tf.Tensor:
    """Epsilon-zero control used only to prove equivalence with frozen hard CE."""

    return _mean_cross_entropy(labels, logits, epsilon=0.0)


@contextmanager
def training_loss_binding() -> Iterator[dict[str, object]]:
    """Temporarily replace only the frozen execution module's loss binding."""

    from lap_gnn_tf.training import evaluator, execution
    from lap_gnn_tf.training.losses import sparse_cross_entropy as frozen_hard_ce

    original_execution_binding = execution.sparse_cross_entropy
    original_evaluator_binding = evaluator.sparse_cross_entropy
    if original_execution_binding is not frozen_hard_ce:
        raise RuntimeError("Frozen execution loss binding was already modified")
    if original_evaluator_binding is not frozen_hard_ce:
        raise RuntimeError("Frozen evaluator hard-CE binding was already modified")

    execution.sparse_cross_entropy = smoothed_sparse_cross_entropy
    state: dict[str, object] = {
        "execution_binding_replaced": True,
        "evaluator_binding_unchanged": evaluator.sparse_cross_entropy
        is original_evaluator_binding,
        "original_execution_binding": original_execution_binding,
        "original_evaluator_binding": original_evaluator_binding,
    }
    try:
        yield state
    finally:
        execution.sparse_cross_entropy = original_execution_binding
