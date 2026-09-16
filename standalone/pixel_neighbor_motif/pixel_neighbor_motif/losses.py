"""Loss functions for Pixel Neighbor Attention + Motif Prototypes."""

from __future__ import annotations

import tensorflow as tf


def compute_prototype_diversity_loss(prototypes: tf.Tensor) -> tf.Tensor:
    """Compute diversity loss on prototype matrix P to prevent prototype collapse.

    L_div = mean_{i != j} (cosine_sim(p_i, p_j)^2)

    Args:
        prototypes: [K, D] normalized prototype tensor.
    Returns:
        Scalar penalty >= 0.0
    """
    if prototypes is None:
        return tf.constant(0.0, dtype=tf.float32)

    p_norm = tf.math.l2_normalize(prototypes, axis=-1, epsilon=1e-6)  # [K, D]
    num_motifs = tf.cast(tf.shape(p_norm)[0], tf.float32)

    # Cosine similarity matrix: [K, K]
    cosine_sim = tf.matmul(p_norm, p_norm, transpose_b=True)

    # Mask diagonal elements (similarity with self is always 1.0)
    eye = tf.eye(tf.shape(p_norm)[0], dtype=cosine_sim.dtype)
    off_diag = cosine_sim * (1.0 - eye)

    # Mean squared off-diagonal cosine similarity
    num_pairs = tf.maximum(num_motifs * (num_motifs - 1.0), 1.0)
    diversity_loss = tf.reduce_sum(tf.square(off_diag)) / num_pairs

    return diversity_loss


def compute_total_loss(
    labels: tf.Tensor,
    output: dict[str, tf.Tensor],
    lambda_diversity: float = 0.0,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Compute total training loss: CrossEntropy + optional diversity penalty.

    Args:
        labels: [B] integer class labels in {0, ..., 6}
        output: dict from model.call()
        lambda_diversity: weight for prototype diversity loss (default 0.0)
    Returns:
        total_loss, loss_metrics_dict
    """
    logits = tf.cast(output["logits"], tf.float32)
    labels = tf.cast(labels, tf.int64)

    # Classification Cross Entropy
    ce_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )

    metrics = {"ce_loss": ce_loss}

    total_loss = ce_loss

    if lambda_diversity > 0.0 and output.get("motif_prototypes") is not None:
        div_loss = compute_prototype_diversity_loss(output["motif_prototypes"])
        total_loss = total_loss + tf.cast(lambda_diversity, tf.float32) * div_loss
        metrics["diversity_loss"] = div_loss
    else:
        metrics["diversity_loss"] = tf.constant(0.0, dtype=tf.float32)

    metrics["total_loss"] = total_loss
    return total_loss, metrics
