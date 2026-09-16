"""Loss functions for Pixel GNN architectures."""

from __future__ import annotations

import tensorflow as tf


def compute_prototype_diversity_loss(prototypes: tf.Tensor) -> tf.Tensor:
    if prototypes is None:
        return tf.constant(0.0, dtype=tf.float32)

    p_norm = tf.math.l2_normalize(prototypes, axis=-1, epsilon=1e-6)
    num_motifs = tf.cast(tf.shape(p_norm)[0], tf.float32)

    cosine_sim = tf.matmul(p_norm, p_norm, transpose_b=True)
    eye = tf.eye(tf.shape(p_norm)[0], dtype=cosine_sim.dtype)
    off_diag = cosine_sim * (1.0 - eye)

    num_pairs = tf.maximum(num_motifs * (num_motifs - 1.0), 1.0)
    diversity_loss = tf.reduce_sum(tf.square(off_diag)) / num_pairs

    return diversity_loss


def compute_motif_diversity_loss(assignment: tf.Tensor | None) -> tf.Tensor:
    """Prevent motif collapse by encouraging uniform distribution of average motif usage."""
    if assignment is None:
        return tf.constant(0.0, dtype=tf.float32)

    usage = tf.reduce_mean(assignment, axis=[0, 1]) + 1e-8
    usage = usage / tf.reduce_sum(usage)
    num_motifs = tf.cast(tf.shape(usage)[0], tf.float32)

    # KL(usage || Uniform) = sum(usage * log(usage * K))
    kl_div = tf.reduce_sum(usage * tf.math.log(usage * num_motifs + 1e-8))
    return kl_div


def compute_spatial_coherence_loss(assignment: tf.Tensor | None) -> tf.Tensor:
    """Dirichlet energy on pixel grid: penalize assignment divergence between neighbor pixels."""
    if assignment is None:
        return tf.constant(0.0, dtype=tf.float32)

    from pixel_gnn.grid import StaticGridTopology

    grid = StaticGridTopology.get_instance()
    neighbors_idx = grid.neighbors_idx      # [2304, 8]
    neighbor_valid = grid.neighbor_valid    # [2304, 8]

    # assignment: [B, 2304, K]
    S_nbr = tf.gather(assignment, neighbors_idx, axis=1)          # [B, 2304, 8, K]
    S_self = tf.expand_dims(assignment, axis=2)                   # [B, 2304, 1, K]

    diff_sq = tf.reduce_sum(tf.square(S_self - S_nbr), axis=-1)  # [B, 2304, 8]
    mask = tf.broadcast_to(tf.expand_dims(neighbor_valid, axis=0), tf.shape(diff_sq))

    valid_diffs = tf.boolean_mask(diff_sq, mask)
    return tf.reduce_mean(valid_diffs)


def compute_motif_diagnostics(assignment: tf.Tensor | None) -> dict[str, float]:
    """Compute active_motifs, motif_usage_min/max/std, assignment_entropy."""
    if assignment is None:
        return {}

    usage = tf.reduce_mean(assignment, axis=[0, 1])
    num_motifs = tf.cast(tf.shape(usage)[0], tf.float32)
    threshold = 0.1 / num_motifs

    active_count = tf.reduce_sum(tf.cast(usage > threshold, tf.float32))
    u_min = tf.reduce_min(usage)
    u_max = tf.reduce_max(usage)
    u_std = tf.math.reduce_std(usage)

    # Pixel assignment entropy: -mean(sum(S * log(S)))
    eps = 1e-8
    entropy_per_pixel = -tf.reduce_sum(assignment * tf.math.log(assignment + eps), axis=-1)
    mean_entropy = tf.reduce_mean(entropy_per_pixel)

    return {
        "active_motifs": float(active_count.numpy()),
        "motif_usage_min": float(u_min.numpy()),
        "motif_usage_max": float(u_max.numpy()),
        "motif_usage_std": float(u_std.numpy()),
        "assignment_entropy": float(mean_entropy.numpy()),
    }


def compute_total_loss(
    labels: tf.Tensor,
    output: dict[str, tf.Tensor],
    lambda_diversity: float = 0.0,
    lambda_spatial_coherence: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    logits = tf.cast(output["logits"], tf.float32)
    labels = tf.cast(labels, tf.int64)

    if label_smoothing > 0.0:
        num_classes = tf.shape(logits)[-1]
        one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        ce_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                one_hot, logits, from_logits=True, label_smoothing=label_smoothing
            )
        )
    else:
        ce_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        )

    metrics = {"ce_loss": ce_loss}
    total_loss = ce_loss

    # Diversity regularization
    if lambda_diversity > 0.0:
        proto_loss = compute_prototype_diversity_loss(output.get("motif_prototypes"))
        assign_div = compute_motif_diversity_loss(output.get("motif_assignment"))
        div_loss = proto_loss + assign_div
        total_loss = total_loss + tf.cast(lambda_diversity, tf.float32) * div_loss
        metrics["diversity_loss"] = div_loss
    else:
        metrics["diversity_loss"] = tf.constant(0.0, dtype=tf.float32)

    # Spatial coherence regularization
    if lambda_spatial_coherence > 0.0 and output.get("motif_assignment") is not None:
        spatial_loss = compute_spatial_coherence_loss(output["motif_assignment"])
        total_loss = total_loss + tf.cast(lambda_spatial_coherence, tf.float32) * spatial_loss
        metrics["spatial_coherence_loss"] = spatial_loss
    else:
        metrics["spatial_coherence_loss"] = tf.constant(0.0, dtype=tf.float32)

    metrics["total_loss"] = total_loss
    return total_loss, metrics
