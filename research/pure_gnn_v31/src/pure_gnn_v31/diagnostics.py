"""Runtime diagnostics for Pure-GNN v3.1."""

from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf


def compute_effective_rank(tensor_2d: tf.Tensor) -> tf.Tensor:
    """Computes numerical effective rank of a 2D matrix (N, C) via singular values:
    
    eff_rank = exp(-sum p_i log p_i) where p_i = s_i / sum(s).
    """
    s = tf.linalg.svd(tensor_2d, compute_uv=False)
    s_sum = tf.reduce_sum(s) + 1e-12
    p = s / s_sum
    # Filter zeros
    p_safe = tf.clip_by_value(p, 1e-12, 1.0)
    entropy = -tf.reduce_sum(p_safe * tf.math.log(p_safe))
    return tf.exp(entropy)


def compute_feature_diagnostics(h: tf.Tensor) -> Dict[str, float]:
    """Computes feature variance, Frobenius norm, and effective rank for node tensor (B, N, C)."""
    # Reshape to (B * N, C)
    c = h.shape[-1]
    flat = tf.reshape(h, [-1, c])
    var = tf.math.reduce_variance(flat)
    norm = tf.norm(flat, ord="fro") / tf.cast(tf.shape(flat)[0], tf.float32)

    # Subsample if large for fast SVD
    max_svd_rows = 512
    if tf.shape(flat)[0] > max_svd_rows:
        sample_flat = flat[:max_svd_rows]
    else:
        sample_flat = flat
    eff_rank = compute_effective_rank(sample_flat)

    return {
        "feature_variance": float(var.numpy()),
        "feature_norm": float(norm.numpy()),
        "effective_rank": float(eff_rank.numpy()),
    }


def compute_boundary_diagnostics(
    h_stage1: tf.Tensor,
    height: int = 48,
    width: int = 48,
) -> Dict[str, float]:
    """Computes activation norms on border vs interior nodes for HxW grid."""
    # h_stage1 is (B, N, C)
    c = h_stage1.shape[-1]
    h_grid = tf.reshape(h_stage1, [-1, height, width, c])

    # Interior: [1:-1, 1:-1]
    interior = h_grid[:, 1:-1, 1:-1, :]
    int_norm = tf.reduce_mean(tf.norm(interior, axis=-1))

    # Border: mask where r in {0, H-1} or col in {0, W-1}
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask_tf = tf.constant(mask, dtype=tf.bool)

    border = tf.boolean_mask(h_grid, mask_tf, axis=1)  # (B, N_border, C)
    border_norm = tf.reduce_mean(tf.norm(border, axis=-1))

    return {
        "interior_activation_norm": float(int_norm.numpy()),
        "border_activation_norm": float(border_norm.numpy()),
        "border_to_interior_ratio": float((border_norm / (int_norm + 1e-8)).numpy()),
    }


def compute_shift_stability(
    model: tf.keras.Model,
    sample_images: tf.Tensor,
) -> Dict[str, float]:
    """Measures representation stability under 1-pixel and 2-pixel spatial translations.
    
    Args:
        model: PureGNNv31 model
        sample_images: (B, 48, 48, 1) float32 in [0, 1]
    Returns:
        cosine similarity between base predictions and shifted predictions.
    """
    logits_base = model(sample_images, training=False)
    probs_base = tf.nn.softmax(logits_base, axis=-1)

    # 1-pixel shift down and right: pad and crop
    shifted_1px = tf.pad(sample_images[:, :-1, :-1, :], [[0, 0], [1, 0], [1, 0], [0, 0]], mode="CONSTANT")
    logits_1px = model(shifted_1px, training=False)
    probs_1px = tf.nn.softmax(logits_1px, axis=-1)

    # 2-pixel shift down and right
    shifted_2px = tf.pad(sample_images[:, :-2, :-2, :], [[0, 0], [2, 0], [2, 0], [0, 0]], mode="CONSTANT")
    logits_2px = model(shifted_2px, training=False)
    probs_2px = tf.nn.softmax(logits_2px, axis=-1)

    # Cosine similarities
    cos_1px = tf.reduce_mean(
        tf.reduce_sum(probs_base * probs_1px, axis=-1) /
        (tf.norm(probs_base, axis=-1) * tf.norm(probs_1px, axis=-1) + 1e-12)
    )
    cos_2px = tf.reduce_mean(
        tf.reduce_sum(probs_base * probs_2px, axis=-1) /
        (tf.norm(probs_base, axis=-1) * tf.norm(probs_2px, axis=-1) + 1e-12)
    )

    return {
        "shift_1px_cosine_stability": float(cos_1px.numpy()),
        "shift_2px_cosine_stability": float(cos_2px.numpy()),
    }
