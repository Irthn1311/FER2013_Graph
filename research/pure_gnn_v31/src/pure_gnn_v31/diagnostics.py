"""Runtime diagnostics for Pure-GNN v3.1."""

from typing import Dict
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


def tensor_feature_diagnostics(h: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Return graph-block feature diagnostics without changing the feature path."""
    # Reshape to (B * N, C)
    c = h.shape[-1]
    flat = tf.reshape(h, [-1, c])
    var = tf.math.reduce_variance(flat)
    norm = tf.sqrt(tf.reduce_sum(tf.square(flat))) / tf.cast(tf.shape(flat)[0], tf.float32)

    sample_flat = flat[:512]
    eff_rank = compute_effective_rank(sample_flat)

    return {
        "feature_variance": var,
        "feature_norm": norm,
        "effective_rank": eff_rank,
    }


def compute_feature_diagnostics(h: tf.Tensor) -> Dict[str, float]:
    """Eager convenience wrapper around :func:`tensor_feature_diagnostics`."""
    return {key: float(value.numpy()) for key, value in tensor_feature_diagnostics(h).items()}


def collect_graph_block_diagnostics(
    model: tf.keras.Model,
    inputs: tf.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Trace all eight graph blocks and measure feature plus gradient norms.

    This opt-in helper executes the existing layers once with ``training=False``;
    it neither changes nor participates in the default model forward path.
    """
    block_tensors = []
    scalar_diagnostics = {}
    with tf.GradientTape(persistent=True) as tape:
        h = tf.reshape(tf.cast(inputs, tf.float32), [-1, 2304, 1])
        h = model.input_proj(h)
        stages = (
            ("stage1", model.stage1_blocks, model.graph_s1),
            ("stage2", model.stage2_blocks, model.graph_s2),
            ("stage3", model.stage3_blocks, model.graph_s3),
        )
        for stage_index, (stage_name, blocks, graph) in enumerate(stages):
            if stage_index == 1:
                h = model.coarsen1(h)
            elif stage_index == 2:
                h = model.coarsen2(h)
            for index, block in enumerate(blocks):
                h, block_diag = block(h, graph=graph, training=False, return_diagnostics=True)
                tape.watch(h)
                name = f"{stage_name}_block{index}"
                block_tensors.append((name, h))
                scalar_diagnostics[name] = {
                    key: float(value.numpy())
                    for key, value in block_diag.items()
                    if value.shape.rank == 0
                }
        h = model.coarsen3(h)
        for index, block in enumerate(model.stage4_blocks):
            h, block_diag = block(
                h,
                coarse_graph=model.graph_coarse,
                training=False,
                return_diagnostics=True,
            )
            tape.watch(h)
            name = f"coarse_block{index}"
            block_tensors.append((name, h))
            scalar_diagnostics[name] = {
                key: float(value.numpy())
                for key, value in block_diag.items()
                if value.shape.rank == 0
            }
        normalized = model.readout_norm(h)
        pooled = tf.concat(
            [tf.reduce_mean(normalized, axis=1), tf.reduce_max(normalized, axis=1)], axis=-1
        )
        logits = model.classifier(model.head_dense1(pooled))
        objective = tf.reduce_sum(tf.square(logits))

    gradients = tape.gradient(objective, [tensor for _, tensor in block_tensors])
    result = {}
    for (name, tensor), gradient in zip(block_tensors, gradients):
        if gradient is None:
            raise RuntimeError(f"Gradient unavailable for graph block {name}")
        result[name] = compute_feature_diagnostics(tensor)
        result[name]["gradient_norm"] = float(tf.norm(gradient).numpy())
        result[name].update(scalar_diagnostics[name])
    del tape
    return result


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

    flat_grid = tf.reshape(h_grid, [-1, height * width, c])
    border = tf.boolean_mask(flat_grid, tf.reshape(mask_tf, [-1]), axis=1)
    border_norm = tf.reduce_mean(tf.norm(border, axis=-1))

    return {
        "interior_activation_norm": float(int_norm.numpy()),
        "border_activation_norm": float(border_norm.numpy()),
        "border_to_interior_ratio": float((border_norm / (int_norm + 1e-8)).numpy()),
    }


def compute_boundary_gradient_diagnostics(
    gradient: tf.Tensor,
    height: int = 48,
    width: int = 48,
) -> Dict[str, float]:
    """Compute border/interior norms for a gradient over Stage-1 features."""
    values = compute_boundary_diagnostics(gradient, height=height, width=width)
    return {
        "interior_gradient_norm": values["interior_activation_norm"],
        "border_gradient_norm": values["border_activation_norm"],
        "border_to_interior_gradient_ratio": values["border_to_interior_ratio"],
    }


def _cosine_mean(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    left = tf.reshape(left, [-1, tf.shape(left)[-1]])
    right = tf.reshape(right, [-1, tf.shape(right)[-1]])
    numerator = tf.reduce_sum(left * right, axis=-1)
    denominator = tf.norm(left, axis=-1) * tf.norm(right, axis=-1) + 1e-12
    return tf.reduce_mean(numerator / denominator)


def coarsening_shift_probe(
    coarsener: tf.keras.layers.Layer,
    sample_features: tf.Tensor,
) -> Dict[str, float]:
    """Diagnose the fixed coarsener under 1px and stride-aligned 2px shifts.

    This intentionally probes the coarsener output, not model probabilities, and
    makes no claim of perfect translation invariance.
    """
    height = int(coarsener.in_height)
    width = int(coarsener.in_width)
    channels = int(coarsener.in_channels)
    features = tf.reshape(sample_features, [-1, height, width, channels])

    def shift(amount: int) -> tf.Tensor:
        cropped = features[:, : height - amount, : width - amount, :]
        return tf.pad(cropped, [[0, 0], [amount, 0], [amount, 0], [0, 0]])

    base = tf.reshape(
        coarsener(tf.reshape(features, [-1, height * width, channels])),
        [-1, height // 2, width // 2, coarsener.out_channels],
    )
    shifted_1 = tf.reshape(
        coarsener(tf.reshape(shift(1), [-1, height * width, channels])),
        tf.shape(base),
    )
    shifted_2 = tf.reshape(
        coarsener(tf.reshape(shift(2), [-1, height * width, channels])),
        tf.shape(base),
    )

    # A 2px input translation corresponds to one coarse cell.  Exclude the
    # outer coarse boundary on both aligned tensors.
    base_2_interior = base[:, 1:-1, 1:-1, :]
    shifted_2_aligned = shifted_2[:, 2:, 2:, :]
    base_1_interior = base[:, 1:-1, 1:-1, :]
    shifted_1_interior = shifted_1[:, 1:-1, 1:-1, :]

    return {
        "shift_2px_aligned_interior_max_abs_error": float(
            tf.reduce_max(tf.abs(base_2_interior - shifted_2_aligned)).numpy()
        ),
        "shift_2px_aligned_interior_cosine": float(
            _cosine_mean(base_2_interior, shifted_2_aligned).numpy()
        ),
        "shift_1px_interior_max_abs_error": float(
            tf.reduce_max(tf.abs(base_1_interior - shifted_1_interior)).numpy()
        ),
        "shift_1px_interior_cosine": float(
            _cosine_mean(base_1_interior, shifted_1_interior).numpy()
        ),
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
