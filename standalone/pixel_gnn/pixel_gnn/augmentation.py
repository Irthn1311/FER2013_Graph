"""Data Augmentation module for Pixel GNN on FER2013.

Transforms only applied during training on 48x48 images:
1. Random horizontal flip (p=0.5)
2. Random contrast (factor in [0.9, 1.1])
3. Random brightness (delta in [-0.08, 0.08])
4. Clip values to [0.0, 1.0]
5. Recomputes exact central-difference spatial gradients gx, gy
6. Re-assembles [I, x, y, gx, gy] node features of shape [B, 2304, 5]
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology


def compute_image_gradients(imgs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute spatial central differences gx, gy for a batch of [B, 48, 48] images.
    
    Identical to np.gradient(imgs, axis=(1, 2)) to machine precision:
      gx = dI / dx (horizontal along columns, axis=2)
      gy = dI / dy (vertical along rows, axis=1)
    """
    gx = tf.concat([
        imgs[:, :, 1:2] - imgs[:, :, 0:1],
        (imgs[:, :, 2:] - imgs[:, :, :-2]) / 2.0,
        imgs[:, :, 47:48] - imgs[:, :, 46:47],
    ], axis=2)

    gy = tf.concat([
        imgs[:, 1:2, :] - imgs[:, 0:1, :],
        (imgs[:, 2:, :] - imgs[:, :-2, :]) / 2.0,
        imgs[:, 47:48, :] - imgs[:, 46:47, :],
    ], axis=1)

    return gy, gx


def compute_image_laplacian(gy: tf.Tensor, gx: tf.Tensor) -> tf.Tensor:
    """Compute exact second-derivative discrete Laplacian d^2I/dx^2 + d^2I/dy^2."""
    gyy = tf.concat([
        gy[:, 1:2, :] - gy[:, 0:1, :],
        (gy[:, 2:, :] - gy[:, :-2, :]) / 2.0,
        gy[:, 47:48, :] - gy[:, 46:47, :],
    ], axis=1)

    gxx = tf.concat([
        gx[:, :, 1:2] - gx[:, :, 0:1],
        (gx[:, :, 2:] - gx[:, :, :-2]) / 2.0,
        gx[:, :, 47:48] - gx[:, :, 46:47],
    ], axis=2)

    return gxx + gyy


def apply_random_cutout(
    imgs: tf.Tensor,
    cutout_prob: float = 0.3,
    min_size: int = 8,
    max_size: int = 14,
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Mask out a random rectangular patch to simulate partial facial occlusion (hands, hair, glasses)."""
    batch_size = tf.shape(imgs)[0]
    height, width = 48, 48

    apply_mask = tf.random.uniform([batch_size], 0.0, 1.0) < cutout_prob
    h_sizes = tf.random.uniform([batch_size], min_size, max_size + 1, dtype=tf.int32)
    w_sizes = tf.random.uniform([batch_size], min_size, max_size + 1, dtype=tf.int32)

    y0 = tf.random.uniform([batch_size], 0, height - max_size + 1, dtype=tf.int32)
    x0 = tf.random.uniform([batch_size], 0, width - max_size + 1, dtype=tf.int32)

    yy, xx = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")
    yy = tf.expand_dims(yy, 0)
    xx = tf.expand_dims(xx, 0)

    y0_exp = tf.reshape(y0, [-1, 1, 1])
    x0_exp = tf.reshape(x0, [-1, 1, 1])
    h_exp = tf.reshape(h_sizes, [-1, 1, 1])
    w_exp = tf.reshape(w_sizes, [-1, 1, 1])

    in_box = (yy >= y0_exp) & (yy < y0_exp + h_exp) & (xx >= x0_exp) & (xx < x0_exp + w_exp)
    apply_exp = tf.reshape(apply_mask, [-1, 1, 1])
    mask = in_box & apply_exp

    return tf.where(mask, tf.constant(fill_value, dtype=imgs.dtype), imgs)


def apply_pixel_mixup(
    batch: dict[str, tf.Tensor],
    mixup_alpha: float = 0.2,
    node_dim: int = 5,
) -> dict[str, tf.Tensor]:
    """Pure Pixel Graph Mixup: Interpolate images X_mix = lambda*X1 + (1-lambda)*X2 and labels Y_mix."""
    imgs = tf.cast(batch["image_48"], tf.float32)  # [B, 48, 48]
    batch_size = tf.shape(imgs)[0]
    num_classes = 7

    # Shuffle indices
    perm = tf.random.shuffle(tf.range(batch_size))
    imgs_shuffled = tf.gather(imgs, perm)

    # Convert labels to soft one-hot
    raw_labels = batch["labels"]
    if len(raw_labels.shape) == 1:
        labels_one_hot = tf.one_hot(raw_labels, depth=num_classes, dtype=tf.float32)
    else:
        labels_one_hot = tf.cast(raw_labels, tf.float32)
    labels_shuffled = tf.gather(labels_one_hot, perm)

    # Sample lambda ~ Beta(alpha, alpha) using Gamma distribution: Gamma(alpha, 1) / (Gamma(alpha, 1) + Gamma(alpha, 1))
    gamma1 = tf.random.gamma([batch_size, 1, 1], mixup_alpha, 1.0, dtype=tf.float32)
    gamma2 = tf.random.gamma([batch_size, 1, 1], mixup_alpha, 1.0, dtype=tf.float32)
    lam = gamma1 / (gamma1 + gamma2 + 1e-8)  # [B, 1, 1]

    # Mix images and labels
    imgs_mixed = lam * imgs + (1.0 - lam) * imgs_shuffled
    imgs_mixed = tf.clip_by_value(imgs_mixed, 0.0, 1.0)

    lam_labels = tf.reshape(lam, [batch_size, 1])
    labels_mixed = lam_labels * labels_one_hot + (1.0 - lam_labels) * labels_shuffled

    # Recompute spatial gradients & features for mixed image
    gy, gx = compute_image_gradients(imgs_mixed)

    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords  # [2304, 2]
    coords_exp = tf.broadcast_to(tf.expand_dims(coords, axis=0), [batch_size, 2304, 2])

    intensity = tf.reshape(imgs_mixed, [batch_size, 2304, 1])
    gx_flat = tf.reshape(gx, [batch_size, 2304, 1])
    gy_flat = tf.reshape(gy, [batch_size, 2304, 1])

    features_list = [intensity, coords_exp, gx_flat, gy_flat]
    if node_dim >= 7:
        grad_mag = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)
        laplacian = compute_image_laplacian(gy, gx)
        grad_mag_flat = tf.reshape(grad_mag, [batch_size, 2304, 1])
        laplacian_flat = tf.reshape(laplacian, [batch_size, 2304, 1])
        features_list.extend([grad_mag_flat, laplacian_flat])

    if node_dim == 9:
        neighbors_idx = grid.neighbors_idx.numpy()
        neighbor_valid = grid.neighbor_valid.numpy()
        valid_mask = tf.constant(neighbor_valid[None, :, :], dtype=tf.float32)

        intensity_2d = tf.squeeze(intensity, axis=-1)
        nbr_intensity = tf.gather(intensity_2d, neighbors_idx, axis=1)
        diff = nbr_intensity - tf.expand_dims(intensity_2d, axis=-1)

        var_flat = tf.reduce_sum(tf.square(diff) * valid_mask, axis=-1, keepdims=True) / (
            tf.reduce_sum(valid_mask, axis=-1, keepdims=True) + 1e-6
        )
        powers = tf.constant((2 ** np.arange(8, dtype=np.float32))[None, None, :], dtype=tf.float32)
        lbp_bits = tf.cast(diff >= 0.0, tf.float32) * valid_mask
        lbp_flat = tf.reduce_sum(lbp_bits * powers, axis=-1, keepdims=True) / 255.0
        features_list.extend([var_flat, lbp_flat])

    node_features = tf.concat(features_list, axis=-1)

    return {
        "node_features": node_features,
        "labels": labels_mixed,
        "sample_ids": batch.get("sample_ids", None),
        "image_48": imgs_mixed,
    }


def augment_batch(
    batch: dict[str, tf.Tensor],
    flip_prob: float = 0.5,
    brightness_delta: float = 0.08,
    contrast_range: tuple[float, float] = (0.9, 1.1),
    cutout_prob: float = 0.0,
    cutout_min_size: int = 8,
    cutout_max_size: int = 14,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.2,
    node_dim: int = 5,
) -> dict[str, tf.Tensor]:
    """Apply data augmentation, recompute gradients/Laplacian, and rebuild node features."""
    imgs = tf.cast(batch["image_48"], tf.float32)  # [B, 48, 48]
    batch_size = tf.shape(imgs)[0]

    # 1. Random horizontal flip
    if flip_prob > 0.0:
        flips = tf.random.uniform([batch_size, 1, 1], minval=0.0, maxval=1.0) < flip_prob
        imgs = tf.where(flips, tf.reverse(imgs, axis=[2]), imgs)

    # 2. Random contrast
    if contrast_range is not None and (contrast_range[0] != 1.0 or contrast_range[1] != 1.0):
        means = tf.reduce_mean(imgs, axis=[1, 2], keepdims=True)
        factors = tf.random.uniform(
            [batch_size, 1, 1],
            minval=contrast_range[0],
            maxval=contrast_range[1],
            dtype=tf.float32,
        )
        imgs = (imgs - means) * factors + means

    # 3. Random brightness
    if brightness_delta > 0.0:
        deltas = tf.random.uniform(
            [batch_size, 1, 1],
            minval=-brightness_delta,
            maxval=brightness_delta,
            dtype=tf.float32,
        )
        imgs = imgs + deltas

    # 4. Clip to valid [0.0, 1.0] intensity range
    imgs = tf.clip_by_value(imgs, 0.0, 1.0)

    # 5. Random Cutout / Occlusion
    if cutout_prob > 0.0:
        imgs = apply_random_cutout(
            imgs,
            cutout_prob=cutout_prob,
            min_size=cutout_min_size,
            max_size=cutout_max_size,
            fill_value=0.0,
        )

    # 6. Recompute spatial gradients on augmented image
    gy, gx = compute_image_gradients(imgs)

    # 7. Reconstruct node features
    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords  # [2304, 2]
    coords_exp = tf.broadcast_to(tf.expand_dims(coords, axis=0), [batch_size, 2304, 2])

    intensity = tf.reshape(imgs, [batch_size, 2304, 1])
    gx_flat = tf.reshape(gx, [batch_size, 2304, 1])
    gy_flat = tf.reshape(gy, [batch_size, 2304, 1])

    features_list = [intensity, coords_exp, gx_flat, gy_flat]
    if node_dim >= 7:
        grad_mag = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)
        laplacian = compute_image_laplacian(gy, gx)
        grad_mag_flat = tf.reshape(grad_mag, [batch_size, 2304, 1])
        laplacian_flat = tf.reshape(laplacian, [batch_size, 2304, 1])
        features_list.extend([grad_mag_flat, laplacian_flat])

    if node_dim == 9:
        neighbors_idx = grid.neighbors_idx.numpy()
        neighbor_valid = grid.neighbor_valid.numpy()
        valid_mask = tf.constant(neighbor_valid[None, :, :], dtype=tf.float32)

        intensity_2d = tf.squeeze(intensity, axis=-1)
        nbr_intensity = tf.gather(intensity_2d, neighbors_idx, axis=1)
        diff = nbr_intensity - tf.expand_dims(intensity_2d, axis=-1)

        var_flat = tf.reduce_sum(tf.square(diff) * valid_mask, axis=-1, keepdims=True) / (
            tf.reduce_sum(valid_mask, axis=-1, keepdims=True) + 1e-6
        )
        powers = tf.constant((2 ** np.arange(8, dtype=np.float32))[None, None, :], dtype=tf.float32)
        lbp_bits = tf.cast(diff >= 0.0, tf.float32) * valid_mask
        lbp_flat = tf.reduce_sum(lbp_bits * powers, axis=-1, keepdims=True) / 255.0
        features_list.extend([var_flat, lbp_flat])

    node_features = tf.concat(features_list, axis=-1)

    aug_batch = {
        "node_features": node_features,
        "labels": batch["labels"],
        "sample_ids": batch.get("sample_ids", None),
        "image_48": imgs,
    }

    # 8. Pure Pixel Graph Mixup
    if mixup_prob > 0.0 and tf.random.uniform([], 0.0, 1.0) < mixup_prob:
        aug_batch = apply_pixel_mixup(aug_batch, mixup_alpha=mixup_alpha, node_dim=node_dim)

    return aug_batch


def make_flipped_batch(batch: dict[str, tf.Tensor], node_dim: int = 5) -> dict[str, tf.Tensor]:
    """Deterministically horizontally flip a batch of images and recompute exact node features for TTA."""
    imgs = tf.cast(batch["image_48"], tf.float32)  # [B, 48, 48]
    batch_size = tf.shape(imgs)[0]

    # Horizontal flip along width axis (axis 2)
    imgs_flipped = tf.reverse(imgs, axis=[2])

    # Recompute central difference spatial gradients on flipped image
    gy, gx = compute_image_gradients(imgs_flipped)

    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords  # [2304, 2]
    coords_exp = tf.broadcast_to(tf.expand_dims(coords, axis=0), [batch_size, 2304, 2])

    intensity = tf.reshape(imgs_flipped, [batch_size, 2304, 1])
    gx_flat = tf.reshape(gx, [batch_size, 2304, 1])
    gy_flat = tf.reshape(gy, [batch_size, 2304, 1])

    features_list = [intensity, coords_exp, gx_flat, gy_flat]
    if node_dim == 7:
        grad_mag = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)
        laplacian = compute_image_laplacian(gy, gx)
        grad_mag_flat = tf.reshape(grad_mag, [batch_size, 2304, 1])
        laplacian_flat = tf.reshape(laplacian, [batch_size, 2304, 1])
        features_list.extend([grad_mag_flat, laplacian_flat])

    node_features = tf.concat(features_list, axis=-1)

    return {
        "node_features": node_features,
        "labels": batch["labels"],
        "sample_ids": batch.get("sample_ids", None),
        "image_48": imgs_flipped,
    }
