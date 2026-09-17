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


def augment_batch(
    batch: dict[str, tf.Tensor],
    flip_prob: float = 0.5,
    brightness_delta: float = 0.08,
    contrast_range: tuple[float, float] = (0.9, 1.1),
) -> dict[str, tf.Tensor]:
    """Apply data augmentation to image_48 batch, recompute gradients, and rebuild node features."""
    imgs = tf.cast(batch["image_48"], tf.float32)  # [B, 48, 48]
    batch_size = tf.shape(imgs)[0]

    # 1. Random horizontal flip (p=0.5)
    if flip_prob > 0.0:
        flips = tf.random.uniform([batch_size, 1, 1], minval=0.0, maxval=1.0) < flip_prob
        imgs = tf.where(flips, tf.reverse(imgs, axis=[2]), imgs)

    # 2. Random contrast (factor in [0.9, 1.1])
    if contrast_range is not None and (contrast_range[0] != 1.0 or contrast_range[1] != 1.0):
        means = tf.reduce_mean(imgs, axis=[1, 2], keepdims=True)
        factors = tf.random.uniform(
            [batch_size, 1, 1],
            minval=contrast_range[0],
            maxval=contrast_range[1],
            dtype=tf.float32,
        )
        imgs = (imgs - means) * factors + means

    # 3. Random brightness (delta in [-0.08, 0.08])
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

    # 5. Recompute spatial gradients on augmented image
    gy, gx = compute_image_gradients(imgs)

    # 6. Reconstruct [I, x, y, gx, gy] node features
    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords  # [2304, 2]
    coords_exp = tf.broadcast_to(tf.expand_dims(coords, axis=0), [batch_size, 2304, 2])

    intensity = tf.reshape(imgs, [batch_size, 2304, 1])
    gx_flat = tf.reshape(gx, [batch_size, 2304, 1])
    gy_flat = tf.reshape(gy, [batch_size, 2304, 1])

    node_features = tf.concat([intensity, coords_exp, gx_flat, gy_flat], axis=-1)

    return {
        "node_features": node_features,
        "labels": batch["labels"],
        "sample_ids": batch["sample_ids"],
        "image_48": imgs,
    }


def make_flipped_batch(batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
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

    node_features = tf.concat([intensity, coords_exp, gx_flat, gy_flat], axis=-1)

    return {
        "node_features": node_features,
        "labels": batch["labels"],
        "sample_ids": batch.get("sample_ids", None),
        "image_48": imgs_flipped,
    }
