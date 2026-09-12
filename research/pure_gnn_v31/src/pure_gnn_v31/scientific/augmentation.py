"""Preregistered Gen2/Gen3 stateless deterministic image augmentation policy for Pure-GNN v3.1."""

from __future__ import annotations

import math
from typing import Dict
import tensorflow as tf

POLICY_NAME = "gen2_gen3_stateless_image_v1"


def _derive_stateless_seed(base_seed: int, epoch: int, item_index: tf.Tensor, salt: int) -> tf.Tensor:
    """Derives a deterministic stateless seed tensor of shape [2] from base_seed, epoch, item_index, and salt.

    Formula:
      base = [base_seed, epoch * 10007]
      s1 = stateless_fold_in(base, item_index)
      seed = stateless_fold_in(s1, salt)
    """
    base = tf.stack([
        tf.cast(base_seed, tf.int32),
        tf.cast(epoch * 10007, tf.int32),
    ])
    folded_idx = tf.random.experimental.stateless_fold_in(base, tf.cast(item_index, tf.int32))
    return tf.random.experimental.stateless_fold_in(folded_idx, salt)


def sample_augmentation_parameters(base_seed: int, epoch: int, item_index: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Deterministically samples augmentation parameters for a single sample at a given epoch.

    Ranges:
      - horizontal_flip: p = 0.5
      - rotation_degrees: [-10.0, +10.0]
      - translation_pixels: [-4.0, +4.0]
      - contrast: [0.85, 1.15]
      - brightness_delta: [-0.10, +0.10]
      - random_erase: p = 0.25
      - erase_area_fraction: [0.02, 0.10]
      - erase_aspect_ratio: [0.5, 2.0]
    """
    flip = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 0)) < 0.5
    angle = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 1), minval=-10.0, maxval=10.0)
    translation = tf.random.stateless_uniform([2], _derive_stateless_seed(base_seed, epoch, item_index, 2), minval=-4.0, maxval=4.0)
    contrast = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 3), minval=0.85, maxval=1.15)
    brightness = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 4), minval=-0.10, maxval=0.10)
    erase = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 5)) < 0.25
    area = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 6), minval=0.02, maxval=0.10)
    aspect = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 7), minval=0.5, maxval=2.0)

    return {
        "flip": flip,
        "rotation_degrees": angle,
        "translation_x": translation[0],
        "translation_y": translation[1],
        "contrast": contrast,
        "brightness": brightness,
        "erase": erase,
        "erase_area_fraction": area,
        "erase_aspect": aspect,
    }


def _apply_geometry(image: tf.Tensor, params: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Applies Gen3-style rotation, translation, and horizontal flip using affine transform on [48, 48, 1].

    Uses CONSTANT fill mode with 0.0 fill value.
    """
    angle = params["rotation_degrees"] * (math.pi / 180.0)
    flip = tf.where(params["flip"], -1.0, 1.0)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = flip * cos_a
    a1 = flip * sin_a
    b0 = -sin_a
    b1 = cos_a

    center = tf.constant(23.5, tf.float32)
    tx = params["translation_x"]
    ty = params["translation_y"]

    a2 = center - a0 * (center + tx) - a1 * (center + ty)
    b2 = center - b0 * (center + tx) - b1 * (center + ty)

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    kwargs = {
        "transforms": transform,
        "output_shape": tf.constant([48, 48], tf.int32),
        "interpolation": "BILINEAR",
        "fill_mode": "CONSTANT",
    }
    warped = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        fill_value=0.0,
        **kwargs,
    )[0]
    return warped


def _apply_photometric(image: tf.Tensor, params: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Applies contrast scaling and brightness offset, clipping to [0.0, 1.0].

    Uses 1D reduction sum to guarantee bit-identical multi-threaded evaluation order across CPU runs.
    """
    mean = tf.reduce_sum(tf.reshape(image, [-1])) / 2304.0
    img = (image - mean) * params["contrast"] + mean
    img = img + params["brightness"]
    return tf.clip_by_value(img, 0.0, 1.0)


def _apply_random_erase(image: tf.Tensor, base_seed: int, epoch: int, item_index: tf.Tensor, params: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Applies random rectangular zero-erasing if params['erase'] is True."""
    def erase_fn():
        area = params["erase_area_fraction"] * 48.0 * 48.0
        h = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area / params["erase_aspect"])), tf.int32), 1, 48)
        w = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area * params["erase_aspect"])), tf.int32), 1, 48)

        top = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 8), 0, 48 - h + 1, dtype=tf.int32)
        left = tf.random.stateless_uniform([], _derive_stateless_seed(base_seed, epoch, item_index, 9), 0, 48 - w + 1, dtype=tf.int32)

        rows = tf.range(48)[:, None]
        cols = tf.range(48)[None, :]
        mask = tf.logical_and(
            tf.logical_and(rows >= top, rows < top + h),
            tf.logical_and(cols >= left, cols < left + w),
        )
        return tf.where(mask[..., None], 0.0, image)

    return tf.cond(params["erase"], erase_fn, lambda: image)


def augment_image_stateless(image: tf.Tensor, base_seed: int, epoch: int, item_index: tf.Tensor) -> tf.Tensor:
    """Applies the preregistered Gen2/Gen3 stateless image augmentation pipeline on a [48, 48, 1] tensor in [0, 1]."""
    params = sample_augmentation_parameters(base_seed, epoch, item_index)
    img = _apply_geometry(image, params)
    img = _apply_photometric(img, params)
    img = _apply_random_erase(img, base_seed, epoch, item_index, params)
    return img
