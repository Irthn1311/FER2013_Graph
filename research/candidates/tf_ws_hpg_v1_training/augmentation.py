"""Registered stateless paired image/support augmentation."""

from __future__ import annotations

import math

import tensorflow as tf

SEED = 42


def _seed(augmentation_index, salt):
    return tf.random.experimental.stateless_fold_in(
        tf.stack(
            [tf.constant(SEED, tf.int32), tf.cast(augmentation_index, tf.int32)]
        ),
        salt,
    )


def sample_parameters(augmentation_index):
    """Sample from seed 42 and the accepted post-shuffle enumeration index."""

    flip = tf.random.stateless_uniform([], _seed(augmentation_index, 0)) < 0.5
    angle = tf.random.stateless_uniform([], _seed(augmentation_index, 1), minval=-10.0, maxval=10.0)
    translation = tf.random.stateless_uniform([2], _seed(augmentation_index, 2), minval=-4.0, maxval=4.0)
    contrast = tf.random.stateless_uniform([], _seed(augmentation_index, 3), minval=0.85, maxval=1.15)
    brightness = tf.random.stateless_uniform([], _seed(augmentation_index, 4), minval=-0.10, maxval=0.10)
    erase = tf.random.stateless_uniform([], _seed(augmentation_index, 5)) < 0.25
    area = tf.random.stateless_uniform([], _seed(augmentation_index, 6), minval=0.02, maxval=0.10)
    aspect = tf.random.stateless_uniform([], _seed(augmentation_index, 7), minval=0.5, maxval=2.0)
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


def _erase(image, augmentation_index, parameters):
    area = parameters["erase_area_fraction"] * 48.0 * 48.0
    height = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area / parameters["erase_aspect"])), tf.int32), 1, 48)
    width = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area * parameters["erase_aspect"])), tf.int32), 1, 48)
    top = tf.random.stateless_uniform([], _seed(augmentation_index, 8), 0, 48 - height + 1, dtype=tf.int32)
    left = tf.random.stateless_uniform([], _seed(augmentation_index, 9), 0, 48 - width + 1, dtype=tf.int32)
    rows, columns = tf.range(48)[:, None], tf.range(48)[None, :]
    mask = tf.logical_and(
        tf.logical_and(rows >= top, rows < top + height),
        tf.logical_and(columns >= left, columns < left + width),
    )
    return tf.where(mask[..., None], 0.0, image)


def _paired_geometry(image, support, parameters):
    angle = parameters["rotation_degrees"] * (math.pi / 180.0)
    flip = tf.where(parameters["flip"], -1.0, 1.0)
    cosine, sine = tf.cos(angle), tf.sin(angle)
    a0, a1 = flip * cosine, flip * sine
    b0, b1 = -sine, cosine
    center = tf.constant(23.5, tf.float32)
    tx, ty = parameters["translation_x"], parameters["translation_y"]
    a2 = center - a0 * (center + tx) - a1 * (center + ty)
    b2 = center - b0 * (center + tx) - b1 * (center + ty)
    vector = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None]
    kwargs = {
        "transforms": vector,
        "output_shape": tf.constant([48, 48], tf.int32),
        "interpolation": "BILINEAR",
        "fill_mode": "CONSTANT",
    }
    warped_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None], fill_value=0.0, **kwargs
    )[0]
    warped_support = tf.raw_ops.ImageProjectiveTransformV3(
        images=support[None], fill_value=1.0, **kwargs
    )[0]
    return warped_image, warped_support, {
        "image_transform": vector,
        "support_transform": tf.identity(vector),
    }


def augment_example(raw_image, support, augmentation_index, *, return_debug=False):
    """Augment image/support with shared geometry and image-only appearance."""

    parameters = sample_parameters(augmentation_index)
    image = tf.cast(raw_image, tf.float32) / 255.0
    field = tf.cast(support, tf.float32)
    image, field, geometry = _paired_geometry(image, field, parameters)
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * parameters["contrast"] + mean
    image = tf.clip_by_value(image + parameters["brightness"], 0.0, 1.0)
    image = tf.cond(
        parameters["erase"],
        lambda: _erase(image, augmentation_index, parameters),
        lambda: image,
    )
    inputs = {"images": image, "support": field}
    if return_debug:
        return inputs, geometry, parameters
    return inputs, geometry, parameters
