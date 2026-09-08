"""Weak facial-support construction and augmentation coupling.

Landmarks are reduced immediately to one global extent.  No landmark identity or
coordinate is returned or retained by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import tensorflow as tf


IMAGE_SIZE = 48
SUPPORT_SIGMA = 0.25
EXTENT_EXPANSION = 0.10
FER_PIXEL_COORDINATE_SYSTEM = "fer_pixel_xy_0_47"


class PixelLandmarkDetector(Protocol):
    """Existing detector contract: ``detect`` returns FER pixel-space x/y."""

    def detect(self, image48: np.ndarray) -> np.ndarray | None: ...


def support_from_landmarks(
    landmarks,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    *,
    coordinate_system: str,
):
    """Reduce FER pixel-space x/y points to the registered soft ellipse.

    ``coordinate_system`` is mandatory and must explicitly name the existing
    detector's pixel contract.  Normalized inputs are not accepted through a
    second mode and units are never inferred from numeric magnitude.
    """

    if coordinate_system != FER_PIXEL_COORDINATE_SYSTEM:
        raise ValueError(
            "landmarks must use explicit FER pixel coordinates x/y in [0,47]"
        )

    ones = np.ones((height, width, 1), dtype=np.float32)
    try:
        points = np.asarray(landmarks, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] == 0:
            return ones
        points = points[:, :2]
        if not np.all(np.isfinite(points)):
            return ones
        if (
            np.any(points[:, 0] < 0.0)
            or np.any(points[:, 0] > width - 1)
            or np.any(points[:, 1] < 0.0)
            or np.any(points[:, 1] > height - 1)
        ):
            return ones
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        raw_width, raw_height = xmax - xmin, ymax - ymin
        if raw_width <= 0.0 or raw_height <= 0.0:
            return ones
        xmin -= EXTENT_EXPANSION * raw_width
        xmax += EXTENT_EXPANSION * raw_width
        ymin -= EXTENT_EXPANSION * raw_height
        ymax += EXTENT_EXPANSION * raw_height
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        rx, ry = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
        if not np.isfinite([cx, cy, rx, ry]).all() or rx <= 0.0 or ry <= 0.0:
            return ones
        # MediaPipeFaceDetector.detect maps normalized coordinates to the FER
        # pixel-index frame by multiplying by 47 and clipping to [0,47].
        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        radius = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        support = np.where(
            radius <= 1.0,
            1.0,
            np.exp(-0.5 * ((radius - 1.0) / SUPPORT_SIGMA) ** 2),
        )
        return np.clip(support, 0.0, 1.0).astype(np.float32)[..., None]
    except (TypeError, ValueError, IndexError, FloatingPointError):
        return ones


def build_support_from_clean_image(
    clean_image: np.ndarray, detector: PixelLandmarkDetector
) -> np.ndarray:
    """Adapt the existing pixel-space ``MediaPipeFaceDetector.detect`` API."""

    try:
        result = detector.detect(clean_image)
        if result is None:
            return np.ones((*clean_image.shape[:2], 1), np.float32)
        return support_from_landmarks(
            result,
            clean_image.shape[0],
            clean_image.shape[1],
            coordinate_system=FER_PIXEL_COORDINATE_SYSTEM,
        )
    except (TypeError, ValueError, AttributeError, IndexError):
        return np.ones((*clean_image.shape[:2], 1), np.float32)


@dataclass(frozen=True)
class GeometricTransform:
    horizontal_flip: bool = False
    rotation_radians: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0


def _projective_vector(transform: GeometricTransform, height, width, dtype):
    """Output-to-input affine transform for ImageProjectiveTransformV3."""

    angle = tf.cast(transform.rotation_radians, dtype)
    flip = tf.cast(-1.0 if transform.horizontal_flip else 1.0, dtype)
    cosine, sine = tf.cos(angle), tf.sin(angle)
    # Inverse linear transform: inverse(rotation * optional horizontal flip).
    a0, a1 = flip * cosine, flip * sine
    b0, b1 = -sine, cosine
    cx = (tf.cast(width, dtype) - 1.0) / 2.0
    cy = (tf.cast(height, dtype) - 1.0) / 2.0
    tx = tf.cast(transform.translation_x, dtype)
    ty = tf.cast(transform.translation_y, dtype)
    a2 = cx - a0 * (cx + tx) - a1 * (cy + ty)
    b2 = cy - b0 * (cx + tx) - b1 * (cy + ty)
    return tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])


def apply_geometric_transform(images, support, transform: GeometricTransform, return_parameters=False):
    """Apply one exactly shared flip/rotation/translation to image and support."""

    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    support = tf.cast(tf.convert_to_tensor(support), tf.float32)
    tf.debugging.assert_equal(tf.shape(images)[:3], tf.shape(support)[:3])
    vector = _projective_vector(transform, tf.shape(images)[1], tf.shape(images)[2], images.dtype)
    vectors = tf.broadcast_to(vector[None, :], [tf.shape(images)[0], 8])
    warped_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=vectors,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    # One-valued fill preserves the registered detector-failure == no-prior
    # invariant, while using the exact same affine vector and interpolation.
    warped_support = tf.raw_ops.ImageProjectiveTransformV3(
        images=support,
        transforms=vectors,
        output_shape=tf.shape(support)[1:3],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=1.0,
    )
    if return_parameters:
        return warped_image, warped_support, {
            "image_transform": vectors,
            "support_transform": tf.identity(vectors),
        }
    return warped_image, warped_support


def photometric_image_only(images, support, brightness_delta=0.0, contrast_factor=1.0):
    adjusted = tf.image.adjust_brightness(images, brightness_delta)
    adjusted = tf.image.adjust_contrast(adjusted, contrast_factor)
    return adjusted, tf.identity(support)


def erase_image_only(images, support, top: int, left: int, height: int, width: int):
    mask = tf.ones_like(images)
    rows = tf.range(tf.shape(images)[1])[None, :, None, None]
    cols = tf.range(tf.shape(images)[2])[None, None, :, None]
    erased = tf.logical_and(
        tf.logical_and(rows >= top, rows < top + height),
        tf.logical_and(cols >= left, cols < left + width),
    )
    mask = tf.where(erased, tf.zeros_like(mask), mask)
    return images * mask, tf.identity(support)


def patch_support_scores(support):
    support = tf.cast(tf.convert_to_tensor(support), tf.float32)
    tf.debugging.assert_equal(tf.shape(support)[1:], [48, 48, 1])
    pooled = tf.nn.avg_pool2d(support, ksize=3, strides=3, padding="VALID")
    return tf.reshape(pooled, [tf.shape(support)[0], 256, 1])
