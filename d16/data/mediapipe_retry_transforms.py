"""Retry-time image transforms for D16 MediaPipe pixel priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from d16.data.mediapipe_priors import preprocess_image48_uint8


@dataclass(frozen=True)
class RetryTransformResult:
    image: np.ndarray
    inverse_xy: Callable[[np.ndarray], np.ndarray]
    skipped: bool = False
    skip_reason: str = ""


def _as_uint8(image48: np.ndarray) -> np.ndarray:
    image = np.asarray(image48, dtype=np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    normalized = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(normalized, float(gamma)) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _sharpen(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1] * 5.0
    blur_neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    return np.clip(center - blur_neighbors, 0, 255).astype(np.uint8)


def _resize_nearest(image: np.ndarray, size: int = 48) -> np.ndarray:
    yy = np.linspace(0, image.shape[0] - 1, int(size)).round().astype(np.int64)
    xx = np.linspace(0, image.shape[1] - 1, int(size)).round().astype(np.int64)
    return image[np.ix_(yy, xx)]


def _rotate(image: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore

        center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(degrees), 1.0).astype(np.float32)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated.astype(np.uint8), matrix
    except Exception:
        theta = np.deg2rad(float(degrees))
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        cx = cy = 23.5
        yy, xx = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(48, dtype=np.float32), indexing="ij")
        x0 = xx - cx
        y0 = yy - cy
        src_x = cos_t * x0 + sin_t * y0 + cx
        src_y = -sin_t * x0 + cos_t * y0 + cy
        src_x = np.clip(np.round(src_x), 0, 47).astype(np.int64)
        src_y = np.clip(np.round(src_y), 0, 47).astype(np.int64)
        rotated = image[src_y, src_x]
        matrix = np.asarray([[cos_t, sin_t, (1 - cos_t) * cx - sin_t * cy], [-sin_t, cos_t, sin_t * cx + (1 - cos_t) * cy]], dtype=np.float32)
        return rotated.astype(np.uint8), matrix


def build_retry_transform(image48: np.ndarray, mode: str) -> RetryTransformResult:
    mode = str(mode or "raw").lower()
    base = _as_uint8(image48)
    identity = lambda xy: np.asarray(xy, dtype=np.float32)

    if mode in ("raw", "none"):
        return RetryTransformResult(base, identity)
    if mode == "equalize":
        return RetryTransformResult(preprocess_image48_uint8(base, "equalize"), identity)
    if mode == "gamma_0_70":
        return RetryTransformResult(_gamma(base, 0.70), identity)
    if mode == "gamma_0_55":
        return RetryTransformResult(_gamma(base, 0.55), identity)
    if mode == "clahe":
        try:
            import cv2  # noqa: F401
        except Exception as exc:
            return RetryTransformResult(base, identity, skipped=True, skip_reason=f"cv2 unavailable for clahe: {exc}")
        return RetryTransformResult(preprocess_image48_uint8(base, "clahe"), identity)
    if mode == "sharpen":
        return RetryTransformResult(_sharpen(base), identity)
    if mode == "pad8":
        pad = 8
        padded = np.pad(base, ((pad, pad), (pad, pad)), mode="edge")
        resized = _resize_nearest(padded, 48)
        scale = float(padded.shape[0] - 1) / 47.0

        def inverse(xy: np.ndarray) -> np.ndarray:
            pts = np.asarray(xy, dtype=np.float32).copy()
            pts = pts * scale - float(pad)
            return pts

        return RetryTransformResult(resized, inverse)
    if mode == "hflip":
        flipped = np.fliplr(base)

        def inverse(xy: np.ndarray) -> np.ndarray:
            pts = np.asarray(xy, dtype=np.float32).copy()
            pts[:, 0] = 47.0 - pts[:, 0]
            return pts

        return RetryTransformResult(flipped, inverse)
    if mode in ("rot_m10", "rot_p10"):
        degrees = -10.0 if mode == "rot_m10" else 10.0
        rotated, matrix = _rotate(base, degrees)
        affine = np.vstack([matrix, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)])
        inverse_affine = np.linalg.inv(affine).astype(np.float32)

        def inverse(xy: np.ndarray) -> np.ndarray:
            pts = np.asarray(xy, dtype=np.float32)
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            homog = np.concatenate([pts, ones], axis=1)
            mapped = homog @ inverse_affine.T
            return mapped[:, :2]

        return RetryTransformResult(rotated, inverse)
    raise ValueError(f"Unsupported retry preprocess mode: {mode}")
