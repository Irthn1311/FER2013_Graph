"""Deterministic local detail feature maps for D16 pixel graph nodes."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


DEFAULT_DETAIL_FEATURES = [
    "grad_mag",
    "local_mean_3x3",
    "local_std_3x3",
    "laplacian_abs",
    "center_surround",
]


def _mean3x3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    acc = np.zeros_like(image, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return acc / 9.0


def _laplacian_abs(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    lap = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return np.abs(lap).astype(np.float32)


def compute_detail_feature_maps(
    image_norm: np.ndarray,
    feature_names: Iterable[str] | None = None,
    normalize: str = "per_image_safe",
) -> Dict[str, np.ndarray]:
    """Compute local texture/detail descriptors on a normalized 48x48 image."""

    image = np.asarray(image_norm, dtype=np.float32)
    if image.shape != (48, 48):
        raise ValueError(f"D16 detail features expect image shape (48, 48), got {image.shape}")
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    names = list(feature_names or DEFAULT_DETAIL_FEATURES)
    unknown = sorted(set(names) - set(DEFAULT_DETAIL_FEATURES))
    if unknown:
        raise ValueError(f"Unknown D16 detail node features: {unknown}")
    if str(normalize) != "per_image_safe":
        raise ValueError(f"Unsupported D16 detail feature normalization: {normalize!r}")

    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    local_mean = _mean3x3(image)
    local_sq_mean = _mean3x3(image * image)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var + 1e-12).astype(np.float32)
    lap_abs = _laplacian_abs(image)
    center_surround = image - local_mean

    maps = {
        "grad_mag": np.clip(grad_mag, 0.0, 1.0).astype(np.float32),
        "local_mean_3x3": np.clip(local_mean, 0.0, 1.0).astype(np.float32),
        "local_std_3x3": np.clip(local_std, 0.0, 0.5).astype(np.float32),
        "laplacian_abs": np.clip(lap_abs, 0.0, 1.0).astype(np.float32),
        "center_surround": np.clip(center_surround, -1.0, 1.0).astype(np.float32),
    }
    return {name: np.nan_to_num(maps[name], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32) for name in names}


def sample_detail_features(
    image_norm: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    feature_names: Iterable[str] | None = None,
    normalize: str = "per_image_safe",
) -> tuple[np.ndarray, List[str]]:
    """Return a [num_nodes, num_detail_features] matrix sampled at node coords."""

    names = list(feature_names or DEFAULT_DETAIL_FEATURES)
    maps = compute_detail_feature_maps(image_norm, names, normalize=normalize)
    if not names:
        return np.zeros((len(yy), 0), dtype=np.float32), names
    cols = [maps[name][yy, xx].astype(np.float32)[:, None] for name in names]
    return np.concatenate(cols, axis=1).astype(np.float32), names
