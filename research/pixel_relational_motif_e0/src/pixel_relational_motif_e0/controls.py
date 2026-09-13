from __future__ import annotations

import numpy as np


def destroy_ordered_relations(s: np.ndarray, *, seed: int) -> np.ndarray:
    """Independently permute the 24 relation offsets in every descriptor row.

    Each row keeps its exact multiset of relation values; only ordered spatial
    correspondence is destroyed. The input is never modified in place.
    """
    x = np.asarray(s)
    if x.ndim != 2 or x.shape[1] != 24:
        raise ValueError("expected relation matrix with shape (n,24)")
    rng = np.random.default_rng(seed)
    keys = rng.random(x.shape)
    perm = np.argsort(keys, axis=1)
    return np.take_along_axis(x, perm, axis=1)


def geometry_permutation(*, image_index: int, seed: int, n_bins: int = 8) -> np.ndarray:
    if n_bins <= 1:
        raise ValueError("n_bins must exceed one")
    ss = np.random.SeedSequence([int(seed), int(image_index), int(n_bins)])
    return np.random.default_rng(ss).permutation(n_bins)


def shuffle_geometry_bins(
    tensor: np.ndarray,
    *,
    image_index: int,
    seed: int,
) -> np.ndarray:
    """Permute the final geometry-bin axis without changing pair totals."""
    x = np.asarray(tensor)
    if x.ndim < 1 or x.shape[-1] != 8:
        raise ValueError("geometry tensor must have final dimension 8")
    perm = geometry_permutation(image_index=image_index, seed=seed, n_bins=8)
    return x[..., perm]
