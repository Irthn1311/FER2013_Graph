from __future__ import annotations

import numpy as np


_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SEED_MIX = np.uint64(0x9E3779B97F4A7C15)
_IMAGE_MIX = np.uint64(0xBF58476D1CE4E5B9)
_PIXEL_MIX = np.uint64(0x94D049BB133111EB)
_COORD_MIX = np.uint64(0xD2B74407B1CE6E93)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with defined unsigned wraparound."""
    z = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z + _SEED_MIX) & _U64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _IMAGE_MIX) & _U64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _PIXEL_MIX) & _U64_MASK
        return z ^ (z >> np.uint64(31))


def keyed_relation_permutations(
    image_ids: np.ndarray,
    pixel_indices: np.ndarray,
    *,
    control_seed: int,
) -> np.ndarray:
    """Return one stateless 24-coordinate permutation per descriptor.

    Each coordinate receives a SplitMix64 key derived only from
    ``(control_seed, canonical_image_id, pixel_index, coordinate)``. Sorting
    those keys gives a deterministic permutation independent of traversal,
    batching, chunk size, and worker scheduling.
    """
    image = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    pixel = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
    if image.shape != pixel.shape:
        raise ValueError("image_ids and pixel_indices must have matching shapes")
    if control_seed < 0 or np.any(image < 0) or np.any((pixel < 0) | (pixel >= 1936)):
        raise ValueError("control seed/image IDs must be nonnegative and pixel indices in [0,1935]")
    with np.errstate(over="ignore"):
        base = (
            np.uint64(control_seed) * _SEED_MIX
            ^ image.astype(np.uint64) * _IMAGE_MIX
            ^ pixel.astype(np.uint64) * _PIXEL_MIX
        )
        coordinate = np.arange(1, 25, dtype=np.uint64)
        keys = _splitmix64(base[:, None] ^ coordinate[None, :] * _COORD_MIX)
    return np.argsort(keys, axis=1, kind="stable").astype(np.int8)


def destroy_ordered_relations_keyed(
    s: np.ndarray,
    image_ids: np.ndarray,
    pixel_indices: np.ndarray,
    *,
    control_seed: int,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Destroy directional correspondence without cross-descriptor mixing.

    The input row multiset is preserved exactly. ``chunk_size`` changes memory
    use only and cannot change the keyed transformation.
    """
    x = np.asarray(s)
    image = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    pixel = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 24:
        raise ValueError("expected relation matrix with shape (n,24)")
    if len(image) != len(x) or len(pixel) != len(x):
        raise ValueError("one canonical image ID and pixel index are required per descriptor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = np.empty_like(x)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        perm = keyed_relation_permutations(
            image[start:stop], pixel[start:stop], control_seed=control_seed
        )
        out[start:stop] = np.take_along_axis(x[start:stop], perm, axis=1)
    return out


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
