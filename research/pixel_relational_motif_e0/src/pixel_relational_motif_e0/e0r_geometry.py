from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import sparse

from .controls import geometry_permutation
from .e0r_occurrence import CompactOccurrences, K


GEOMETRY_SEEDS = tuple(range(42, 62))
GEOMETRY_BINS = 8
GEOMETRY_DIM = K * K * GEOMETRY_BINS
NESTED_DIM = K + GEOMETRY_DIM


def _quadrants(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return (2 * (dy >= 0) + (dx >= 0)).astype(np.int8)


def train_pair_distance_median(all_occurrences: list[CompactOccurrences]) -> float | None:
    """Exact median from a histogram over discrete normalized coordinate distances."""
    distance_counts: dict[int, int] = {}
    total = 0
    for occurrences in all_occurrences:
        n = len(occurrences)
        if n < 2:
            continue
        dx = occurrences.x[:, None].astype(np.int32) - occurrences.x[None, :].astype(np.int32)
        dy = occurrences.y[:, None].astype(np.int32) - occurrences.y[None, :].astype(np.int32)
        mask = ~np.eye(n, dtype=bool)
        squared = np.square(dx[mask]) + np.square(dy[mask])
        values, counts = np.unique(squared, return_counts=True)
        for value, count in zip(values, counts):
            distance_counts[int(value)] = distance_counts.get(int(value), 0) + int(count)
            total += int(count)
    if total == 0:
        return None
    targets = ((total - 1) // 2, total // 2)
    found: list[int] = []
    cumulative = 0
    for value in sorted(distance_counts):
        next_cumulative = cumulative + distance_counts[value]
        while len(found) < 2 and targets[len(found)] < next_cumulative:
            found.append(value)
        cumulative = next_cumulative
        if len(found) == 2:
            break
    return float((np.sqrt(found[0]) + np.sqrt(found[1])) / (2.0 * 47.0))


def geometry_row(
    occurrences: CompactOccurrences,
    *,
    distance_median: float,
    canonical_image_id: int,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(occurrences)
    if n < 2:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    component = occurrences.components.astype(np.int64)
    x = occurrences.x.astype(np.float64) / 47.0
    y = occurrences.y.astype(np.float64) / 47.0
    src = np.repeat(component, n)
    dst = np.tile(component, n)
    dx = np.tile(x, n) - np.repeat(x, n)
    dy = np.tile(y, n) - np.repeat(y, n)
    mask = ~np.eye(n, dtype=bool).reshape(-1)
    src, dst, dx, dy = src[mask], dst[mask], dx[mask], dy[mask]
    distance_bin = (np.hypot(dx, dy) > distance_median).astype(np.int8)
    geometry = distance_bin * 4 + _quadrants(dx, dy)
    if shuffle_seed is not None:
        if shuffle_seed not in GEOMETRY_SEEDS:
            raise ValueError("unregistered geometry seed")
        permutation = geometry_permutation(
            image_index=int(canonical_image_id), seed=int(shuffle_seed), n_bins=GEOMETRY_BINS
        )
        geometry = np.argsort(permutation)[geometry]
    columns = ((src * K + dst) * GEOMETRY_BINS + geometry).astype(np.int32)
    unique, counts = np.unique(columns, return_counts=True)
    data = counts.astype(np.float32) / np.float32(n * (n - 1))
    return unique.astype(np.int32), data


def build_geometry_csr(
    all_occurrences: list[CompactOccurrences],
    canonical_ids: np.ndarray,
    *,
    distance_median: float,
    shuffle_seed: int | None = None,
) -> sparse.csr_matrix:
    ids = np.asarray(canonical_ids, dtype=np.int64).reshape(-1)
    if len(ids) != len(all_occurrences):
        raise ValueError("occurrences and canonical IDs must align")
    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row, (occurrences, image_id) in enumerate(zip(all_occurrences, ids)):
        indices, data = geometry_row(
            occurrences,
            distance_median=distance_median,
            canonical_image_id=int(image_id),
            shuffle_seed=shuffle_seed,
        )
        indices_parts.append(indices)
        data_parts.append(data)
        indptr[row + 1] = indptr[row] + len(indices)
    indices = np.concatenate(indices_parts) if indices_parts else np.empty(0, np.int32)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, np.float32)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(ids), GEOMETRY_DIM))
    if matrix.shape[1] != GEOMETRY_DIM:
        raise AssertionError("geometry dimension contract violated")
    return matrix


def nested_csr(unary: np.ndarray, geometry: sparse.csr_matrix) -> sparse.csr_matrix:
    o = sparse.csr_matrix(np.asarray(unary, dtype=np.float32))
    if o.shape[0] != geometry.shape[0] or o.shape[1] != K or geometry.shape[1] != GEOMETRY_DIM:
        raise ValueError("nested O/G shapes do not align")
    out = sparse.hstack([o, geometry], format="csr", dtype=np.float32)
    if out.shape != (o.shape[0], NESTED_DIM):
        raise AssertionError("nested dimension contract violated")
    return out


def sparse_sha256(matrix: sparse.spmatrix) -> str:
    x = matrix.tocsr(copy=True)
    x.sort_indices()
    h = hashlib.sha256()
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii"))
    for value in (x.indptr.astype(np.int64), x.indices.astype(np.int32), x.data.astype(np.float32)):
        h.update(np.ascontiguousarray(value).view(np.uint8))
    return h.hexdigest()
