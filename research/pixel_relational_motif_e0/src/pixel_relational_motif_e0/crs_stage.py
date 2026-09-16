from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse


MASTER_SEED = 42
PRIMITIVE_K = 128
PRIMITIVE_SIDE = 44
CRS_SUPPORT = 9
CRS_RADIUS = CRS_SUPPORT // 2
CRS_SIDE = PRIMITIVE_SIDE - 2 * CRS_RADIUS
CRS_VALID_POSITIONS = CRS_SIDE * CRS_SIDE
CRS_CELL_GRID = 3
CRS_CELL_SIDE = CRS_SUPPORT // CRS_CELL_GRID
CRS_DESCRIPTOR_DIM = CRS_CELL_GRID * CRS_CELL_GRID * PRIMITIVE_K
CRS_K = 512
DICTIONARY_SAMPLES_PER_IMAGE = 24
PYRAMID_REGION_COUNT = 5
P_DIM = PYRAMID_REGION_COUNT * PRIMITIVE_K
M_DIM = PYRAMID_REGION_COUNT * CRS_K
C_DIM = M_DIM
E01_V533_DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"


if CRS_SUPPORT % CRS_CELL_GRID != 0:
    raise RuntimeError("CRS support must divide exactly into the registered cell grid")
if CRS_SIDE != 36 or CRS_VALID_POSITIONS != 1296:
    raise RuntimeError("registered CRS shape contract changed")
if CRS_DESCRIPTOR_DIM != 1152:
    raise RuntimeError("registered CRS descriptor dimension changed")


def valid_center_indices() -> np.ndarray:
    """Registered primitive-grid CRS centers as row-major integer (y, x)."""
    axis = np.arange(CRS_RADIUS, PRIMITIVE_SIDE - CRS_RADIUS, dtype=np.int16)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    out = np.stack([yy.ravel(), xx.ravel()], axis=1)
    if out.shape != (CRS_VALID_POSITIONS, 2):
        raise AssertionError("invalid registered CRS center shape")
    return out


def primitive_center_crop(primitive_map: np.ndarray) -> np.ndarray:
    """Return the P-arm IDs on the exact CRS center domain (36x36)."""
    p = np.asarray(primitive_map)
    if p.shape != (PRIMITIVE_SIDE, PRIMITIVE_SIDE):
        raise ValueError(f"primitive_map must have shape {(PRIMITIVE_SIDE, PRIMITIVE_SIDE)}")
    if not np.issubdtype(p.dtype, np.integer):
        raise ValueError("primitive_map must contain integer primitive IDs")
    if np.any((p < 0) | (p >= PRIMITIVE_K)):
        raise ValueError("primitive IDs outside registered K=128 range")
    crop = p[CRS_RADIUS : PRIMITIVE_SIDE - CRS_RADIUS, CRS_RADIUS : PRIMITIVE_SIDE - CRS_RADIUS]
    if crop.shape != (CRS_SIDE, CRS_SIDE):
        raise AssertionError("P-arm center crop shape invariant violated")
    return crop


def _histogram_rows(block: np.ndarray, n_bins: int = PRIMITIVE_K) -> np.ndarray:
    """Fast exact histograms for integer rows using one bincount."""
    x = np.asarray(block)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError("block must be a 2D integer array")
    if np.any((x < 0) | (x >= n_bins)):
        raise ValueError("histogram IDs outside range")
    n, width = x.shape
    offsets = np.arange(n, dtype=np.int64)[:, None] * int(n_bins)
    encoded = (x.astype(np.int64, copy=False) + offsets).ravel()
    counts = np.bincount(encoded, minlength=n * n_bins).reshape(n, n_bins)
    if np.any(counts.sum(axis=1) != width):
        raise AssertionError("histogram count invariant violated")
    return counts.astype(np.float32, copy=False)


def composition_descriptors(primitive_map: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Dense registered 9x9 -> 3x3-cell primitive-histogram descriptors.

    Returns row-major descriptors for all 36x36 valid centers with shape
    (1296, 1152). Cell blocks are concatenated in row-major cell order.
    """
    p = np.asarray(primitive_map)
    if p.shape != (PRIMITIVE_SIDE, PRIMITIVE_SIDE):
        raise ValueError(f"primitive_map must have shape {(PRIMITIVE_SIDE, PRIMITIVE_SIDE)}")
    if not np.issubdtype(p.dtype, np.integer):
        raise ValueError("primitive_map must contain integer primitive IDs")
    if np.any((p < 0) | (p >= PRIMITIVE_K)):
        raise ValueError("primitive IDs outside registered K=128 range")

    windows = sliding_window_view(p, (CRS_SUPPORT, CRS_SUPPORT))
    if windows.shape != (CRS_SIDE, CRS_SIDE, CRS_SUPPORT, CRS_SUPPORT):
        raise AssertionError(f"unexpected CRS window shape {windows.shape}")
    n = CRS_VALID_POSITIONS
    cell_histograms: list[np.ndarray] = []
    for cell_y in range(CRS_CELL_GRID):
        ys = slice(cell_y * CRS_CELL_SIDE, (cell_y + 1) * CRS_CELL_SIDE)
        for cell_x in range(CRS_CELL_GRID):
            xs = slice(cell_x * CRS_CELL_SIDE, (cell_x + 1) * CRS_CELL_SIDE)
            block = windows[:, :, ys, xs].reshape(n, CRS_CELL_SIDE * CRS_CELL_SIDE)
            hist = _histogram_rows(block, PRIMITIVE_K)
            if not np.all(hist.sum(axis=1) == CRS_CELL_SIDE * CRS_CELL_SIDE):
                raise AssertionError("registered 9-count cell invariant violated")
            cell_histograms.append(hist)
    out = np.concatenate(cell_histograms, axis=1)
    if out.shape != (CRS_VALID_POSITIONS, CRS_DESCRIPTOR_DIM):
        raise AssertionError(f"CRS descriptor shape invariant violated: {out.shape}")
    if not np.all(out.sum(axis=1) == CRS_SUPPORT * CRS_SUPPORT):
        raise AssertionError("registered 81-count patch invariant violated")
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise FloatingPointError("zero CRS descriptor norm")
        out = out / norms
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite CRS descriptor")
    return out.astype(np.float32, copy=False)


def composition_descriptor_subset(
    primitive_map: np.ndarray,
    linear_indices: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Select registered dense descriptors at exact row-major center indices."""
    idx = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    if np.any((idx < 0) | (idx >= CRS_VALID_POSITIONS)):
        raise ValueError("composition center index outside 36x36 domain")
    dense = composition_descriptors(primitive_map, normalize=normalize)
    return dense[idx]


def _seed64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_dictionary_centers(
    image_id: int,
    *,
    count: int = DICTIONARY_SAMPLES_PER_IMAGE,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Deterministic uniform-without-replacement Training center sample."""
    if count <= 0 or count > CRS_VALID_POSITIONS:
        raise ValueError("invalid dictionary sample count")
    seed = _seed64("crs-dict-sample", int(master_seed), int(image_id))
    rng = np.random.default_rng(seed)
    selected = rng.choice(CRS_VALID_POSITIONS, size=count, replace=False)
    selected = np.sort(selected.astype(np.int32, copy=False))
    if len(np.unique(selected)) != count:
        raise AssertionError("dictionary sampling must be without replacement")
    return selected


_PERMUTATION_TABLE: np.ndarray | None = None


def _permutation_table() -> np.ndarray:
    global _PERMUTATION_TABLE
    if _PERMUTATION_TABLE is None:
        flat = np.fromiter(
            itertools.chain.from_iterable(itertools.permutations(range(CRS_CELL_GRID * CRS_CELL_GRID))),
            dtype=np.uint8,
        )
        table = flat.reshape(-1, CRS_CELL_GRID * CRS_CELL_GRID)
        if table.shape != (362880, 9):
            raise AssertionError("S9 permutation table invariant violated")
        _PERMUTATION_TABLE = table
    return _PERMUTATION_TABLE


def keyed_cell_permutation(
    *,
    split_id: str,
    image_id: int,
    center_y: int,
    center_x: int,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Version-independent, cryptographic-hash-keyed S9 permutation."""
    if not split_id:
        raise ValueError("split_id must be non-empty")
    if not (CRS_RADIUS <= int(center_y) < PRIMITIVE_SIDE - CRS_RADIUS):
        raise ValueError("center_y outside registered CRS domain")
    if not (CRS_RADIUS <= int(center_x) < PRIMITIVE_SIDE - CRS_RADIUS):
        raise ValueError("center_x outside registered CRS domain")
    payload = (
        f"crs-cell-perm-v1\x1f{int(master_seed)}\x1f{split_id}\x1f"
        f"{int(image_id)}\x1f{int(center_y)}\x1f{int(center_x)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    rank = int.from_bytes(digest[:8], "big", signed=False) % 362880
    return _permutation_table()[rank].astype(np.int16, copy=True)


def permute_cell_blocks(
    descriptors: np.ndarray,
    *,
    split_id: str,
    image_id: int,
    center_indices: np.ndarray | None = None,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Apply the registered per-patch keyed permutation to nine 128-bin blocks."""
    x = np.asarray(descriptors, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != CRS_DESCRIPTOR_DIM:
        raise ValueError(f"descriptors must have shape (n,{CRS_DESCRIPTOR_DIM})")
    n = len(x)
    if center_indices is None:
        if n != CRS_VALID_POSITIONS:
            raise ValueError("center_indices required unless descriptors are dense 1296-row descriptors")
        center_indices = np.arange(CRS_VALID_POSITIONS, dtype=np.int32)
    idx = np.asarray(center_indices, dtype=np.int64).reshape(-1)
    if len(idx) != n or np.any((idx < 0) | (idx >= CRS_VALID_POSITIONS)):
        raise ValueError("invalid center_indices")
    centers = valid_center_indices()[idx]
    blocks = x.reshape(n, 9, PRIMITIVE_K)
    out = np.empty_like(blocks)
    for row, ((center_y, center_x), block) in enumerate(zip(centers, blocks)):
        perm = keyed_cell_permutation(
            split_id=split_id,
            image_id=int(image_id),
            center_y=int(center_y),
            center_x=int(center_x),
            master_seed=master_seed,
        )
        out[row] = block[perm]
    return out.reshape(n, CRS_DESCRIPTOR_DIM)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    data = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("all spherical-kmeans rows must have finite positive norm")
    return data / norms


@dataclass(frozen=True)
class SphericalKMeansResult:
    cluster_centers_: np.ndarray
    labels_: np.ndarray
    objective_: float
    n_iter_: int
    converged_: bool
    init_index_: int
    empty_reseeds_: int

    def predict(self, x: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        data = _normalize_rows(x)
        centers = np.asarray(self.cluster_centers_, dtype=np.float32)
        labels = np.empty(len(data), dtype=np.int32)
        for start in range(0, len(data), batch_size):
            xb = data[start : start + batch_size]
            sim = xb @ centers.T
            labels[start : start + len(xb)] = np.argmax(sim, axis=1).astype(np.int32)
        return labels


class SphericalKMeans:
    """Deterministic Lloyd-style spherical k-means for the registered CRS stage."""

    def __init__(
        self,
        n_clusters: int = CRS_K,
        *,
        n_init: int = 3,
        max_iter: int = 50,
        tol: float = 1e-6,
        random_state: int = MASTER_SEED,
        batch_size: int = 8192,
    ) -> None:
        if n_clusters <= 1 or n_init <= 0 or max_iter <= 0 or tol <= 0 or batch_size <= 0:
            raise ValueError("invalid spherical-kmeans configuration")
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.batch_size = int(batch_size)

    def _initial_centers(self, x: np.ndarray, init_index: int) -> np.ndarray:
        if len(x) < self.n_clusters:
            raise ValueError("n_clusters exceeds number of training descriptors")
        seed = _seed64("crs-skm-init", self.random_state, init_index)
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(x), size=self.n_clusters, replace=False)
        return x[ids].copy()

    def _assign(self, x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        labels = np.empty(len(x), dtype=np.int32)
        best = np.empty(len(x), dtype=np.float32)
        objective = 0.0
        for start in range(0, len(x), self.batch_size):
            xb = x[start : start + self.batch_size]
            sim = xb @ centers.T
            lb = np.argmax(sim, axis=1).astype(np.int32)
            bs = sim[np.arange(len(xb)), lb].astype(np.float32)
            labels[start : start + len(xb)] = lb
            best[start : start + len(xb)] = bs
            objective += float(bs.sum(dtype=np.float64))
        return labels, best, objective

    def _updated_centers(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        best_similarity: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        n = len(x)
        rows = labels.astype(np.int64, copy=False)
        cols = np.arange(n, dtype=np.int64)
        membership = sparse.csr_matrix(
            (np.ones(n, dtype=np.float32), (rows, cols)),
            shape=(self.n_clusters, n),
        )
        sums = np.asarray(membership @ x, dtype=np.float32)
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.int64)
        centers = np.empty((self.n_clusters, x.shape[1]), dtype=np.float32)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]

        empty_ids = np.flatnonzero(~nonempty)
        reseeds = len(empty_ids)
        if reseeds:
            order = np.lexsort((np.arange(n, dtype=np.int64), best_similarity.astype(np.float64)))
            selected: list[int] = []
            used = set()
            for idx in order:
                ii = int(idx)
                if ii in used:
                    continue
                selected.append(ii)
                used.add(ii)
                if len(selected) == reseeds:
                    break
            if len(selected) != reseeds:
                raise RuntimeError("could not deterministically reseed empty clusters")
            for cluster_id, row_id in zip(empty_ids.tolist(), selected):
                centers[int(cluster_id)] = x[int(row_id)]

        centers = _normalize_rows(centers)
        return centers, reseeds

    def fit(self, x: np.ndarray) -> SphericalKMeansResult:
        data = _normalize_rows(x)
        best_result: SphericalKMeansResult | None = None
        for init_index in range(self.n_init):
            centers = self._initial_centers(data, init_index)
            centers = _normalize_rows(centers)
            previous_labels: np.ndarray | None = None
            previous_objective: float | None = None
            total_reseeds = 0
            converged = False
            iterations = 0

            for iteration in range(1, self.max_iter + 1):
                labels, best_similarity, objective = self._assign(data, centers)
                assignments_unchanged = previous_labels is not None and np.array_equal(labels, previous_labels)
                relative_small = False
                if previous_objective is not None:
                    denom = max(1.0, abs(previous_objective))
                    improvement = objective - previous_objective
                    relative_small = improvement >= -1e-10 and abs(improvement) / denom < self.tol

                new_centers, reseeds = self._updated_centers(data, labels, best_similarity)
                total_reseeds += reseeds
                centers = new_centers
                iterations = iteration

                if assignments_unchanged or (relative_small and reseeds == 0):
                    converged = True
                    break
                previous_labels = labels.copy()
                previous_objective = objective

            final_labels, _, final_objective = self._assign(data, centers)
            result = SphericalKMeansResult(
                cluster_centers_=centers.copy(),
                labels_=final_labels,
                objective_=float(final_objective),
                n_iter_=iterations,
                converged_=bool(converged),
                init_index_=init_index,
                empty_reseeds_=int(total_reseeds),
            )
            if best_result is None or result.objective_ > best_result.objective_:
                best_result = result

        assert best_result is not None
        return best_result


def spatial_pyramid_histogram(id_map: np.ndarray, *, n_bins: int) -> np.ndarray:
    """Registered 1x1 + 2x2 regional histogram, regional L1 then global L2."""
    ids = np.asarray(id_map)
    if ids.shape != (CRS_SIDE, CRS_SIDE):
        raise ValueError(f"id_map must have shape {(CRS_SIDE, CRS_SIDE)}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("id_map must contain integer IDs")
    if np.any((ids < 0) | (ids >= n_bins)):
        raise ValueError("IDs outside histogram range")
    half = CRS_SIDE // 2
    regions = [
        ids,
        ids[:half, :half],
        ids[:half, half:],
        ids[half:, :half],
        ids[half:, half:],
    ]
    histograms: list[np.ndarray] = []
    expected_counts = [CRS_VALID_POSITIONS, half * half, half * half, half * half, half * half]
    for region, expected in zip(regions, expected_counts):
        hist = np.bincount(region.ravel(), minlength=n_bins).astype(np.float32)
        if int(hist.sum()) != expected:
            raise AssertionError("pyramid regional count invariant violated")
        hist /= float(expected)
        histograms.append(hist)
    out = np.concatenate(histograms)
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError("invalid pyramid norm")
    out /= norm
    return out.astype(np.float32, copy=False)


def p_image_feature(primitive_map: np.ndarray) -> np.ndarray:
    feature = spatial_pyramid_histogram(primitive_center_crop(primitive_map), n_bins=PRIMITIVE_K)
    if feature.shape != (P_DIM,):
        raise AssertionError("P feature dimension invariant violated")
    return feature


def crs_image_feature(assignments: np.ndarray) -> np.ndarray:
    labels = np.asarray(assignments, dtype=np.int32).reshape(CRS_SIDE, CRS_SIDE)
    feature = spatial_pyramid_histogram(labels, n_bins=CRS_K)
    if feature.shape != (M_DIM,):
        raise AssertionError("CRS feature dimension invariant violated")
    return feature
