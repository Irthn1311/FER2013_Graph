from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import linear_sum_assignment

from .crs_stage import (
    CRS_DESCRIPTOR_DIM,
    CRS_K,
    CRS_SIDE,
    CRS_VALID_POSITIONS,
    MASTER_SEED,
    composition_descriptors,
    permute_cell_blocks,
)


ISSUE_NUMBER = 86
PREREGISTRATION_SHA = "6e6ba806cc9ff0bc2b8a5534807d55f7feb00d73"
STABILITY_REPLICATES = 20
FIT_FRACTION = 0.80
SUPPORT_THRESHOLD = 288
JACCARD_THRESHOLD = 0.75
MAX_OCCURRENCES_PER_IMAGE = 64
SPARSE_FEATURE_DIM = 5 * CRS_K


def _seed64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stability_seed(arm: str, replicate_id: int) -> int:
    normalized = str(arm).upper()
    if normalized not in {"M", "C"}:
        raise ValueError("stability arm must be M or C")
    if not 0 <= int(replicate_id) < STABILITY_REPLICATES:
        raise ValueError("replicate_id must be in 0..19")
    return _seed64(MASTER_SEED, "motif_stability", normalized, int(replicate_id))


def image_level_partition(
    canonical_image_ids: np.ndarray, *, arm: str, replicate_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered deterministic floor(80%)/remainder image split."""
    image_ids = np.asarray(canonical_image_ids, dtype=np.int32).reshape(-1)
    if len(image_ids) < 2 or len(np.unique(image_ids)) != len(image_ids):
        raise ValueError("canonical image IDs must be unique")
    fit_count = int(np.floor(FIT_FRACTION * len(image_ids)))
    rng = np.random.Generator(np.random.PCG64(stability_seed(arm, replicate_id)))
    order = rng.permutation(len(image_ids))
    fit = np.sort(image_ids[order[:fit_count]])
    held_out = np.sort(image_ids[order[fit_count:]])
    if len(np.intersect1d(fit, held_out)) or len(fit) + len(held_out) != len(image_ids):
        raise AssertionError(
            "registered image partition is not disjoint and exhaustive"
        )
    return fit, held_out


@dataclass(frozen=True)
class CentroidMatching:
    original_to_replicate: np.ndarray
    replicate_to_original: np.ndarray
    matched_cosine: np.ndarray


def centroid_only_hungarian_match(
    original_centers: np.ndarray, replicate_centers: np.ndarray
) -> CentroidMatching:
    """Maximum-weight one-to-one matching using centroid cosine only."""
    original = np.asarray(original_centers, dtype=np.float32)
    replicate = np.asarray(replicate_centers, dtype=np.float32)
    if original.shape != replicate.shape or original.ndim != 2:
        raise ValueError("original and replicate centers must have the same 2D shape")
    if original.shape[0] != CRS_K or original.shape[1] != CRS_DESCRIPTOR_DIM:
        raise ValueError("registered centroid shape mismatch")
    if not np.all(np.isfinite(original)) or not np.all(np.isfinite(replicate)):
        raise ValueError("centers must be finite")
    original_norm = np.linalg.norm(original, axis=1, keepdims=True)
    replicate_norm = np.linalg.norm(replicate, axis=1, keepdims=True)
    if np.any(original_norm <= 0) or np.any(replicate_norm <= 0):
        raise ValueError("centers must have positive norm")
    similarity = (original / original_norm) @ (replicate / replicate_norm).T
    original_ids, replicate_ids = linear_sum_assignment(similarity, maximize=True)
    if not np.array_equal(original_ids, np.arange(CRS_K)):
        raise AssertionError("Hungarian original assignment is not canonical")
    original_to_replicate = np.empty(CRS_K, dtype=np.int32)
    original_to_replicate[original_ids] = replicate_ids.astype(np.int32)
    replicate_to_original = np.empty(CRS_K, dtype=np.int32)
    replicate_to_original[replicate_ids] = original_ids.astype(np.int32)
    expected = np.arange(CRS_K, dtype=np.int32)
    if not np.array_equal(np.sort(original_to_replicate), expected):
        raise AssertionError("Hungarian assignment is not a permutation")
    return CentroidMatching(
        original_to_replicate=original_to_replicate,
        replicate_to_original=replicate_to_original,
        matched_cosine=similarity[original_ids, replicate_ids].astype(np.float32),
    )


def exact_cosine_assignments_and_margins(
    descriptors: np.ndarray | sparse.spmatrix,
    centers: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cosine argmax IDs and best-minus-second-best margins."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    center_array = np.asarray(centers, dtype=np.float32)
    if center_array.ndim != 2 or center_array.shape[0] < 2:
        raise ValueError("centers must be a 2D array with at least two rows")
    center_norm = np.linalg.norm(center_array, axis=1, keepdims=True)
    if np.any(center_norm <= 0) or not np.all(np.isfinite(center_norm)):
        raise ValueError("centers must have finite positive norm")
    center_array = center_array / center_norm
    if sparse.issparse(descriptors):
        data = sparse.csr_matrix(descriptors, dtype=np.float32, copy=True)
        norms = np.sqrt(np.asarray(data.multiply(data).sum(axis=1)).reshape(-1))
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise ValueError("descriptors must have finite positive norm")
        data = sparse.diags((1.0 / norms).astype(np.float32)) @ data
        data = sparse.csr_matrix(data, dtype=np.float32)
    else:
        data = np.asarray(descriptors, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("descriptors must be 2D")
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise ValueError("descriptors must have finite positive norm")
        data = data / norms
    if data.shape[1] != center_array.shape[1]:
        raise ValueError("descriptor/center dimension mismatch")
    labels = np.empty(data.shape[0], dtype=np.int32)
    margins = np.empty(data.shape[0], dtype=np.float32)
    for start in range(0, data.shape[0], batch_size):
        block = data[start : start + batch_size]
        similarities = np.asarray(block @ center_array.T, dtype=np.float32)
        best_ids = np.argmax(similarities, axis=1).astype(np.int32)
        best = similarities[np.arange(len(similarities)), best_ids]
        second = np.partition(similarities, -2, axis=1)[:, -2]
        labels[start : start + block.shape[0]] = best_ids
        margins[start : start + block.shape[0]] = best - second
    if not np.all(np.isfinite(margins)) or np.any(margins < -1e-6):
        raise FloatingPointError("invalid assignment margins")
    return labels, np.maximum(margins, 0.0).astype(np.float32)


def cluster_jaccard(
    original_labels: np.ndarray,
    remapped_replicate_labels: np.ndarray,
    *,
    n_types: int = CRS_K,
) -> np.ndarray:
    original = np.asarray(original_labels, dtype=np.int64).reshape(-1)
    replicate = np.asarray(remapped_replicate_labels, dtype=np.int64).reshape(-1)
    if original.shape != replicate.shape:
        raise ValueError("Jaccard label vectors must have the same shape")
    if np.any((original < 0) | (original >= n_types)) or np.any(
        (replicate < 0) | (replicate >= n_types)
    ):
        raise ValueError("Jaccard labels outside vocabulary")
    count_a = np.bincount(original, minlength=n_types).astype(np.int64)
    count_b = np.bincount(replicate, minlength=n_types).astype(np.int64)
    intersection = np.bincount(
        original[original == replicate], minlength=n_types
    ).astype(np.int64)
    union = count_a + count_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(n_types, dtype=np.float64),
        where=union != 0,
    )


def heldout_dense_jaccard(
    primitive_maps: np.ndarray,
    canonical_image_ids: np.ndarray,
    *,
    arm: str,
    original_centers: np.ndarray,
    replicate_centers: np.ndarray,
    replicate_to_original: np.ndarray,
) -> np.ndarray:
    """Evaluate all 1,296 positions/image; no 24-sample pool is accepted here."""
    maps = np.asarray(primitive_maps)
    image_ids = np.asarray(canonical_image_ids, dtype=np.int32).reshape(-1)
    normalized_arm = str(arm).upper()
    if normalized_arm not in {"M", "C"}:
        raise ValueError("held-out arm must be M or C")
    if maps.ndim != 3 or maps.shape[0] != len(image_ids) or maps.shape[1:] != (44, 44):
        raise ValueError("held-out primitive-map shape mismatch")
    mapping = np.asarray(replicate_to_original, dtype=np.int32)
    if not np.array_equal(np.sort(mapping), np.arange(CRS_K, dtype=np.int32)):
        raise ValueError("replicate-to-original mapping must be a permutation")
    count_a = np.zeros(CRS_K, dtype=np.int64)
    count_b = np.zeros(CRS_K, dtype=np.int64)
    intersection = np.zeros(CRS_K, dtype=np.int64)
    expected_centers = np.arange(CRS_VALID_POSITIONS, dtype=np.int32)
    for primitive_map, image_id in zip(maps, image_ids):
        descriptors = composition_descriptors(primitive_map, normalize=True)
        if descriptors.shape != (CRS_VALID_POSITIONS, CRS_DESCRIPTOR_DIM):
            raise AssertionError(
                "dense held-out evaluator did not produce 1296 positions"
            )
        if normalized_arm == "C":
            descriptors = permute_cell_blocks(
                descriptors,
                split_id="train",
                image_id=int(image_id),
                center_indices=expected_centers,
            )
        csr = sparse.csr_matrix(descriptors, dtype=np.float32)
        original, _ = exact_cosine_assignments_and_margins(csr, original_centers)
        replicate, _ = exact_cosine_assignments_and_margins(csr, replicate_centers)
        remapped = mapping[replicate]
        count_a += np.bincount(original, minlength=CRS_K)
        count_b += np.bincount(remapped, minlength=CRS_K)
        same = original == remapped
        intersection += np.bincount(original[same], minlength=CRS_K)
    union = count_a + count_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(CRS_K, dtype=np.float64),
        where=union != 0,
    )


def distinct_image_support(dense_assignments: np.ndarray) -> np.ndarray:
    assignments = np.asarray(dense_assignments)
    if assignments.ndim != 3 or assignments.shape[1:] != (CRS_SIDE, CRS_SIDE):
        raise ValueError("dense assignments must have shape (images,36,36)")
    if np.any((assignments < 0) | (assignments >= CRS_K)):
        raise ValueError("dense assignment outside K=512")
    support = np.zeros(CRS_K, dtype=np.int32)
    for image in assignments:
        support[np.unique(image)] += 1
    return support


def summarize_jaccards(raw_jaccards: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_jaccards, dtype=np.float64)
    if raw.shape != (STABILITY_REPLICATES, CRS_K):
        raise ValueError("raw Jaccards must have shape (20,512)")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
        raise ValueError("Jaccards must be finite in [0,1]")
    q1, median, q3 = np.quantile(raw, [0.25, 0.5, 0.75], axis=0, method="linear")
    return {
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": raw.min(axis=0),
        "max": raw.max(axis=0),
    }


def qualified_type_ids(support: np.ndarray, median_jaccard: np.ndarray) -> np.ndarray:
    counts = np.asarray(support, dtype=np.int64).reshape(-1)
    medians = np.asarray(median_jaccard, dtype=np.float64).reshape(-1)
    if counts.shape != (CRS_K,) or medians.shape != (CRS_K,):
        raise ValueError("qualification vectors must have length 512")
    return np.flatnonzero(
        (counts >= SUPPORT_THRESHOLD) & (medians >= JACCARD_THRESHOLD)
    ).astype(np.int32)


def _rank_qualified(
    qualified: np.ndarray, support: np.ndarray, median_jaccard: np.ndarray
) -> np.ndarray:
    ids = np.asarray(qualified, dtype=np.int32).reshape(-1)
    if len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= CRS_K)):
        raise ValueError("invalid qualified type IDs")
    order = np.lexsort(
        (
            ids.astype(np.int64),
            -np.asarray(support, dtype=np.int64)[ids],
            -np.asarray(median_jaccard, dtype=np.float64)[ids],
        )
    )
    return ids[order]


def matched_vocabularies(
    q_m: np.ndarray,
    q_c: np.ndarray,
    *,
    m_support: np.ndarray,
    c_support: np.ndarray,
    m_median: np.ndarray,
    c_median: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    q_star = min(len(q_m), len(q_c))
    m_ranked = _rank_qualified(q_m, m_support, m_median)
    c_ranked = _rank_qualified(q_c, c_support, c_median)
    return m_ranked[:q_star], c_ranked[:q_star], int(q_star)


@dataclass(frozen=True)
class Occurrence:
    type_id: int
    row: int
    col: int
    margin: float
    component_size: int


@dataclass(frozen=True)
class OccurrenceExtraction:
    retained: tuple[Occurrence, ...]
    component_sizes: np.ndarray
    pre_cap_count: int


def extract_occurrences(
    assignment_map: np.ndarray,
    margin_map: np.ndarray,
    selected_types: np.ndarray,
) -> OccurrenceExtraction:
    assignments = np.asarray(assignment_map, dtype=np.int32)
    margins = np.asarray(margin_map, dtype=np.float32)
    selected = np.asarray(selected_types, dtype=np.int32).reshape(-1)
    if assignments.shape != (CRS_SIDE, CRS_SIDE) or margins.shape != assignments.shape:
        raise ValueError("occurrence maps must both have shape (36,36)")
    if not np.all(np.isfinite(margins)):
        raise ValueError("occurrence margins must be finite")
    if len(np.unique(selected)) != len(selected) or np.any(
        (selected < 0) | (selected >= CRS_K)
    ):
        raise ValueError("invalid selected vocabulary")
    structure = np.ones((3, 3), dtype=np.uint8)
    candidates: list[Occurrence] = []
    component_sizes: list[int] = []
    for type_id in np.sort(selected):
        labels, count = ndimage.label(assignments == int(type_id), structure=structure)
        for component_id in range(1, count + 1):
            coordinates = np.argwhere(labels == component_id)
            size = int(len(coordinates))
            component_sizes.append(size)
            flat = coordinates[:, 0] * CRS_SIDE + coordinates[:, 1]
            component_margins = margins[coordinates[:, 0], coordinates[:, 1]]
            best_margin = float(component_margins.max())
            tied = np.flatnonzero(component_margins == best_margin)
            best_local = int(tied[np.argmin(flat[tied])])
            row, col = coordinates[best_local]
            candidates.append(
                Occurrence(int(type_id), int(row), int(col), best_margin, size)
            )
    candidates.sort(key=lambda x: (-x.margin, x.type_id, x.row * CRS_SIDE + x.col))
    retained = tuple(candidates[:MAX_OCCURRENCES_PER_IMAGE])
    return OccurrenceExtraction(
        retained=retained,
        component_sizes=np.asarray(component_sizes, dtype=np.int32),
        pre_cap_count=len(candidates),
    )


def sparse_occurrence_feature(occurrences: tuple[Occurrence, ...]) -> np.ndarray:
    regions = np.zeros((5, CRS_K), dtype=np.float32)
    for occurrence in occurrences:
        if not 0 <= occurrence.type_id < CRS_K:
            raise ValueError("occurrence type outside K=512")
        if not 0 <= occurrence.row < CRS_SIDE or not 0 <= occurrence.col < CRS_SIDE:
            raise ValueError("occurrence coordinate outside 36x36")
        regions[0, occurrence.type_id] += 1.0
        quadrant = (
            1 + (2 if occurrence.row >= 18 else 0) + (1 if occurrence.col >= 18 else 0)
        )
        regions[quadrant, occurrence.type_id] += 1.0
    sums = regions.sum(axis=1, keepdims=True)
    regions = np.divide(regions, sums, out=np.zeros_like(regions), where=sums != 0)
    feature = regions.reshape(SPARSE_FEATURE_DIM)
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    if not np.all(np.isfinite(feature)):
        raise FloatingPointError("non-finite sparse occurrence feature")
    return feature


def occurrence_diagnostics(extractions: list[OccurrenceExtraction]) -> dict:
    if not extractions:
        raise ValueError("occurrence diagnostics require images")
    component_parts = [x.component_sizes for x in extractions if len(x.component_sizes)]
    components = (
        np.concatenate(component_parts).astype(np.int32)
        if component_parts
        else np.empty(0, dtype=np.int32)
    )
    pre = np.asarray([x.pre_cap_count for x in extractions], dtype=np.int32)
    post = np.asarray([len(x.retained) for x in extractions], dtype=np.int32)
    if len(components):
        median, p90, p95 = np.quantile(
            components.astype(np.float64), [0.5, 0.9, 0.95], method="linear"
        )
        component_summary = {
            "count": int(len(components)),
            "fraction_size_1": float(np.mean(components == 1)),
            "median": float(median),
            "p90": float(p90),
            "p95": float(p95),
            "max": int(components.max()),
        }
    else:
        component_summary = {
            "count": 0,
            "fraction_size_1": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "component_sizes": components,
        "pre_cap_nodes_per_image": pre,
        "post_cap_nodes_per_image": post,
        "component_summary": component_summary,
        "cap_binding_fraction": float(np.mean(pre > MAX_OCCURRENCES_PER_IMAGE)),
        "zero_node_image_count": int(np.count_nonzero(post == 0)),
    }
