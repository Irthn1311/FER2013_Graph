from __future__ import annotations

from pathlib import Path

import numpy as np

from .crs_stage import (
    CRS_DESCRIPTOR_DIM,
    CRS_K,
    CRS_SIDE,
    CRS_SUPPORT,
    CRS_VALID_POSITIONS,
    MASTER_SEED,
    PRIMITIVE_K,
    PRIMITIVE_SIDE,
    _histogram_rows,
    permute_cell_blocks,
    sample_dictionary_centers,
)
from .descriptor import extract_raw_relations
from .e02_runner import ActualDictionary, load_actual_dictionary


def load_frozen_primitive_substrate(dictionary_npz: str | Path) -> ActualDictionary:
    """Load the exact v533 K=128 primitive substrate; never filter stable IDs."""
    actual = load_actual_dictionary(dictionary_npz)
    if actual.artifact.selected_k != PRIMITIVE_K:
        raise ValueError("CRS stage requires the frozen K=128 primitive substrate")
    if actual.artifact.model.n_components != PRIMITIVE_K:
        raise ValueError("frozen primitive GMM component count mismatch")
    # E0.1's validated motif set is intentionally irrelevant here.  The full
    # canonical K=128 model is the primitive alphabet for this new hypothesis.
    return actual


def primitive_map_from_uint8(image_uint8: np.ndarray, substrate: ActualDictionary) -> np.ndarray:
    """Build the registered 44x44 hard primitive-ID field from one FER image."""
    x = np.asarray(image_uint8)
    if x.shape != (48, 48):
        raise ValueError(f"expected 48x48 image, got {x.shape}")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("primitive-map input must be integer pixels in [0,255]")
    if np.any((x < 0) | (x > 255)):
        raise ValueError("pixel outside [0,255]")

    image01 = x.astype(np.float64) / 255.0
    s, log_sigma, _ = extract_raw_relations(image01)
    r = substrate.artifact.transform.transform(s, log_sigma)
    if r.shape != (PRIMITIVE_SIDE * PRIMITIVE_SIDE, 12):
        raise AssertionError(f"unexpected frozen relation shape {r.shape}")
    primitive_ids = substrate.artifact.model.predict(r).astype(np.int16, copy=False)
    if primitive_ids.shape != (PRIMITIVE_SIDE * PRIMITIVE_SIDE,):
        raise AssertionError("primitive assignment shape invariant violated")
    if np.any((primitive_ids < 0) | (primitive_ids >= PRIMITIVE_K)):
        raise AssertionError("primitive assignment outside K=128")
    return primitive_ids.reshape(PRIMITIVE_SIDE, PRIMITIVE_SIDE)


def composition_descriptor_subset_fast(
    primitive_map: np.ndarray,
    linear_indices: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute only requested 9x9 CRS descriptors, avoiding dense materialization."""
    p = np.asarray(primitive_map)
    if p.shape != (PRIMITIVE_SIDE, PRIMITIVE_SIDE):
        raise ValueError(f"primitive_map must have shape {(PRIMITIVE_SIDE, PRIMITIVE_SIDE)}")
    if not np.issubdtype(p.dtype, np.integer):
        raise ValueError("primitive_map must contain integer primitive IDs")
    if np.any((p < 0) | (p >= PRIMITIVE_K)):
        raise ValueError("primitive IDs outside registered K=128 range")

    idx = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    if len(idx) == 0 or np.any((idx < 0) | (idx >= CRS_VALID_POSITIONS)):
        raise ValueError("invalid CRS center indices")
    top_y = idx // CRS_SIDE
    top_x = idx % CRS_SIDE
    windows = np.stack(
        [
            p[int(y) : int(y) + CRS_SUPPORT, int(x) : int(x) + CRS_SUPPORT]
            for y, x in zip(top_y, top_x)
        ],
        axis=0,
    )
    if windows.shape != (len(idx), CRS_SUPPORT, CRS_SUPPORT):
        raise AssertionError("subset window shape invariant violated")

    cell_histograms: list[np.ndarray] = []
    for cell_y in range(3):
        ys = slice(cell_y * 3, (cell_y + 1) * 3)
        for cell_x in range(3):
            xs = slice(cell_x * 3, (cell_x + 1) * 3)
            block = windows[:, ys, xs].reshape(len(idx), 9)
            hist = _histogram_rows(block, PRIMITIVE_K)
            if not np.all(hist.sum(axis=1) == 9):
                raise AssertionError("subset cell-count invariant violated")
            cell_histograms.append(hist)

    out = np.concatenate(cell_histograms, axis=1)
    if out.shape != (len(idx), CRS_DESCRIPTOR_DIM):
        raise AssertionError("subset descriptor shape invariant violated")
    if not np.all(out.sum(axis=1) == 81):
        raise AssertionError("subset 81-count invariant violated")
    if normalize:
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        if np.any(norm <= 0):
            raise FloatingPointError("zero subset descriptor norm")
        out = out / norm
    return out.astype(np.float32, copy=False)


def sampled_training_descriptors(
    primitive_map: np.ndarray,
    *,
    image_id: int,
    master_seed: int = MASTER_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched sampled centers plus M and C descriptors for dictionary fit."""
    indices = sample_dictionary_centers(image_id, master_seed=master_seed)
    m = composition_descriptor_subset_fast(primitive_map, indices, normalize=True)
    c = permute_cell_blocks(
        m,
        split_id="train",
        image_id=image_id,
        center_indices=indices,
        master_seed=master_seed,
    )
    if m.shape != c.shape or m.shape[1] != CRS_DESCRIPTOR_DIM:
        raise AssertionError("matched M/C sampled descriptor shape invariant violated")
    if not np.allclose(np.linalg.norm(m, axis=1), np.linalg.norm(c, axis=1), atol=1e-6):
        raise AssertionError("C control changed descriptor norms")
    return indices, m, c
