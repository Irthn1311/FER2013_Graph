from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from numpy.lib.format import open_memmap
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .crs_public_runner import load_frozen_train_model, verify_frozen_dictionary
from .crs_runtime import (
    load_frozen_primitive_substrate,
    primitive_map_from_uint8,
    sampled_training_descriptors,
)
from .crs_stage import (
    CRS_DESCRIPTOR_DIM,
    CRS_K,
    CRS_SIDE,
    CRS_VALID_POSITIONS,
    DICTIONARY_SAMPLES_PER_IMAGE,
    MASTER_SEED,
    PRIMITIVE_SIDE,
    SphericalKMeans,
    SphericalKMeansIterationDiagnostic,
    composition_descriptors,
    descriptors_to_csr,
    permute_cell_blocks,
)
from .e01_runner import TRAIN_ROWS
from .e02_runner import (
    TRAIN_SHA256,
    load_labels_downstream,
    load_pixels_only,
    sha256_array,
    sha256_file,
)
from .motif_qualification import (
    FIT_FRACTION,
    ISSUE_NUMBER,
    JACCARD_THRESHOLD,
    MAX_OCCURRENCES_PER_IMAGE,
    PREREGISTRATION_SHA,
    SPARSE_FEATURE_DIM,
    STABILITY_REPLICATES,
    SUPPORT_THRESHOLD,
    centroid_only_hungarian_match,
    distinct_image_support,
    exact_cosine_assignments_and_margins,
    extract_occurrences,
    heldout_dense_jaccard,
    image_level_partition,
    matched_vocabularies,
    occurrence_diagnostics,
    qualified_type_ids,
    sparse_occurrence_feature,
    stability_seed,
    summarize_jaccards,
)


ACCEPTED_TRAIN_MODEL_SHA256 = (
    "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"
)
SCIENTIFIC_SHA_ENV = "MOTIF_SCIENTIFIC_SHA"
SKM_N_INIT = 3
SKM_MAX_ITER = 500
SKM_TOL = 1e-6
SKM_BATCH_SIZE = 8192
LOGREG_C = 1.0
LOGREG_SOLVER = "lbfgs"
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_MAX_ITER = 5000
LOGREG_TOL = 1e-4
POOL_CHUNK_IMAGES = 128
EXEMPLARS_PER_TYPE = 5

SUBSTRATE_MANIFEST = "motif_train_substrate_manifest.json"
PRIMITIVE_MAPS_FILE = "motif_train_primitive_maps.npy"
M_ASSIGNMENTS_FILE = "motif_train_m_assignments.npy"
C_ASSIGNMENTS_FILE = "motif_train_c_assignments.npy"
M_MARGINS_FILE = "motif_train_m_margins.npy"
C_MARGINS_FILE = "motif_train_c_margins.npy"
M_POOL_FILE = "motif_train_m_sample_pool.npz"
C_POOL_FILE = "motif_train_c_sample_pool.npz"
RECURRENCE_FILE = "motif_train_recurrence.npz"
FINAL_MODEL_FILE = "motif_train_model.npz"
FINAL_SUMMARY_FILE = "motif_train_summary.json"
FINAL_MANIFEST_FILE = "motif_train_execution_manifest.json"
STABILITY_MERGED_FILE = "motif_stability_merged.npz"
OCCURRENCE_DIAGNOSTICS_FILE = "motif_train_occurrence_diagnostics.npz"


@dataclass(frozen=True)
class SparseProbeState:
    classes: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    n_iter: np.ndarray
    train_accuracy: float
    train_macro_f1: float
    per_class_f1: np.ndarray
    converged: bool


def _log(message: str) -> None:
    print(f"[PGM-MOTIF-TRAIN] {message}", flush=True)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _scientific_sha() -> str:
    value = os.environ.get(SCIENTIFIC_SHA_ENV, "")
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError(
            f"{SCIENTIFIC_SHA_ENV} must contain the reviewed 40-hex source SHA"
        )
    return value


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_sparse(path: Path, matrix: sparse.csr_matrix) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        sparse.save_npz(
            temporary, sparse.csr_matrix(matrix, dtype=np.float32), compressed=True
        )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_file_record(path: Path, record: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"artifact identity mismatch: {path.name}")


def _registered_replicate_skm(
    *,
    arm: str,
    replicate_id: int,
    diagnostics: list[SphericalKMeansIterationDiagnostic],
) -> SphericalKMeans:
    """Create a from-scratch fit; accepted full-Train centers are not an input."""
    return SphericalKMeans(
        n_clusters=CRS_K,
        n_init=SKM_N_INIT,
        max_iter=SKM_MAX_ITER,
        tol=SKM_TOL,
        random_state=stability_seed(arm, replicate_id),
        batch_size=SKM_BATCH_SIZE,
        diagnostic_callback=diagnostics.append,
    )


def _save_pool_chunks(blocks: list[sparse.csr_matrix], path: Path) -> None:
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    sparse.save_npz(path, matrix, compressed=True)


def _merge_pool_chunks(paths: list[Path], output: Path) -> None:
    blocks = [sparse.load_npz(path).tocsr().astype(np.float32) for path in paths]
    expected_rows = TRAIN_ROWS * DICTIONARY_SAMPLES_PER_IMAGE
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    if matrix.shape != (expected_rows, CRS_DESCRIPTOR_DIM):
        raise AssertionError("registered sampled descriptor pool shape mismatch")
    _atomic_sparse(output, matrix)


def _local_fields(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # CRS centers 4..39 on the 44x44 primitive grid map to raw centers 6..41.
    raw = np.asarray(image, dtype=np.float32) / 255.0
    windows = np.lib.stride_tricks.sliding_window_view(raw, (9, 9))[2:38, 2:38]
    if windows.shape != (CRS_SIDE, CRS_SIDE, 9, 9):
        raise AssertionError("local nuisance window geometry mismatch")
    intensity = windows.mean(axis=(-2, -1), dtype=np.float64).astype(np.float32)
    contrast = windows.std(axis=(-2, -1), dtype=np.float64).astype(np.float32)
    return intensity, contrast


def _update_nuisance(
    accumulator: dict[str, np.ndarray | float | int],
    assignments: np.ndarray,
    intensity: np.ndarray,
    contrast: np.ndarray,
) -> None:
    ids = assignments.reshape(-1).astype(np.int64)
    i = intensity.reshape(-1).astype(np.float64)
    c = contrast.reshape(-1).astype(np.float64)
    accumulator["count"] += np.bincount(ids, minlength=CRS_K)
    accumulator["intensity_sum"] += np.bincount(ids, weights=i, minlength=CRS_K)
    accumulator["contrast_sum"] += np.bincount(ids, weights=c, minlength=CRS_K)
    accumulator["intensity_total"] += float(i.sum())
    accumulator["intensity_sq_total"] += float(i @ i)
    accumulator["contrast_total"] += float(c.sum())
    accumulator["contrast_sq_total"] += float(c @ c)
    accumulator["position_count"] += len(ids)
    quadrant = np.add.outer(
        (np.arange(CRS_SIDE) >= 18).astype(np.int32) * 2,
        (np.arange(CRS_SIDE) >= 18).astype(np.int32),
    ).reshape(-1)
    for q in range(4):
        accumulator["quadrant_count"][:, q] += np.bincount(
            ids[quadrant == q], minlength=CRS_K
        )
    rows, cols = np.indices((CRS_SIDE, CRS_SIDE))
    border = ((rows < 4) | (rows >= 32) | (cols < 4) | (cols >= 32)).reshape(-1)
    accumulator["border_count"] += np.bincount(ids[border], minlength=CRS_K)


def _empty_nuisance() -> dict[str, np.ndarray | float | int]:
    return {
        "count": np.zeros(CRS_K, dtype=np.int64),
        "intensity_sum": np.zeros(CRS_K, dtype=np.float64),
        "contrast_sum": np.zeros(CRS_K, dtype=np.float64),
        "quadrant_count": np.zeros((CRS_K, 4), dtype=np.int64),
        "border_count": np.zeros(CRS_K, dtype=np.int64),
        "intensity_total": 0.0,
        "intensity_sq_total": 0.0,
        "contrast_total": 0.0,
        "contrast_sq_total": 0.0,
        "position_count": 0,
    }


def build_train_substrate(
    *,
    train_csv: str | Path,
    dictionary_npz: str | Path,
    train_model_npz: str | Path,
    output_dir: str | Path,
) -> dict:
    """Build the canonical label-free Train substrate once."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("substrate output directory must be empty")
    source_sha = _scientific_sha()
    dictionary_sha = verify_frozen_dictionary(dictionary_npz)
    model = load_frozen_train_model(train_model_npz)
    if model.artifact_sha256 != ACCEPTED_TRAIN_MODEL_SHA256:
        raise AssertionError("accepted CRS Train model identity drift")
    primitive_substrate = load_frozen_primitive_substrate(dictionary_npz)
    train = load_pixels_only(train_csv, role="train")
    if train.sha256 != TRAIN_SHA256 or train.images_uint8.shape != (TRAIN_ROWS, 48, 48):
        raise AssertionError("official Train pixel contract mismatch")

    work = Path(tempfile.mkdtemp(prefix="motif_substrate_", dir=out))
    final_arrays = {
        PRIMITIVE_MAPS_FILE: ((TRAIN_ROWS, PRIMITIVE_SIDE, PRIMITIVE_SIDE), np.int16),
        M_ASSIGNMENTS_FILE: ((TRAIN_ROWS, CRS_SIDE, CRS_SIDE), np.int16),
        C_ASSIGNMENTS_FILE: ((TRAIN_ROWS, CRS_SIDE, CRS_SIDE), np.int16),
        M_MARGINS_FILE: ((TRAIN_ROWS, CRS_SIDE, CRS_SIDE), np.float32),
        C_MARGINS_FILE: ((TRAIN_ROWS, CRS_SIDE, CRS_SIDE), np.float32),
    }
    maps = {
        name: open_memmap(work / name, mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in final_arrays.items()
    }
    pool_blocks: dict[str, list[sparse.csr_matrix]] = {"M": [], "C": []}
    pool_chunks: dict[str, list[Path]] = {"M": [], "C": []}
    sample_centers = np.empty(
        (TRAIN_ROWS, DICTIONARY_SAMPLES_PER_IMAGE), dtype=np.int32
    )
    nuisances = {"M": _empty_nuisance(), "C": _empty_nuisance()}
    dense_center_ids = np.arange(CRS_VALID_POSITIONS, dtype=np.int32)
    try:
        for image_index, (image, image_id) in enumerate(
            zip(train.images_uint8, train.canonical_ids)
        ):
            primitive_map = primitive_map_from_uint8(image, primitive_substrate)
            m_dense = composition_descriptors(primitive_map, normalize=True)
            c_dense = permute_cell_blocks(
                m_dense,
                split_id="train",
                image_id=int(image_id),
                center_indices=dense_center_ids,
            )
            m_ids, m_margin = exact_cosine_assignments_and_margins(
                descriptors_to_csr(m_dense), model.m_centers
            )
            c_ids, c_margin = exact_cosine_assignments_and_margins(
                descriptors_to_csr(c_dense), model.c_centers
            )
            maps[PRIMITIVE_MAPS_FILE][image_index] = primitive_map
            maps[M_ASSIGNMENTS_FILE][image_index] = m_ids.reshape(CRS_SIDE, CRS_SIDE)
            maps[C_ASSIGNMENTS_FILE][image_index] = c_ids.reshape(CRS_SIDE, CRS_SIDE)
            maps[M_MARGINS_FILE][image_index] = m_margin.reshape(CRS_SIDE, CRS_SIDE)
            maps[C_MARGINS_FILE][image_index] = c_margin.reshape(CRS_SIDE, CRS_SIDE)
            sampled_ids, sampled_m, sampled_c = sampled_training_descriptors(
                primitive_map, image_id=int(image_id)
            )
            sample_centers[image_index] = sampled_ids
            pool_blocks["M"].append(descriptors_to_csr(sampled_m))
            pool_blocks["C"].append(descriptors_to_csr(sampled_c))
            intensity, contrast = _local_fields(image)
            _update_nuisance(
                nuisances["M"], m_ids.reshape(CRS_SIDE, CRS_SIDE), intensity, contrast
            )
            _update_nuisance(
                nuisances["C"], c_ids.reshape(CRS_SIDE, CRS_SIDE), intensity, contrast
            )
            if (
                len(pool_blocks["M"]) == POOL_CHUNK_IMAGES
                or image_index + 1 == TRAIN_ROWS
            ):
                for arm in ("M", "C"):
                    path = work / f"{arm.lower()}_pool_{len(pool_chunks[arm]):04d}.npz"
                    _save_pool_chunks(pool_blocks[arm], path)
                    pool_chunks[arm].append(path)
                    pool_blocks[arm].clear()
            if (image_index + 1) % 256 == 0 or image_index + 1 == TRAIN_ROWS:
                _log(f"label-free substrate images {image_index + 1}/{TRAIN_ROWS}")
        for value in maps.values():
            value.flush()
        del maps
        for name in final_arrays:
            os.replace(work / name, out / name)
        _merge_pool_chunks(pool_chunks["M"], out / M_POOL_FILE)
        _merge_pool_chunks(pool_chunks["C"], out / C_POOL_FILE)
        m_assignments = np.load(out / M_ASSIGNMENTS_FILE, mmap_mode="r")
        c_assignments = np.load(out / C_ASSIGNMENTS_FILE, mmap_mode="r")
        m_support = distinct_image_support(m_assignments)
        c_support = distinct_image_support(c_assignments)
        recurrence_payload: dict[str, np.ndarray] = {
            "m_distinct_image_support": m_support,
            "c_distinct_image_support": c_support,
            "sample_center_indices": sample_centers,
            "sample_center_indices_sha256": np.asarray(sha256_array(sample_centers)),
        }
        for arm in ("M", "C"):
            for key, value in nuisances[arm].items():
                recurrence_payload[f"{arm.lower()}_nuisance_{key}"] = np.asarray(value)
        _atomic_savez(out / RECURRENCE_FILE, recurrence_payload)
        files = {
            name: _file_record(out / name)
            for name in (
                *final_arrays.keys(),
                M_POOL_FILE,
                C_POOL_FILE,
                RECURRENCE_FILE,
            )
        }
        identity_payload = {
            "scientific_source_sha": source_sha,
            "preregistration_sha": PREREGISTRATION_SHA,
            "train_sha256": train.sha256,
            "dictionary_sha256": dictionary_sha,
            "accepted_train_model_sha256": model.artifact_sha256,
            "files": files,
        }
        substrate_identity = hashlib.sha256(
            _canonical_json_bytes(identity_payload)
        ).hexdigest()
        manifest = {
            "experiment": "PGM_MOTIF_QUALIFICATION_TRAIN_SUBSTRATE",
            "issue": ISSUE_NUMBER,
            **identity_payload,
            "substrate_identity_sha256": substrate_identity,
            "registered": {
                "train_rows": TRAIN_ROWS,
                "dense_positions_per_image": CRS_VALID_POSITIONS,
                "sampled_fit_descriptors_per_image": DICTIONARY_SAMPLES_PER_IMAGE,
                "crs_k": CRS_K,
            },
            "environment": _environment(),
            "labels_accessed": False,
            "public_test_accessed": False,
            "private_test_accessed": False,
        }
        _atomic_json(out / SUBSTRATE_MANIFEST, manifest)
        return manifest
    finally:
        if work.exists():
            shutil.rmtree(work)


def load_substrate_manifest(substrate_dir: str | Path) -> dict:
    root = Path(substrate_dir)
    manifest = json.loads((root / SUBSTRATE_MANIFEST).read_text(encoding="utf-8"))
    if (
        manifest["issue"] != ISSUE_NUMBER
        or manifest["preregistration_sha"] != PREREGISTRATION_SHA
    ):
        raise ValueError("motif substrate provenance mismatch")
    if manifest["accepted_train_model_sha256"] != ACCEPTED_TRAIN_MODEL_SHA256:
        raise ValueError("motif substrate Train model mismatch")
    for name, record in manifest["files"].items():
        _verify_file_record(root / name, record)
    identity_payload = {
        key: manifest[key]
        for key in (
            "scientific_source_sha",
            "preregistration_sha",
            "train_sha256",
            "dictionary_sha256",
            "accepted_train_model_sha256",
            "files",
        )
    }
    identity = hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()
    if identity != manifest["substrate_identity_sha256"]:
        raise ValueError("motif substrate identity mismatch")
    return manifest


def _replicate_filename(arm: str, replicate_id: int) -> str:
    return f"motif_stability_{arm.upper()}_rep_{replicate_id:02d}.npz"


def run_stability_replicate(
    *,
    arm: str,
    replicate_id: int,
    substrate_dir: str | Path,
    train_model_npz: str | Path,
    output_dir: str | Path,
) -> dict:
    normalized_arm = str(arm).upper()
    seed = stability_seed(normalized_arm, replicate_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / _replicate_filename(normalized_arm, replicate_id)
    manifest_path = out / artifact_path.with_suffix(".manifest.json").name
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite completed stability replicate")
    source_sha = _scientific_sha()
    substrate_manifest = load_substrate_manifest(substrate_dir)
    if source_sha != substrate_manifest["scientific_source_sha"]:
        raise ValueError("stability source differs from substrate source")
    model = load_frozen_train_model(train_model_npz)
    root = Path(substrate_dir)
    pool = sparse.load_npz(
        root / (M_POOL_FILE if normalized_arm == "M" else C_POOL_FILE)
    )
    canonical_ids = np.arange(TRAIN_ROWS, dtype=np.int32)
    fit_ids, held_out_ids = image_level_partition(
        canonical_ids, arm=normalized_arm, replicate_id=replicate_id
    )
    row_offsets = np.arange(DICTIONARY_SAMPLES_PER_IMAGE, dtype=np.int64)
    fit_rows = (
        fit_ids.astype(np.int64)[:, None] * DICTIONARY_SAMPLES_PER_IMAGE + row_offsets
    ).reshape(-1)
    fit_pool = pool[fit_rows].tocsr()
    expected_fit_rows = len(fit_ids) * DICTIONARY_SAMPLES_PER_IMAGE
    if fit_pool.shape != (expected_fit_rows, CRS_DESCRIPTOR_DIM):
        raise AssertionError("replicate fit pool is not exact 24 descriptors/fit image")
    traces: list[SphericalKMeansIterationDiagnostic] = []
    result = _registered_replicate_skm(
        arm=normalized_arm, replicate_id=replicate_id, diagnostics=traces
    ).fit(fit_pool)
    if not result.converged_:
        raise RuntimeError(
            f"{normalized_arm} replicate {replicate_id} did not converge at max_iter=500; STOP TRAIN STAGE"
        )
    original_centers = model.m_centers if normalized_arm == "M" else model.c_centers
    matching = centroid_only_hungarian_match(original_centers, result.cluster_centers_)
    primitive_maps = np.load(root / PRIMITIVE_MAPS_FILE, mmap_mode="r")
    jaccards = heldout_dense_jaccard(
        primitive_maps[held_out_ids],
        held_out_ids,
        arm=normalized_arm,
        original_centers=original_centers,
        replicate_centers=result.cluster_centers_,
        replicate_to_original=matching.replicate_to_original,
    )
    payload = {
        "experiment": np.asarray("PGM_MOTIF_STABILITY_REPLICATE"),
        "issue": np.asarray(ISSUE_NUMBER, dtype=np.int32),
        "scientific_source_sha": np.asarray(source_sha),
        "preregistration_sha": np.asarray(PREREGISTRATION_SHA),
        "substrate_identity_sha256": np.asarray(
            substrate_manifest["substrate_identity_sha256"]
        ),
        "accepted_train_model_sha256": np.asarray(model.artifact_sha256),
        "arm": np.asarray(normalized_arm),
        "replicate_id": np.asarray(replicate_id, dtype=np.int32),
        "replicate_seed": np.asarray(seed, dtype=np.uint64),
        "fit_image_ids": fit_ids,
        "held_out_image_ids": held_out_ids,
        "fit_image_ids_sha256": np.asarray(sha256_array(fit_ids)),
        "held_out_image_ids_sha256": np.asarray(sha256_array(held_out_ids)),
        "fit_descriptors_per_image": np.asarray(
            DICTIONARY_SAMPLES_PER_IMAGE, dtype=np.int32
        ),
        "held_out_positions_per_image": np.asarray(CRS_VALID_POSITIONS, dtype=np.int32),
        "replicate_centers": result.cluster_centers_.astype(np.float32),
        "original_to_replicate": matching.original_to_replicate,
        "replicate_to_original": matching.replicate_to_original,
        "matched_centroid_cosine": matching.matched_cosine,
        "held_out_jaccard": jaccards,
        "objective": np.asarray(result.objective_, dtype=np.float64),
        "n_iter": np.asarray(result.n_iter_, dtype=np.int32),
        "converged": np.asarray(result.converged_, dtype=np.bool_),
        "selected_init": np.asarray(result.init_index_, dtype=np.int32),
        "empty_reseeds": np.asarray(result.empty_reseeds_, dtype=np.int32),
        "per_init_n_iter": np.asarray(
            [x.n_iter for x in result.init_diagnostics_], dtype=np.int32
        ),
        "per_init_objective": np.asarray(
            [x.objective for x in result.init_diagnostics_], dtype=np.float64
        ),
        "per_init_converged": np.asarray(
            [x.converged for x in result.init_diagnostics_], dtype=np.bool_
        ),
        "per_init_empty_reseeds": np.asarray(
            [x.empty_reseeds for x in result.init_diagnostics_], dtype=np.int32
        ),
    }
    _atomic_savez(artifact_path, payload)
    record = _file_record(artifact_path)
    replicate_manifest = {
        "experiment": "PGM_MOTIF_STABILITY_REPLICATE",
        "issue": ISSUE_NUMBER,
        "scientific_source_sha": source_sha,
        "preregistration_sha": PREREGISTRATION_SHA,
        "substrate_identity_sha256": substrate_manifest["substrate_identity_sha256"],
        "arm": normalized_arm,
        "replicate_id": replicate_id,
        "seed": seed,
        "artifact": {"name": artifact_path.name, **record},
        "objective": result.objective_,
        "n_iter": result.n_iter_,
        "converged": result.converged_,
        "selected_init": result.init_index_,
        "environment": _environment(),
        "public_test_accessed": False,
        "private_test_accessed": False,
    }
    _atomic_json(manifest_path, replicate_manifest)
    _log(f"checkpointed {artifact_path.name} sha256={record['sha256']}")
    return replicate_manifest


def _load_complete_replicates(
    replicates_dir: Path, substrate_identity: str, source_sha: str
) -> dict[str, np.ndarray]:
    raw = {
        "M": np.empty((STABILITY_REPLICATES, CRS_K)),
        "C": np.empty((STABILITY_REPLICATES, CRS_K)),
    }
    expected_paths = {
        replicates_dir / _replicate_filename(arm, replicate_id)
        for arm in ("M", "C")
        for replicate_id in range(STABILITY_REPLICATES)
    }
    observed_paths = set(replicates_dir.glob("motif_stability_[MC]_rep_[0-9][0-9].npz"))
    if observed_paths != expected_paths:
        missing = sorted(path.name for path in expected_paths - observed_paths)
        extra = sorted(path.name for path in observed_paths - expected_paths)
        raise ValueError(f"replicate set mismatch: missing={missing}, extra={extra}")
    for arm in ("M", "C"):
        for replicate_id in range(STABILITY_REPLICATES):
            path = replicates_dir / _replicate_filename(arm, replicate_id)
            sidecar = path.with_suffix(".manifest.json")
            if not sidecar.is_file():
                raise FileNotFoundError(f"missing replicate manifest {sidecar.name}")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            if manifest["arm"] != arm or int(manifest["replicate_id"]) != replicate_id:
                raise ValueError("replicate manifest identity mismatch")
            _verify_file_record(path, manifest["artifact"])
            with np.load(path, allow_pickle=False) as artifact:
                if (
                    str(artifact["arm"].item()) != arm
                    or int(artifact["replicate_id"]) != replicate_id
                ):
                    raise ValueError("replicate identity mismatch")
                if str(artifact["scientific_source_sha"].item()) != source_sha:
                    raise ValueError("replicate source SHA mismatch")
                if (
                    str(artifact["substrate_identity_sha256"].item())
                    != substrate_identity
                ):
                    raise ValueError("replicate substrate mismatch")
                if (
                    str(artifact["accepted_train_model_sha256"].item())
                    != ACCEPTED_TRAIN_MODEL_SHA256
                ):
                    raise ValueError("replicate accepted Train model mismatch")
                if not bool(artifact["converged"]):
                    raise ValueError("non-converged replicate cannot be merged")
                fit, held = image_level_partition(
                    np.arange(TRAIN_ROWS, dtype=np.int32),
                    arm=arm,
                    replicate_id=replicate_id,
                )
                if not np.array_equal(
                    artifact["fit_image_ids"], fit
                ) or not np.array_equal(artifact["held_out_image_ids"], held):
                    raise ValueError("replicate partition mismatch")
                mapping = np.asarray(artifact["replicate_to_original"], dtype=np.int32)
                if not np.array_equal(
                    np.sort(mapping), np.arange(CRS_K, dtype=np.int32)
                ):
                    raise ValueError("replicate matching is not a permutation")
                centers = np.asarray(artifact["replicate_centers"], dtype=np.float32)
                if centers.shape != (CRS_K, CRS_DESCRIPTOR_DIM) or not np.all(
                    np.isfinite(centers)
                ):
                    raise ValueError("replicate centers invalid")
                if not np.allclose(
                    np.linalg.norm(centers, axis=1), 1.0, atol=1e-5, rtol=1e-5
                ):
                    raise ValueError("replicate centers are not unit normalized")
                value = np.asarray(artifact["held_out_jaccard"], dtype=np.float64)
                if (
                    value.shape != (CRS_K,)
                    or not np.all(np.isfinite(value))
                    or np.any((value < 0) | (value > 1))
                ):
                    raise ValueError("replicate Jaccard artifact invalid")
                raw[arm][replicate_id] = value
    return raw


def _fit_sparse_probe(
    x: sparse.csr_matrix, y: np.ndarray, *, arm: str
) -> SparseProbeState:
    data = sparse.csr_matrix(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if data.shape != (TRAIN_ROWS, SPARSE_FEATURE_DIM) or len(labels) != TRAIN_ROWS:
        raise ValueError("sparse probe input shape mismatch")
    if not np.all(np.isfinite(data.data)):
        raise FloatingPointError("sparse features contain non-finite values")
    classifier = LogisticRegression(
        C=LOGREG_C,
        solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT,
        max_iter=LOGREG_MAX_ITER,
        tol=LOGREG_TOL,
        random_state=MASTER_SEED,
    )
    classifier.fit(data, labels)
    n_iter = np.asarray(classifier.n_iter_, dtype=np.int64)
    if np.any(n_iter >= LOGREG_MAX_ITER):
        raise RuntimeError(f"{arm} sparse probe did not converge; STOP before Public")
    prediction = classifier.predict(data)
    per_class = f1_score(
        labels, prediction, labels=np.arange(7), average=None, zero_division=0
    ).astype(np.float64)
    return SparseProbeState(
        classes=np.asarray(classifier.classes_, dtype=np.int64),
        coef=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        n_iter=n_iter,
        train_accuracy=float(accuracy_score(labels, prediction)),
        train_macro_f1=float(per_class.mean()),
        per_class_f1=per_class,
        converged=True,
    )


def _build_sparse_features(
    assignments: np.ndarray, margins: np.ndarray, vocabulary: np.ndarray
) -> tuple[sparse.csr_matrix, list]:
    rows: list[sparse.csr_matrix] = []
    extractions = []
    for image_index in range(TRAIN_ROWS):
        extraction = extract_occurrences(
            assignments[image_index], margins[image_index], vocabulary
        )
        extractions.append(extraction)
        rows.append(
            sparse.csr_matrix(sparse_occurrence_feature(extraction.retained)[None, :])
        )
        if (image_index + 1) % 512 == 0 or image_index + 1 == TRAIN_ROWS:
            _log(f"sparse feature images {image_index + 1}/{TRAIN_ROWS}")
    return sparse.vstack(rows, format="csr", dtype=np.float32), extractions


def _point_biserial(
    total_count: int,
    total_sum: float,
    total_sq: float,
    type_count: np.ndarray,
    type_sum: np.ndarray,
) -> np.ndarray:
    n = float(total_count)
    mean = total_sum / n
    variance = max(0.0, total_sq / n - mean * mean)
    std = np.sqrt(variance)
    out = np.zeros(CRS_K, dtype=np.float64)
    if std == 0:
        return out
    for type_id in range(CRS_K):
        n1 = float(type_count[type_id])
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        mean1 = type_sum[type_id] / n1
        mean0 = (total_sum - type_sum[type_id]) / n0
        out[type_id] = (mean1 - mean0) * np.sqrt((n1 / n) * (n0 / n)) / std
    return out


def _nuisance_report(
    recurrence: np.lib.npyio.NpzFile, arm: str, qualified: np.ndarray
) -> dict[str, np.ndarray]:
    key = arm.lower()
    count = np.asarray(recurrence[f"{key}_nuisance_count"], dtype=np.int64)
    intensity = _point_biserial(
        int(recurrence[f"{key}_nuisance_position_count"]),
        float(recurrence[f"{key}_nuisance_intensity_total"]),
        float(recurrence[f"{key}_nuisance_intensity_sq_total"]),
        count,
        np.asarray(recurrence[f"{key}_nuisance_intensity_sum"]),
    )
    contrast = _point_biserial(
        int(recurrence[f"{key}_nuisance_position_count"]),
        float(recurrence[f"{key}_nuisance_contrast_total"]),
        float(recurrence[f"{key}_nuisance_contrast_sq_total"]),
        count,
        np.asarray(recurrence[f"{key}_nuisance_contrast_sum"]),
    )
    quadrants = np.asarray(
        recurrence[f"{key}_nuisance_quadrant_count"], dtype=np.float64
    )
    probabilities = np.divide(
        quadrants,
        quadrants.sum(axis=1, keepdims=True),
        out=np.zeros_like(quadrants),
        where=quadrants.sum(axis=1, keepdims=True) != 0,
    )
    log_probability = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probability, where=probabilities > 0)
    entropy = -np.sum(probabilities * log_probability, axis=1)
    spatial_concentration = 1.0 - entropy / np.log(4.0)
    border = np.divide(
        np.asarray(recurrence[f"{key}_nuisance_border_count"], dtype=np.float64),
        count,
        out=np.zeros(CRS_K, dtype=np.float64),
        where=count != 0,
    )
    return {
        "type_ids": qualified.astype(np.int32),
        "local_intensity_point_biserial": intensity[qualified],
        "local_contrast_point_biserial": contrast[qualified],
        "spatial_concentration": spatial_concentration[qualified],
        "border_concentration": border[qualified],
    }


def _representative_diagnostics(
    extractions: list,
    qualified: np.ndarray,
    images_uint8: np.ndarray,
    canonical_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    type_to_row = {int(type_id): row for row, type_id in enumerate(qualified.tolist())}
    spatial_maps = np.zeros((len(qualified), CRS_SIDE, CRS_SIDE), dtype=np.int32)
    candidates: list[list[tuple[float, int, int, int]]] = [
        [] for _ in range(len(qualified))
    ]
    for image_index, extraction in enumerate(extractions):
        for occurrence in extraction.retained:
            row = type_to_row.get(occurrence.type_id)
            if row is None:
                continue
            spatial_maps[row, occurrence.row, occurrence.col] += 1
            candidates[row].append(
                (
                    -occurrence.margin,
                    int(canonical_ids[image_index]),
                    occurrence.row,
                    occurrence.col,
                )
            )
    image_ids = np.full((len(qualified), EXEMPLARS_PER_TYPE), -1, dtype=np.int32)
    rows = np.full_like(image_ids, -1)
    cols = np.full_like(image_ids, -1)
    margins = np.zeros(image_ids.shape, dtype=np.float32)
    valid = np.zeros(image_ids.shape, dtype=np.bool_)
    patches = np.zeros((len(qualified), EXEMPLARS_PER_TYPE, 9, 9), dtype=np.uint8)
    for type_row, values in enumerate(candidates):
        values.sort(key=lambda item: (item[0], item[1], item[2] * CRS_SIDE + item[3]))
        for rank, (negative_margin, image_id, row, col) in enumerate(
            values[:EXEMPLARS_PER_TYPE]
        ):
            image_ids[type_row, rank] = image_id
            rows[type_row, rank] = row
            cols[type_row, rank] = col
            margins[type_row, rank] = -negative_margin
            valid[type_row, rank] = True
            center_y, center_x = row + 6, col + 6
            patches[type_row, rank] = images_uint8[
                image_id, center_y - 4 : center_y + 5, center_x - 4 : center_x + 5
            ]
    return {
        "m_spatial_maps": spatial_maps,
        "m_exemplar_image_ids": image_ids,
        "m_exemplar_rows": rows,
        "m_exemplar_cols": cols,
        "m_exemplar_margins": margins,
        "m_exemplar_valid": valid,
        "m_exemplar_patches_uint8": patches,
    }


def finalize_train_stage(
    *,
    train_csv: str | Path,
    substrate_dir: str | Path,
    replicates_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("final Train output directory must be empty")
    source_sha = _scientific_sha()
    substrate = load_substrate_manifest(substrate_dir)
    if substrate["scientific_source_sha"] != source_sha:
        raise ValueError("finalizer source differs from substrate source")
    raw = _load_complete_replicates(
        Path(replicates_dir), substrate["substrate_identity_sha256"], source_sha
    )
    root = Path(substrate_dir)
    with np.load(root / RECURRENCE_FILE, allow_pickle=False) as recurrence:
        m_support = np.asarray(recurrence["m_distinct_image_support"], dtype=np.int32)
        c_support = np.asarray(recurrence["c_distinct_image_support"], dtype=np.int32)
        summaries = {arm: summarize_jaccards(raw[arm]) for arm in ("M", "C")}
        q_m = qualified_type_ids(m_support, summaries["M"]["median"])
        q_c = qualified_type_ids(c_support, summaries["C"]["median"])
        m_match, c_match, q_star = matched_vocabularies(
            q_m,
            q_c,
            m_support=m_support,
            c_support=c_support,
            m_median=summaries["M"]["median"],
            c_median=summaries["C"]["median"],
        )
        merged_payload = {
            "scientific_source_sha": np.asarray(source_sha),
            "substrate_identity_sha256": np.asarray(
                substrate["substrate_identity_sha256"]
            ),
            "m_raw_jaccard": raw["M"],
            "c_raw_jaccard": raw["C"],
            "m_support": m_support,
            "c_support": c_support,
            "m_support_fraction": m_support.astype(np.float64) / TRAIN_ROWS,
            "c_support_fraction": c_support.astype(np.float64) / TRAIN_ROWS,
            "q_m": q_m,
            "q_c": q_c,
            "m_qualified": np.isin(np.arange(CRS_K), q_m),
            "c_qualified": np.isin(np.arange(CRS_K), q_c),
            "m_match_types": m_match,
            "c_match_types": c_match,
            "q_star": np.asarray(q_star, dtype=np.int32),
        }
        for arm in ("M", "C"):
            for name, value in summaries[arm].items():
                merged_payload[f"{arm.lower()}_jaccard_{name}"] = value
        _atomic_savez(out / STABILITY_MERGED_FILE, merged_payload)
        qualification = {
            "q_m": len(q_m),
            "q_c": len(q_c),
            "q_star": q_star,
            "match_fraction_of_m_all": None if len(q_m) == 0 else q_star / len(q_m),
        }
        if q_star == 0:
            summary = {
                "status": "MOTIF_TRAIN_QUALIFICATION_EMPTY_STOP_BEFORE_PUBLIC",
                "issue": ISSUE_NUMBER,
                "scientific_source_sha": source_sha,
                "preregistration_sha": PREREGISTRATION_SHA,
                "qualification": qualification,
                "public_test_accessed": False,
                "private_test_accessed": False,
            }
            _atomic_json(out / FINAL_SUMMARY_FILE, summary)
            _atomic_json(
                out / FINAL_MANIFEST_FILE,
                {
                    "issue": ISSUE_NUMBER,
                    "outputs": {
                        STABILITY_MERGED_FILE: _file_record(
                            out / STABILITY_MERGED_FILE
                        ),
                        FINAL_SUMMARY_FILE: _file_record(out / FINAL_SUMMARY_FILE),
                    },
                    "public_test_accessed": False,
                    "private_test_accessed": False,
                },
            )
            return summary

        m_assign = np.load(root / M_ASSIGNMENTS_FILE, mmap_mode="r")
        c_assign = np.load(root / C_ASSIGNMENTS_FILE, mmap_mode="r")
        m_margin = np.load(root / M_MARGINS_FILE, mmap_mode="r")
        c_margin = np.load(root / C_MARGINS_FILE, mmap_mode="r")
        train_pixels = load_pixels_only(train_csv, role="train")
        if train_pixels.sha256 != substrate["train_sha256"]:
            raise ValueError("Train pixels changed before sparse feature construction")
        features: dict[str, sparse.csr_matrix] = {}
        extractions = {}
        features["S_M_all"], extractions["S_M_all"] = _build_sparse_features(
            m_assign, m_margin, q_m
        )
        features["S_M_match"], extractions["S_M_match"] = _build_sparse_features(
            m_assign, m_margin, m_match
        )
        features["S_C_match"], extractions["S_C_match"] = _build_sparse_features(
            c_assign, c_margin, c_match
        )
        feature_records = {}
        for arm, matrix in features.items():
            path = out / f"motif_train_{arm.lower()}_features.npz"
            _atomic_sparse(path, matrix)
            feature_records[arm] = _file_record(path)

        diagnostic_payload: dict[str, np.ndarray] = {
            "q_m": q_m,
            "q_c": q_c,
            "m_match_types": m_match,
            "c_match_types": c_match,
        }
        diagnostic_summary = {}
        for arm, values in extractions.items():
            diagnostics = occurrence_diagnostics(values)
            diagnostic_payload[f"{arm.lower()}_component_sizes"] = diagnostics[
                "component_sizes"
            ]
            diagnostic_payload[f"{arm.lower()}_pre_cap_nodes"] = diagnostics[
                "pre_cap_nodes_per_image"
            ]
            diagnostic_payload[f"{arm.lower()}_post_cap_nodes"] = diagnostics[
                "post_cap_nodes_per_image"
            ]
            diagnostic_summary[arm] = {
                **diagnostics["component_summary"],
                "cap_binding_fraction": diagnostics["cap_binding_fraction"],
                "zero_node_image_count": diagnostics["zero_node_image_count"],
            }
        for arm, qualified in (("M", q_m), ("C", q_c)):
            report = _nuisance_report(recurrence, arm, qualified)
            for name, value in report.items():
                diagnostic_payload[f"{arm.lower()}_nuisance_{name}"] = value
        diagnostic_payload.update(
            _representative_diagnostics(
                extractions["S_M_all"],
                q_m,
                train_pixels.images_uint8,
                train_pixels.canonical_ids,
            )
        )
        _atomic_savez(out / OCCURRENCE_DIAGNOSTICS_FILE, diagnostic_payload)

    # Sparse representations and diagnostics are frozen before this first label read.
    labels = load_labels_downstream(
        train_csv, role="train", expected_sha256=substrate["train_sha256"]
    )
    probes = {
        arm: _fit_sparse_probe(matrix, labels, arm=arm)
        for arm, matrix in features.items()
    }
    model_payload: dict[str, np.ndarray] = {
        "experiment": np.asarray("PGM_MOTIF_QUALIFICATION_TRAIN"),
        "issue": np.asarray(ISSUE_NUMBER, dtype=np.int32),
        "scientific_source_sha": np.asarray(source_sha),
        "preregistration_sha": np.asarray(PREREGISTRATION_SHA),
        "substrate_identity_sha256": np.asarray(substrate["substrate_identity_sha256"]),
        "accepted_train_model_sha256": np.asarray(ACCEPTED_TRAIN_MODEL_SHA256),
        "q_m": q_m,
        "q_c": q_c,
        "m_match_types": m_match,
        "c_match_types": c_match,
        "q_star": np.asarray(q_star, dtype=np.int32),
        "support_threshold": np.asarray(SUPPORT_THRESHOLD, dtype=np.int32),
        "jaccard_threshold": np.asarray(JACCARD_THRESHOLD, dtype=np.float64),
        "max_occurrences_per_image": np.asarray(
            MAX_OCCURRENCES_PER_IMAGE, dtype=np.int32
        ),
        "logreg_c": np.asarray(LOGREG_C),
        "logreg_solver": np.asarray(LOGREG_SOLVER),
        "logreg_class_weight": np.asarray(LOGREG_CLASS_WEIGHT),
        "logreg_max_iter": np.asarray(LOGREG_MAX_ITER, dtype=np.int32),
        "logreg_tol": np.asarray(LOGREG_TOL),
    }
    for arm, state in probes.items():
        key = arm.lower()
        model_payload[f"{key}_classes"] = state.classes
        model_payload[f"{key}_coef"] = state.coef
        model_payload[f"{key}_intercept"] = state.intercept
        model_payload[f"{key}_n_iter"] = state.n_iter
    _atomic_savez(out / FINAL_MODEL_FILE, model_payload)
    outputs = {
        name: _file_record(out / name)
        for name in (
            STABILITY_MERGED_FILE,
            OCCURRENCE_DIAGNOSTICS_FILE,
            FINAL_MODEL_FILE,
        )
    }
    summary = {
        "status": "MOTIF_TRAIN_STAGE_COMPLETE_PUBLIC_STILL_LOCKED",
        "issue": ISSUE_NUMBER,
        "scientific_source_sha": source_sha,
        "preregistration_sha": PREREGISTRATION_SHA,
        "substrate_identity_sha256": substrate["substrate_identity_sha256"],
        "qualification": qualification,
        "registered": {
            "replicates_per_arm": STABILITY_REPLICATES,
            "fit_fraction": FIT_FRACTION,
            "fit_descriptors_per_image": DICTIONARY_SAMPLES_PER_IMAGE,
            "held_out_positions_per_image": CRS_VALID_POSITIONS,
            "hungarian_input": "centroid cosine only",
            "support_threshold": SUPPORT_THRESHOLD,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "connectivity": 8,
            "max_occurrences_per_image": MAX_OCCURRENCES_PER_IMAGE,
            "sparse_feature_dim": SPARSE_FEATURE_DIM,
            "nuisance_diagnostics": {
                "local_intensity": "point-biserial association between dense type membership and 9x9 raw-pixel mean",
                "local_contrast": "point-biserial association between dense type membership and 9x9 raw-pixel standard deviation",
                "spatial_concentration": "one minus four-quadrant entropy normalized by log(4)",
                "border_concentration": "fraction of dense assignments within the outer four cells of the 36x36 domain",
                "selection_effect": "report-only; never changes qualified sets",
            },
        },
        "occurrence_diagnostics": diagnostic_summary,
        "train_diagnostics_only": {
            arm: {
                "accuracy": state.train_accuracy,
                "macro_f1": state.train_macro_f1,
                "per_class_f1": state.per_class_f1.tolist(),
                "n_iter": state.n_iter.tolist(),
                "converged": state.converged,
            }
            for arm, state in probes.items()
        },
        "feature_artifacts": feature_records,
        "environment": _environment(),
        "public_test_accessed": False,
        "private_test_accessed": False,
    }
    _atomic_json(out / FINAL_SUMMARY_FILE, summary)
    outputs[FINAL_SUMMARY_FILE] = _file_record(out / FINAL_SUMMARY_FILE)
    for arm, record in feature_records.items():
        outputs[f"motif_train_{arm.lower()}_features.npz"] = record
    _atomic_json(
        out / FINAL_MANIFEST_FILE,
        {
            "issue": ISSUE_NUMBER,
            "scientific_source_sha": source_sha,
            "outputs": outputs,
            "public_test_accessed": False,
            "private_test_accessed": False,
        },
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #86 Train-only motif qualification"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    substrate = sub.add_parser("build-substrate")
    substrate.add_argument("--train-csv", required=True)
    substrate.add_argument("--dictionary-npz", required=True)
    substrate.add_argument("--train-model-npz", required=True)
    substrate.add_argument("--output-dir", required=True)
    replicate = sub.add_parser("run-replicate")
    replicate.add_argument("--arm", required=True, choices=("M", "C"))
    replicate.add_argument("--replicate-id", required=True, type=int, choices=range(20))
    replicate.add_argument("--substrate-dir", required=True)
    replicate.add_argument("--train-model-npz", required=True)
    replicate.add_argument("--output-dir", required=True)
    finalize = sub.add_parser("finalize-train")
    finalize.add_argument("--train-csv", required=True)
    finalize.add_argument("--substrate-dir", required=True)
    finalize.add_argument("--replicates-dir", required=True)
    finalize.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "build-substrate":
        build_train_substrate(
            train_csv=args.train_csv,
            dictionary_npz=args.dictionary_npz,
            train_model_npz=args.train_model_npz,
            output_dir=args.output_dir,
        )
    elif args.mode == "run-replicate":
        run_stability_replicate(
            arm=args.arm,
            replicate_id=args.replicate_id,
            substrate_dir=args.substrate_dir,
            train_model_npz=args.train_model_npz,
            output_dir=args.output_dir,
        )
    else:
        finalize_train_stage(
            train_csv=args.train_csv,
            substrate_dir=args.substrate_dir,
            replicates_dir=args.replicates_dir,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
