from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from numpy.lib.format import open_memmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .crs_runtime import (
    load_frozen_primitive_substrate,
    primitive_map_from_uint8,
    sampled_training_descriptors,
)
from .crs_stage import (
    C_DIM,
    CRS_DESCRIPTOR_DIM,
    CRS_K,
    CRS_VALID_POSITIONS,
    DICTIONARY_SAMPLES_PER_IMAGE,
    E01_V533_DICTIONARY_SHA256,
    M_DIM,
    MASTER_SEED,
    P_DIM,
    PRIMITIVE_K,
    PRIMITIVE_SIDE,
    SphericalKMeans,
    SphericalKMeansResult,
    composition_descriptors,
    crs_image_feature,
    p_image_feature,
    permute_cell_blocks,
)
from .e01_runner import TRAIN_ROWS
from .e02_runner import TRAIN_SHA256, load_labels_downstream, load_pixels_only, sha256_file


SOURCE_BASE_SHA = "49e65e1032f1af13fd615a2e6f7a64f20760684b"
PREREGISTRATION_PATH = "research/pixel_relational_motif_e0/CRS_STAGE_PREREGISTRATION.md"

SKM_N_INIT = 3
SKM_MAX_ITER = 50
SKM_TOL = 1e-6
SKM_BATCH_SIZE = 8192

LOGREG_C = 1.0
LOGREG_SOLVER = "lbfgs"
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_MAX_ITER = 5000
# Explicitly fix sklearn's long-standing lbfgs default tolerance. This is an
# implementation clarification before PublicTest access, not a tuned value.
LOGREG_TOL = 1e-4

FEATURE_DT = np.float32
POOL_DT = np.float32
PRIMITIVE_DT = np.int16


@dataclass(frozen=True)
class ProbeState:
    classes: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    n_iter: np.ndarray
    train_accuracy: float
    train_macro_f1: float
    converged: bool


def _log(message: str) -> None:
    print(f"[PGM-CRS-TRAIN] {message}", flush=True)


def _environment_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _fit_registered_skm(pool: np.ndarray, *, arm: str) -> SphericalKMeansResult:
    result = SphericalKMeans(
        n_clusters=CRS_K,
        n_init=SKM_N_INIT,
        max_iter=SKM_MAX_ITER,
        tol=SKM_TOL,
        random_state=MASTER_SEED,
        batch_size=SKM_BATCH_SIZE,
    ).fit(pool)
    _log(
        f"{arm} spherical-kmeans objective={result.objective_:.6f} "
        f"iter={result.n_iter_} converged={result.converged_} "
        f"init={result.init_index_} empty_reseeds={result.empty_reseeds_}"
    )
    if not result.converged_:
        raise RuntimeError(
            f"{arm} spherical-kmeans hit the registered max_iter={SKM_MAX_ITER}; "
            "do not inspect PublicTest. A global technical max_iter amendment is required."
        )
    if result.cluster_centers_.shape != (CRS_K, CRS_DESCRIPTOR_DIM):
        raise AssertionError(f"{arm} CRS center shape mismatch")
    norms = np.linalg.norm(result.cluster_centers_, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise AssertionError(f"{arm} CRS centers are not unit norm")
    return result


def _cluster_diagnostics(labels: np.ndarray) -> dict[str, float | int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=CRS_K)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("empty cluster-label vector")
    p = counts[counts > 0].astype(np.float64) / float(total)
    entropy = float(-(p * np.log(p)).sum())
    return {
        "n_assignments": total,
        "empty_clusters": int(np.sum(counts == 0)),
        "tiny_clusters_lt10": int(np.sum(counts < 10)),
        "min_occupancy": int(counts.min()),
        "median_occupancy": float(np.median(counts)),
        "max_occupancy": int(counts.max()),
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / np.log(CRS_K)),
    }


def _fit_fixed_probe(x: np.ndarray, y: np.ndarray, *, arm: str) -> ProbeState:
    data = np.asarray(x)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if data.ndim != 2 or len(data) != len(labels):
        raise ValueError("probe feature/label shape mismatch")
    if not np.all(np.isfinite(data)):
        raise FloatingPointError(f"{arm} features contain non-finite values")
    if set(np.unique(labels).tolist()) != set(range(7)):
        raise ValueError("Training labels must contain FER classes 0..6")

    clf = LogisticRegression(
        C=LOGREG_C,
        solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT,
        max_iter=LOGREG_MAX_ITER,
        tol=LOGREG_TOL,
        random_state=MASTER_SEED,
    )
    clf.fit(data, labels)
    classes = np.asarray(clf.classes_, dtype=np.int64)
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    n_iter = np.asarray(clf.n_iter_, dtype=np.int64)

    if not np.array_equal(classes, np.arange(7, dtype=np.int64)):
        raise AssertionError(f"{arm} classifier class-order invariant violated")
    if coef.shape != (7, data.shape[1]) or intercept.shape != (7,):
        raise AssertionError(f"{arm} classifier parameter shape invariant violated")
    converged = bool(np.all(n_iter < LOGREG_MAX_ITER))
    if not converged:
        raise RuntimeError(
            f"{arm} logistic probe reached max_iter={LOGREG_MAX_ITER}; "
            "do not inspect PublicTest. Only a global max_iter increase is allowed."
        )

    pred = clf.predict(data)
    return ProbeState(
        classes=classes,
        coef=coef,
        intercept=intercept,
        n_iter=n_iter,
        train_accuracy=float(accuracy_score(labels, pred)),
        train_macro_f1=float(
            f1_score(labels, pred, labels=np.arange(7), average="macro", zero_division=0)
        ),
        converged=converged,
    )


def probe_predict(state: ProbeState, x: np.ndarray) -> np.ndarray:
    """Version-light inference from the stored multinomial linear parameters."""
    data = np.asarray(x)
    logits = data @ state.coef.T + state.intercept[None, :]
    return state.classes[np.argmax(logits, axis=1)]


def _create_training_intermediates(
    images_uint8: np.ndarray,
    substrate,
    *,
    work_dir: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Build one primitive cache plus matched sampled M/C dictionary pools.

    Labels are not accepted by this function. The sampled pools are exactly
    24 rows/image; the primitive cache is reused later for dense extraction so
    the expensive frozen-GMM assignment is not recomputed.
    """
    n_rows = TRAIN_ROWS * DICTIONARY_SAMPLES_PER_IMAGE
    primitive_path = work_dir / "train_primitive_maps.npy"
    m_path = work_dir / "crs_m_pool.npy"
    c_path = work_dir / "crs_c_pool.npy"

    primitive_maps = open_memmap(
        primitive_path,
        mode="w+",
        dtype=PRIMITIVE_DT,
        shape=(TRAIN_ROWS, PRIMITIVE_SIDE, PRIMITIVE_SIDE),
    )
    m_pool = open_memmap(
        m_path, mode="w+", dtype=POOL_DT, shape=(n_rows, CRS_DESCRIPTOR_DIM)
    )
    c_pool = open_memmap(
        c_path, mode="w+", dtype=POOL_DT, shape=(n_rows, CRS_DESCRIPTOR_DIM)
    )

    h_primitive = hashlib.sha256()
    h_sample_identity = hashlib.sha256()
    h_m = hashlib.sha256()
    h_c = hashlib.sha256()

    row = 0
    for image_id in range(TRAIN_ROWS):
        primitive = primitive_map_from_uint8(images_uint8[image_id], substrate)
        if primitive.shape != (PRIMITIVE_SIDE, PRIMITIVE_SIDE):
            raise AssertionError("primitive-cache shape invariant violated")
        primitive_maps[image_id] = primitive
        h_primitive.update(np.ascontiguousarray(primitive, dtype=PRIMITIVE_DT).tobytes())

        centers, m, c = sampled_training_descriptors(
            primitive, image_id=image_id, master_seed=MASTER_SEED
        )
        if len(centers) != DICTIONARY_SAMPLES_PER_IMAGE:
            raise AssertionError("registered 24-patch sampling invariant violated")
        h_sample_identity.update(np.asarray([image_id], dtype=np.int32).tobytes())
        h_sample_identity.update(np.asarray(centers, dtype=np.int32).tobytes())

        sl = slice(row, row + DICTIONARY_SAMPLES_PER_IMAGE)
        m_pool[sl] = m
        c_pool[sl] = c
        h_m.update(np.ascontiguousarray(m).tobytes())
        h_c.update(np.ascontiguousarray(c).tobytes())
        row += DICTIONARY_SAMPLES_PER_IMAGE

        if (image_id + 1) % 250 == 0 or image_id + 1 == TRAIN_ROWS:
            _log(f"primitive/sample pool {image_id + 1}/{TRAIN_ROWS} images")

    if row != n_rows:
        raise AssertionError("dictionary pool row-count invariant violated")

    primitive_maps.flush()
    m_pool.flush()
    c_pool.flush()
    del primitive_maps, m_pool, c_pool
    return primitive_path, m_path, c_path, {
        "primitive_maps": h_primitive.hexdigest(),
        "sample_identity": h_sample_identity.hexdigest(),
        "M_pool": h_m.hexdigest(),
        "C_pool": h_c.hexdigest(),
    }


def _build_dense_features(
    primitive_path: Path,
    m_dict: SphericalKMeansResult,
    c_dict: SphericalKMeansResult,
    *,
    work_dir: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    primitive_maps = np.load(primitive_path, mmap_mode="r")
    if primitive_maps.shape != (TRAIN_ROWS, PRIMITIVE_SIDE, PRIMITIVE_SIDE):
        raise AssertionError("primitive cache shape mismatch")

    p_path = work_dir / "train_P.npy"
    m_path = work_dir / "train_M.npy"
    c_path = work_dir / "train_C.npy"
    p_features = open_memmap(p_path, mode="w+", dtype=FEATURE_DT, shape=(TRAIN_ROWS, P_DIM))
    m_features = open_memmap(m_path, mode="w+", dtype=FEATURE_DT, shape=(TRAIN_ROWS, M_DIM))
    c_features = open_memmap(c_path, mode="w+", dtype=FEATURE_DT, shape=(TRAIN_ROWS, C_DIM))
    hp = hashlib.sha256()
    hm = hashlib.sha256()
    hc = hashlib.sha256()

    dense_indices = np.arange(CRS_VALID_POSITIONS, dtype=np.int32)
    for image_id in range(TRAIN_ROWS):
        primitive = np.asarray(primitive_maps[image_id], dtype=PRIMITIVE_DT)
        p = p_image_feature(primitive)

        m_desc = composition_descriptors(primitive, normalize=True)
        if m_desc.shape != (CRS_VALID_POSITIONS, CRS_DESCRIPTOR_DIM):
            raise AssertionError("dense M descriptor shape invariant violated")
        m_ids = m_dict.predict(m_desc)
        if m_ids.shape != (CRS_VALID_POSITIONS,):
            raise AssertionError("dense M assignment count invariant violated")
        m = crs_image_feature(m_ids)

        c_desc = permute_cell_blocks(
            m_desc,
            split_id="train",
            image_id=image_id,
            center_indices=dense_indices,
            master_seed=MASTER_SEED,
        )
        c_ids = c_dict.predict(c_desc)
        if c_ids.shape != (CRS_VALID_POSITIONS,):
            raise AssertionError("dense C assignment count invariant violated")
        c = crs_image_feature(c_ids)

        p_features[image_id] = p
        m_features[image_id] = m
        c_features[image_id] = c
        hp.update(np.ascontiguousarray(p).tobytes())
        hm.update(np.ascontiguousarray(m).tobytes())
        hc.update(np.ascontiguousarray(c).tobytes())

        if (image_id + 1) % 250 == 0 or image_id + 1 == TRAIN_ROWS:
            _log(f"dense features {image_id + 1}/{TRAIN_ROWS} images")

    p_features.flush()
    m_features.flush()
    c_features.flush()
    del p_features, m_features, c_features, primitive_maps
    return p_path, m_path, c_path, {
        "P": hp.hexdigest(),
        "M": hm.hexdigest(),
        "C": hc.hexdigest(),
    }


def _load_memmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _save_model_artifact(
    path: Path,
    *,
    train_sha256: str,
    dictionary_sha256: str,
    intermediate_hashes: dict[str, str],
    feature_hashes: dict[str, str],
    m_dict: SphericalKMeansResult,
    c_dict: SphericalKMeansResult,
    probes: dict[str, ProbeState],
) -> None:
    payload: dict[str, np.ndarray] = {
        "source_base_sha": np.asarray(SOURCE_BASE_SHA),
        "preregistration_path": np.asarray(PREREGISTRATION_PATH),
        "train_sha256": np.asarray(train_sha256),
        "source_dictionary_sha256": np.asarray(dictionary_sha256),
        "primitive_maps_sha256": np.asarray(intermediate_hashes["primitive_maps"]),
        "sample_identity_sha256": np.asarray(intermediate_hashes["sample_identity"]),
        "m_pool_sha256": np.asarray(intermediate_hashes["M_pool"]),
        "c_pool_sha256": np.asarray(intermediate_hashes["C_pool"]),
        "p_feature_sha256": np.asarray(feature_hashes["P"]),
        "m_feature_sha256": np.asarray(feature_hashes["M"]),
        "c_feature_sha256": np.asarray(feature_hashes["C"]),
        "master_seed": np.asarray(MASTER_SEED, dtype=np.int32),
        "dictionary_samples_per_image": np.asarray(
            DICTIONARY_SAMPLES_PER_IMAGE, dtype=np.int32
        ),
        "crs_k": np.asarray(CRS_K, dtype=np.int32),
        "crs_descriptor_dim": np.asarray(CRS_DESCRIPTOR_DIM, dtype=np.int32),
        "m_crs_centers": m_dict.cluster_centers_.astype(np.float32),
        "m_skm_objective": np.asarray(m_dict.objective_, dtype=np.float64),
        "m_skm_n_iter": np.asarray(m_dict.n_iter_, dtype=np.int32),
        "m_skm_init_index": np.asarray(m_dict.init_index_, dtype=np.int32),
        "m_skm_empty_reseeds": np.asarray(m_dict.empty_reseeds_, dtype=np.int32),
        "c_crs_centers": c_dict.cluster_centers_.astype(np.float32),
        "c_skm_objective": np.asarray(c_dict.objective_, dtype=np.float64),
        "c_skm_n_iter": np.asarray(c_dict.n_iter_, dtype=np.int32),
        "c_skm_init_index": np.asarray(c_dict.init_index_, dtype=np.int32),
        "c_skm_empty_reseeds": np.asarray(c_dict.empty_reseeds_, dtype=np.int32),
        "logreg_c": np.asarray(LOGREG_C, dtype=np.float64),
        "logreg_solver": np.asarray(LOGREG_SOLVER),
        "logreg_class_weight": np.asarray(LOGREG_CLASS_WEIGHT),
        "logreg_max_iter": np.asarray(LOGREG_MAX_ITER, dtype=np.int32),
        "logreg_tol": np.asarray(LOGREG_TOL, dtype=np.float64),
    }
    for arm, state in probes.items():
        key = arm.lower()
        payload[f"{key}_classes"] = state.classes.astype(np.int64)
        payload[f"{key}_coef"] = state.coef.astype(np.float64)
        payload[f"{key}_intercept"] = state.intercept.astype(np.float64)
        payload[f"{key}_n_iter"] = state.n_iter.astype(np.int64)
    np.savez_compressed(path, **payload)


def run_train_stage(
    *,
    train_csv: str | Path,
    dictionary_npz: str | Path,
    output_dir: str | Path,
    keep_work: bool = False,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [
        out_dir / "crs_train_model.npz",
        out_dir / "crs_train_summary.json",
    ]
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite registered Train-stage outputs: " + ", ".join(existing)
        )

    n_pool_rows = TRAIN_ROWS * DICTIONARY_SAMPLES_PER_IMAGE
    pool_bytes = 2 * n_pool_rows * CRS_DESCRIPTOR_DIM * np.dtype(POOL_DT).itemsize
    primitive_bytes = (
        TRAIN_ROWS * PRIMITIVE_SIDE * PRIMITIVE_SIDE * np.dtype(PRIMITIVE_DT).itemsize
    )
    feature_bytes = TRAIN_ROWS * (P_DIM + M_DIM + C_DIM) * np.dtype(FEATURE_DT).itemsize
    required_bytes = int(pool_bytes + primitive_bytes + feature_bytes + (1 << 30))
    free_bytes = shutil.disk_usage(out_dir).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient free disk for registered Train stage: need at least "
            f"{required_bytes / (1 << 30):.2f} GiB, have {free_bytes / (1 << 30):.2f} GiB"
        )

    dictionary_path = Path(dictionary_npz)
    dictionary_sha = sha256_file(dictionary_path)
    if dictionary_sha != E01_V533_DICTIONARY_SHA256:
        raise ValueError(
            f"CRS stage requires v533 dictionary SHA256 {E01_V533_DICTIONARY_SHA256}, "
            f"got {dictionary_sha}"
        )

    substrate = load_frozen_primitive_substrate(dictionary_path)
    train = load_pixels_only(train_csv, role="train")
    if train.sha256 != TRAIN_SHA256:
        raise ValueError("Training SHA256 mismatch")
    if len(train.images_uint8) != TRAIN_ROWS:
        raise AssertionError("official Train row-count invariant violated")

    work_dir = Path(tempfile.mkdtemp(prefix="crs_train_", dir=out_dir))
    _log(f"temporary work dir: {work_dir}")
    try:
        primitive_path, m_pool_path, c_pool_path, intermediate_hashes = (
            _create_training_intermediates(
                train.images_uint8, substrate, work_dir=work_dir
            )
        )
        _log(f"sample identity sha256={intermediate_hashes['sample_identity']}")

        m_pool = _load_memmap(m_pool_path)
        m_dict = _fit_registered_skm(m_pool, arm="M")
        m_diag = _cluster_diagnostics(m_dict.labels_)
        del m_pool

        c_pool = _load_memmap(c_pool_path)
        c_dict = _fit_registered_skm(c_pool, arm="C")
        c_diag = _cluster_diagnostics(c_dict.labels_)
        del c_pool

        # Pools are construction intermediates. The hashes and learned centers
        # are retained; the ~6 GiB sampled pool files need not be scientific outputs.
        if not keep_work:
            m_pool_path.unlink(missing_ok=True)
            c_pool_path.unlink(missing_ok=True)

        p_path, m_path, c_path, feature_hashes = _build_dense_features(
            primitive_path, m_dict, c_dict, work_dir=work_dir
        )

        # Labels are deliberately read only after both label-free dictionaries
        # and all P/M/C dense feature matrices are frozen.
        y = load_labels_downstream(
            train_csv, role="train", expected_sha256=train.sha256
        )
        class_counts = np.bincount(y, minlength=7).astype(np.int64)
        if int(class_counts.sum()) != TRAIN_ROWS:
            raise AssertionError("Training class-count invariant violated")

        probes = {
            "P": _fit_fixed_probe(_load_memmap(p_path), y, arm="P"),
            "M": _fit_fixed_probe(_load_memmap(m_path), y, arm="M"),
            "C": _fit_fixed_probe(_load_memmap(c_path), y, arm="C"),
        }

        # Verify that stored linear parameters reproduce a valid class prediction
        # path independently of sklearn estimator serialization.
        for arm, path in (("P", p_path), ("M", m_path), ("C", c_path)):
            pred = probe_predict(probes[arm], _load_memmap(path)[:16])
            if pred.shape != (16,) or np.any((pred < 0) | (pred > 6)):
                raise AssertionError(f"{arm} stored-probe inference invariant violated")

        artifact_path = out_dir / "crs_train_model.npz"
        _save_model_artifact(
            artifact_path,
            train_sha256=train.sha256,
            dictionary_sha256=dictionary_sha,
            intermediate_hashes=intermediate_hashes,
            feature_hashes=feature_hashes,
            m_dict=m_dict,
            c_dict=c_dict,
            probes=probes,
        )
        artifact_sha = sha256_file(artifact_path)

        summary = {
            "status": "TRAIN_STAGE_COMPLETE_PUBLIC_STILL_LOCKED",
            "source_base_sha": SOURCE_BASE_SHA,
            "preregistration_path": PREREGISTRATION_PATH,
            "train_sha256": train.sha256,
            "source_dictionary_sha256": dictionary_sha,
            "model_artifact": artifact_path.name,
            "model_artifact_sha256": artifact_sha,
            "environment": _environment_manifest(),
            "registered": {
                "master_seed": MASTER_SEED,
                "primitive_k": PRIMITIVE_K,
                "primitive_map": [PRIMITIVE_SIDE, PRIMITIVE_SIDE],
                "valid_domain": [36, 36],
                "valid_positions": CRS_VALID_POSITIONS,
                "composition_support": [9, 9],
                "cell_grid": [3, 3],
                "descriptor_dim": CRS_DESCRIPTOR_DIM,
                "dictionary_samples_per_image": DICTIONARY_SAMPLES_PER_IMAGE,
                "crs_k": CRS_K,
                "spherical_kmeans": {
                    "n_init": SKM_N_INIT,
                    "max_iter": SKM_MAX_ITER,
                    "tol": SKM_TOL,
                    "batch_size": SKM_BATCH_SIZE,
                },
                "p_dim": P_DIM,
                "m_dim": M_DIM,
                "c_dim": C_DIM,
                "logistic": {
                    "C": LOGREG_C,
                    "solver": LOGREG_SOLVER,
                    "class_weight": LOGREG_CLASS_WEIGHT,
                    "max_iter": LOGREG_MAX_ITER,
                    "tol": LOGREG_TOL,
                },
            },
            "intermediate_sha256": intermediate_hashes,
            "feature_sha256": feature_hashes,
            "class_counts": class_counts.tolist(),
            "dictionary_fit": {
                "M": {
                    "objective": m_dict.objective_,
                    "n_iter": m_dict.n_iter_,
                    "init_index": m_dict.init_index_,
                    "empty_reseeds": m_dict.empty_reseeds_,
                    "converged": m_dict.converged_,
                    "occupancy": m_diag,
                },
                "C": {
                    "objective": c_dict.objective_,
                    "n_iter": c_dict.n_iter_,
                    "init_index": c_dict.init_index_,
                    "empty_reseeds": c_dict.empty_reseeds_,
                    "converged": c_dict.converged_,
                    "occupancy": c_diag,
                },
            },
            "train_diagnostics_only": {
                arm: {
                    "accuracy": state.train_accuracy,
                    "macro_f1": state.train_macro_f1,
                    "n_iter": state.n_iter.tolist(),
                    "converged": state.converged,
                }
                for arm, state in probes.items()
            },
            "public_test_accessed": False,
            "private_test_accessed": False,
        }
        summary_path = out_dir / "crs_train_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _log(f"Train-only CRS stage complete: {summary_path}")
        _log("PublicTest remains locked; this runner has no PublicTest input.")
        return summary
    finally:
        if work_dir.exists() and not keep_work:
            shutil.rmtree(work_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registered Train-only P/M/C CRS fitting stage. PublicTest is inaccessible."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--dictionary-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-work", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_train_stage(
        train_csv=args.train_csv,
        dictionary_npz=args.dictionary_npz,
        output_dir=args.output_dir,
        keep_work=bool(args.keep_work),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
