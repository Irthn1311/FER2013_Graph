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
from sklearn.metrics import accuracy_score, f1_score

from .crs_runtime import load_frozen_primitive_substrate, primitive_map_from_uint8
from .crs_stage import (
    C_DIM,
    CRS_DESCRIPTOR_DIM,
    CRS_K,
    CRS_SIDE,
    CRS_VALID_POSITIONS,
    DICTIONARY_SAMPLES_PER_IMAGE,
    E01_V533_DICTIONARY_SHA256,
    M_DIM,
    MASTER_SEED,
    P_DIM,
    composition_descriptors,
    crs_image_feature,
    descriptors_to_csr,
    p_image_feature,
    permute_cell_blocks,
)
from .e01_runner import TRAIN_ROWS
from .e02_runner import (
    PUBLIC_ROWS,
    PUBLIC_SHA256,
    TRAIN_SHA256,
    PixelData,
    load_labels_downstream,
    load_pixels_only,
    sha256_array,
    sha256_file,
)


ISSUE_NUMBER = 84
DRAFT_PR_NUMBER = 85
TRAIN_MODEL_SHA256 = "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"
TRAIN_SCIENTIFIC_SHA = "f2ff796ce01a92b1d4a010113d0c5160f1dc2c3d"
TECHNICAL_AMENDMENT = "A1"

SKM_N_INIT = 3
SKM_MAX_ITER = 500
SKM_TOL = 1e-6
LOGREG_C = 1.0
LOGREG_SOLVER = "lbfgs"
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_MAX_ITER = 5000
LOGREG_TOL = 1e-4

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_GENERATOR = "numpy.random.Generator(PCG64)"
BOOTSTRAP_CI_METHOD = "linear"

PREDICTIONS_FILENAME = "crs_public_predictions.npz"
SUMMARY_FILENAME = "crs_public_summary.json"
MANIFEST_FILENAME = "crs_public_execution_manifest.json"
PUBLIC_SOURCE_SHA_ENV = "CRS_PUBLIC_SOURCE_SHA"


@dataclass(frozen=True)
class FrozenProbe:
    classes: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    n_iter: np.ndarray


@dataclass(frozen=True)
class FrozenTrainModel:
    artifact_sha256: str
    m_centers: np.ndarray
    c_centers: np.ndarray
    probes: dict[str, FrozenProbe]


def _log(message: str) -> None:
    print(f"[PGM-CRS-PUBLIC] {message}", flush=True)


def _environment_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _scalar(array: np.ndarray, *, field: str):
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"Train model field {field} must be scalar")
    return value.item()


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def load_frozen_train_model(path: str | Path) -> FrozenTrainModel:
    """Load the one accepted A1 Train artifact and reject all drift."""
    artifact_path = Path(path)
    digest = sha256_file(artifact_path)
    if digest != TRAIN_MODEL_SHA256:
        raise ValueError(
            f"CRS Public runner requires Train model SHA256 {TRAIN_MODEL_SHA256}, "
            f"got {digest}"
        )

    required = {
        "technical_amendment",
        "master_seed",
        "dictionary_samples_per_image",
        "crs_k",
        "crs_descriptor_dim",
        "skm_n_init",
        "skm_max_iter",
        "skm_tol",
        "source_dictionary_sha256",
        "train_sha256",
        "m_crs_centers",
        "c_crs_centers",
        "logreg_c",
        "logreg_solver",
        "logreg_class_weight",
        "logreg_max_iter",
        "logreg_tol",
    }
    for arm in ("p", "m", "c"):
        required.update(
            {
                f"{arm}_classes",
                f"{arm}_coef",
                f"{arm}_intercept",
                f"{arm}_n_iter",
            }
        )

    with np.load(artifact_path, allow_pickle=False) as model:
        missing = sorted(required.difference(model.files))
        if missing:
            raise ValueError(f"Train model missing required fields: {missing}")

        expected_scalars = {
            "technical_amendment": TECHNICAL_AMENDMENT,
            "master_seed": MASTER_SEED,
            "dictionary_samples_per_image": DICTIONARY_SAMPLES_PER_IMAGE,
            "crs_k": CRS_K,
            "crs_descriptor_dim": CRS_DESCRIPTOR_DIM,
            "skm_n_init": SKM_N_INIT,
            "skm_max_iter": SKM_MAX_ITER,
            "skm_tol": SKM_TOL,
            "source_dictionary_sha256": E01_V533_DICTIONARY_SHA256,
            "train_sha256": TRAIN_SHA256,
            "logreg_c": LOGREG_C,
            "logreg_solver": LOGREG_SOLVER,
            "logreg_class_weight": LOGREG_CLASS_WEIGHT,
            "logreg_max_iter": LOGREG_MAX_ITER,
            "logreg_tol": LOGREG_TOL,
        }
        for field, expected in expected_scalars.items():
            observed = _scalar(model[field], field=field)
            if observed != expected:
                raise ValueError(
                    f"Train model {field} mismatch: expected {expected!r}, "
                    f"got {observed!r}"
                )

        centers: dict[str, np.ndarray] = {}
        for arm in ("m", "c"):
            value = _require_finite(
                f"{arm.upper()} centers", model[f"{arm}_crs_centers"]
            ).astype(np.float32, copy=True)
            if value.shape != (CRS_K, CRS_DESCRIPTOR_DIM):
                raise ValueError(f"{arm.upper()} center shape mismatch: {value.shape}")
            norms = np.linalg.norm(value, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{arm.upper()} centers are not unit normalized")
            centers[arm] = value

        dimensions = {"p": P_DIM, "m": M_DIM, "c": C_DIM}
        probes: dict[str, FrozenProbe] = {}
        expected_classes = np.arange(7, dtype=np.int64)
        for arm, dimension in dimensions.items():
            classes = np.asarray(model[f"{arm}_classes"], dtype=np.int64).copy()
            coef = _require_finite(
                f"{arm.upper()} probe coefficients", model[f"{arm}_coef"]
            ).astype(np.float64, copy=True)
            intercept = _require_finite(
                f"{arm.upper()} probe intercept", model[f"{arm}_intercept"]
            ).astype(np.float64, copy=True)
            n_iter = _require_finite(
                f"{arm.upper()} probe iterations", model[f"{arm}_n_iter"]
            ).astype(np.int64, copy=True)
            if not np.array_equal(classes, expected_classes):
                raise ValueError(f"{arm.upper()} probe class order mismatch")
            if coef.shape != (7, dimension) or intercept.shape != (7,):
                raise ValueError(f"{arm.upper()} probe parameter shape mismatch")
            if n_iter.shape != (1,) or np.any(n_iter <= 0) or np.any(n_iter >= LOGREG_MAX_ITER):
                raise ValueError(f"{arm.upper()} probe convergence provenance invalid")
            probes[arm.upper()] = FrozenProbe(classes, coef, intercept, n_iter)

    return FrozenTrainModel(
        artifact_sha256=digest,
        m_centers=centers["m"],
        c_centers=centers["c"],
        probes=probes,
    )


def verify_frozen_dictionary(path: str | Path) -> str:
    digest = sha256_file(path)
    if digest != E01_V533_DICTIONARY_SHA256:
        raise ValueError(
            f"CRS Public runner requires v533 dictionary SHA256 "
            f"{E01_V533_DICTIONARY_SHA256}, got {digest}"
        )
    return digest


def validate_public_identity(data: PixelData) -> None:
    expected_ids = np.arange(TRAIN_ROWS, TRAIN_ROWS + PUBLIC_ROWS, dtype=np.int32)
    if data.role != "public":
        raise ValueError("CRS Public runner requires role='public'")
    if data.sha256 != PUBLIC_SHA256:
        raise ValueError("Public SHA256 contract mismatch")
    if data.images_uint8.shape != (PUBLIC_ROWS, 48, 48):
        raise ValueError("Public image shape/row contract mismatch")
    if data.images_uint8.dtype != np.uint8:
        raise ValueError("Public pixels must be uint8")
    if not np.array_equal(data.canonical_ids, expected_ids):
        raise ValueError("Public canonical-ID contract mismatch")


def exact_cosine_nearest_centers(
    descriptors: np.ndarray | sparse.spmatrix,
    centers: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Exact cosine argmax matching SphericalKMeansResult.predict semantics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    center_array = np.asarray(centers, dtype=np.float32)
    if center_array.ndim != 2 or not np.all(np.isfinite(center_array)):
        raise ValueError("centers must be a finite 2D array")
    center_norms = np.linalg.norm(center_array, axis=1)
    if not np.allclose(center_norms, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("centers must be unit normalized")

    if sparse.issparse(descriptors):
        data = sparse.csr_matrix(descriptors, dtype=np.float32, copy=True)
        norms = np.sqrt(np.asarray(data.multiply(data).sum(axis=1)).reshape(-1))
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise ValueError("descriptors must have finite positive norms")
        data = sparse.diags((1.0 / norms).astype(np.float32)) @ data
        data = sparse.csr_matrix(data, dtype=np.float32)
    else:
        data = np.asarray(descriptors, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("descriptors must be 2D")
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise ValueError("descriptors must have finite positive norms")
        data = data / norms
    if data.shape[1] != center_array.shape[1]:
        raise ValueError("descriptor/center dimension mismatch")

    labels = np.empty(data.shape[0], dtype=np.int32)
    for start in range(0, data.shape[0], batch_size):
        block = data[start : start + batch_size]
        similarities = np.asarray(block @ center_array.T)
        labels[start : start + block.shape[0]] = np.argmax(
            similarities, axis=1
        ).astype(np.int32)
    return labels


def frozen_probe_predict(
    probe: FrozenProbe, features: np.ndarray, *, batch_size: int = 512
) -> np.ndarray:
    data = np.asarray(features)
    if data.ndim != 2 or data.shape[1] != probe.coef.shape[1]:
        raise ValueError("frozen probe feature shape mismatch")
    if not np.all(np.isfinite(data)):
        raise ValueError("frozen probe features contain non-finite values")
    predictions = np.empty(data.shape[0], dtype=np.int8)
    for start in range(0, data.shape[0], batch_size):
        block = data[start : start + batch_size]
        logits = block @ probe.coef.T + probe.intercept[None, :]
        predictions[start : start + len(block)] = probe.classes[
            np.argmax(logits, axis=1)
        ].astype(np.int8)
    if np.any((predictions < 0) | (predictions > 6)):
        raise AssertionError("frozen probe prediction outside FER classes")
    return predictions


def _flush_memmap(path: Path, value: np.memmap) -> None:
    value.flush()
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def build_public_representations(
    data: PixelData,
    substrate,
    train_model: FrozenTrainModel,
    *,
    work_dir: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Build all label-free Public representations before any label access."""
    validate_public_identity(data)
    paths = {
        "P": work_dir / "public_P.npy",
        "M": work_dir / "public_M.npy",
        "C": work_dir / "public_C.npy",
    }
    dimensions = {"P": P_DIM, "M": M_DIM, "C": C_DIM}
    features = {
        arm: open_memmap(
            paths[arm], mode="w+", dtype=np.float32, shape=(PUBLIC_ROWS, dimension)
        )
        for arm, dimension in dimensions.items()
    }
    hashes = {arm: hashlib.sha256() for arm in dimensions}
    dense_indices = np.arange(CRS_VALID_POSITIONS, dtype=np.int32)

    for position, canonical_id in enumerate(data.canonical_ids):
        primitive = primitive_map_from_uint8(data.images_uint8[position], substrate)
        p_feature = p_image_feature(primitive)

        m_descriptors = composition_descriptors(primitive, normalize=True)
        if m_descriptors.shape != (CRS_VALID_POSITIONS, CRS_DESCRIPTOR_DIM):
            raise AssertionError("Public M descriptor contract changed")
        m_ids = exact_cosine_nearest_centers(
            descriptors_to_csr(m_descriptors), train_model.m_centers
        )
        if m_ids.shape != (CRS_VALID_POSITIONS,):
            raise AssertionError("Public M assignment count mismatch")
        m_feature = crs_image_feature(m_ids)

        c_descriptors = permute_cell_blocks(
            m_descriptors,
            split_id="public",
            image_id=int(canonical_id),
            center_indices=dense_indices,
            master_seed=MASTER_SEED,
        )
        c_ids = exact_cosine_nearest_centers(
            descriptors_to_csr(c_descriptors), train_model.c_centers
        )
        if c_ids.shape != (CRS_VALID_POSITIONS,):
            raise AssertionError("Public C assignment count mismatch")
        c_feature = crs_image_feature(c_ids)

        row_features = {"P": p_feature, "M": m_feature, "C": c_feature}
        for arm, feature in row_features.items():
            if feature.shape != (dimensions[arm],) or not np.all(np.isfinite(feature)):
                raise AssertionError(f"Public {arm} feature contract changed")
            if not np.isclose(np.linalg.norm(feature), 1.0, rtol=1e-5, atol=1e-5):
                raise AssertionError(f"Public {arm} feature is not unit normalized")
            features[arm][position] = feature
            hashes[arm].update(np.ascontiguousarray(feature).tobytes())

        if (position + 1) % 250 == 0 or position + 1 == PUBLIC_ROWS:
            _log(f"label-free representations {position + 1}/{PUBLIC_ROWS} images")

    for arm, value in features.items():
        _flush_memmap(paths[arm], value)
    del features
    return paths, {arm: value.hexdigest() for arm, value in hashes.items()}


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    for key, value in payload.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"non-finite artifact field {key}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze_predictions_before_labels(
    feature_paths: dict[str, Path],
    train_model: FrozenTrainModel,
    *,
    work_dir: Path,
) -> dict[str, np.ndarray]:
    """Persist and reload predictions before the downstream label loader runs."""
    predictions: dict[str, np.ndarray] = {}
    for arm, dimension in (("P", P_DIM), ("M", M_DIM), ("C", C_DIM)):
        features = np.load(feature_paths[arm], mmap_mode="r")
        if features.shape != (PUBLIC_ROWS, dimension):
            raise AssertionError(f"Public {arm} feature matrix shape mismatch")
        predictions[arm] = frozen_probe_predict(train_model.probes[arm], features)

    frozen_path = work_dir / "public_predictions_prelabel.npz"
    _atomic_savez(
        frozen_path,
        {f"pred_{arm}": value for arm, value in predictions.items()},
    )
    with np.load(frozen_path, allow_pickle=False) as frozen:
        restored = {
            arm: np.asarray(frozen[f"pred_{arm}"], dtype=np.int8).copy()
            for arm in ("P", "M", "C")
        }
    for arm in ("P", "M", "C"):
        if not np.array_equal(restored[arm], predictions[arm]):
            raise AssertionError(f"Public {arm} pre-label prediction freeze mismatch")
    return restored


def classification_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    labels = np.arange(7, dtype=np.int64)
    y = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if y.shape != (PUBLIC_ROWS,) or pred.shape != (PUBLIC_ROWS,):
        raise ValueError("Public metric input shape mismatch")
    per_class = f1_score(
        y, pred, labels=labels, average=None, zero_division=0
    ).astype(np.float64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y, pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "per_class_f1_diagnostic": per_class.tolist(),
    }


def make_bootstrap_indices() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(
        0,
        PUBLIC_ROWS,
        size=(BOOTSTRAP_REPLICATES, PUBLIC_ROWS),
        dtype=np.int32,
    )
    if indices.shape != (2_000, 3_589) or indices.dtype != np.int32:
        raise AssertionError("registered bootstrap index contract changed")
    return indices


def _metrics_from_confusion(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    confusion = np.bincount(y * 7 + pred, minlength=49).reshape(7, 7)
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0, dtype=np.int64).astype(np.float64) - tp
    fn = confusion.sum(axis=1, dtype=np.int64).astype(np.float64) - tp
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros(7, dtype=np.float64),
        where=denominator != 0,
    )
    accuracy = float(tp.sum() / len(y))
    return accuracy, float(per_class.mean())


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all four registered deltas from one shared index matrix."""
    y = np.asarray(y_true, dtype=np.int64)
    if y.shape != (PUBLIC_ROWS,):
        raise ValueError("registered Public label shape mismatch")
    if indices.shape != (BOOTSTRAP_REPLICATES, PUBLIC_ROWS) or indices.dtype != np.int32:
        raise ValueError("registered shared bootstrap matrix mismatch")
    for arm in ("P", "M", "C"):
        value = np.asarray(predictions[arm])
        if value.shape != (PUBLIC_ROWS,) or np.any((value < 0) | (value > 6)):
            raise ValueError(f"registered {arm} prediction contract mismatch")

    deltas = {
        "delta_acc_m_c": np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64),
        "delta_f1_m_c": np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64),
        "delta_acc_m_p": np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64),
        "delta_f1_m_p": np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64),
    }
    for replicate, sampled in enumerate(indices):
        y_sample = y[sampled]
        metrics = {
            arm: _metrics_from_confusion(y_sample, predictions[arm][sampled])
            for arm in ("P", "M", "C")
        }
        deltas["delta_acc_m_c"][replicate] = metrics["M"][0] - metrics["C"][0]
        deltas["delta_f1_m_c"][replicate] = metrics["M"][1] - metrics["C"][1]
        deltas["delta_acc_m_p"][replicate] = metrics["M"][0] - metrics["P"][0]
        deltas["delta_f1_m_p"][replicate] = metrics["M"][1] - metrics["P"][1]
    return deltas


def percentile_interval(distribution: np.ndarray) -> tuple[float, float]:
    value = np.asarray(distribution, dtype=np.float64)
    if value.shape != (BOOTSTRAP_REPLICATES,) or not np.all(np.isfinite(value)):
        raise ValueError("registered bootstrap delta distribution mismatch")
    lower, upper = np.quantile(
        value, [0.025, 0.975], method=BOOTSTRAP_CI_METHOD
    )
    return float(lower), float(upper)


def registered_verdict(gate_a_pass: bool, gate_b_pass: bool) -> str:
    table = {
        (False, False): "COMPOSITIONAL FORMULATION NOT SUPPORTED",
        (False, True): "UTILITY WITHOUT ARRANGEMENT EVIDENCE — NOT COMPOSITIONAL SUPPORT",
        (True, False): "ARRANGEMENT SUPPORTED — ABSTRACTION UTILITY NOT SUPPORTED",
        (True, True): "COMPOSITIONAL STAGE SUPPORTED",
    }
    return table[(bool(gate_a_pass), bool(gate_b_pass))]


def _source_sha_from_environment() -> str:
    source_sha = os.environ.get(PUBLIC_SOURCE_SHA_ENV, "")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise RuntimeError(
            f"{PUBLIC_SOURCE_SHA_ENV} must contain the exact 40-character Public source lock"
        )
    return source_sha


def _validate_prediction_artifact(path: Path, *, source_sha: str) -> None:
    required = {
        "experiment",
        "issue",
        "draft_pr",
        "source_sha",
        "public_sha256",
        "train_model_sha256",
        "dictionary_sha256",
        "canonical_public_ids",
        "y_true",
        "pred_P",
        "pred_M",
        "pred_C",
        "bootstrap_indices",
        "bootstrap_indices_sha256",
        "delta_acc_m_c",
        "delta_f1_m_c",
        "delta_acc_m_p",
        "delta_f1_m_p",
    }
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != required:
            raise ValueError("Public prediction artifact key set mismatch")
        if str(_scalar(artifact["source_sha"], field="source_sha")) != source_sha:
            raise ValueError("Public prediction artifact source SHA mismatch")
        if str(_scalar(artifact["public_sha256"], field="public_sha256")) != PUBLIC_SHA256:
            raise ValueError("Public prediction artifact input SHA mismatch")
        expected_ids = np.arange(TRAIN_ROWS, TRAIN_ROWS + PUBLIC_ROWS, dtype=np.int32)
        if not np.array_equal(artifact["canonical_public_ids"], expected_ids):
            raise ValueError("Public prediction artifact canonical IDs mismatch")
        for key in ("y_true", "pred_P", "pred_M", "pred_C"):
            value = np.asarray(artifact[key])
            if value.shape != (PUBLIC_ROWS,) or np.any((value < 0) | (value > 6)):
                raise ValueError(f"Public prediction artifact {key} mismatch")
        indices = np.asarray(artifact["bootstrap_indices"])
        if indices.shape != (BOOTSTRAP_REPLICATES, PUBLIC_ROWS) or indices.dtype != np.int32:
            raise ValueError("Public prediction artifact bootstrap matrix mismatch")
        if str(_scalar(artifact["bootstrap_indices_sha256"], field="bootstrap_indices_sha256")) != sha256_array(indices):
            raise ValueError("Public prediction artifact bootstrap hash mismatch")
        for key in (
            "delta_acc_m_c",
            "delta_f1_m_c",
            "delta_acc_m_p",
            "delta_f1_m_p",
        ):
            value = np.asarray(artifact[key])
            if value.shape != (BOOTSTRAP_REPLICATES,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Public prediction artifact {key} mismatch")


def run_public_stage(
    *,
    public_csv: str | Path,
    dictionary_npz: str | Path,
    train_model_npz: str | Path,
    output_dir: str | Path,
) -> dict:
    source_sha = _source_sha_from_environment()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_paths = [
        out / PREDICTIONS_FILENAME,
        out / SUMMARY_FILENAME,
        out / MANIFEST_FILENAME,
    ]
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite registered Public outputs: " + ", ".join(existing)
        )

    train_model = load_frozen_train_model(train_model_npz)
    dictionary_sha = verify_frozen_dictionary(dictionary_npz)
    substrate = load_frozen_primitive_substrate(dictionary_npz)
    public = load_pixels_only(public_csv, role="public")
    validate_public_identity(public)

    work_dir = Path(tempfile.mkdtemp(prefix="crs_public_", dir=out))
    _log(f"temporary work dir: {work_dir}")
    try:
        feature_paths, feature_hashes = build_public_representations(
            public, substrate, train_model, work_dir=work_dir
        )
        predictions = freeze_predictions_before_labels(
            feature_paths, train_model, work_dir=work_dir
        )

        # This is the first and only Public-label access. All representations
        # and all three prediction vectors have already been persisted/reloaded.
        y_true = load_labels_downstream(
            public_csv, role="public", expected_sha256=public.sha256
        ).astype(np.int8, copy=False)
        if y_true.shape != (PUBLIC_ROWS,) or np.any((y_true < 0) | (y_true > 6)):
            raise ValueError("registered Public label contract mismatch")

        point_metrics = {
            arm: classification_metrics(y_true, predictions[arm])
            for arm in ("P", "M", "C")
        }
        observed_deltas = {
            "delta_acc_m_c": point_metrics["M"]["accuracy"]
            - point_metrics["C"]["accuracy"],
            "delta_f1_m_c": point_metrics["M"]["macro_f1"]
            - point_metrics["C"]["macro_f1"],
            "delta_acc_m_p": point_metrics["M"]["accuracy"]
            - point_metrics["P"]["accuracy"],
            "delta_f1_m_p": point_metrics["M"]["macro_f1"]
            - point_metrics["P"]["macro_f1"],
        }

        bootstrap_indices = make_bootstrap_indices()
        bootstrap_deltas = paired_bootstrap_deltas(
            y_true, predictions, bootstrap_indices
        )
        intervals = {
            key: dict(zip(("lower95", "upper95"), percentile_interval(value)))
            for key, value in bootstrap_deltas.items()
        }
        gate_a_pass = bool(
            intervals["delta_acc_m_c"]["lower95"] > 0
            and intervals["delta_f1_m_c"]["lower95"] > 0
        )
        gate_b_pass = bool(
            intervals["delta_acc_m_p"]["lower95"] > 0
            and intervals["delta_f1_m_p"]["lower95"] > 0
        )
        verdict = registered_verdict(gate_a_pass, gate_b_pass)
        bootstrap_sha = sha256_array(bootstrap_indices)

        prediction_path = out / PREDICTIONS_FILENAME
        _atomic_savez(
            prediction_path,
            {
                "experiment": np.asarray("PGM_CRS_PUBLIC_REGISTERED_VALIDATION"),
                "issue": np.asarray(ISSUE_NUMBER, dtype=np.int32),
                "draft_pr": np.asarray(DRAFT_PR_NUMBER, dtype=np.int32),
                "source_sha": np.asarray(source_sha),
                "public_sha256": np.asarray(PUBLIC_SHA256),
                "train_model_sha256": np.asarray(TRAIN_MODEL_SHA256),
                "dictionary_sha256": np.asarray(E01_V533_DICTIONARY_SHA256),
                "canonical_public_ids": public.canonical_ids.astype(np.int32),
                "y_true": y_true.astype(np.int8),
                "pred_P": predictions["P"].astype(np.int8),
                "pred_M": predictions["M"].astype(np.int8),
                "pred_C": predictions["C"].astype(np.int8),
                "bootstrap_indices": bootstrap_indices,
                "bootstrap_indices_sha256": np.asarray(bootstrap_sha),
                **bootstrap_deltas,
            },
        )
        _validate_prediction_artifact(prediction_path, source_sha=source_sha)
        prediction_sha = sha256_file(prediction_path)

        summary = {
            "status": "CRS_PUBLIC_REGISTERED_VALIDATION_COMPLETE",
            "issue": ISSUE_NUMBER,
            "draft_pr": DRAFT_PR_NUMBER,
            "source_sha": source_sha,
            "train_scientific_sha": TRAIN_SCIENTIFIC_SHA,
            "public_sha256": public.sha256,
            "public_rows": PUBLIC_ROWS,
            "canonical_public_id_range": [TRAIN_ROWS, TRAIN_ROWS + PUBLIC_ROWS - 1],
            "train_model_sha256": train_model.artifact_sha256,
            "source_dictionary_sha256": dictionary_sha,
            "environment": _environment_manifest(),
            "representation_sha256": feature_hashes,
            "metrics": point_metrics,
            "observed_deltas": observed_deltas,
            "paired_bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "generator": BOOTSTRAP_GENERATOR,
                "matrix_shape": [BOOTSTRAP_REPLICATES, PUBLIC_ROWS],
                "shared_index_matrix": True,
                "index_sha256": bootstrap_sha,
                "ci": "percentile",
                "quantiles": [0.025, 0.975],
                "quantile_method": BOOTSTRAP_CI_METHOD,
                "intervals": intervals,
            },
            "gate_a": {"passed": gate_a_pass, "comparison": "M-C"},
            "gate_b": {"passed": gate_b_pass, "comparison": "M-P"},
            "registered_verdict": verdict,
            "prediction_artifact": PREDICTIONS_FILENAME,
            "prediction_artifact_sha256": prediction_sha,
            "private_test_accessed": False,
        }
        summary_path = out / SUMMARY_FILENAME
        _atomic_json(summary_path, summary)
        summary_sha = sha256_file(summary_path)

        manifest = {
            "issue": ISSUE_NUMBER,
            "draft_pr": DRAFT_PR_NUMBER,
            "source_sha": source_sha,
            "execution": {
                "account": os.environ.get("CRS_EXECUTION_ACCOUNT"),
                "kernel": os.environ.get("CRS_EXECUTION_KERNEL"),
                "version": os.environ.get("CRS_EXECUTION_VERSION"),
            },
            "environment": _environment_manifest(),
            "inputs": {
                "public_csv": {"sha256": public.sha256, "rows": PUBLIC_ROWS},
                "dictionary_npz": {"sha256": dictionary_sha},
                "train_model_npz": {"sha256": train_model.artifact_sha256},
            },
            "outputs": {
                PREDICTIONS_FILENAME: {
                    "sha256": prediction_sha,
                    "bytes": prediction_path.stat().st_size,
                },
                SUMMARY_FILENAME: {
                    "sha256": summary_sha,
                    "bytes": summary_path.stat().st_size,
                },
            },
            "private_test_accessed": False,
        }
        manifest_path = out / MANIFEST_FILENAME
        _atomic_json(manifest_path, manifest)
        _log(f"registered Public artifacts complete: {out}")
        _log("PrivateTest remains sealed and inaccessible.")
        return summary
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registered inference-only CRS PublicTest evaluation"
    )
    parser.add_argument("--public-csv", required=True)
    parser.add_argument("--dictionary-npz", required=True)
    parser.add_argument("--train-model-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_public_stage(
        public_csv=args.public_csv,
        dictionary_npz=args.dictionary_npz,
        train_model_npz=args.train_model_npz,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
