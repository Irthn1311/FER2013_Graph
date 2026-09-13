from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .controls import destroy_ordered_relations_keyed
from .descriptor import DescriptorTransform, VALID_LOCATIONS, extract_raw_relations
from .diag_gmm import DiagonalGaussianMixture, variance_floor_from_data
from .e01_occurrence_runner import DictionaryArtifact, load_dictionary_artifact
from .e01_runner import TRAIN_ROWS
from .features import dense_posterior_histogram
from .probe import ProbeSpec, fit_probe, metrics, paired_bootstrap_vs_control_mean


ISSUE_NUMBER = 78
BASE_SHA = "0bd9b3ff9f66f3e14c8faca07b6624424d1894ac"
E01_V533_SCIENTIFIC_SHA = "5fda000413c4dcfe810917f07e37bcafb1c8d394"
E01_V533_DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
TRAIN_SHA256 = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
PUBLIC_SHA256 = "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"
PUBLIC_ROWS = 3_589
PIXELS_PER_IMAGE = 48 * 48
CONTROL_SEEDS = (42, 43, 44, 45, 46)
K = 128
POOL_PER_DECILE = 50_000
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
GMM_N_INIT = 5
GMM_MAX_ITER = 100
GMM_TOL = 1e-3
GMM_BATCH_SIZE = 16_384
VARIANCE_FLOOR_FRACTION = 0.01
PROBE_SPEC = ProbeSpec(c=1.0, max_iter=2000, tol=1e-4, seed=42)


@dataclass(frozen=True)
class PixelData:
    images_uint8: np.ndarray
    canonical_ids: np.ndarray
    sha256: str
    role: str


@dataclass(frozen=True)
class MatchedPool:
    s: np.ndarray
    log_sigma: np.ndarray
    image_ids: np.ndarray
    pixel_indices: np.ndarray
    identity_sha256: str


@dataclass(frozen=True)
class ActualDictionary:
    artifact: DictionaryArtifact
    fit_ids: np.ndarray
    decile_edges: np.ndarray
    pool_image_ids: np.ndarray
    sha256: str


@dataclass(frozen=True)
class ControlDictionary:
    seed: int
    transform: DescriptorTransform
    model: DiagonalGaussianMixture
    variance_floor: np.ndarray


def _log(message: str) -> None:
    print(f"[PGM-E0.2] {message}", flush=True)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    x = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(x.dtype.str.encode("ascii"))
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii"))
    h.update(x.view(np.uint8))
    return h.hexdigest()


def bind_e02_split(role: str, path: str | Path) -> Path:
    """Bind only registered Train/Public paths and reject test aliases."""
    normalized_role = str(role).strip().lower().replace("-", "_").replace(" ", "_")
    p = Path(path)
    path_text = str(p).lower().replace("-", "_").replace(" ", "_")
    compact = path_text.replace("_", "")
    forbidden_role = normalized_role in {"test", "private_test", "privatetest", "final_test"}
    forbidden_path = (
        "privatetest" in compact
        or "private_test" in path_text
        or "final_test" in path_text
        or p.name.lower() == "test.csv"
    )
    if forbidden_role or forbidden_path:
        raise ValueError("E0.2 must not open, hash, count, inspect, or infer PrivateTest/test.csv")
    if normalized_role not in {"train", "public"}:
        raise ValueError("E0.2 split role must be exactly 'train' or 'public'")
    return p


def _role_contract(role: str) -> tuple[int, int, str]:
    if role == "train":
        return TRAIN_ROWS, 0, TRAIN_SHA256
    if role == "public":
        return PUBLIC_ROWS, TRAIN_ROWS, PUBLIC_SHA256
    raise ValueError("unregistered E0.2 role")


def load_pixels_only(csv_path: str | Path, *, role: str) -> PixelData:
    """Load pixels without locating or parsing the emotion column."""
    path = bind_e02_split(role, csv_path)
    expected_rows, offset, expected_sha = _role_contract(role)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != expected_sha:
        raise ValueError(f"{role} SHA256 mismatch: expected {expected_sha}, got {digest}")
    images = np.empty((expected_rows, PIXELS_PER_IMAGE), dtype=np.uint8)
    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [c.strip().lower() for c in next(reader)]
        except StopIteration as exc:
            raise ValueError("FER CSV is empty") from exc
        if "pixels" not in header:
            raise ValueError("FER CSV requires pixels column")
        pixel_column = header.index("pixels")
        for row in reader:
            if not row:
                continue
            if count >= expected_rows:
                raise ValueError(f"{role} row count exceeds locked {expected_rows}")
            pixels = np.fromstring(row[pixel_column], sep=" ", dtype=np.int16)
            if pixels.size != PIXELS_PER_IMAGE or np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(f"invalid FER pixels at {role} row {count}")
            images[count] = pixels.astype(np.uint8, copy=False)
            count += 1
    if count != expected_rows:
        raise ValueError(f"{role} requires {expected_rows} rows, observed {count}")
    return PixelData(
        images.reshape(expected_rows, 48, 48),
        np.arange(offset, offset + expected_rows, dtype=np.int32),
        digest,
        role,
    )


def load_labels_downstream(csv_path: str | Path, *, role: str, expected_sha256: str) -> np.ndarray:
    """Read labels only after unsupervised dictionaries/features are frozen."""
    path = bind_e02_split(role, csv_path)
    expected_rows, _, registered_sha = _role_contract(role)
    digest = sha256_file(path)
    if digest != expected_sha256 or digest != registered_sha:
        raise ValueError(f"{role} changed between pixel and downstream-label stages")
    labels = np.empty(expected_rows, dtype=np.int8)
    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [c.strip().lower() for c in next(reader)]
        except StopIteration as exc:
            raise ValueError("FER CSV is empty") from exc
        if "emotion" not in header:
            raise ValueError("downstream supervised stage requires emotion column")
        emotion_column = header.index("emotion")
        for row in reader:
            if not row:
                continue
            if count >= expected_rows:
                raise ValueError(f"{role} label row count exceeds locked {expected_rows}")
            label = int(row[emotion_column])
            if not 0 <= label <= 6:
                raise ValueError(f"emotion out of range at {role} row {count}: {label}")
            labels[count] = label
            count += 1
    if count != expected_rows:
        raise ValueError(f"{role} requires {expected_rows} labels, observed {count}")
    return labels


def load_actual_dictionary(path: str | Path) -> ActualDictionary:
    p = Path(path)
    digest = sha256_file(p)
    if digest != E01_V533_DICTIONARY_SHA256:
        raise ValueError(
            f"E0.2 actual is locked to v533 dictionary {E01_V533_DICTIONARY_SHA256}, got {digest}"
        )
    artifact = load_dictionary_artifact(p)
    if artifact.selected_k != K or artifact.model.n_components != K:
        raise ValueError(f"E0.2 requires frozen actual K={K}")
    with np.load(p, allow_pickle=False) as z:
        fit_ids = np.asarray(z["fit_ids"], dtype=np.int32)
        edges = np.asarray(z["decile_edges"], dtype=np.float64)
        pool_image_ids = np.asarray(z["pool_image_ids"], dtype=np.int32)
        source_train_sha = str(np.asarray(z["train_sha256"]).reshape(()).item())
    if source_train_sha != TRAIN_SHA256:
        raise ValueError("v533 dictionary Train provenance mismatch")
    if edges.shape != (11,) or pool_image_ids.shape != (10 * POOL_PER_DECILE,):
        raise ValueError("v533 matched-pool metadata shape mismatch")
    return ActualDictionary(artifact, fit_ids, edges, pool_image_ids, digest)


def _image01(images_uint8: np.ndarray, image_id: int) -> np.ndarray:
    return np.asarray(images_uint8[int(image_id)], dtype=np.float64) / 255.0


def reconstruct_matched_pool(
    images_uint8: np.ndarray,
    dictionary: ActualDictionary,
    *,
    chunk_images: int = 128,
) -> MatchedPool:
    """Recreate v533's exact priority sample and retain physical pixel IDs."""
    if chunk_images <= 0:
        raise ValueError("chunk_images must be positive")
    capacity = POOL_PER_DECILE
    rs = [np.empty((0, 24), np.float32) for _ in range(10)]
    rl = [np.empty(0, np.float32) for _ in range(10)]
    ri = [np.empty(0, np.int32) for _ in range(10)]
    rp = [np.empty(0, np.int16) for _ in range(10)]
    rk = [np.empty(0, np.float64) for _ in range(10)]
    buffers: list[list[list[np.ndarray]]] = [[[] for _ in range(10)] for _ in range(5)]

    def flush() -> None:
        nonlocal buffers
        for b in range(10):
            parts = []
            current = (rs[b], rl[b], ri[b], rp[b], rk[b])
            for current_value, incoming in zip(current, (buffers[j][b] for j in range(5))):
                parts.append(np.concatenate([current_value, *incoming], axis=0) if incoming else current_value)
            if len(parts[-1]) > capacity:
                take = np.argpartition(parts[-1], capacity - 1)[:capacity]
                take = take[np.argsort(parts[-1][take], kind="mergesort")]
                parts = [value[take] for value in parts]
            rs[b], rl[b], ri[b], rp[b], rk[b] = parts
        buffers = [[[] for _ in range(10)] for _ in range(5)]

    for position, image_id in enumerate(dictionary.fit_ids):
        s, log_sigma, _ = extract_raw_relations(_image01(images_uint8, int(image_id)))
        bins = np.searchsorted(dictionary.decile_edges[1:-1], log_sigma, side="right")
        rng = np.random.default_rng(np.random.SeedSequence([42, 401, int(image_id)]))
        priorities = rng.random(len(log_sigma))
        for b in range(10):
            selected = np.flatnonzero(bins == b)
            if len(selected) == 0:
                continue
            buffers[0][b].append(s[selected].astype(np.float32))
            buffers[1][b].append(log_sigma[selected].astype(np.float32))
            buffers[2][b].append(np.full(len(selected), int(image_id), np.int32))
            buffers[3][b].append(selected.astype(np.int16))
            buffers[4][b].append(priorities[selected])
        if (position + 1) % chunk_images == 0:
            flush()
    flush()
    if any(len(value) != capacity for value in rs):
        raise RuntimeError("could not recreate the locked 50k-per-decile pool")
    pool = MatchedPool(
        np.concatenate(rs),
        np.concatenate(rl),
        np.concatenate(ri),
        np.concatenate(rp),
        "",
    )
    if not np.array_equal(pool.image_ids, dictionary.pool_image_ids):
        raise RuntimeError("recreated physical sample does not match v533 pool_image_ids")
    identities = np.stack([pool.image_ids, pool.pixel_indices.astype(np.int32)], axis=1)
    return MatchedPool(
        pool.s,
        pool.log_sigma,
        pool.image_ids,
        pool.pixel_indices,
        sha256_array(identities),
    )


def fit_control_dictionary(pool: MatchedPool, *, control_seed: int) -> ControlDictionary:
    """Fit one label-inaccessible PCA/log-scale/GMM control from scratch."""
    if control_seed not in CONTROL_SEEDS:
        raise ValueError(f"control_seed must be one of {CONTROL_SEEDS}")
    destroyed = destroy_ordered_relations_keyed(
        pool.s,
        pool.image_ids,
        pool.pixel_indices,
        control_seed=control_seed,
        chunk_size=GMM_BATCH_SIZE,
    )
    transform = DescriptorTransform(n_components=11, random_state=42).fit(
        destroyed, pool.log_sigma
    )
    latent = transform.transform(destroyed, pool.log_sigma)
    floor = variance_floor_from_data(latent, fraction=VARIANCE_FLOOR_FRACTION)
    model = DiagonalGaussianMixture(
        K,
        floor,
        max_iter=GMM_MAX_ITER,
        tol=GMM_TOL,
        n_init=GMM_N_INIT,
        random_state=42,
        batch_size=GMM_BATCH_SIZE,
    ).fit(latent)
    return ControlDictionary(control_seed, transform, model, floor)


def dense_features(
    data: PixelData,
    transform,
    model: DiagonalGaussianMixture,
    *,
    control_seed: int | None,
    progress_every: int = 500,
) -> np.ndarray:
    if model.n_components != K:
        raise ValueError(f"all E0.2 conditions require K={K}")
    out = np.empty((len(data.canonical_ids), K), dtype=np.float32)
    pixel_indices = np.arange(VALID_LOCATIONS, dtype=np.int16)
    for position, canonical_id in enumerate(data.canonical_ids):
        s, log_sigma, _ = extract_raw_relations(_image01(data.images_uint8, position))
        if control_seed is not None:
            s = destroy_ordered_relations_keyed(
                s,
                np.full(VALID_LOCATIONS, int(canonical_id), dtype=np.int32),
                pixel_indices,
                control_seed=control_seed,
            )
        latent = transform.transform(s, log_sigma)
        out[position] = dense_posterior_histogram(model.predict_proba(latent)).astype(np.float32)
        if progress_every and ((position + 1) % progress_every == 0 or position + 1 == len(out)):
            label = "actual" if control_seed is None else f"control-{control_seed}"
            _log(f"{label} {data.role} dense histograms {position + 1}/{len(out)}")
    if out.shape[1] != K or not np.all(np.isfinite(out)) or np.any(out < 0):
        raise RuntimeError("invalid dense posterior histograms")
    if not np.allclose(out.sum(axis=1), 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError("dense posterior histograms do not sum to one")
    return out


def save_control_artifact(
    path: str | Path,
    control: ControlDictionary,
    *,
    train_sha256: str,
    source_dictionary_sha256: str,
    pool_identity_sha256: str,
) -> dict[str, object]:
    pca = control.transform.pca
    if pca is None or control.model.weights_ is None:
        raise RuntimeError("cannot save unfitted control")
    path = Path(path)
    np.savez_compressed(
        path,
        experiment=np.asarray("PGM_E0.2_local_relational_necessity"),
        issue=np.asarray(ISSUE_NUMBER, np.int32),
        base_sha=np.asarray(BASE_SHA),
        source_dictionary_sha256=np.asarray(source_dictionary_sha256),
        train_sha256=np.asarray(train_sha256),
        pool_identity_sha256=np.asarray(pool_identity_sha256),
        control_seed=np.asarray(control.seed, np.int32),
        selected_k=np.asarray(K, np.int32),
        pca_mean=pca.mean_.astype(np.float64),
        pca_components=pca.components_.astype(np.float64),
        pca_explained_variance=pca.explained_variance_.astype(np.float64),
        log_sigma_mean=np.asarray(control.transform.log_sigma_mean, np.float64),
        log_sigma_std=np.asarray(control.transform.log_sigma_std, np.float64),
        variance_floor=control.variance_floor.astype(np.float64),
        gmm_weights=control.model.weights_.astype(np.float64),
        gmm_means=control.model.means_.astype(np.float64),
        gmm_variances=control.model.variances_.astype(np.float64),
        gmm_lower_bound=np.asarray(control.model.lower_bound_, np.float64),
        gmm_n_iter=np.asarray(control.model.n_iter_, np.int32),
    )
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _condition_probe(x_train, y_train, x_public, *, name: str) -> tuple[np.ndarray, dict]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = fit_probe(x_train, y_train, spec=PROBE_SPEC)
    prediction = np.asarray(probe.predict(x_public), dtype=np.int8)
    classifier = probe.named_steps["logisticregression"]
    diagnostics = {
        "condition": name,
        "n_iter": np.asarray(classifier.n_iter_, dtype=np.int64).tolist(),
        "converged": bool(np.all(np.asarray(classifier.n_iter_) < PROBE_SPEC.max_iter)),
        "warnings": [f"{type(item.message).__name__}: {item.message}" for item in caught],
    }
    return prediction, diagnostics


def _metric_record(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    accuracy, macro_f1 = metrics(y, prediction)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def run_e02(
    train_csv: str | Path,
    public_csv: str | Path,
    dictionary_npz: str | Path,
    output_dir: str | Path,
) -> dict:
    """Execute the preregistered E0.2 Train-supervised/Public-dev screen."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = load_actual_dictionary(dictionary_npz)
    _log(f"frozen actual dictionary SHA256 PASS: {dictionary.sha256}")

    _log("loading registered Train/Public pixels label-blind")
    train = load_pixels_only(train_csv, role="train")
    public = load_pixels_only(public_csv, role="public")
    _log("reconstructing exact v533 500k physical descriptor sample")
    pool = reconstruct_matched_pool(train.images_uint8, dictionary)
    _log(f"matched pool identity SHA256: {pool.identity_sha256}")

    train_features: dict[str, np.ndarray] = {}
    public_features: dict[str, np.ndarray] = {}
    feature_hashes: dict[str, dict[str, str]] = {}
    control_artifacts: dict[str, dict[str, object]] = {}
    control_fit_diagnostics: dict[str, dict[str, object]] = {}

    _log("computing frozen actual A dense K=128 posterior histograms")
    train_features["actual"] = dense_features(
        train, dictionary.artifact.transform, dictionary.artifact.model, control_seed=None
    )
    public_features["actual"] = dense_features(
        public, dictionary.artifact.transform, dictionary.artifact.model, control_seed=None
    )
    feature_hashes["actual"] = {
        "train": sha256_array(train_features["actual"]),
        "public": sha256_array(public_features["actual"]),
    }

    for seed in CONTROL_SEEDS:
        name = f"control_{seed}"
        _log(f"fitting independent matched relation-destroyed control seed={seed}")
        control = fit_control_dictionary(pool, control_seed=seed)
        control_path = out_dir / f"e02_control_seed{seed}.npz"
        control_artifacts[str(seed)] = save_control_artifact(
            control_path,
            control,
            train_sha256=train.sha256,
            source_dictionary_sha256=dictionary.sha256,
            pool_identity_sha256=pool.identity_sha256,
        )
        control_fit_diagnostics[str(seed)] = {
            "gmm_n_iter": int(control.model.n_iter_),
            "gmm_lower_bound": float(control.model.lower_bound_),
            "pca_fitted_independently": True,
            "gmm_fitted_independently": True,
        }
        train_features[name] = dense_features(
            train, control.transform, control.model, control_seed=seed
        )
        public_features[name] = dense_features(
            public, control.transform, control.model, control_seed=seed
        )
        feature_hashes[name] = {
            "train": sha256_array(train_features[name]),
            "public": sha256_array(public_features[name]),
        }
        del control

    # This is intentionally the first point at which either label column is read.
    _log("unsupervised fitting/features frozen; now reading Train/Public labels downstream")
    train_labels = load_labels_downstream(train_csv, role="train", expected_sha256=train.sha256)
    public_labels = load_labels_downstream(public_csv, role="public", expected_sha256=public.sha256)

    predictions: dict[str, np.ndarray] = {}
    probe_diagnostics: dict[str, dict] = {}
    condition_order = ["actual", *[f"control_{seed}" for seed in CONTROL_SEEDS]]
    for name in condition_order:
        _log(f"fitting fixed saga probe for {name}")
        predictions[name], probe_diagnostics[name] = _condition_probe(
            train_features[name], train_labels, public_features[name], name=name
        )

    actual_metrics = _metric_record(public_labels, predictions["actual"])
    control_predictions = np.stack(
        [predictions[f"control_{seed}"] for seed in CONTROL_SEEDS], axis=0
    )
    control_metrics = {
        str(seed): _metric_record(public_labels, predictions[f"control_{seed}"])
        for seed in CONTROL_SEEDS
    }
    control_accuracy = np.asarray([control_metrics[str(s)]["accuracy"] for s in CONTROL_SEEDS])
    control_macro_f1 = np.asarray([control_metrics[str(s)]["macro_f1"] for s in CONTROL_SEEDS])
    observed_delta = {
        "accuracy": float(actual_metrics["accuracy"] - control_accuracy.mean()),
        "macro_f1": float(actual_metrics["macro_f1"] - control_macro_f1.mean()),
    }
    bootstrap = paired_bootstrap_vs_control_mean(
        public_labels,
        predictions["actual"],
        control_predictions,
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    accuracy_delta = np.asarray(bootstrap["delta_accuracy"], dtype=np.float64)
    macro_f1_delta = np.asarray(bootstrap["delta_macro_f1"], dtype=np.float64)

    results_path = out_dir / "e02_results.npz"
    np.savez_compressed(
        results_path,
        experiment=np.asarray("PGM_E0.2_local_relational_necessity"),
        issue=np.asarray(ISSUE_NUMBER, np.int32),
        base_sha=np.asarray(BASE_SHA),
        actual_dictionary_sha256=np.asarray(dictionary.sha256),
        train_sha256=np.asarray(train.sha256),
        public_sha256=np.asarray(public.sha256),
        train_ids=train.canonical_ids,
        public_ids=public.canonical_ids,
        train_labels=train_labels,
        public_labels=public_labels,
        public_label_sha256=np.asarray(sha256_array(public_labels)),
        control_seeds=np.asarray(CONTROL_SEEDS, np.int32),
        actual_public_predictions=predictions["actual"],
        control_public_predictions=control_predictions,
        bootstrap_delta_accuracy=accuracy_delta,
        bootstrap_delta_macro_f1=macro_f1_delta,
    )
    results_artifact = {
        "file": results_path.name,
        "sha256": sha256_file(results_path),
        "bytes": results_path.stat().st_size,
    }

    summary = {
        "experiment": "PGM_E0.2_local_relational_necessity",
        "issue": ISSUE_NUMBER,
        "base_sha": BASE_SHA,
        "immutable_e01_verdict": "E0.1 NEGATIVE — MOTIF EXISTENCE NOT SUPPORTED",
        "e01b_overwritten": False,
        "formal_verdict": "E0.2 SCIENTIFIC VERDICT — REQUIRES INDEPENDENT REVIEW",
        "private_test_read": False,
        "data": {
            "train": {
                "rows": TRAIN_ROWS,
                "sha256": train.sha256,
                "label_order_sha256": sha256_array(train_labels),
            },
            "public": {
                "rows": PUBLIC_ROWS,
                "sha256": public.sha256,
                "label_order_sha256": sha256_array(public_labels),
            },
            "public_labels_stage": "downstream_probe_evaluation_only",
        },
        "actual_dictionary": {
            "scientific_sha": E01_V533_SCIENTIFIC_SHA,
            "sha256": dictionary.sha256,
            "k": K,
            "refit": False,
        },
        "control": {
            "seeds": list(CONTROL_SEEDS),
            "key": "SplitMix64(control_seed, global_canonical_image_id, valid_center_pixel_index, coordinate)",
            "pool_size": int(len(pool.s)),
            "pool_identity_sha256": pool.identity_sha256,
            "per_descriptor_multiset_preserved": True,
            "log_sigma_untouched": True,
            "pca_refit_per_seed": True,
            "gmm_refit_per_seed": True,
            "gmm_hyperparameters": {
                "k": K,
                "covariance": "diagonal",
                "n_init": GMM_N_INIT,
                "max_iter": GMM_MAX_ITER,
                "tol": GMM_TOL,
                "batch_size": GMM_BATCH_SIZE,
                "random_state": 42,
                "variance_floor_fraction": VARIANCE_FLOOR_FRACTION,
            },
            "fit_diagnostics": control_fit_diagnostics,
            "artifacts": control_artifacts,
        },
        "representation": {
            "type": "mean dense posterior histogram",
            "dimensions": K,
            "descriptors_per_image": VALID_LOCATIONS,
            "uses_all_components": True,
            "feature_sha256": feature_hashes,
            "same_train_public_ids_all_conditions": True,
            "same_labels_all_conditions": True,
        },
        "probe": {
            "scaler": "StandardScaler(with_mean=False)",
            "classifier": "L2 LogisticRegression(solver=saga)",
            "C": PROBE_SPEC.c,
            "max_iter": PROBE_SPEC.max_iter,
            "tol": PROBE_SPEC.tol,
            "random_state": PROBE_SPEC.seed,
            "identical_capacity_all_conditions": True,
            "diagnostics": probe_diagnostics,
        },
        "metrics": {
            "actual": actual_metrics,
            "controls": control_metrics,
            "control_mean": {
                "accuracy": float(control_accuracy.mean()),
                "macro_f1": float(control_macro_f1.mean()),
            },
            "control_sample_sd": {
                "accuracy": float(control_accuracy.std(ddof=1)),
                "macro_f1": float(control_macro_f1.std(ddof=1)),
            },
            "actual_minus_control_mean": observed_delta,
        },
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "accuracy": {
                "observed_delta": observed_delta["accuracy"],
                "quantiles_2.5_50_97.5": np.percentile(accuracy_delta, [2.5, 50, 97.5]).tolist(),
                "ci95": np.asarray(bootstrap["accuracy_ci95"]).tolist(),
                "fraction_le_zero": float(np.mean(accuracy_delta <= 0)),
            },
            "macro_f1": {
                "observed_delta": observed_delta["macro_f1"],
                "quantiles_2.5_50_97.5": np.percentile(macro_f1_delta, [2.5, 50, 97.5]).tolist(),
                "ci95": np.asarray(bootstrap["macro_f1_ci95"]).tolist(),
                "fraction_le_zero": float(np.mean(macro_f1_delta <= 0)),
            },
        },
        "results_artifact": results_artifact,
    }
    summary_path = out_dir / "e02_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _log(f"E0.2 complete: {summary_path}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preregistered PGM E0.2 runner")
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--public-csv", required=True, type=Path)
    parser.add_argument("--dictionary-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_e02(args.train_csv, args.public_csv, args.dictionary_npz, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
