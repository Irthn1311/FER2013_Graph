from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .descriptor import DescriptorTransform, EPS_GRAY, VALID_LOCATIONS, extract_raw_relations
from .diag_gmm import DiagonalGaussianMixture, variance_floor_from_data
from .e0 import StabilityResult, fit_canonical_dictionary, nondegenerate_mask
from .partition import ImagePartition, make_image_partition
from .stability import (
    benjamini_hochberg,
    empirical_upper_p,
    hungarian_match,
    medoid_run,
    pairwise_symmetric_kl,
)

BASE_SHA = "0cf76985572c388db00cfbb5be37fd5e83249be0"
ISSUE_NUMBER = 74
TRAIN_ROWS = 28_709
IMAGE_SIZE = 48
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE
K_CANDIDATES = (32, 64, 96, 128)
POOL_PER_DECILE = 50_000
BOOTSTRAPS = 20
NULL_REPLICATES = 2_000
BH_Q = 0.05


@dataclass(frozen=True)
class TrainData:
    images_uint8: np.ndarray
    sha256: str


@dataclass(frozen=True)
class DescriptorPool:
    s: np.ndarray
    log_sigma: np.ndarray
    image_ids: np.ndarray
    decile_edges: np.ndarray
    observed_bin_counts: np.ndarray


@dataclass(frozen=True)
class E01Config:
    seed: int = 42
    fit_fraction: float = 0.70
    heldout_fraction: float = 0.15
    k_candidates: tuple[int, ...] = K_CANDIDATES
    pool_per_decile: int = POOL_PER_DECILE
    bootstraps: int = BOOTSTRAPS
    null_replicates: int = NULL_REPLICATES
    bh_q: float = BH_Q
    variance_floor_fraction: float = 0.01
    selection_n_init: int = 5
    canonical_n_init: int = 5
    gmm_max_iter: int = 100
    gmm_tol: float = 1e-3
    gmm_batch_size: int = 16_384
    sampling_chunk_images: int = 128

    def validate_locked(self) -> None:
        if self.seed != 42:
            raise ValueError("E0.1 locked runner requires seed=42")
        if tuple(self.k_candidates) != K_CANDIDATES:
            raise ValueError(f"E0.1 K candidates are locked to {K_CANDIDATES}")
        if self.pool_per_decile != POOL_PER_DECILE:
            raise ValueError("E0.1 pool is locked to 50,000 descriptors per log-sigma decile")
        if self.bootstraps != BOOTSTRAPS:
            raise ValueError("E0.1 bootstrap count is locked to B=20")
        if self.null_replicates != NULL_REPLICATES:
            raise ValueError("E0.1 null count is locked to 2,000")
        if abs(self.bh_q - BH_Q) > 1e-15:
            raise ValueError("E0.1 BH threshold is locked to q=0.05")


def _log(message: str) -> None:
    print(f"[PGM-E0.1] {message}", flush=True)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _reject_private_path(path: Path) -> None:
    text = str(path).lower().replace("-", "_").replace(" ", "_")
    compact = text.replace("_", "")
    if "privatetest" in compact or "private_test" in text or "final_test" in text:
        raise ValueError("E0.1 must not read FER2013 PrivateTest/final-test data")


def load_official_train_images(csv_path: str | Path) -> TrainData:
    """Load only official FER2013 Train into a compact uint8 array.

    Labels are validated but deliberately not returned or used by E0.1. Exact
    Train row count is mandatory, so PublicTest/PrivateTest-sized files fail
    closed even if a caller supplies a misleading filename.
    """
    path = Path(csv_path)
    _reject_private_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    images = np.empty((TRAIN_ROWS, PIXELS_PER_IMAGE), dtype=np.uint8)
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("FER2013 Train CSV is empty") from exc
        header = [c.strip().lower() for c in raw_header]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError("FER2013 CSV requires emotion and pixels columns")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")
        for row in reader:
            if not row:
                continue
            if row_count >= TRAIN_ROWS:
                raise ValueError(f"Train row count exceeds locked {TRAIN_ROWS}")
            label = int(row[emo_idx])
            if not 0 <= label <= 6:
                raise ValueError(f"emotion out of range at row {row_count}: {label}")
            pixels = np.fromstring(row[pix_idx], sep=" ", dtype=np.int16)
            if pixels.size != PIXELS_PER_IMAGE:
                raise ValueError(
                    f"expected {PIXELS_PER_IMAGE} pixels at row {row_count}, got {pixels.size}"
                )
            if np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(f"pixel outside [0,255] at row {row_count}")
            images[row_count] = pixels.astype(np.uint8, copy=False)
            row_count += 1
    if row_count != TRAIN_ROWS:
        raise ValueError(f"official Train requires {TRAIN_ROWS} rows, observed {row_count}")
    return TrainData(images.reshape(TRAIN_ROWS, IMAGE_SIZE, IMAGE_SIZE), digest)


def _image01(images_uint8: np.ndarray, image_id: int) -> np.ndarray:
    return np.asarray(images_uint8[int(image_id)], dtype=np.float64) / 255.0


def _log_sigma_only(image01: np.ndarray) -> np.ndarray:
    x = np.asarray(image01, dtype=np.float64)
    if x.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("expected 48x48 image")
    windows = sliding_window_view(x, (5, 5)).reshape(VALID_LOCATIONS, 25)
    sigma = windows.std(axis=1, ddof=0)
    out = np.log(sigma + EPS_GRAY)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite log-sigma")
    return out


def compute_exact_decile_edges(images_uint8: np.ndarray, fit_ids: np.ndarray) -> np.ndarray:
    """Compute empirical Train_fit log-sigma decile edges without storing S."""
    ids = np.asarray(fit_ids, dtype=np.int64).reshape(-1)
    values = np.empty(len(ids) * VALID_LOCATIONS, dtype=np.float32)
    pos = 0
    for image_id in ids:
        l = _log_sigma_only(_image01(images_uint8, int(image_id)))
        values[pos : pos + VALID_LOCATIONS] = l.astype(np.float32)
        pos += VALID_LOCATIONS
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11)).astype(np.float64)
    if np.any(np.diff(edges) < 0):
        raise AssertionError("decile edges must be monotone")
    return edges


def _merge_priority_reservoir(
    *,
    capacity: int,
    current_s: np.ndarray,
    current_l: np.ndarray,
    current_ids: np.ndarray,
    current_keys: np.ndarray,
    incoming_s: list[np.ndarray],
    incoming_l: list[np.ndarray],
    incoming_ids: list[np.ndarray],
    incoming_keys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if incoming_s:
        ns = np.concatenate(incoming_s, axis=0)
        nl = np.concatenate(incoming_l, axis=0)
        ni = np.concatenate(incoming_ids, axis=0)
        nk = np.concatenate(incoming_keys, axis=0)
        s = np.concatenate([current_s, ns], axis=0)
        l = np.concatenate([current_l, nl], axis=0)
        ids = np.concatenate([current_ids, ni], axis=0)
        keys = np.concatenate([current_keys, nk], axis=0)
    else:
        s, l, ids, keys = current_s, current_l, current_ids, current_keys
    if len(keys) > capacity:
        take = np.argpartition(keys, capacity - 1)[:capacity]
        take = take[np.argsort(keys[take], kind="mergesort")]
        s, l, ids, keys = s[take], l[take], ids[take], keys[take]
    return s, l, ids, keys


def sample_fixed_stratified_pool(
    images_uint8: np.ndarray,
    fit_ids: np.ndarray,
    decile_edges: np.ndarray,
    *,
    per_decile: int = POOL_PER_DECILE,
    seed: int = 42,
    chunk_images: int = 128,
) -> DescriptorPool:
    """Uniform-without-replacement priority sample within each log-sigma decile.

    Independent deterministic U(0,1) priorities are assigned per descriptor;
    each bin keeps its `per_decile` smallest priorities. This is equivalent to
    a uniform sample without replacement while never materializing all S rows.
    """
    ids = np.asarray(fit_ids, dtype=np.int64).reshape(-1)
    edges = np.asarray(decile_edges, dtype=np.float64).reshape(-1)
    if edges.shape != (11,):
        raise ValueError("decile_edges must have length 11")
    if per_decile <= 0 or chunk_images <= 0:
        raise ValueError("positive per_decile and chunk_images required")

    rs = [np.empty((0, 24), dtype=np.float32) for _ in range(10)]
    rl = [np.empty((0,), dtype=np.float32) for _ in range(10)]
    ri = [np.empty((0,), dtype=np.int32) for _ in range(10)]
    rk = [np.empty((0,), dtype=np.float64) for _ in range(10)]
    observed = np.zeros(10, dtype=np.int64)

    bs: list[list[np.ndarray]] = [[] for _ in range(10)]
    bl: list[list[np.ndarray]] = [[] for _ in range(10)]
    bi: list[list[np.ndarray]] = [[] for _ in range(10)]
    bk: list[list[np.ndarray]] = [[] for _ in range(10)]

    def flush() -> None:
        nonlocal rs, rl, ri, rk, bs, bl, bi, bk
        for b in range(10):
            rs[b], rl[b], ri[b], rk[b] = _merge_priority_reservoir(
                capacity=per_decile,
                current_s=rs[b],
                current_l=rl[b],
                current_ids=ri[b],
                current_keys=rk[b],
                incoming_s=bs[b],
                incoming_l=bl[b],
                incoming_ids=bi[b],
                incoming_keys=bk[b],
            )
        bs = [[] for _ in range(10)]
        bl = [[] for _ in range(10)]
        bi = [[] for _ in range(10)]
        bk = [[] for _ in range(10)]

    for pos, image_id in enumerate(ids):
        s, l, _ = extract_raw_relations(_image01(images_uint8, int(image_id)))
        bins = np.searchsorted(edges[1:-1], l, side="right")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 401, int(image_id)]))
        keys = rng.random(len(l))
        for b in range(10):
            mask = bins == b
            n = int(mask.sum())
            observed[b] += n
            if n == 0:
                continue
            bs[b].append(s[mask].astype(np.float32))
            bl[b].append(l[mask].astype(np.float32))
            bi[b].append(np.full(n, int(image_id), dtype=np.int32))
            bk[b].append(keys[mask])
        if (pos + 1) % chunk_images == 0:
            flush()
    flush()

    sizes = np.asarray([len(x) for x in rs], dtype=np.int64)
    if np.any(sizes != per_decile):
        raise RuntimeError(
            f"locked pool requires exactly {per_decile} per decile; observed reservoir sizes {sizes.tolist()}"
        )
    return DescriptorPool(
        s=np.concatenate(rs, axis=0),
        log_sigma=np.concatenate(rl, axis=0),
        image_ids=np.concatenate(ri, axis=0),
        decile_edges=edges,
        observed_bin_counts=observed,
    )


def image_log_likelihoods(
    models: dict[int, DiagonalGaussianMixture],
    images_uint8: np.ndarray,
    image_ids: np.ndarray,
    transform: DescriptorTransform,
) -> dict[int, np.ndarray]:
    ids = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    out = {k: np.empty(len(ids), dtype=np.float64) for k in models}
    for i, image_id in enumerate(ids):
        s, l, _ = extract_raw_relations(_image01(images_uint8, int(image_id)))
        r = transform.transform(s, l)
        for k, model in models.items():
            out[k][i] = model.score(r)
    return out


def _fit_bootstrap_models(
    pool_r: np.ndarray,
    pool_image_ids: np.ndarray,
    fit_image_ids: np.ndarray,
    *,
    k: int,
    variance_floor: np.ndarray,
    config: E01Config,
) -> list[DiagonalGaussianMixture]:
    fit_ids = np.asarray(fit_image_ids, dtype=np.int64).reshape(-1)
    pool_ids = np.asarray(pool_image_ids, dtype=np.int64).reshape(-1)
    max_id = int(max(pool_ids.max(initial=0), fit_ids.max(initial=0)))
    models: list[DiagonalGaussianMixture] = []
    for b in range(config.bootstraps):
        bseed = int(np.random.SeedSequence([config.seed, 101, b]).generate_state(1)[0])
        rng = np.random.default_rng(bseed)
        sampled = rng.choice(fit_ids, size=len(fit_ids), replace=True)
        multiplicity = np.bincount(sampled, minlength=max_id + 1)
        sw = multiplicity[pool_ids].astype(np.float64)
        _log(f"K={k} bootstrap {b + 1}/{config.bootstraps}")
        model = DiagonalGaussianMixture(
            k,
            variance_floor,
            max_iter=config.gmm_max_iter,
            tol=config.gmm_tol,
            n_init=1,
            random_state=bseed,
            batch_size=config.gmm_batch_size,
        ).fit(pool_r, sample_weight=sw)
        models.append(model)
    return models


def _null_jaccard_from_sizes(
    reference_size: int,
    matched_sizes: np.ndarray,
    *,
    universe_size: int,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    if not 0 <= reference_size <= universe_size:
        raise ValueError("invalid reference size")
    sizes = np.asarray(matched_sizes, dtype=np.int64).reshape(-1)
    if np.any((sizes < 0) | (sizes > universe_size)):
        raise ValueError("invalid matched size")
    rng = np.random.default_rng(seed)
    out = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        js = np.empty(len(sizes), dtype=np.float64)
        for j, b in enumerate(sizes):
            inter = int(
                rng.hypergeometric(
                    ngood=int(reference_size),
                    nbad=int(universe_size - reference_size),
                    nsample=int(b),
                )
            )
            union = int(reference_size) + int(b) - inter
            js[j] = 1.0 if union == 0 else inter / union
        out[r] = float(np.median(js))
    return out


def stream_anchor_stability(
    models: list[DiagonalGaussianMixture],
    images_uint8: np.ndarray,
    anchor_ids: np.ndarray,
    transform: DescriptorTransform,
    *,
    n_null: int = NULL_REPLICATES,
    q: float = BH_Q,
    seed: int = 42,
) -> tuple[StabilityResult, np.ndarray]:
    """Exact anchor Jaccard accounting without storing anchor R or labels.

    For every bootstrap model, only component cardinalities and intersections
    with the medoid assignment are accumulated. Those sufficient statistics
    yield exactly the same per-component Jaccards as materializing all anchor
    descriptor index sets.
    """
    if not models:
        raise ValueError("models must be non-empty")
    k = models[0].n_components
    if any(m.n_components != k for m in models):
        raise ValueError("all bootstrap models must share K")
    means_runs = [m.means_ for m in models]
    vars_runs = [m.variances_ for m in models]
    medoid = medoid_run(means_runs, vars_runs)
    ref = models[medoid]
    bcount = len(models)

    maps: list[np.ndarray] = []
    distances = np.empty((k, bcount), dtype=np.float64)
    for b, model in enumerate(models):
        cost = pairwise_symmetric_kl(ref.means_, ref.variances_, model.means_, model.variances_)
        rows, cols, vals = hungarian_match(cost)
        mapping = np.empty(k, dtype=np.int64)
        mapping[rows] = cols
        maps.append(mapping)
        distances[rows, b] = vals

    ids = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
    counts_by_component_image = np.zeros((k, len(ids)), dtype=np.int32)
    ref_counts = np.zeros(k, dtype=np.int64)
    other_counts = np.zeros((k, bcount), dtype=np.int64)
    intersections = np.zeros((k, bcount), dtype=np.int64)
    universe = 0

    for image_pos, image_id in enumerate(ids):
        s, l, _ = extract_raw_relations(_image01(images_uint8, int(image_id)))
        r = transform.transform(s, l)
        ref_labels = ref.predict(r)
        bc_ref = np.bincount(ref_labels, minlength=k).astype(np.int64)
        counts_by_component_image[:, image_pos] = bc_ref.astype(np.int32)
        ref_counts += bc_ref
        universe += len(ref_labels)
        for b, model in enumerate(models):
            labels = model.predict(r)
            bc = np.bincount(labels, minlength=k).astype(np.int64)
            other_counts[:, b] += bc[maps[b]]
            agrees = labels == maps[b][ref_labels]
            intersections[:, b] += np.bincount(ref_labels[agrees], minlength=k)

    denom = ref_counts[:, None] + other_counts - intersections
    jaccards = np.ones((k, bcount), dtype=np.float64)
    nz = denom > 0
    jaccards[nz] = intersections[nz] / denom[nz]
    observed = np.median(jaccards, axis=1)
    p_values = np.empty(k, dtype=np.float64)
    for comp in range(k):
        nseed = int(np.random.SeedSequence([seed, 202, comp]).generate_state(1)[0])
        null = _null_jaccard_from_sizes(
            int(ref_counts[comp]),
            other_counts[comp],
            universe_size=universe,
            n_replicates=n_null,
            seed=nseed,
        )
        p_values[comp] = empirical_upper_p(float(observed[comp]), null)
    stable_mask, q_values = benjamini_hochberg(p_values, q=q)
    return (
        StabilityResult(
            models=models,
            medoid_index=medoid,
            observed_jaccard=observed,
            p_values=p_values,
            q_values=q_values,
            stable_mask=stable_mask,
            match_distances=distances,
        ),
        counts_by_component_image,
    )


def _within_one_se_candidates(ll_by_k: dict[int, np.ndarray]) -> tuple[list[int], dict[int, dict[str, float]]]:
    stats: dict[int, dict[str, float]] = {}
    for k, vals in ll_by_k.items():
        x = np.asarray(vals, dtype=np.float64)
        mean = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        stats[int(k)] = {"mean": mean, "se": se}
    best_k = max(stats, key=lambda kk: stats[kk]["mean"])
    cutoff = stats[best_k]["mean"] - stats[best_k]["se"]
    candidates = sorted(k for k, st in stats.items() if st["mean"] >= cutoff)
    for k in stats:
        stats[k]["within_one_se"] = bool(k in candidates)
    return candidates, stats


def _json_array(x: np.ndarray) -> list:
    return np.asarray(x).tolist()


def run_e01(train_csv: str | Path, output_dir: str | Path, *, config: E01Config = E01Config()) -> dict:
    """Run locked E0.1 on official Train only; PublicTest and PrivateTest are not read."""
    config.validate_locked()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("loading and validating official Train (labels validated, not used)")
    train = load_official_train_images(train_csv)
    partition: ImagePartition = make_image_partition(
        TRAIN_ROWS,
        fit_fraction=config.fit_fraction,
        heldout_fraction=config.heldout_fraction,
        seed=config.seed,
    )
    _log(
        f"image partition fit={len(partition.fit)} heldout={len(partition.heldout)} anchor={len(partition.anchor)}"
    )

    _log("computing exact Train_fit log-sigma deciles")
    edges = compute_exact_decile_edges(train.images_uint8, partition.fit)
    _log("sampling fixed 500k Train_fit descriptor pool")
    pool = sample_fixed_stratified_pool(
        train.images_uint8,
        partition.fit,
        edges,
        per_decile=config.pool_per_decile,
        seed=config.seed,
        chunk_images=config.sampling_chunk_images,
    )

    _log("fitting frozen PCA 24->11 and log-sigma z-score")
    transform = DescriptorTransform(n_components=11, random_state=config.seed).fit(pool.s, pool.log_sigma)
    pool_r = transform.transform(pool.s, pool.log_sigma)
    variance_floor = variance_floor_from_data(pool_r, fraction=config.variance_floor_fraction)

    _log("fitting K candidates on the same fixed pool")
    selection_models: dict[int, DiagonalGaussianMixture] = {}
    for k in config.k_candidates:
        _log(f"fit K={k}")
        selection_models[k] = DiagonalGaussianMixture(
            k,
            variance_floor,
            max_iter=config.gmm_max_iter,
            tol=config.gmm_tol,
            n_init=config.selection_n_init,
            random_state=config.seed,
            batch_size=config.gmm_batch_size,
        ).fit(pool_r)

    _log("scoring image-level Train_heldout log-likelihood for all K in one pass")
    heldout_ll = image_log_likelihoods(
        selection_models,
        train.images_uint8,
        partition.heldout,
        transform,
    )
    one_se, k_stats = _within_one_se_candidates(heldout_ll)
    _log(f"1-SE candidate K values: {one_se}")

    stability_by_k: dict[int, StabilityResult] = {}
    anchor_counts_by_k: dict[int, np.ndarray] = {}
    stable_fraction: dict[int, float] = {}
    for k in one_se:
        _log(f"running locked B={config.bootstraps} image bootstrap stability for K={k}")
        bootstrap_models = _fit_bootstrap_models(
            pool_r,
            pool.image_ids,
            partition.fit,
            k=k,
            variance_floor=variance_floor,
            config=config,
        )
        _log(f"streaming full Train_anchor stability sufficient statistics for K={k}")
        stability, counts = stream_anchor_stability(
            bootstrap_models,
            train.images_uint8,
            partition.anchor,
            transform,
            n_null=config.null_replicates,
            q=config.bh_q,
            seed=config.seed,
        )
        stability_by_k[k] = stability
        anchor_counts_by_k[k] = counts
        stable_fraction[k] = float(np.mean(stability.stable_mask))
        k_stats[k]["bh_stable_fraction"] = stable_fraction[k]

    selected_k = min(one_se, key=lambda k: (-stable_fraction[k], k))
    _log(f"selected K={selected_k} by 1-SE -> BH-stable fraction -> smaller K")
    selected_stability = stability_by_k[selected_k]
    counts = anchor_counts_by_k[selected_k]

    _log("testing image-level motif non-degeneracy")
    nondegenerate, neff, neff_threshold = nondegenerate_mask(
        counts,
        percentile=5.0,
        n_null=config.null_replicates,
        seed=config.seed,
    )
    _log("fitting separate canonical GMM on full fixed 500k pool")
    canonical = fit_canonical_dictionary(
        pool_r,
        variance_floor=variance_floor,
        stability=selected_stability,
        nondegenerate=nondegenerate,
        n_init=config.canonical_n_init,
        seed=config.seed,
        gmm_max_iter=config.gmm_max_iter,
        gmm_tol=config.gmm_tol,
        gmm_batch_size=config.gmm_batch_size,
    )

    pca = transform.pca
    assert pca is not None
    dictionary_path = out_dir / "e01_dictionary.npz"
    np.savez_compressed(
        dictionary_path,
        base_sha=np.asarray(BASE_SHA),
        issue=np.asarray(ISSUE_NUMBER, dtype=np.int32),
        seed=np.asarray(config.seed, dtype=np.int32),
        train_sha256=np.asarray(train.sha256),
        fit_ids=partition.fit.astype(np.int32),
        heldout_ids=partition.heldout.astype(np.int32),
        anchor_ids=partition.anchor.astype(np.int32),
        decile_edges=pool.decile_edges,
        pool_observed_bin_counts=pool.observed_bin_counts,
        pool_image_ids=pool.image_ids.astype(np.int32),
        pca_mean=pca.mean_.astype(np.float64),
        pca_components=pca.components_.astype(np.float64),
        pca_explained_variance=pca.explained_variance_.astype(np.float64),
        log_sigma_mean=np.asarray(transform.log_sigma_mean, dtype=np.float64),
        log_sigma_std=np.asarray(transform.log_sigma_std, dtype=np.float64),
        variance_floor=variance_floor.astype(np.float64),
        selected_k=np.asarray(selected_k, dtype=np.int32),
        canonical_weights=canonical.model.weights_.astype(np.float64),
        canonical_means=canonical.model.means_.astype(np.float64),
        canonical_variances=canonical.model.variances_.astype(np.float64),
        canonical_stable_components=canonical.stable_components.astype(np.int32),
        medoid_to_canonical=canonical.medoid_to_canonical.astype(np.int32),
        canonical_distances=canonical.canonical_distances.astype(np.float64),
        medoid_stable_mask=selected_stability.stable_mask.astype(np.bool_),
        medoid_jaccard=selected_stability.observed_jaccard.astype(np.float64),
        medoid_p_values=selected_stability.p_values.astype(np.float64),
        medoid_q_values=selected_stability.q_values.astype(np.float64),
        neff=neff.astype(np.float64),
        neff_null_p05=neff_threshold.astype(np.float64),
        nondegenerate_mask=nondegenerate.astype(np.bool_),
    )

    summary = {
        "experiment": "PGM_E0.1_motif_existence",
        "issue": ISSUE_NUMBER,
        "base_sha": BASE_SHA,
        "private_test_read": False,
        "public_test_read": False,
        "train": {"rows": TRAIN_ROWS, "sha256": train.sha256},
        "partition": {
            "fit_images": int(len(partition.fit)),
            "heldout_images": int(len(partition.heldout)),
            "anchor_images": int(len(partition.anchor)),
            "seed": config.seed,
        },
        "descriptor_pool": {
            "size": int(len(pool.s)),
            "per_decile": config.pool_per_decile,
            "decile_edges": _json_array(pool.decile_edges),
            "observed_bin_counts": _json_array(pool.observed_bin_counts),
        },
        "pca": {
            "input_dim": 24,
            "output_dim": 11,
            "whiten": False,
            "explained_variance_ratio": _json_array(pca.explained_variance_ratio_),
            "log_sigma_mean": float(transform.log_sigma_mean),
            "log_sigma_std": float(transform.log_sigma_std),
        },
        "gmm": {
            "covariance": "diagonal",
            "variance_floor_fraction": config.variance_floor_fraction,
            "variance_floor": _json_array(variance_floor),
            "k_candidates": list(config.k_candidates),
            "selection_n_init": config.selection_n_init,
            "canonical_n_init": config.canonical_n_init,
            "max_iter": config.gmm_max_iter,
            "tol": config.gmm_tol,
            "batch_size": config.gmm_batch_size,
        },
        "k_selection": {
            "stats": {str(k): v for k, v in sorted(k_stats.items())},
            "one_se_candidates": one_se,
            "selected_k": selected_k,
        },
        "stability": {
            "bootstrap_count": config.bootstraps,
            "null_replicates": config.null_replicates,
            "bh_q": config.bh_q,
            "selected_k_bh_stable_fraction": float(stable_fraction[selected_k]),
            "observed_jaccard": _json_array(selected_stability.observed_jaccard),
            "p_values": _json_array(selected_stability.p_values),
            "q_values": _json_array(selected_stability.q_values),
            "stable_mask": _json_array(selected_stability.stable_mask),
        },
        "nondegeneracy": {
            "effective_support": _json_array(neff),
            "matched_null_p05": _json_array(neff_threshold),
            "nondegenerate_mask": _json_array(nondegenerate),
        },
        "canonical_dictionary": {
            "stable_components": _json_array(canonical.stable_components),
            "stable_component_count": int(len(canonical.stable_components)),
            "medoid_to_canonical": _json_array(canonical.medoid_to_canonical),
            "canonical_distances": _json_array(canonical.canonical_distances),
            "artifact": dictionary_path.name,
        },
    }
    summary_path = out_dir / "e01_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"E0.1 complete; wrote {summary_path} and {dictionary_path}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked PGM E0.1 motif-existence runner. Reads official Train only."
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_e01(args.train_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
