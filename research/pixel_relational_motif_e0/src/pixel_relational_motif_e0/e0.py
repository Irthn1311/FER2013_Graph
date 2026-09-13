from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .diag_gmm import DiagonalGaussianMixture
from .stability import (
    benjamini_hochberg,
    effective_support,
    effective_support_null,
    empirical_upper_p,
    hungarian_match,
    jaccard_indices,
    medoid_run,
    pairwise_symmetric_kl,
    support_matched_null_statistic,
)


@dataclass
class StabilityResult:
    models: list[DiagonalGaussianMixture]
    medoid_index: int
    observed_jaccard: np.ndarray
    p_values: np.ndarray
    q_values: np.ndarray
    stable_mask: np.ndarray
    match_distances: np.ndarray


@dataclass
class CanonicalDictionary:
    model: DiagonalGaussianMixture
    stable_components: np.ndarray
    medoid_to_canonical: np.ndarray
    canonical_distances: np.ndarray


def _image_bootstrap_weights(pool_image_ids: np.ndarray, fit_image_ids: np.ndarray, *, seed: int) -> np.ndarray:
    pool_ids = np.asarray(pool_image_ids, dtype=np.int64).reshape(-1)
    images = np.asarray(fit_image_ids, dtype=np.int64).reshape(-1)
    if len(pool_ids) == 0 or len(images) == 0:
        raise ValueError("bootstrap needs non-empty pool and image IDs")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(images, size=len(images), replace=True)
    unique, counts = np.unique(sampled, return_counts=True)
    multiplicity = dict(zip(unique.tolist(), counts.tolist()))
    return np.fromiter((multiplicity.get(int(i), 0) for i in pool_ids), dtype=np.float64, count=len(pool_ids))


def bootstrap_stability(pool_r: np.ndarray, pool_image_ids: np.ndarray, fit_image_ids: np.ndarray, anchor_r: np.ndarray, *, k: int, variance_floor: np.ndarray, n_bootstrap: int = 20, n_null: int = 2000, q: float = 0.05, seed: int = 42, gmm_max_iter: int = 100, gmm_tol: float = 1e-3, gmm_batch_size: int = 16_384) -> StabilityResult:
    """Run the locked image-bootstrap motif stability analysis in frozen R-space."""
    x = np.asarray(pool_r, dtype=np.float64)
    pool_ids = np.asarray(pool_image_ids, dtype=np.int64)
    anchor = np.asarray(anchor_r, dtype=np.float64)
    if len(x) != len(pool_ids):
        raise ValueError("pool descriptors and image IDs must align")
    models: list[DiagonalGaussianMixture] = []
    for b in range(n_bootstrap):
        bseed = int(np.random.SeedSequence([seed, 101, b]).generate_state(1)[0])
        sw = _image_bootstrap_weights(pool_ids, fit_image_ids, seed=bseed)
        model = DiagonalGaussianMixture(k, variance_floor, max_iter=gmm_max_iter, tol=gmm_tol, n_init=1, random_state=bseed, batch_size=gmm_batch_size).fit(x, sample_weight=sw)
        models.append(model)
    means_runs = [m.means_ for m in models]
    vars_runs = [m.variances_ for m in models]
    medoid = medoid_run(means_runs, vars_runs)
    ref = models[medoid]
    anchor_labels = [m.predict(anchor) for m in models]
    maps: list[np.ndarray] = []
    distances = np.empty((k, n_bootstrap), dtype=np.float64)
    for b, model in enumerate(models):
        cost = pairwise_symmetric_kl(ref.means_, ref.variances_, model.means_, model.variances_)
        rows, cols, vals = hungarian_match(cost)
        mapping = np.empty(k, dtype=np.int64)
        mapping[rows] = cols
        maps.append(mapping)
        distances[rows, b] = vals
    obs = np.empty(k, dtype=np.float64)
    pvals = np.empty(k, dtype=np.float64)
    universe = len(anchor)
    ref_labels = anchor_labels[medoid]
    for comp in range(k):
        a = np.flatnonzero(ref_labels == comp)
        js = []
        sizes = []
        for b in range(n_bootstrap):
            bb = np.flatnonzero(anchor_labels[b] == maps[b][comp])
            js.append(jaccard_indices(a, bb))
            sizes.append(len(bb))
        obs[comp] = float(np.median(js))
        null_seed = int(np.random.SeedSequence([seed, 202, comp]).generate_state(1)[0])
        null = support_matched_null_statistic(a, np.asarray(sizes), universe_size=universe, n_replicates=n_null, seed=null_seed)
        pvals[comp] = empirical_upper_p(obs[comp], null)
    stable, qvals = benjamini_hochberg(pvals, q=q)
    return StabilityResult(models, medoid, obs, pvals, qvals, stable, distances)


def nondegenerate_mask(counts_by_component_image: np.ndarray, *, percentile: float = 5.0, n_null: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image-level recurrence filter using effective number of supporting images."""
    counts = np.asarray(counts_by_component_image, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (K,n_images)")
    k, n_images = counts.shape
    obs = np.empty(k)
    thresholds = np.empty(k)
    mask = np.zeros(k, dtype=bool)
    for comp in range(k):
        obs[comp] = effective_support(counts[comp])
        null_seed = int(np.random.SeedSequence([seed, 303, comp]).generate_state(1)[0])
        null = effective_support_null(total_occurrences=int(counts[comp].sum()), n_images=n_images, n_replicates=n_null, seed=null_seed)
        thresholds[comp] = float(np.percentile(null, percentile))
        mask[comp] = obs[comp] > thresholds[comp]
    return mask, obs, thresholds


def fit_canonical_dictionary(pool_r: np.ndarray, *, variance_floor: np.ndarray, stability: StabilityResult, nondegenerate: np.ndarray, n_init: int = 5, seed: int = 42, gmm_max_iter: int = 100, gmm_tol: float = 1e-3, gmm_batch_size: int = 16_384) -> CanonicalDictionary:
    """Fit the separate canonical GMM and map every validated medoid motif into it.

    Canonical matching is a relabeling/provenance step, not an additional motif
    selection test. E0.1 motif validity is defined by the registered BH-stability
    and image-level non-degeneracy criteria; canonical match distances are kept
    for diagnostics but do not silently discard an otherwise validated motif.
    """
    ref = stability.models[stability.medoid_index]
    k = ref.n_components
    canonical = DiagonalGaussianMixture(k, variance_floor, max_iter=gmm_max_iter, tol=gmm_tol, n_init=n_init, random_state=seed, batch_size=gmm_batch_size).fit(pool_r)
    cost = pairwise_symmetric_kl(ref.means_, ref.variances_, canonical.means_, canonical.variances_)
    rows, cols, vals = hungarian_match(cost)
    mapping = np.empty(k, dtype=np.int64)
    cd = np.empty(k, dtype=np.float64)
    mapping[rows] = cols
    cd[rows] = vals
    candidate = stability.stable_mask & np.asarray(nondegenerate, dtype=bool)
    keep = np.asarray(sorted(mapping[np.flatnonzero(candidate)].tolist()), dtype=np.int64)
    return CanonicalDictionary(canonical, keep, mapping, cd)
