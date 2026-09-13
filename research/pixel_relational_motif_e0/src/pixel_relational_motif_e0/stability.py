from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def symmetric_kl_diag(mu_a: np.ndarray, var_a: np.ndarray, mu_b: np.ndarray, var_b: np.ndarray) -> float:
    a, va, b, vb = map(lambda z: np.asarray(z, dtype=np.float64), (mu_a, var_a, mu_b, var_b))
    if a.shape != b.shape or va.shape != vb.shape or a.shape != va.shape:
        raise ValueError("Gaussian parameter shapes must agree")
    if np.any(va <= 0) or np.any(vb <= 0):
        raise ValueError("variances must be positive")
    diff2 = np.square(a - b)
    kl_ab = 0.5 * np.sum(np.log(vb / va) + (va + diff2) / vb - 1.0)
    kl_ba = 0.5 * np.sum(np.log(va / vb) + (vb + diff2) / va - 1.0)
    return float(0.5 * (kl_ab + kl_ba))


def pairwise_symmetric_kl(means_a: np.ndarray, vars_a: np.ndarray, means_b: np.ndarray, vars_b: np.ndarray) -> np.ndarray:
    ma, va, mb, vb = map(np.asarray, (means_a, vars_a, means_b, vars_b))
    out = np.empty((len(ma), len(mb)), dtype=np.float64)
    for i in range(len(ma)):
        for j in range(len(mb)):
            out[i, j] = symmetric_kl_diag(ma[i], va[i], mb[j], vb[j])
    return out


def hungarian_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(cost, dtype=np.float64)
    rows, cols = linear_sum_assignment(c)
    return rows, cols, c[rows, cols]


def medoid_run(means_runs: list[np.ndarray], vars_runs: list[np.ndarray]) -> int:
    if len(means_runs) != len(vars_runs) or not means_runs:
        raise ValueError("need matching non-empty run lists")
    n = len(means_runs)
    totals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cost = pairwise_symmetric_kl(means_runs[i], vars_runs[i], means_runs[j], vars_runs[j])
            _, _, vals = hungarian_match(cost)
            d = float(vals.mean())
            totals[i] += d
            totals[j] += d
    return int(np.argmin(totals))


def jaccard_indices(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.unique(np.asarray(a, dtype=np.int64))
    bb = np.unique(np.asarray(b, dtype=np.int64))
    inter = np.intersect1d(aa, bb, assume_unique=True).size
    union = len(aa) + len(bb) - inter
    return 1.0 if union == 0 else float(inter / union)


def support_matched_null_statistic(reference_indices: np.ndarray, matched_sizes: np.ndarray, *, universe_size: int, n_replicates: int = 2000, seed: int = 42) -> np.ndarray:
    """Exact-cardinality motif-specific null for median bootstrap Jaccard.

    Under the null, intersection size between fixed A (size a) and a uniformly
    random B (size b) follows Hypergeometric(N, a, b). Sampling that count is
    exactly equivalent to drawing full random sets but avoids huge allocations.
    """
    a_idx = np.unique(np.asarray(reference_indices, dtype=np.int64))
    sizes = np.asarray(matched_sizes, dtype=np.int64).reshape(-1)
    if np.any(sizes < 0) or np.any(sizes > universe_size):
        raise ValueError("matched cardinalities must lie in [0, universe_size]")
    if np.any(a_idx < 0) or np.any(a_idx >= universe_size):
        raise ValueError("reference indices outside anchor universe")
    a = len(a_idx)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        js = np.empty(len(sizes), dtype=np.float64)
        for i, b in enumerate(sizes):
            inter = int(rng.hypergeometric(ngood=a, nbad=universe_size - a, nsample=int(b)))
            union = a + int(b) - inter
            js[i] = 1.0 if union == 0 else inter / union
        stats[r] = float(np.median(js))
    return stats


def empirical_upper_p(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def benjamini_hochberg(p_values: np.ndarray, *, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if np.any((p < 0) | (p > 1)) or not (0 < q < 1):
        raise ValueError("invalid p-values or q")
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_ranked
    reject = adjusted <= q
    return reject, adjusted


def effective_support(counts_by_image: np.ndarray) -> float:
    c = np.asarray(counts_by_image, dtype=np.float64).reshape(-1)
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    total = c.sum()
    if total <= 0:
        return 0.0
    q = c / total
    return float(1.0 / np.square(q).sum())


def effective_support_null(*, total_occurrences: int, n_images: int, n_replicates: int = 2000, seed: int = 42) -> np.ndarray:
    if total_occurrences < 0 or n_images <= 0:
        raise ValueError("invalid count or n_images")
    if total_occurrences == 0:
        return np.zeros(n_replicates)
    rng = np.random.default_rng(seed)
    p = np.full(n_images, 1.0 / n_images)
    out = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        out[r] = effective_support(rng.multinomial(total_occurrences, p))
    return out


@dataclass(frozen=True)
class CanonicalMatch:
    source_component: int
    canonical_component: int
    distance: float


def match_to_canonical(source_means: np.ndarray, source_vars: np.ndarray, canonical_means: np.ndarray, canonical_vars: np.ndarray) -> list[CanonicalMatch]:
    cost = pairwise_symmetric_kl(source_means, source_vars, canonical_means, canonical_vars)
    rows, cols, vals = hungarian_match(cost)
    return [CanonicalMatch(int(r), int(c), float(v)) for r, c, v in zip(rows, cols, vals)]
