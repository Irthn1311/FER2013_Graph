from __future__ import annotations

import numpy as np


def stratified_log_sigma_sample(
    s: np.ndarray,
    log_sigma: np.ndarray,
    *,
    per_decile: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically sample equally from ten log-sigma quantile bins.

    Returns sampled S, sampled log_sigma, and source row indices. If a bin has
    fewer than ``per_decile`` points, all points are retained and no synthetic
    duplication is performed.
    """
    x = np.asarray(s)
    l = np.asarray(log_sigma).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 24 or len(x) != len(l):
        raise ValueError("expected S=(n,24) and matching log_sigma")
    if per_decile <= 0:
        raise ValueError("per_decile must be positive")
    edges = np.quantile(l, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(edges[1:-1], l, side="right")
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for b in range(10):
        idx = np.flatnonzero(bins == b)
        if len(idx) > per_decile:
            idx = np.sort(rng.choice(idx, size=per_decile, replace=False))
        chosen.append(idx)
    selected = np.concatenate(chosen)
    return x[selected], l[selected], selected
