from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp


def variance_floor_from_data(x: np.ndarray, *, fraction: float = 0.01) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("x must be 2D")
    if not (0 < fraction < 1):
        raise ValueError("fraction must lie in (0,1)")
    var = data.var(axis=0, ddof=0)
    return np.maximum(fraction * var, 1e-8)


@dataclass
class DiagonalGaussianMixture:
    """Small deterministic diagonal-GMM EM with vector variance floors.

    E-steps are chunked, so the locked 500k x <=128 E0 fit need not materialize
    a full responsibility matrix in memory.
    """

    n_components: int
    variance_floor: np.ndarray | float
    max_iter: int = 100
    tol: float = 1e-3
    n_init: int = 1
    random_state: int = 42
    batch_size: int = 16_384
    min_component_weight: float = 1e-8

    weights_: np.ndarray | None = None
    means_: np.ndarray | None = None
    variances_: np.ndarray | None = None
    lower_bound_: float | None = None
    n_iter_: int | None = None

    def _floor(self, d: int) -> np.ndarray:
        f = np.asarray(self.variance_floor, dtype=np.float64)
        if f.ndim == 0:
            f = np.full(d, float(f))
        if f.shape != (d,) or np.any(f <= 0):
            raise ValueError(f"variance_floor must be positive scalar or shape ({d},)")
        return f

    @staticmethod
    def _log_prob(x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        log_det = np.log(vars_).sum(axis=1)
        maha = (np.square(x[:, None, :] - means[None, :, :]) / vars_[None, :, :]).sum(axis=2)
        return -0.5 * (d * math.log(2.0 * math.pi) + log_det[None, :] + maha)

    def _initialize(self, x: np.ndarray, rng: np.random.Generator, floor: np.ndarray, sample_weight: np.ndarray):
        n, d = x.shape
        if self.n_components > n:
            raise ValueError("n_components cannot exceed number of samples")
        prob = sample_weight / sample_weight.sum()
        positive = np.flatnonzero(prob > 0)
        if len(positive) < self.n_components:
            raise ValueError("fewer positive-weight samples than components")
        ids = rng.choice(n, size=self.n_components, replace=False, p=prob)
        means = x[ids].copy()
        global_var = np.maximum(x.var(axis=0, ddof=0), floor)
        vars_ = np.repeat(global_var[None, :], self.n_components, axis=0)
        weights = np.full(self.n_components, 1.0 / self.n_components)
        return weights, means, vars_

    def _em_once(self, x: np.ndarray, seed: int, floor: np.ndarray, sample_weight: np.ndarray):
        rng = np.random.default_rng(seed)
        weights, means, vars_ = self._initialize(x, rng, floor, sample_weight)
        prev = -np.inf
        n, d = x.shape
        total_weight = float(sample_weight.sum())
        for it in range(1, self.max_iter + 1):
            nk = np.zeros(self.n_components, dtype=np.float64)
            sx = np.zeros((self.n_components, d), dtype=np.float64)
            sx2 = np.zeros((self.n_components, d), dtype=np.float64)
            total_ll = 0.0
            for start in range(0, n, self.batch_size):
                xb = x[start : start + self.batch_size]
                lp = self._log_prob(xb, means, vars_) + np.log(np.maximum(weights, 1e-300))[None, :]
                norm = logsumexp(lp, axis=1)
                resp = np.exp(lp - norm[:, None])
                wb = sample_weight[start : start + len(xb)]
                wr = resp * wb[:, None]
                total_ll += float(np.dot(norm, wb))
                nk += wr.sum(axis=0)
                sx += wr.T @ xb
                sx2 += wr.T @ np.square(xb)
            dead = nk < max(self.min_component_weight * total_weight, 1e-6)
            safe_nk = np.maximum(nk, 1e-12)
            new_means = sx / safe_nk[:, None]
            second = sx2 / safe_nk[:, None]
            new_vars = np.maximum(second - np.square(new_means), floor[None, :])
            new_weights = safe_nk / safe_nk.sum()
            if np.any(dead):
                for k in np.flatnonzero(dead):
                    idx = int(rng.integers(0, n))
                    new_means[k] = x[idx]
                    new_vars[k] = np.maximum(x.var(axis=0, ddof=0), floor)
                    new_weights[k] = 1.0 / n
                new_weights /= new_weights.sum()
            mean_ll = total_ll / total_weight
            weights, means, vars_ = new_weights, new_means, new_vars
            if np.isfinite(prev) and abs(mean_ll - prev) <= self.tol * (1.0 + abs(prev)):
                return weights, means, vars_, mean_ll, it
            prev = mean_ll
        return weights, means, vars_, prev, self.max_iter

    def fit(self, x: np.ndarray, sample_weight: np.ndarray | None = None) -> "DiagonalGaussianMixture":
        data = np.asarray(x, dtype=np.float64)
        if data.ndim != 2 or len(data) == 0 or not np.all(np.isfinite(data)):
            raise ValueError("x must be a non-empty finite 2D array")
        if sample_weight is None:
            sw = np.ones(len(data), dtype=np.float64)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(sw) != len(data) or np.any(sw < 0) or not np.all(np.isfinite(sw)) or sw.sum() <= 0:
                raise ValueError("invalid sample_weight")
        floor = self._floor(data.shape[1])
        best = None
        for init in range(self.n_init):
            seed = int(np.random.SeedSequence([self.random_state, init]).generate_state(1)[0])
            state = self._em_once(data, seed, floor, sw)
            if best is None or state[3] > best[3]:
                best = state
        assert best is not None
        self.weights_, self.means_, self.variances_, self.lower_bound_, self.n_iter_ = best
        return self

    def _check(self):
        if self.weights_ is None or self.means_ is None or self.variances_ is None:
            raise RuntimeError("GMM is not fit")

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        self._check()
        data = np.asarray(x, dtype=np.float64)
        out = np.empty(len(data), dtype=np.float64)
        for start in range(0, len(data), self.batch_size):
            xb = data[start : start + self.batch_size]
            lp = self._log_prob(xb, self.means_, self.variances_) + np.log(self.weights_)[None, :]
            out[start : start + len(xb)] = logsumexp(lp, axis=1)
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check()
        data = np.asarray(x, dtype=np.float64)
        chunks = []
        for start in range(0, len(data), self.batch_size):
            xb = data[start : start + self.batch_size]
            lp = self._log_prob(xb, self.means_, self.variances_) + np.log(self.weights_)[None, :]
            lp -= logsumexp(lp, axis=1)[:, None]
            chunks.append(np.exp(lp))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.n_components))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def score(self, x: np.ndarray) -> float:
        values = self.score_samples(x)
        return float(values.mean())
