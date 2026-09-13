from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KResult:
    k: int
    image_log_likelihoods: np.ndarray
    stable_fraction: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.image_log_likelihoods))

    @property
    def se(self) -> float:
        x = np.asarray(self.image_log_likelihoods, dtype=np.float64)
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def select_k_one_se(results: list[KResult]) -> int:
    """Best LL -> within 1 SE of best -> stable fraction -> smaller K."""
    if not results:
        raise ValueError("need K results")
    best = max(results, key=lambda r: r.mean)
    cutoff = best.mean - best.se
    candidates = [r for r in results if r.mean >= cutoff]
    candidates.sort(key=lambda r: (-r.stable_fraction, r.k))
    return int(candidates[0].k)
