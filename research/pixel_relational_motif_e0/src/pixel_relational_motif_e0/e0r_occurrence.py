from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controls import destroy_ordered_relations_keyed
from .descriptor import VALID_LOCATIONS, extract_raw_relations
from .e0r_frozen import FrozenCondition
from .occurrence import local_max_nms


K = 128
MEDIAN_BUDGET = 64
NODE_CAP = 80


@dataclass(frozen=True)
class CompactOccurrences:
    components: np.ndarray
    confidence: np.ndarray
    y: np.ndarray
    x: np.ndarray

    def __len__(self) -> int:
        return len(self.components)


def full_k_score_maps(posteriors: np.ndarray, *, side: int = 44) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(posteriors, dtype=np.float64)
    if p.shape != (side * side, K):
        raise ValueError(f"full frozen posterior must have shape ({side * side},{K})")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("posterior must be finite and nonnegative")
    component = np.argmax(p, axis=1)
    confidence = p[np.arange(len(p)), component]
    return confidence.reshape(side, side), component.reshape(side, side)


def extract_candidates(
    image_uint8: np.ndarray,
    *,
    canonical_image_id: int,
    condition: FrozenCondition,
) -> CompactOccurrences:
    image = np.asarray(image_uint8, dtype=np.float64) / 255.0
    s, log_sigma, _ = extract_raw_relations(image)
    if len(s) != VALID_LOCATIONS:
        raise AssertionError("E0.R requires exactly 1,936 descriptor positions")
    if condition.control_seed is not None:
        s = destroy_ordered_relations_keyed(
            s,
            np.full(VALID_LOCATIONS, int(canonical_image_id), dtype=np.int32),
            np.arange(VALID_LOCATIONS, dtype=np.int16),
            control_seed=condition.control_seed,
        )
    latent = condition.transform.transform(s, log_sigma)
    posterior = condition.model.predict_proba(latent)
    score, component = full_k_score_maps(posterior)
    selected = local_max_nms(score, component, radius=2, center_offset=2)
    return CompactOccurrences(
        components=np.asarray([item.motif for item in selected], dtype=np.int16),
        confidence=np.asarray([item.confidence for item in selected], dtype=np.float64),
        y=np.asarray([item.y for item in selected], dtype=np.int16),
        x=np.asarray([item.x for item in selected], dtype=np.int16),
    )


def brute_force_tau(score_arrays: list[np.ndarray], *, median_budget: int = MEDIAN_BUDGET) -> float:
    if not score_arrays:
        raise ValueError("at least one image is required")
    values = np.unique(np.concatenate([np.asarray([0.0]), *[np.asarray(x) for x in score_arrays]]))
    for tau in values:
        counts = np.asarray([np.count_nonzero(np.asarray(x) >= tau) for x in score_arrays])
        if float(np.median(counts)) <= median_budget:
            return float(tau)
    return 1.0


def exact_tau(score_arrays: list[np.ndarray], *, median_budget: int = MEDIAN_BUDGET) -> float:
    """Exact threshold; odd-image fast path equals the registered brute scan."""
    if not score_arrays or median_budget < 0:
        raise ValueError("non-empty score arrays and nonnegative budget required")
    arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in score_arrays]
    if any(not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("candidate confidence must be finite")
    counts0 = np.asarray([len(x) for x in arrays])
    if float(np.median(counts0)) <= median_budget:
        return 0.0
    if len(arrays) % 2 == 0:
        return brute_force_tau(arrays, median_budget=median_budget)
    critical = np.full(len(arrays), -np.inf, dtype=np.float64)
    for index, scores in enumerate(arrays):
        if len(scores) > median_budget:
            critical[index] = np.partition(scores, len(scores) - median_budget - 1)[
                len(scores) - median_budget - 1
            ]
    middle = len(arrays) // 2
    boundary = float(np.partition(critical, middle)[middle])
    successors = [scores[scores > boundary].min() for scores in arrays if np.any(scores > boundary)]
    if not successors:
        return 1.0
    return float(min(successors))


def apply_no_fallback(
    candidates: CompactOccurrences,
    *,
    tau: float,
    cap: int = NODE_CAP,
) -> tuple[CompactOccurrences, bool]:
    if cap <= 0 or not np.isfinite(tau):
        raise ValueError("positive cap and finite tau required")
    eligible = np.flatnonzero(candidates.confidence >= tau)
    cap_used = len(eligible) > cap
    take = eligible[:cap]
    return (
        CompactOccurrences(
            candidates.components[take],
            candidates.confidence[take],
            candidates.y[take],
            candidates.x[take],
        ),
        cap_used,
    )


def occurrence_histograms(all_occurrences: list[CompactOccurrences]) -> np.ndarray:
    out = np.zeros((len(all_occurrences), K), dtype=np.float32)
    for row, occurrences in enumerate(all_occurrences):
        if len(occurrences):
            out[row] = np.bincount(occurrences.components, minlength=K).astype(np.float32)
            out[row] /= np.float32(len(occurrences))
    sums = out.sum(axis=1)
    counts = np.asarray([len(x) for x in all_occurrences])
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise RuntimeError("invalid occurrence histogram")
    if not np.allclose(sums[counts > 0], 1.0, atol=1e-6, rtol=0.0):
        raise RuntimeError("nonempty occurrence histogram must sum to one")
    if np.any(sums[counts == 0] != 0):
        raise RuntimeError("zero-node occurrence histogram must be all zero")
    return out


def occurrence_diagnostics(
    all_occurrences: list[CompactOccurrences],
    cap_flags: np.ndarray,
) -> dict[str, object]:
    counts = np.asarray([len(x) for x in all_occurrences], dtype=np.int32)
    component_counts = np.zeros(K, dtype=np.int64)
    support = np.zeros(K, dtype=np.int64)
    for occurrences in all_occurrences:
        if len(occurrences):
            component_counts += np.bincount(occurrences.components, minlength=K)
            support[np.unique(occurrences.components)] += 1
    total = int(component_counts.sum())
    prevalence = component_counts.astype(np.float64) / total if total else np.zeros(K)
    return {
        "median_nodes": float(np.median(counts)),
        "p90_nodes": float(np.percentile(counts, 90)),
        "p95_nodes": float(np.percentile(counts, 95)),
        "cap_rate": float(np.mean(np.asarray(cap_flags, dtype=bool))),
        "zero_node_rate": float(np.mean(counts == 0)),
        "one_node_rate": float(np.mean(counts == 1)),
        "total_occurrences": total,
        "component_prevalence": prevalence.tolist(),
        "component_image_support_rate": (support / len(all_occurrences)).tolist(),
        "node_counts": counts,
    }
