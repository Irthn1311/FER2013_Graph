from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class Occurrence:
    motif: int
    confidence: float
    y: int
    x: int

    def normalized_xy(self, image_size: int = 48) -> tuple[float, float]:
        denom = float(image_size - 1)
        return self.x / denom, self.y / denom


def stable_score_map(posteriors: np.ndarray, stable_components: np.ndarray, *, side: int = 44) -> tuple[np.ndarray, np.ndarray]:
    """Max original GMM posterior over stable components, without renormalizing."""
    p = np.asarray(posteriors, dtype=np.float64)
    stable = np.asarray(stable_components, dtype=np.int64).reshape(-1)
    if len(stable) == 0:
        raise ValueError("stable component set is empty")
    if p.ndim != 2 or p.shape[0] != side * side:
        raise ValueError("posteriors must be (side*side,K)")
    sub = p[:, stable]
    local_idx = np.argmax(sub, axis=1)
    score = sub[np.arange(len(sub)), local_idx]
    motif = stable[local_idx]
    return score.reshape(side, side), motif.reshape(side, side)


def local_max_nms(score_map: np.ndarray, motif_map: np.ndarray, *, radius: int = 2, center_offset: int = 2) -> list[Occurrence]:
    score = np.asarray(score_map, dtype=np.float64)
    motif = np.asarray(motif_map, dtype=np.int64)
    if score.shape != motif.shape or score.ndim != 2:
        raise ValueError("score and motif maps must be same-shape 2D arrays")
    local = score == maximum_filter(score, size=3, mode="constant", cval=-np.inf)
    yy, xx = np.nonzero(local)
    order = np.lexsort((xx, yy, -score[yy, xx]))
    selected: list[tuple[int, int]] = []
    out: list[Occurrence] = []
    r2 = radius * radius
    for idx in order:
        y, x = int(yy[idx]), int(xx[idx])
        if any((y - sy) ** 2 + (x - sx) ** 2 <= r2 for sy, sx in selected):
            continue
        selected.append((y, x))
        out.append(Occurrence(motif=int(motif[y, x]), confidence=float(score[y, x]), y=y + center_offset, x=x + center_offset))
    return out


def choose_tau_for_budget(candidate_lists: list[list[Occurrence]], *, median_budget: int = 64) -> float:
    """Lowest confidence threshold whose median image count is <= budget."""
    if not candidate_lists:
        raise ValueError("need at least one image's candidates")
    values = sorted({0.0, *[o.confidence for xs in candidate_lists for o in xs]})
    for tau in values:
        counts = [sum(o.confidence >= tau for o in xs) for xs in candidate_lists]
        if float(np.median(counts)) <= median_budget:
            return float(tau)
    return 1.0


def apply_occurrence_policy(candidates: list[Occurrence], *, tau: float, cap: int = 80, minimum_nodes: int = 2) -> tuple[list[Occurrence], bool, bool]:
    ordered = sorted(candidates, key=lambda o: (-o.confidence, o.y, o.x, o.motif))
    kept = [o for o in ordered if o.confidence >= tau]
    cap_used = len(kept) > cap
    kept = kept[:cap]
    fallback = False
    if len(kept) < minimum_nodes:
        fallback = True
        kept = ordered[: min(minimum_nodes, len(ordered))]
    return kept, cap_used, fallback


def occurrence_diagnostics(all_occurrences: list[list[Occurrence]], cap_flags, fallback_flags) -> dict[str, float]:
    counts = np.asarray([len(x) for x in all_occurrences], dtype=np.float64)
    return {"median_nodes": float(np.median(counts)), "p90_nodes": float(np.percentile(counts, 90)), "p95_nodes": float(np.percentile(counts, 95)), "cap_rate": float(np.mean(np.asarray(cap_flags, dtype=bool))), "fallback_rate": float(np.mean(np.asarray(fallback_flags, dtype=bool)))}
