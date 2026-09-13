from __future__ import annotations

import numpy as np
from .occurrence import Occurrence


def dense_posterior_histogram(posteriors: np.ndarray) -> np.ndarray:
    p = np.asarray(posteriors, dtype=np.float64)
    if p.ndim != 2 or len(p) == 0:
        raise ValueError("posteriors must be a non-empty (locations,K) matrix")
    h = p.mean(axis=0)
    return h / h.sum() if h.sum() > 0 else h


def median_pair_distance(images: list[list[Occurrence]], *, image_size: int = 48) -> float:
    vals: list[float] = []
    for occs in images:
        xy = np.array([o.normalized_xy(image_size) for o in occs], dtype=np.float64)
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                vals.append(float(np.linalg.norm(xy[j] - xy[i])))
    if not vals:
        raise ValueError("need at least one image with two occurrences")
    return float(np.median(vals))


def _quadrant(dx: float, dy: float) -> int:
    right = dx >= 0.0
    down = dy >= 0.0
    return (2 if down else 0) + (1 if right else 0)


def geometry_pair_tensor(occurrences: list[Occurrence], *, stable_components: np.ndarray, distance_median: float, image_size: int = 48) -> np.ndarray:
    stable = np.asarray(stable_components, dtype=np.int64).reshape(-1)
    index = {int(k): i for i, k in enumerate(stable)}
    k = len(stable)
    out = np.zeros((k, k, 8), dtype=np.float64)
    for i, a in enumerate(occurrences):
        if a.motif not in index:
            continue
        ax, ay = a.normalized_xy(image_size)
        for j, b in enumerate(occurrences):
            if i == j or b.motif not in index:
                continue
            bx, by = b.normalized_xy(image_size)
            dx, dy = bx - ax, by - ay
            dist = float(np.hypot(dx, dy))
            db = 0 if dist <= distance_median else 1
            qb = _quadrant(dx, dy)
            out[index[a.motif], index[b.motif], db * 4 + qb] += 1.0
    total = out.sum()
    if total > 0:
        out /= total
    return out
