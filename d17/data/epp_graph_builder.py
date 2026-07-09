"""Evidence-preserving pixel graph builder for D17.

D17 deliberately ignores face/part/landmark priors for node selection. The
input prior dictionary is used as an image/label/sample container only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


NODE_FEATURE_NAMES = [
    "intensity",
    "gx",
    "gy",
    "x_norm",
    "y_norm",
    "grad_mag",
    "local_mean_3x3",
    "local_std_3x3",
    "laplacian_abs",
    "center_surround",
]

EDGE_FEATURE_NAMES = [
    "dx",
    "dy",
    "spatial_dist",
    "abs_intensity_diff",
    "abs_grad_mag_diff",
    "abs_laplacian_diff",
]

KNN_FEATURE_NAMES = [
    "intensity",
    "gx",
    "gy",
    "grad_mag",
    "local_mean_3x3",
    "local_std_3x3",
    "laplacian_abs",
    "center_surround",
]


@dataclass
class EPPGraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor
    sample_index: torch.Tensor
    detected: torch.Tensor
    landmark_missing_flag: torch.Tensor
    image_48: torch.Tensor
    node_feature_names: List[str]
    edge_feature_names: List[str]
    local_edge_count: int
    knn_edge_count: int
    total_edge_count: int
    node_support_mode: str


def _mean3x3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    acc = np.zeros_like(image, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return acc / 9.0


def _laplacian_abs(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    lap = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return np.abs(lap).astype(np.float32)


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    return (values - float(values.mean())) / max(float(values.std()), float(eps))


def compute_pixel_feature_maps(image_norm: np.ndarray) -> Dict[str, np.ndarray]:
    image = np.asarray(image_norm, dtype=np.float32)
    if image.shape != (48, 48):
        raise ValueError(f"D17 expects image shape (48, 48), got {image.shape}")
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    gy, gx = np.gradient(image)
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    local_mean = _mean3x3(image)
    local_sq_mean = _mean3x3(image * image)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var + 1e-12).astype(np.float32)
    lap_abs = _laplacian_abs(image)
    center_surround = (image - local_mean).astype(np.float32)
    return {
        "intensity": image.astype(np.float32),
        "gx": gx,
        "gy": gy,
        "grad_mag": np.clip(grad_mag, 0.0, 1.0).astype(np.float32),
        "local_mean_3x3": np.clip(local_mean, 0.0, 1.0).astype(np.float32),
        "local_std_3x3": np.clip(local_std, 0.0, 0.5).astype(np.float32),
        "laplacian_abs": np.clip(lap_abs, 0.0, 1.0).astype(np.float32),
        "center_surround": np.clip(center_surround, -1.0, 1.0).astype(np.float32),
    }


def compute_detail_score(maps: Dict[str, np.ndarray]) -> np.ndarray:
    return (
        _zscore(maps["grad_mag"])
        + _zscore(maps["laplacian_abs"])
        + _zscore(maps["local_std_3x3"])
        + _zscore(np.abs(maps["center_surround"]))
    ).astype(np.float32)


def _topn_coords(score: np.ndarray, target_count: int) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    n = min(max(int(target_count), 1), flat.size)
    if n >= flat.size:
        selected = np.arange(flat.size, dtype=np.int64)
    else:
        selected = np.argpartition(-flat, n - 1)[:n].astype(np.int64)
        selected = selected[np.argsort(-flat[selected], kind="mergesort")]
    yy = selected // 48
    xx = selected % 48
    coords = np.stack([yy, xx], axis=1).astype(np.int64)
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def _stratified_coords(score: np.ndarray, target_count: int, bins: int) -> np.ndarray:
    bins = max(int(bins), 1)
    target = min(max(int(target_count), 1), 48 * 48)
    per_bin = int(np.ceil(target / float(bins * bins)))
    selected: List[int] = []
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    for by in range(bins):
        y0 = int(round(by * 48 / bins))
        y1 = int(round((by + 1) * 48 / bins))
        for bx in range(bins):
            x0 = int(round(bx * 48 / bins))
            x1 = int(round((bx + 1) * 48 / bins))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            idx = (yy.reshape(-1) * 48 + xx.reshape(-1)).astype(np.int64)
            if idx.size == 0:
                continue
            take = min(per_bin, idx.size)
            local = idx[np.argsort(-flat[idx], kind="mergesort")[:take]]
            selected.extend(int(x) for x in local.tolist())
    selected_arr = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected_arr.size < target:
        used = np.zeros((48 * 48,), dtype=bool)
        used[selected_arr] = True
        remaining = np.flatnonzero(~used)
        fill = remaining[np.argsort(-flat[remaining], kind="mergesort")[: target - selected_arr.size]]
        selected_arr = np.concatenate([selected_arr, fill.astype(np.int64)])
    elif selected_arr.size > target:
        selected_arr = selected_arr[np.argsort(-flat[selected_arr], kind="mergesort")[:target]]
    yy = selected_arr // 48
    xx = selected_arr % 48
    coords = np.stack([yy, xx], axis=1).astype(np.int64)
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def select_node_coords(score: np.ndarray, cfg: Dict[str, Any]) -> tuple[np.ndarray, str]:
    mode = str(cfg.get("node_support_mode", cfg.get("mode", "detail_topN_knn")))
    target = int(cfg.get("target_node_count", cfg.get("target_count", 1800)) or 1800)
    if mode in {"detail_topN_knn", "detail_topn_knn"}:
        return _topn_coords(score, target), "detail_topN_knn"
    if mode == "stratified_detail_knn":
        return _stratified_coords(score, target, int(cfg.get("bins", 6) or 6)), mode
    raise ValueError(f"Unsupported D17 node_support_mode={mode!r}")


def _local_edges(coords: np.ndarray) -> np.ndarray:
    node_ids = -np.ones((48, 48), dtype=np.int64)
    node_ids[coords[:, 0], coords[:, 1]] = np.arange(coords.shape[0], dtype=np.int64)
    yy = coords[:, 0]
    xx = coords[:, 1]
    src_all = node_ids[yy, xx]
    offsets = np.asarray(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        dtype=np.int64,
    )
    ny = yy[:, None] + offsets[None, :, 0]
    nx = xx[:, None] + offsets[None, :, 1]
    inside = (ny >= 0) & (ny < 48) & (nx >= 0) & (nx < 48)
    dst = np.full(ny.shape, -1, dtype=np.int64)
    dst[inside] = node_ids[ny[inside], nx[inside]]
    valid = dst >= 0
    if not valid.any():
        return np.asarray([[0], [0]], dtype=np.int64)
    src_matrix = np.repeat(src_all[:, None], offsets.shape[0], axis=1)
    return np.stack([src_matrix[valid], dst[valid]], axis=0).astype(np.int64)


def _unique_directed_edges(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = edges.T.astype(np.int64, copy=False)
    unique = np.unique(pairs, axis=0)
    return unique.T.astype(np.int64)


def _knn_edges(x: np.ndarray, feature_names: Iterable[str], cfg: Dict[str, Any]) -> np.ndarray:
    k = int(cfg.get("k", 6) or 6)
    if k <= 0 or x.shape[0] <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    names = list(feature_names)
    requested = list(cfg.get("feature_names") or KNN_FEATURE_NAMES)
    indices = [names.index(name) for name in requested if name in names]
    if not indices:
        raise ValueError(f"D17 kNN found no matching feature_names: {requested}")
    feat = np.asarray(x[:, indices], dtype=np.float32)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / np.maximum(feat.std(axis=0, keepdims=True), 1e-6)
    sq = np.sum(feat * feat, axis=1, keepdims=True)
    dist = sq + sq.T - 2.0 * (feat @ feat.T)
    np.fill_diagonal(dist, np.inf)
    kk = min(k, x.shape[0] - 1)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(x.shape[0])[:, None]
    order = np.argsort(dist[row, nn], axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    src = np.repeat(np.arange(x.shape[0], dtype=np.int64), kk)
    return np.stack([src, nn.reshape(-1).astype(np.int64)], axis=0)


def _edge_attr(x: np.ndarray, pos: np.ndarray, edges: np.ndarray) -> np.ndarray:
    src = edges[0].astype(np.int64)
    dst = edges[1].astype(np.int64)
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    spatial = np.sqrt(dx * dx + dy * dy)
    idx = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
    out = np.stack(
        [
            dx,
            dy,
            spatial,
            np.abs(x[src, idx["intensity"]] - x[dst, idx["intensity"]]),
            np.abs(x[src, idx["grad_mag"]] - x[dst, idx["grad_mag"]]),
            np.abs(x[src, idx["laplacian_abs"]] - x[dst, idx["laplacian_abs"]]),
        ],
        axis=1,
    )
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def build_epp_graph(prior: Dict[str, np.ndarray], graph_cfg: Dict[str, Any] | None = None) -> EPPGraphData:
    cfg = dict(graph_cfg or {})
    image = np.asarray(prior["image_48"], dtype=np.float32)
    image_norm = image / 255.0 if float(np.nanmax(image)) > 1.0 else image
    image_norm = np.clip(np.nan_to_num(image_norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    maps = compute_pixel_feature_maps(image_norm)
    score = compute_detail_score(maps)
    coords, support_mode = select_node_coords(score, cfg)
    yy = coords[:, 0]
    xx = coords[:, 1]
    x_norm = (xx.astype(np.float32) / 47.0) * 2.0 - 1.0
    y_norm = (yy.astype(np.float32) / 47.0) * 2.0 - 1.0
    cols = [
        maps["intensity"][yy, xx],
        maps["gx"][yy, xx],
        maps["gy"][yy, xx],
        x_norm,
        y_norm,
        maps["grad_mag"][yy, xx],
        maps["local_mean_3x3"][yy, xx],
        maps["local_std_3x3"][yy, xx],
        maps["laplacian_abs"][yy, xx],
        maps["center_surround"][yy, xx],
    ]
    x = np.stack(cols, axis=1).astype(np.float32)
    pos = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    local = _unique_directed_edges(_local_edges(coords))
    knn_cfg = dict(cfg.get("knn_edges", {}) or {})
    knn_cfg.setdefault("k", cfg.get("knn_k", 6))
    knn_cfg.setdefault("feature_names", KNN_FEATURE_NAMES)
    knn = _unique_directed_edges(_knn_edges(x, NODE_FEATURE_NAMES, knn_cfg))
    total = _unique_directed_edges(np.concatenate([local, knn], axis=1))
    edge_attr = _edge_attr(x, pos, total)
    local_pairs = {tuple(pair) for pair in local.T.tolist()}
    total_pairs = {tuple(pair) for pair in total.T.tolist()}
    knn_added_count = len(total_pairs - local_pairs)
    return EPPGraphData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(total).long(),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos),
        y=torch.tensor(int(np.asarray(prior["label"]).item()), dtype=torch.long),
        sample_index=torch.tensor(int(np.asarray(prior["sample_index"]).item()), dtype=torch.long),
        detected=torch.tensor(bool(np.asarray(prior.get("detected", True)).item()), dtype=torch.bool),
        landmark_missing_flag=torch.tensor(int(np.asarray(prior.get("landmark_missing_flag", 0)).item()), dtype=torch.long),
        image_48=torch.from_numpy(image_norm.astype(np.float32)),
        node_feature_names=list(NODE_FEATURE_NAMES),
        edge_feature_names=list(EDGE_FEATURE_NAMES),
        local_edge_count=int(local.shape[1]),
        knn_edge_count=int(knn_added_count),
        total_edge_count=int(total.shape[1]),
        node_support_mode=support_mode,
    )

