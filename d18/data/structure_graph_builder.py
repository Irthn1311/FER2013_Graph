"""D18 structure-guided pixel graph builder.

D18 keeps D17 prior-free pixel node features, but uses facial structure only to
create additional relation edges between existing pixel nodes. Priors are never
written into node features or graph readout inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

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

BASE_EDGE_FEATURE_NAMES = [
    "dx",
    "dy",
    "spatial_dist",
    "abs_intensity_diff",
    "abs_grad_mag_diff",
    "abs_laplacian_diff",
]

STRUCTURE_EDGE_FEATURE_NAMES = BASE_EDGE_FEATURE_NAMES + [
    "relation_type_norm",
    "part_relation_score",
    "facial_distance_score",
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

GROUP_INDICES = {
    "mouth": [5, 6, 7],
    "eye": [0, 1],
    "brow": [2, 3],
    "nose_cheek": [4, 8, 9, 10],
}

DEFAULT_RELATIONS = [
    ("brow", "eye"),
    ("eye", "mouth"),
    ("nose_cheek", "mouth"),
    ("mouth", "nose_cheek"),
    ("eye", "nose_cheek"),
    ("brow", "mouth"),
]

EDGE_TYPE_LOCAL = 0
EDGE_TYPE_KNN = 1
EDGE_TYPE_STRUCTURE = 2
PIXEL_EVIDENCE_FEATURES = [
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
class D18GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor
    sample_index: torch.Tensor
    detected: torch.Tensor
    landmark_missing_flag: torch.Tensor
    image_48: torch.Tensor
    edge_type: torch.Tensor
    structure_relation_id: torch.Tensor
    node_feature_names: List[str]
    edge_feature_names: List[str]
    local_edge_count: int
    knn_edge_count: int
    structure_edge_count: int
    total_edge_count: int
    structure_edge_count_before_purification: int
    structure_edge_count_after_purification: int
    purification_compatibility_kept_mean: float
    purification_compatibility_dropped_mean: float
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
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1]
    return np.abs(lap).astype(np.float32)


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    return (values - float(values.mean())) / max(float(values.std()), float(eps))


def compute_pixel_feature_maps(image_norm: np.ndarray) -> Dict[str, np.ndarray]:
    image = np.asarray(image_norm, dtype=np.float32)
    if image.shape != (48, 48):
        raise ValueError(f"D18 expects image shape (48, 48), got {image.shape}")
    image = np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    local_mean = _mean3x3(image)
    local_sq_mean = _mean3x3(image * image)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0) + 1e-12).astype(np.float32)
    lap_abs = _laplacian_abs(image)
    center_surround = (image - local_mean).astype(np.float32)
    return {
        "intensity": image,
        "gx": gx.astype(np.float32),
        "gy": gy.astype(np.float32),
        "grad_mag": np.clip(grad_mag, 0.0, 1.0).astype(np.float32),
        "local_mean_3x3": np.clip(local_mean, 0.0, 1.0).astype(np.float32),
        "local_std_3x3": np.clip(local_std, 0.0, 0.5).astype(np.float32),
        "laplacian_abs": np.clip(lap_abs, 0.0, 1.0).astype(np.float32),
        "center_surround": np.clip(center_surround, -1.0, 1.0).astype(np.float32),
    }


def compute_detail_score(maps: Dict[str, np.ndarray]) -> np.ndarray:
    return (_zscore(maps["grad_mag"]) + _zscore(maps["laplacian_abs"]) + _zscore(maps["local_std_3x3"]) + _zscore(np.abs(maps["center_surround"]))).astype(np.float32)


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
            local = idx[np.argsort(-flat[idx], kind="mergesort")[: min(per_bin, idx.size)]]
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
    mode = str(cfg.get("node_support_mode", cfg.get("mode", "stratified_detail_knn")))
    target = int(cfg.get("target_node_count", cfg.get("target_count", 1800)) or 1800)
    if mode in {"detail_topN_knn", "detail_topn_knn"}:
        return _topn_coords(score, target), "detail_topN_knn"
    if mode == "stratified_detail_knn":
        return _stratified_coords(score, target, int(cfg.get("bins", 6) or 6)), mode
    raise ValueError(f"Unsupported D18 node_support_mode={mode!r}")


def _local_edges(coords: np.ndarray) -> np.ndarray:
    node_ids = -np.ones((48, 48), dtype=np.int64)
    node_ids[coords[:, 0], coords[:, 1]] = np.arange(coords.shape[0], dtype=np.int64)
    offsets = np.asarray([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], dtype=np.int64)
    yy = coords[:, 0]
    xx = coords[:, 1]
    ny = yy[:, None] + offsets[None, :, 0]
    nx = xx[:, None] + offsets[None, :, 1]
    inside = (ny >= 0) & (ny < 48) & (nx >= 0) & (nx < 48)
    dst = np.full(ny.shape, -1, dtype=np.int64)
    dst[inside] = node_ids[ny[inside], nx[inside]]
    valid = dst >= 0
    if not valid.any():
        return np.zeros((2, 0), dtype=np.int64)
    src = np.repeat(np.arange(coords.shape[0], dtype=np.int64)[:, None], offsets.shape[0], axis=1)
    return np.stack([src[valid], dst[valid]], axis=0).astype(np.int64)


def _unique_directed_edges(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.unique(edges.T.astype(np.int64, copy=False), axis=0).T.astype(np.int64)


def _knn_edges(x: np.ndarray, feature_names: Iterable[str], cfg: Dict[str, Any]) -> np.ndarray:
    k = int(cfg.get("k", 6) or 6)
    if k <= 0 or x.shape[0] <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    names = list(feature_names)
    requested = list(cfg.get("feature_names") or KNN_FEATURE_NAMES)
    indices = [names.index(name) for name in requested if name in names]
    if not indices:
        raise ValueError(f"D18 kNN found no matching feature_names: {requested}")
    feat = np.asarray(x[:, indices], dtype=np.float32)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / np.maximum(feat.std(axis=0, keepdims=True), 1e-6)
    sq = np.sum(feat * feat, axis=1, keepdims=True)
    dist = sq + sq.T - 2.0 * (feat @ feat.T)
    np.fill_diagonal(dist, np.inf)
    kk = min(k, x.shape[0] - 1)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(x.shape[0])[:, None]
    nn = np.take_along_axis(nn, np.argsort(dist[row, nn], axis=1), axis=1)
    src = np.repeat(np.arange(x.shape[0], dtype=np.int64), kk)
    return np.stack([src, nn.reshape(-1).astype(np.int64)], axis=0)


def _group_score(part_node: np.ndarray, group: str) -> np.ndarray:
    indices = [idx for idx in GROUP_INDICES.get(group, []) if idx < part_node.shape[1]]
    if not indices:
        return np.zeros((part_node.shape[0],), dtype=np.float32)
    return np.max(part_node[:, indices], axis=1).astype(np.float32)


def _select_group_nodes(score: np.ndarray, max_nodes: int, min_score: float) -> np.ndarray:
    valid = np.flatnonzero(score >= float(min_score))
    if valid.size == 0:
        valid = np.argsort(-score, kind="mergesort")[: max(1, min(int(max_nodes), score.size))]
    order = valid[np.argsort(-score[valid], kind="mergesort")]
    return order[: max(1, int(max_nodes))].astype(np.int64)


def _structure_edges(coords: np.ndarray, part_node: np.ndarray, cfg: Dict[str, Any]) -> tuple[np.ndarray, Dict[Tuple[int, int], tuple[int, float]]]:
    if not bool(cfg.get("enabled", True)):
        return np.zeros((2, 0), dtype=np.int64), {}
    max_nodes = int(cfg.get("max_nodes_per_group", 32) or 32)
    targets_per_source = int(cfg.get("targets_per_source", 4) or 4)
    min_score = float(cfg.get("min_score", 0.05))
    bidirectional = bool(cfg.get("bidirectional", True))
    relations_raw = cfg.get("relations") or DEFAULT_RELATIONS
    relations = [(str(a), str(b)) for a, b in relations_raw]
    group_scores = {name: _group_score(part_node, name) for rel in relations for name in rel}
    group_nodes = {name: _select_group_nodes(score, max_nodes, min_score) for name, score in group_scores.items()}
    pos = coords.astype(np.float32) / 47.0
    edges: List[tuple[int, int]] = []
    meta: Dict[Tuple[int, int], tuple[int, float]] = {}
    for rel_id, (src_group, dst_group) in enumerate(relations, start=1):
        src_nodes = group_nodes.get(src_group, np.zeros((0,), dtype=np.int64))
        dst_nodes = group_nodes.get(dst_group, np.zeros((0,), dtype=np.int64))
        if src_nodes.size == 0 or dst_nodes.size == 0:
            continue
        for src in src_nodes.tolist():
            candidates = dst_nodes[dst_nodes != int(src)]
            if candidates.size == 0:
                continue
            dist = np.sum((pos[candidates] - pos[int(src)]) ** 2, axis=1)
            order = np.argsort(dist, kind="mergesort")[: min(targets_per_source, candidates.size)]
            for dst in candidates[order].tolist():
                score = float(np.sqrt(max(group_scores[src_group][int(src)], 0.0) * max(group_scores[dst_group][int(dst)], 0.0)))
                pair = (int(src), int(dst))
                edges.append(pair)
                meta[pair] = (rel_id, score)
                if bidirectional:
                    rev = (int(dst), int(src))
                    edges.append(rev)
                    meta[rev] = (rel_id, score)
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), {}
    return _unique_directed_edges(np.asarray(edges, dtype=np.int64).T), meta


def _purify_structure_edges(
    x: np.ndarray,
    structure: np.ndarray,
    meta: Dict[Tuple[int, int], tuple[int, float]],
    cfg: Dict[str, Any],
) -> tuple[np.ndarray, Dict[Tuple[int, int], tuple[int, float]], Dict[str, float]]:
    purification = dict((cfg.get("purification") or {}))
    stats = {
        "before": float(structure.shape[1]),
        "after": float(structure.shape[1]),
        "kept_compatibility_mean": float("nan"),
        "dropped_compatibility_mean": float("nan"),
    }
    if not bool(purification.get("enabled", False)) or structure.size == 0:
        return structure, meta, stats
    keep_ratio = float(purification.get("keep_ratio", 0.6))
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio >= 1.0:
        return structure, meta, stats
    idx = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
    feature_indices = [idx[name] for name in PIXEL_EVIDENCE_FEATURES if name in idx]
    feat = np.asarray(x[:, feature_indices], dtype=np.float32)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / np.maximum(feat.std(axis=0, keepdims=True), 1e-6)
    src = structure[0].astype(np.int64)
    dst = structure[1].astype(np.int64)
    dist = np.sqrt(np.sum((feat[src] - feat[dst]) ** 2, axis=1)).astype(np.float32)
    compat = np.exp(-dist).astype(np.float32)
    groups: Dict[tuple[int, int], List[int]] = {}
    for edge_i, pair in enumerate(structure.T.tolist()):
        rid = int(meta.get((int(pair[0]), int(pair[1])), (0, 0.0))[0])
        if str(purification.get("mode", "per_relation_or_per_source")) == "per_relation_or_per_source":
            key = (rid, int(pair[0]))
        else:
            key = (0, int(pair[0]))
        groups.setdefault(key, []).append(edge_i)
    keep_mask = np.zeros((structure.shape[1],), dtype=bool)
    for indices in groups.values():
        local_scores = compat[indices]
        n_keep = max(1, int(np.ceil(len(indices) * keep_ratio)))
        chosen = np.asarray(indices, dtype=np.int64)[np.argsort(-local_scores, kind="mergesort")[:n_keep]]
        keep_mask[chosen] = True
    kept = structure[:, keep_mask]
    new_meta = {tuple(map(int, pair)): meta[tuple(map(int, pair))] for pair in kept.T.tolist() if tuple(map(int, pair)) in meta}
    stats = {
        "before": float(structure.shape[1]),
        "after": float(kept.shape[1]),
        "kept_compatibility_mean": float(np.mean(compat[keep_mask])) if bool(keep_mask.any()) else float("nan"),
        "dropped_compatibility_mean": float(np.mean(compat[~keep_mask])) if bool((~keep_mask).any()) else float("nan"),
    }
    return kept.astype(np.int64), new_meta, stats


def _edge_metadata(
    total: np.ndarray,
    local_pairs: set[tuple[int, int]],
    local_knn_pairs: set[tuple[int, int]],
    structure_meta: Dict[Tuple[int, int], tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    edge_type = np.zeros((total.shape[1],), dtype=np.int64)
    relation_id = np.zeros((total.shape[1],), dtype=np.int64)
    for i, pair_raw in enumerate(total.T.tolist()):
        pair = (int(pair_raw[0]), int(pair_raw[1]))
        if pair in local_pairs:
            edge_type[i] = EDGE_TYPE_LOCAL
        elif pair in local_knn_pairs:
            edge_type[i] = EDGE_TYPE_KNN
        elif pair in structure_meta:
            edge_type[i] = EDGE_TYPE_STRUCTURE
            relation_id[i] = int(structure_meta[pair][0])
        else:
            edge_type[i] = EDGE_TYPE_KNN
    return edge_type, relation_id


def _edge_attr(x: np.ndarray, pos: np.ndarray, edges: np.ndarray, meta: Dict[Tuple[int, int], tuple[int, float]], edge_schema: str, relation_count: int) -> tuple[np.ndarray, List[str]]:
    src = edges[0].astype(np.int64)
    dst = edges[1].astype(np.int64)
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    spatial = np.sqrt(dx * dx + dy * dy)
    idx = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
    base = [
        dx,
        dy,
        spatial,
        np.abs(x[src, idx["intensity"]] - x[dst, idx["intensity"]]),
        np.abs(x[src, idx["grad_mag"]] - x[dst, idx["grad_mag"]]),
        np.abs(x[src, idx["laplacian_abs"]] - x[dst, idx["laplacian_abs"]]),
    ]
    if str(edge_schema) in {"structure9", "relation9", "d18b"}:
        rel_type = np.zeros((edges.shape[1],), dtype=np.float32)
        rel_score = np.zeros_like(rel_type)
        facial_dist = np.zeros_like(rel_type)
        denom = max(int(relation_count), 1)
        scale = 0.75
        for i, pair in enumerate(edges.T.tolist()):
            key = (int(pair[0]), int(pair[1]))
            if key in meta:
                rid, score = meta[key]
                rel_type[i] = float(rid) / float(denom)
                rel_score[i] = float(score)
                facial_dist[i] = float(np.exp(-float(spatial[i]) / scale))
        out = np.stack(base + [rel_type, np.clip(rel_score, 0.0, 1.0), facial_dist], axis=1)
        names = list(STRUCTURE_EDGE_FEATURE_NAMES)
    else:
        out = np.stack(base, axis=1)
        names = list(BASE_EDGE_FEATURE_NAMES)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32), names


def build_structure_graph(prior: Dict[str, np.ndarray], graph_cfg: Dict[str, Any] | None = None) -> D18GraphData:
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
    x = np.stack([
        maps["intensity"][yy, xx], maps["gx"][yy, xx], maps["gy"][yy, xx], x_norm, y_norm,
        maps["grad_mag"][yy, xx], maps["local_mean_3x3"][yy, xx], maps["local_std_3x3"][yy, xx],
        maps["laplacian_abs"][yy, xx], maps["center_surround"][yy, xx],
    ], axis=1).astype(np.float32)
    pos = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    part_masks = np.asarray(prior.get("part_soft_masks"), dtype=np.float32)
    if part_masks.ndim != 3:
        part_node = np.zeros((coords.shape[0], 0), dtype=np.float32)
    else:
        part_node = np.transpose(part_masks[:, yy, xx], (1, 0)).astype(np.float32)
    local = _unique_directed_edges(_local_edges(coords))
    knn_cfg = dict(cfg.get("knn_edges", {}) or {})
    knn_cfg.setdefault("k", cfg.get("knn_k", 6))
    knn_cfg.setdefault("feature_names", KNN_FEATURE_NAMES)
    knn = _unique_directed_edges(_knn_edges(x, NODE_FEATURE_NAMES, knn_cfg))
    structure_cfg = dict(cfg.get("structure_edges", {}) or {})
    if bool(structure_cfg.get("force_remove", False)):
        structure = np.zeros((2, 0), dtype=np.int64)
        structure_meta: Dict[Tuple[int, int], tuple[int, float]] = {}
        purification_stats = {"before": 0.0, "after": 0.0, "kept_compatibility_mean": float("nan"), "dropped_compatibility_mean": float("nan")}
    else:
        structure, structure_meta = _structure_edges(coords, part_node, structure_cfg)
        structure_before_purification = int(structure.shape[1])
        structure, structure_meta, purification_stats = _purify_structure_edges(x, structure, structure_meta, structure_cfg)
        purification_stats["before"] = float(structure_before_purification)
    local_pairs = {tuple(pair) for pair in local.T.tolist()}
    local_knn = _unique_directed_edges(np.concatenate([local, knn], axis=1))
    local_knn_pairs = {tuple(pair) for pair in local_knn.T.tolist()}
    total = _unique_directed_edges(np.concatenate([local_knn, structure], axis=1))
    total_pairs = {tuple(pair) for pair in total.T.tolist()}
    knn_added_count = len(local_knn_pairs - local_pairs)
    structure_added_count = len(total_pairs - local_knn_pairs)
    edge_type, structure_relation_id = _edge_metadata(total, local_pairs, local_knn_pairs, structure_meta)
    relation_count = len(structure_cfg.get("relations") or DEFAULT_RELATIONS)
    edge_attr, edge_names = _edge_attr(x, pos, total, structure_meta, str(cfg.get("edge_schema", "base6")), relation_count)
    return D18GraphData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(total).long(),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos),
        y=torch.tensor(int(np.asarray(prior["label"]).item()), dtype=torch.long),
        sample_index=torch.tensor(int(np.asarray(prior["sample_index"]).item()), dtype=torch.long),
        detected=torch.tensor(bool(np.asarray(prior.get("detected", True)).item()), dtype=torch.bool),
        landmark_missing_flag=torch.tensor(int(np.asarray(prior.get("landmark_missing_flag", 0)).item()), dtype=torch.long),
        image_48=torch.from_numpy(image_norm.astype(np.float32)),
        edge_type=torch.from_numpy(edge_type).long(),
        structure_relation_id=torch.from_numpy(structure_relation_id).long(),
        node_feature_names=list(NODE_FEATURE_NAMES),
        edge_feature_names=edge_names,
        local_edge_count=int(local.shape[1]),
        knn_edge_count=int(knn_added_count),
        structure_edge_count=int(structure_added_count),
        total_edge_count=int(total.shape[1]),
        structure_edge_count_before_purification=int(purification_stats["before"]),
        structure_edge_count_after_purification=int(purification_stats["after"]),
        purification_compatibility_kept_mean=float(purification_stats["kept_compatibility_mean"]),
        purification_compatibility_dropped_mean=float(purification_stats["dropped_compatibility_mean"]),
        node_support_mode=support_mode,
    )
