"""Pixel graph builder and variable-size D16 batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from d16.data.detail_node_features import DEFAULT_DETAIL_FEATURES, sample_detail_features


_FULL_MASK_COORDS_EDGES: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class D16GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    pos: torch.Tensor
    y: torch.Tensor
    sample_index: torch.Tensor
    part_soft: torch.Tensor
    face_mask: torch.Tensor
    valid_part_mask: torch.Tensor
    valid_anchor_mask: torch.Tensor
    detected: torch.Tensor
    landmark_missing_flag: torch.Tensor
    image_48: torch.Tensor
    node_feature_names: List[str] | None = None
    edge_feature_names: List[str] | None = None

    def to_pyg_data(self):
        try:
            from torch_geometric.data import Data  # type: ignore
        except Exception as exc:
            raise ImportError("torch_geometric is not installed; D16 uses lightweight graph data by default") from exc
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            pos=self.pos,
            y=self.y,
            sample_index=self.sample_index,
            part_soft=self.part_soft,
            face_mask=self.face_mask,
            valid_part_mask=self.valid_part_mask,
            valid_anchor_mask=self.valid_anchor_mask,
            detected=self.detected,
            landmark_missing_flag=self.landmark_missing_flag,
            image_48=self.image_48,
            node_feature_names=self.node_feature_names,
            edge_feature_names=self.edge_feature_names,
        )


@dataclass
class D16Batch:
    x_cat: torch.Tensor
    edge_index_cat: torch.Tensor
    edge_attr_cat: torch.Tensor | None
    batch_index: torch.Tensor
    ptr: torch.Tensor
    y: torch.Tensor
    sample_index: torch.Tensor
    pos_cat: torch.Tensor
    part_soft_cat: torch.Tensor
    face_mask_cat: torch.Tensor
    valid_part_mask: torch.Tensor
    valid_anchor_mask: torch.Tensor
    detected: torch.Tensor
    landmark_missing_flag: torch.Tensor
    image_48: torch.Tensor
    node_feature_names: List[str] | None = None
    edge_feature_names: List[str] | None = None

    def to(self, device: torch.device | str) -> "D16Batch":
        return D16Batch(
            x_cat=self.x_cat.to(device),
            edge_index_cat=self.edge_index_cat.to(device),
            edge_attr_cat=None if self.edge_attr_cat is None else self.edge_attr_cat.to(device),
            batch_index=self.batch_index.to(device),
            ptr=self.ptr.to(device),
            y=self.y.to(device),
            sample_index=self.sample_index.to(device),
            pos_cat=self.pos_cat.to(device),
            part_soft_cat=self.part_soft_cat.to(device),
            face_mask_cat=self.face_mask_cat.to(device),
            valid_part_mask=self.valid_part_mask.to(device),
            valid_anchor_mask=self.valid_anchor_mask.to(device),
            detected=self.detected.to(device),
            landmark_missing_flag=self.landmark_missing_flag.to(device),
            image_48=self.image_48.to(device),
            node_feature_names=self.node_feature_names,
            edge_feature_names=self.edge_feature_names,
        )

    @property
    def num_graphs(self) -> int:
        return int(self.y.numel())


def _binary_dilate(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        acc = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= padded[1 + dy : 49 + dy, 1 + dx : 49 + dx]
        out = acc
    return out


def _node_mask(face_mask: np.ndarray, graph_mode: str, threshold: float, context_pixels: int) -> np.ndarray:
    if graph_mode == "full_with_mask":
        return np.ones((48, 48), dtype=bool)
    if graph_mode == "face_plus_context":
        base = np.asarray(face_mask) > float(threshold)
        mask = _binary_dilate(base, iterations=int(context_pixels))
        if not mask.any():
            mask = np.ones((48, 48), dtype=bool)
        return mask
    if graph_mode == "face_only":
        mask = np.asarray(face_mask) > float(threshold)
        return mask if mask.any() else np.ones((48, 48), dtype=bool)
    raise ValueError(f"Unknown D16 graph_mode={graph_mode!r}")


def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(image.astype(np.float32))
    return gx.astype(np.float32), gy.astype(np.float32)


def _edges_for_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _FULL_MASK_COORDS_EDGES
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape == (48, 48) and bool(mask_bool.all()):
        if _FULL_MASK_COORDS_EDGES is None:
            _FULL_MASK_COORDS_EDGES = _edges_for_mask_uncached(mask_bool)
        coords, edges = _FULL_MASK_COORDS_EDGES
        return coords, edges
    return _edges_for_mask_uncached(mask_bool)


def _edges_for_mask_uncached(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_ids = -np.ones((48, 48), dtype=np.int64)
    coords = np.argwhere(mask)
    if coords.size == 0:
        coords = np.asarray([[0, 0]], dtype=np.int64)
        return coords, np.asarray([[0], [0]], dtype=np.int64)
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
        edges = np.asarray([[0], [0]], dtype=np.int64)
    else:
        src_matrix = np.repeat(src_all[:, None], offsets.shape[0], axis=1)
        edges = np.stack([src_matrix[valid], dst[valid]], axis=0).astype(np.int64)
    return coords.astype(np.int64), edges


_ANCHOR_GROUP_INDICES = {
    "mouth": [5, 6, 7],
    "eye": [0, 1],
    "brow": [2, 3],
    "nose_cheek": [4, 8, 9],
}


def _anchor_group_prior(part_soft: np.ndarray, group: str, part_count: int) -> np.ndarray:
    if group == "global":
        return np.ones((part_soft.shape[0],), dtype=np.float32)
    indices = [idx for idx in _ANCHOR_GROUP_INDICES.get(group, []) if idx < int(part_count)]
    if not indices:
        return np.zeros((part_soft.shape[0],), dtype=np.float32)
    return np.max(part_soft[:, indices], axis=1).astype(np.float32)


def _top_indices_by_weight(weights: np.ndarray, threshold: float, max_count: int) -> np.ndarray:
    eligible = np.flatnonzero(weights >= float(threshold))
    if eligible.size == 0:
        eligible = np.flatnonzero(weights > 0.0)
    if eligible.size == 0:
        return eligible.astype(np.int64)
    if max_count > 0 and eligible.size > int(max_count):
        order = np.argsort(weights[eligible])[::-1][: int(max_count)]
        eligible = eligible[order]
    return eligible.astype(np.int64)


def _add_part_anchor_nodes(
    x: np.ndarray,
    pos: np.ndarray,
    part_soft: np.ndarray,
    face_values: np.ndarray,
    edges: np.ndarray,
    anchor_cfg: Dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = dict(anchor_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return x, pos, part_soft, face_values, edges
    groups = list(cfg.get("groups") or ["mouth", "eye", "brow", "nose_cheek", "global"])
    groups = [str(group) for group in groups if str(group)]
    if not groups:
        return x, pos, part_soft, face_values, edges
    connect_threshold = float(cfg.get("connect_threshold", 0.20))
    max_pixels = int(cfg.get("max_pixels_per_anchor", 384) or 0)
    connect_global_to_pixels = bool(cfg.get("connect_global_to_pixels", False))
    anchor_to_anchor = bool(cfg.get("anchor_to_anchor", True))
    bidirectional = bool(cfg.get("bidirectional", True))
    part_count = int(part_soft.shape[1])
    anchor_x: List[np.ndarray] = []
    anchor_pos: List[np.ndarray] = []
    anchor_part: List[np.ndarray] = []
    anchor_face: List[float] = []
    anchor_edges: List[tuple[int, int]] = []
    base_n = int(x.shape[0])
    for group_idx, group in enumerate(groups):
        weights = _anchor_group_prior(part_soft, group, part_count).astype(np.float32)
        if group == "global":
            weights = np.maximum(face_values.astype(np.float32), 1e-6)
        denom = float(np.sum(weights))
        if denom <= 1e-6:
            node_x = np.zeros((x.shape[1],), dtype=np.float32)
            node_pos = np.zeros((pos.shape[1],), dtype=np.float32)
        else:
            normalized = (weights / denom).astype(np.float32)
            node_x = np.sum(x * normalized[:, None], axis=0).astype(np.float32)
            node_pos = np.sum(pos * normalized[:, None], axis=0).astype(np.float32)
        node_part = np.zeros((part_count,), dtype=np.float32)
        if group == "global":
            node_part[:] = np.mean(part_soft, axis=0).astype(np.float32)
        else:
            for part_idx in [idx for idx in _ANCHOR_GROUP_INDICES.get(group, []) if idx < part_count]:
                node_part[part_idx] = 1.0
        anchor_x.append(node_x)
        anchor_pos.append(node_pos)
        anchor_part.append(node_part)
        anchor_face.append(1.0)
        anchor_id = base_n + group_idx
        if group != "global" or connect_global_to_pixels:
            selected = _top_indices_by_weight(weights, connect_threshold, max_pixels)
            for node_id in selected.tolist():
                anchor_edges.append((int(node_id), anchor_id))
                if bidirectional:
                    anchor_edges.append((anchor_id, int(node_id)))
    if anchor_to_anchor and len(groups) > 1:
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i == j:
                    continue
                anchor_edges.append((base_n + i, base_n + j))
    if not anchor_x:
        return x, pos, part_soft, face_values, edges
    x_out = np.concatenate([x, np.stack(anchor_x, axis=0).astype(np.float32)], axis=0)
    pos_out = np.concatenate([pos, np.stack(anchor_pos, axis=0).astype(np.float32)], axis=0)
    part_out = np.concatenate([part_soft, np.stack(anchor_part, axis=0).astype(np.float32)], axis=0)
    face_out = np.concatenate([face_values.astype(np.float32), np.asarray(anchor_face, dtype=np.float32)], axis=0)
    if anchor_edges:
        extra_edges = np.asarray(anchor_edges, dtype=np.int64).T
        edges_out = np.concatenate([edges.astype(np.int64), extra_edges], axis=1)
    else:
        edges_out = edges.astype(np.int64)
    return x_out.astype(np.float32), pos_out.astype(np.float32), part_out.astype(np.float32), face_out.astype(np.float32), edges_out.astype(np.int64)


def _detail_features_enabled(detail_features: Dict[str, Any] | None) -> bool:
    if not detail_features:
        return False
    return bool(detail_features.get("enabled", False)) and bool(detail_features.get("append_to_x", True))


def _edge_features_enabled(edge_features: Dict[str, Any] | None) -> bool:
    if not edge_features:
        return False
    return bool(edge_features.get("enabled", False)) and bool(edge_features.get("append_to_edge_attr", True))


def _routing_only_node_features(node_features: Dict[str, Any] | None, prior_usage: str | None = None) -> bool:
    cfg = dict(node_features or {})
    mode = str(
        cfg.get(
            "feature_prior_mode",
            cfg.get("prior_mode", cfg.get("mode", prior_usage or "")),
        )
        or ""
    ).lower()
    if mode in {"routing_only", "no_prior_features", "pixel_only", "deprioritized"}:
        return True
    return bool(cfg.get("exclude_prior_features", False))


def _node_feature_flags(node_features: Dict[str, Any] | None, prior_usage: str | None = None) -> Dict[str, bool]:
    cfg = dict(node_features or {})
    routing_only = _routing_only_node_features(cfg, prior_usage=prior_usage)
    return {
        "face": bool(cfg.get("include_face_mask", not routing_only)),
        "part": bool(cfg.get("include_part_soft", cfg.get("include_part_soft_masks", not routing_only))),
        "distance": bool(cfg.get("include_distance_maps", not routing_only)),
        "missing": bool(cfg.get("include_landmark_missing_flag", not routing_only)),
    }


def _edge_feature_names(edge_features: Dict[str, Any] | None) -> List[str]:
    names = list((edge_features or {}).get("features") or [])
    if names:
        return [str(name) for name in names]
    return [
        "dx",
        "dy",
        "spatial_dist",
        "abs_intensity_diff",
        "abs_grad_mag_diff",
        "abs_laplacian_diff",
        "part_similarity",
        "same_dominant_part",
    ]


def _edge_regularization_probability(cfg: Dict[str, Any]) -> float:
    probability = float(cfg.get("probability", 0.0) or 0.0)
    epoch = int(cfg.get("current_epoch", 0) or 0)
    for item in cfg.get("schedule") or []:
        if epoch >= int(item.get("start_epoch", 1) or 1):
            probability = float(item.get("probability", probability) or 0.0)
    return float(np.clip(probability, 0.0, 1.0))


def _regularize_edge_attr(edge_attr: np.ndarray, names: List[str], edge_features: Dict[str, Any] | None) -> np.ndarray:
    cfg = dict((edge_features or {}).get("prior_regularization", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return edge_attr
    probability = _edge_regularization_probability(cfg)
    if probability <= 0.0:
        return edge_attr
    seed = int(cfg.get("rng_seed", cfg.get("seed", 137)) or 137)
    rng = np.random.default_rng(seed)
    if float(rng.random()) >= probability:
        return edge_attr
    target_features = [str(name) for name in cfg.get("target_features") or ["part_similarity", "same_dominant_part"]]
    target_indices = [names.index(name) for name in target_features if name in names]
    if not target_indices:
        return edge_attr
    mode = str(cfg.get("mode", "dropout"))
    neutral_values = dict(cfg.get("neutral_values") or {})
    out = np.array(edge_attr, copy=True)
    for idx in target_indices:
        name = names[idx]
        neutral = float(neutral_values.get(name, 0.0))
        if mode == "dropout":
            out[:, idx] = neutral
        elif mode == "attenuate":
            keep = float(cfg.get("keep", 0.5))
            keep = float(np.clip(keep, 0.0, 1.0))
            out[:, idx] = out[:, idx] * keep + neutral * (1.0 - keep)
        else:
            raise ValueError(f"Unsupported D16 edge prior regularization mode={mode!r}")
    return out.astype(np.float32)


def _build_edge_attr(
    x: np.ndarray,
    pos: np.ndarray,
    part_soft: np.ndarray,
    edges: np.ndarray,
    feature_names: Iterable[str] | None = None,
    edge_features: Dict[str, Any] | None = None,
    node_feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    names = [str(name) for name in feature_names] if feature_names else _edge_feature_names(edge_features)
    src = edges[0].astype(np.int64)
    dst = edges[1].astype(np.int64)
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    spatial = np.sqrt(dx * dx + dy * dy)
    node_names = list(node_feature_names or [])
    node_index = {name: idx for idx, name in enumerate(node_names)}

    def column(name: str) -> np.ndarray | None:
        idx = node_index.get(name)
        if idx is None or idx >= x.shape[1]:
            return None
        return x[:, idx]

    grad_mag = column("grad_mag")
    if grad_mag is None and x.shape[1] > 2:
        grad_mag = np.sqrt(x[:, 1] * x[:, 1] + x[:, 2] * x[:, 2]).astype(np.float32)
    laplacian_abs = column("laplacian_abs")

    parts_src = part_soft[src]
    parts_dst = part_soft[dst]
    part_dot = np.sum(parts_src * parts_dst, axis=1)
    part_norm = np.linalg.norm(parts_src, axis=1) * np.linalg.norm(parts_dst, axis=1)
    part_similarity = part_dot / np.maximum(part_norm, 1e-6)
    same_part = (np.argmax(parts_src, axis=1) == np.argmax(parts_dst, axis=1)).astype(np.float32)
    feature_map = {
        "dx": dx,
        "dy": dy,
        "spatial_dist": spatial,
        "abs_intensity_diff": np.abs(x[src, 0] - x[dst, 0]),
        "abs_grad_mag_diff": np.abs(grad_mag[src] - grad_mag[dst]) if grad_mag is not None else np.zeros_like(dx),
        "abs_laplacian_diff": np.abs(laplacian_abs[src] - laplacian_abs[dst]) if laplacian_abs is not None else np.zeros_like(dx),
        "part_similarity": np.clip(part_similarity, 0.0, 1.0),
        "same_dominant_part": same_part,
    }
    missing = [name for name in names if name not in feature_map]
    if missing:
        raise ValueError(f"Unsupported D16 edge feature names: {missing}")
    edge_attr = np.stack([feature_map[name] for name in names], axis=1).astype(np.float32)
    edge_attr = _regularize_edge_attr(edge_attr, names, edge_features)
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=-1.0)
    return edge_attr


def build_pixel_graph(
    prior: Dict[str, np.ndarray],
    graph_mode: str = "face_plus_context",
    face_threshold: float = 0.15,
    context_pixels: int = 2,
    detail_features: Dict[str, Any] | None = None,
    edge_features: Dict[str, Any] | None = None,
    anchor_nodes: Dict[str, Any] | None = None,
    node_features: Dict[str, Any] | None = None,
    prior_usage: str | None = None,
) -> D16GraphData:
    image = np.asarray(prior["image_48"], dtype=np.float32)
    image_norm = image / 255.0 if image.max() > 1.0 else image
    face = np.asarray(prior["face_mask"], dtype=np.float32)
    part_masks = np.asarray(prior["part_soft_masks"], dtype=np.float32)
    distance_maps = np.asarray(prior["distance_maps"], dtype=np.float32)
    missing = float(np.asarray(prior["landmark_missing_flag"]).item())
    mask = _node_mask(face, graph_mode=graph_mode, threshold=face_threshold, context_pixels=context_pixels)
    coords, edges = _edges_for_mask(mask)
    yy = coords[:, 0]
    xx = coords[:, 1]
    gx, gy = _gradients(image_norm)
    x_norm = (xx.astype(np.float32) / 47.0) * 2.0 - 1.0
    y_norm = (yy.astype(np.float32) / 47.0) * 2.0 - 1.0
    feature_flags = _node_feature_flags(node_features, prior_usage=prior_usage)
    features = [
        image_norm[yy, xx][:, None],
        gx[yy, xx][:, None],
        gy[yy, xx][:, None],
        x_norm[:, None],
        y_norm[:, None],
    ]
    node_feature_names = ["intensity", "gx", "gy", "x_norm", "y_norm"]
    if feature_flags["face"]:
        features.append(face[yy, xx][:, None])
        node_feature_names.append("face_mask")
    if feature_flags["part"]:
        features.append(np.transpose(part_masks[:, yy, xx], (1, 0)))
        node_feature_names.extend([f"part_soft_{idx}" for idx in range(part_masks.shape[0])])
    if feature_flags["distance"]:
        features.append(np.transpose(distance_maps[:, yy, xx], (1, 0)))
        node_feature_names.extend([f"distance_map_{idx}" for idx in range(distance_maps.shape[0])])
    if feature_flags["missing"]:
        features.append(np.full((len(coords), 1), missing, dtype=np.float32))
        node_feature_names.append("landmark_missing_flag")
    if _detail_features_enabled(detail_features):
        detail_names = list(detail_features.get("features") or DEFAULT_DETAIL_FEATURES)
        detail_x, detail_names = sample_detail_features(
            image_norm,
            yy,
            xx,
            feature_names=detail_names,
            normalize=str(detail_features.get("normalize", "per_image_safe")),
        )
        features.append(detail_x)
        node_feature_names.extend(detail_names)
    x = np.concatenate(features, axis=1).astype(np.float32)
    pos = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    part_soft_sampled = np.transpose(part_masks[:, yy, xx], (1, 0)).astype(np.float32)
    face_sampled = face[yy, xx].astype(np.float32)
    x, pos, part_soft_sampled, face_sampled, edges = _add_part_anchor_nodes(
        x,
        pos,
        part_soft_sampled,
        face_sampled,
        edges,
        anchor_nodes,
    )
    edge_attr = None
    if _edge_features_enabled(edge_features):
        edge_attr = _build_edge_attr(
            x=x,
            pos=pos,
            part_soft=part_soft_sampled,
            edges=edges,
            feature_names=edge_features.get("features"),
            edge_features=edge_features,
            node_feature_names=node_feature_names,
        )
    return D16GraphData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edges).long(),
        edge_attr=None if edge_attr is None else torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos),
        y=torch.tensor(int(np.asarray(prior["label"]).item()), dtype=torch.long),
        sample_index=torch.tensor(int(np.asarray(prior["sample_index"]).item()), dtype=torch.long),
        part_soft=torch.from_numpy(part_soft_sampled),
        face_mask=torch.from_numpy(face_sampled.astype(np.float32)),
        valid_part_mask=torch.from_numpy(np.asarray(prior["valid_part_mask"], dtype=np.float32)),
        valid_anchor_mask=torch.from_numpy(np.asarray(prior["valid_anchor_mask"], dtype=np.float32)),
        detected=torch.tensor(bool(np.asarray(prior["detected"]).item()), dtype=torch.bool),
        landmark_missing_flag=torch.tensor(int(np.asarray(prior["landmark_missing_flag"]).item()), dtype=torch.long),
        image_48=torch.from_numpy(image_norm.astype(np.float32)),
        node_feature_names=list(node_feature_names),
        edge_feature_names=_edge_feature_names(edge_features) if _edge_features_enabled(edge_features) else None,
    )


def collate_d16_graphs(graphs: Iterable[D16GraphData]) -> D16Batch:
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot collate empty D16 graph batch")
    xs: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    batch_index: List[torch.Tensor] = []
    ptr = [0]
    pos: List[torch.Tensor] = []
    part_soft: List[torch.Tensor] = []
    face: List[torch.Tensor] = []
    ys, sample_indices, valid_parts, valid_anchors, detected, missing = [], [], [], [], [], []
    images: List[torch.Tensor] = []
    offset = 0
    for batch_id, graph in enumerate(graphs):
        n = int(graph.x.size(0))
        xs.append(graph.x)
        edges.append(graph.edge_index + offset)
        if graph.edge_attr is not None:
            edge_attrs.append(graph.edge_attr)
        batch_index.append(torch.full((n,), batch_id, dtype=torch.long))
        pos.append(graph.pos)
        part_soft.append(graph.part_soft)
        face.append(graph.face_mask)
        ys.append(graph.y)
        sample_indices.append(graph.sample_index)
        valid_parts.append(graph.valid_part_mask)
        valid_anchors.append(graph.valid_anchor_mask)
        detected.append(graph.detected)
        missing.append(graph.landmark_missing_flag)
        images.append(graph.image_48.float())
        offset += n
        ptr.append(offset)
    return D16Batch(
        x_cat=torch.cat(xs, dim=0),
        edge_index_cat=torch.cat(edges, dim=1),
        edge_attr_cat=torch.cat(edge_attrs, dim=0) if len(edge_attrs) == len(graphs) else None,
        batch_index=torch.cat(batch_index, dim=0),
        ptr=torch.tensor(ptr, dtype=torch.long),
        y=torch.stack(ys).long(),
        sample_index=torch.stack(sample_indices).long(),
        pos_cat=torch.cat(pos, dim=0),
        part_soft_cat=torch.cat(part_soft, dim=0),
        face_mask_cat=torch.cat(face, dim=0),
        valid_part_mask=torch.stack(valid_parts).float(),
        valid_anchor_mask=torch.stack(valid_anchors).float(),
        detected=torch.stack(detected).bool(),
        landmark_missing_flag=torch.stack(missing).long(),
        image_48=torch.stack(images).float(),
        node_feature_names=list(graphs[0].node_feature_names or []) or None,
        edge_feature_names=list(graphs[0].edge_feature_names or []) or None,
    )

