"""On-disk graph cache helpers for D18 structure-guided graphs.

The cache stores already-built graph tensors so Kaggle training does not rebuild
node support, local edges, kNN edges, and structure relation edges every epoch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d18.data.structure_graph_builder import (
    BASE_EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    STRUCTURE_EDGE_FEATURE_NAMES,
    D18GraphData,
)


def graph_cache_path(cache_dir: str | Path, split: str, prior_file: str | Path) -> Path:
    return Path(cache_dir) / str(split) / Path(prior_file).name


EVIDENCE_CACHE_SCHEMA = "d19_a0_evidence_only_v1"


def evidence_cache_signature_payload(graph_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return only image/evidence graph fields that define an A0 cache entry."""
    cfg = dict(graph_cfg or {})
    if str(cfg.get("graph_mode")) != "evidence_only":
        raise ValueError("Evidence cache signature requires graph_mode=evidence_only")
    return {
        "cache_schema": EVIDENCE_CACHE_SCHEMA,
        "graph_builder": "d18.structure_graph_builder.evidence_only.v1",
        "image_preprocessing": "fer48_float32_clip01_v1",
        "node_selection": {
            "node_support_mode": cfg.get("node_support_mode", "stratified_detail_knn"),
            "target_node_count": int(cfg.get("target_node_count", 1800)),
            "bins": int(cfg.get("bins", 6)),
        },
        "node_feature_schema": list(NODE_FEATURE_NAMES),
        "local_edges": cfg.get("local_edges", {}) or {},
        "knn_edges": cfg.get("knn_edges", {}) or {},
        "edge_attribute_schema": {
            "name": str(cfg.get("edge_schema", "base6")),
            "features": list(BASE_EDGE_FEATURE_NAMES),
        },
        "merge": {
            "precedence": ["local", "knn"],
            "deduplicate_directed_endpoints": True,
            "self_loops_added": False,
            "version": "local_gt_knn_v1",
        },
    }


def evidence_cache_signature(graph_cfg: dict[str, Any]) -> str:
    encoded = json.dumps(
        evidence_cache_signature_payload(graph_cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evidence_image_hash(image_48: np.ndarray, label: int) -> str:
    image = np.asarray(image_48, dtype=np.float32)
    payload = image.tobytes(order="C") + int(label).to_bytes(2, "little", signed=True)
    return hashlib.sha256(payload).hexdigest()


def evidence_graph_cache_path(
    cache_dir: str | Path,
    split: str,
    sample_index: int,
    image_48: np.ndarray,
    label: int,
    graph_cfg: dict[str, Any],
) -> Path:
    namespace = evidence_cache_signature(graph_cfg)[:16]
    content_hash = evidence_image_hash(image_48, label)[:16]
    return Path(cache_dir) / namespace / str(split) / f"{int(sample_index):06d}_{content_hash}.npz"


def _edge_names_for_dim(edge_dim: int) -> list[str]:
    if int(edge_dim) == len(STRUCTURE_EDGE_FEATURE_NAMES):
        return list(STRUCTURE_EDGE_FEATURE_NAMES)
    return list(BASE_EDGE_FEATURE_NAMES)


def save_d18_graph_cache(graph: D18GraphData, path: str | Path, compressed: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "x": graph.x.detach().cpu().numpy().astype(np.float16),
        "edge_index": graph.edge_index.detach().cpu().numpy().astype(np.uint16),
        "edge_attr": graph.edge_attr.detach().cpu().numpy().astype(np.float16),
        "pos": graph.pos.detach().cpu().numpy().astype(np.float16),
        "y": np.asarray(int(graph.y), dtype=np.int16),
        "sample_index": np.asarray(int(graph.sample_index), dtype=np.int64),
        "detected": np.asarray(bool(graph.detected), dtype=np.bool_),
        "landmark_missing_flag": np.asarray(int(graph.landmark_missing_flag), dtype=np.int8),
        "image_48": graph.image_48.detach().cpu().numpy().astype(np.float16),
        "edge_type": graph.edge_type.detach().cpu().numpy().astype(np.int8),
        "structure_relation_id": graph.structure_relation_id.detach().cpu().numpy().astype(np.int16),
        "local_edge_count": np.asarray(int(graph.local_edge_count), dtype=np.int32),
        "knn_edge_count": np.asarray(int(graph.knn_edge_count), dtype=np.int32),
        "structure_edge_count": np.asarray(int(graph.structure_edge_count), dtype=np.int32),
        "total_edge_count": np.asarray(int(graph.total_edge_count), dtype=np.int32),
        "node_support_mode": np.asarray(str(graph.node_support_mode)),
        "edge_feature_names": np.asarray(graph.edge_feature_names),
        "structure_edge_count_before_purification": np.asarray(int(graph.structure_edge_count_before_purification), dtype=np.int32),
        "structure_edge_count_after_purification": np.asarray(int(graph.structure_edge_count_after_purification), dtype=np.int32),
        "purification_compatibility_kept_mean": np.asarray(float(graph.purification_compatibility_kept_mean), dtype=np.float32),
        "purification_compatibility_dropped_mean": np.asarray(float(graph.purification_compatibility_dropped_mean), dtype=np.float32),
    }
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def load_d18_graph_cache(path: str | Path) -> D18GraphData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        edge_index = data["edge_index"].astype(np.int64, copy=False)
        edge_attr = data["edge_attr"].astype(np.float32, copy=False)
        if "edge_feature_names" in data.files:
            edge_feature_names = [str(x) for x in data["edge_feature_names"].tolist()]
        else:
            edge_feature_names = _edge_names_for_dim(edge_attr.shape[1])
        if "edge_type" in data.files:
            edge_type = data["edge_type"].astype(np.int64, copy=False)
        else:
            edge_type = np.zeros((edge_index.shape[1],), dtype=np.int64)
            if edge_attr.shape[1] >= 9:
                edge_type[edge_attr[:, 6] > 0.0] = 2
        if "structure_relation_id" in data.files:
            relation_id = data["structure_relation_id"].astype(np.int64, copy=False)
        else:
            relation_id = np.zeros((edge_index.shape[1],), dtype=np.int64)
            if edge_attr.shape[1] >= 9:
                relation_id = np.rint(edge_attr[:, 6] * 6).astype(np.int64)
        return D18GraphData(
            x=torch.from_numpy(data["x"].astype(np.float32, copy=False)),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr),
            pos=torch.from_numpy(data["pos"].astype(np.float32, copy=False)),
            y=torch.tensor(int(data["y"]), dtype=torch.long),
            sample_index=torch.tensor(int(data["sample_index"]), dtype=torch.long),
            detected=torch.tensor(bool(data["detected"]), dtype=torch.bool),
            landmark_missing_flag=torch.tensor(int(data["landmark_missing_flag"]), dtype=torch.long),
            image_48=torch.from_numpy(data["image_48"].astype(np.float32, copy=False)),
            edge_type=torch.from_numpy(edge_type).long(),
            structure_relation_id=torch.from_numpy(relation_id).long(),
            node_feature_names=list(NODE_FEATURE_NAMES),
            edge_feature_names=edge_feature_names,
            local_edge_count=int(data["local_edge_count"]),
            knn_edge_count=int(data["knn_edge_count"]),
            structure_edge_count=int(data["structure_edge_count"]) if "structure_edge_count" in data.files else 0,
            total_edge_count=int(data["total_edge_count"]),
            structure_edge_count_before_purification=int(data["structure_edge_count_before_purification"]) if "structure_edge_count_before_purification" in data.files else int(data["structure_edge_count"]) if "structure_edge_count" in data.files else 0,
            structure_edge_count_after_purification=int(data["structure_edge_count_after_purification"]) if "structure_edge_count_after_purification" in data.files else int(data["structure_edge_count"]) if "structure_edge_count" in data.files else 0,
            purification_compatibility_kept_mean=float(data["purification_compatibility_kept_mean"]) if "purification_compatibility_kept_mean" in data.files else float("nan"),
            purification_compatibility_dropped_mean=float(data["purification_compatibility_dropped_mean"]) if "purification_compatibility_dropped_mean" in data.files else float("nan"),
            node_support_mode=str(data["node_support_mode"]) if "node_support_mode" in data.files else "cached",
        )
