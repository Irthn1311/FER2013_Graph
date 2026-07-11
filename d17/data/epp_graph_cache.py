"""On-disk graph cache helpers for D17 EPP graphs.

The cache stores already-built graph tensors so Kaggle training does not rebuild
node support, local edges, and kNN edges every epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from d17.data.epp_graph_builder import EPPGraphData


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


def graph_cache_path(cache_dir: str | Path, split: str, prior_file: str | Path) -> Path:
    return Path(cache_dir) / str(split) / Path(prior_file).name


def save_epp_graph_cache(graph: EPPGraphData, path: str | Path, compressed: bool = False) -> None:
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
        "local_edge_count": np.asarray(int(graph.local_edge_count), dtype=np.int32),
        "knn_edge_count": np.asarray(int(graph.knn_edge_count), dtype=np.int32),
        "total_edge_count": np.asarray(int(graph.total_edge_count), dtype=np.int32),
        "node_support_mode": np.asarray(str(graph.node_support_mode)),
    }
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def load_epp_graph_cache(path: str | Path) -> EPPGraphData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        edge_index = data["edge_index"].astype(np.int64, copy=False)
        return EPPGraphData(
            x=torch.from_numpy(data["x"].astype(np.float32, copy=False)),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(data["edge_attr"].astype(np.float32, copy=False)),
            y=torch.tensor(int(data["y"]), dtype=torch.long),
            sample_index=torch.tensor(int(data["sample_index"]), dtype=torch.long),
            pos=torch.from_numpy(data["pos"].astype(np.float32, copy=False)),
            detected=torch.tensor(bool(data["detected"]), dtype=torch.bool),
            landmark_missing_flag=torch.tensor(int(data["landmark_missing_flag"]), dtype=torch.long),
            image_48=torch.from_numpy(data["image_48"].astype(np.float32, copy=False)),
            local_edge_count=int(data["local_edge_count"]),
            knn_edge_count=int(data["knn_edge_count"]),
            total_edge_count=int(data["total_edge_count"]),
            node_feature_names=list(NODE_FEATURE_NAMES),
            edge_feature_names=list(EDGE_FEATURE_NAMES),
            node_support_mode=str(data["node_support_mode"]) if "node_support_mode" in data.files else "cached",
        )
