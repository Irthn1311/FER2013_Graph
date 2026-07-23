"""Portable golden-fixture loading for isolated validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lap_gnn.data.graph_builder import D16Batch


def load_golden_batch(package_root: str | Path) -> D16Batch:
    root = Path(package_root) / "validation_assets" / "golden"
    with np.load(root / "graph_batch.npz", allow_pickle=False) as data:
        labels = np.load(root / "labels.npy", allow_pickle=False)
        return D16Batch(
            x_cat=torch.from_numpy(data["node_features"].copy()).float(),
            edge_index_cat=torch.from_numpy(data["edge_index"].copy()).long(),
            edge_attr_cat=torch.from_numpy(data["edge_features"].copy()).float(),
            batch_index=torch.from_numpy(data["batch_index"].copy()).long(),
            ptr=torch.from_numpy(data["ptr"].copy()).long(),
            y=torch.from_numpy(labels.copy()).long(),
            sample_index=torch.from_numpy(data["sample_ids"].copy()).long(),
            pos_cat=torch.from_numpy(data["positions"].copy()).float(),
            part_soft_cat=torch.from_numpy(data["part_soft"].copy()).float(),
            face_mask_cat=torch.from_numpy(data["face_mask"].copy()).float(),
            valid_part_mask=torch.from_numpy(data["valid_part_mask"].copy()).float(),
            valid_anchor_mask=torch.from_numpy(data["valid_anchor_mask"].copy()).float(),
            detected=torch.from_numpy(data["detected"].copy()).bool(),
            landmark_missing_flag=torch.from_numpy(data["landmark_missing_flag"].copy()).long(),
            image_48=torch.from_numpy(data["image_48"].copy()).float(),
            node_feature_names=None,
            edge_feature_names=None,
        )


def load_portable_model_state(package_root: str | Path) -> dict[str, torch.Tensor]:
    path = Path(package_root) / "validation_assets" / "golden" / "model_state.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}
