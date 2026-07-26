"""TensorFlow conversion for flat NumPy graph batches."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lap_gnn_tf.graph.builder import D16Batch


def to_tensor_dict(batch: D16Batch | dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    arrays = batch.as_tensor_dict() if isinstance(batch, D16Batch) else batch
    return {
        key: tf.convert_to_tensor(value)
        for key, value in arrays.items()
        if value is not None
    }


def load_golden_batch(path: str) -> dict[str, tf.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        mapping = {
            "node_features": data["node_features"],
            "edge_index": data["edge_index"],
            "edge_features": data["edge_features"],
            "node_types": data["node_types"],
            "node_graph_index": data["batch_index"],
            "edge_graph_index": data["batch_index"][data["edge_index"][1]],
            "graph_node_counts": data["node_counts"],
            "graph_edge_counts": data["edge_counts"],
            "labels": np.load(path.replace("graph_batch.npz", "labels.npy"), allow_pickle=False),
            "sample_ids": data["sample_ids"],
            "coordinates": data["positions"],
            "part_soft": data["part_soft"],
            "face_mask": data["face_mask"],
            "anchor_mask": data["anchor_mask"],
            "valid_part_mask": data["valid_part_mask"],
            "valid_anchor_mask": data["valid_anchor_mask"],
            "detected": data["detected"],
            "landmark_missing_flag": data["landmark_missing_flag"],
            "image_48": data["image_48"],
        }
    return {key: tf.convert_to_tensor(value) for key, value in mapping.items()}
