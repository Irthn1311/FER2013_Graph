"""All 2304 pixels with the reference directed 8-neighborhood ordering."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from lap_gnn_tf.graph.builder import _edges_for_mask, _gradients
from lap_gnn_tf.graph.features import _laplacian_abs

NODE_FEATURE_NAMES = ["intensity", "x_norm", "y_norm", "gx", "gy"]
EDGE_FEATURE_NAMES = [
    "dx", "dy", "spatial_dist", "abs_intensity_diff",
    "abs_grad_mag_diff", "abs_laplacian_diff",
]


@dataclass
class PixelGraph:
    nodes: np.ndarray
    edges: np.ndarray
    edge_features: np.ndarray
    image: np.ndarray
    label: int
    sample_id: int


def build_pixel_graph(image, label: int = 0, sample_id: int = 0):
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (48, 48) or not np.isfinite(image).all():
        raise ValueError("Expected a finite 48x48 grayscale image")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Pixel intensity must lie in [0,255] or [0,1]")
    image = image / 255.0 if image.max() > 1.0 else image.copy()
    coords, edges = _edges_for_mask(np.ones((48, 48), dtype=bool))
    yy, xx = coords.T
    gx, gy = _gradients(image)
    pos = np.stack([xx / 47.0 * 2.0 - 1.0, yy / 47.0 * 2.0 - 1.0], axis=1).astype(np.float32)
    nodes = np.column_stack([image[yy, xx], pos, gx[yy, xx], gy[yy, xx]]).astype(np.float32)
    src, dst = edges
    delta = pos[dst] - pos[src]
    grad_mag = np.clip(np.sqrt(gx * gx + gy * gy), 0.0, 1.0)[yy, xx]
    laplacian = np.clip(_laplacian_abs(image), 0.0, 1.0)[yy, xx]
    edge_features = np.column_stack([
        delta, np.linalg.norm(delta, axis=1),
        np.abs(nodes[src, 0] - nodes[dst, 0]),
        np.abs(grad_mag[src] - grad_mag[dst]),
        np.abs(laplacian[src] - laplacian[dst]),
    ]).astype(np.float32)
    return PixelGraph(nodes, edges.copy(), edge_features, image, int(label), int(sample_id))


def collate_pixel_graphs(graphs):
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot collate an empty pixel batch")
    return {
        "node_features": tf.convert_to_tensor(np.concatenate([g.nodes for g in graphs]), tf.float32),
        "edge_index": tf.convert_to_tensor(np.concatenate([g.edges + i * 2304 for i, g in enumerate(graphs)], axis=1), tf.int64),
        "edge_features": tf.convert_to_tensor(np.concatenate([g.edge_features for g in graphs]), tf.float32),
        "node_graph_index": tf.repeat(tf.range(len(graphs), dtype=tf.int64), 2304),
        "graph_node_counts": tf.fill([len(graphs)], tf.constant(2304, tf.int64)),
        "graph_edge_counts": tf.convert_to_tensor([g.edges.shape[1] for g in graphs], tf.int64),
        "labels": tf.convert_to_tensor([g.label for g in graphs], tf.int64),
        "sample_ids": tf.convert_to_tensor([g.sample_id for g in graphs], tf.int64),
        "image_48": tf.convert_to_tensor(np.stack([g.image for g in graphs]), tf.float32),
    }
