"""Static 48x48 grid topology and 8-neighborhood geometric precomputations.

On a 48x48 image, pixel graph connectivity is fixed and identical for every sample:
- 2304 nodes
- 8 neighbors per pixel (with boundary masking)
- Static geometric edge features (dx, dy, distance)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def precompute_grid_topology():
    """Compute 8-neighborhood indices, valid mask, and static geometric edge attributes.

    Returns:
        neighbors_idx: np.ndarray of shape [2304, 8], int32
            Indices of the 8 neighbors for each of the 2304 pixels.
            Invalid neighbors (out of bounds) are clamped to 0.
        neighbor_valid: np.ndarray of shape [2304, 8], bool
            True if the neighbor is strictly within the 48x48 image boundary.
        static_edge_features: np.ndarray of shape [2304, 8, 3], float32
            [dx, dy, spatial_distance] normalized to [-1, 1] coordinates.
        normalized_coords: np.ndarray of shape [2304, 2], float32
            [x_norm, y_norm] coordinates in [-1, 1].
    """
    height, width = 48, 48
    num_nodes = height * width

    # Offsets for 8-neighborhood: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # Node coordinates in normalized [-1, 1] range
    yy, xx = np.mgrid[0:height, 0:width]
    y_flat = yy.reshape(-1)
    x_flat = xx.reshape(-1)

    x_norm = (x_flat / 47.0) * 2.0 - 1.0
    y_norm = (y_flat / 47.0) * 2.0 - 1.0
    normalized_coords = np.stack([x_norm, y_norm], axis=1).astype(np.float32)

    neighbors_idx = np.zeros((num_nodes, 8), dtype=np.int32)
    neighbor_valid = np.zeros((num_nodes, 8), dtype=bool)
    static_edge_features = np.zeros((num_nodes, 8, 3), dtype=np.float32)

    for i in range(num_nodes):
        r, c = y_flat[i], x_flat[i]
        src_pos = normalized_coords[i]

        for k, (dr, dc) in enumerate(neighbor_offsets):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbor_idx = nr * width + nc
                neighbors_idx[i, k] = neighbor_idx
                neighbor_valid[i, k] = True

                dst_pos = normalized_coords[neighbor_idx]
                delta = dst_pos - src_pos
                dist = np.linalg.norm(delta)
                static_edge_features[i, k] = [delta[0], delta[1], dist]
            else:
                # Clamp out-of-bounds to node 0; mask is False so attention will zero this out
                neighbors_idx[i, k] = 0
                neighbor_valid[i, k] = False
                static_edge_features[i, k] = [0.0, 0.0, 0.0]

    return (
        neighbors_idx,
        neighbor_valid,
        static_edge_features,
        normalized_coords,
    )


class StaticGridTopology:
    """Singleton-like container holding constant TensorFlow tensors for the 48x48 grid."""

    _instance = None

    def __init__(self):
        neighbors_idx, neighbor_valid, static_edge, coords = precompute_grid_topology()
        self.neighbors_idx = tf.constant(neighbors_idx, dtype=tf.int32)
        self.neighbor_valid = tf.constant(neighbor_valid, dtype=tf.bool)
        self.static_edge_features = tf.constant(static_edge, dtype=tf.float32)
        self.normalized_coords = tf.constant(coords, dtype=tf.float32)

    @classmethod
    def get_instance(cls) -> StaticGridTopology:
        if cls._instance is None:
            cls._instance = StaticGridTopology()
        return cls._instance
