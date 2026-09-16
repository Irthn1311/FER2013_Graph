"""Static 48x48 grid topology and 8-neighborhood geometric precomputations.

On a 48x48 image, pixel graph connectivity is fixed and identical for every sample:
- 2304 nodes
- 8 neighbors per pixel (with boundary masking)
- 17,860 directed edges per image
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def precompute_grid_topology():
    height, width = 48, 48
    num_nodes = height * width

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    yy, xx = np.mgrid[0:height, 0:width]
    y_flat = yy.reshape(-1)
    x_flat = xx.reshape(-1)

    x_norm = (x_flat / 47.0) * 2.0 - 1.0
    y_norm = (y_flat / 47.0) * 2.0 - 1.0
    normalized_coords = np.stack([x_norm, y_norm], axis=1).astype(np.float32)

    neighbors_idx = np.zeros((num_nodes, 8), dtype=np.int32)
    neighbor_valid = np.zeros((num_nodes, 8), dtype=bool)
    static_edge_features = np.zeros((num_nodes, 8, 3), dtype=np.float32)

    src_list, dst_list, edge_attr_list = [], [], []

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

                src_list.append(i)
                dst_list.append(neighbor_idx)
                # 6-dim edge attr: [dx, dy, dist, 0, 0, 0] base
                edge_attr_list.append([delta[0], delta[1], dist, 0.0, 0.0, 0.0])
            else:
                neighbors_idx[i, k] = 0
                neighbor_valid[i, k] = False
                static_edge_features[i, k] = [0.0, 0.0, 0.0]

    single_image_edges = np.stack([src_list, dst_list], axis=0).astype(np.int64)  # [2, 17860]
    single_image_edge_attr = np.array(edge_attr_list, dtype=np.float32)            # [17860, 6]

    return (
        neighbors_idx,
        neighbor_valid,
        static_edge_features,
        normalized_coords,
        single_image_edges,
        single_image_edge_attr,
    )


class StaticGridTopology:
    _instance = None

    def __init__(self):
        (
            neighbors_idx,
            neighbor_valid,
            static_edge,
            coords,
            edges,
            edge_attr,
        ) = precompute_grid_topology()
        self.neighbors_idx = tf.constant(neighbors_idx, dtype=tf.int32)
        self.neighbor_valid = tf.constant(neighbor_valid, dtype=tf.bool)
        self.static_edge_features = tf.constant(static_edge, dtype=tf.float32)
        self.normalized_coords = tf.constant(coords, dtype=tf.float32)
        self.single_image_edges = tf.constant(edges, dtype=tf.int64)
        self.single_image_edge_attr = tf.constant(edge_attr, dtype=tf.float32)

    @classmethod
    def get_instance(cls) -> StaticGridTopology:
        if cls._instance is None:
            cls._instance = StaticGridTopology()
        return cls._instance

    def get_flat_batch_edges(self, batch_size):
        """Construct [2, B * 17860] and [B * 17860, 6] for batch."""
        edges = self.single_image_edges
        edge_attr = self.single_image_edge_attr
        num_edges_per_img = tf.shape(edges)[1]

        batch_edges = []
        for i in range(batch_size):
            offset = tf.cast(i * 2304, tf.int64)
            batch_edges.append(edges + offset)

        batched_edge_index = tf.concat(batch_edges, axis=1)
        batched_edge_attr = tf.tile(edge_attr, [batch_size, 1])
        return batched_edge_index, batched_edge_attr
