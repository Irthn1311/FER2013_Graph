"""Graph indexing and fixed spatial topology builders for Pure-GNN v3.1."""

import math
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf


DIRECTION_TUPLES = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

DIRECTION_TO_ID = {d: idx for idx, d in enumerate(DIRECTION_TUPLES)}


def build_grid_8neighbor_graph(height: int, width: int) -> Dict[str, tf.Tensor]:
    """Builds a fixed 8-neighbor directed grid graph for H x W image lattice.
    
    Receiver is node i, sender is neighbor j in N(i).
    There are NO virtual/reflected/zero-padded neighbors.
    Degree normalization uses the real degree (3 corner, 5 edge, 8 interior).
    
    Returns a dictionary of TensorFlow tensors:
        - src_indices: (E,) int32, indices of neighbor nodes j
        - dst_indices: (E,) int32, indices of receiver nodes i
        - delta_x: (E, 1) float32, c_j - c_i
        - delta_y: (E, 1) float32, r_j - r_i
        - direction_id: (E,) int32 in {0..7}
        - degrees: (N, 1) float32, degree of each node i
        - num_nodes: int, N = H * W
        - num_edges: int, total directed edges E
    """
    num_nodes = height * width
    src_list = []
    dst_list = []
    dx_list = []
    dy_list = []
    dir_list = []
    degrees = np.zeros((num_nodes,), dtype=np.float32)

    for r in range(height):
        for c in range(width):
            i = r * width + c
            deg = 0
            for dr, dc in DIRECTION_TUPLES:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    j = nr * width + nc
                    src_list.append(j)
                    dst_list.append(i)
                    dx_list.append(float(dc))
                    dy_list.append(float(dr))
                    dir_list.append(DIRECTION_TO_ID[(dr, dc)])
                    deg += 1
            degrees[i] = deg

    src_arr = np.array(src_list, dtype=np.int32)
    dst_arr = np.array(dst_list, dtype=np.int32)
    dx_arr = np.array(dx_list, dtype=np.float32)[:, None]
    dy_arr = np.array(dy_list, dtype=np.float32)[:, None]
    dir_arr = np.array(dir_list, dtype=np.int32)
    deg_arr = degrees[:, None]

    return {
        "src_indices": tf.constant(src_arr, dtype=tf.int32),
        "dst_indices": tf.constant(dst_arr, dtype=tf.int32),
        "delta_x": tf.constant(dx_arr, dtype=tf.float32),
        "delta_y": tf.constant(dy_arr, dtype=tf.float32),
        "direction_id": tf.constant(dir_arr, dtype=tf.int32),
        "degrees": tf.constant(deg_arr, dtype=tf.float32),
        "num_nodes": tf.constant(num_nodes, dtype=tf.int32),
        "num_edges": tf.constant(len(src_list), dtype=tf.int32),
    }


def build_complete_coarse_graph(grid_size: int = 6) -> Dict[str, tf.Tensor]:
    """Builds a complete directed graph among grid_size x grid_size nodes, excluding self-loops.
    
    For grid_size=6:
    N = 36
    E = 36 * 35 = 1260 directed edges.
    
    Receiver is i, sender is j != i.
    
    Geometry p_ij = [delta_x, delta_y, spatial_distance]:
        delta_x = (c_j - c_i) / (grid_size - 1)
        delta_y = (r_j - r_i) / (grid_size - 1)
        dist = sqrt(delta_x^2 + delta_y^2)
    """
    num_nodes = grid_size * grid_size
    src_list = []
    dst_list = []
    p_ij_list = []

    norm_scale = float(grid_size - 1) if grid_size > 1 else 1.0

    for i in range(num_nodes):
        r_i, c_i = divmod(i, grid_size)
        for j in range(num_nodes):
            if i == j:
                continue
            r_j, c_j = divmod(j, grid_size)
            dx = (c_j - c_i) / norm_scale
            dy = (r_j - r_i) / norm_scale
            dist = math.sqrt(dx * dx + dy * dy)
            src_list.append(j)
            dst_list.append(i)
            p_ij_list.append([dx, dy, dist])

    src_arr = np.array(src_list, dtype=np.int32)
    dst_arr = np.array(dst_list, dtype=np.int32)
    p_ij_arr = np.array(p_ij_list, dtype=np.float32)

    return {
        "src_indices": tf.constant(src_arr, dtype=tf.int32),
        "dst_indices": tf.constant(dst_arr, dtype=tf.int32),
        "p_ij": tf.constant(p_ij_arr, dtype=tf.float32),
        "num_nodes": tf.constant(num_nodes, dtype=tf.int32),
        "num_edges": tf.constant(len(src_list), dtype=tf.int32),
        "edges_per_node": tf.constant(num_nodes - 1, dtype=tf.int32),
    }
