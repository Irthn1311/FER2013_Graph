"""Generic grid and learned-relation graph operations for CF-HPG v1.0."""

from __future__ import annotations

import tensorflow as tf


RELATION_TOP_K = 4


def normalized_grid_centers(grid_size: int, dtype=tf.float32) -> tf.Tensor:
    """Return row-major (x, y) cell-center coordinates in [-1, 1]."""

    axis = (tf.cast(tf.range(grid_size), dtype) + 0.5) * (2.0 / grid_size) - 1.0
    y, x = tf.meshgrid(axis, axis, indexing="ij")
    return tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=-1)


def spatial_8_neighbor_adjacency(grid_size: int) -> tf.Tensor:
    """Return directed row-major 8-neighborhood adjacency without self edges."""

    node_count = grid_size * grid_size
    rows = []
    for source in range(node_count):
        source_y, source_x = divmod(source, grid_size)
        row = []
        for target in range(node_count):
            target_y, target_x = divmod(target, grid_size)
            row.append(
                source != target
                and abs(source_y - target_y) <= 1
                and abs(source_x - target_x) <= 1
            )
        rows.append(row)
    return tf.constant(rows, dtype=tf.bool)


def learned_relation_adjacency(
    nodes: tf.Tensor, top_k: int = RELATION_TOP_K
) -> tf.Tensor:
    """Build directed cosine top-k adjacency independently for each batch item."""

    nodes = tf.convert_to_tensor(nodes)
    tf.debugging.assert_rank(nodes, 3)
    node_count = tf.shape(nodes)[1]
    tf.debugging.assert_greater(node_count, top_k)
    normalized = tf.math.l2_normalize(nodes, axis=-1)
    similarity = tf.matmul(normalized, normalized, transpose_b=True)
    batch_size = tf.shape(nodes)[0]
    self_mask = tf.eye(node_count, batch_shape=[batch_size], dtype=tf.bool)
    similarity = tf.where(
        self_mask,
        tf.fill(tf.shape(similarity), tf.cast(-float("inf"), similarity.dtype)),
        similarity,
    )
    neighbor_indices = tf.math.top_k(similarity, k=top_k, sorted=True).indices
    adjacency = tf.reduce_any(
        tf.one_hot(neighbor_indices, depth=node_count, dtype=tf.int32) > 0,
        axis=-2,
    )
    return tf.logical_and(adjacency, tf.logical_not(self_mask))


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class HybridGraphBuilder(tf.keras.layers.Layer):
    """Union fixed generic-grid adjacency with learned cosine relations."""

    def __init__(self, grid_size: int, top_k: int = RELATION_TOP_K, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.grid_size = int(grid_size)
        self.top_k = int(top_k)
        self._spatial = spatial_8_neighbor_adjacency(self.grid_size)

    def call(self, nodes: tf.Tensor) -> tf.Tensor:
        nodes = tf.convert_to_tensor(nodes)
        tf.debugging.assert_equal(tf.shape(nodes)[1], self.grid_size**2)
        learned = learned_relation_adjacency(nodes, self.top_k)
        spatial = tf.broadcast_to(self._spatial[None, :, :], tf.shape(learned))
        return tf.logical_or(spatial, learned)

    def get_config(self):
        config = super().get_config()
        config.update({"grid_size": self.grid_size, "top_k": self.top_k})
        return config


def max_relative_aggregate(nodes: tf.Tensor, adjacency: tf.Tensor) -> tf.Tensor:
    """Compute max_j(h_j - h_i) over each registered neighbor set."""

    nodes = tf.convert_to_tensor(nodes)
    adjacency = tf.cast(tf.convert_to_tensor(adjacency), tf.bool)
    tf.debugging.assert_rank(nodes, 3)
    tf.debugging.assert_rank(adjacency, 3)
    tf.debugging.assert_equal(tf.shape(adjacency)[0], tf.shape(nodes)[0])
    tf.debugging.assert_equal(tf.shape(adjacency)[1], tf.shape(nodes)[1])
    tf.debugging.assert_equal(tf.shape(adjacency)[2], tf.shape(nodes)[1])
    tf.debugging.assert_equal(
        tf.reduce_all(tf.reduce_any(adjacency, axis=-1)),
        True,
        message="Every node requires at least one neighbor",
    )
    differences = nodes[:, None, :, :] - nodes[:, :, None, :]
    masked = tf.where(
        adjacency[:, :, :, None],
        differences,
        tf.fill(tf.shape(differences), tf.cast(-float("inf"), nodes.dtype)),
    )
    return tf.reduce_max(masked, axis=2)
