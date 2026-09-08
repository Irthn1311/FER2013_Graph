"""Sparse edge-list graph construction for WS-HPG."""

from __future__ import annotations

import tensorflow as tf


def normalized_grid_coordinates(grid_size: int, dtype=tf.float32):
    axis = tf.linspace(tf.cast(-1.0, dtype), tf.cast(1.0, dtype), grid_size)
    y, x = tf.meshgrid(axis, axis, indexing="ij")
    return tf.stack([tf.reshape(x, [-1]), tf.reshape(y, [-1])], axis=-1)


def spatial_8_edge_index(grid_size: int):
    """Directed j->i valid 8-neighbor edges, represented as [source,dest]."""

    sources, destinations = [], []
    for destination in range(grid_size * grid_size):
        row, col = divmod(destination, grid_size)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                source_row, source_col = row + dy, col + dx
                if 0 <= source_row < grid_size and 0 <= source_col < grid_size:
                    sources.append(source_row * grid_size + source_col)
                    destinations.append(destination)
    return tf.constant([sources, destinations], tf.int32)


def batch_shared_edges(edge_index, batch_size, node_count):
    edge_index = tf.cast(edge_index, tf.int32)
    offsets = tf.range(batch_size, dtype=tf.int32)[:, None, None] * node_count
    expanded = edge_index[None, :, :] + offsets
    return tf.transpose(tf.reshape(tf.transpose(expanded, [0, 2, 1]), [-1, 2]))


def cosine_top_k_edges(nodes, top_k=4):
    """Return sparse batched cosine top-k edges; no self edges."""

    nodes = tf.convert_to_tensor(nodes)
    batch_size, node_count = tf.shape(nodes)[0], tf.shape(nodes)[1]
    normalized = tf.math.l2_normalize(nodes, axis=-1)
    similarity = tf.matmul(normalized, normalized, transpose_b=True)
    diagonal = tf.eye(node_count, batch_shape=[batch_size], dtype=tf.bool)
    similarity = tf.where(diagonal, tf.cast(-float("inf"), similarity.dtype), similarity)
    # [B,destination,k] source ids.
    sources = tf.math.top_k(similarity, k=top_k, sorted=True).indices
    destinations = tf.broadcast_to(
        tf.range(node_count, dtype=tf.int32)[None, :, None], tf.shape(sources)
    )
    offsets = tf.range(batch_size, dtype=tf.int32)[:, None, None] * node_count
    return tf.stack(
        [tf.reshape(sources + offsets, [-1]), tf.reshape(destinations + offsets, [-1])]
    )


def hybrid_spatial_cosine_edges(nodes, grid_size: int, top_k: int = 4):
    """Sparse deduplicated union of spatial-8 and current-feature cosine top-k."""

    batch_size = tf.shape(nodes)[0]
    node_count = grid_size * grid_size
    spatial = batch_shared_edges(spatial_8_edge_index(grid_size), batch_size, node_count)
    learned = cosine_top_k_edges(nodes, top_k)
    combined = tf.concat([spatial, learned], axis=1)
    total_nodes = batch_size * node_count
    keys = tf.cast(combined[0], tf.int64) * tf.cast(total_nodes, tf.int64) + tf.cast(combined[1], tf.int64)
    unique_keys, _ = tf.unique(keys)
    source = tf.cast(unique_keys // tf.cast(total_nodes, tf.int64), tf.int32)
    destination = tf.cast(unique_keys % tf.cast(total_nodes, tf.int64), tf.int32)
    return tf.stack([source, destination])


def sparse_pna_lite(messages, destinations, num_nodes):
    """Incoming mean, max and population std using segment operations."""

    messages = tf.convert_to_tensor(messages)
    destinations = tf.cast(destinations, tf.int32)
    counts = tf.math.unsorted_segment_sum(
        tf.ones([tf.shape(messages)[0], 1], messages.dtype), destinations, num_nodes
    )
    tf.debugging.assert_positive(counts, message="Every node needs an incoming edge")
    sums = tf.math.unsorted_segment_sum(messages, destinations, num_nodes)
    mean = sums / counts
    maximum = tf.math.unsorted_segment_max(messages, destinations, num_nodes)
    second = tf.math.unsorted_segment_sum(tf.square(messages), destinations, num_nodes) / counts
    std = tf.sqrt(tf.maximum(second - tf.square(mean), 0.0) + 1e-6)
    return mean, maximum, std
