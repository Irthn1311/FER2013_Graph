"""Flat-batch pooling and micro-motif attention mechanics."""

from __future__ import annotations

import tensorflow as tf


PART_ORDER = ["mouth", "eye", "brow", "nose_cheek", "global"]
GROUP_INDICES = [[5, 6, 7], [0, 1], [2, 3], [4, 8, 9]]


def group_priors(part_soft):
    priors = [tf.reduce_max(tf.gather(part_soft, indices, axis=1), axis=1) for indices in GROUP_INDICES]
    priors.append(tf.ones((tf.shape(part_soft)[0],), dtype=part_soft.dtype))
    return tf.stack(priors, axis=1)


def part_pool(h, part_soft, node_graph_index, valid_part_mask, num_graphs):
    priors = group_priors(part_soft)
    pooled = {}
    valid = {}
    for group_index, name in enumerate(PART_ORDER[:4]):
        weights = priors[:, group_index]
        input_indices = GROUP_INDICES[group_index]

        # PyTorch reduces each contiguous graph independently. Keeping that
        # reduction order avoids backend-dependent segment accumulation drift.
        def reduce_graph(graph_id):
            node_mask = tf.equal(node_graph_index, graph_id)
            h_graph = tf.boolean_mask(h, node_mask)
            weights_graph = tf.cast(
                tf.boolean_mask(weights, node_mask), h.dtype,
            )
            weight_sum = tf.cast(tf.reduce_sum(weights_graph), h.dtype)
            denominator = tf.maximum(
                weight_sum, tf.cast(1e-6, h.dtype),
            )
            input_valid = tf.reduce_sum(
                tf.gather(valid_part_mask[graph_id], input_indices),
            ) > 0.0
            group_valid = tf.logical_and(input_valid, denominator > 1e-5)
            value = tf.reduce_sum(h_graph * weights_graph[:, None], axis=0) / denominator
            return tf.where(group_valid, value, tf.zeros_like(value)), group_valid

        values, group_valid = tf.map_fn(
            reduce_graph,
            tf.range(num_graphs, dtype=tf.int32),
            fn_output_signature=(
                tf.TensorSpec((h.shape[-1],), h.dtype),
                tf.TensorSpec((), tf.bool),
            ),
        )
        pooled[name] = values
        valid[name] = group_valid

    def reduce_global(graph_id):
        node_mask = tf.equal(node_graph_index, graph_id)
        return tf.reduce_mean(tf.boolean_mask(h, node_mask), axis=0)

    pooled["global"] = tf.map_fn(
        reduce_global,
        tf.range(num_graphs, dtype=tf.int32),
        fn_output_signature=tf.TensorSpec((h.shape[-1],), h.dtype),
    )
    valid["global"] = tf.ones((num_graphs,), dtype=tf.bool)
    return pooled, valid


def pad_flat_nodes(values, node_graph_index, num_graphs, max_nodes=None):
    counts = tf.math.unsorted_segment_sum(
        tf.ones((tf.shape(values)[0],), dtype=tf.int32), node_graph_index, num_graphs,
    )
    if max_nodes is None:
        max_nodes = tf.reduce_max(counts)
    starts = tf.concat([tf.zeros((1,), tf.int32), tf.cumsum(counts)[:-1]], axis=0)
    within = tf.range(tf.shape(values)[0], dtype=tf.int32) - tf.gather(starts, node_graph_index)
    indices = tf.stack([node_graph_index, within], axis=1)
    output_shape = tf.concat([[num_graphs, max_nodes], tf.shape(values)[1:]], axis=0)
    padded = tf.scatter_nd(indices, values, output_shape)
    valid = tf.scatter_nd(indices, tf.ones((tf.shape(values)[0],), dtype=tf.bool), (num_graphs, max_nodes))
    return padded, valid, counts, indices
