"""Independent tiny references and invariance checks for technical preflight."""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from pure_gnn_v31.coarse_conditions import CoarseBlock
from pure_gnn_v31.graph_index import build_complete_coarse_graph, build_grid_8neighbor_graph
from pure_gnn_v31.local_relation import LocalAdaptiveRelationBlock


def _receiver_edge_indices(dst: tf.Tensor, receiver: int) -> tf.Tensor:
    return tf.constant(np.flatnonzero(dst.numpy() == receiver), dtype=tf.int32)


def explicit_local_reference(
    block: LocalAdaptiveRelationBlock,
    h: tf.Tensor,
    graph: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """Explicit receiver loop implementing the registered local equations."""
    src = graph["src_indices"]
    dst = graph["dst_indices"]
    raw_h_j = tf.gather(h, src, axis=1)
    raw_h_i = tf.gather(h, dst, axis=1)
    raw_diff = raw_h_j - raw_h_i
    gate_h = block.norm1(raw_h_i)
    gate_diff = block.norm_diff(raw_diff)
    if block.mask_content:
        gate_h = tf.zeros_like(gate_h)
        gate_diff = tf.zeros_like(gate_diff)
    batch_size = tf.shape(h)[0]
    dx = tf.broadcast_to(graph["delta_x"][None], [batch_size, tf.shape(src)[0], 1])
    dy = tf.broadcast_to(graph["delta_y"][None], [batch_size, tf.shape(src)[0], 1])
    gate_input = tf.concat([gate_h, gate_diff, dx, dy], axis=-1)
    gates = 1.0 + tf.tanh(block.gate_dense2(block.gate_dense1(gate_input)))
    directional = tf.einsum(
        "bec,ecd->bed", raw_h_j, tf.gather(block.w_dir, graph["direction_id"])
    )
    messages = gates * (directional + block.w_rel(raw_diff))
    per_receiver = []
    for receiver in range(int(graph["num_nodes"].numpy())):
        edge_indices = _receiver_edge_indices(dst, receiver)
        per_receiver.append(tf.reduce_mean(tf.gather(messages, edge_indices, axis=1), axis=1))
    aggregated = tf.stack(per_receiver, axis=1)
    h_mid = h + aggregated
    return h_mid + block.ffn_dense2(block.ffn_dense1(block.norm2(h_mid)))


def explicit_coarse_reference(
    block: CoarseBlock,
    h: tf.Tensor,
    graph: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """Explicit receiver loop independent of production segment aggregation."""
    if block.condition == "G0":
        return h + block.ffn_dense2(block.ffn_dense1(block.norm1(h)))
    src = graph["src_indices"]
    dst = graph["dst_indices"]
    raw_h_i = tf.gather(h, dst, axis=1)
    raw_h_j = tf.gather(h, src, axis=1)
    raw_diff = raw_h_j - raw_h_i
    values = block.v_proj(raw_diff)
    if block.condition == "G0.5":
        edge_weights = tf.ones_like(values[..., :1]) / tf.cast(
            graph["edges_per_node"], tf.float32
        )
    else:
        gate_h = block.norm1(raw_h_i)
        gate_diff = block.coarse_gate_norm_diff(raw_diff)
        batch_size = tf.shape(h)[0]
        geometry = tf.broadcast_to(
            graph["p_ij"][None], [batch_size, tf.shape(src)[0], 3]
        )
        logits = block.coarse_gate_dense2(
            block.coarse_gate_dense1(tf.concat([gate_h, gate_diff, geometry], axis=-1))
        )
        unnormalized = tf.sigmoid(logits)
        normalized_parts = []
        for receiver in range(int(graph["num_nodes"].numpy())):
            edge_indices = _receiver_edge_indices(dst, receiver)
            receiver_weights = tf.gather(unnormalized, edge_indices, axis=1)
            denominator = tf.reduce_sum(receiver_weights, axis=1, keepdims=True) + block.epsilon
            normalized_parts.append((edge_indices, receiver_weights / denominator))
        edge_weights = tf.zeros_like(unnormalized)
        # Edge inventories are receiver-major; concatenate in explicit receiver order.
        edge_weights = tf.concat([weights for _, weights in normalized_parts], axis=1)
    messages = edge_weights * values
    per_receiver = []
    for receiver in range(int(graph["num_nodes"].numpy())):
        edge_indices = _receiver_edge_indices(dst, receiver)
        per_receiver.append(tf.reduce_sum(tf.gather(messages, edge_indices, axis=1), axis=1))
    aggregated = tf.stack(per_receiver, axis=1)
    h_mid = h + aggregated
    return h_mid + block.ffn_dense2(block.ffn_dense1(block.norm2(h_mid)))


def reference_equivalence_report() -> Dict[str, float]:
    """Measure production/reference value and input-gradient errors."""
    tf.keras.utils.set_random_seed(3101)
    local_graph = build_grid_8neighbor_graph(4, 4)
    local = LocalAdaptiveRelationBlock(channels=4, gate_hidden_dim=3)
    local_input = tf.random.stateless_normal([2, 16, 4], seed=[1, 2])
    with tf.GradientTape() as production_tape:
        production_tape.watch(local_input)
        production = local(local_input, graph=local_graph, training=False)
        production_objective = tf.reduce_sum(tf.square(production))
    production_gradient = production_tape.gradient(production_objective, local_input)
    with tf.GradientTape() as reference_tape:
        reference_tape.watch(local_input)
        reference = explicit_local_reference(local, local_input, local_graph)
        reference_objective = tf.reduce_sum(tf.square(reference))
    reference_gradient = reference_tape.gradient(reference_objective, local_input)

    report = {
        "local_output_max_abs_error": float(tf.reduce_max(tf.abs(production - reference)).numpy()),
        "local_gradient_max_abs_error": float(
            tf.reduce_max(tf.abs(production_gradient - reference_gradient)).numpy()
        ),
    }
    coarse_graph = build_complete_coarse_graph(3)
    coarse_input = tf.random.stateless_normal([2, 9, 4], seed=[3, 4])
    for condition, key in (("G0.5", "coarse_g05"), ("G1", "coarse_g1")):
        block = CoarseBlock(channels=4, condition=condition, gate_hidden_dim=2)
        production = block(coarse_input, coarse_graph=coarse_graph, training=False)
        reference = explicit_coarse_reference(block, coarse_input, coarse_graph)
        report[f"{key}_output_max_abs_error"] = float(
            tf.reduce_max(tf.abs(production - reference)).numpy()
        )
    return report


def _permuted_graph(graph: Dict[str, tf.Tensor], permutation: np.ndarray) -> Dict[str, tf.Tensor]:
    inverse = np.argsort(permutation)
    result = dict(graph)
    result["src_indices"] = tf.constant(inverse[graph["src_indices"].numpy()], tf.int32)
    result["dst_indices"] = tf.constant(inverse[graph["dst_indices"].numpy()], tf.int32)
    if "degrees" in graph:
        result["degrees"] = tf.gather(graph["degrees"], permutation)
    return result


def permutation_equivariance_report() -> Dict[str, float]:
    """Relabel nodes while preserving edge-attached direction/geometry values."""
    permutation = np.array([5, 0, 15, 7, 2, 10, 1, 14, 4, 9, 3, 12, 6, 11, 8, 13])
    inverse = np.argsort(permutation)
    local_graph = build_grid_8neighbor_graph(4, 4)
    local_permuted_graph = _permuted_graph(local_graph, permutation)
    local = LocalAdaptiveRelationBlock(channels=4, gate_hidden_dim=3)
    h = tf.random.stateless_normal([1, 16, 4], seed=[5, 6])
    original = local(h, graph=local_graph, training=False)
    permuted = local(tf.gather(h, permutation, axis=1), graph=local_permuted_graph, training=False)
    local_error = tf.reduce_max(tf.abs(original - tf.gather(permuted, inverse, axis=1)))

    coarse_permutation = np.array([4, 0, 8, 2, 6, 1, 7, 3, 5])
    coarse_inverse = np.argsort(coarse_permutation)
    coarse_graph = build_complete_coarse_graph(3)
    coarse_permuted_graph = _permuted_graph(coarse_graph, coarse_permutation)
    coarse = CoarseBlock(channels=4, condition="G1", gate_hidden_dim=2)
    hc = tf.random.stateless_normal([1, 9, 4], seed=[7, 8])
    original_c = coarse(hc, coarse_graph=coarse_graph, training=False)
    permuted_c = coarse(
        tf.gather(hc, coarse_permutation, axis=1),
        coarse_graph=coarse_permuted_graph,
        training=False,
    )
    coarse_error = tf.reduce_max(
        tf.abs(original_c - tf.gather(permuted_c, coarse_inverse, axis=1))
    )
    return {
        "local_max_abs_error": float(local_error.numpy()),
        "coarse_max_abs_error": float(coarse_error.numpy()),
    }


def copy_shared_coarse_weights(source: CoarseBlock, target: CoarseBlock) -> None:
    """Copy every registered non-gate weight between built G0.5/G1 blocks."""
    target.v_proj.set_weights(source.v_proj.get_weights())
    target.norm1.set_weights(source.norm1.get_weights())
    target.norm2.set_weights(source.norm2.get_weights())
    target.ffn_dense1.set_weights(source.ffn_dense1.get_weights())
    target.ffn_dense2.set_weights(source.ffn_dense2.get_weights())


def g05_g1_initial_equivalence_report() -> Dict[str, float]:
    graph = build_complete_coarse_graph(6)
    h = tf.random.stateless_normal([2, 36, 4], seed=[9, 10])
    uniform = CoarseBlock(channels=4, condition="G0.5", gate_hidden_dim=2)
    learned = CoarseBlock(channels=4, condition="G1", gate_hidden_dim=2)
    _ = uniform(h, coarse_graph=graph, training=False)
    _ = learned(h, coarse_graph=graph, training=False)
    copy_shared_coarse_weights(uniform, learned)
    uniform_out = uniform(h, coarse_graph=graph, training=False)
    learned_out = learned(h, coarse_graph=graph, training=False)
    return {"max_abs_error": float(tf.reduce_max(tf.abs(uniform_out - learned_out)).numpy())}
