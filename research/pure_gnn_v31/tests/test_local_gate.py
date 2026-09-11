"""Unit tests for Local Adaptive Relation Block and gate contracts in Pure-GNN v3.1."""

import numpy as np
import tensorflow as tf
from pure_gnn_v31.graph_index import build_grid_8neighbor_graph
from pure_gnn_v31.local_relation import LocalAdaptiveRelationBlock


def test_local_gate_initialization_equals_one():
    block = LocalAdaptiveRelationBlock(channels=32, gate_hidden_dim=16, mask_content=False)
    graph = build_grid_8neighbor_graph(12, 12)
    h = tf.random.normal([2, 144, 32])

    # Run with return_diagnostics
    h_out, diags = block(h, graph=graph, training=False, return_diagnostics=True)

    # Output shape must match input
    assert h_out.shape == (2, 144, 32)

    # Initial gate mean must equal 1.0 within 1e-5
    gate_mean = float(diags["gate_mean"].numpy())
    gate_std = float(diags["gate_std"].numpy())
    assert abs(gate_mean - 1.0) < 1e-5, f"Gate mean at init is {gate_mean}, expected 1.0"
    assert gate_std < 1e-5, f"Gate std at init is {gate_std}, expected ~0.0"


def test_local_gate_g2_content_masking():
    # G2 condition: mask_content=True
    block_g2 = LocalAdaptiveRelationBlock(channels=32, gate_hidden_dim=16, mask_content=True)
    graph = build_grid_8neighbor_graph(12, 12)

    # Even with two completely different node representations,
    # because content is masked to zero in G2, the gate input depends only on geometry
    h1 = tf.ones([1, 144, 32])
    h2 = tf.ones([1, 144, 32]) * 50.0

    _, d1 = block_g2(h1, graph=graph, training=False, return_diagnostics=True)
    _, d2 = block_g2(h2, graph=graph, training=False, return_diagnostics=True)

    np.testing.assert_array_equal(d1["gate_input_content"].numpy(), 0.0)
    np.testing.assert_array_equal(d2["gate_input_content"].numpy(), 0.0)


def test_local_raw_message_value_is_not_layer_normalized():
    block = LocalAdaptiveRelationBlock(channels=4, gate_hidden_dim=3)
    graph = build_grid_8neighbor_graph(4, 4)
    h = tf.random.stateless_normal([1, 16, 4], seed=[90, 91]) * 3.0 + 7.0
    _ = block(h, graph=graph, training=False)
    raw_sender = tf.gather(h, graph["src_indices"], axis=1)
    raw_receiver = tf.gather(h, graph["dst_indices"], axis=1)
    expected_direction = tf.einsum(
        "bec,ecd->bed", raw_sender, tf.gather(block.w_dir, graph["direction_id"])
    )
    expected_relation = block.w_rel(raw_sender - raw_receiver)
    normalized_sender = block.norm1(raw_sender)
    normalized_direction = tf.einsum(
        "bec,ecd->bed", normalized_sender, tf.gather(block.w_dir, graph["direction_id"])
    )
    assert not np.allclose(
        (expected_direction + expected_relation).numpy(),
        normalized_direction.numpy(),
    )
