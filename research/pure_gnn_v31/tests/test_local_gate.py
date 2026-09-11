"""Unit tests for Local Adaptive Relation Block and gate contracts in Pure-GNN v3.1."""

import pytest
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

    assert abs(float(d1["gate_mean"].numpy()) - float(d2["gate_mean"].numpy())) < 1e-6
