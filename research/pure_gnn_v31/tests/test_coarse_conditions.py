"""Unit tests for coarse conditions (G0, G0.5, G1, G2, G3) and functional matching contracts."""

import numpy as np
import pytest
import tensorflow as tf
from pure_gnn_v31.graph_index import build_complete_coarse_graph
from pure_gnn_v31.coarse_conditions import CoarseBlock


def test_coarse_g05_uniform_weights():
    block_g05 = CoarseBlock(channels=128, condition="G0.5")
    graph_coarse = build_complete_coarse_graph(6)
    h = tf.random.normal([2, 36, 128])

    out, diags = block_g05(h, coarse_graph=graph_coarse, training=False, return_diagnostics=True)
    assert out.shape == (2, 36, 128)
    assert abs(float(diags["effective_neighbor_count"].numpy()) - 35.0) < 1e-4


def test_coarse_g1_gate_zero_init_gives_uniform_distribution():
    block_g1 = CoarseBlock(channels=128, condition="G1", gate_hidden_dim=4)
    graph_coarse = build_complete_coarse_graph(6)
    h = tf.random.normal([2, 36, 128])

    out, diags = block_g1(h, coarse_graph=graph_coarse, training=False, return_diagnostics=True)
    assert out.shape == (2, 36, 128)

    eff_n = float(diags["effective_neighbor_count"].numpy())
    # At zero initialization, gate outputs ~1/35 for all 35 neighbors, giving effective neighbor count ~35.0
    assert abs(eff_n - 35.0) < 0.1, f"Expected effective neighbor count ~35, got {eff_n}"

    # Weights must sum to approximately 1.0 per receiver
    # Check bounds
    w_min = float(diags["coarse_weight_min"].numpy())
    w_max = float(diags["coarse_weight_max"].numpy())
    assert abs(w_min - 1.0 / 35.0) < 1e-3
    assert abs(w_max - 1.0 / 35.0) < 1e-3


def test_coarse_g0_no_non_local_edges():
    block_g0 = CoarseBlock(channels=128, condition="G0")
    graph_coarse = build_complete_coarse_graph(6)
    h = tf.random.normal([2, 36, 128])

    out, diags = block_g0(h, coarse_graph=graph_coarse, training=False, return_diagnostics=True)
    assert out.shape == (2, 36, 128)
    # G0 has no v_proj attribute
    assert not hasattr(block_g0, "v_proj")
