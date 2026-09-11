"""Unit tests for fixed anti-aliased graph coarsening and boundary renormalization."""

import numpy as np
import pytest
import tensorflow as tf
from pure_gnn_v31.coarsening import (
    FixedAntiAliasedCoarsening,
    compute_aa_weights_and_indices,
    compute_simple_mean_indices,
)


def test_aa_boundary_renormalization_and_partition_of_unity():
    out_idx, in_idx, weights = compute_aa_weights_and_indices(48, 48)
    num_out_nodes = 24 * 24

    # Check partition of unity: sum of incoming weights per coarse node must equal 1.0
    weight_sums = np.zeros(num_out_nodes, dtype=np.float64)
    for k_out, w in zip(out_idx, weights):
        weight_sums[k_out] += float(w)

    np.testing.assert_allclose(
        weight_sums,
        1.0,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AA coarsening weights do not sum to 1.0 per output node!",
    )


def test_aa_specific_coefficients():
    # In a 48x48 grid:
    # Corner node (0, 0) corresponds to fine center (0, 0).
    # Valid neighbors in 3x3 are (0, 0), (0, 1), (1, 0), (1, 1).
    # Raw binomial weights:
    # (0, 0): b(0)*b(0) = 2*2 = 4
    # (0, 1): b(0)*b(1) = 2*1 = 2
    # (1, 0): b(1)*b(0) = 1*2 = 2
    # (1, 1): b(1)*b(1) = 1*1 = 1
    # Sum = 4 + 2 + 2 + 1 = 9.
    # Renormalized weights must be: 4/9, 2/9, 2/9, 1/9!
    out_idx, in_idx, weights = compute_aa_weights_and_indices(48, 48)

    corner_mask = out_idx == 0
    corner_weights = np.sort(weights[corner_mask])
    expected_corner = np.sort(np.array([4.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0], dtype=np.float32))
    np.testing.assert_allclose(corner_weights, expected_corner, rtol=1e-5)

    # Interior node e.g. coarse (1, 1) -> fine (2, 2).
    # Sum is 16.
    # Weights should be 4/16 (center), 2/16 (cardinal), 1/16 (diagonal).
    interior_node_idx = 1 * 24 + 1
    int_mask = out_idx == interior_node_idx
    int_weights = np.sort(weights[int_mask])
    expected_interior = np.sort(np.array([
        1/16, 1/16, 1/16, 1/16,
        2/16, 2/16, 2/16, 2/16,
        4/16
    ], dtype=np.float32))
    np.testing.assert_allclose(int_weights, expected_interior, rtol=1e-5)


def test_coarsening_layer_forward():
    layer = FixedAntiAliasedCoarsening(
        in_height=48, in_width=48, in_channels=32, out_channels=64
    )
    x = tf.random.normal([2, 2304, 32])
    out = layer(x)
    assert out.shape == (2, 576, 64)


def test_simple_mean_coarsening_forward():
    layer_mean = FixedAntiAliasedCoarsening(
        in_height=48, in_width=48, in_channels=32, out_channels=64, use_simple_mean=True
    )
    x = tf.random.normal([2, 2304, 32])
    out = layer_mean(x)
    assert out.shape == (2, 576, 64)
