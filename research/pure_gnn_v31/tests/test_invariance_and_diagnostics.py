"""Permutation, functional-match, and instrumentation evidence."""

import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.diagnostics import collect_graph_block_diagnostics
from pure_gnn_v31.technical_checks import (
    g05_g1_initial_equivalence_report,
    permutation_equivariance_report,
)


def test_local_and_coarse_permutation_equivariance():
    report = permutation_equivariance_report()
    np.testing.assert_allclose(report["local_max_abs_error"], 0.0, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(report["coarse_max_abs_error"], 0.0, atol=1e-5, rtol=0.0)


def test_g05_and_zero_initialized_g1_are_functionally_matched():
    report = g05_g1_initial_equivalence_report()
    np.testing.assert_allclose(report["max_abs_error"], 0.0, atol=1e-5, rtol=0.0)


def test_all_eight_graph_blocks_expose_feature_diagnostics_when_requested():
    model = PureGNNv31(condition="G1")
    images = tf.random.stateless_uniform([1, 48, 48, 1], seed=[94, 95])
    _, diagnostics = model(images, training=False, return_diagnostics=True)
    prefixes = [
        *(f"stage1_block{i}" for i in range(2)),
        *(f"stage2_block{i}" for i in range(2)),
        *(f"stage3_block{i}" for i in range(2)),
        *(f"coarse_block{i}" for i in range(2)),
    ]
    for prefix in prefixes:
        for metric in ("feature_variance", "feature_norm", "effective_rank"):
            assert f"{prefix}_{metric}" in diagnostics
    for prefix in ("stage1_block0", "stage1_block1", "stage2_block0", "stage2_block1", "stage3_block0", "stage3_block1"):
        assert f"{prefix}_gate_mean" in diagnostics
    for prefix in ("coarse_block0", "coarse_block1"):
        assert f"{prefix}_coarse_weight_entropy" in diagnostics
        assert f"{prefix}_effective_neighbor_count" in diagnostics


def test_all_graph_blocks_expose_finite_gradient_norms_via_opt_in_trace():
    model = PureGNNv31(condition="G1")
    images = tf.random.stateless_uniform([1, 48, 48, 1], seed=[96, 97])
    diagnostics = collect_graph_block_diagnostics(model, images)
    assert len(diagnostics) == 8
    assert all(np.isfinite(values["gradient_norm"]) for values in diagnostics.values())
