"""Unit tests enforcing the parameter budget constraint and G0.5/G1 functional matching."""

import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import count_parameters, verify_parameter_budget


def test_g1_gate_parameter_share_under_half_percent():
    model_g1 = PureGNNv31(condition="G1")
    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = model_g1(dummy, training=False)

    passed, param_info = verify_parameter_budget(model_g1, max_share=0.005)
    total_params = param_info["total_trainable_parameters"]
    gate_params = param_info["coarse_gate_mlp_parameters"]
    share = param_info["coarse_gate_parameter_share"]

    print(f"Total trainable params: {total_params}")
    print(f"Coarse gate MLP params: {gate_params}")
    print(f"Gate share: {share * 100:.3f}%")

    assert passed is True, f"Gate share {share:.5f} exceeds 0.005 (0.5%)"
    assert share <= 0.005


def test_g05_vs_g1_exact_parameter_difference():
    model_g1 = PureGNNv31(condition="G1")
    model_g05 = PureGNNv31(condition="G0.5")

    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = model_g1(dummy, training=False)
    _ = model_g05(dummy, training=False)

    p_g1 = count_parameters(model_g1)
    p_g05 = count_parameters(model_g05)

    diff = p_g1["total_trainable_parameters"] - p_g05["total_trainable_parameters"]
    assert diff == p_g1["coarse_gate_mlp_parameters"], (
        f"Difference between G1 and G0.5 ({diff}) is not exactly equal to coarse gate MLP params ({p_g1['coarse_gate_mlp_parameters']})"
    )
