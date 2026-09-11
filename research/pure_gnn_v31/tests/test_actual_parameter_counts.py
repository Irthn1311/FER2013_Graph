"""Tests verifying actual runtime parameter counts for G0, G0.5, and G1 models."""

import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import count_parameters


def test_actual_model_parameter_counts():
    """Verifies actual runtime parameter counts:
    - G0: 823,149
    - G0.5: 856,429
    - G1: 859,031
    """
    m0 = PureGNNv31(condition="G0")
    m05 = PureGNNv31(condition="G0.5")
    m1 = PureGNNv31(condition="G1")

    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = m0(dummy, training=False)
    _ = m05(dummy, training=False)
    _ = m1(dummy, training=False)

    p0 = count_parameters(m0)
    p05 = count_parameters(m05)
    p1 = count_parameters(m1)

    assert p0["total_trainable_parameters"] == 823149
    assert p05["total_trainable_parameters"] == 856429
    assert p1["total_trainable_parameters"] == 859031

    # Difference G0.5 - G0 is 33,280 (32,768 for two v_proj + 512 for two norm2)
    assert p05["total_trainable_parameters"] - p0["total_trainable_parameters"] == 33280

    # Difference G1 - G0.5 is 2,602 (exact coarse gate parameters)
    assert p1["total_trainable_parameters"] - p05["total_trainable_parameters"] == 2602
    assert p1["coarse_gate_mlp_parameters"] == 2602
