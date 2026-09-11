"""Unit tests for Pure-GNN v3.1 model forward passes, shapes, and sample independence."""

import pytest
import numpy as np
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31


@pytest.mark.parametrize("condition", ["G0", "G0.5", "G1", "G2", "G3"])
def test_all_conditions_forward_shape(condition):
    model = PureGNNv31(condition=condition)
    # Test (B, 48, 48, 1)
    x = tf.random.uniform([2, 48, 48, 1], dtype=tf.float32)
    logits = model(x, training=False)
    assert logits.shape == (2, 7)

    # Test (B, 48, 48)
    x_2d = tf.random.uniform([2, 48, 48], dtype=tf.float32)
    logits_2d = model(x_2d, training=False)
    assert logits_2d.shape == (2, 7)

    # Test (B, 2304, 1)
    x_nodes = tf.random.uniform([2, 2304, 1], dtype=tf.float32)
    logits_nodes = model(x_nodes, training=False)
    assert logits_nodes.shape == (2, 7)


def test_batch_sample_independence():
    """Verifies that running samples together in a batch yields identical logits to running them independently."""
    model = PureGNNv31(condition="G1")
    x1 = tf.random.uniform([1, 48, 48, 1], seed=101, dtype=tf.float32)
    x2 = tf.random.uniform([1, 48, 48, 1], seed=202, dtype=tf.float32)
    x_joint = tf.concat([x1, x2], axis=0)  # (2, 48, 48, 1)

    logits_joint = model(x_joint, training=False).numpy()
    logits1 = model(x1, training=False).numpy()
    logits2 = model(x2, training=False).numpy()

    np.testing.assert_allclose(logits_joint[0:1], logits1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logits_joint[1:2], logits2, rtol=1e-5, atol=1e-5)
