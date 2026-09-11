"""Unit tests verifying that no prohibited operations (Conv2D, Attention, Landmarks, etc.) exist."""

import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import audit_forbidden_layers


@pytest.mark.parametrize("condition", ["G0", "G0.5", "G1", "G2", "G3"])
def test_no_forbidden_layers_in_model(condition):
    model = PureGNNv31(condition=condition)
    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = model(dummy, training=False)

    violations = audit_forbidden_layers(model)
    assert len(violations) == 0, f"Forbidden ops detected in condition {condition}: {violations}"


def test_no_absolute_coordinates_in_node_features():
    """Confirms that node inputs are strictly 1-channel raw pixel intensity, without (x, y) coordinates."""
    model = PureGNNv31(condition="G1")
    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = model(dummy, training=False)
    # input projection must take 1 feature channel -> C
    assert model.input_proj.weights[0].shape[0] == 1
