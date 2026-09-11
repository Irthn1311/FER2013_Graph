"""Unit tests verifying that no prohibited operations (Conv2D, Attention, Landmarks, etc.) exist."""

import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pathlib import Path

from pure_gnn_v31.contracts import (
    audit_forbidden_layers,
    audit_forbidden_source,
    audit_no_absolute_node_coordinates,
    audit_sparse_local_source,
)


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


def test_executable_source_ast_has_no_forbidden_architecture_primitive():
    source_root = Path(__file__).resolve().parents[1] / "src" / "pure_gnn_v31"
    assert audit_forbidden_source(source_root) == []


def test_local_production_source_is_sparse_and_has_no_full_grid_dense_adjacency():
    source_root = Path(__file__).resolve().parents[1] / "src" / "pure_gnn_v31"
    assert audit_sparse_local_source(source_root) == []


def test_model_source_projects_only_pixel_state_without_absolute_coordinates():
    source_root = Path(__file__).resolve().parents[1] / "src" / "pure_gnn_v31"
    assert audit_no_absolute_node_coordinates(source_root) == []
