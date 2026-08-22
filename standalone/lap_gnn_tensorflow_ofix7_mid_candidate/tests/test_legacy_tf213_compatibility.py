import tensorflow as tf

from _helpers import loaded
from lap_gnn_tf.compat import keras_version
from lap_gnn_tf.config import load_config
from lap_gnn_tf.training.execution import build_restricted_graph_train_step
from lap_gnn_tf.training.optimizer import build_optimizer


def first_graph(batch):
    node_mask = batch["node_graph_index"] == 0
    edge_mask = batch["edge_graph_index"] == 0
    node_keys = {
        "node_features", "node_types", "node_graph_index", "coordinates",
        "part_soft", "face_mask", "anchor_mask",
    }
    edge_keys = {"edge_features", "edge_graph_index"}
    graph_keys = {
        "graph_node_counts", "graph_edge_counts", "labels", "sample_ids",
        "valid_part_mask", "valid_anchor_mask", "detected",
        "landmark_missing_flag", "image_48",
    }
    result = {}
    for key, value in batch.items():
        if key in node_keys:
            result[key] = tf.boolean_mask(value, node_mask)
        elif key == "edge_index":
            result[key] = tf.boolean_mask(value, edge_mask, axis=1)
        elif key in edge_keys:
            result[key] = tf.boolean_mask(value, edge_mask)
        elif key in graph_keys:
            result[key] = value[:1]
        else:
            result[key] = value
    return result


def test_keras_version_is_available_without_tf_keras_version_attribute():
    assert keras_version() != "unknown"


def test_registered_mixed_precision_step_supports_both_loss_scale_apis():
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model, batch = loaded()
        batch = first_graph(batch)
        config = load_config(
            "configs/fer2013_ofix7_mid_tensorflow_seed42.yaml"
        )
        optimizer = build_optimizer(config)
        optimizer.build(model.trainable_variables)
        train_step = build_restricted_graph_train_step(model, optimizer)
        loss = train_step(batch)
        assert bool(tf.math.is_finite(loss))
        if not hasattr(tf.keras.optimizers.Optimizer, "_backend_apply_gradients"):
            assert int(optimizer.iterations.numpy()) == 1
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
