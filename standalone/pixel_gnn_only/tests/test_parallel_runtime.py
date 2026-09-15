"""Synthetic sharding/protocol checks and two-GPU optimizer arithmetic checks.

These checks do not read FER2013, train a research model, or report accuracy.
Run on Kaggle; GPU-specific cases skip when two GPUs are unavailable.
"""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"))
sys.path.insert(0, str(REPO_ROOT / "standalone/pixel_gnn_only"))

from lap_gnn_tf.config import load_config
from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.optimizer import build_optimizer
from pixel_gnn_only.execution import build_restricted_graph_train_step, configure_restricted_grappler
from pixel_gnn_only.parallel import (shard_graph_batch, sync_replica, build_parallel_train_step,
                                     build_parallel_evaluation_step)
from pixel_gnn_only.protocol import validate_pixel_config, apply_batch_size_override

CONFIG_ROOT = REPO_ROOT / "standalone/pixel_gnn_only/configs"


def synthetic_batch(count):
    with tf.device("/CPU:0"):
        ids = np.arange(count * 2)
        return {
            "node_features": tf.constant(np.column_stack([ids + 1, np.zeros((count * 2, 4))]), tf.float32),
            "edge_index": tf.constant(np.stack([ids, ids ^ 1]), tf.int64),
            "edge_features": tf.zeros([count * 2, 6], tf.float32),
            "node_graph_index": tf.repeat(tf.range(count, dtype=tf.int64), 2),
            "graph_node_counts": tf.fill([count], tf.constant(2, tf.int64)),
            "graph_edge_counts": tf.fill([count], tf.constant(2, tf.int64)),
            "labels": tf.range(count, dtype=tf.int64) % 7,
            "sample_ids": tf.range(count, dtype=tf.int64) + 100,
            "image_48": tf.zeros([count, 48, 48], tf.float32),
        }


class ShardAndProtocolTests(unittest.TestCase):
    def test_odd_batch_rebases_edges_and_keeps_whole_graphs(self):
        batch = synthetic_batch(5)
        left = shard_graph_batch(batch, 0, 3)
        right = shard_graph_batch(batch, 3, 5)
        np.testing.assert_array_equal(left["sample_ids"].numpy(), [100, 101, 102])
        np.testing.assert_array_equal(right["sample_ids"].numpy(), [103, 104])
        np.testing.assert_array_equal(right["edge_index"].numpy(), [[0, 1, 2, 3], [1, 0, 3, 2]])
        np.testing.assert_array_equal(right["node_graph_index"].numpy(), [0, 0, 1, 1])
        edge_graphs = tf.gather(right["node_graph_index"], right["edge_index"])
        np.testing.assert_array_equal(edge_graphs[0].numpy(), edge_graphs[1].numpy())
        self.assertEqual(int(left["labels"].shape[0] + right["labels"].shape[0]), 5)

    def test_fast_batch_override_is_recorded_and_other_protocol_drift_fails(self):
        cfg = load_config(CONFIG_ROOT / "fer2013_pixel_gnn_only_kaggle_fast_seed42.yaml")
        apply_batch_size_override(cfg, 64)
        validate_pixel_config(cfg)
        self.assertEqual([cfg[k]["batch_size"] for k in ["training", "data", "resources"]], [64, 64, 64])
        self.assertFalse(cfg["ablation_provenance"]["training_protocol_equal"])
        self.assertEqual(cfg["ablation_provenance"]["authorized_training_changes"], ["batch_size"])
        changed = deepcopy(cfg)
        changed["training"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "protocol drift"):
            validate_pixel_config(changed)

    def test_original_batch16_config_remains_strict(self):
        cfg = load_config(CONFIG_ROOT / "fer2013_pixel_gnn_only_seed42.yaml")
        validate_pixel_config(cfg)
        self.assertEqual(cfg["training"]["batch_size"], 16)
        self.assertTrue(cfg["ablation_provenance"]["training_protocol_equal"])
        with self.assertRaisesRegex(ValueError, "separate Kaggle fast"):
            apply_batch_size_override(cfg, 32)


class ToyGraphModel(tf.keras.Model):
    """58 connected scalar weights exercise the ordered optimizer contract."""
    def __init__(self):
        super().__init__()
        self.scalars = [self.add_weight(name=f"w_{i}", shape=(), dtype=tf.float32,
                       initializer=tf.keras.initializers.Constant(0.0001)) for i in range(58)]

    def call(self, batch, training=False):
        values = tf.math.unsorted_segment_mean(batch["node_features"][:, 0],
                   tf.cast(batch["node_graph_index"], tf.int32), tf.shape(batch["labels"])[0])
        coefficient = tf.add_n([weight * float(i + 1) for i, weight in enumerate(self.scalars)])
        score = values * coefficient
        logits = tf.stack([score, -score, 0.2 * score, -0.4 * score,
                           0.5 * score, 0.7 * score, -0.2 * score], axis=1)
        return {"logits": logits, "probabilities": tf.nn.softmax(logits)}


@unittest.skipUnless(len(tf.config.list_physical_devices("GPU")) >= 2, "Two GPUs required")
class TwoGpuArithmeticTests(unittest.TestCase):
    def setUp(self):
        previous_policy = tf.keras.mixed_precision.global_policy().name
        tf.keras.mixed_precision.set_global_policy("float32")
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, previous_policy)
        configure_restricted_grappler()

    def models(self, batch):
        with tf.device("/GPU:0"):
            primary, reference = ToyGraphModel(), ToyGraphModel()
            primary(batch)
            reference(batch)
        with tf.device("/GPU:1"):
            replica = ToyGraphModel()
            replica(shard_graph_batch(batch, 0, 1))
        sync_replica(primary, replica)
        return primary, reference, replica

    def test_odd_batch_update_matches_single_global_clipped_update(self):
        batch = synthetic_batch(5)
        primary, reference, replica = self.models(batch)
        cfg = load_config(CONFIG_ROOT / "fer2013_pixel_gnn_only_seed42.yaml")
        with tf.device("/GPU:0"):
            first, second = build_optimizer(cfg), build_optimizer(cfg)
            first.build(primary.trainable_variables)
            second.build(reference.trainable_variables)
            with tf.GradientTape() as tape:
                loss = sparse_cross_entropy(batch["labels"], reference(batch)["logits"])
            raw = tape.gradient(loss, reference.trainable_variables)
            self.assertGreater(float(tf.linalg.global_norm(raw).numpy()), 5.0)
        parallel = build_parallel_train_step(primary, replica, first, training=False)
        single = build_restricted_graph_train_step(reference, second, training=False)
        np.testing.assert_allclose(parallel(batch).numpy(), single(batch).numpy(), rtol=1e-5)
        self.assertEqual(int(first.iterations.numpy()), 1)
        for actual, expected, mirror in zip(primary.trainable_variables, reference.trainable_variables, replica.trainable_variables):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-7)
            np.testing.assert_array_equal(actual.numpy(), mirror.numpy())

    def test_evaluation_preserves_order_and_short_single_graph_batch(self):
        batch = synthetic_batch(5)
        primary, reference, replica = self.models(batch)
        evaluate = build_parallel_evaluation_step(primary, replica)
        for count in [5, 1]:
            requested = shard_graph_batch(batch, 0, count)
            output = reference(requested)
            expected_loss = sparse_cross_entropy(requested["labels"], output["logits"])
            actual_loss, probabilities = evaluate(requested)
            np.testing.assert_allclose(actual_loss.numpy(), expected_loss.numpy(), rtol=1e-5)
            np.testing.assert_allclose(probabilities.numpy(), output["probabilities"].numpy(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
