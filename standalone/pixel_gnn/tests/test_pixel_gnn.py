"""Unit tests for standalone/pixel_gnn package."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
TF_STANDALONE = ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"
PIXEL_GNN_ROOT = ROOT / "standalone/pixel_gnn"

sys.path.insert(0, str(TF_STANDALONE))
sys.path.insert(0, str(PIXEL_GNN_ROOT))

from pixel_gnn.grid import StaticGridTopology, precompute_grid_topology
from pixel_gnn.models import build_model, MODEL_REGISTRY
from pixel_gnn.losses import compute_prototype_diversity_loss, compute_total_loss


def test_registry():
    assert "pixel_neighbor_motif" in MODEL_REGISTRY
    assert "pixel_gnn_only" in MODEL_REGISTRY

    # Build neighbor motif
    m1 = build_model({"model": {"name": "pixel_neighbor_motif", "hidden_dim": 32, "num_motifs": 8}})
    assert m1.name == "pixel_neighbor_motif"

    # Build pixel gnn only
    m2 = build_model({"model": {"name": "pixel_gnn_only", "hidden_dim": 32, "gnn_layers": 1}})
    assert m2.name == "pixel_gnn_only"
    print("[TEST] test_registry passed!")


def test_models_forward_and_backward():
    batch = {
        "node_features": tf.random.normal((2, 2304, 5)),
        "labels": tf.constant([0, 1], dtype=tf.int64),
    }

    for name in ["pixel_neighbor_motif", "pixel_gnn_only"]:
        if name == "pixel_neighbor_motif":
            cfg = {"model": {"name": name, "hidden_dim": 32, "num_motifs": 8}, "loss": {"lambda_motif_diversity": 0.01}}
        else:
            cfg = {"model": {"name": name, "hidden_dim": 32, "gnn_layers": 1}, "loss": {}}

        model = build_model(cfg)

        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss, _ = compute_total_loss(batch["labels"], out)

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        for g in grads:
            assert g is not None and tf.reduce_all(tf.math.is_finite(g))

        print(f"[TEST] model {name} forward & backward passed!")


if __name__ == "__main__":
    test_registry()
    test_models_forward_and_backward()
    print("ALL TESTS IN pixel_gnn PASSED!")
