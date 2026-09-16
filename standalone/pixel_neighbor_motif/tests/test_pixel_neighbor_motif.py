"""Unit tests for Pixel Neighbor Attention + Learned Motif Prototypes."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
TF_STANDALONE = ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"
STANDALONE_ROOT = ROOT / "standalone/pixel_neighbor_motif"

sys.path.insert(0, str(TF_STANDALONE))
sys.path.insert(0, str(STANDALONE_ROOT))

from pixel_neighbor_motif.grid import StaticGridTopology, precompute_grid_topology
from pixel_neighbor_motif.losses import compute_prototype_diversity_loss, compute_total_loss
from pixel_neighbor_motif.model import (
    LocalNeighborAttentionLayer,
    LearnedMotifPrototypeLayer,
    MotifAttentionPooling,
    PixelNeighborMotifModel,
)


def test_grid_topology():
    """Verify static grid geometry and 8-neighborhood boundaries."""
    neighbors_idx, neighbor_valid, static_edge, coords = precompute_grid_topology()

    assert neighbors_idx.shape == (2304, 8)
    assert neighbor_valid.shape == (2304, 8)
    assert static_edge.shape == (2304, 8, 3)
    assert coords.shape == (2304, 2)

    # Top-left corner (pixel 0: row 0, col 0) has exactly 3 valid neighbors (right, bottom, bottom-right)
    assert np.sum(neighbor_valid[0]) == 3

    # Center pixel (row 24, col 24 -> index 24*48 + 24 = 1176) has exactly 8 valid neighbors
    center_idx = 24 * 48 + 24
    assert np.sum(neighbor_valid[center_idx]) == 8
    print("[TEST] test_grid_topology passed!")


def test_neighbor_attention_layer():
    """Verify LocalNeighborAttentionLayer forward pass and shapes."""
    batch_size = 2
    hidden_dim = 64
    layer = LocalNeighborAttentionLayer(hidden_dim=hidden_dim, edge_dim=3)

    h_in = tf.random.normal((batch_size, 2304, hidden_dim))
    h_out = layer(h_in, training=False)

    assert h_out.shape == (batch_size, 2304, hidden_dim)
    assert tf.reduce_all(tf.math.is_finite(h_out))
    print("[TEST] test_neighbor_attention_layer passed!")


def test_motif_prototypes_and_gradients():
    """Verify that motif prototypes receive gradients and are updated."""
    batch_size = 2
    hidden_dim = 64
    num_motifs = 16

    model = PixelNeighborMotifModel(
        hidden_dim=hidden_dim,
        num_attention_layers=1,
        num_motifs=num_motifs,
        pooling_type="motif",
    )

    dummy_batch = {
        "node_features": tf.random.normal((batch_size, 2304, 5)),
        "labels": tf.constant([1, 4], dtype=tf.int64),
    }

    with tf.GradientTape() as tape:
        out = model(dummy_batch, training=True)
        loss, _ = compute_total_loss(dummy_batch["labels"], out, lambda_diversity=0.01)

    grads = tape.gradient(loss, model.trainable_variables)
    proto_var = [v for v in model.trainable_variables if "motif_prototypes" in v.name][0]
    proto_idx = [i for i, v in enumerate(model.trainable_variables) if v is proto_var][0]
    proto_grad = grads[proto_idx]

    assert proto_grad is not None
    assert proto_grad.shape == (num_motifs, hidden_dim)
    assert tf.reduce_all(tf.math.is_finite(proto_grad))
    print("[TEST] test_motif_prototypes_and_gradients passed!")


def test_ablation_pooling_modes():
    """Verify both motif pooling and global_mean pooling."""
    for p_type in ["motif", "global_mean"]:
        model = PixelNeighborMotifModel(
            hidden_dim=32,
            num_attention_layers=1,
            num_motifs=8,
            pooling_type=p_type,
        )
        dummy_batch = {
            "node_features": tf.random.normal((2, 2304, 5)),
            "labels": tf.constant([0, 1], dtype=tf.int64),
        }
        out = model(dummy_batch, training=False)
        assert out["logits"].shape == (2, 7)
        assert out["z_image"].shape == (2, 32)
        if p_type == "motif":
            assert out["motif_assignment"] is not None
        else:
            assert out["motif_assignment"] is None
        print(f"[TEST] test_ablation_pooling_modes ({p_type}) passed!")


def test_diversity_loss():
    """Verify prototype diversity penalty calculation."""
    # Orthogonal prototypes -> cosine similarity off-diagonal = 0 -> loss = 0
    p_ortho = tf.eye(8, dtype=tf.float32)
    loss_ortho = compute_prototype_diversity_loss(p_ortho)
    assert tf.abs(loss_ortho) < 1e-6

    # Identical prototypes -> cosine similarity = 1 -> loss > 0
    p_identical = tf.ones((8, 8), dtype=tf.float32)
    loss_identical = compute_prototype_diversity_loss(p_identical)
    assert loss_identical > 0.5
    print("[TEST] test_diversity_loss passed!")


if __name__ == "__main__":
    test_grid_topology()
    test_neighbor_attention_layer()
    test_motif_prototypes_and_gradients()
    test_ablation_pooling_modes()
    test_diversity_loss()
    print("ALL TESTS PASSED SUCCESSFULLY!")
