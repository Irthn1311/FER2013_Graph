"""Unit tests for standalone/pixel_gnn package."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
PIXEL_GNN_ROOT = ROOT / "standalone/pixel_gnn"

sys.path.insert(0, str(PIXEL_GNN_ROOT))

from pixel_gnn.grid import StaticGridTopology
from pixel_gnn.models import build_model, MODEL_REGISTRY
from pixel_gnn.losses import (
    compute_prototype_diversity_loss,
    compute_motif_diversity_loss,
    compute_spatial_coherence_loss,
    compute_motif_diagnostics,
    compute_total_loss,
)


def test_registry():
    assert "pixel_neighbor_motif" in MODEL_REGISTRY
    assert "pixel_motif_graph" in MODEL_REGISTRY
    assert "pixel_gnn_only" in MODEL_REGISTRY

    # Build neighbor motif
    m1 = build_model({"model": {"name": "pixel_neighbor_motif", "hidden_dim": 32, "num_motifs": 8}})
    assert m1.name == "pixel_neighbor_motif"

    # Build motif graph
    m2 = build_model({"model": {"name": "pixel_motif_graph", "hidden_dim": 32, "num_motifs": 16}})
    assert m2.name == "pixel_neighbor_motif"

    # Build pixel gnn only
    m3 = build_model({"model": {"name": "pixel_gnn_only", "hidden_dim": 32, "gnn_layers": 1}})
    assert m3.name == "pixel_gnn_only"
    print("[TEST] test_registry passed!")


def test_motif_graph_configurations():
    batch = {
        "node_features": tf.random.normal((2, 2304, 5)),
        "labels": tf.constant([0, 1], dtype=tf.int64),
    }

    # Test K = 16, 32, 64
    for k in [16, 32, 64]:
        cfg = {
            "model": {
                "name": "pixel_motif_graph",
                "hidden_dim": 32,
                "num_attention_layers": 2,
                "num_motifs": k,
                "use_motif_graph": True,
                "num_motif_gnn_layers": 1,
            },
            "loss": {
                "lambda_motif_diversity": 0.05,
                "lambda_spatial_coherence": 0.02,
            },
        }
        model = build_model(cfg)
        out = model(batch, training=False)

        # Check shapes
        assert out["logits"].shape == (2, 7)
        assert out["motif_assignment"].shape == (2, 2304, k)
        assert out["A_motif"].shape == (2, k, k)
        assert out["motif_spatial_centers"].shape == (k, 2)
        assert out["motif_prototypes"].shape == (k, 32)
        assert out["motif_gnn_attention"].shape == (2, k, k)

        # Non-NaN
        for key in ["logits", "motif_assignment", "A_motif", "motif_gnn_attention"]:
            assert tf.reduce_all(tf.math.is_finite(out[key])), f"NaN found in {key}"

        # Test loss & backward
        with tf.GradientTape() as tape:
            out_tr = model(batch, training=True)
            loss, metrics = compute_total_loss(
                batch["labels"],
                out_tr,
                lambda_diversity=0.05,
                lambda_spatial_coherence=0.02,
            )

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        for g in grads:
            assert g is not None and tf.reduce_all(tf.math.is_finite(g))

        # Check diagnostics
        diag = compute_motif_diagnostics(out["motif_assignment"])
        assert "active_motifs" in diag
        assert "assignment_entropy" in diag
        assert diag["active_motifs"] >= 1.0

        print(f"[TEST] pixel_motif_graph K={k} passed!")


def test_models_forward_and_backward():
    batch = {
        "node_features": tf.random.normal((2, 2304, 5)),
        "labels": tf.constant([0, 1], dtype=tf.int64),
    }

    for name in ["pixel_neighbor_motif", "pixel_gnn_only"]:
        if name == "pixel_neighbor_motif":
            cfg = {
                "model": {"name": name, "hidden_dim": 32, "num_motifs": 8},
                "loss": {"lambda_motif_diversity": 0.01, "lambda_spatial_coherence": 0.01},
            }
        else:
            cfg = {"model": {"name": name, "hidden_dim": 32, "gnn_layers": 1}, "loss": {}}

        model = build_model(cfg)

        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss, _ = compute_total_loss(
                batch["labels"],
                out,
                lambda_diversity=0.01,
                lambda_spatial_coherence=0.01,
            )

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        for g in grads:
            assert g is not None and tf.reduce_all(tf.math.is_finite(g))

        print(f"[TEST] model {name} forward & backward passed!")


def test_visualization_synthetic():
    from visualize_test_graph import run_visualization

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = PIXEL_GNN_ROOT / "configs/fer2013_pixel_motif_graph_fast_seed42.yaml"
        run_visualization(
            config_path=cfg_path,
            fer_csv=None,
            checkpoint_path=None,
            output_dir=tmpdir,
            num_samples=2,
        )

        expected_files = [
            "sample_001_original.png",
            "sample_001_pixel_graph.png",
            "sample_001_motif_map.png",
            "sample_001_overlay.png",
            "sample_001_motif_graph.png",
            "sample_001_attention_weights.png",
            "sample_001_summary.png",
        ]
        for f in expected_files:
            p = Path(tmpdir) / f
            assert p.exists() and p.stat().st_size > 0, f"Expected visualization file {f} missing or empty!"

    print("[TEST] test_visualization_synthetic passed!")


def test_augmentation_module():
    from pixel_gnn.augmentation import augment_batch, compute_image_gradients

    dummy_batch = {
        "image_48": tf.random.uniform((4, 48, 48), 0.0, 1.0),
        "labels": tf.constant([0, 1, 2, 3], dtype=tf.int64),
        "sample_ids": tf.constant([10, 20, 30, 40], dtype=tf.int64),
        "node_features": tf.zeros((4, 2304, 5)),
    }

    aug = augment_batch(
        dummy_batch,
        flip_prob=1.0,
        brightness_delta=0.08,
        contrast_range=(0.9, 1.1),
    )

    assert aug["node_features"].shape == (4, 2304, 5)
    assert aug["image_48"].shape == (4, 48, 48)
    assert tf.reduce_all(tf.math.is_finite(aug["node_features"])), "NaN in augmented features!"
    assert tf.reduce_all(aug["image_48"] >= 0.0) and tf.reduce_all(aug["image_48"] <= 1.0), "Augmented image out of [0, 1] bounds!"

    # Verify gradient computation against np.gradient
    img_np = aug["image_48"].numpy()
    gy_np, gx_np = np.gradient(img_np, axis=(1, 2))
    gy_tf, gx_tf = compute_image_gradients(aug["image_48"])

    np.testing.assert_allclose(gy_np, gy_tf.numpy(), atol=1e-5)
    np.testing.assert_allclose(gx_np, gx_tf.numpy(), atol=1e-5)

    print("[TEST] test_augmentation_module passed!")


if __name__ == "__main__":
    test_registry()
    test_motif_graph_configurations()
    test_models_forward_and_backward()
    test_visualization_synthetic()
    test_augmentation_module()
    print("ALL TESTS IN pixel_gnn PASSED!")
