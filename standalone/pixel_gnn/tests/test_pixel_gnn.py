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


def test_warmup_cosine_decay():
    from pixel_gnn.utils import WarmupCosineDecay, build_scheduler

    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    sched = WarmupCosineDecay(opt, warmup_epochs=3, total_epochs=90, peak_lr=3e-4, min_lr=1e-6)

    # Epoch 1
    sched.on_epoch_start(1)
    lr1 = float(opt.learning_rate.numpy())
    assert abs(lr1 - (1e-6 + (3e-4 - 1e-6) * (1 / 3))) < 1e-8, f"Unexpected lr1: {lr1}"

    # Epoch 3 (warmup peak)
    sched.on_epoch_start(3)
    lr3 = float(opt.learning_rate.numpy())
    assert abs(lr3 - 3e-4) < 1e-8, f"Unexpected lr3: {lr3}"

    # Epoch 90 (decay floor)
    sched.on_epoch_start(90)
    lr90 = float(opt.learning_rate.numpy())
    assert abs(lr90 - 1e-6) < 1e-8, f"Unexpected lr90: {lr90}"

    # Test factory
    cfg = {
        "training": {
            "scheduler": {"type": "cosine_warmup", "warmup_epochs": 3, "min_lr": 1e-6},
            "lr": 3e-4,
            "max_epochs": 90,
        }
    }
    built = build_scheduler(cfg, opt)
    assert isinstance(built, WarmupCosineDecay)

    print("[TEST] test_warmup_cosine_decay passed!")


def test_dual_scale_model():
    from pixel_gnn.models import build_model
    from pixel_gnn.losses import compute_total_loss

    cfg = {
        "model": {
            "name": "pixel_motif_dual_scale",
            "hidden_dim": 96,
            "num_attention_layers": 2,
            "num_heads": 4,
            "num_motifs": 16,
            "use_motif_graph": True,
            "num_motif_gnn_layers": 1,
            "dropout": 0.1,
        },
        "loss": {"label_smoothing": 0.05, "lambda_motif_diversity": 0.01},
    }

    model = build_model(cfg)
    batch = {
        "node_features": tf.random.uniform((2, 2304, 5)),
        "labels": tf.constant([1, 4], dtype=tf.int64),
    }

    with tf.GradientTape() as tape:
        out = model(batch, training=True)
        loss, metrics = compute_total_loss(batch["labels"], out, label_smoothing=0.05)

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(grads) == len(model.trainable_variables)
    for g in grads:
        assert g is not None and tf.reduce_all(tf.math.is_finite(g))

    assert out["logits"].shape == (2, 7)
    assert out["z_image"].shape == (2, 96)
    assert out["z_motif"].shape == (2, 96)
    assert out["z_pixel"].shape == (2, 96)
    assert out["A_motif"].shape == (2, 16, 16)
    print("[TEST] test_dual_scale_model passed!")


def test_7d_features_and_cutout():
    from pixel_gnn.augmentation import (
        compute_image_gradients,
        compute_image_laplacian,
        apply_random_cutout,
        augment_batch,
        make_flipped_batch,
    )
    from pixel_gnn.losses import compute_total_loss

    # 1. Test Laplacian computation
    imgs = tf.random.uniform((4, 48, 48), dtype=tf.float32)
    gy, gx = compute_image_gradients(imgs)
    lap = compute_image_laplacian(gy, gx)
    assert lap.shape == (4, 48, 48)
    assert tf.reduce_all(tf.math.is_finite(lap))

    # 2. Test Cutout
    cutout_imgs = apply_random_cutout(imgs, cutout_prob=1.0, min_size=8, max_size=14, fill_value=0.0)
    assert cutout_imgs.shape == (4, 48, 48)
    # With cutout_prob=1.0, some pixels should be zero
    assert tf.reduce_any(cutout_imgs == 0.0)

    # 3. Test augment_batch with node_dim=7 and cutout
    batch_raw = {
        "image_48": imgs,
        "labels": tf.constant([0, 1, 2, 3], dtype=tf.int64),
    }
    aug = augment_batch(
        batch_raw,
        flip_prob=0.5,
        brightness_delta=0.08,
        contrast_range=(0.9, 1.1),
        cutout_prob=0.5,
        cutout_min_size=8,
        cutout_max_size=14,
        node_dim=7,
    )
    assert aug["node_features"].shape == (4, 2304, 7)
    assert tf.reduce_all(tf.math.is_finite(aug["node_features"]))

    # 4. Test make_flipped_batch with node_dim=7
    flipped = make_flipped_batch(aug, node_dim=7)
    assert flipped["node_features"].shape == (4, 2304, 7)
    assert tf.reduce_all(tf.math.is_finite(flipped["node_features"]))

    # 5. Test Dual-Scale model with 7D input
    cfg = {
        "model": {
            "name": "pixel_motif_dual_scale",
            "node_dim": 7,
            "hidden_dim": 96,
            "num_attention_layers": 2,
            "num_heads": 4,
            "num_motifs": 16,
            "spatial_span": 0.58,
            "use_motif_graph": True,
            "num_motif_gnn_layers": 1,
            "num_motif_heads": 4,
            "dropout": 0.1,
        },
        "loss": {"label_smoothing": 0.05, "lambda_motif_diversity": 0.01},
    }
    model = build_model(cfg)

    with tf.GradientTape() as tape:
        out = model(aug, training=True)
        loss, _ = compute_total_loss(aug["labels"], out, label_smoothing=0.05)

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(grads) == len(model.trainable_variables)
    for g in grads:
        assert g is not None and tf.reduce_all(tf.math.is_finite(g))

    assert out["logits"].shape == (4, 7)
    assert out["z_image"].shape == (4, 96)
    print("[TEST] test_7d_features_and_cutout passed!")


if __name__ == "__main__":
    test_registry()
    test_motif_graph_configurations()
    test_models_forward_and_backward()
    test_visualization_synthetic()
    test_augmentation_module()
    test_warmup_cosine_decay()
    test_dual_scale_model()
    test_7d_features_and_cutout()
    print("ALL TESTS IN pixel_gnn PASSED!")
