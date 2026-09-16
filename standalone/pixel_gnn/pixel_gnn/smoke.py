"""Smoke test for Pixel GNN architectures."""

from __future__ import annotations

import time
import numpy as np
import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology
from pixel_gnn.losses import compute_total_loss
from pixel_gnn.models import build_model


def make_dummy_batch(batch_size: int = 4) -> dict[str, tf.Tensor]:
    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords.numpy()

    images = np.random.uniform(0.0, 1.0, size=(batch_size, 48, 48)).astype(np.float32)

    node_features = []
    for img in images:
        gy, gx = np.gradient(img)
        intensity = img.reshape(-1, 1)
        gx_flat = gx.reshape(-1, 1)
        gy_flat = gy.reshape(-1, 1)
        feat = np.concatenate([intensity, coords, gx_flat, gy_flat], axis=1).astype(np.float32)
        node_features.append(feat)

    labels = np.random.randint(0, 7, size=(batch_size,), dtype=np.int64)

    return {
        "node_features": tf.convert_to_tensor(np.stack(node_features), dtype=tf.float32),
        "labels": tf.convert_to_tensor(labels, dtype=tf.int64),
        "sample_ids": tf.convert_to_tensor(np.arange(batch_size), dtype=tf.int64),
        "image_48": tf.convert_to_tensor(images, dtype=tf.float32),
    }


def run_smoke_test(config: dict | None = None) -> dict:
    if config is None:
        config = {
            "model": {
                "name": "pixel_neighbor_motif",
                "hidden_dim": 64,
                "num_attention_layers": 1,
                "num_motifs": 32,
                "pooling_type": "motif",
            },
            "loss": {"lambda_motif_diversity": 0.01},
        }

    model_name = config.get("model", {}).get("name", "pixel_neighbor_motif")
    print("=" * 80)
    print(f"[SMOKE] Testing Pixel GNN Model: {model_name}...")
    print("=" * 80, flush=True)

    model = build_model(config)
    batch_size = 4
    dummy_batch = make_dummy_batch(batch_size=batch_size)

    # 1. Forward pass
    output = model(dummy_batch, training=False)
    total_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"  [1/4] Forward pass OK! Parameters: {total_params:,}")

    # 2. Shapes check
    logits = output["logits"]
    assert logits.shape == (batch_size, 7), f"Logits shape mismatch: {logits.shape}"
    assert tf.reduce_all(tf.math.is_finite(logits)), "Logits contain NaN or Inf!"

    if output.get("motif_assignment") is not None:
        assign = output["motif_assignment"]
        num_motifs = config.get("model", {}).get("num_motifs", 32)
        assert assign.shape == (batch_size, 2304, num_motifs), f"Assignment shape mismatch: {assign.shape}"
        assert tf.reduce_all(tf.math.is_finite(assign)), "Assignment contains NaN!"
        # Check probability sum close to 1
        assign_sums = tf.reduce_sum(assign, axis=-1)
        assert tf.reduce_all(tf.abs(assign_sums - 1.0) < 1e-4), "Soft assignment does not sum to 1!"

    if output.get("A_motif") is not None:
        A_mat = output["A_motif"]
        num_motifs = config.get("model", {}).get("num_motifs", 32)
        assert A_mat.shape == (batch_size, num_motifs, num_motifs), f"A_motif shape mismatch: {A_mat.shape}"
        assert tf.reduce_all(tf.math.is_finite(A_mat)), "A_motif contains NaN!"

    print(f"  [2/4] Output shape and non-NaN check OK: logits={logits.shape}")

    # 3. Backward pass & gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_cfg = config.get("loss", {})
    lambda_div = float(loss_cfg.get("lambda_motif_diversity", 0.0))
    lambda_spatial = float(loss_cfg.get("lambda_spatial_coherence", 0.0))
    label_smooth = float(loss_cfg.get("label_smoothing", 0.0))

    with tf.GradientTape() as tape:
        out_train = model(dummy_batch, training=True)
        loss, _ = compute_total_loss(
            dummy_batch["labels"],
            out_train,
            lambda_diversity=lambda_div,
            lambda_spatial_coherence=lambda_spatial,
            label_smoothing=label_smooth,
        )

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(grads) == len(model.trainable_variables), "Mismatch in gradients length!"
    for g in grads:
        assert g is not None and tf.reduce_all(tf.math.is_finite(g)), "Invalid gradient detected!"

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"  [3/4] Gradient computation & optimizer step OK ({len(grads)} variables).")

    # 4. Benchmark
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model(dummy_batch, training=False)
    ms_per_step = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"  [4/4] Benchmark: {ms_per_step:.2f} ms per batch (B={batch_size})")

    print("=" * 80)
    print(f"[SMOKE PASSED] Model '{model_name}' successfully verified!")
    print("=" * 80, flush=True)

    return {"status": "PASSED", "parameters": total_params}
