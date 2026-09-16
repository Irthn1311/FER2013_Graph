"""Smoke verification test for Pixel Neighbor Attention + Learned Motif Prototypes.

Checks:
1. Syntax and import check.
2. Model build with custom configuration.
3. Dummy/synthetic forward pass with small batch.
4. Output shape validation ([B, 7] for logits and probabilities).
5. Non-NaN/Inf checks.
6. Verification that motif prototypes are in trainable_variables.
7. Verification of finite gradients flowing to motif prototypes.
8. One-step optimizer update check showing prototype weights actually change!
"""

from __future__ import annotations

import io
import json
import time
import numpy as np
import tensorflow as tf

from pixel_neighbor_motif.grid import StaticGridTopology
from pixel_neighbor_motif.losses import compute_total_loss
from pixel_neighbor_motif.model import PixelNeighborMotifModel


def make_dummy_batch(batch_size: int = 4) -> dict[str, tf.Tensor]:
    """Create a synthetic batch for smoke verification."""
    grid = StaticGridTopology.get_instance()
    coords = grid.normalized_coords.numpy()  # [2304, 2]

    # Generate synthetic 48x48 images
    images = np.random.uniform(0.0, 1.0, size=(batch_size, 48, 48)).astype(np.float32)

    # Compute node features [B, 2304, 5]
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


def run_smoke_test(
    batch_size: int = 4,
    hidden_dim: int = 64,
    num_attention_layers: int = 1,
    num_motifs: int = 32,
    temperature: float = 0.1,
    pooling_type: str = "motif",
    lambda_diversity: float = 0.01,
) -> dict:
    """Run end-to-end smoke verification."""
    print("=" * 80)
    print("[SMOKE] Starting Pixel Neighbor Motif Smoke Verification...")
    print("=" * 80, flush=True)

    # 1. Grid topology check
    grid = StaticGridTopology.get_instance()
    print(f"  [1/8] StaticGridTopology checked: {grid.neighbors_idx.shape} neighbors, {grid.static_edge_features.shape} edge attrs")

    # 2. Build model
    model = PixelNeighborMotifModel(
        hidden_dim=hidden_dim,
        num_attention_layers=num_attention_layers,
        num_motifs=num_motifs,
        temperature=temperature,
        pooling_type=pooling_type,
        dropout=0.1,
        num_classes=7,
    )
    dummy_batch = make_dummy_batch(batch_size=batch_size)
    output = model(dummy_batch, training=False)

    total_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"  [2/8] Model forward pass complete. Trainable parameters: {total_params:,}")

    # 3. Shape validations
    logits = output["logits"]
    probs = output["probabilities"]
    preds = output["predictions"]
    z = output["z_image"]
    assert logits.shape == (batch_size, 7), f"Logits shape mismatch: {logits.shape}"
    assert probs.shape == (batch_size, 7), f"Probabilities shape mismatch: {probs.shape}"
    assert preds.shape == (batch_size,), f"Predictions shape mismatch: {preds.shape}"
    assert z.shape == (batch_size, hidden_dim), f"Graph embedding shape mismatch: {z.shape}"
    print(f"  [3/8] Output shapes verified: logits={logits.shape}, z_image={z.shape}")

    # 4. Check for NaNs or Infs
    assert tf.reduce_all(tf.math.is_finite(logits)), "Logits contain NaN or Inf!"
    assert tf.reduce_all(tf.math.is_finite(probs)), "Probabilities contain NaN or Inf!"
    print("  [4/8] Finite value checks passed (no NaN or Inf).")

    # 5. Motif Prototype checks (if motif pooling enabled)
    if pooling_type == "motif":
        assert output["motif_assignment"].shape == (batch_size, 2304, num_motifs)
        assert output["motif_attention_weights"].shape == (batch_size, num_motifs)
        assert output["motif_prototypes"].shape == (num_motifs, hidden_dim)

        # Check prototype variable in trainable_variables
        proto_vars = [v for v in model.trainable_variables if "motif_prototypes" in v.name]
        assert len(proto_vars) == 1, f"Expected 1 prototype variable, found {len(proto_vars)}"
        proto_var = proto_vars[0]
        print(f"  [5/8] Motif prototype variable verified in trainable_variables: {proto_var.name} {proto_var.shape}")

        # 6. Gradient flow check
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        initial_proto_val = proto_var.numpy().copy()

        with tf.GradientTape() as tape:
            out_train = model(dummy_batch, training=True)
            loss, metrics = compute_total_loss(dummy_batch["labels"], out_train, lambda_diversity=lambda_diversity)

        grads = tape.gradient(loss, model.trainable_variables)
        var_to_grad = {v.name: g for v, g in zip(model.trainable_variables, grads)}

        assert proto_var.name in var_to_grad, "Missing gradient for prototype variable!"
        proto_grad = var_to_grad[proto_var.name]
        assert proto_grad is not None, "Prototype gradient is None!"
        assert tf.reduce_all(tf.math.is_finite(proto_grad)), "Prototype gradient has NaN/Inf!"
        proto_grad_norm = float(tf.linalg.global_norm([proto_grad]).numpy())
        assert proto_grad_norm > 0.0, "Prototype gradient is all zero!"
        print(f"  [6/8] Gradient flow verified: Proto grad norm = {proto_grad_norm:.6f} > 0")

        # 7. Optimizer update check (Weights change!)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        updated_proto_val = proto_var.numpy()
        weight_diff = np.max(np.abs(updated_proto_val - initial_proto_val))
        assert weight_diff > 0.0, "Prototypes did not update after optimizer step!"
        print(f"  [7/8] Prototype update verified: max weight delta = {weight_diff:.6f} > 0")
    else:
        print("  [5-7/8] Skipped motif checks (pooling_type != motif)")

    # 8. Benchmark small step
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model(dummy_batch, training=False)
    infer_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    print(f"  [8/8] Benchmark: {infer_ms:.2f} ms per batch (B={batch_size})")

    print("=" * 80)
    print(f"[SMOKE PASSED] All checks successfully completed! (Params: {total_params:,})")
    print("=" * 80, flush=True)

    return {
        "status": "PASSED",
        "trainable_parameters": total_params,
        "batch_size": batch_size,
        "shapes": {
            "logits": list(logits.shape),
            "z_image": list(z.shape),
        },
    }


if __name__ == "__main__":
    run_smoke_test()
