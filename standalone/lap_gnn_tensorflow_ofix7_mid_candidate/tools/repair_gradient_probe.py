"""Measure repaired TensorFlow gradients against the locked PyTorch fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _helpers import GOLDEN, loaded  # noqa: E402
from lap_gnn_tf.training.losses import sparse_cross_entropy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(42)
    np.random.seed(42)

    model, batch = loaded()
    with tf.GradientTape() as tape:
        logits = model(batch, training=False)["logits"]
        loss = sparse_cross_entropy(batch["labels"], logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    by_id = {
        id(variable): gradient
        for variable, gradient in zip(model.trainable_variables, gradients)
    }

    actual_parts: list[np.ndarray] = []
    expected_parts: list[np.ndarray] = []
    missing: list[str] = []
    with np.load(
        GOLDEN / "pytorch_gradients_eval_ce.npz", allow_pickle=False
    ) as reference:
        for binding in model.state_bindings():
            gradient = by_id.get(id(binding.variable))
            if gradient is None:
                missing.append(binding.source_key)
                continue
            actual = gradient.numpy()
            if binding.transform == "transpose":
                actual = actual.T
            actual_parts.append(actual.reshape(-1))
            expected_parts.append(reference[binding.source_key].reshape(-1))

    actual_vector = np.concatenate(actual_parts).astype(np.float64)
    expected_vector = np.concatenate(expected_parts).astype(np.float64)
    delta = actual_vector - expected_vector
    denominator = max(float(np.linalg.norm(expected_vector)), 1e-12)
    cosine = float(
        np.dot(actual_vector, expected_vector)
        / (np.linalg.norm(actual_vector) * np.linalg.norm(expected_vector))
    )
    result = {
        "loss": float(loss.numpy()),
        "tensor_count": len(actual_parts),
        "missing_tensors": missing,
        "all_finite": bool(np.isfinite(actual_vector).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "cosine": cosine,
        "pass": bool(not missing and cosine >= 0.99999),
        "optimizer_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
