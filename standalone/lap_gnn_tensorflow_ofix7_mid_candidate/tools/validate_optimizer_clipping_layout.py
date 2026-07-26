"""Validate production optimizer clipping without applying an update."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf

from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--pytorch-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    golden = args.package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    model = build_model(batch)
    bindings = model.state_bindings()
    optimizer = TorchCompatibleAdamW()
    optimizer.build(model.trainable_variables)
    with np.load(
        golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False
    ) as source:
        gradients = []
        for binding in bindings:
            gradient = np.asarray(source[binding.source_key], np.float32)
            if binding.transform == "transpose":
                gradient = gradient.T
            gradients.append(tf.constant(gradient, tf.float32))
    optimizer._clip_variables = list(model.trainable_variables)
    clipped = optimizer._clip_gradients(gradients)
    optimizer._clip_variables = None

    maximum = 0.0
    exact = 0
    with np.load(args.pytorch_trace, allow_pickle=False) as reference:
        for index, (binding, gradient) in enumerate(zip(bindings, clipped)):
            actual = gradient.numpy()
            if binding.transform == "transpose":
                actual = actual.T
            expected = reference[f"real_clipped_gradient_{index:03d}"]
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            actual.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                ),
            )
            exact += int(np.array_equal(actual, expected))
    result = {
        "global_norm": float(optimizer.last_global_gradient_norm.numpy()),
        "clip_coefficient": float(optimizer.last_clip_coefficient.numpy()),
        "clipped_gradient_max_abs": maximum,
        "array_exact_tensors": exact,
        "tensor_count": len(bindings),
        "optimizer_updates_executed": 0,
        "pass": maximum == 0.0 and exact == len(bindings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
