"""Verify the production optimizer on copied real-model fixture arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


GATE = 2e-8


def compare(
    expected: list[np.ndarray], actual: list[np.ndarray]
) -> dict[str, float | int | bool]:
    maximum = 0.0
    exact = 0
    for reference, candidate in zip(expected, actual):
        maximum = max(
            maximum,
            float(
                np.max(
                    np.abs(
                        candidate.astype(np.float64)
                        - reference.astype(np.float64)
                    )
                )
            ),
        )
        exact += int(np.array_equal(reference, candidate))
    return {
        "max_abs": maximum,
        "array_exact_tensors": exact,
        "tensor_count": len(expected),
        "pass_2e_8": maximum <= GATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.package_root / "src"))
    from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW

    tf.config.set_visible_devices([], "GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.config.experimental.enable_op_determinism()

    metadata = json.loads(args.trace_json.read_text(encoding="utf-8"))
    keys = metadata["keys"]
    with (
        np.load(args.state, allow_pickle=False) as state,
        np.load(args.trace_npz, allow_pickle=False) as trace,
    ):
        variables = [
            tf.Variable(
                np.asarray(state[key], np.float32),
                name=f"fixture_{index:03d}",
            )
            for index, key in enumerate(keys)
        ]
        gradients = [
            tf.constant(
                np.asarray(trace[f"clipped_gradient_{index:03d}"], np.float32)
            )
            for index in range(len(keys))
        ]
        optimizer = TorchCompatibleAdamW(
            learning_rate=3e-4,
            weight_decay=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            global_clipnorm=5.0,
        )
        optimizer.build(variables)
        rows = []
        for step in (1, 2):
            (
                optimizer._current_step_size,
                optimizer._current_bias_correction2_sqrt,
                optimizer._current_decay_factor,
            ) = optimizer._exact_step_scalars(
                tf.constant(np.float32(3e-4), tf.float32)
            )
            for gradient, variable in zip(gradients, variables):
                optimizer.update_step(
                    gradient,
                    variable,
                    tf.constant(np.float32(3e-4), tf.float32),
                )
            optimizer.iterations.assign_add(1)
            expected_parameters = [
                np.asarray(
                    trace[f"step{step}_parameter_after_addcdiv_{index:03d}"],
                    np.float32,
                )
                for index in range(len(keys))
            ]
            expected_momentums = [
                np.asarray(
                    trace[f"step{step}_momentum_after_lerp_{index:03d}"],
                    np.float32,
                )
                for index in range(len(keys))
            ]
            expected_velocities = [
                np.asarray(
                    trace[f"step{step}_velocity_after_addcmul_{index:03d}"],
                    np.float32,
                )
                for index in range(len(keys))
            ]
            row = {
                "step": step,
                "parameter": compare(
                    expected_parameters,
                    [np.asarray(variable.numpy(), np.float32) for variable in variables],
                ),
                "momentum": compare(
                    expected_momentums,
                    [
                        np.asarray(momentum.numpy(), np.float32)
                        for momentum in optimizer._momentums
                    ],
                ),
                "velocity": compare(
                    expected_velocities,
                    [
                        np.asarray(velocity.numpy(), np.float32)
                        for velocity in optimizer._velocities
                    ],
                ),
                "iterations": int(optimizer.iterations.numpy()),
                "global_norm": 15.556995391845703,
                "clip_coefficient": 0.3213987946510315,
            }
            row["pass"] = (
                row["parameter"]["pass_2e_8"]
                and row["momentum"]["pass_2e_8"]
                and row["velocity"]["pass_2e_8"]
                and row["iterations"] == step
            )
            rows.append(row)

    result = {
        "tensorflow": tf.__version__,
        "steps": rows,
        "offline_closure_pass": all(row["pass"] for row in rows),
        "optimizer_updates_on_copied_fixture_arrays": 2,
        "live_model_optimizer_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["offline_closure_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
