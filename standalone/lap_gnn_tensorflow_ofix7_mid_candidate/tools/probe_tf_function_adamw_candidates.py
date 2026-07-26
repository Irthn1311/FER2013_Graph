"""Probe graph-mode AdamW arithmetic candidates on flattened copied fixtures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


GATE = 2e-8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root / "src"))
    from lap_gnn_tf.training.optimizer import _software_fma

    tf.config.set_visible_devices([], "GPU")
    tf.config.optimizer.set_jit(False)
    tf.config.experimental.enable_op_determinism()
    metadata = json.loads(args.trace_json.read_text(encoding="utf-8"))
    count = len(metadata["keys"])
    with np.load(args.trace_npz, allow_pickle=False) as trace:
        initial = np.concatenate(
            [
                np.asarray(trace[f"initial_parameter_{index:03d}"], np.float32)
                .reshape(-1)
                for index in range(count)
            ]
        )
        gradient = np.concatenate(
            [
                np.asarray(trace[f"clipped_gradient_{index:03d}"], np.float32)
                .reshape(-1)
                for index in range(count)
            ]
        )
        expected = {
            (step, kind): np.concatenate(
                [
                    np.asarray(
                        trace[
                            f"step{step}_{kind}_{index:03d}"
                        ],
                        np.float32,
                    ).reshape(-1)
                    for index in range(count)
                ]
            )
            for step in (1, 2)
            for kind in (
                "parameter_after_addcdiv",
                "momentum_after_lerp",
                "velocity_after_addcmul",
            )
        }

    p0 = tf.constant(initial)
    g = tf.constant(gradient)
    decay = tf.constant(np.float32(1.0 - 3e-4 * 1e-3))
    alpha1 = tf.constant(np.float32(1.0 - 0.9))
    beta1 = tf.constant(np.float32(0.9))
    alpha2 = tf.constant(np.float32(1.0 - 0.999))
    beta2 = tf.constant(np.float32(0.999))
    epsilon = tf.constant(np.float32(1e-8))

    def lerp(momentum, name):
        if name == "delta":
            return momentum + (g - momentum) * alpha1
        if name == "weighted":
            return beta1 * momentum + alpha1 * g
        if name == "software_fma":
            return _software_fma(alpha1, g - momentum, momentum)
        raise ValueError(name)

    def update(parameter, momentum, denominator, step_size, name):
        if name == "scale_num_add":
            return parameter + ((-step_size) * momentum) / denominator
        if name == "scale_num_sub":
            return parameter - (step_size * momentum) / denominator
        if name == "ratio_scale_add":
            return parameter + (momentum / denominator) * (-step_size)
        if name == "ratio_scale_sub":
            return parameter - (momentum / denominator) * step_size
        if name == "denominator_fold":
            return parameter - momentum / (denominator / step_size)
        if name == "reciprocal":
            return parameter + (-step_size) * momentum * tf.math.reciprocal(
                denominator
            )
        raise ValueError(name)

    rows = []
    for lerp_name in ("delta", "weighted", "software_fma"):
        for update_name in (
            "scale_num_add",
            "scale_num_sub",
            "ratio_scale_add",
            "ratio_scale_sub",
            "denominator_fold",
            "reciprocal",
        ):
            @tf.function(autograph=False)
            def run():
                parameter = p0
                momentum = tf.zeros_like(p0)
                velocity = tf.zeros_like(p0)
                outputs = []
                for step in (1, 2):
                    parameter = parameter * decay
                    momentum = lerp(momentum, lerp_name)
                    velocity = velocity * beta2
                    velocity = velocity + (alpha2 * g) * g
                    correction2 = tf.constant(
                        np.float32((1.0 - 0.999**step) ** 0.5)
                    )
                    step_size = tf.constant(
                        np.float32(3e-4 / (1.0 - 0.9**step))
                    )
                    denominator = tf.sqrt(velocity) / correction2 + epsilon
                    parameter = update(
                        parameter,
                        momentum,
                        denominator,
                        step_size,
                        update_name,
                    )
                    outputs.extend([parameter, momentum, velocity])
                return outputs

            values = run()
            row = {"lerp": lerp_name, "update": update_name}
            all_pass = True
            for step in (1, 2):
                for offset, kind in enumerate(
                    (
                        "parameter_after_addcdiv",
                        "momentum_after_lerp",
                        "velocity_after_addcmul",
                    )
                ):
                    actual = values[(step - 1) * 3 + offset].numpy()
                    reference = expected[(step, kind)]
                    maximum = float(
                        np.max(
                            np.abs(
                                actual.astype(np.float64)
                                - reference.astype(np.float64)
                            )
                        )
                    )
                    row[f"step{step}_{kind}_max_abs"] = maximum
                    all_pass &= maximum <= GATE
            row["all_gates_pass"] = all_pass
            rows.append(row)

    rows.sort(
        key=lambda row: (
            not row["all_gates_pass"],
            row["step2_parameter_after_addcdiv_max_abs"],
            row["step2_momentum_after_lerp_max_abs"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:5], indent=2))


if __name__ == "__main__":
    main()
