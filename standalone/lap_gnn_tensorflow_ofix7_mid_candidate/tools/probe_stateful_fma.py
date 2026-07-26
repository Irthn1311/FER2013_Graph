"""Probe a graph-safe stateful software FMA on copied flattened fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def split_constant(value: float) -> tuple[np.float32, np.float32]:
    value32 = np.float32(value)
    scaled = np.float32(np.float32(4097.0) * value32)
    high = np.float32(scaled - np.float32(scaled - value32))
    return high, np.float32(value32 - high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
                        trace[f"step{step}_{kind}_{index:03d}"], np.float32
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

    parameter = tf.Variable(initial)
    momentum = tf.Variable(tf.zeros_like(parameter))
    velocity = tf.Variable(tf.zeros_like(parameter))
    work_product = tf.Variable(tf.zeros_like(parameter), trainable=False)
    work_error = tf.Variable(tf.zeros_like(parameter), trainable=False)
    work_total = tf.Variable(tf.zeros_like(parameter), trainable=False)
    work_sum_error = tf.Variable(tf.zeros_like(parameter), trainable=False)
    gradient = tf.constant(gradient)
    alpha = tf.constant(np.float32(0.1))
    alpha_high_value, alpha_low_value = split_constant(0.1)
    alpha_high = tf.constant(alpha_high_value)
    alpha_low = tf.constant(alpha_low_value)
    splitter = tf.constant(np.float32(4097.0))
    decay = tf.constant(np.float32(1.0 - 3e-4 * 1e-3))
    beta2 = tf.constant(np.float32(0.999))
    alpha2 = tf.constant(np.float32(0.001))
    epsilon = tf.constant(np.float32(1e-8))

    def stateful_fma(right, addend):
        work_product.assign(alpha * right)
        product = tf.identity(work_product)
        work_error.assign(splitter * right)
        scaled_right = tf.identity(work_error)
        work_total.assign(scaled_right - (scaled_right - right))
        right_high = tf.identity(work_total)
        right_low = right - right_high
        work_error.assign(
            (alpha_high * right_high - product)
            + alpha_high * right_low
            + alpha_low * right_high
            + alpha_low * right_low
        )
        product_error = tf.identity(work_error)
        work_total.assign(product + addend)
        total = tf.identity(work_total)
        work_sum_error.assign(total - product)
        recovered = tf.identity(work_sum_error)
        work_sum_error.assign(
            (product - (total - recovered)) + (addend - recovered)
        )
        sum_error = tf.identity(work_sum_error)
        return total + (product_error + sum_error)

    @tf.function(autograph=False)
    def apply_step(step):
        parameter.assign(parameter * decay)
        momentum_before = tf.identity(momentum)
        momentum.assign(
            stateful_fma(gradient - momentum_before, momentum_before)
        )
        velocity.assign(velocity * beta2)
        velocity.assign_add((alpha2 * gradient) * gradient)
        correction2 = tf.where(
            tf.equal(step, 1),
            tf.constant(np.float32((1.0 - 0.999) ** 0.5)),
            tf.constant(np.float32((1.0 - 0.999**2) ** 0.5)),
        )
        step_size = tf.where(
            tf.equal(step, 1),
            tf.constant(np.float32(3e-4 / (1.0 - 0.9))),
            tf.constant(np.float32(3e-4 / (1.0 - 0.9**2))),
        )
        denominator = tf.sqrt(velocity) / correction2 + epsilon
        parameter.assign_add(((-step_size) * momentum) / denominator)

    rows = []
    for step in (1, 2):
        apply_step(tf.constant(step, tf.int32))
        row = {"step": step}
        for kind, value in (
            ("parameter_after_addcdiv", parameter.numpy()),
            ("momentum_after_lerp", momentum.numpy()),
            ("velocity_after_addcmul", velocity.numpy()),
        ):
            row[f"{kind}_max_abs"] = float(
                np.max(
                    np.abs(
                        value.astype(np.float64)
                        - expected[(step, kind)].astype(np.float64)
                    )
                )
            )
        row["pass"] = all(
            row[f"{kind}_max_abs"] <= 2e-8
            for kind in (
                "parameter_after_addcdiv",
                "momentum_after_lerp",
                "velocity_after_addcmul",
            )
        )
        rows.append(row)
    result = {"steps": rows, "pass": all(row["pass"] for row in rows)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
