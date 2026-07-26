"""Independent NumPy projection of two repaired real-model AdamW steps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def max_ulp(left: np.ndarray, right: np.ndarray) -> int:
    left_i = np.asarray(left, np.float32).view(np.int32).astype(np.int64)
    right_i = np.asarray(right, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(left_i - right_i))) if left_i.size else 0


def compare(kind, step, keys, expected, actual):
    maximum = 0.0
    total_abs = 0.0
    count = 0
    delta_square = 0.0
    reference_square = 0.0
    maximum_ulp = 0
    maximum_name = None
    exact = 0
    for key, left, right in zip(keys, expected, actual):
        delta = left.astype(np.float64) - right.astype(np.float64)
        current = float(np.max(np.abs(delta)))
        if current > maximum:
            maximum = current
            maximum_name = key
        total_abs += float(np.sum(np.abs(delta)))
        count += delta.size
        delta_square += float(np.sum(np.square(delta)))
        reference_square += float(np.sum(np.square(left.astype(np.float64))))
        maximum_ulp = max(maximum_ulp, max_ulp(left, right))
        exact += int(np.array_equal(left, right))
    return {
        "step": step,
        "tensor": kind,
        "max_abs": maximum,
        "mean_abs": total_abs / max(count, 1),
        "relative_l2": math.sqrt(delta_square)
        / max(math.sqrt(reference_square), 1e-12),
        "max_ulp": maximum_ulp,
        "max_tensor": maximum_name,
        "array_exact_tensors": exact,
        "tensor_count": len(keys),
        "pass_2e_8": maximum <= 2e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pytorch-trace", type=Path, required=True)
    parser.add_argument("--pytorch-live", type=Path, required=True)
    parser.add_argument("--pytorch-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.pytorch_json.read_text(encoding="utf-8"))
    keys = metadata["keys"]
    with (
        np.load(args.state, allow_pickle=False) as state,
        np.load(args.pytorch_trace, allow_pickle=False) as trace,
        np.load(args.pytorch_live, allow_pickle=False) as live,
    ):
        parameters = [np.asarray(state[key], np.float32).copy() for key in keys]
        gradients = [
            np.asarray(trace[f"real_clipped_gradient_{index:03d}"], np.float32)
            for index in range(len(keys))
        ]
        momentums = [np.zeros_like(value) for value in parameters]
        velocities = [np.zeros_like(value) for value in parameters]
        rows = []
        for step in (1, 2):
            next_parameters = []
            next_momentums = []
            next_velocities = []
            for parameter, gradient, momentum, velocity in zip(
                parameters, gradients, momentums, velocities
            ):
                parameter = np.float32(
                    parameter * np.float32(1.0 - 3e-4 * 1e-3)
                )
                momentum = np.float32(
                    momentum + (gradient - momentum) * np.float32(1.0 - 0.9)
                )
                velocity = np.float32(velocity * np.float32(0.999))
                velocity = np.float32(
                    velocity
                    + np.float32(gradient * gradient)
                    * np.float32(1.0 - 0.999)
                )
                denominator = np.float32(
                    np.sqrt(velocity)
                    / np.float32(math.sqrt(1.0 - 0.999**step))
                    + np.float32(1e-8)
                )
                parameter = np.float32(
                    parameter
                    + np.float32(momentum / denominator)
                    * np.float32(-3e-4 / (1.0 - 0.9**step))
                )
                next_parameters.append(parameter)
                next_momentums.append(momentum)
                next_velocities.append(velocity)
            for kind, values in [
                ("parameter", next_parameters),
                ("momentum", next_momentums),
                ("velocity", next_velocities),
            ]:
                expected = [
                    np.asarray(
                        live[f"step{step}_{kind}_{index:03d}"], np.float32
                    )
                    for index in range(len(keys))
                ]
                rows.append(compare(kind, step, keys, expected, values))
            parameters = next_parameters
            momentums = next_momentums
            velocities = next_velocities
    result = {
        "rows": rows,
        "all_pass_2e_8": all(row["pass_2e_8"] for row in rows),
        "optimizer_updates_executed": 0,
        "method": "independent_numpy_offline_projection",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
