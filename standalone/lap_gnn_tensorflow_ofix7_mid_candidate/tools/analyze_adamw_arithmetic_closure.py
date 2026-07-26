"""Offline TensorFlow arithmetic matrix against traced PyTorch AdamW primitives."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf


LR = 3e-4
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
GATE = 2e-8
SPLITTER = tf.constant(4097.0, tf.float32)


def max_ulp(left: np.ndarray, right: np.ndarray) -> int:
    left_i = np.asarray(left, np.float32).view(np.int32).astype(np.int64)
    right_i = np.asarray(right, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(left_i - right_i))) if left_i.size else 0


def metrics(
    keys: list[str],
    expected: list[np.ndarray],
    actual: list[np.ndarray],
) -> dict:
    maximum = 0.0
    maximum_name = None
    maximum_ulp = 0
    total_abs = 0.0
    delta_square = 0.0
    reference_square = 0.0
    count = 0
    exact = 0
    for key, reference, candidate in zip(keys, expected, actual):
        delta = candidate.astype(np.float64) - reference.astype(np.float64)
        current = float(np.max(np.abs(delta))) if delta.size else 0.0
        if current > maximum:
            maximum = current
            maximum_name = key
        maximum_ulp = max(maximum_ulp, max_ulp(reference, candidate))
        total_abs += float(np.sum(np.abs(delta)))
        delta_square += float(np.sum(np.square(delta)))
        reference_square += float(np.sum(np.square(reference.astype(np.float64))))
        count += delta.size
        exact += int(np.array_equal(reference, candidate))
    return {
        "max_abs": maximum,
        "mean_abs": total_abs / max(count, 1),
        "relative_l2": math.sqrt(delta_square)
        / max(math.sqrt(reference_square), 1e-12),
        "max_ulp": maximum_ulp,
        "max_tensor": maximum_name,
        "array_exact_tensors": exact,
        "tensor_count": len(keys),
        "pass_2e_8": maximum <= GATE,
    }


def as_tf(values: list[np.ndarray]) -> list[tf.Tensor]:
    return [tf.constant(np.asarray(value, np.float32)) for value in values]


def as_np(values: list[tf.Tensor]) -> list[np.ndarray]:
    return [np.asarray(value.numpy(), np.float32) for value in values]


def raw_add(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    return tf.raw_ops.AddV2(x=left, y=right)


def raw_mul(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    return tf.raw_ops.Mul(x=left, y=right)


def raw_div(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    return tf.raw_ops.RealDiv(x=left, y=right)


def two_sum(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    total = left + right
    recovered = total - left
    error = (left - (total - recovered)) + (right - recovered)
    return total, error


def split_float32(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    scaled = SPLITTER * value
    high = scaled - (scaled - value)
    return high, value - high


def two_product(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    product = left * right
    left_high, left_low = split_float32(left)
    right_high, right_low = split_float32(right)
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error


def software_fma(
    left: tf.Tensor, right: tf.Tensor, addend: tf.Tensor
) -> tf.Tensor:
    product, product_error = two_product(left, right)
    total, sum_error = two_sum(product, addend)
    return total + (product_error + sum_error)


def decay_variants(parameter: tf.Tensor, factor: tf.Tensor) -> dict[str, tf.Tensor]:
    variable = tf.Variable(parameter)
    variable.assign(variable * factor)
    return {
        "mul_expression": parameter * factor,
        "raw_mul": raw_mul(parameter, factor),
        "assign_expression": variable.read_value(),
    }


def lerp_variants(
    momentum: tf.Tensor,
    gradient: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
) -> dict[str, tf.Tensor]:
    return {
        "delta_then_scale": momentum + (gradient - momentum) * alpha,
        "scale_delta_then_add": alpha * (gradient - momentum) + momentum,
        "weighted_sum": beta * momentum + alpha * gradient,
        "raw_delta_then_scale": raw_add(
            momentum, raw_mul(gradient - momentum, alpha)
        ),
        "raw_weighted_sum": raw_add(
            raw_mul(beta, momentum), raw_mul(alpha, gradient)
        ),
        "software_fma_delta": software_fma(
            alpha, gradient - momentum, momentum
        ),
    }


def velocity_variants(
    velocity: tf.Tensor,
    gradient: tf.Tensor,
    beta: tf.Tensor,
    alpha: tf.Tensor,
) -> dict[str, tuple[tf.Tensor, tf.Tensor]]:
    multiplied = velocity * beta
    raw_multiplied = raw_mul(velocity, beta)
    return {
        "square_then_scale": (
            multiplied,
            multiplied + tf.square(gradient) * alpha,
        ),
        "scale_then_product": (
            multiplied,
            multiplied + (alpha * gradient) * gradient,
        ),
        "product_then_scale": (
            multiplied,
            multiplied + alpha * (gradient * gradient),
        ),
        "raw_product_then_scale": (
            raw_multiplied,
            raw_add(
                raw_multiplied,
                raw_mul(alpha, raw_mul(gradient, gradient)),
            ),
        ),
    }


def denominator_variants(
    velocity: tf.Tensor,
    correction_sqrt: tf.Tensor,
    epsilon: tf.Tensor,
) -> dict[str, tuple[tf.Tensor, tf.Tensor]]:
    square_root = tf.sqrt(velocity)
    divided = square_root / correction_sqrt
    return {
        "sqrt_div_add": (square_root, divided + epsilon),
        "raw_sqrt_div_add": (
            tf.raw_ops.Sqrt(x=velocity),
            raw_add(raw_div(tf.raw_ops.Sqrt(x=velocity), correction_sqrt), epsilon),
        ),
        "multiply_reciprocal_add": (
            square_root,
            square_root * tf.math.reciprocal(correction_sqrt) + epsilon,
        ),
    }


def update_variants(
    parameter: tf.Tensor,
    momentum: tf.Tensor,
    denominator: tf.Tensor,
    step_size: tf.Tensor,
) -> dict[str, tf.Tensor]:
    negative_step = -step_size
    return {
        "ratio_then_scale_add": parameter + (momentum / denominator) * negative_step,
        "scale_numerator_then_div_add": parameter
        + (negative_step * momentum) / denominator,
        "multiply_then_div_sub": parameter
        - (step_size * momentum) / denominator,
        "divide_then_multiply_sub": parameter
        - (momentum / denominator) * step_size,
        "denominator_fold": parameter - momentum / (denominator / step_size),
        "reciprocal_multiply": parameter
        + negative_step * momentum * tf.math.reciprocal(denominator),
        "raw_scale_numerator_then_div_add": raw_add(
            parameter, raw_div(raw_mul(negative_step, momentum), denominator)
        ),
        "raw_ratio_then_scale_add": raw_add(
            parameter, raw_mul(raw_div(momentum, denominator), negative_step)
        ),
    }


def load_list(
    archive: np.lib.npyio.NpzFile, prefix: str, count: int
) -> list[np.ndarray]:
    return [
        np.asarray(archive[f"{prefix}_{index:03d}"], np.float32)
        for index in range(count)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed_ulp(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    expected_i = expected.astype(np.float32).view(np.int32).astype(np.int64)
    actual_i = actual.astype(np.float32).view(np.int32).astype(np.int64)
    return actual_i - expected_i


def localization_rows(
    step: int,
    keys: list[str],
    trace: np.lib.npyio.NpzFile,
    projected: list[np.ndarray],
) -> list[dict]:
    rows = []
    for index, key in enumerate(keys):
        expected = np.asarray(
            trace[f"step{step}_parameter_after_addcdiv_{index:03d}"], np.float32
        )
        actual = projected[index]
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        differing = np.flatnonzero(difference.ravel() != 0.0)
        if not differing.size:
            continue
        initial = np.asarray(
            trace[f"step{step}_parameter_before_{index:03d}"], np.float32
        )
        gradient = np.asarray(trace[f"clipped_gradient_{index:03d}"], np.float32)
        momentum = np.asarray(
            trace[f"step{step}_momentum_after_lerp_{index:03d}"], np.float32
        )
        velocity = np.asarray(
            trace[f"step{step}_velocity_after_addcmul_{index:03d}"], np.float32
        )
        denominator = np.asarray(
            trace[f"step{step}_denominator_{index:03d}"], np.float32
        )
        ulps = signed_ulp(expected, actual)
        for flat_index in differing:
            multi_index = np.unravel_index(int(flat_index), expected.shape)
            rows.append(
                {
                    "step": step,
                    "tensor": key,
                    "flattened_index": int(flat_index),
                    "multidimensional_index": json.dumps(
                        [int(value) for value in multi_index]
                    ),
                    "initial_parameter": float(initial.ravel()[flat_index]),
                    "clipped_gradient": float(gradient.ravel()[flat_index]),
                    "first_moment": float(momentum.ravel()[flat_index]),
                    "second_moment": float(velocity.ravel()[flat_index]),
                    "adaptive_denominator": float(denominator.ravel()[flat_index]),
                    "pytorch_result": float(expected.ravel()[flat_index]),
                    "tensorflow_projected_result": float(actual.ravel()[flat_index]),
                    "absolute_difference": abs(float(difference.ravel()[flat_index])),
                    "signed_difference_tf_minus_torch": float(
                        difference.ravel()[flat_index]
                    ),
                    "signed_ulp_difference": int(ulps.ravel()[flat_index]),
                    "parameter_abs": abs(float(initial.ravel()[flat_index])),
                    "gradient_sign": int(np.sign(gradient.ravel()[flat_index])),
                    "parameter_sign": int(np.sign(initial.ravel()[flat_index])),
                    "rank": int(expected.ndim),
                    "shape": json.dumps(list(expected.shape)),
                    "c_contiguous": bool(expected.flags.c_contiguous),
                }
            )
    rows.sort(key=lambda row: row["absolute_difference"], reverse=True)
    return rows[:100]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.config.experimental.enable_op_determinism()
    metadata = json.loads(args.trace_json.read_text(encoding="utf-8"))
    keys = metadata["keys"]
    count = len(keys)
    primitive_rows: list[dict] = []
    variant_rows: list[dict] = []
    replay_rows: dict[int, list[dict]] = {1: [], 2: []}
    best_arrays: dict[str, np.ndarray] = {}

    with np.load(args.trace_npz, allow_pickle=False) as trace:
        initial_parameters = load_list(trace, "initial_parameter", count)
        gradients = load_list(trace, "clipped_gradient", count)

        current_parameters = as_tf(initial_parameters)
        current_momentums = as_tf(
            [np.zeros_like(parameter) for parameter in initial_parameters]
        )
        current_velocities = as_tf(
            [np.zeros_like(parameter) for parameter in initial_parameters]
        )

        candidate_names = {
            "lerp": list(
                lerp_variants(
                    tf.constant([0.0], tf.float32),
                    tf.constant([1.0], tf.float32),
                    tf.constant(np.float32(1.0 - BETA1)),
                    tf.constant(np.float32(BETA1)),
                )
            ),
            "velocity": list(
                velocity_variants(
                    tf.constant([0.0], tf.float32),
                    tf.constant([1.0], tf.float32),
                    tf.constant(np.float32(BETA2)),
                    tf.constant(np.float32(1.0 - BETA2)),
                )
            ),
            "denominator": list(
                denominator_variants(
                    tf.constant([1.0], tf.float32),
                    tf.constant([1.0], tf.float32),
                    tf.constant(np.float32(EPSILON)),
                )
            ),
            "update": list(
                update_variants(
                    tf.constant([1.0], tf.float32),
                    tf.constant([1.0], tf.float32),
                    tf.constant([1.0], tf.float32),
                    tf.constant(np.float32(LR)),
                )
            ),
        }

        for step in (1, 2):
            scalar = metadata["steps"][step - 1]
            decay_factor = tf.constant(
                np.float32(scalar["decay_factor_float32"]), tf.float32
            )
            alpha1 = tf.constant(
                np.float32(scalar["one_minus_beta1_float32"]), tf.float32
            )
            beta1 = tf.constant(np.float32(BETA1), tf.float32)
            alpha2 = tf.constant(
                np.float32(scalar["one_minus_beta2_float32"]), tf.float32
            )
            beta2 = tf.constant(np.float32(BETA2), tf.float32)
            correction_sqrt = tf.constant(
                np.float32(scalar["bias_correction2_sqrt_float32"]), tf.float32
            )
            step_size = tf.constant(
                np.float32(scalar["step_size_float32"]), tf.float32
            )
            epsilon = tf.constant(np.float32(EPSILON), tf.float32)

            expected_decay = load_list(trace, f"step{step}_after_decay", count)
            expected_momentum = load_list(
                trace, f"step{step}_momentum_after_lerp", count
            )
            expected_velocity_mul = load_list(
                trace, f"step{step}_velocity_after_mul", count
            )
            expected_velocity = load_list(
                trace, f"step{step}_velocity_after_addcmul", count
            )
            expected_sqrt = load_list(trace, f"step{step}_sqrt_velocity", count)
            expected_denominator = load_list(
                trace, f"step{step}_denominator", count
            )
            expected_parameter = load_list(
                trace, f"step{step}_parameter_after_addcdiv", count
            )

            decay_outputs: dict[str, list[tf.Tensor]] = {
                name: []
                for name in ["mul_expression", "raw_mul", "assign_expression"]
            }
            for parameter in current_parameters:
                values = decay_variants(parameter, decay_factor)
                for name, value in values.items():
                    decay_outputs[name].append(value)
            for name, values in decay_outputs.items():
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "weight_decay_mul",
                        "candidate": name,
                        **metrics(keys, expected_decay, as_np(values)),
                    }
                )

            lerp_outputs = {name: [] for name in candidate_names["lerp"]}
            for momentum, gradient in zip(current_momentums, as_tf(gradients)):
                values = lerp_variants(momentum, gradient, alpha1, beta1)
                for name, value in values.items():
                    lerp_outputs[name].append(value)
            for name, values in lerp_outputs.items():
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "momentum_lerp",
                        "candidate": name,
                        **metrics(keys, expected_momentum, as_np(values)),
                    }
                )

            velocity_mul_outputs = {name: [] for name in candidate_names["velocity"]}
            velocity_outputs = {name: [] for name in candidate_names["velocity"]}
            for velocity, gradient in zip(current_velocities, as_tf(gradients)):
                values = velocity_variants(velocity, gradient, beta2, alpha2)
                for name, (multiplied, value) in values.items():
                    velocity_mul_outputs[name].append(multiplied)
                    velocity_outputs[name].append(value)
            for name in candidate_names["velocity"]:
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "velocity_mul",
                        "candidate": name,
                        **metrics(
                            keys, expected_velocity_mul, as_np(velocity_mul_outputs[name])
                        ),
                    }
                )
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "velocity_addcmul",
                        "candidate": name,
                        **metrics(keys, expected_velocity, as_np(velocity_outputs[name])),
                    }
                )

            denominator_outputs = {
                name: {"sqrt": [], "denominator": []}
                for name in candidate_names["denominator"]
            }
            for velocity in as_tf(expected_velocity):
                values = denominator_variants(
                    velocity, correction_sqrt, epsilon
                )
                for name, (square_root, denominator) in values.items():
                    denominator_outputs[name]["sqrt"].append(square_root)
                    denominator_outputs[name]["denominator"].append(denominator)
            for name, values in denominator_outputs.items():
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "sqrt_velocity",
                        "candidate": name,
                        **metrics(keys, expected_sqrt, as_np(values["sqrt"])),
                    }
                )
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "denominator",
                        "candidate": name,
                        **metrics(
                            keys,
                            expected_denominator,
                            as_np(values["denominator"]),
                        ),
                    }
                )

            update_outputs = {name: [] for name in candidate_names["update"]}
            for parameter, momentum, denominator in zip(
                as_tf(expected_decay),
                as_tf(expected_momentum),
                as_tf(expected_denominator),
            ):
                values = update_variants(
                    parameter, momentum, denominator, step_size
                )
                for name, value in values.items():
                    update_outputs[name].append(value)
            for name, values in update_outputs.items():
                primitive_rows.append(
                    {
                        "step": step,
                        "primitive": "parameter_addcdiv",
                        "candidate": name,
                        **metrics(keys, expected_parameter, as_np(values)),
                    }
                )

            current_projected = as_np(update_outputs["ratio_then_scale_add"])
            write_csv(
                args.output_dir
                / f"0{4 + step}_step{step}_top100_parameter_differences.csv",
                localization_rows(step, keys, trace, current_projected),
            )

            matrix_names = {
                "lerp": [
                    "delta_then_scale",
                    "scale_delta_then_add",
                    "weighted_sum",
                    "software_fma_delta",
                ],
                "velocity": [
                    "square_then_scale",
                    "scale_then_product",
                    "product_then_scale",
                ],
                "denominator": [
                    "sqrt_div_add",
                    "raw_sqrt_div_add",
                ],
                "update": [
                    "ratio_then_scale_add",
                    "scale_numerator_then_div_add",
                    "multiply_then_div_sub",
                    "divide_then_multiply_sub",
                    "raw_scale_numerator_then_div_add",
                ],
            }
            combinations = itertools.product(
                matrix_names["lerp"],
                matrix_names["velocity"],
                matrix_names["denominator"],
                matrix_names["update"],
            )
            step_candidates = []
            for lerp_name, velocity_name, denominator_name, update_name in combinations:
                candidate_m = lerp_outputs[lerp_name]
                candidate_v = velocity_outputs[velocity_name]
                candidate_d = []
                for velocity in candidate_v:
                    candidate_d.append(
                        denominator_variants(
                            velocity, correction_sqrt, epsilon
                        )[denominator_name][1]
                    )
                candidate_p = []
                for parameter, momentum, denominator in zip(
                    decay_outputs["mul_expression"], candidate_m, candidate_d
                ):
                    candidate_p.append(
                        update_variants(
                            parameter, momentum, denominator, step_size
                        )[update_name]
                    )
                parameter_result = metrics(
                    keys, expected_parameter, as_np(candidate_p)
                )
                momentum_result = metrics(
                    keys, expected_momentum, as_np(candidate_m)
                )
                velocity_result = metrics(
                    keys, expected_velocity, as_np(candidate_v)
                )
                row = {
                    "step": step,
                    "decay": "mul_expression",
                    "lerp": lerp_name,
                    "velocity": velocity_name,
                    "denominator": denominator_name,
                    "update": update_name,
                    "parameter_max_abs": parameter_result["max_abs"],
                    "momentum_max_abs": momentum_result["max_abs"],
                    "velocity_max_abs": velocity_result["max_abs"],
                    "parameter_mean_abs": parameter_result["mean_abs"],
                    "momentum_mean_abs": momentum_result["mean_abs"],
                    "velocity_mean_abs": velocity_result["mean_abs"],
                    "parameter_relative_l2": parameter_result["relative_l2"],
                    "momentum_relative_l2": momentum_result["relative_l2"],
                    "velocity_relative_l2": velocity_result["relative_l2"],
                    "parameter_max_ulp": parameter_result["max_ulp"],
                    "momentum_max_ulp": momentum_result["max_ulp"],
                    "velocity_max_ulp": velocity_result["max_ulp"],
                    "parameter_exact_tensors": parameter_result[
                        "array_exact_tensors"
                    ],
                    "momentum_exact_tensors": momentum_result[
                        "array_exact_tensors"
                    ],
                    "velocity_exact_tensors": velocity_result[
                        "array_exact_tensors"
                    ],
                }
                row["all_gates_pass"] = (
                    row["parameter_max_abs"] <= GATE
                    and row["momentum_max_abs"] <= GATE
                    and row["velocity_max_abs"] <= GATE
                )
                step_candidates.append((row, candidate_p, candidate_m, candidate_v))

            step_candidates.sort(
                key=lambda item: (
                    not item[0]["all_gates_pass"],
                    item[0]["parameter_max_abs"],
                    item[0]["momentum_max_abs"],
                    item[0]["velocity_max_abs"],
                )
            )
            best_row, best_p, best_m, best_v = step_candidates[0]
            variant_rows.extend(item[0] for item in step_candidates)
            replay_rows[step] = [
                {
                    "step": step,
                    "primitive": primitive,
                    "max_abs": result["max_abs"],
                    "max_ulp": result["max_ulp"],
                    "array_exact_tensors": result["array_exact_tensors"],
                    "pass_2e_8": result["pass_2e_8"],
                }
                for primitive, result in [
                    (
                        "weight_decay_mul",
                        metrics(keys, expected_decay, as_np(decay_outputs["mul_expression"])),
                    ),
                    (
                        "momentum_lerp",
                        metrics(keys, expected_momentum, as_np(best_m)),
                    ),
                    (
                        "velocity_addcmul",
                        metrics(keys, expected_velocity, as_np(best_v)),
                    ),
                    (
                        "denominator",
                        metrics(
                            keys,
                            expected_denominator,
                            as_np(
                                [
                                    denominator_variants(
                                        velocity, correction_sqrt, epsilon
                                    )[best_row["denominator"]][1]
                                    for velocity in best_v
                                ]
                            ),
                        ),
                    ),
                    (
                        "parameter_addcdiv",
                        metrics(keys, expected_parameter, as_np(best_p)),
                    ),
                ]
            ]
            for index, values in enumerate((best_p, best_m, best_v)):
                kind = ("parameter", "momentum", "velocity")[index]
                for tensor_index, value in enumerate(values):
                    best_arrays[
                        f"step{step}_{kind}_{tensor_index:03d}"
                    ] = np.asarray(value.numpy(), np.float32)

            current_parameters = best_p
            current_momentums = best_m
            current_velocities = best_v

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "07_tensorflow_primitive_candidates.csv", primitive_rows)
    write_csv(args.output_dir / "08_arithmetic_variant_matrix.csv", variant_rows)
    write_csv(args.output_dir / "05_step1_primitive_replay.csv", replay_rows[1])
    write_csv(args.output_dir / "06_step2_primitive_replay.csv", replay_rows[2])
    np.savez_compressed(
        args.output_dir / "tensorflow_best_offline_replay.npz", **best_arrays
    )

    best_by_step = {
        step: next(row for row in variant_rows if row["step"] == step)
        for step in (1, 2)
    }
    first_difference = {}
    for step in (1, 2):
        first_difference[str(step)] = next(
            (
                row["primitive"]
                for row in replay_rows[step]
                if row["max_abs"] != 0.0
            ),
            None,
        )
    result = {
        "tensorflow": tf.__version__,
        "tensor_count": count,
        "primitive_rows": len(primitive_rows),
        "variant_rows": len(variant_rows),
        "best_by_step": best_by_step,
        "first_differing_primitive": first_difference,
        "offline_closure_pass": all(
            bool(best_by_step[step]["all_gates_pass"]) for step in (1, 2)
        ),
        "optimizer_updates_executed": 0,
        "full_training_launched": False,
    }
    (args.output_dir / "offline_arithmetic_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
