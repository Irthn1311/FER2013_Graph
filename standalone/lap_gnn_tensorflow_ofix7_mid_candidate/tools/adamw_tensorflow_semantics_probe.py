"""TensorFlow float32 AdamW arithmetic probe against saved PyTorch traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


LR = 3e-4
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
MAX_NORM = 5.0
ZERO = tf.constant(0.0, tf.float32)
ONE = tf.constant(1.0, tf.float32)


def max_ulp(left: np.ndarray, right: np.ndarray) -> int:
    left_i = np.asarray(left, np.float32).view(np.int32).astype(np.int64)
    right_i = np.asarray(right, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(left_i - right_i))) if left_i.size else 0


def ds_constant(value: float) -> tuple[tf.Tensor, tf.Tensor]:
    high = np.float32(value)
    low = np.float32(value - float(high))
    return tf.constant(high), tf.constant(low)


def split(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    scaled = tf.constant(4097.0, tf.float32) * value
    high = scaled - (scaled - value)
    return high, value - high


def two_sum(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    total = left + right
    recovered = total - left
    error = (left - (total - recovered)) + (right - recovered)
    return total, error


def two_product(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    product = left * right
    left_high, left_low = split(left)
    right_high, right_low = split(right)
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error


def ds_add(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    total, error = two_sum(left[0], right[0])
    return two_sum(total, error + left[1] + right[1])


def ds_mul(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    product, error = two_product(left[0], right[0])
    error = (
        error
        + left[0] * right[1]
        + left[1] * right[0]
        + left[1] * right[1]
    )
    return two_sum(product, error)


def ds_pow(base: tuple[tf.Tensor, tf.Tensor], exponent: int) -> tuple[tf.Tensor, tf.Tensor]:
    result = (ONE, ZERO)
    factor = base
    remaining = int(exponent)
    while remaining:
        if remaining & 1:
            result = ds_mul(result, factor)
        factor = ds_mul(factor, factor)
        remaining >>= 1
    return result


def ds_div_float(
    numerator: tuple[tf.Tensor, tf.Tensor],
    denominator: tuple[tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    quotient = numerator[0] / denominator[0]
    product = ds_mul((quotient, ZERO), denominator)
    remainder = ds_add(numerator, (-product[0], -product[1]))
    return quotient + (remainder[0] + remainder[1]) / denominator[0]


def ds_sqrt_float(value: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    root = tf.sqrt(value[0])
    square = ds_mul((root, ZERO), (root, ZERO))
    remainder = ds_add(value, (-square[0], -square[1]))
    return root + (remainder[0] + remainder[1]) / (tf.constant(2.0, tf.float32) * root)


def exact_scalars(step: int) -> tuple[tf.Tensor, tf.Tensor]:
    beta1_power = ds_pow(ds_constant(BETA1), step)
    beta2_power = ds_pow(ds_constant(BETA2), step)
    correction1 = ds_add((ONE, ZERO), (-beta1_power[0], -beta1_power[1]))
    correction2 = ds_add((ONE, ZERO), (-beta2_power[0], -beta2_power[1]))
    step_size = ds_div_float(ds_constant(LR), correction1)
    correction2_sqrt = ds_sqrt_float(correction2)
    return step_size, correction2_sqrt


def cases() -> dict[str, dict]:
    return {
        "scalar_positive": {"parameters": [np.array(1.25, np.float32)], "gradients": [np.array(0.25, np.float32)]},
        "scalar_negative": {"parameters": [np.array(-0.75, np.float32)], "gradients": [np.array(-0.5, np.float32)]},
        "vector": {
            "parameters": [np.array([1.0, -2.0, 0.125, 8.0], np.float32)],
            "gradients": [np.array([0.25, -0.5, 0.0, 1.25], np.float32)],
        },
        "matrix": {
            "parameters": [np.array([[1.0, -2.0], [0.125, 8.0]], np.float32)],
            "gradients": [np.array([[0.25, -0.5], [0.0, 1.25]], np.float32)],
        },
        "zero_gradient": {"parameters": [np.array([1.0, -1.0], np.float32)], "gradients": [np.zeros(2, np.float32)]},
        "small_gradient": {
            "parameters": [np.array([1.0, -1.0], np.float32)],
            "gradients": [np.array([1e-10, -1e-12], np.float32)],
        },
        "clip_required": {
            "parameters": [np.array([1.0, -1.0], np.float32)],
            "gradients": [np.array([6.0, 8.0], np.float32)],
        },
        "two_variables": {
            "parameters": [np.array([1.0, -1.0], np.float32), np.array([[0.5, -0.25]], np.float32)],
            "gradients": [np.array([0.01, -0.02], np.float32), np.array([[6.0, 8.0]], np.float32)],
        },
        "preclipped": {
            "parameters": [np.array([1.0, -1.0], np.float32)],
            "gradients": [np.array([3.0, 4.0], np.float32)],
        },
        "restored_moments": {
            "parameters": [np.array([1.0, -1.0], np.float32)],
            "gradients": [np.array([0.25, -0.5], np.float32)],
            "initial_step": 3,
            "initial_m": [np.array([0.02, -0.04], np.float32)],
            "initial_v": [np.array([0.003, 0.005], np.float32)],
        },
    }


def candidate_step(
    parameter: tf.Tensor,
    gradient: tf.Tensor,
    momentum: tf.Tensor,
    velocity: tf.Tensor,
    step: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    decay = tf.constant(np.float32(1.0 - LR * WEIGHT_DECAY))
    one_minus_beta1 = tf.constant(np.float32(1.0 - BETA1))
    one_minus_beta2 = tf.constant(np.float32(1.0 - BETA2))
    beta2 = tf.constant(np.float32(BETA2))
    parameter = parameter * decay
    momentum = momentum + (gradient - momentum) * one_minus_beta1
    velocity = velocity * beta2
    velocity = velocity + tf.square(gradient) * one_minus_beta2
    step_size, correction2_sqrt = exact_scalars(step)
    denominator = tf.sqrt(velocity) / correction2_sqrt + tf.constant(np.float32(EPSILON))
    parameter = parameter + (momentum / denominator) * (-step_size)
    return parameter, momentum, velocity


def current_step(
    parameter: tf.Tensor,
    gradient: tf.Tensor,
    momentum: tf.Tensor,
    velocity: tf.Tensor,
    step: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    lr = tf.constant(np.float32(LR))
    beta1 = tf.constant(np.float32(BETA1))
    beta2 = tf.constant(np.float32(BETA2))
    parameter = parameter * (ONE - lr * tf.constant(np.float32(WEIGHT_DECAY)))
    momentum = beta1 * momentum + (ONE - beta1) * gradient
    velocity = beta2 * velocity + (ONE - beta2) * tf.square(gradient)
    step_value = tf.constant(np.float32(step))
    correction1 = ONE - tf.pow(beta1, step_value)
    correction2 = ONE - tf.pow(beta2, step_value)
    denominator = tf.sqrt(velocity) / tf.sqrt(correction2) + tf.constant(np.float32(EPSILON))
    parameter = parameter - (lr / correction1) * momentum / denominator
    return parameter, momentum, velocity


def compare_array(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-12)),
        "max_ulp": max_ulp(actual, expected),
        "array_exact": bool(np.array_equal(actual, expected)),
    }


def torch_cpu_avx2_norm(tensor: tf.Tensor) -> tf.Tensor:
    flat = tf.reshape(tensor, (-1,))
    size = tf.size(flat)
    vectorized_size = size - tf.math.floormod(size, 8)
    lanes = tf.reshape(flat[:vectorized_size], (-1, 8))
    lane_squares = tf.square(lanes)
    lane_totals = tf.math.cumsum(
        tf.concat([tf.zeros((1, 8), tf.float32), lane_squares], axis=0),
        axis=0,
    )[-1]
    total = lane_totals[0]
    for lane in range(1, 8):
        total = total + lane_totals[lane]
    tail_squares = tf.concat(
        [tf.square(flat[vectorized_size:]), tf.zeros((7,), tf.float32)],
        axis=0,
    )[:7]
    for tail_index in range(7):
        total = total + tail_squares[tail_index]
    return tf.sqrt(total)


def synthetic_probe(reference: np.lib.npyio.NpzFile) -> list[dict]:
    rows = []
    for name, case in cases().items():
        initial_parameters = [tf.constant(value) for value in case["parameters"]]
        initial_momentums = [
            tf.constant(value)
            for value in case.get("initial_m", [np.zeros_like(value) for value in case["parameters"]])
        ]
        initial_velocities = [
            tf.constant(value)
            for value in case.get("initial_v", [np.zeros_like(value) for value in case["parameters"]])
        ]
        current_parameters = list(initial_parameters)
        current_momentums = list(initial_momentums)
        current_velocities = list(initial_velocities)
        candidate_parameters = list(initial_parameters)
        candidate_momentums = list(initial_momentums)
        candidate_velocities = list(initial_velocities)
        initial_step = int(case.get("initial_step", 0))
        for local_step in (1, 2):
            step = initial_step + local_step
            for index in range(len(initial_parameters)):
                gradient = tf.constant(reference[f"{name}_step{local_step}_var{index}_gradient"])
                current = current_step(
                    current_parameters[index],
                    gradient,
                    current_momentums[index],
                    current_velocities[index],
                    step,
                )
                candidate = candidate_step(
                    candidate_parameters[index],
                    gradient,
                    candidate_momentums[index],
                    candidate_velocities[index],
                    step,
                )
                expected = {
                    "parameter": reference[f"{name}_step{local_step}_var{index}_parameter"],
                    "momentum": reference[f"{name}_step{local_step}_var{index}_momentum"],
                    "velocity": reference[f"{name}_step{local_step}_var{index}_velocity"],
                }
                for implementation, values in [("current", current), ("candidate", candidate)]:
                    for kind, value in zip(["parameter", "momentum", "velocity"], values):
                        rows.append(
                            {
                                "case": name,
                                "local_step": local_step,
                                "optimizer_step": step,
                                "variable": index,
                                "implementation": implementation,
                                "tensor": kind,
                                **compare_array(value.numpy(), expected[kind]),
                            }
                        )
                (
                    current_parameters[index],
                    current_momentums[index],
                    current_velocities[index],
                ) = current
                (
                    candidate_parameters[index],
                    candidate_momentums[index],
                    candidate_velocities[index],
                ) = candidate
    return rows


def clipping_probe(
    gradient_path: Path,
    reference: np.lib.npyio.NpzFile,
    keys: list[str],
) -> dict:
    with np.load(gradient_path, allow_pickle=False) as gradients:
        tensors = [tf.constant(np.asarray(gradients[key], np.float32)) for key in keys]
    expected_norms = reference["real_foreach_norms"]
    norm_methods = {
        "linalg": [
            tf.linalg.norm(tf.reshape(tensor, (-1,)), ord=2) for tensor in tensors
        ],
        "reduce_euclidean": [
            tf.math.reduce_euclidean_norm(tensor) for tensor in tensors
        ],
        "raw_euclidean": [
            tf.raw_ops.EuclideanNorm(
                input=tensor,
                axis=tf.range(tf.rank(tensor)),
                keep_dims=False,
            )
            for tensor in tensors
        ],
        "sqrt_reduce_sum_square": [
            tf.sqrt(tf.reduce_sum(tf.square(tensor))) for tensor in tensors
        ],
        "torch_cpu_avx2_lane8": [
            torch_cpu_avx2_norm(tensor) for tensor in tensors
        ],
    }
    norm_method_results = {}
    for method, method_norms in norm_methods.items():
        method_array = tf.stack(method_norms).numpy()
        norm_method_results[method] = {
            **compare_array(method_array, expected_norms),
            "global_norm": float(tf.linalg.norm(tf.stack(method_norms), ord=2).numpy()),
        }
    norms = norm_methods["torch_cpu_avx2_lane8"]
    global_norm = torch_cpu_avx2_norm(tf.stack(norms))
    coefficient = tf.minimum(
        ONE,
        tf.constant(np.float32(MAX_NORM))
        / (global_norm + tf.constant(np.float32(1e-6))),
    )
    comparisons = []
    for index, tensor in enumerate(tensors):
        clipped = (tensor * coefficient).numpy()
        expected = reference[f"real_clipped_gradient_{index:03d}"]
        comparisons.append(compare_array(clipped, expected))
    return {
        "global_norm": float(global_norm.numpy()),
        "clip_coefficient": float(coefficient.numpy()),
        "clipped_gradient_max_abs": max(row["max_abs"] for row in comparisons),
        "clipped_gradient_max_ulp": max(row["max_ulp"] for row in comparisons),
        "array_exact_tensors": sum(row["array_exact"] for row in comparisons),
        "tensor_count": len(comparisons),
        "norm_methods": norm_method_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-npz", type=Path, required=True)
    parser.add_argument("--pytorch-json", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    metadata = json.loads(args.pytorch_json.read_text(encoding="utf-8"))
    with np.load(args.pytorch_npz, allow_pickle=False) as reference:
        rows = synthetic_probe(reference)
        clipping = clipping_probe(
            args.gradients, reference, metadata["real_gradient_clip"]["keys"]
        )
    result = {
        "tensorflow": tf.__version__,
        "clipping": clipping,
        "current_max_abs": {
            kind: max(
                row["max_abs"]
                for row in rows
                if row["implementation"] == "current" and row["tensor"] == kind
            )
            for kind in ["parameter", "momentum", "velocity"]
        },
        "candidate_max_abs": {
            kind: max(
                row["max_abs"]
                for row in rows
                if row["implementation"] == "candidate" and row["tensor"] == kind
            )
            for kind in ["parameter", "momentum", "velocity"]
        },
        "candidate_all_pass_2e_8": all(
            row["max_abs"] <= 2e-8
            for row in rows
            if row["implementation"] == "candidate"
        ),
        "first_divergence": "one_minus_beta_float32_subtraction",
        "real_model_optimizer_updates": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tensorflow_semantics_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "tensorflow_semantics_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
