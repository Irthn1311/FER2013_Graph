"""Validate selected G1-A checkpointing, mixed precision, and performance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model
from lap_gnn_tf.training.execution import (
    apply_gradients_eager_exact,
    build_compiled_gradient_function,
    configure_restricted_grappler,
)
from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.optimizer import (
    TorchCompatibleAdamW,
    _torch_cpu_avx2_norm,
)


def configure(policy: str = "float32") -> None:
    configure_restricted_grappler()
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()


def build_float32_instance(package_root: Path):
    golden = package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    model = build_model(batch)
    load_pytorch_npz(model, golden / "model_state.npz", strict=True)
    optimizer = TorchCompatibleAdamW()
    optimizer.build(model.trainable_variables)
    bindings = model.state_bindings()
    with np.load(
        golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False
    ) as source:
        by_id = {}
        for binding in bindings:
            gradient = np.asarray(source[binding.source_key], np.float32)
            if binding.transform == "transpose":
                gradient = gradient.T
            by_id[id(binding.variable)] = tf.constant(gradient, tf.float32)
    pairs = [
        (by_id[id(variable)], variable)
        for variable in model.trainable_variables
    ]
    return model, optimizer, batch, pairs


def capture(model, optimizer) -> dict[str, np.ndarray]:
    arrays = {
        f"model_{index:03d}": np.asarray(variable.numpy())
        for index, variable in enumerate(model.trainable_variables)
    }
    arrays.update({
        f"optimizer_{index:03d}": np.asarray(variable.numpy())
        for index, variable in enumerate(optimizer.variables)
    })
    return arrays


def exact_registered_global_norm(gradients, variables):
    norm_gradients = []
    for gradient, variable in zip(gradients, variables):
        path = str(getattr(variable, "path", variable.name))
        leaf_name = path.rsplit("/", 1)[-1].split(":", 1)[0]
        if len(variable.shape) == 2 and leaf_name in {
            "kernel",
            "in_proj_kernel",
            "out_kernel",
        }:
            gradient = tf.transpose(gradient)
        norm_gradients.append(gradient)
    return _torch_cpu_avx2_norm(
        tf.stack([
            _torch_cpu_avx2_norm(gradient)
            for gradient in norm_gradients
        ])
    )


def compare(expected_path: Path, actual: dict[str, np.ndarray]) -> dict:
    with np.load(expected_path, allow_pickle=False) as expected:
        exact = 0
        maximum = 0.0
        for key in expected.files:
            candidate = actual[key]
            reference = expected[key]
            exact += int(np.array_equal(candidate, reference))
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
        return {
            "array_count": len(expected.files),
            "exact_arrays": exact,
            "max_abs": maximum,
            "pass": exact == len(expected.files) and maximum == 0.0,
        }


def child_checkpoint(args) -> None:
    configure()
    golden = args.package_root / "validation_assets" / "golden"
    model = tf.keras.models.load_model(args.checkpoint)
    optimizer = model.optimizer
    bindings = model.state_bindings()
    with np.load(
        golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False
    ) as source:
        by_id = {}
        for binding in bindings:
            gradient = np.asarray(source[binding.source_key], np.float32)
            if binding.transform == "transpose":
                gradient = gradient.T
            by_id[id(binding.variable)] = tf.constant(gradient, tf.float32)
    pairs = [
        (by_id[id(variable)], variable)
        for variable in model.trainable_variables
    ]

    @tf.function(autograph=False, jit_compile=False)
    def apply():
        optimizer.apply_gradients(pairs)

    restored = compare(args.step2_state, capture(model, optimizer))
    apply()
    continued = compare(args.step3_state, capture(model, optimizer))
    result = {
        "restored": restored,
        "continued": continued,
        "iterations": int(optimizer.iterations.numpy()),
        "pass": restored["pass"] and continued["pass"]
        and int(optimizer.iterations.numpy()) == 3,
    }
    args.child_result.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["pass"]:
        raise SystemExit(1)


def checkpoint_validation(package_root: Path, output: Path) -> dict:
    configure()
    model, optimizer, _batch, pairs = build_float32_instance(package_root)
    model.compile(optimizer=optimizer, run_eagerly=False)

    @tf.function(autograph=False, jit_compile=False)
    def apply():
        optimizer.apply_gradients(pairs)

    apply()
    apply()
    step2 = output / "g1_checkpoint_step2_state.npz"
    np.savez_compressed(step2, **capture(model, optimizer))
    checkpoint = output / "g1_step2.keras"
    model.save(checkpoint, include_optimizer=True)
    apply()
    step3 = output / "g1_uninterrupted_step3_state.npz"
    np.savez_compressed(step3, **capture(model, optimizer))
    child_result = output / "g1_checkpoint_child.json"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--checkpoint-child",
        "--package-root",
        str(package_root),
        "--checkpoint",
        str(checkpoint),
        "--step2-state",
        str(step2),
        "--step3-state",
        str(step3),
        "--child-result",
        str(child_result),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        return {
            "pass": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
        }
    result = json.loads(child_result.read_text(encoding="utf-8"))
    result["checkpoint"] = str(checkpoint)
    result["fresh_process_returncode"] = completed.returncode
    return result


def mixed_precision_validation(package_root: Path) -> dict:
    configure("mixed_float16")
    golden = package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    model = build_model(batch)
    load_pytorch_npz(model, golden / "model_state.npz", strict=True)
    inner = TorchCompatibleAdamW()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(inner)
    optimizer.build(model.trainable_variables)

    @tf.function(autograph=False, jit_compile=False)
    def train_step():
        with tf.GradientTape() as tape:
            output = model(batch, training=False)
            loss = sparse_cross_entropy(batch["labels"], output["logits"])
            scaled_loss = optimizer.scale_loss(loss)
        gradients = tape.gradient(
            scaled_loss, model.trainable_variables
        )
        loss_scale = tf.cast(optimizer.dynamic_scale, tf.float32)
        unscaled_gradients = tuple(
            tf.cast(gradient, tf.float32) / loss_scale
            for gradient in gradients
        )
        clip_norm = exact_registered_global_norm(
            unscaled_gradients, model.trainable_variables
        )
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, tuple(gradients), clip_norm

    recovery_attempts = []
    loss = None
    scaled_gradients = ()
    clip_norm = tf.constant(np.nan, tf.float32)
    for attempt in range(1, 9):
        scale_before_attempt = float(optimizer.dynamic_scale.numpy())
        loss, scaled_gradients, clip_norm = train_step()
        scale_after_attempt = float(optimizer.dynamic_scale.numpy())
        gradients_finite_attempt = all(
            bool(np.isfinite(gradient.numpy()).all())
            for gradient in scaled_gradients
        )
        recovery_attempts.append({
            "attempt": attempt,
            "loss_scale_before": scale_before_attempt,
            "loss_scale_after": scale_after_attempt,
            "scaled_gradients_finite": gradients_finite_attempt,
            "optimizer_iterations": int(optimizer.iterations.numpy()),
        })
        if int(optimizer.iterations.numpy()) == 1:
            break
    scale_before = recovery_attempts[0]["loss_scale_before"]
    scale_after = recovery_attempts[-1]["loss_scale_after"]
    variable_finite = all(
        bool(np.isfinite(variable.numpy()).all())
        for variable in model.trainable_variables
    )
    slot_finite = all(
        bool(np.isfinite(variable.numpy()).all())
        for variable in inner._momentums + inner._velocities
    )
    gradient_finite = all(
        bool(np.isfinite(gradient.numpy()).all())
        for gradient in scaled_gradients
    )
    graph_ops = [
        operation.type
        for operation in train_step.get_concrete_function().graph.get_operations()
    ]
    result = {
        "policy": tf.keras.mixed_precision.global_policy().name,
        "loss": float(loss.numpy()),
        "loss_dtype": str(loss.dtype),
        "loss_finite": bool(np.isfinite(loss.numpy())),
        "scaled_gradients_finite": gradient_finite,
        "variables_finite": variable_finite,
        "slots_finite": slot_finite,
        "model_variable_dtypes": sorted(
            {str(variable.dtype) for variable in model.trainable_variables}
        ),
        "m1_dtypes": sorted(
            {str(variable.dtype) for variable in inner._momentums}
        ),
        "m2_dtypes": sorted(
            {str(variable.dtype) for variable in inner._velocities}
        ),
        "loss_scale_before": scale_before,
        "loss_scale_after": scale_after,
        "loss_scale_recovery_attempts": recovery_attempts,
        "inner_iterations": int(inner.iterations.numpy()),
        "outer_iterations": int(optimizer.iterations.numpy()),
        "exact_clip_norm": float(clip_norm.numpy()),
        "graph_contains_check_finite": any(
            "Finite" in operation for operation in graph_ops
        ),
        "graph_contains_py_function": any(
            operation in {"PyFunc", "EagerPyFunc"} for operation in graph_ops
        ),
    }
    result["pass"] = (
        result["loss_finite"]
        and result["loss_dtype"] == "<dtype: 'float32'>"
        and gradient_finite
        and variable_finite
        and slot_finite
        and result["model_variable_dtypes"] == ["float32"]
        and result["m1_dtypes"] == ["float32"]
        and result["m2_dtypes"] == ["float32"]
        and np.isfinite(result["exact_clip_norm"])
        and result["outer_iterations"] == 1
        and not result["graph_contains_py_function"]
    )
    tf.keras.mixed_precision.set_global_policy("float32")
    return result


def benchmark_mode(
    package_root: Path, mode: str, warmups: int = 5, timed: int = 20
) -> dict:
    configure()
    model, optimizer, batch, _pairs = build_float32_instance(package_root)
    process = psutil.Process(os.getpid())
    graph_started = time.perf_counter()
    if mode == "G1-A":

        @tf.function(autograph=False, jit_compile=False)
        def step():
            with tf.GradientTape() as tape:
                output = model(batch, training=False)
                loss = sparse_cross_entropy(
                    batch["labels"], output["logits"]
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            return loss

        step.get_concrete_function()

        def execute():
            started = time.perf_counter()
            loss = step()
            loss.numpy()
            total = time.perf_counter() - started
            return total, total, 0.0

    else:
        compute = build_compiled_gradient_function(model, training=False)
        compute.get_concrete_function(batch)

        def execute():
            compute_started = time.perf_counter()
            loss, _logits, gradients, finite = compute(batch)
            loss.numpy()
            finite.numpy()
            compute_time = time.perf_counter() - compute_started
            update_started = time.perf_counter()
            apply_gradients_eager_exact(
                optimizer, gradients, model.trainable_variables
            )
            optimizer.iterations.numpy()
            update_time = time.perf_counter() - update_started
            return (
                compute_time + update_time,
                compute_time,
                update_time,
            )

    graph_construction = time.perf_counter() - graph_started
    for _ in range(warmups):
        execute()
    totals = []
    computes = []
    updates = []
    peak_rss = process.memory_info().rss
    for _ in range(timed):
        total, compute_time, update_time = execute()
        totals.append(total)
        computes.append(compute_time)
        updates.append(update_time)
        peak_rss = max(peak_rss, process.memory_info().rss)
    return {
        "mode": mode,
        "warmup_steps": warmups,
        "timed_steps": timed,
        "graph_construction_sec": graph_construction,
        "forward_backward_mean_sec": float(np.mean(computes)),
        "optimizer_update_mean_sec": float(np.mean(updates)),
        "total_step_mean_sec": float(np.mean(totals)),
        "total_step_median_sec": float(np.median(totals)),
        "python_overhead_mean_sec": (
            float(np.mean(updates)) if mode == "H1" else 0.0
        ),
        "host_to_device_transfers_per_step": 0,
        "gpu_synchronizations_observed": 0,
        "peak_host_rss_bytes": int(peak_rss),
        "peak_gpu_memory_bytes": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-child", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--step2-state", type=Path)
    parser.add_argument("--step3-state", type=Path)
    parser.add_argument("--child-result", type=Path)
    args = parser.parse_args()
    if args.checkpoint_child:
        child_checkpoint(args)
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_validation(args.package_root.resolve(), output)
    mixed = mixed_precision_validation(args.package_root.resolve())
    benchmarks = [
        benchmark_mode(args.package_root.resolve(), "G1-A"),
        benchmark_mode(args.package_root.resolve(), "H1"),
    ]
    g1_time = benchmarks[0]["total_step_mean_sec"]
    h1_time = benchmarks[1]["total_step_mean_sec"]
    overhead = (h1_time / g1_time - 1.0) * 100.0
    result = {
        "checkpoint_continuation": checkpoint,
        "mixed_precision": mixed,
        "benchmarks": benchmarks,
        "h1_overhead_percent_relative_to_g1": overhead,
        "pass": checkpoint["pass"] and mixed["pass"],
    }
    (output / "selected_g1_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output / "14_performance_microbenchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(benchmarks[0])
        )
        writer.writeheader()
        writer.writerows(benchmarks)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
