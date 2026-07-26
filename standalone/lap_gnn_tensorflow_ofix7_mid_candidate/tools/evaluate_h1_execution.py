"""Evaluate compiled gradients followed by the eager exact AdamW path."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path


GATE = 2e-8


def worker(args) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf

    sys.path.insert(0, str(args.package_root / "src"))
    from lap_gnn_tf.conversion import load_pytorch_npz
    from lap_gnn_tf.graph.batch import load_golden_batch
    from lap_gnn_tf.model import build_model
    from lap_gnn_tf.training.execution import (
        apply_gradients_eager_exact,
        build_compiled_gradient_function,
    )
    from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    golden = args.package_root / "validation_assets" / "golden"

    def orient(array, transform):
        return array.T if transform == "transpose" else array

    def vector_metrics(actual_parts, expected_parts):
        actual = np.concatenate(
            [np.asarray(value).reshape(-1) for value in actual_parts]
        ).astype(np.float64)
        expected = np.concatenate(
            [np.asarray(value).reshape(-1) for value in expected_parts]
        ).astype(np.float64)
        delta = actual - expected
        expected_norm = max(float(np.linalg.norm(expected)), 1e-12)
        cosine_denominator = max(
            float(np.linalg.norm(actual) * np.linalg.norm(expected)), 1e-12
        )
        return {
            "max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / expected_norm),
            "cosine": float(np.dot(actual, expected) / cosine_denominator),
            "all_finite": bool(np.isfinite(actual).all()),
        }

    def compare_update(optimizer, bindings, reference, step):
        maxima = {"parameter": 0.0, "m1": 0.0, "m2": 0.0}
        for index, binding in enumerate(bindings):
            variable_index = optimizer._get_variable_index(binding.variable)
            values = {
                "parameter": binding.variable.numpy(),
                "m1": optimizer._momentums[variable_index].numpy(),
                "m2": optimizer._velocities[variable_index].numpy(),
            }
            reference_names = {
                "parameter": "parameter",
                "m1": "momentum",
                "m2": "velocity",
            }
            for kind, value in values.items():
                actual = orient(value, binding.transform)
                expected = np.asarray(
                    reference[
                        f"step{step}_{reference_names[kind]}_{index:03d}"
                    ],
                    np.float32,
                )
                maxima[kind] = max(
                    maxima[kind],
                    float(
                        np.max(
                            np.abs(
                                actual.astype(np.float64)
                                - expected.astype(np.float64)
                            )
                        )
                    ),
                )
        return {
            "step": step,
            "parameter_max_abs": maxima["parameter"],
            "m1_max_abs": maxima["m1"],
            "m2_max_abs": maxima["m2"],
            "iterations": int(optimizer.iterations.numpy()),
            "pass": (
                max(maxima.values()) <= GATE
                and int(optimizer.iterations.numpy()) == step
            ),
        }

    repetitions = []
    raw_tensor_rows = None
    with (
        np.load(args.reference, allow_pickle=False) as update_reference,
        np.load(
            golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False
        ) as gradient_reference,
        np.load(args.primitive_trace, allow_pickle=False) as primitive_trace,
    ):
        for repeat in range(1, args.repeats + 1):
            batch = load_golden_batch(str(golden / "graph_batch.npz"))
            model = build_model(batch)
            load_pytorch_npz(model, golden / "model_state.npz", strict=True)
            optimizer = TorchCompatibleAdamW()
            optimizer.build(model.trainable_variables)
            compute = build_compiled_gradient_function(model, training=False)
            iteration_before_compute = int(optimizer.iterations.numpy())
            loss, logits, gradients, finite = compute(batch)
            iteration_after_compute = int(optimizer.iterations.numpy())
            bindings = model.state_bindings()
            by_id = {
                id(variable): gradient
                for variable, gradient in zip(
                    model.trainable_variables, gradients
                )
            }
            actual_raw = []
            expected_raw = []
            tensor_rows = []
            for binding in bindings:
                actual = orient(
                    by_id[id(binding.variable)].numpy(), binding.transform
                )
                expected = np.asarray(
                    gradient_reference[binding.source_key], np.float32
                )
                actual_raw.append(actual)
                expected_raw.append(expected)
                tensor_rows.append({
                    "tensor": binding.source_key,
                    "max_abs": float(
                        np.max(
                            np.abs(
                                actual.astype(np.float64)
                                - expected.astype(np.float64)
                            )
                        )
                    ),
                    "array_exact": bool(np.array_equal(actual, expected)),
                })
            if raw_tensor_rows is None:
                raw_tensor_rows = tensor_rows
            raw_metrics = vector_metrics(actual_raw, expected_raw)

            optimizer.__dict__["_clip_variables"] = list(
                model.trainable_variables
            )
            try:
                clipped = optimizer._clip_gradients(list(gradients))
            finally:
                optimizer.__dict__["_clip_variables"] = None
            actual_clipped = []
            expected_clipped = []
            for index, binding in enumerate(bindings):
                actual_clipped.append(
                    orient(clipped[index].numpy(), binding.transform)
                )
                expected_clipped.append(
                    np.asarray(
                        primitive_trace[f"clipped_gradient_{index:03d}"],
                        np.float32,
                    )
                )
            clipped_metrics = vector_metrics(
                actual_clipped, expected_clipped
            )
            steps = []
            for step in (1, 2):
                apply_gradients_eager_exact(
                    optimizer, gradients, model.trainable_variables
                )
                steps.append(
                    compare_update(
                        optimizer, bindings, update_reference, step
                    )
                )
            repetitions.append({
                "repeat": repeat,
                "loss": float(loss.numpy()),
                "logits_all_finite": bool(
                    np.isfinite(logits.numpy()).all()
                ),
                "raw_gradient_metrics": raw_metrics,
                "clipped_gradient_metrics": clipped_metrics,
                "compiled_compute_finite": bool(finite.numpy()),
                "iterations_before_compute": iteration_before_compute,
                "iterations_after_compute": iteration_after_compute,
                "global_norm": float(
                    optimizer.last_global_gradient_norm.numpy()
                ),
                "clip_coefficient": float(
                    optimizer.last_clip_coefficient.numpy()
                ),
                "steps": steps,
                "pass": (
                    raw_metrics["cosine"] >= 0.99999
                    and clipped_metrics["cosine"] >= 0.99999
                    and bool(finite.numpy())
                    and iteration_before_compute == 0
                    and iteration_after_compute == 0
                    and all(item["pass"] for item in steps)
                ),
            })
            del model, optimizer, compute, gradients, bindings, batch
            tf.keras.backend.clear_session()
            gc.collect()

    result = {
        "tensorflow": tf.__version__,
        "repeats": args.repeats,
        "optimizer_executed_eagerly": True,
        "compiled_stage_updates_optimizer": False,
        "repetitions": repetitions,
        "pass": all(item["pass"] for item in repetitions),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.raw_csv and raw_tensor_rows:
        with args.raw_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(raw_tensor_rows[0])
            )
            writer.writeheader()
            writer.writerows(raw_tensor_rows)


def orchestrate(args) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [("repeated", 10)] + [
        (f"fresh_{index}", 1) for index in range(1, 6)
    ]
    results = []
    for label, repeats in jobs:
        result_path = output / f"h1_{label}.json"
        raw_csv = (
            output / "06_h1_raw_gradient_parity.csv"
            if label == "repeated"
            else output / f"h1_{label}_raw.csv"
        )
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--repeats",
            str(repeats),
            "--package-root",
            str(args.package_root.resolve()),
            "--reference",
            str(args.reference.resolve()),
            "--primitive-trace",
            str(args.primitive_trace.resolve()),
            "--result",
            str(result_path),
            "--raw-csv",
            str(raw_csv),
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"H1 {label} failed:\n{completed.stdout}\n{completed.stderr}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["process_label"] = label
        results.append(result)
    repetitions = [
        repeat for result in results for repeat in result["repetitions"]
    ]
    step_rows = []
    for step in (1, 2):
        values = [item["steps"][step - 1] for item in repetitions]
        step_rows.append({
            "step": step,
            "parameter_max_abs": max(
                item["parameter_max_abs"] for item in values
            ),
            "m1_max_abs": max(item["m1_max_abs"] for item in values),
            "m2_max_abs": max(item["m2_max_abs"] for item in values),
            "pass": all(item["pass"] for item in values),
        })
    summary = {
        "in_process_repeats": 10,
        "fresh_processes": 5,
        "total_two_step_repetitions": len(repetitions),
        "raw_gradient_worst": {
            "max_abs": max(
                item["raw_gradient_metrics"]["max_abs"]
                for item in repetitions
            ),
            "relative_l2": max(
                item["raw_gradient_metrics"]["relative_l2"]
                for item in repetitions
            ),
            "minimum_cosine": min(
                item["raw_gradient_metrics"]["cosine"]
                for item in repetitions
            ),
        },
        "clipped_gradient_worst": {
            "max_abs": max(
                item["clipped_gradient_metrics"]["max_abs"]
                for item in repetitions
            ),
            "relative_l2": max(
                item["clipped_gradient_metrics"]["relative_l2"]
                for item in repetitions
            ),
            "minimum_cosine": min(
                item["clipped_gradient_metrics"]["cosine"]
                for item in repetitions
            ),
        },
        "steps": step_rows,
        "optimizer_executed_eagerly": True,
        "compiled_stage_updates_optimizer": False,
        "h1_repeated_pass": results[0]["pass"],
        "h1_fresh_process_pass": all(
            result["pass"] for result in results[1:]
        ),
        "h1_pass": all(result["pass"] for result in results),
    }
    (output / "h1_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--primitive-trace", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        orchestrate(args)


if __name__ == "__main__":
    main()
