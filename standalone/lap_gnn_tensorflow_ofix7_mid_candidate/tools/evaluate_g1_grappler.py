"""Evaluate the three registered Grappler configurations for AdamW parity."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


GATE = 2e-8
CONFIGURATIONS = {
    "G1-A": {
        "arithmetic_optimization": False,
        "remapping": False,
    },
    "G1-B": {
        "arithmetic_optimization": False,
        "remapping": False,
        "function_optimization": False,
        "dependency_optimization": False,
    },
    "G1-C": {
        "disable_meta_optimizer": True,
    },
}


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
    from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW

    settings = CONFIGURATIONS[args.mode]
    tf.config.optimizer.set_experimental_options(settings)
    effective = tf.config.optimizer.get_experimental_options()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    golden = args.package_root / "validation_assets" / "golden"

    def build_instance():
        batch = load_golden_batch(str(golden / "graph_batch.npz"))
        model = build_model(batch)
        load_pytorch_npz(model, golden / "model_state.npz", strict=True)
        optimizer = TorchCompatibleAdamW()
        optimizer.build(model.trainable_variables)
        bindings = model.state_bindings()
        with np.load(
            golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False
        ) as source:
            gradient_by_id = {}
            for binding in bindings:
                gradient = np.asarray(source[binding.source_key], np.float32)
                if binding.transform == "transpose":
                    gradient = gradient.T
                gradient_by_id[id(binding.variable)] = tf.constant(
                    gradient, tf.float32
                )
        pairs = [
            (gradient_by_id[id(variable)], variable)
            for variable in model.trainable_variables
        ]
        return model, optimizer, bindings, pairs

    def compare(optimizer, bindings, reference, step):
        maxima = {"parameter": 0.0, "momentum": 0.0, "velocity": 0.0}
        exact = {"parameter": 0, "momentum": 0, "velocity": 0}
        for index, binding in enumerate(bindings):
            variable_index = optimizer._get_variable_index(binding.variable)
            values = {
                "parameter": binding.variable.numpy(),
                "momentum": optimizer._momentums[variable_index].numpy(),
                "velocity": optimizer._velocities[variable_index].numpy(),
            }
            for kind, value in values.items():
                actual = value.T if binding.transform == "transpose" else value
                expected = np.asarray(
                    reference[f"step{step}_{kind}_{index:03d}"], np.float32
                )
                maximum = float(
                    np.max(
                        np.abs(
                            actual.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                )
                maxima[kind] = max(maxima[kind], maximum)
                exact[kind] += int(np.array_equal(actual, expected))
        passed = (
            all(value <= GATE for value in maxima.values())
            and int(optimizer.iterations.numpy()) == step
        )
        return {
            "step": step,
            "parameter_max_abs": maxima["parameter"],
            "m1_max_abs": maxima["momentum"],
            "m2_max_abs": maxima["velocity"],
            "parameter_exact_tensors": exact["parameter"],
            "m1_exact_tensors": exact["momentum"],
            "m2_exact_tensors": exact["velocity"],
            "iterations": int(optimizer.iterations.numpy()),
            "variable_count": len(bindings),
            "pass": passed,
        }

    repetitions = []
    graph_audit = None
    with np.load(args.reference, allow_pickle=False) as reference:
        for repeat in range(1, args.repeats + 1):
            model, optimizer, bindings, pairs = build_instance()

            @tf.function(autograph=False, jit_compile=False)
            def apply_optimizer():
                optimizer.apply_gradients(pairs)
                return optimizer.iterations

            concrete = apply_optimizer.get_concrete_function()
            if graph_audit is None:
                operations = [operation.type for operation in concrete.graph.get_operations()]
                graph_audit = {
                    "operation_count": len(operations),
                    "operation_type_counts": dict(sorted(Counter(operations).items())),
                    "contains_py_function": any(
                        value in {"PyFunc", "EagerPyFunc"} for value in operations
                    ),
                    "contains_xla": any("Xla" in value for value in operations),
                }
            steps = []
            for step in (1, 2):
                apply_optimizer()
                steps.append(compare(optimizer, bindings, reference, step))
            repetitions.append({
                "repeat": repeat,
                "steps": steps,
                "pass": all(item["pass"] for item in steps),
            })
            del model, optimizer, bindings, pairs, concrete, apply_optimizer
            tf.keras.backend.clear_session()
            gc.collect()

    result = {
        "configuration": args.mode,
        "requested_optimizer_options": settings,
        "effective_optimizer_options": effective,
        "tensorflow": tf.__version__,
        "repeats": args.repeats,
        "repetitions": repetitions,
        "graph_audit": graph_audit,
        "pass": all(item["pass"] for item in repetitions),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "configuration": args.mode,
        "repeats": args.repeats,
        "pass": result["pass"],
    }))


def orchestrate(args) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    process_results = []
    rows = []
    for mode in CONFIGURATIONS:
        mode_slug = mode.lower().replace("-", "_")
        jobs = [("repeated", 10)] + [
            (f"fresh_{index}", 1) for index in range(1, 6)
        ]
        mode_results = []
        for label, repeats in jobs:
            result_path = output / f"{mode_slug}_{label}.json"
            if not result_path.exists():
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--mode",
                    mode,
                    "--repeats",
                    str(repeats),
                    "--package-root",
                    str(args.package_root.resolve()),
                    "--reference",
                    str(args.reference.resolve()),
                    "--result",
                    str(result_path),
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
                        f"{mode} {label} failed:\n"
                        f"{completed.stdout}\n{completed.stderr}"
                    )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result["configuration"] != mode
                or result["repeats"] != repeats
            ):
                raise RuntimeError(
                    f"Existing result does not match {mode} {label}"
                )
            result["process_label"] = label
            mode_results.append(result)
        all_repetitions = [
            repeat
            for result in mode_results
            for repeat in result["repetitions"]
        ]
        for step in (1, 2):
            step_rows = [
                repeat["steps"][step - 1] for repeat in all_repetitions
            ]
            rows.append({
                "configuration": mode,
                "step": step,
                "parameter_max_abs": max(
                    item["parameter_max_abs"] for item in step_rows
                ),
                "m1_max_abs": max(item["m1_max_abs"] for item in step_rows),
                "m2_max_abs": max(item["m2_max_abs"] for item in step_rows),
                "iterations_exact": all(
                    item["iterations"] == step for item in step_rows
                ),
                "variable_count_exact": all(
                    item["variable_count"] == 127 for item in step_rows
                ),
                "repetitions": len(step_rows),
                "pass": all(item["pass"] for item in step_rows),
            })
        mode_pass = all(repeat["pass"] for repeat in all_repetitions)
        process_results.append({
            "configuration": mode,
            "requested_optimizer_options": CONFIGURATIONS[mode],
            "effective_optimizer_options": mode_results[0][
                "effective_optimizer_options"
            ],
            "graph_audit": mode_results[0]["graph_audit"],
            "in_process_traces": 10,
            "fresh_processes": 5,
            "total_two_step_repetitions": len(all_repetitions),
            "pass": mode_pass,
        })
    csv_path = output / "03_g1_grappler_configurations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "configurations": process_results,
        "rows": rows,
        "g1_pass": any(item["pass"] for item in process_results),
        "selected_configuration": next(
            (
                item["configuration"]
                for item in process_results
                if item["pass"]
            ),
            None,
        ),
    }
    (output / "g1_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=list(CONFIGURATIONS))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--result", type=Path)
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
