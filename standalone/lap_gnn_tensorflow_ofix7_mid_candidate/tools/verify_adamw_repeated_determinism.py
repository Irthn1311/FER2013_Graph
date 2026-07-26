"""Repeat the repaired TensorFlow AdamW trace against one fixed PyTorch reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


GATE = 2e-8


def source_orientation(array: np.ndarray, transform: str) -> np.ndarray:
    return array.T if transform == "transpose" else array


def build_instance(package_root: Path):
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
        gradient_by_id = {}
        for binding in bindings:
            gradient = np.asarray(source[binding.source_key], np.float32)
            if binding.transform == "transpose":
                gradient = gradient.T
            gradient_by_id[id(binding.variable)] = tf.constant(
                gradient, tf.float32
            )
    gradients_and_variables = [
        (gradient_by_id[id(variable)], variable)
        for variable in model.trainable_variables
    ]
    return model, optimizer, bindings, gradients_and_variables


def compare_capture(
    optimizer,
    bindings,
    reference: np.lib.npyio.NpzFile,
    step: int,
) -> dict:
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
            actual = source_orientation(value, binding.transform)
            expected = np.asarray(
                reference[f"step{step}_{kind}_{index:03d}"], np.float32
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
            exact[kind] += int(np.array_equal(actual, expected))
    return {
        "parameter_max_abs": maxima["parameter"],
        "momentum_max_abs": maxima["momentum"],
        "velocity_max_abs": maxima["velocity"],
        "parameter_exact_tensors": exact["parameter"],
        "momentum_exact_tensors": exact["momentum"],
        "velocity_exact_tensors": exact["velocity"],
        "iterations": int(optimizer.iterations.numpy()),
        "pass": (
            maxima["parameter"] <= GATE
            and maxima["momentum"] <= GATE
            and maxima["velocity"] <= GATE
            and int(optimizer.iterations.numpy()) == step
        ),
    }


def one_run(package_root: Path, reference_path: Path, graph: bool) -> list[dict]:
    model, optimizer, bindings, gradients_and_variables = build_instance(package_root)

    def apply_step():
        optimizer.apply_gradients(gradients_and_variables)

    apply = tf.function(apply_step) if graph else apply_step
    rows = []
    with np.load(reference_path, allow_pickle=False) as reference:
        for step in (1, 2):
            apply()
            rows.append(compare_capture(optimizer, bindings, reference, step))
    return rows


def repeated_in_process(
    package_root: Path,
    reference_path: Path,
    graph: bool,
    repeats: int,
) -> list[list[dict]]:
    model, optimizer, bindings, gradients_and_variables = build_instance(package_root)
    initial_model = [variable.numpy().copy() for variable in model.trainable_variables]
    initial_optimizer = [variable.numpy().copy() for variable in optimizer.variables]

    def apply_step():
        optimizer.apply_gradients(gradients_and_variables)

    apply = tf.function(apply_step) if graph else apply_step
    outputs = []
    with np.load(reference_path, allow_pickle=False) as reference:
        for _ in range(repeats):
            for variable, value in zip(model.trainable_variables, initial_model):
                variable.assign(value)
            for variable, value in zip(optimizer.variables, initial_optimizer):
                variable.assign(value)
            rows = []
            for step in (1, 2):
                apply()
                rows.append(compare_capture(optimizer, bindings, reference, step))
            outputs.append(rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--single-graph", action="store_true")
    parser.add_argument("--single-output", type=Path)
    args = parser.parse_args()

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    if args.single:
        rows = one_run(
            args.package_root,
            args.reference,
            graph=args.single_graph,
        )
        result = {"steps": rows, "pass": all(row["pass"] for row in rows)}
        args.single_output.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if not result["pass"]:
            raise SystemExit(1)
        return

    rows = []
    for mode, repeats in (("eager", 10), ("tf_function", 10)):
        mode_results = repeated_in_process(
            args.package_root,
            args.reference,
            graph=(mode == "tf_function"),
            repeats=repeats,
        )
        for repeat, result in enumerate(mode_results, start=1):
            for step, step_result in enumerate(result, start=1):
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "step": step,
                        **step_result,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fresh_process_results = []
    for repeat in range(1, 6):
        with tempfile.TemporaryDirectory(
            prefix="lap_gnn_tf_adamw_", dir=args.output_dir
        ) as temporary:
            output = Path(temporary) / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--package-root",
                    str(args.package_root),
                    "--reference",
                    str(args.reference),
                    "--output-dir",
                    str(args.output_dir),
                    "--single",
                    "--single-output",
                    str(output),
                ],
                check=False,
                text=True,
                capture_output=True,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Fresh process {repeat} failed: {completed.stderr}"
                )
            result = json.loads(output.read_text(encoding="utf-8"))
            fresh_process_results.append(result)
            for step, step_result in enumerate(result["steps"], start=1):
                rows.append(
                    {
                        "mode": "fresh_process",
                        "repeat": repeat,
                        "step": step,
                        **step_result,
                    }
                )

    csv_path = args.output_dir / "14_repeated_determinism.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "eager_repeats": 10,
        "tf_function_repeats": 10,
        "fresh_process_repeats": 5,
        "eager_pass": all(
            row["pass"] for row in rows if row["mode"] == "eager"
        ),
        "tf_function_pass": all(
            row["pass"] for row in rows if row["mode"] == "tf_function"
        ),
        "fresh_process_pass": all(
            row["pass"] for row in rows if row["mode"] == "fresh_process"
        ),
        "tensorflow_optimizer_updates": 50,
        "additional_pytorch_optimizer_updates": 0,
    }
    result["pass"] = (
        result["eager_pass"]
        and result["tf_function_pass"]
        and result["fresh_process_pass"]
    )
    (args.output_dir / "repeated_determinism.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
