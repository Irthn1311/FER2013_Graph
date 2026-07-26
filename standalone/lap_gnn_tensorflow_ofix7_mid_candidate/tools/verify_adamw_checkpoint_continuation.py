"""Verify full-model AdamW checkpoint continuation in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def gradients_for_model(package_root: Path, model):
    golden = package_root / "validation_assets" / "golden"
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
    return gradients_and_variables


def build_instance(package_root: Path):
    golden = package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    model = build_model(batch)
    load_pytorch_npz(model, golden / "model_state.npz", strict=True)
    optimizer = TorchCompatibleAdamW()
    optimizer.build(model.trainable_variables)
    gradients_and_variables = gradients_for_model(package_root, model)
    return model, optimizer, gradients_and_variables


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


def compare(
    expected_path: Path, actual: dict[str, np.ndarray]
) -> dict[str, int | float | bool]:
    maximum = 0.0
    exact = 0
    missing: list[str] = []
    unexpected: list[str] = []
    with np.load(expected_path, allow_pickle=False) as expected:
        expected_keys = set(expected.files)
        actual_keys = set(actual)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        for key in sorted(expected_keys & actual_keys):
            reference = np.asarray(expected[key])
            candidate = np.asarray(actual[key])
            if reference.shape != candidate.shape:
                maximum = float("inf")
                continue
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
        "exact_arrays": exact,
        "array_count": len(actual),
        "missing_arrays": len(missing),
        "unexpected_arrays": len(unexpected),
        "pass": (
            maximum == 0.0
            and exact == len(actual)
            and not missing
            and not unexpected
        ),
    }


def configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()


def child_main(args) -> None:
    configure_tensorflow()
    model = tf.keras.models.load_model(args.checkpoint)
    optimizer = model.optimizer
    gradients_and_variables = gradients_for_model(args.package_root, model)
    restored = compare(args.step2_state, capture(model, optimizer))
    optimizer.apply_gradients(gradients_and_variables)
    continued = compare(args.step3_state, capture(model, optimizer))
    result = {
        "fresh_process_restore": restored,
        "fresh_process_continuation": continued,
        "iterations_after_restore": 2,
        "iterations_after_continuation": int(optimizer.iterations.numpy()),
        "pass": (
            restored["pass"]
            and continued["pass"]
            and int(optimizer.iterations.numpy()) == 3
        ),
    }
    args.child_output.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["pass"]:
        raise SystemExit(1)


def parent_main(args) -> None:
    configure_tensorflow()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer, gradients_and_variables = build_instance(args.package_root)
    model.compile(optimizer=optimizer, run_eagerly=True)
    for _ in range(2):
        optimizer.apply_gradients(gradients_and_variables)

    step2_path = args.output_dir / "checkpoint_step2_state.npz"
    np.savez_compressed(step2_path, **capture(model, optimizer))
    checkpoint_path = args.output_dir / "adamw_step2.keras"
    model.save(checkpoint_path, include_optimizer=True)

    optimizer.apply_gradients(gradients_and_variables)
    step3_path = args.output_dir / "uninterrupted_step3_state.npz"
    np.savez_compressed(step3_path, **capture(model, optimizer))

    child_output = args.output_dir / "checkpoint_child_result.json"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--child",
        "--package-root",
        str(args.package_root),
        "--checkpoint",
        str(checkpoint_path),
        "--step2-state",
        str(step2_path),
        "--step3-state",
        str(step3_path),
        "--child-output",
        str(child_output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        result = {
            "pass": False,
            "fresh_process_returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    else:
        result = json.loads(child_output.read_text(encoding="utf-8"))
        result.update({
            "checkpoint_prefix": str(checkpoint_path),
            "full_model_trainable_variables": len(model.trainable_variables),
            "optimizer_variables": len(optimizer.variables),
            "parent_iterations_after_continuation": int(
                optimizer.iterations.numpy()
            ),
            "fresh_process_returncode": completed.returncode,
        })
    output = args.output_dir / "checkpoint_continuation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--step2-state", type=Path)
    parser.add_argument("--step3-state", type=Path)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child:
        child_main(args)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        parent_main(args)


if __name__ == "__main__":
    main()
