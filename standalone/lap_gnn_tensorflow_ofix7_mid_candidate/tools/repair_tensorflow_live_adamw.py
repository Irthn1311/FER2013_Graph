"""Two-step live TorchCompatibleAdamW probe using fixed golden gradients."""

from __future__ import annotations

import argparse
import json
import os
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


def source_orientation(array: np.ndarray, transform: str) -> np.ndarray:
    return array.T if transform == "transpose" else array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    golden = args.package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    with tf.device("/CPU:0"):
        model = build_model(batch)
        load_pytorch_npz(model, golden / "model_state.npz", strict=True)
        optimizer = TorchCompatibleAdamW(
            learning_rate=3e-4,
            weight_decay=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=None,
        )
        optimizer.build(model.trainable_variables)

    bindings = model.state_bindings()
    binding_by_id = {id(binding.variable): binding for binding in bindings}
    with np.load(golden / "pytorch_gradients_eval_ce.npz", allow_pickle=False) as source:
        gradient_by_id = {}
        for binding in bindings:
            gradient = np.asarray(source[binding.source_key])
            if binding.transform == "transpose":
                gradient = gradient.T
            gradient_by_id[id(binding.variable)] = tf.convert_to_tensor(gradient, tf.float32)

    arrays = {}
    steps = []
    for step_index in (1, 2):
        gradients_and_variables = [
            (gradient_by_id[id(variable)], variable)
            for variable in model.trainable_variables
        ]
        optimizer.apply_gradients(gradients_and_variables)
        all_parameter_finite = True
        all_slot_finite = True
        for index, binding in enumerate(bindings):
            variable_index = optimizer._get_variable_index(binding.variable)
            parameter = binding.variable.numpy()
            momentum = optimizer._momentums[variable_index].numpy()
            velocity = optimizer._velocities[variable_index].numpy()
            parameter = source_orientation(parameter, binding.transform)
            momentum = source_orientation(momentum, binding.transform)
            velocity = source_orientation(velocity, binding.transform)
            arrays[f"step{step_index}_parameter_{index:03d}"] = parameter
            arrays[f"step{step_index}_momentum_{index:03d}"] = momentum
            arrays[f"step{step_index}_velocity_{index:03d}"] = velocity
            all_parameter_finite &= bool(np.isfinite(parameter).all())
            all_slot_finite &= bool(np.isfinite(momentum).all() and np.isfinite(velocity).all())
        steps.append({
            "step": step_index,
            "optimizer_iterations": int(optimizer.iterations.numpy()),
            "all_parameters_finite": all_parameter_finite,
            "all_slots_finite": all_slot_finite,
        })
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    metadata = {
        "framework": "tensorflow",
        "tensorflow": tf.__version__,
        "keys": [binding.source_key for binding in bindings],
        "steps": steps,
        "optimizer_updates_executed": 2,
        "model_variables": len(model.trainable_variables),
        "optimizer_variables": len(optimizer.variables),
    }
    args.output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
