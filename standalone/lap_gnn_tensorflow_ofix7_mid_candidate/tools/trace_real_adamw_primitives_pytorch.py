"""Trace PyTorch single-tensor AdamW primitives on copied real-model fixtures."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim.adam as torch_adam


LR = 3e-4
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


def store(arrays: dict[str, np.ndarray], prefix: str, values: list[torch.Tensor]) -> None:
    for index, value in enumerate(values):
        arrays[f"{prefix}_{index:03d}"] = value.detach().cpu().numpy().copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--clipped-gradients", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    keys = metadata["keys"]
    arrays: dict[str, np.ndarray] = {}

    with (
        np.load(args.state, allow_pickle=False) as state,
        np.load(args.clipped_gradients, allow_pickle=False) as clipped,
    ):
        parameters = [
            torch.from_numpy(np.asarray(state[key], np.float32).copy())
            for key in keys
        ]
        gradients = [
            torch.from_numpy(
                np.asarray(clipped[f"real_clipped_gradient_{index:03d}"], np.float32)
                .copy()
            )
            for index in range(len(keys))
        ]

    momentums = [torch.zeros_like(parameter) for parameter in parameters]
    velocities = [torch.zeros_like(parameter) for parameter in parameters]
    store(arrays, "initial_parameter", parameters)
    store(arrays, "clipped_gradient", gradients)
    scalar_rows = []

    with torch.no_grad():
        for step in (1, 2):
            store(arrays, f"step{step}_parameter_before", parameters)
            store(arrays, f"step{step}_momentum_before", momentums)
            store(arrays, f"step{step}_velocity_before", velocities)

            decay_factor = 1.0 - LR * WEIGHT_DECAY
            for parameter in parameters:
                parameter.mul_(decay_factor)
            store(arrays, f"step{step}_after_decay", parameters)

            for momentum, gradient in zip(momentums, gradients):
                momentum.lerp_(gradient, 1.0 - BETA1)
            store(arrays, f"step{step}_momentum_after_lerp", momentums)

            for velocity in velocities:
                velocity.mul_(BETA2)
            store(arrays, f"step{step}_velocity_after_mul", velocities)

            for velocity, gradient in zip(velocities, gradients):
                velocity.addcmul_(gradient, gradient, value=1.0 - BETA2)
            store(arrays, f"step{step}_velocity_after_addcmul", velocities)

            bias_correction1 = 1.0 - BETA1**step
            bias_correction2 = 1.0 - BETA2**step
            step_size = LR / bias_correction1
            bias_correction2_sqrt = bias_correction2**0.5

            square_roots = [velocity.sqrt() for velocity in velocities]
            denominators = [
                square_root.div(bias_correction2_sqrt).add(EPSILON)
                for square_root in square_roots
            ]
            adaptive_ratios = [
                momentum.div(denominator)
                for momentum, denominator in zip(momentums, denominators)
            ]
            explicit_updates = [
                adaptive_ratio.mul(-step_size)
                for adaptive_ratio in adaptive_ratios
            ]
            explicit_add_results = [
                parameter.add(update)
                for parameter, update in zip(parameters, explicit_updates)
            ]

            store(arrays, f"step{step}_sqrt_velocity", square_roots)
            store(arrays, f"step{step}_denominator", denominators)
            store(arrays, f"step{step}_adaptive_ratio", adaptive_ratios)
            store(arrays, f"step{step}_explicit_update", explicit_updates)
            store(arrays, f"step{step}_explicit_add_result", explicit_add_results)

            for parameter, momentum, denominator in zip(
                parameters, momentums, denominators
            ):
                parameter.addcdiv_(momentum, denominator, value=-step_size)
            store(arrays, f"step{step}_parameter_after_addcdiv", parameters)

            scalar_rows.append(
                {
                    "step": step,
                    "decay_factor_python": decay_factor,
                    "decay_factor_float32": float(np.float32(decay_factor)),
                    "one_minus_beta1_python": 1.0 - BETA1,
                    "one_minus_beta1_float32": float(np.float32(1.0 - BETA1)),
                    "one_minus_beta2_python": 1.0 - BETA2,
                    "one_minus_beta2_float32": float(np.float32(1.0 - BETA2)),
                    "bias_correction1": bias_correction1,
                    "bias_correction2": bias_correction2,
                    "step_size_python": step_size,
                    "step_size_float32": float(np.float32(step_size)),
                    "bias_correction2_sqrt_python": bias_correction2_sqrt,
                    "bias_correction2_sqrt_float32": float(
                        np.float32(bias_correction2_sqrt)
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "pytorch_real_primitive_trace.npz", **arrays)
    result = {
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "implementation": "single_tensor_adam",
        "keys": keys,
        "tensor_count": len(keys),
        "steps": scalar_rows,
        "optimizer_updates_executed": 0,
        "fixture_only": True,
        "source": inspect.getsource(torch_adam._single_tensor_adam),
    }
    (args.output_dir / "pytorch_real_primitive_trace.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "torch": result["torch"],
                "tensor_count": result["tensor_count"],
                "steps": scalar_rows,
                "optimizer_updates_executed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
