"""Bounded CPU PyTorch AdamW semantics and clipping reference trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)


LR = 3e-4
WEIGHT_DECAY = 1e-3
BETAS = (0.9, 0.999)
EPSILON = 1e-8
MAX_NORM = 5.0


def max_ulp(left: np.ndarray, right: np.ndarray) -> int:
    left_i = np.asarray(left, np.float32).view(np.int32).astype(np.int64)
    right_i = np.asarray(right, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(left_i - right_i))) if left_i.size else 0


def synthetic_cases() -> dict[str, dict]:
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
            "clip": True,
        },
        "two_variables": {
            "parameters": [np.array([1.0, -1.0], np.float32), np.array([[0.5, -0.25]], np.float32)],
            "gradients": [np.array([0.01, -0.02], np.float32), np.array([[6.0, 8.0]], np.float32)],
            "clip": True,
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


def run_case(name: str, case: dict, arrays: dict[str, np.ndarray]) -> dict:
    parameters = [torch.nn.Parameter(torch.from_numpy(np.asarray(value).copy())) for value in case["parameters"]]
    gradients = [torch.from_numpy(np.asarray(value).copy()) for value in case["gradients"]]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    initial_step = int(case.get("initial_step", 0))
    if initial_step:
        for index, parameter in enumerate(parameters):
            optimizer.state[parameter] = {
                "step": torch.tensor(float(initial_step)),
                "exp_avg": torch.from_numpy(np.asarray(case["initial_m"][index]).copy()),
                "exp_avg_sq": torch.from_numpy(np.asarray(case["initial_v"][index]).copy()),
            }
    rows = []
    for local_step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.clone()
        raw_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(parameter.grad, 2.0) for parameter in parameters]),
            2.0,
        )
        if case.get("clip", False):
            measured_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_NORM, foreach=None)
        else:
            measured_norm = raw_norm
        coefficient = min(1.0, MAX_NORM / (float(measured_norm.item()) + 1e-6))
        clipped_gradients = [parameter.grad.detach().clone() for parameter in parameters]
        optimizer.step()
        for index, (parameter, clipped_gradient) in enumerate(zip(parameters, clipped_gradients)):
            state = optimizer.state[parameter]
            prefix = f"{name}_step{local_step}_var{index}"
            arrays[f"{prefix}_gradient"] = clipped_gradient.numpy().copy()
            arrays[f"{prefix}_parameter"] = parameter.detach().numpy().copy()
            arrays[f"{prefix}_momentum"] = state["exp_avg"].detach().numpy().copy()
            arrays[f"{prefix}_velocity"] = state["exp_avg_sq"].detach().numpy().copy()
        rows.append(
            {
                "local_step": local_step,
                "optimizer_step": initial_step + local_step,
                "global_norm": float(measured_norm.item()),
                "clip_coefficient": coefficient,
                "all_finite": all(bool(torch.isfinite(parameter).all()) for parameter in parameters),
            }
        )
    return {"name": name, "variables": len(parameters), "clip": bool(case.get("clip", False)), "steps": rows}


def real_gradient_clip(state_path: Path, gradient_path: Path, arrays: dict[str, np.ndarray]) -> dict:
    with np.load(state_path, allow_pickle=False) as state, np.load(gradient_path, allow_pickle=False) as gradients:
        keys = list(state.files)
        parameters = [torch.nn.Parameter(torch.from_numpy(np.asarray(state[key]).copy())) for key in keys]
        for parameter, key in zip(parameters, keys):
            parameter.grad = torch.from_numpy(np.asarray(gradients[key]).copy())
        foreach_norms = torch._foreach_norm(
            [parameter.grad for parameter in parameters], 2.0
        )
        scalar_norms = [
            torch.linalg.vector_norm(parameter.grad, 2.0) for parameter in parameters
        ]
        arrays["real_foreach_norms"] = torch.stack(foreach_norms).numpy().copy()
        arrays["real_scalar_norms"] = torch.stack(scalar_norms).numpy().copy()
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_NORM, foreach=None)
        coefficient = min(1.0, MAX_NORM / (float(total_norm.item()) + 1e-6))
        for index, parameter in enumerate(parameters):
            arrays[f"real_clipped_gradient_{index:03d}"] = parameter.grad.numpy().copy()
    return {
        "keys": keys,
        "global_norm": float(total_norm.item()),
        "foreach_vs_scalar_norm_max_abs": float(
            torch.max(
                torch.abs(torch.stack(foreach_norms) - torch.stack(scalar_norms))
            ).item()
        ),
        "clip_coefficient": coefficient,
        "clipped_global_norm": float(
            math.sqrt(
                sum(float(torch.sum(parameter.grad.double() ** 2).item()) for parameter in parameters)
            )
        ),
    }


def mode_probe() -> dict:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=LR, weight_decay=WEIGHT_DECAY)
    fused, foreach = _default_to_fused_or_foreach([parameter], False, use_fused=False)
    group = optimizer.param_groups[0]
    return {
        "torch": torch.__version__,
        "device": "cpu",
        "resolved_implementation": "single_tensor",
        "default_resolution": {"fused": fused, "foreach": foreach},
        "param_group": {
            key: group[key]
            for key in [
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
                "decoupled_weight_decay",
            ]
        },
        "foreach_supported_devices": _get_foreach_kernels_supported_devices(),
        "fused_supported_devices": _get_fused_kernels_supported_devices(),
    }


def implementation_mode_deltas() -> list[dict]:
    initial = np.array([1.0, -2.0, 0.125, 8.0], np.float32)
    gradient = np.array([0.25, -0.5, 0.0, 1.25], np.float32)
    outputs = {}
    modes = {
        "single_tensor": {"foreach": False, "fused": False},
        "foreach": {"foreach": True, "fused": False},
        "fused": {"foreach": False, "fused": True},
    }
    for name, kwargs in modes.items():
        try:
            parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
            optimizer = torch.optim.AdamW(
                [parameter],
                lr=LR,
                betas=BETAS,
                eps=EPSILON,
                weight_decay=WEIGHT_DECAY,
                **kwargs,
            )
            parameter.grad = torch.from_numpy(gradient.copy())
            optimizer.step()
            outputs[name] = parameter.detach().numpy().copy()
        except Exception as error:
            outputs[name] = str(error)
    reference = outputs["single_tensor"]
    rows = []
    for name, output in outputs.items():
        if isinstance(output, str):
            rows.append({"mode": name, "available": False, "error": output})
        else:
            rows.append(
                {
                    "mode": name,
                    "available": True,
                    "max_abs_vs_single": float(np.max(np.abs(output.astype(np.float64) - reference.astype(np.float64)))),
                    "max_ulp_vs_single": max_ulp(output, reference),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)

    arrays: dict[str, np.ndarray] = {}
    cases = [run_case(name, case, arrays) for name, case in synthetic_cases().items()]
    real_clip = real_gradient_clip(args.state, args.gradients, arrays)
    result = {
        "mode": mode_probe(),
        "hyperparameters": {
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "beta1": BETAS[0],
            "beta2": BETAS[1],
            "epsilon": EPSILON,
            "global_clip_norm": MAX_NORM,
        },
        "real_gradient_clip": real_clip,
        "synthetic_cases": cases,
        "implementation_modes": implementation_mode_deltas(),
        "real_model_optimizer_updates": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "pytorch_semantics_trace.npz", **arrays)
    (args.output_dir / "pytorch_semantics_trace.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
