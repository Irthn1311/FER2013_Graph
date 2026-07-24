"""Development-only two-step live PyTorch AdamW reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(42)
    with (
        np.load(args.state, allow_pickle=False) as state,
        np.load(args.gradients, allow_pickle=False) as gradients,
    ):
        keys = list(state.files)
        parameters = [
            torch.nn.Parameter(torch.from_numpy(np.asarray(state[key]).copy()))
            for key in keys
        ]
        fixed_gradients = [
            torch.from_numpy(np.asarray(gradients[key]).copy())
            for key in keys
        ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        foreach=False,
        fused=False,
    )
    arrays = {}
    steps = []
    for step_index in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, fixed_gradients):
            parameter.grad = gradient.clone()
        optimizer.step()
        state_steps = []
        for index, (key, parameter) in enumerate(zip(keys, parameters)):
            slot = optimizer.state[parameter]
            arrays[f"step{step_index}_parameter_{index:03d}"] = parameter.detach().numpy()
            arrays[f"step{step_index}_momentum_{index:03d}"] = slot["exp_avg"].detach().numpy()
            arrays[f"step{step_index}_velocity_{index:03d}"] = slot["exp_avg_sq"].detach().numpy()
            state_steps.append(float(slot["step"].item()))
        steps.append({
            "step": step_index,
            "minimum_state_step": min(state_steps),
            "maximum_state_step": max(state_steps),
            "all_parameters_finite": all(bool(torch.isfinite(value).all()) for value in parameters),
            "all_slots_finite": all(
                bool(torch.isfinite(optimizer.state[value]["exp_avg"]).all())
                and bool(torch.isfinite(optimizer.state[value]["exp_avg_sq"]).all())
                for value in parameters
            ),
        })
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    metadata = {
        "framework": "pytorch",
        "torch": torch.__version__,
        "keys": keys,
        "steps": steps,
        "optimizer_updates_executed": 2,
    }
    args.output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
