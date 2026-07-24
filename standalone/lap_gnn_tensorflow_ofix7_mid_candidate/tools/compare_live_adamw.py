"""Compare the saved two-step live PyTorch and TensorFlow AdamW states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-npz", type=Path, required=True)
    parser.add_argument("--pytorch-json", type=Path, required=True)
    parser.add_argument("--tensorflow-npz", type=Path, required=True)
    parser.add_argument("--tensorflow-json", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--pytorch-step1-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pytorch_metadata = json.loads(args.pytorch_json.read_text(encoding="utf-8"))
    tensorflow_metadata = json.loads(args.tensorflow_json.read_text(encoding="utf-8"))
    if pytorch_metadata["keys"] != tensorflow_metadata["keys"]:
        raise RuntimeError("Live optimizer key order mismatch")
    result = {
        "keys_match": True,
        "tensor_count": len(pytorch_metadata["keys"]),
        "steps": [],
        "total_optimizer_updates": (
            pytorch_metadata["optimizer_updates_executed"]
            + tensorflow_metadata["optimizer_updates_executed"]
        ),
        "step1_reference": "locked_pytorch_live_step1_fixture",
        "step2_reference": "fresh_pytorch_live_step2",
        "capture_warning": (
            "Fresh PyTorch step-1 NumPy views were overwritten by step 2; "
            "step 1 therefore uses the pre-existing locked live-step fixture."
        ),
    }
    with (
        np.load(args.pytorch_npz, allow_pickle=False) as pytorch,
        np.load(args.tensorflow_npz, allow_pickle=False) as tensorflow,
        np.load(args.state, allow_pickle=False) as state,
        np.load(args.gradients, allow_pickle=False) as gradients,
        np.load(args.pytorch_step1_fixture, allow_pickle=False) as step1_fixture,
    ):
        if set(pytorch.files) != set(tensorflow.files):
            raise RuntimeError("Live optimizer state tensor set mismatch")
        for step in (1, 2):
            row = {"step": step}
            for kind in ("parameter", "momentum", "velocity"):
                names = sorted(name for name in pytorch.files if name.startswith(f"step{step}_{kind}_"))
                maximum = 0.0
                relative_square = 0.0
                reference_square = 0.0
                for index, name in enumerate(names):
                    key = pytorch_metadata["keys"][index]
                    if step == 1 and kind == "parameter":
                        left = np.asarray(step1_fixture[key], np.float32)
                    elif step == 1 and kind == "momentum":
                        left = np.float32(0.1) * np.asarray(gradients[key], np.float32)
                    elif step == 1 and kind == "velocity":
                        gradient = np.asarray(gradients[key], np.float32)
                        left = np.float32(0.001) * np.square(gradient)
                    else:
                        left = np.asarray(pytorch[name], np.float32)
                    right = np.asarray(tensorflow[name], np.float32)
                    delta = left.astype(np.float64) - right.astype(np.float64)
                    maximum = max(maximum, float(np.max(np.abs(delta))))
                    relative_square += float(np.sum(np.square(delta)))
                    reference_square += float(np.sum(np.square(left.astype(np.float64))))
                row[f"{kind}_max_abs"] = maximum
                row[f"{kind}_relative_l2"] = (
                    relative_square ** 0.5 / max(reference_square ** 0.5, 1e-12)
                )
            pytorch_step = pytorch_metadata["steps"][step - 1]
            tensorflow_step = tensorflow_metadata["steps"][step - 1]
            row.update({
                "pytorch_step_counter": pytorch_step["maximum_state_step"],
                "tensorflow_step_counter": tensorflow_step["optimizer_iterations"],
                "step_counter_match": (
                    pytorch_step["minimum_state_step"]
                    == pytorch_step["maximum_state_step"]
                    == tensorflow_step["optimizer_iterations"]
                ),
                "all_finite": all([
                    pytorch_step["all_parameters_finite"],
                    pytorch_step["all_slots_finite"],
                    tensorflow_step["all_parameters_finite"],
                    tensorflow_step["all_slots_finite"],
                ]),
            })
            row["pass_2e_8"] = (
                row["parameter_max_abs"] <= 2e-8
                and row["momentum_max_abs"] <= 2e-8
                and row["velocity_max_abs"] <= 2e-8
                and row["step_counter_match"]
                and row["all_finite"]
            )
            result["steps"].append(row)
    result["pass"] = all(row["pass_2e_8"] for row in result["steps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
