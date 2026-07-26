"""Development-only PyTorch classifier probe for the TensorFlow repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lap_gnn.model.classifier import D16Classifier


def summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.asarray(actual, np.float32) - np.asarray(expected, np.float32)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta.astype(np.float64))
            / max(np.linalg.norm(np.asarray(expected, np.float64)), 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-state", type=Path, required=True)
    parser.add_argument("--probe-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = D16Classifier(480, 192, 7, 0.2).cpu().eval()
    with np.load(args.model_state, allow_pickle=False) as source:
        state = {
            key.removeprefix("classifier."): torch.from_numpy(source[key].copy())
            for key in source.files
            if key.startswith("classifier.")
        }
    model.load_state_dict(state, strict=True)

    with np.load(args.probe_inputs, allow_pickle=False) as probe:
        expected_input = probe["expected_classifier_input"]
        tensorflow_input = probe["tensorflow_classifier_input"]
        expected_logits = probe["expected_logits"]
    with torch.no_grad():
        expected_input_logits = model(torch.from_numpy(expected_input)).numpy()
        tensorflow_input_logits = model(torch.from_numpy(tensorflow_input)).numpy()
    result = {
        "torch": torch.__version__,
        "golden_input_against_golden_logits": summary(expected_input_logits, expected_logits),
        "tensorflow_input_against_golden_logits": summary(tensorflow_input_logits, expected_logits),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
