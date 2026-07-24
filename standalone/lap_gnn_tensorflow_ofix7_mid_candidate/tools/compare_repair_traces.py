"""Compare bounded PyTorch and TensorFlow repair traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(actual, np.float32) - np.asarray(expected, np.float32)
    max_abs = float(np.max(np.abs(delta)))
    relative_l2 = float(
        np.linalg.norm(delta.astype(np.float64))
        / max(np.linalg.norm(np.asarray(expected, np.float64)), 1e-12)
    )
    return max_abs, relative_l2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--tensorflow-segment", type=Path, required=True)
    parser.add_argument("--tensorflow-loop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (
        np.load(args.pytorch, allow_pickle=False) as pytorch,
        np.load(args.tensorflow_segment, allow_pickle=False) as segment,
        np.load(args.tensorflow_loop, allow_pickle=False) as loop,
    ):
        names = sorted(
            name for name in pytorch.files
            if f"trace_{name}" in segment.files and f"trace_{name}" in loop.files
        )
        rows = []
        for name in names:
            segment_max, segment_l2 = metrics(segment[f"trace_{name}"], pytorch[name])
            loop_max, loop_l2 = metrics(loop[f"trace_{name}"], pytorch[name])
            rows.append({
                "tensor": name,
                "shape": "x".join(str(value) for value in pytorch[name].shape),
                "segment_max_abs": segment_max,
                "segment_relative_l2": segment_l2,
                "torch_loop_max_abs": loop_max,
                "torch_loop_relative_l2": loop_l2,
                "max_abs_improvement": segment_max - loop_max,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
