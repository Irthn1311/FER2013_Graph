"""Benchmarks runtime, batch execution, and throughput for Pure-GNN v3.1."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31


def benchmark_batch_sizes(
    batch_sizes: List[int] = [16, 32, 64],
    condition: str = "G1",
    num_warmup: int = 5,
    num_steps: int = 20,
    output_path: Optional[str] = None,
) -> Dict:
    results = {
        "condition": condition,
        "tensorflow_version": tf.__version__,
        "devices": [d.name for d in tf.config.list_physical_devices()],
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "benchmarks": {},
    }

    model = PureGNNv31(condition=condition)

    for bs in batch_sizes:
        print(f"Benchmarking batch size {bs}...")
        dummy_batch = tf.random.uniform([bs, 48, 48, 1], dtype=tf.float32)

        # Warmup
        for _ in range(num_warmup):
            _ = model(dummy_batch, training=False)

        # Timing forward pass
        t0 = time.perf_counter()
        for _ in range(num_steps):
            _ = model(dummy_batch, training=False)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        step_time_ms = (elapsed / num_steps) * 1000.0
        examples_per_sec = (bs * num_steps) / elapsed

        results["benchmarks"][f"B{bs}"] = {
            "batch_size": bs,
            "step_time_ms": round(step_time_ms, 2),
            "examples_per_sec": round(examples_per_sec, 2),
        }
        print(f"  -> B{bs}: {step_time_ms:.2f} ms/step, {examples_per_sec:.2f} samples/sec")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Benchmark saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pure-GNN v3.1")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--condition", type=str, default="G1")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    benchmark_batch_sizes(args.batch_sizes, args.condition, output_path=args.output)
