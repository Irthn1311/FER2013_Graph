"""Repository wrapper for the installed benchmark module."""

import argparse
import json

from pure_gnn_v31.tools.benchmark_runtime import benchmark_batch_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pure-GNN v3.1")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--condition", default="G1")
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--technical-seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(benchmark_batch_sizes(
        args.batch_sizes,
        args.condition,
        num_warmup=args.warmup,
        num_steps=args.steps,
        output_path=args.output,
        technical_seed=args.technical_seed,
    ), indent=2))
