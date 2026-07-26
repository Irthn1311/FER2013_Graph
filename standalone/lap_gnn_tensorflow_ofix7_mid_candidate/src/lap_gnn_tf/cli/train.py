from __future__ import annotations

import argparse
import json

from lap_gnn_tf.resources import ResourceControls
from lap_gnn_tf.training.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fer-csv", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--graph-workers", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--tf-data-prefetch", type=int, default=2)
    parser.add_argument("--tf-data-parallel-calls", type=int, default=1)
    parser.add_argument("--graph-cache-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xla", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-growth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu-training", action="store_true")
    args = parser.parse_args()
    if args.device.lower().startswith("gpu"):
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU") and not args.allow_cpu_training:
            raise RuntimeError("GPU requested but unavailable; pass --allow-cpu-training explicitly to override")
    controls = ResourceControls(
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        graph_workers=args.graph_workers,
        tf_data_prefetch=args.tf_data_prefetch,
        tf_data_parallel_calls=args.tf_data_parallel_calls,
        graph_cache_size=args.graph_cache_size,
        memory_growth=args.memory_growth,
        mixed_precision=args.mixed_precision,
        xla=args.xla,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
    )
    print(json.dumps(run_training(
        args.config, args.fer_csv, args.prior_root, args.output_root,
        controls, no_resume=args.no_resume,
    ), indent=2))


if __name__ == "__main__":
    main()
