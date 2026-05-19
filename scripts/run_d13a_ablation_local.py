"""Local launcher for D13A ablation configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


RUNS = {
    "k64": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k64.yaml", "d13a_edgeaware_lite_localpool_k64"),
    "k256": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k256.yaml", "d13a_edgeaware_lite_localpool_k256"),
    "temp07": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_temp07.yaml", "d13a_edgeaware_lite_localpool_k144_temp07"),
    "temp05": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_temp05.yaml", "d13a_edgeaware_lite_localpool_k144_temp05"),
    "anneal_1to05": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_anneal_1to05.yaml", "d13a_edgeaware_lite_localpool_k144_anneal_1to05"),
    "no_aux": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_no_aux.yaml", "d13a_edgeaware_lite_localpool_k144_no_aux"),
    "compact_balance_x2": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_compact_balance_x2.yaml", "d13a_edgeaware_lite_localpool_k144_compact_balance_x2"),
    "seed2": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_seed2.yaml", "d13a_edgeaware_lite_localpool_k144_seed2"),
    "seed3": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_seed3.yaml", "d13a_edgeaware_lite_localpool_k144_seed3"),
    "lr1e4": ("configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_lr1e4.yaml", "d13a_edgeaware_lite_localpool_k144_lr1e4"),
    "baseline": ("configs/d13/d13a_edgeaware_lite_localpool_k144.yaml", "d13a_edgeaware_lite_localpool_k144"),
    "gine": ("configs/d13/d13a_gine_localpool_k144.yaml", "d13a_gine_localpool_k144"),
}


def build_command(args: argparse.Namespace) -> list[str]:
    if args.run_key not in RUNS:
        valid = " ".join(sorted(RUNS))
        raise SystemExit(f"Unknown run_key={args.run_key!r}. Valid: {valid}")
    config, run_name = RUNS[args.run_key]
    output_name = run_name if not args.output_suffix else f"{run_name}_{args.output_suffix}"
    output_dir = Path(args.output_root) / output_name
    cmd = [
        sys.executable,
        "training/train_d13.py",
        "--config",
        config,
        "--output_dir",
        str(output_dir),
    ]
    if args.environment:
        cmd += ["--environment", args.environment]
    if args.device:
        cmd += ["--device", args.device]
    if args.max_epochs_override is not None:
        cmd += ["--epochs", str(args.max_epochs_override)]
    if args.max_train_batches is not None:
        cmd += ["--max_train_batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        cmd += ["--max_val_batches", str(args.max_val_batches)]
    if args.max_test_batches is not None:
        cmd += ["--max_test_batches", str(args.max_test_batches)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_key", required=True, choices=sorted(RUNS))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_epochs_override", type=int, default=None)
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument("--output_root", default="outputs/d13_hierarchical_reduction/ablations")
    parser.add_argument("--environment", "--env", default=None, choices=["local", "kaggle"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    args = parser.parse_args()
    cmd = build_command(args)
    print(" ".join(str(x) for x in cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

