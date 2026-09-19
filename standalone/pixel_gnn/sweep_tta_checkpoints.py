#!/usr/bin/env python3
"""CLI runner for Comprehensive Validation-Tuned TTA Sweep & Checkpoint Ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDALONE_PIXEL = ROOT / "standalone/pixel_gnn"
if str(STANDALONE_PIXEL) not in sys.path:
    sys.path.insert(0, str(STANDALONE_PIXEL))

from pixel_gnn.sweep_tta import run_comprehensive_checkpoint_sweep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Validation-Tuned TTA Sweep & Combinatorial Checkpoint Ensemble for FER2013 Pixel GNN"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to experiment output directory containing checkpoints/ (defaults to config.paths.output_root)",
    )
    parser.add_argument(
        "--fer-csv",
        type=str,
        default=None,
        help="Path to FER2013 split CSV file (or directory containing val.csv / test.csv)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step size for single-model TTA sweep in [0.0, 1.0] (default: 0.05)",
    )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=50000,
        help="Number of continuous Dirichlet weight combinations to evaluate in combinatorial ensemble (default: 50000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        from pixel_gnn.utils import load_config
        cfg = load_config(config_path)
        output_dir = cfg.get("paths", {}).get("output_root", "outputs")

    run_comprehensive_checkpoint_sweep(
        config_path=config_path,
        output_dir=output_dir,
        fer_csv=args.fer_csv,
        step=args.step,
        num_random_samples=args.num_random_samples,
    )


if __name__ == "__main__":
    main()
