#!/usr/bin/env python3
"""Runner for Pure Pixel Neighbor Attention + Learned Motif Prototypes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STANDALONE_ROOT = REPO_ROOT / "standalone/pixel_neighbor_motif"
TF_STANDALONE = REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"

sys.path.insert(0, str(TF_STANDALONE))
sys.path.insert(0, str(STANDALONE_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(STANDALONE_ROOT / "configs/fer2013_pixel_neighbor_motif_kaggle_fast_seed42.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--fer-csv", help="Path to train.csv or fer2013.csv")
    parser.add_argument("--output-root", help="Output directory for checkpoints/logs")
    parser.add_argument("--smoke", action="store_true", help="Run quick one-batch verification check")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data with --smoke")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    args = parser.parse_args()

    import tensorflow as tf
    print(f"[RUNNER] TensorFlow {tf.__version__} loaded.")

    if args.smoke or args.synthetic:
        from pixel_neighbor_motif.smoke import run_smoke_test
        run_smoke_test()
        return

    from lap_gnn_tf.config import load_config
    from pixel_neighbor_motif.trainer import run_training

    config = load_config(args.config)
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    fer_csv = args.fer_csv or config.get("paths", {}).get("fer_csv")
    if not fer_csv:
        parser.error("Must supply --fer-csv or configure paths.fer_csv in YAML")

    output_root = args.output_root or config.get("paths", {}).get("output_root", "outputs/run")
    run_training(
        config_path=args.config,
        fer_csv=fer_csv,
        output_root=output_root,
        limit_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
