#!/usr/bin/env python3
"""Universal Runner for FER2013 Models (Pixel GNN / Neighbor Motif / LAP-GNN).

Usage Examples:
    # 1. Quick smoke test on any model:
    python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_fast_seed42.yaml --smoke --synthetic
    python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_gnn_only_fast_seed42.yaml --smoke --synthetic

    # 2. Train on Kaggle (GPU):
    python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_fast_seed42.yaml --fer-csv /kaggle/input/dataset/train.csv
    python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_gnn_only_fast_seed42.yaml --fer-csv /kaggle/input/dataset/train.csv

    # 3. Train on Local Machine (CPU/GPU):
    python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_seed42.yaml --fer-csv "data/train.csv" --epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STANDALONE_PIXEL = REPO_ROOT / "standalone/pixel_gnn"
STANDALONE_TF = REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"

sys.path.insert(0, str(STANDALONE_TF))
sys.path.insert(0, str(STANDALONE_PIXEL))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=str(STANDALONE_PIXEL / "configs/fer2013_pixel_neighbor_motif_fast_seed42.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--fer-csv", help="Path to train.csv or fer2013.csv")
    parser.add_argument("--output-root", help="Output directory for checkpoints and logs")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-batch verification check")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data with --smoke")
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--epochs", type=int, help="Override maximum epochs")
    args = parser.parse_args()

    import tensorflow as tf
    print(f"[RUNNER] TensorFlow {tf.__version__} loaded. Config: {args.config}")

    from lap_gnn_tf.config import load_config
    config = load_config(args.config)

    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    model_name = config.get("model", {}).get("name", "pixel_neighbor_motif")

    # Mode 1: Smoke Test
    if args.smoke or args.synthetic:
        from pixel_gnn.smoke import run_smoke_test
        run_smoke_test(config=config)
        return

    # Mode 2: Training
    fer_csv = args.fer_csv or config.get("paths", {}).get("fer_csv")
    if not fer_csv:
        # Check standard Kaggle / local paths
        candidates = [
            Path("/kaggle/input/fer2013-split/train.csv"),
            Path("/kaggle/input/fer2013/train.csv"),
            Path("/kaggle/input/fer2013/fer2013.csv"),
            Path("data/train.csv"),
            Path("data/fer2013.csv"),
        ]
        for c in candidates:
            if c.is_file():
                fer_csv = str(c)
                print(f"[AUTO-DETECT] Found dataset at: {fer_csv}")
                break

    if not fer_csv:
        parser.error("Must supply --fer-csv <path_to_train.csv> or configure paths.fer_csv in YAML")

    output_root = args.output_root or config.get("paths", {}).get("output_root", f"outputs/{config.get('run_name', 'run')}")

    # Route to appropriate engine based on model type
    if model_name in ["pixel_neighbor_motif", "pixel_gnn_only"]:
        from pixel_gnn.trainer import run_training
        run_training(
            config_path=args.config,
            fer_csv=fer_csv,
            output_root=output_root,
            limit_epochs=args.epochs,
        )
    else:
        # Fallback to full LAP-GNN pipeline
        from lap_gnn_tf.training.trainer import run_training
        from lap_gnn_tf.resources import ResourceControls
        res_cfg = config.get("resources", {})
        data_cfg = config.get("data", {})
        controls = ResourceControls(
            batch_size=int(res_cfg.get("batch_size") or data_cfg.get("batch_size", 32)),
            eval_batch_size=int(res_cfg.get("eval_batch_size", 64)),
        )
        run_training(
            config_path=str(args.config),
            fer_csv=str(fer_csv),
            prior_root=str(config.get("paths", {}).get("prior_root")),
            output_root=str(output_root),
            controls=controls,
            no_resume=True,
        )


if __name__ == "__main__":
    main()
