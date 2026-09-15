#!/usr/bin/env python3
"""Separate Kaggle runner for Pixel-GNN Only. The full runner is unchanged."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = REPO_ROOT / "standalone/pixel_gnn_only"
sys.path.insert(0, str(REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"))
sys.path.insert(0, str(ABLATION_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ABLATION_ROOT / "configs/fer2013_pixel_gnn_only_seed42.yaml"))
    parser.add_argument("--fer-csv", help="train.csv alongside val.csv/test.csv, or original FER2013 Usage CSV")
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true", help="One training batch forward/backward; no val/test evaluation")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly synthetic input, allowed only with --smoke")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved protocol without loading data or training")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.10–3.12, matching the frozen TensorFlow package")
    import tensorflow as tf
    tf_version = tuple(int(x) for x in tf.__version__.split(".")[:2])
    if not (2, 13) <= tf_version < (2, 19):
        raise RuntimeError("Use TensorFlow 2.13–2.18; install requirements-kaggle.txt and restart the Kaggle session")
    from lap_gnn_tf.config import load_config
    from pixel_gnn_only.protocol import validate_pixel_config
    config = load_config(args.config)
    validate_pixel_config(config)
    if args.synthetic and not args.smoke:
        parser.error("--synthetic is allowed only with --smoke")
    output = args.output_root or ("/kaggle/working/outputs/pixel_gnn_only_smoke" if args.smoke
                                  else config["paths"]["output_root"])
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not args.synthetic and not args.fer_csv:
        matches = sorted(Path("/kaggle/input").glob("**/train.csv")) + sorted(Path("/kaggle/input").glob("**/fer2013.csv"))
        if len(matches) != 1:
            parser.error("Pass --fer-csv explicitly; could not identify exactly one FER2013 CSV")
        args.fer_csv = str(matches[0])
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Select a GPU accelerator on Kaggle before running this script")
    from lap_gnn_tf.resources import ResourceControls
    from lap_gnn_tf.seed import seed_everything
    from pixel_gnn_only.execution import validate_execution_config
    resources = config["resources"]
    controls = ResourceControls(batch_size=config["training"]["batch_size"], eval_batch_size=32,
        graph_workers=resources["graph_workers"], tf_data_prefetch=resources["tf_data_prefetch"],
        graph_cache_size=resources["graph_cache_size"], memory_growth=resources["memory_growth"],
        mixed_precision=resources["mixed_precision"], xla=resources["xla"], device="gpu")
    if args.smoke:
        controls.apply()
        seed_everything(config["seed"])
        validate_execution_config(config["training"])
        if args.synthetic:
            import numpy as np
            from pixel_gnn_only.graph import build_pixel_graph, collate_pixel_graphs
            rng = np.random.default_rng(config["seed"])
            batch = collate_pixel_graphs([build_pixel_graph(rng.random((48, 48), dtype=np.float32), i % 7, i)
                                         for i in range(controls.batch_size)])
            input_kind = "synthetic; no FER data or prior artifacts read"
        else:
            from pixel_gnn_only.batching import GraphBatchGenerator
            generator = GraphBatchGenerator(args.fer_csv, "train", config, controls.batch_size,
                                            config["seed"], True, graph_workers=controls.graph_workers)
            batch = next(generator.iter_epoch(1, limit_batches=1))
            input_kind = generator.dataset.provenance
        from pixel_gnn_only.smoke import run_smoke
        run_smoke(batch, config, output, input_kind)
    else:
        from pixel_gnn_only.trainer import run_training
        run_training(config_path=args.config, fer_csv=args.fer_csv, output_root=output,
                     controls=controls, no_resume=True)


if __name__ == "__main__":
    main()
