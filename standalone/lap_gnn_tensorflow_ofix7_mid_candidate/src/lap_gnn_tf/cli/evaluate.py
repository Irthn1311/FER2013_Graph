import argparse
import json
from pathlib import Path

import tensorflow as tf

from lap_gnn_tf.config import load_config
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.model import LapGNN
from lap_gnn_tf.training.evaluator import evaluate_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fer-csv", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("Official evaluation requires TRAINING_COMPLETE.json")
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    checkpoint = run_dir / "checkpoints" / f"{args.checkpoint}.keras"
    model = tf.keras.models.load_model(
        checkpoint, custom_objects={"LapGNN": LapGNN}, compile=False,
    )
    data = GraphBatchGenerator(
        args.prior_root, args.split, config,
        batch_size=int(config["training"]["batch_size"]), seed=int(config["seed"]),
        shuffle=False, graph_cache_size=int(config.get("resources", {}).get("graph_cache_size", 64)),
    )
    metrics = evaluate_batches(model, data.iter_epoch(0))
    output = run_dir / f"{args.split}_metrics_{args.checkpoint}.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
