"""Evaluate one TensorFlow checkpoint without modifying its run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from lap_gnn_tf.config import load_config
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.model import LapGNN
from lap_gnn_tf.training.evaluator import evaluate_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Checkpoint stem. Repeat to evaluate a probability ensemble.",
    )
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--clean-graph-cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--graph-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_names = args.checkpoint or ["best_val_accuracy"]
    checkpoints = [
        run_dir / "checkpoints" / f"{checkpoint_name}.keras"
        for checkpoint_name in checkpoint_names
    ]
    config = load_config(run_dir / "resolved_config.yaml")
    models = [
        tf.keras.models.load_model(
            checkpoint,
            custom_objects={"LapGNN": LapGNN},
            compile=False,
        )
        for checkpoint in checkpoints
    ]
    if len(models) == 1:
        model = models[0]
    else:
        class ProbabilityEnsemble(tf.Module):
            def __init__(self, members):
                super().__init__()
                self.members = members

            def __call__(self, batch, training=False):
                probabilities = tf.add_n(
                    [
                        member(batch, training=training)["probabilities"]
                        for member in self.members
                    ]
                ) / tf.cast(len(self.members), tf.float32)
                logits = tf.math.log(tf.maximum(probabilities, 1e-7))
                return {"logits": logits, "probabilities": probabilities}

        model = ProbabilityEnsemble(models)
    data = GraphBatchGenerator(
        args.prior_root.resolve(),
        args.split,
        config,
        batch_size=args.batch_size,
        seed=int(config["seed"]),
        shuffle=False,
        graph_cache_size=64,
        graph_workers=args.graph_workers,
        clean_graph_cache_dir=args.clean_graph_cache_dir.resolve(),
    )
    metrics = evaluate_batches(model, data.iter_epoch(0))
    result = {
        "run_dir": str(run_dir),
        "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
        "split": args.split,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
