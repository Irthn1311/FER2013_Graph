"""Repeated eager, tf.function and fresh-process parity validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model
from lap_gnn_tf.training.losses import sparse_cross_entropy


def setup(package_root: Path):
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    golden = package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    with tf.device("/CPU:0"):
        model = build_model(batch)
        load_pytorch_npz(model, golden / "model_state.npz", strict=True)
    return model, batch, golden


def measure(output, batch, golden: Path) -> dict:
    expected_logits = np.load(golden / "logits.npy", allow_pickle=False)
    expected_probabilities = np.load(golden / "probabilities.npy", allow_pickle=False)
    expected_loss = json.loads((golden / "losses.json").read_text(encoding="utf-8"))["cross_entropy"]
    logits = output["logits"].numpy()
    probabilities = output["probabilities"].numpy()
    loss = float(sparse_cross_entropy(batch["labels"], output["logits"]).numpy())
    return {
        "max_logit_difference": float(np.max(np.abs(logits - expected_logits))),
        "max_probability_difference": float(
            np.max(np.abs(probabilities - expected_probabilities))
        ),
        "loss_difference": abs(loss - float(expected_loss)),
        "prediction_agreement": float(
            np.mean(logits.argmax(axis=1) == expected_logits.argmax(axis=1))
        ),
    }


def summarize(rows: list[dict]) -> dict:
    values = [row["max_logit_difference"] for row in rows]
    return {
        "runs": len(rows),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "maximum_probability_difference": max(
            row["max_probability_difference"] for row in rows
        ),
        "maximum_loss_difference": max(row["loss_difference"] for row in rows),
        "minimum_prediction_agreement": min(row["prediction_agreement"] for row in rows),
        "all_pass": all(
            row["max_logit_difference"] <= 1e-5
            and row["prediction_agreement"] == 1.0
            for row in rows
        ),
    }


def worker(package_root: Path, output_path: Path) -> None:
    model, batch, golden = setup(package_root)
    result = measure(model(batch, training=False), batch, golden)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if args.worker_output:
        worker(package_root, args.worker_output)
        return
    if args.output_json is None or args.output_csv is None:
        parser.error("--output-json and --output-csv are required")

    model, batch, golden = setup(package_root)
    eager_rows = [
        {"mode": "eager", "run": run + 1, **measure(model(batch, training=False), batch, golden)}
        for run in range(10)
    ]

    @tf.function(autograph=False)
    def compiled(value):
        return model(value, training=False)

    compiled(batch)
    function_rows = [
        {"mode": "tf_function", "run": run + 1, **measure(compiled(batch), batch, golden)}
        for run in range(10)
    ]
    fresh_rows = []
    with tempfile.TemporaryDirectory(prefix="lap_gnn_tf_repair_") as temporary:
        for run in range(5):
            output_path = Path(temporary) / f"fresh_{run}.json"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            environment["TF_DETERMINISTIC_OPS"] = "1"
            environment.pop("TF_ENABLE_ONEDNN_OPTS", None)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--package-root",
                    str(package_root),
                    "--worker-output",
                    str(output_path),
                ],
                check=True,
                cwd=package_root,
                env=environment,
                capture_output=True,
                text=True,
            )
            fresh_rows.append({
                "mode": "fresh_process",
                "run": run + 1,
                **json.loads(output_path.read_text(encoding="utf-8")),
            })
    all_rows = eager_rows + function_rows + fresh_rows
    result = {
        "eager": summarize(eager_rows),
        "tf_function": summarize(function_rows),
        "fresh_process": summarize(fresh_rows),
        "parity_class": (
            "EXACT_FORWARD_PARITY"
            if max(row["max_logit_difference"] for row in all_rows) == 0.0
            else "NUMERIC_FORWARD_PARITY_STABLE"
            if all(row["max_logit_difference"] <= 1e-5 for row in all_rows)
            else "FORWARD_PARITY_STILL_FAILED"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(result, indent=2))
    if result["parity_class"] == "FORWARD_PARITY_STILL_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
