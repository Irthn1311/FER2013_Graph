"""CPU float32 golden parity command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from lap_gnn_tf.constants import EXPECTED_PARAMETER_COUNT
from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model
from lap_gnn_tf.training.losses import sparse_cross_entropy


def compare(package_root: Path) -> dict:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    golden = package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    with tf.device("/CPU:0"):
        model = build_model(batch)
        mapping = load_pytorch_npz(model, golden / "model_state.npz", strict=True)
        output = model(batch, training=False, collect_intermediates=True)
    with np.load(golden / "layer_outputs.npz", allow_pickle=False) as expected_layers:
        rows = []
        for name in expected_layers.files:
            if name not in output["intermediates"]:
                continue
            actual = output["intermediates"][name].numpy()
            expected = expected_layers[name]
            delta = actual - expected
            rows.append({
                "name": name,
                "max_abs": float(np.max(np.abs(delta))),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12)),
            })
    expected_logits = np.load(golden / "logits.npy", allow_pickle=False)
    expected_probabilities = np.load(golden / "probabilities.npy", allow_pickle=False)
    actual_logits = output["logits"].numpy()
    actual_probabilities = output["probabilities"].numpy()
    max_logit = float(np.max(np.abs(actual_logits - expected_logits)))
    max_probability = float(np.max(np.abs(actual_probabilities - expected_probabilities)))
    prediction_agreement = float(np.mean(actual_logits.argmax(1) == expected_logits.argmax(1)))
    expected_loss = json.loads((golden / "losses.json").read_text(encoding="utf-8"))["cross_entropy"]
    actual_loss = float(sparse_cross_entropy(batch["labels"], output["logits"]).numpy())
    trainable = sum(int(variable.shape.num_elements()) for variable in model.trainable_variables)
    non_trainable = sum(int(variable.shape.num_elements()) for variable in model.non_trainable_variables)
    result = {
        "tensorflow": tf.__version__,
        "mapping": mapping,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "parameter_count_match": trainable == EXPECTED_PARAMETER_COUNT,
        "layers": rows,
        "maximum_layer_max_abs": max(row["max_abs"] for row in rows),
        "max_logit_difference": max_logit,
        "max_probability_difference": max_probability,
        "prediction_agreement": prediction_agreement,
        "loss_difference": abs(actual_loss - float(expected_loss)),
        "forward_parity_class": (
            "EXACT_FORWARD_PARITY" if max_logit <= 1e-6
            else "NUMERIC_FORWARD_PARITY" if max_logit <= 1e-5
            else "FORWARD_PARITY_FAILED"
        ),
        "pass": (
            mapping["complete"]
            and trainable == EXPECTED_PARAMETER_COUNT
            and max_logit <= 1e-5
            and prediction_agreement == 1.0
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.package_root)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
