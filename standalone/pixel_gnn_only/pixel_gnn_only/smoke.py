"""Kaggle-side architecture report and one-batch gradient/update check."""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.optimizer import build_optimizer
from pixel_gnn_only.execution import build_restricted_graph_train_step, validate_gradient_contract
from pixel_gnn_only.model import PixelGNNOnly
from pixel_gnn_only.protocol import EXPECTED_PARAMETER_COUNT


def describe_model(model, batch, output_dir):
    output = model(batch, training=False)
    layers = model._flatten_layers(include_self=False, recursive=True)
    inventory = [{"name": layer.name, "class": type(layer).__name__,
                  "trainable_parameters": sum(int(np.prod(v.shape)) for v in layer.trainable_weights)} for layer in layers]
    forbidden = [row for row in inventory if any(token in row["class"].lower()
                 for token in ["conv", "attention", "partcontext", "partglobal", "motif"])]
    if forbidden:
        raise RuntimeError(f"Forbidden CNN/prior-aware layers: {forbidden}")
    parameters = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter count drift: {parameters} != {EXPECTED_PARAMETER_COUNT}")
    buffer = io.StringIO()
    model.summary(print_fn=lambda line, **kwargs: buffer.write(line + "\n"), expand_nested=True)
    print(buffer.getvalue(), flush=True)
    shapes = {"node_features": list(batch["node_features"].shape),
              "edge_index": list(batch["edge_index"].shape),
              "edge_features": list(batch["edge_features"].shape),
              "gnn_output": list(output["node_embeddings"].shape),
              "pooled_embedding": list(output["z_image"].shape),
              "logits": list(output["logits"].shape)}
    report = {"shapes": shapes, "trainable_parameters": parameters,
              "no_conv_or_cnn_layers": True, "no_landmark_aware_layers": True, "layers": inventory}
    print(json.dumps(report, indent=2), flush=True)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_summary.txt").write_text(buffer.getvalue(), encoding="utf-8")
    (directory / "architecture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_smoke(batch, config, output_dir, input_kind):
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Smoke output must be fresh: {directory}")
    model = PixelGNNOnly()
    report = describe_model(model, batch, directory)
    with tf.GradientTape() as tape:
        output = model(batch, training=True)
        loss = sparse_cross_entropy(batch["labels"], output["logits"])
    gradients = tape.gradient(loss, model.trainable_variables)
    validate_gradient_contract(gradients, model.trainable_variables)
    if not bool(tf.math.is_finite(loss)) or not all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
        raise FloatingPointError("Non-finite loss or gradients")
    optimizer = build_optimizer(config)
    optimizer.build(model.trainable_variables)
    # Also exercise the same compiled, mixed-precision optimizer path as training.
    from pixel_gnn_only.batching import GraphBatchGenerator
    step = build_restricted_graph_train_step(model, optimizer, GraphBatchGenerator.output_signature())
    before = [v.numpy().copy() for v in model.trainable_variables]
    step_loss = step(batch)
    changed = any(not np.array_equal(a, v.numpy()) for a, v in zip(before, model.trainable_variables))
    if not changed or int(optimizer.iterations.numpy()) != 1:
        raise RuntimeError("The compiled optimizer step did not update parameters")
    report.update({"input_kind": input_kind, "batch_size": int(batch["labels"].shape[0]),
                   "forward_backward": "PASS", "compiled_optimizer_update": "PASS",
                   "loss": float(loss.numpy()), "compiled_loss": float(step_loss.numpy()),
                   "gradient_variable_count": len(gradients),
                   "scientific_accuracy": "UNKNOWN; this is only a smoke test"})
    (directory / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report
