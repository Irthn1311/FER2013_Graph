"""Kaggle-side architecture report and one-batch gradient/update check."""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from pixel_gnn_only.execution import validate_gradient_contract
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


def run_smoke(batch, config, output_dir, input_kind, gpu_count=1):
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Smoke output must be fresh: {directory}")
    with tf.device("/GPU:0"):
        model = PixelGNNOnly()
    report = describe_model(model, batch, directory)
    from pixel_gnn_only.parallel import (create_replica, build_parallel_train_step,
                                         replica_gradients, shard_graph_batch, variable_devices)
    replica = create_replica(model, batch, gpu_count)
    count = int(batch["labels"].shape[0])
    if replica is not None and count > 1:
        midpoint = (count + 1) // 2
        left = shard_graph_batch(batch, 0, midpoint)
        right = shard_graph_batch(batch, midpoint, count)
        with tf.device("/GPU:0"):
            loss_left, gradients_left = replica_gradients(model, left, None, count)
        with tf.device("/GPU:1"):
            loss_right, gradients_right = replica_gradients(replica, right, None, count)
        with tf.device("/GPU:0"):
            loss = loss_left + loss_right
            gradients = [a + b for a, b in zip(gradients_left, gradients_right)]
    else:
        with tf.device("/GPU:0"):
            loss, gradients = replica_gradients(model, batch, None, count)
    validate_gradient_contract(gradients, model.trainable_variables)
    if not bool(tf.math.is_finite(loss)) or not all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
        raise FloatingPointError("Non-finite loss or gradients")
    from lap_gnn_tf.training.optimizer import build_optimizer
    with tf.device("/GPU:0"):
        optimizer = build_optimizer(config)
        optimizer.build(model.trainable_variables)
    # Also exercise the same compiled, mixed-precision optimizer path as training.
    step = build_parallel_train_step(model, replica, optimizer)
    before = [v.numpy().copy() for v in model.trainable_variables]
    step_loss = step(batch)
    changed = any(not np.array_equal(a, v.numpy()) for a, v in zip(before, model.trainable_variables))
    if not changed or int(optimizer.iterations.numpy()) != 1:
        raise RuntimeError("The compiled optimizer step did not update parameters")
    if replica is not None and not all(np.array_equal(a.numpy(), b.numpy()) for a, b in
                                      zip(model.trainable_variables, replica.trainable_variables)):
        raise RuntimeError("GPU replica weights were not synchronized")
    report.update({"input_kind": input_kind, "batch_size": int(batch["labels"].shape[0]),
                   "forward_backward": "PASS", "compiled_optimizer_update": "PASS",
                   "loss": float(loss.numpy()), "compiled_loss": float(step_loss.numpy()),
                   "gradient_variable_count": len(gradients),
                   "gpu_count": gpu_count, "replica_weights_synced": replica is not None,
                   "primary_variable_devices": variable_devices(model),
                   "replica_variable_devices": variable_devices(replica) if replica is not None else [],
                   "scientific_accuracy": "UNKNOWN; this is only a smoke test"})
    (directory / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report
