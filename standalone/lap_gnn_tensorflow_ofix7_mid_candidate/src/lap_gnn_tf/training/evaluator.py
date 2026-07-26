"""Bounded or complete split evaluation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.metrics import classification_metrics


def build_compiled_evaluation_step(model):
    """Compile inference once while preserving the dynamic graph signature."""
    from lap_gnn_tf.data.graph_generator import GraphBatchGenerator

    @tf.function(
        input_signature=[GraphBatchGenerator.output_signature()],
        reduce_retracing=True,
    )
    def evaluate_step(batch):
        output = model(batch, training=False)
        loss = sparse_cross_entropy(batch["labels"], output["logits"])
        return loss, output["probabilities"]

    return evaluate_step


def evaluate_batches(
    model,
    batches,
    limit_batches: int | None = None,
    evaluate_step=None,
) -> dict:
    if evaluate_step is None:
        evaluate_step = build_compiled_evaluation_step(model)
    labels, probabilities, losses = [], [], []
    for batch_index, batch in enumerate(batches):
        if limit_batches is not None and batch_index >= int(limit_batches):
            break
        loss, batch_probabilities = evaluate_step(batch)
        losses.append(float(loss.numpy()))
        labels.append(batch["labels"].numpy())
        probabilities.append(batch_probabilities.numpy())
    if not labels:
        raise ValueError("No evaluation batches were produced")
    result = classification_metrics(np.concatenate(labels), np.concatenate(probabilities))
    result["loss"] = float(np.mean(losses))
    return result
