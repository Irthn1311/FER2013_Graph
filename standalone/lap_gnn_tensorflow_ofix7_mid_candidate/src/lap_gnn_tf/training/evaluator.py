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
    return_details: bool = False,
) -> dict:
    if evaluate_step is None:
        evaluate_step = build_compiled_evaluation_step(model)
    labels, probabilities, losses = [], [], []
    sample_ids, detected, missing = [], [], []
    for batch_index, batch in enumerate(batches):
        if limit_batches is not None and batch_index >= int(limit_batches):
            break
        loss, batch_probabilities = evaluate_step(batch)
        losses.append(float(loss.numpy()))
        labels.append(batch["labels"].numpy())
        probabilities.append(batch_probabilities.numpy())
        if return_details:
            sample_ids.append(batch["sample_ids"].numpy())
            detected.append(batch["detected"].numpy())
            missing.append(batch["landmark_missing_flag"].numpy())
    if not labels:
        raise ValueError("No evaluation batches were produced")
    labels_array = np.concatenate(labels)
    probabilities_array = np.concatenate(probabilities)
    result = classification_metrics(labels_array, probabilities_array)
    result["loss"] = float(np.mean(losses))
    if return_details:
        result["details"] = {
            "labels": labels_array,
            "predictions": probabilities_array.argmax(axis=1),
            "probabilities": probabilities_array,
            "sample_ids": np.concatenate(sample_ids),
            "detected": np.concatenate(detected),
            "landmark_missing_flag": np.concatenate(missing),
        }
    return result
