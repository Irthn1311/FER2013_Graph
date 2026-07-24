"""Bounded or complete split evaluation."""

from __future__ import annotations

import numpy as np

from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.metrics import classification_metrics


def evaluate_batches(model, batches, limit_batches: int | None = None) -> dict:
    labels, probabilities, losses = [], [], []
    for batch_index, batch in enumerate(batches):
        if limit_batches is not None and batch_index >= int(limit_batches):
            break
        output = model(batch, training=False)
        losses.append(float(sparse_cross_entropy(batch["labels"], output["logits"]).numpy()))
        labels.append(batch["labels"].numpy())
        probabilities.append(output["probabilities"].numpy())
    if not labels:
        raise ValueError("No evaluation batches were produced")
    result = classification_metrics(np.concatenate(labels), np.concatenate(probabilities))
    result["loss"] = float(np.mean(losses))
    return result

