"""Evaluation and Motif Diagnostics for Pixel Neighbor Motif."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.metrics import classification_metrics
from pixel_neighbor_motif.losses import compute_total_loss


def extract_motif_diagnostics(
    model,
    batch: dict,
    top_k_nodes: int = 10,
) -> dict:
    """Extract diagnostic information about learned motifs for visualization.

    Args:
        model: PixelNeighborMotifModel
        batch: sample batch dict
        top_k_nodes: number of top pixel nodes to retrieve per motif
    Returns:
        dict containing:
            - motif_prototypes: [K, D] np.ndarray
            - motif_usage: [K] np.ndarray
            - motif_attention_weights: [B, K] np.ndarray
            - top_nodes_per_motif: list of dicts with top pixel (row, col) coordinates
    """
    output = model(batch, training=False)
    if output.get("motif_assignment") is None:
        return {"motif_enabled": False}

    prototypes = output["motif_prototypes"].numpy()               # [K, D]
    assignment = output["motif_assignment"].numpy()               # [B, 2304, K]
    beta = output["motif_attention_weights"].numpy()               # [B, K]
    usage = output["motif_usage"].numpy()                         # [K]

    # Mean assignment across the batch for each of the 2304 nodes: [2304, K]
    node_motif_mean = np.mean(assignment, axis=0)

    top_nodes = {}
    num_motifs = prototypes.shape[0]
    for k in range(num_motifs):
        # Top scoring nodes for motif k
        scores_k = node_motif_mean[:, k]
        top_indices = np.argsort(scores_k)[::-1][:top_k_nodes]
        # Convert flat index to (row, col)
        coords = [(int(idx // 48), int(idx % 48), float(scores_k[idx])) for idx in top_indices]
        top_nodes[f"motif_{k:02d}"] = {
            "usage": float(usage[k]),
            "top_pixels": coords,
        }

    return {
        "motif_enabled": True,
        "prototypes": prototypes,
        "usage": usage,
        "attention_weights": beta,
        "top_nodes_per_motif": top_nodes,
    }


def evaluate_model(
    model,
    dataset_generator,
    lambda_diversity: float = 0.0,
    limit_batches: int | None = None,
    include_diagnostics: bool = False,
) -> dict:
    """Run full evaluation on a dataset split."""
    all_labels, all_probs, all_losses = [], [], []

    @tf.function
    def eval_step(batch):
        out = model(batch, training=False)
        loss, _ = compute_total_loss(batch["labels"], out, lambda_diversity=lambda_diversity)
        return loss, out["probabilities"], out

    first_batch_diagnostics = None

    for batch_idx, batch in enumerate(dataset_generator):
        if limit_batches is not None and batch_idx >= int(limit_batches):
            break
        loss, probs, out = eval_step(batch)
        all_losses.append(float(loss.numpy()))
        all_labels.append(batch["labels"].numpy())
        all_probs.append(probs.numpy())

        if include_diagnostics and first_batch_diagnostics is None:
            first_batch_diagnostics = extract_motif_diagnostics(model, batch)

    labels = np.concatenate(all_labels, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)
    metrics = classification_metrics(labels, probabilities)
    metrics["loss"] = float(np.mean(all_losses))

    if include_diagnostics:
        metrics["motif_diagnostics"] = first_batch_diagnostics

    return metrics
