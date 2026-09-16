"""Evaluation and motif diagnostics for Pixel GNN models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from pixel_gnn.utils import classification_metrics
from pixel_gnn.losses import compute_total_loss, compute_motif_diagnostics


def extract_motif_diagnostics(
    model,
    batch: dict,
    top_k_nodes: int = 10,
) -> dict:
    output = model(batch, training=False)
    if output.get("motif_assignment") is None:
        return {"motif_enabled": False}

    prototypes = output["motif_prototypes"].numpy()
    assignment = output["motif_assignment"]
    assignment_np = assignment.numpy()
    beta = output["motif_attention_weights"].numpy() if output.get("motif_attention_weights") is not None else None
    usage = output["motif_usage"].numpy() if output.get("motif_usage") is not None else np.mean(assignment_np, axis=(0, 1))

    # Extended motif metrics
    diag_metrics = compute_motif_diagnostics(assignment)

    node_motif_mean = np.mean(assignment_np, axis=0)

    top_nodes = {}
    num_motifs = prototypes.shape[0]
    for k in range(num_motifs):
        scores_k = node_motif_mean[:, k]
        top_indices = np.argsort(scores_k)[::-1][:top_k_nodes]
        coords = [(int(idx // 48), int(idx % 48), float(scores_k[idx])) for idx in top_indices]
        top_nodes[f"motif_{k:02d}"] = {
            "usage": float(usage[k]),
            "top_pixels": coords,
        }

    A_motif = output.get("A_motif")
    A_motif_np = A_motif.numpy() if A_motif is not None else None

    spatial_centers = output.get("motif_spatial_centers")
    spatial_centers_np = spatial_centers.numpy() if spatial_centers is not None else None

    motif_gnn_attn = output.get("motif_gnn_attention")
    motif_gnn_attn_np = motif_gnn_attn.numpy() if motif_gnn_attn is not None else None

    result = {
        "motif_enabled": True,
        "prototypes": prototypes,
        "usage": usage,
        "attention_weights": beta,
        "A_motif": A_motif_np,
        "motif_gnn_attention": motif_gnn_attn_np,
        "spatial_centers": spatial_centers_np,
        "top_nodes_per_motif": top_nodes,
    }
    result.update(diag_metrics)
    return result


def evaluate_model(
    model,
    dataset_generator,
    lambda_diversity: float = 0.0,
    lambda_spatial_coherence: float = 0.0,
    limit_batches: int | None = None,
    include_diagnostics: bool = False,
) -> dict:
    all_labels, all_probs, all_losses = [], [], []

    @tf.function
    def eval_step(batch):
        out = model(batch, training=False)
        loss, _ = compute_total_loss(
            batch["labels"],
            out,
            lambda_diversity=lambda_diversity,
            lambda_spatial_coherence=lambda_spatial_coherence,
        )
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
