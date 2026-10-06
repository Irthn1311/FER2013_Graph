"""Differentiable objectives introduced by MPG-FER v2.2."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def motif_mutual_information_loss(
    assignments: torch.Tensor, beta: float = 1.0, eps: float = 1e-8
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return H(M|H) - beta*H(M), normalized by log(number of motifs)."""
    if assignments.ndim != 3 or assignments.shape[-1] < 2:
        raise ValueError("assignments must have shape [batch, nodes, motifs>=2]")
    probabilities = assignments.clamp_min(eps)
    local_entropy_per_node = -(probabilities * probabilities.log()).sum(dim=-1)
    h_local_raw = local_entropy_per_node.mean()
    usage = assignments.mean(dim=(0, 1)).clamp_min(eps)
    h_global_raw = -(usage * usage.log()).sum()
    normalizer = math.log(assignments.shape[-1])
    h_local_normalized = h_local_raw / normalizer
    h_global_normalized = h_global_raw / normalizer
    loss = h_local_normalized - beta * h_global_normalized
    return loss, {
        "H_local_raw": h_local_raw,
        "H_local_normalized": h_local_normalized,
        "H_global_raw": h_global_raw,
        "H_global_normalized": h_global_normalized,
        "L_MI": loss,
        "prototype_utilization": usage,
    }


def symmetric_js_divergence(
    logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean Jensen-Shannon divergence between two classifier distributions."""
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    log_m = m.clamp_min(eps).log()
    kl_pm = (p * (F.log_softmax(logits_a, dim=-1) - log_m)).sum(dim=-1)
    kl_qm = (q * (F.log_softmax(logits_b, dim=-1) - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard in-batch SupCon with safe exclusion of invalid anchors."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, dimension]")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    normalized = F.normalize(embeddings, dim=-1)
    batch = embeddings.shape[0]
    zero = embeddings.sum() * 0.0
    if batch < 2:
        return zero, {
            "valid_supcon_anchor_fraction": zero.detach(),
            "mean_positive_count": zero.detach(),
        }

    self_mask = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0

    similarity = normalized @ normalized.t() / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    denominator_logits = similarity.masked_fill(self_mask, -torch.inf)
    log_prob = similarity - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    per_anchor = -(
        log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    loss = per_anchor[valid].mean() if bool(valid.any()) else zero
    valid_fraction = valid.to(embeddings.dtype).mean()
    mean_positive_count = (
        positive_count[valid].to(embeddings.dtype).mean()
        if bool(valid.any())
        else zero.detach()
    )
    return loss, {
        "valid_supcon_anchor_fraction": valid_fraction,
        "mean_positive_count": mean_positive_count,
    }
