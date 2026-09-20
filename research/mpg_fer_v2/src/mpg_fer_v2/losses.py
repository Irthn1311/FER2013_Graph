"""Differentiable objectives introduced by MPG-FER v2."""

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

