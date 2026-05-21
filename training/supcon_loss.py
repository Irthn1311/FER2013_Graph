"""Supervised contrastive loss utilities for diagnostic image embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from torch import nn


class SupervisedContrastiveLoss(nn.Module):
    """Khosla-style SupCon loss for one normalized view per sample."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss, _ = supervised_contrastive_loss_with_stats(features, labels, temperature=self.temperature)
        return loss


def supervised_contrastive_loss_with_stats(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Image-level supervised contrastive loss plus diagnostic stats.

    Samples without a same-label positive in the batch are excluded from the
    average. If no positive pairs exist, the loss is zero and stats mark the
    missing SupCon signal.
    """
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if features.ndim != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(f"labels must be [B], got {tuple(labels.shape)} for features {tuple(features.shape)}")
    batch_size = int(features.shape[0])
    zero = features.sum() * 0.0
    stats: Dict[str, Any] = {
        "positive_pair_count": 0.0,
        "valid_supcon_anchor_count": 0.0,
        "z_norm_mean": float(features.detach().float().norm(dim=1).mean().cpu()) if batch_size else 0.0,
        "z_norm_std": float(features.detach().float().norm(dim=1).std(unbiased=False).cpu()) if batch_size else 0.0,
        "embedding_collapse_score": 0.0,
        "has_supcon_signal": 0.0,
    }
    if batch_size <= 1:
        return zero, stats

    z = F.normalize(features.float(), dim=1, eps=1e-8)
    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.t()).to(device=z.device, dtype=z.dtype)
    self_mask = torch.eye(batch_size, device=z.device, dtype=z.dtype)
    positive_mask = positive_mask * (1.0 - self_mask)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    positive_pair_count = float(positive_mask.sum().detach().cpu().item())

    sim = torch.matmul(z, z.t())
    off_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
    collapse_score = sim[off_mask].mean() if off_mask.any() else sim.new_tensor(0.0)
    stats.update(
        {
            "positive_pair_count": positive_pair_count,
            "valid_supcon_anchor_count": float(valid.sum().detach().cpu().item()),
            "z_norm_mean": float(features.detach().float().norm(dim=1).mean().cpu().item()),
            "z_norm_std": float(features.detach().float().norm(dim=1).std(unbiased=False).cpu().item()),
            "embedding_collapse_score": float(collapse_score.detach().cpu().item()),
            "has_supcon_signal": 1.0 if positive_pair_count > 0.0 else 0.0,
        }
    )
    if not bool(valid.any()):
        return zero, stats

    logits = sim / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1.0 - self_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid].clamp_min(1.0)
    return -mean_log_prob_pos.mean(), stats


class SupervisedContrastiveLossWithStats(nn.Module):
    """SupCon module returning ``(loss, stats)`` for D13C diagnostics."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return supervised_contrastive_loss_with_stats(features, labels, temperature=self.temperature)


class _LegacySupervisedContrastiveLoss(nn.Module):
    """Deprecated compatibility stub kept out of public imports."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)} for features {tuple(features.shape)}")
        batch_size = int(features.shape[0])
        if batch_size <= 1:
            return features.sum() * 0.0

        z = F.normalize(features.float(), dim=1, eps=1e-8)
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.t()).to(device=z.device, dtype=z.dtype)
        self_mask = torch.eye(batch_size, device=z.device, dtype=z.dtype)
        positive_mask = positive_mask * (1.0 - self_mask)

        logits = torch.matmul(z, z.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = 1.0 - self_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not bool(valid.any()):
            return features.sum() * 0.0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid].clamp_min(1.0)
        return -mean_log_prob_pos.mean()
