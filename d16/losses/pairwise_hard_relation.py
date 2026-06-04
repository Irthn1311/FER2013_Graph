"""Pairwise hard-relation auxiliary heads for D16.

The heads are training-only modules attached to the main model for checkpoint
and resume. They do not change the inference classifier path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PAIRS = {
    "fear_sad": {"class_ids": [2, 4], "hidden_ratio": 0.25, "dropout": 0.2},
    "sad_neutral": {"class_ids": [4, 6], "hidden_ratio": 0.25, "dropout": 0.2},
}


class PairwiseHardRelationLoss(nn.Module):
    """Small binary heads over z_image for selected hard-confusion pairs."""

    def __init__(self, embedding_dim: int, pairs: Dict[str, Dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        pair_cfg = pairs or DEFAULT_PAIRS
        if not pair_cfg:
            raise ValueError("pairwise hard relation pairs must not be empty")
        self.pair_names: List[str] = []
        self.class_ids: Dict[str, List[int]] = {}
        self.heads = nn.ModuleDict()
        for name, cfg in pair_cfg.items():
            class_ids = [int(item) for item in (cfg.get("class_ids") or [])]
            if len(class_ids) != 2 or class_ids[0] == class_ids[1]:
                raise ValueError(f"Pair {name!r} must define two distinct class_ids, got {class_ids}")
            hidden_ratio = float(cfg.get("hidden_ratio", 0.25) or 0.25)
            hidden_dim = max(1, int(round(self.embedding_dim * hidden_ratio)))
            dropout = float(cfg.get("dropout", 0.2) or 0.0)
            pair_name = str(name)
            self.pair_names.append(pair_name)
            self.class_ids[pair_name] = class_ids
            self.heads[pair_name] = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, D], got {tuple(z.shape)}")
        zero = z.float().sum() * 0.0
        losses: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {
            "loss_pairwise_hard_relation": zero,
            "pairwise_available_count": z.new_tensor(0.0),
        }
        for name in self.pair_names:
            class_a, class_b = self.class_ids[name]
            mask = y.eq(class_a) | y.eq(class_b)
            count = int(mask.sum().detach().cpu().item())
            key = name.replace("-", "_")
            if count <= 0:
                stats[f"loss_pairwise_{key}"] = zero
                stats[f"pair_count_{key}"] = z.new_tensor(0.0)
                stats[f"pair_acc_{key}"] = z.new_tensor(float("nan"))
                continue
            z_pair = z[mask].float()
            y_pair = torch.where(y[mask].eq(class_a), torch.zeros_like(y[mask]), torch.ones_like(y[mask]))
            logits = self.heads[name](z_pair)
            loss = F.cross_entropy(logits, y_pair)
            acc = logits.argmax(dim=1).eq(y_pair).float().mean()
            losses.append(loss)
            stats[f"loss_pairwise_{key}"] = loss
            stats[f"pair_count_{key}"] = z.new_tensor(float(count))
            stats[f"pair_acc_{key}"] = acc.detach()
        if losses:
            stats["loss_pairwise_hard_relation"] = torch.stack(losses).mean()
            stats["pairwise_available_count"] = z.new_tensor(float(len(losses)))
        return stats


def pairwise_hard_relation_lambda(loss_cfg: Dict[str, Any], epoch: int) -> float:
    cfg = (loss_cfg.get("pairwise_hard_relation", {}) or {}) if loss_cfg else {}
    if not bool(cfg.get("enabled", False)):
        return 0.0
    target = float(cfg.get("lambda", cfg.get("weight", 0.0)) or 0.0)
    start = int(cfg.get("warmup_start_epoch", 0) or 0)
    end = int(cfg.get("warmup_end_epoch", start) or start)
    epoch = int(epoch)
    if target <= 0.0 or epoch < start:
        return 0.0
    if end <= start or epoch >= end:
        return target
    return target * float(epoch - start) / float(max(end - start, 1))


def build_pairwise_hard_relation_loss(loss_cfg: Dict[str, Any], embedding_dim: int) -> PairwiseHardRelationLoss | None:
    if str((loss_cfg or {}).get("mode", "ce_only")) != "ce_pairwise_hard_relation":
        return None
    cfg = (loss_cfg.get("pairwise_hard_relation", {}) or {}) if loss_cfg else {}
    if not bool(cfg.get("enabled", False)):
        return None
    return PairwiseHardRelationLoss(
        embedding_dim=int(embedding_dim),
        pairs=cfg.get("pairs") or DEFAULT_PAIRS,
    )
