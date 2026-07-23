"""Hard-class prototype separation auxiliary loss for D16.

The prototypes are training-only parameters. They are not used during
inference and do not change the classifier path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardPrototypeSeparationLoss(nn.Module):
    """Learnable prototype CE plus margin loss for selected hard classes."""

    def __init__(
        self,
        embedding_dim: int,
        hard_class_ids: Iterable[int] = (0, 2, 4, 6),
        temperature: float = 0.2,
        margin: float = 0.2,
        margin_weight: float = 0.5,
        normalize_embeddings: bool = True,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.hard_class_ids: List[int] = [int(item) for item in hard_class_ids]
        if not self.hard_class_ids:
            raise ValueError("hard_class_ids must not be empty")
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.margin_weight = float(margin_weight)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.eps = float(eps)
        self.prototypes = nn.Parameter(torch.empty(len(self.hard_class_ids), int(embedding_dim)))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, dim=1, eps=self.eps))
        class_ids = torch.as_tensor(self.hard_class_ids, dtype=torch.long)
        self.register_buffer("hard_class_id_tensor", class_ids, persistent=False)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, D], got {tuple(z.shape)}")
        hard_map = torch.full((int(y.max().detach().item()) + 1 if y.numel() else 1,), -1, device=y.device, dtype=torch.long)
        if hard_map.numel() <= int(self.hard_class_id_tensor.max().detach().item()):
            hard_map = torch.full((int(self.hard_class_id_tensor.max().detach().item()) + 1,), -1, device=y.device, dtype=torch.long)
        hard_map[self.hard_class_id_tensor.to(y.device)] = torch.arange(
            len(self.hard_class_ids), device=y.device, dtype=torch.long
        )
        valid_label = y < hard_map.numel()
        hard_target = torch.full_like(y, -1)
        hard_target[valid_label] = hard_map[y[valid_label]]
        hard_mask = hard_target.ge(0)
        zero = z.float().sum() * 0.0 + self.prototypes.float().sum() * 0.0
        if int(hard_mask.sum().detach().cpu().item()) <= 0:
            return {
                "loss_hard_proto_sep": zero,
                "loss_proto_ce": zero,
                "loss_proto_margin": zero,
                "hard_proto_sample_count": z.new_tensor(0.0),
                "hard_proto_positive_sim_mean": z.new_tensor(float("nan")),
                "hard_proto_max_negative_sim_mean": z.new_tensor(float("nan")),
            }

        z_hard = z[hard_mask].float()
        targets = hard_target[hard_mask]
        proto = self.prototypes.float()
        if self.normalize_embeddings:
            z_hard = F.normalize(z_hard, dim=1, eps=self.eps)
            proto = F.normalize(proto, dim=1, eps=self.eps)
        sim = z_hard @ proto.t()
        proto_ce = F.cross_entropy(sim / max(self.temperature, self.eps), targets)
        positive = sim.gather(1, targets.view(-1, 1)).squeeze(1)
        neg_sim = sim.masked_fill(F.one_hot(targets, num_classes=sim.size(1)).bool(), -torch.inf)
        max_negative = neg_sim.max(dim=1).values
        margin_loss = F.relu(float(self.margin) - positive + max_negative).mean()
        loss = proto_ce + float(self.margin_weight) * margin_loss
        return {
            "loss_hard_proto_sep": loss,
            "loss_proto_ce": proto_ce,
            "loss_proto_margin": margin_loss,
            "hard_proto_sample_count": hard_mask.sum().to(dtype=z.dtype),
            "hard_proto_positive_sim_mean": positive.detach().mean(),
            "hard_proto_max_negative_sim_mean": max_negative.detach().mean(),
        }


def hard_proto_lambda(loss_cfg: Dict[str, Any], epoch: int) -> float:
    cfg = (loss_cfg.get("hard_proto_sep", {}) or {}) if loss_cfg else {}
    if not bool(cfg.get("enabled", False)):
        return 0.0
    target = float(cfg.get("lambda", cfg.get("weight", 0.0)) or 0.0)
    start = int(cfg.get("warmup_start_epoch", 0) or 0)
    end = int(cfg.get("warmup_end_epoch", start) or start)
    epoch = int(epoch)
    if target <= 0.0:
        return 0.0
    if epoch < start:
        return 0.0
    if end <= start or epoch >= end:
        return target
    return target * float(epoch - start) / float(max(end - start, 1))


def build_hard_proto_separation_loss(loss_cfg: Dict[str, Any], embedding_dim: int) -> HardPrototypeSeparationLoss | None:
    if str((loss_cfg or {}).get("mode", "ce_only")) != "ce_hard_proto_sep":
        return None
    cfg = (loss_cfg.get("hard_proto_sep", {}) or {}) if loss_cfg else {}
    if not bool(cfg.get("enabled", False)):
        return None
    return HardPrototypeSeparationLoss(
        embedding_dim=int(embedding_dim),
        hard_class_ids=cfg.get("hard_class_ids", [0, 2, 4, 6]),
        temperature=float(cfg.get("temperature", 0.2)),
        margin=float(cfg.get("margin", 0.2)),
        margin_weight=float(cfg.get("margin_weight", 0.5)),
        normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
    )
