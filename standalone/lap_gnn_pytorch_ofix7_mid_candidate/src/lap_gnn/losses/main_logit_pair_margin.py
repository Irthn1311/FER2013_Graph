"""Main-logit hard-pair margin regularizer for D16.

This loss is training-only and parameter-free. It acts directly on the main
classifier logits, so inference architecture and checkpoint state stay unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ACTIVE_PAIRS = {
    "fear_sad": {"true_class": 2, "confuser_class": 4},
    "sad_neutral": {"true_class": 4, "confuser_class": 6},
    "neutral_sad": {"true_class": 6, "confuser_class": 4},
}


class MainLogitPairMarginLoss(nn.Module):
    """Margin loss that asks true-class logits to beat selected confusers."""

    def __init__(self, margin: float = 0.15, active_pairs: Dict[str, Dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.margin = float(margin)
        pair_cfg = active_pairs or DEFAULT_ACTIVE_PAIRS
        if not pair_cfg:
            raise ValueError("main-logit pair margin active_pairs must not be empty")
        self.pair_names: List[str] = []
        self.true_classes: Dict[str, int] = {}
        self.confuser_classes: Dict[str, int] = {}
        for name, cfg in pair_cfg.items():
            pair_name = str(name)
            self.pair_names.append(pair_name)
            self.true_classes[pair_name] = int(cfg["true_class"])
            self.confuser_classes[pair_name] = int(cfg["confuser_class"])

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
        zero = logits.float().sum() * 0.0
        losses: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {
            "main_logit_pair_margin_loss": zero,
            "pair_margin_available_count": logits.new_tensor(0.0),
        }
        for name in self.pair_names:
            true_class = self.true_classes[name]
            confuser_class = self.confuser_classes[name]
            if true_class >= logits.size(1) or confuser_class >= logits.size(1):
                raise ValueError(
                    f"Pair {name!r} class id exceeds logits classes={logits.size(1)}: "
                    f"true={true_class}, confuser={confuser_class}"
                )
            mask = y.eq(true_class)
            count = int(mask.sum().detach().cpu().item())
            key = name.replace("-", "_")
            if count <= 0:
                stats[f"pair_margin_loss_{key}"] = zero
                stats[f"pair_margin_count_{key}"] = logits.new_tensor(0.0)
                stats[f"mean_margin_violation_{key}"] = logits.new_tensor(float("nan"))
                stats[f"pair_margin_satisfied_ratio_{key}"] = logits.new_tensor(float("nan"))
                continue
            pair_logits = logits[mask].float()
            violation = self.margin + pair_logits[:, confuser_class] - pair_logits[:, true_class]
            margin_loss = F.relu(violation)
            loss = margin_loss.mean()
            satisfied = violation.le(0.0).float().mean()
            losses.append(loss)
            stats[f"pair_margin_loss_{key}"] = loss
            stats[f"pair_margin_count_{key}"] = logits.new_tensor(float(count))
            stats[f"mean_margin_violation_{key}"] = margin_loss.detach().mean()
            stats[f"pair_margin_satisfied_ratio_{key}"] = satisfied.detach()
        if losses:
            stats["main_logit_pair_margin_loss"] = torch.stack(losses).mean()
            stats["pair_margin_available_count"] = logits.new_tensor(float(len(losses)))
        return stats


def main_logit_pair_margin_lambda(loss_cfg: Dict[str, Any], epoch: int) -> float:
    cfg = (loss_cfg.get("main_logit_pair_margin", {}) or {}) if loss_cfg else {}
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


def build_main_logit_pair_margin_loss(loss_cfg: Dict[str, Any]) -> MainLogitPairMarginLoss | None:
    if str((loss_cfg or {}).get("mode", "ce_only")) != "ce_main_logit_pair_margin":
        return None
    cfg = (loss_cfg.get("main_logit_pair_margin", {}) or {}) if loss_cfg else {}
    if not bool(cfg.get("enabled", False)):
        return None
    return MainLogitPairMarginLoss(
        margin=float(cfg.get("margin", 0.15)),
        active_pairs=cfg.get("active_pairs") or DEFAULT_ACTIVE_PAIRS,
    )
