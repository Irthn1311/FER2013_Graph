"""Part-aware supervised contrastive loss for D16."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class PartAwareSupConLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def _single_part_loss(self, z: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
        valid_idx = torch.where(valid.bool())[0]
        if valid_idx.numel() < 2:
            return z.new_tensor(0.0), 0
        z = F.normalize(z[valid_idx], dim=1)
        labels = labels[valid_idx]
        logits = torch.matmul(z, z.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
        positives = same & ~eye
        positive_count = int(positives.sum().item())
        if positive_count == 0:
            return z.new_tensor(0.0), 0
        exp_logits = torch.exp(logits) * (~eye).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
        mean_log_prob_pos = (log_prob * positives.float()).sum(dim=1) / positives.float().sum(dim=1).clamp_min(1.0)
        anchors = positives.any(dim=1)
        return -mean_log_prob_pos[anchors].mean(), positive_count

    def forward(
        self,
        part_embeddings: Dict[str, torch.Tensor],
        valid_part_groups: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        detected: torch.Tensor | None = None,
        skip_fallback: bool = True,
    ) -> Dict[str, torch.Tensor]:
        losses = []
        positive_pairs = 0
        skipped_parts = 0
        no_positive_parts = 0
        per_part: Dict[str, torch.Tensor] = {}
        for name, z in part_embeddings.items():
            if name == "global":
                continue
            valid = valid_part_groups.get(name)
            if valid is None:
                skipped_parts += 1
                continue
            if skip_fallback and detected is not None:
                valid = valid.bool() & detected.bool()
            loss, count = self._single_part_loss(z, labels, valid)
            per_part[f"loss_part_supcon_{name}"] = loss.detach()
            if count > 0:
                losses.append(loss)
                positive_pairs += count
            else:
                no_positive_parts += 1
        if not losses:
            total = labels.new_tensor(0.0, dtype=torch.float32)
        else:
            total = torch.stack(losses).mean()
        out = {
            "loss_part_supcon": total,
            "part_supcon_positive_pair_count": total.new_tensor(float(positive_pairs)),
            "part_supcon_active_parts": total.new_tensor(float(len(losses))),
            "part_supcon_no_positive_parts": total.new_tensor(float(no_positive_parts)),
            "part_supcon_skipped_parts": total.new_tensor(float(skipped_parts)),
        }
        out.update(per_part)
        return out
