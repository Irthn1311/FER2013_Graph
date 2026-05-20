"""Slot bottleneck modules for D13B diagnostic runs.

These slots are diagnostic bottleneck candidates only. They are not semantic
regions and this module does not make motif claims.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MotifSlotAttention(nn.Module):
    """Learned slot queries attending over D13A region embeddings."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_slots: int = 8,
        slot_dim: int = 64,
        num_iters: int = 2,
        dropout: float = 0.1,
        normalize_slots: bool = True,
        eps: float = 1e-6,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.num_iters = int(num_iters)
        self.normalize_slots = bool(normalize_slots)
        self.eps = float(eps)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_iters <= 0:
            raise ValueError("num_iters must be positive")

        self.slot_init = nn.Parameter(torch.randn(self.num_slots, self.slot_dim) * 0.02)
        self.region_proj = nn.Linear(self.hidden_dim, self.slot_dim) if self.hidden_dim != self.slot_dim else nn.Identity()
        self.q_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.k_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.v_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.update = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
        )
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.ffn_norm = nn.LayerNorm(self.slot_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        region_embeddings: torch.Tensor,
        region_pos: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if region_embeddings.ndim != 3:
            raise ValueError(f"region_embeddings must be [B,K,D], got {tuple(region_embeddings.shape)}")
        bsz, num_regions, _ = region_embeddings.shape
        if region_mask is None:
            region_mask = torch.ones((bsz, num_regions), device=region_embeddings.device, dtype=torch.bool)
        region_mask = region_mask.bool()
        h = self.region_proj(region_embeddings)
        slots = self.slot_init.unsqueeze(0).expand(bsz, -1, -1)
        if self.normalize_slots:
            slots = self.slot_norm(slots)
        attn = None
        mask_bias = (~region_mask).unsqueeze(1)
        for _ in range(self.num_iters):
            q = self.q_proj(slots)
            k = self.k_proj(h)
            v = self.v_proj(h)
            logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.slot_dim))
            logits = logits.masked_fill(mask_bias, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            update = torch.matmul(attn, v)
            slots = self.slot_norm(slots + self.dropout(update))
            slots = self.ffn_norm(slots + self.dropout(self.update(slots)))
        assert attn is not None
        aux = self._stats(slots, attn, region_pos=region_pos, region_mask=region_mask)
        return slots, attn, aux

    def _stats(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        region_pos: Optional[torch.Tensor],
        region_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eps = self.eps
        bsz, num_slots, num_regions = attn.shape
        valid_counts = region_mask.float().sum(dim=1).clamp_min(1.0)
        entropy = -(attn * attn.clamp_min(eps).log()).sum(dim=-1) / valid_counts.log().clamp_min(eps).unsqueeze(-1)
        # Each slot attention is normalized over regions, so attention mass
        # alone is always one. Use slot activation norm as the diagnostic mass.
        slot_mass = slots.norm(dim=-1).clamp_min(eps)
        mass_dist = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
        mass_entropy = -(mass_dist * mass_dist.clamp_min(eps).log()).sum(dim=-1)
        effective_slots = torch.exp(mass_entropy)
        dominance = mass_dist.max(dim=-1).values
        flat = F.normalize(attn, p=2, dim=-1, eps=eps)
        sim = torch.matmul(flat, flat.transpose(1, 2))
        eye = torch.eye(num_slots, device=attn.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)
        overlap = off_diag.mean() if off_diag.numel() else attn.new_tensor(0.0)
        area_threshold = 1.0 / max(float(num_regions), 1.0)
        slot_area = (attn > area_threshold).float().sum(dim=-1)
        if region_pos is not None:
            if region_pos.ndim != 3:
                raise ValueError(f"region_pos must be [B,K,2], got {tuple(region_pos.shape)}")
            centers = torch.matmul(attn, region_pos.to(device=attn.device, dtype=attn.dtype))
            center_std = centers.std(dim=1, unbiased=False).mean()
        else:
            centers = attn.new_zeros((bsz, num_slots, 2))
            center_std = attn.new_tensor(0.0)
        balance_loss = ((mass_dist - (1.0 / num_slots)) ** 2).mean()
        overlap_loss = overlap
        diversity_loss = overlap
        entropy_loss = entropy.mean()
        return {
            "slot_entropy": entropy.mean(),
            "effective_slots": effective_slots.mean(),
            "slot_overlap": overlap,
            "slot_dominance": dominance.mean(),
            "slot_area_mean": slot_area.mean(),
            "slot_area_min": slot_area.amin(),
            "slot_area_max": slot_area.amax(),
            "slot_center_std": center_std,
            "slot_centers": centers.detach(),
            "slot_mass": mass_dist.detach(),
            "slot_diversity_loss": diversity_loss,
            "slot_overlap_loss": overlap_loss,
            "slot_entropy_loss": entropy_loss,
            "slot_balance_loss": balance_loss,
        }
