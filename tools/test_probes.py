"""Probe definitions and unit test for R0_REFIT, R1, and R2."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class R0RefitProbe(nn.Module):
    """Canonical motif readout architecture trained from scratch on frozen H_M."""

    def __init__(self, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.d_motif = d_motif
        self.motif_attn_pool = nn.Linear(d_motif, 1)
        self.motif_readout_proj = nn.Sequential(
            nn.Linear(d_motif * 3, d_readout),
            nn.LayerNorm(d_readout),
            nn.GELU(),
        )
        self.aux_motif_head = nn.Linear(d_readout, num_classes)

    def forward(self, h_motif: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_motif: [B, 49, 192]
        m_mean = h_motif.mean(dim=1)
        m_max = h_motif.max(dim=1).values
        attn_scores = self.motif_attn_pool(h_motif) # [B, 49, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # [B, 49, 1]
        m_attention = (attn_weights * h_motif).sum(dim=1) # [B, 192]

        r_m = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1)) # [B, 384]
        logits = self.aux_motif_head(r_m) # [B, 7]
        return logits, {
            "attention_weights": attn_weights.squeeze(-1),
            "motif_descriptor": r_m,
        }


class R1MultiSlotProbe(nn.Module):
    """Multi-slot class-agnostic readout with S=7 latent queries."""

    def __init__(
        self,
        d_motif: int = 192,
        num_slots: int = 7,
        d_slot_proj: int = 64,
        d_hidden: int = 256,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.d_motif = d_motif
        self.slots = nn.Parameter(torch.randn(num_slots, d_motif) / math.sqrt(d_motif))
        self.k_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.v_proj = nn.Linear(d_motif, d_motif, bias=False)

        self.slot_proj = nn.Sequential(
            nn.Linear(d_motif, d_slot_proj),
            nn.LayerNorm(d_slot_proj),
            nn.GELU(),
        )
        flat_dim = num_slots * d_slot_proj
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, h_motif: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_motif: [B, 49, 192]
        batch = h_motif.shape[0]
        k = self.k_proj(h_motif) # [B, 49, 192]
        v = self.v_proj(h_motif) # [B, 49, 192]

        # Q: [7, 192] -> expand to [B, 7, 192]
        q = self.slots.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_motif) # [B, 7, 49]
        attn = F.softmax(scores, dim=-1) # [B, 7, 49]

        slots_e = torch.matmul(attn, v) # [B, 7, 192]
        slots_projected = self.slot_proj(slots_e) # [B, 7, 64]
        flat_features = slots_projected.reshape(batch, -1) # [B, 7 * 64]
        logits = self.classifier(flat_features)

        return logits, {
            "attention_weights": attn,
            "slot_queries": self.slots,
            "slot_representations": slots_e,
        }


class R2ClassConditionedProbe(nn.Module):
    """Class-conditioned motif readout with C=7 learned class queries."""

    def __init__(self, d_motif: int = 192, num_classes: int = 7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_motif = d_motif
        self.class_queries = nn.Parameter(torch.randn(num_classes, d_motif) / math.sqrt(d_motif))
        self.k_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.v_proj = nn.Linear(d_motif, d_motif, bias=False)

        # Each class logit is produced from its class-conditioned representation: z_c = w_c^T r_c + b_c
        self.class_heads = nn.Linear(d_motif, 1, bias=True) # or separate per class
        # To strictly enforce z_c = w_c^T r_c + b_c per class:
        self.w_c = nn.Parameter(torch.randn(num_classes, d_motif) / math.sqrt(d_motif))
        self.b_c = nn.Parameter(torch.zeros(num_classes))

    def forward(self, h_motif: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_motif: [B, 49, 192]
        batch = h_motif.shape[0]
        k = self.k_proj(h_motif) # [B, 49, 192]
        v = self.v_proj(h_motif) # [B, 49, 192]

        # Class queries: [7, 192] -> expand to [B, 7, 192]
        q = self.class_queries.unsqueeze(0).expand(batch, -1, -1)
        # s_cj = q_c^T W_K h_j / sqrt(d) -> [B, 7, 49]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_motif)
        alpha = F.softmax(scores, dim=-1) # [B, 7, 49]

        # r_c = sum_j alpha_cj W_V h_j -> [B, 7, 192]
        r = torch.matmul(alpha, v)

        # z_c = w_c^T r_c + b_c:
        # r: [B, 7, 192], w_c: [7, 192]
        logits = (r * self.w_c.unsqueeze(0)).sum(dim=-1) + self.b_c.unsqueeze(0) # [B, 7]

        return logits, {
            "attention_weights": alpha,
            "class_queries": self.class_queries,
            "class_representations": r,
        }


def test_probes():
    x = torch.randn(4, 49, 192)
    
    r0 = R0RefitProbe()
    r1 = R1MultiSlotProbe()
    r2 = R2ClassConditionedProbe()

    l0, d0 = r0(x)
    l1, d1 = r1(x)
    l2, d2 = r2(x)

    assert l0.shape == (4, 7)
    assert l1.shape == (4, 7)
    assert l2.shape == (4, 7)

    p0 = sum(p.numel() for p in r0.parameters())
    p1 = sum(p.numel() for p in r1.parameters())
    p2 = sum(p.numel() for p in r2.parameters())

    print(f"R0 Refit Parameters: {p0:,}")
    print(f"R1 MultiSlot Parameters: {p1:,}")
    print(f"R2 ClassConditioned Parameters: {p2:,}")
    print("All probe forward passes and shapes verified successfully!")


if __name__ == "__main__":
    test_probes()
