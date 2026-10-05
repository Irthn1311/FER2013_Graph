"""Probe architecture implementations and unit test for C0, C1, and C2."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonicalReadout(nn.Module):
    """Exact canonical motif readout from MPG-FER v2.3."""

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

    def forward(self, h_motif: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h_motif: [B, 49, 192]
        m_mean = h_motif.mean(dim=1)
        m_max = h_motif.max(dim=1).values
        attn_scores = self.motif_attn_pool(h_motif)
        attn_weights = F.softmax(attn_scores, dim=1)
        m_attention = (attn_weights * h_motif).sum(dim=1)

        r_m = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1))
        logits = self.aux_motif_head(r_m)
        return logits, r_m


class C0BaselineProbe(nn.Module):
    """C0 Baseline: Canonical readout trained on frozen h_m^(L)."""

    def __init__(self, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits, r_m = self.readout(h_m_L)
        return logits, {"motif_descriptor": r_m}


class C1EarlySkipProbe(nn.Module):
    """C1 Early-State Skip Control: Gated combination of h_m^(L) and h_m^(0)."""

    def __init__(self, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.d_motif = d_motif
        self.w_skip = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L: torch.Tensor, h_m_0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_m_L, h_m_0: [B, 49, 192]
        cat_features = torch.cat([h_m_L, h_m_0], dim=-1) # [B, 49, 384]
        g_m = torch.sigmoid(self.w_g(cat_features)) # [B, 49, 1]
        skip_feat = self.w_skip(h_m_0) # [B, 49, 192]
        h_final = self.ln(h_m_L + g_m * skip_feat) # [B, 49, 192]

        logits, r_m = self.readout(h_final)
        return logits, {
            "gate_values": g_m.squeeze(-1), # [B, 49]
            "h_final": h_final,
            "motif_descriptor": r_m,
        }


class C2ContextualScaleRecompositionProbe(nn.Module):
    """C2 CSR: Occurrence-wise query from h_m^(L) attending to original scale candidates z_(m,s)."""

    def __init__(self, d_motif: int = 192, num_scales: int = 3, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.d_motif = d_motif
        self.num_scales = num_scales
        self.w_q = nn.Linear(d_motif, d_motif, bias=False)
        self.w_k = nn.Linear(d_motif, d_motif, bias=False)
        self.w_v = nn.Linear(d_motif, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L: torch.Tensor, z_m_s: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_m_L: [B, 49, 192]
        # z_m_s: [B, 49, 3, 192]
        batch, occurrences, scales, dim = z_m_s.shape
        assert occurrences == 49 and scales == 3 and dim == self.d_motif

        # Query: [B, 49, 1, 192]
        q_m = self.w_q(h_m_L).unsqueeze(2)

        # Keys and Values: [B, 49, 3, 192]
        k_ms = self.w_k(z_m_s)
        v_ms = self.w_v(z_m_s)

        # Attention scores: q_m^T k_ms / sqrt(d) -> [B, 49, 1, 3] -> squeeze to [B, 49, 3]
        scores = (q_m * k_ms).sum(dim=-1, keepdim=True) / math.sqrt(dim) # [B, 49, 3, 1]
        scores = scores.squeeze(-1) # [B, 49, 3]
        beta = F.softmax(scores, dim=-1) # [B, 49, 3]

        # Retrieved scale evidence: sum_s beta_(m,s) v_(m,s) -> [B, 49, 192]
        d_m = (beta.unsqueeze(-1) * v_ms).sum(dim=2)

        # Gated fusion
        cat_features = torch.cat([h_m_L, d_m], dim=-1) # [B, 49, 384]
        g_m = torch.sigmoid(self.w_g(cat_features)) # [B, 49, 1]
        h_prime = self.ln(h_m_L + g_m * self.w_d(d_m)) # [B, 49, 192]

        logits, r_m = self.readout(h_prime)
        return logits, {
            "beta": beta, # [B, 49, 3]
            "gate_values": g_m.squeeze(-1), # [B, 49]
            "retrieved_evidence": d_m,
            "h_prime": h_prime,
            "motif_descriptor": r_m,
        }


def test_csr_probes():
    B = 4
    h_m_L = torch.randn(B, 49, 192)
    h_m_0 = torch.randn(B, 49, 192)
    z_m_s = torch.randn(B, 49, 3, 192)

    c0 = C0BaselineProbe()
    c1 = C1EarlySkipProbe()
    c2 = C2ContextualScaleRecompositionProbe()

    l0, d0 = c0(h_m_L)
    l1, d1 = c1(h_m_L, h_m_0)
    l2, d2 = c2(h_m_L, z_m_s)

    assert l0.shape == (B, 7)
    assert l1.shape == (B, 7)
    assert l2.shape == (B, 7)
    assert d2["beta"].shape == (B, 49, 3)
    assert d2["gate_values"].shape == (B, 49)

    p0 = sum(p.numel() for p in c0.parameters())
    p1 = sum(p.numel() for p in c1.parameters())
    p2 = sum(p.numel() for p in c2.parameters())

    print(f"C0 Baseline Parameters: {p0:,}")
    print(f"C1 Early Skip Parameters: {p1:,}")
    print(f"C2 CSR Parameters: {p2:,}")
    print("All probe forward passes and shape invariants verified successfully!")


if __name__ == "__main__":
    test_csr_probes()
