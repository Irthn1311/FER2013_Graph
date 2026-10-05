"""Probe architecture implementations and unit test for P0, P1, and P2."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpg_fer_v2_3.motif import build_aligned_support_indices


class CanonicalReadout(nn.Module):
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
        m_mean = h_motif.mean(dim=1)
        m_max = h_motif.max(dim=1).values
        attn_scores = self.motif_attn_pool(h_motif)
        attn_weights = F.softmax(attn_scores, dim=1)
        m_attention = (attn_weights * h_motif).sum(dim=1)
        r_m = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1))
        logits = self.aux_motif_head(r_m)
        return logits, r_m


class P0BaselineProbe(nn.Module):
    """P0 Baseline: Canonical motif readout trained on frozen h_m^(L)."""

    def __init__(self, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits, r_m = self.readout(h_m_L)
        return logits, {"motif_descriptor": r_m}


class P1LocalPixelReinjectionProbe(nn.Module):
    """P1 Local Pixel Reinjection Control: d_m = sum_i p_(m,i) v_i."""

    def __init__(self, d_pixel: int = 96, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.d_pixel = d_pixel
        self.d_motif = d_motif
        self.w_v = nn.Linear(d_pixel, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L: torch.Tensor, h_support_pixels: torch.Tensor, p_m_i: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # h_m_L: [B, 49, 192]
        # h_support_pixels: [B, 49, 256, 96]
        # p_m_i: [B, 49, 256]
        v = self.w_v(h_support_pixels) # [B, 49, 256, 192]
        d_m = (p_m_i.unsqueeze(-1) * v).sum(dim=2) # [B, 49, 192]

        cat_feat = torch.cat([h_m_L, d_m], dim=-1) # [B, 49, 384]
        g_m = torch.sigmoid(self.w_g(cat_feat)) # [B, 49, 1]
        residual = self.w_d(d_m)
        h_prime = self.ln(h_m_L + g_m * residual)

        logits, r_m = self.readout(h_prime)

        norm_residual = torch.norm(residual, dim=-1) # [B, 49]
        norm_h_L = torch.norm(h_m_L, dim=-1) # [B, 49]
        norm_ratio = (norm_residual / (norm_h_L + 1e-8)).mean()

        return logits, {
            "gate_values": g_m.squeeze(-1),
            "retrieved_d_m": d_m,
            "norm_ratio": norm_ratio,
            "h_prime": h_prime,
            "motif_descriptor": r_m,
        }


class P2ContextualPixelReinspectionProbe(nn.Module):
    """P2 Contextual Local Pixel Reinspection: score = q_m^T k_i / sqrt(d) + log(p_(m,i) + eps)."""

    def __init__(self, d_pixel: int = 96, d_motif: int = 192, d_readout: int = 384, num_classes: int = 7) -> None:
        super().__init__()
        self.d_pixel = d_pixel
        self.d_motif = d_motif
        self.d_q = 96 # project h_m_L to 96 to match k_i
        self.w_q = nn.Linear(d_motif, self.d_q, bias=False)
        self.w_k = nn.Linear(d_pixel, self.d_q, bias=False)
        self.w_v = nn.Linear(d_pixel, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(
        self,
        h_m_L: torch.Tensor,
        h_support_pixels: torch.Tensor,
        p_m_i: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        # h_m_L: [B, 49, 192]
        # h_support_pixels: [B, 49, 256, 96]
        # p_m_i: [B, 49, 256]
        batch, occurrences, support_size, _ = h_support_pixels.shape

        q_m = self.w_q(h_m_L).unsqueeze(2) # [B, 49, 1, 96]
        k_i = self.w_k(h_support_pixels) # [B, 49, 256, 96]
        v_i = self.w_v(h_support_pixels) # [B, 49, 256, 192]

        content_scores = (q_m * k_i).sum(dim=-1) / math.sqrt(self.d_q) # [B, 49, 256]

        # Eligible mask where p_m_i > 0
        eligible_mask = p_m_i > 0
        log_prior = torch.zeros_like(p_m_i)
        log_prior[eligible_mask] = torch.log(p_m_i[eligible_mask] + eps)
        log_prior[~eligible_mask] = -1e9

        total_scores = content_scores + log_prior
        a_m_i = F.softmax(total_scores, dim=-1) # [B, 49, 256]

        d_m = (a_m_i.unsqueeze(-1) * v_i).sum(dim=2) # [B, 49, 192]

        cat_feat = torch.cat([h_m_L, d_m], dim=-1) # [B, 49, 384]
        g_m = torch.sigmoid(self.w_g(cat_feat)) # [B, 49, 1]
        residual = self.w_d(d_m)
        h_prime = self.ln(h_m_L + g_m * residual)

        logits, r_m = self.readout(h_prime)

        # Diagnostics:
        norm_residual = torch.norm(residual, dim=-1) # [B, 49]
        norm_h_L = torch.norm(h_m_L, dim=-1) # [B, 49]
        norm_ratio = (norm_residual / (norm_h_L + 1e-8)).mean()

        return logits, {
            "a_m_i": a_m_i,
            "gate_values": g_m.squeeze(-1),
            "retrieved_d_m": d_m,
            "norm_ratio": norm_ratio,
            "h_prime": h_prime,
            "motif_descriptor": r_m,
        }


def test_pixel_probes():
    B = 2
    h_m_L = torch.randn(B, 49, 192)
    h_support_pixels = torch.randn(B, 49, 256, 96)
    p_m_i = F.softmax(torch.randn(B, 49, 256), dim=-1)

    p0 = P0BaselineProbe()
    p1 = P1LocalPixelReinjectionProbe()
    p2 = P2ContextualPixelReinspectionProbe()

    l0, d0 = p0(h_m_L)
    l1, d1 = p1(h_m_L, h_support_pixels, p_m_i)
    l2, d2 = p2(h_m_L, h_support_pixels, p_m_i)

    assert l0.shape == (B, 7)
    assert l1.shape == (B, 7)
    assert l2.shape == (B, 7)
    assert d1["gate_values"].shape == (B, 49)
    assert d2["gate_values"].shape == (B, 49)
    assert d2["a_m_i"].shape == (B, 49, 256)

    params_0 = sum(p.numel() for p in p0.parameters())
    params_1 = sum(p.numel() for p in p1.parameters())
    params_2 = sum(p.numel() for p in p2.parameters())

    print(f"P0 Parameters: {params_0:,}")
    print(f"P1 Parameters: {params_1:,}")
    print(f"P2 Parameters: {params_2:,}")
    print("All probe forward passes and shape invariants verified successfully!")


if __name__ == "__main__":
    test_pixel_probes()
