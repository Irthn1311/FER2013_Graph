"""D12A global-local motif model for FER-2013 pixel graphs.

This module is intentionally independent from D10/D11 classes while preserving
the same dense full-graph batch contract used by the existing trainer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class ContextAwareEdgeGatedLayer(nn.Module):
    """Edge-gated message passing with source/destination-aware gates."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.2,
        eps: float = 1e-6,
        use_gate_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.eps = float(eps)
        self.use_gate_norm = bool(use_gate_norm)

        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(self.edge_dim),
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm_msg = nn.LayerNorm(self.hidden_dim)
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"h must be [B, N, H], got {tuple(h.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
        if edge_attr.ndim != 3:
            raise ValueError(f"edge_attr must be [B, E, F], got {tuple(edge_attr.shape)}")
        if edge_attr.shape[0] != h.shape[0]:
            raise ValueError(f"edge_attr batch {edge_attr.shape[0]} != h batch {h.shape[0]}")

        bsz, num_nodes, hidden_dim = h.shape
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() != edge_attr.shape[1]:
            raise ValueError(
                f"edge_index has {src.numel()} edges but edge_attr has {edge_attr.shape[1]}"
            )
        if int(src.max()) >= num_nodes or int(dst.max()) >= num_nodes:
            raise ValueError("edge_index contains node id outside h.shape[1]")

        h_src = h.index_select(dim=1, index=src)
        h_dst = h.index_select(dim=1, index=dst)
        edge_emb = self.edge_mlp(edge_attr)

        gate_input = torch.cat([edge_emb, h_src, h_dst, h_src - h_dst], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        msg_input = torch.cat([h_src, edge_emb], dim=-1)
        msg = self.msg_mlp(msg_input) * gate

        numerator = msg.new_zeros((bsz, num_nodes, hidden_dim))
        numerator.index_add_(1, dst, msg)
        if self.use_gate_norm:
            denom = gate.new_zeros((bsz, num_nodes, hidden_dim))
            denom.index_add_(1, dst, gate)
            agg = numerator / denom.clamp_min(self.eps)
        else:
            degree = h.new_zeros((num_nodes,))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = numerator / degree.to(dtype=numerator.dtype).view(1, -1, 1).clamp_min(1.0)

        update = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm_msg(h + self.dropout(update))
        h_new = self.norm_ffn(h_new + self.dropout(self.ffn(h_new)))
        if node_mask is not None:
            h_new = h_new * node_mask.to(device=h_new.device, dtype=h_new.dtype).unsqueeze(-1)

        diagnostics = {
            "gate_mean": gate.detach().mean(),
            "gate_std": gate.detach().std(unbiased=False),
            "gate_min": gate.detach().amin(),
            "gate_max": gate.detach().amax(),
        }
        return h_new, diagnostics


class D12PixelEncoder(nn.Module):
    """Context-aware multi-scale edge-gated pixel encoder."""

    SCALE2_OFFSETS = (
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-2, -2),
        (-2, 2),
        (2, -2),
        (2, 2),
    )

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_scale2: bool = True,
        scale2_alpha: float = 1.0,
        use_gate_norm: bool = True,
        height: int = 48,
        width: int = 48,
        enable_micro_diagnostics: bool = False,
        save_attention_maps: bool = False,
        save_node_similarity: bool = False,
        diagnostic_max_samples: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_scale2 = bool(use_scale2)
        self.scale2_alpha = float(scale2_alpha)
        self.height = int(height)
        self.width = int(width)
        self.internal_edge_dim = self.edge_dim + 1
        self.enable_micro_diagnostics = bool(enable_micro_diagnostics)
        self.save_attention_maps = bool(save_attention_maps)
        self.save_node_similarity = bool(save_node_similarity)
        self.diagnostic_max_samples = max(int(diagnostic_max_samples), 1)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.scale1_layers = nn.ModuleList(
            ContextAwareEdgeGatedLayer(
                hidden_dim=self.hidden_dim,
                edge_dim=self.internal_edge_dim,
                dropout=dropout,
                use_gate_norm=use_gate_norm,
            )
            for _ in range(self.num_layers)
        )
        self.scale2_layers = nn.ModuleList(
            ContextAwareEdgeGatedLayer(
                hidden_dim=self.hidden_dim,
                edge_dim=self.internal_edge_dim,
                dropout=dropout,
                use_gate_norm=use_gate_norm,
            )
            for _ in range(self.num_layers)
        )
        scale2_edge_index = self._build_scale2_edge_index()
        self.register_buffer("scale2_edge_index", scale2_edge_index, persistent=False)

    def set_micro_diagnostics(
        self,
        *,
        enabled: bool = True,
        save_node_similarity: Optional[bool] = None,
        save_attention_maps: Optional[bool] = None,
        diagnostic_max_samples: Optional[int] = None,
    ) -> None:
        self.enable_micro_diagnostics = bool(enabled)
        if save_node_similarity is not None:
            self.save_node_similarity = bool(save_node_similarity)
        if save_attention_maps is not None:
            self.save_attention_maps = bool(save_attention_maps)
        if diagnostic_max_samples is not None:
            self.diagnostic_max_samples = max(int(diagnostic_max_samples), 1)

    def _build_scale2_edge_index(self) -> torch.Tensor:
        src_nodes = []
        dst_nodes = []
        for y in range(self.height):
            for x in range(self.width):
                src = y * self.width + x
                for dy, dx in self.SCALE2_OFFSETS:
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < self.height and 0 <= xx < self.width:
                        src_nodes.append(src)
                        dst_nodes.append(yy * self.width + xx)
        return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    def _expand_edge_attr(self, edge_attr: torch.Tensor, bsz: int) -> torch.Tensor:
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(bsz, -1, -1)
        if edge_attr.ndim != 3:
            raise ValueError(f"edge_attr must be [B, E, F] or [E, F], got {tuple(edge_attr.shape)}")
        if edge_attr.shape[0] != bsz:
            raise ValueError(f"edge_attr batch {edge_attr.shape[0]} != x batch {bsz}")
        if edge_attr.shape[-1] != self.edge_dim:
            raise ValueError(f"edge_attr dim {edge_attr.shape[-1]} != configured edge_dim {self.edge_dim}")
        return edge_attr

    def _with_scale_value(self, edge_attr: torch.Tensor, scale_value: float) -> torch.Tensor:
        scale = edge_attr.new_full((*edge_attr.shape[:-1], 1), float(scale_value))
        return torch.cat([edge_attr, scale], dim=-1)

    def _scale2_edge_attr(self, x: torch.Tensor) -> torch.Tensor:
        edge_index = self.scale2_edge_index.to(device=x.device)
        src = edge_index[0]
        dst = edge_index[1]
        src_y = torch.div(src, self.width, rounding_mode="floor")
        src_x = src % self.width
        dst_y = torch.div(dst, self.width, rounding_mode="floor")
        dst_x = dst % self.width
        dx = (dst_x - src_x).to(dtype=x.dtype) / max(float(self.width - 1), 1.0)
        dy = (dst_y - src_y).to(dtype=x.dtype) / max(float(self.height - 1), 1.0)
        dist = torch.sqrt(dx.pow(2) + dy.pow(2))

        intensity = x[:, :, 0]
        delta_intensity = (intensity.index_select(1, src) - intensity.index_select(1, dst)).abs()
        intensity_similarity = torch.exp(-self.scale2_alpha * delta_intensity)

        static = torch.stack([dx, dy, dist], dim=-1).to(device=x.device)
        static = static.unsqueeze(0).expand(x.shape[0], -1, -1)
        dynamic = torch.stack([delta_intensity, intensity_similarity], dim=-1)
        return torch.cat([static, dynamic], dim=-1)

    @staticmethod
    def _scalar_stats(prefix: str, tensor: torch.Tensor, diagnostics: Dict[str, torch.Tensor]) -> None:
        value = tensor.detach().float()
        diagnostics[f"{prefix}_mean"] = value.mean()
        diagnostics[f"{prefix}_std"] = value.std(unbiased=False)

    def _coords_from_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= 3:
            return x.detach()[:, :, 1:3].float()
        ys = torch.linspace(0.0, 1.0, self.height, device=x.device)
        xs = torch.linspace(0.0, 1.0, self.width, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).expand(x.shape[0], -1, -1)

    @staticmethod
    def _mean_pairwise_cosine(h: torch.Tensor, mask: torch.Tensor, max_nodes: int = 128) -> torch.Tensor:
        h = h.detach().float()
        values = []
        for sample_idx in range(min(int(h.shape[0]), int(mask.shape[0]))):
            idx = mask[sample_idx].nonzero(as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            if idx.numel() > max_nodes:
                take = torch.linspace(
                    0,
                    idx.numel() - 1,
                    steps=max_nodes,
                    device=idx.device,
                ).round().long()
                idx = idx.index_select(0, take)
            z = F.normalize(h[sample_idx].index_select(0, idx), dim=-1)
            sim = z @ z.transpose(0, 1)
            denom = max(int(idx.numel()) * (int(idx.numel()) - 1), 1)
            values.append((sim.sum() - sim.diag().sum()) / float(denom))
        if not values:
            return h.new_zeros(())
        return torch.stack(values).mean()

    def _region_masks(self, x: torch.Tensor, node_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        coords = self._coords_from_x(x)
        xx = coords[:, :, 0]
        yy = coords[:, :, 1]
        valid = torch.ones_like(xx, dtype=torch.bool)
        if node_mask is not None:
            valid = node_mask.to(device=x.device, dtype=torch.bool)
        return {
            "eye": valid & (yy >= 0.20) & (yy <= 0.45) & (xx >= 0.15) & (xx <= 0.85),
            "nose_mouth": valid & (yy >= 0.40) & (yy <= 0.75) & (xx >= 0.25) & (xx <= 0.75),
            "center": valid & (yy >= 0.15) & (yy <= 0.85) & (xx >= 0.15) & (xx <= 0.85),
            "border": valid & ((xx <= 0.08) | (xx >= 0.92) | (yy <= 0.08) | (yy >= 0.92)),
        }

    def _add_micro_diagnostics(
        self,
        diagnostics: Dict[str, torch.Tensor],
        *,
        x: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        h_input_proj: torch.Tensor,
        h_after_scale1: torch.Tensor,
        h_after_scale2: torch.Tensor,
        h_pixel_final: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self._scalar_stats("encoder_input", h_input_proj, diagnostics)
            self._scalar_stats("encoder_scale1", h_after_scale1, diagnostics)
            self._scalar_stats("encoder_scale2", h_after_scale2, diagnostics)
            self._scalar_stats("encoder_final", h_pixel_final, diagnostics)
            delta = (h_after_scale2.detach().float() - h_after_scale1.detach().float()).norm(dim=-1)
            scale1_norm = h_after_scale1.detach().float().norm(dim=-1)
            delta_norm = delta.mean()
            diagnostics["encoder_scale2_delta_norm"] = delta_norm
            diagnostics["encoder_scale2_delta_ratio"] = delta_norm / scale1_norm.mean().clamp_min(1e-8)

            if self.save_node_similarity:
                max_samples = min(self.diagnostic_max_samples, x.shape[0])
                region_masks = self._region_masks(x[:max_samples], node_mask[:max_samples] if node_mask is not None else None)
                h1 = h_after_scale1[:max_samples]
                h2 = h_after_scale2[:max_samples]
                for region, mask in region_masks.items():
                    c1 = self._mean_pairwise_cosine(h1, mask, max_nodes=128)
                    c2 = self._mean_pairwise_cosine(h2, mask, max_nodes=128)
                    diagnostics[f"cos_{region}_scale1"] = c1
                    diagnostics[f"cos_{region}_scale2"] = c2
                    diagnostics[f"cos_{region}_delta"] = c2 - c1

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.node_dim:
            raise ValueError(f"x dim {x.shape[-1]} != configured node_dim {self.node_dim}")
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E] or [B, 2, E], got {tuple(edge_index.shape)}")
        edge_attr = self._expand_edge_attr(edge_attr, x.shape[0])

        h = self.input_proj(x)
        if node_mask is not None:
            h = h * node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        h_input_proj = h

        diagnostics: Dict[str, torch.Tensor] = {}
        edge_attr_scale1 = self._with_scale_value(edge_attr, 1.0)
        h_after_scale1: Optional[torch.Tensor] = None
        h_after_scale2: Optional[torch.Tensor] = None
        for idx, layer in enumerate(self.scale1_layers):
            h, diag = layer(h, edge_index, edge_attr_scale1, node_mask=node_mask)
            if idx == 0:
                h_after_scale1 = h
            diagnostics[f"scale1_layer{idx}_gate_mean"] = diag["gate_mean"]
            diagnostics[f"scale1_layer{idx}_gate_std"] = diag["gate_std"]
            diagnostics[f"scale1_layer{idx}_gate_min"] = diag["gate_min"]
            diagnostics[f"scale1_layer{idx}_gate_max"] = diag["gate_max"]

            if self.use_scale2:
                edge_attr_scale2 = self._with_scale_value(self._scale2_edge_attr(x), 2.0)
                h, diag2 = self.scale2_layers[idx](
                    h,
                    self.scale2_edge_index.to(device=x.device),
                    edge_attr_scale2,
                    node_mask=node_mask,
                )
                diagnostics[f"scale2_layer{idx}_gate_mean"] = diag2["gate_mean"]
                diagnostics[f"scale2_layer{idx}_gate_std"] = diag2["gate_std"]
                diagnostics[f"scale2_layer{idx}_gate_min"] = diag2["gate_min"]
                diagnostics[f"scale2_layer{idx}_gate_max"] = diag2["gate_max"]
                if idx == 0:
                    h_after_scale2 = h

        diagnostics["encoder_gate_mean"] = torch.stack(
            [v for k, v in diagnostics.items() if k.endswith("gate_mean")]
        ).mean()
        diagnostics["encoder_gate_std"] = torch.stack(
            [v for k, v in diagnostics.items() if k.endswith("gate_std")]
        ).mean()
        diagnostics["encoder_gate_min"] = torch.stack(
            [v for k, v in diagnostics.items() if k.endswith("gate_min")]
        ).amin()
        diagnostics["encoder_gate_max"] = torch.stack(
            [v for k, v in diagnostics.items() if k.endswith("gate_max")]
        ).amax()
        diagnostics["scale2_edge_count"] = x.new_tensor(float(self.scale2_edge_index.shape[1]))
        if self.enable_micro_diagnostics:
            h_after_scale1 = h_after_scale1 if h_after_scale1 is not None else h_input_proj
            h_after_scale2 = h_after_scale2 if h_after_scale2 is not None else h_after_scale1
            self._add_micro_diagnostics(
                diagnostics,
                x=x,
                node_mask=node_mask,
                h_input_proj=h_input_proj,
                h_after_scale1=h_after_scale1,
                h_after_scale2=h_after_scale2,
                h_pixel_final=h,
            )
        return h, diagnostics


class IterativeSlotAttentionD12(nn.Module):
    """D10-style iterative slot attention with slot-wise pixel competition."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_slots: int = 8,
        num_iterations: int = 5,
        dropout: float = 0.2,
        residual_slot_connection: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.residual_slot_connection = bool(residual_slot_connection)
        self.eps = float(eps)

        self.slot_mu = nn.Parameter(torch.randn(1, self.num_slots, self.hidden_dim) * 0.02)
        self.norm_inputs = nn.LayerNorm(self.hidden_dim)
        self.norm_slots = nn.LayerNorm(self.hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.norm_mlp = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(
        self,
        h_pixel: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = h_pixel.shape[0]
        inputs = self.norm_inputs(h_pixel)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        slots = self.slot_mu.expand(bsz, -1, -1).clone()
        attn_maps = h_pixel.new_zeros((bsz, self.num_slots, h_pixel.shape[1]))
        attn_weights = attn_maps

        for _ in range(self.num_iterations):
            slots_prev = slots
            query = self.query_proj(self.norm_slots(slots))
            attn_logits = torch.einsum("bkh,bnh->bkn", query, key) / math.sqrt(float(self.hidden_dim))
            if node_mask is not None:
                attn_logits = attn_logits.masked_fill(~node_mask.bool().unsqueeze(1), -1e4)
            attn_maps = torch.softmax(attn_logits, dim=1)
            if node_mask is not None:
                attn_maps = attn_maps * node_mask.to(dtype=attn_maps.dtype, device=attn_maps.device).unsqueeze(1)
            attn_weights = attn_maps / attn_maps.sum(dim=2, keepdim=True).clamp_min(self.eps)
            updates = torch.einsum("bkn,bnh->bkh", attn_weights, value)
            slots = self.gru(
                updates.reshape(-1, self.hidden_dim),
                slots_prev.reshape(-1, self.hidden_dim),
            ).reshape(bsz, self.num_slots, self.hidden_dim)
            if self.residual_slot_connection:
                slots = slots + slots_prev
            slots = slots + self.mlp(self.norm_mlp(slots))

        slot_centers = torch.einsum("bkn,bnd->bkd", attn_weights, coords)
        return slots, attn_maps, slot_centers


class MotifRelationTransformerD12(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, slots_raw: torch.Tensor) -> torch.Tensor:
        return self.transformer(slots_raw)


class VirtualNodeGatherD12(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        global_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.value_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(int(hidden_dim), int(global_dim)),
            nn.LayerNorm(int(global_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        h_pixel: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score(h_pixel).squeeze(-1)
        if node_mask is not None:
            score = score.masked_fill(~node_mask.bool(), -1e4)
        virtual_attention = torch.softmax(score, dim=1)
        value = self.value_proj(h_pixel)
        pooled = torch.einsum("bn,bnh->bh", virtual_attention, value)
        global_context = self.bottleneck_proj(pooled)
        return global_context, virtual_attention


class FiLMFusionD12(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        global_dim: int = 64,
        num_slots: int = 8,
        global_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.dropout = nn.Dropout(float(global_dropout))
        self.gamma_mlp = nn.Sequential(
            nn.LayerNorm(int(global_dim)),
            nn.Linear(int(global_dim), int(global_dim)),
            nn.GELU(),
            nn.Linear(int(global_dim), self.num_slots * self.hidden_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.LayerNorm(int(global_dim)),
            nn.Linear(int(global_dim), int(global_dim)),
            nn.GELU(),
            nn.Linear(int(global_dim), self.num_slots * self.hidden_dim),
        )
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.normal_(self.gamma_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.beta_mlp[-1].bias)
        nn.init.normal_(self.beta_mlp[-1].weight, mean=0.0, std=1e-3)

    def forward(
        self,
        slots_context: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = slots_context.shape[0]
        global_context = self.dropout(global_context)
        gamma = torch.tanh(self.gamma_mlp(global_context)).view(
            bsz, self.num_slots, self.hidden_dim
        )
        beta = self.beta_mlp(global_context).view(bsz, self.num_slots, self.hidden_dim)
        slots_refined = slots_context * (1.0 + gamma) + beta
        return slots_refined, gamma, beta


class ClassMotifAttentionHeadD12(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        num_classes: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.logit_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.key(slots)
        value = self.value(slots)
        scores = torch.einsum("ch,bkh->bck", self.class_queries, key)
        scores = scores / math.sqrt(float(self.hidden_dim))
        class_motif_attn = torch.softmax(scores, dim=2)
        class_repr = torch.einsum("bck,bkh->bch", class_motif_attn, value)
        logits = self.logit_head(self.dropout(class_repr)).squeeze(-1)
        return logits, class_motif_attn, class_repr


class D12GlobalLocalMotifModel(nn.Module):
    """D12A mainline model with dense full-graph trainer compatibility."""

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 96,
        dropout: float = 0.2,
        encoder: Optional[Dict[str, Any]] = None,
        num_slots: int = 8,
        slot_iterations: int = 5,
        residual_slot_connection: bool = True,
        motif_relation_layers: int = 1,
        motif_relation_heads: int = 4,
        use_global_branch: bool = True,
        global_dim: int = 64,
        global_dropout: float = 0.3,
        supcon_projection_dim: int = 128,
        height: int = 48,
        width: int = 48,
        diagnostics: Optional[Dict[str, Any]] = None,
        enable_micro_diagnostics: bool = False,
        save_attention_maps: bool = False,
        save_node_similarity: bool = False,
        diagnostic_max_samples: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.use_global_branch = bool(use_global_branch)
        self.global_dim = int(global_dim)
        self.height = int(height)
        self.width = int(width)
        if self.num_nodes != self.height * self.width:
            raise ValueError(f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}")
        if self.num_slots != 8:
            raise ValueError("D12A mainline expects num_slots=8")
        if self.global_dim != 64:
            raise ValueError("D12A mainline expects global_dim=64")

        diag_cfg = dict(diagnostics or {})
        enable_micro_diagnostics = bool(
            diag_cfg.get("enable_micro_diagnostics", enable_micro_diagnostics)
        )
        save_attention_maps = bool(diag_cfg.get("save_attention_maps", save_attention_maps))
        save_node_similarity = bool(diag_cfg.get("save_node_similarity", save_node_similarity))
        diagnostic_max_samples = int(diag_cfg.get("diagnostic_max_samples", diagnostic_max_samples))
        self.enable_micro_diagnostics = bool(enable_micro_diagnostics)
        self.save_attention_maps = bool(save_attention_maps)
        self.save_node_similarity = bool(save_node_similarity)
        self.diagnostic_max_samples = max(int(diagnostic_max_samples), 1)

        encoder_cfg = dict(encoder or {})
        encoder_cfg.setdefault("num_layers", 2)
        encoder_cfg.setdefault("use_scale2", True)
        encoder_cfg.setdefault("scale2_alpha", 1.0)
        encoder_cfg.setdefault("use_gate_norm", True)
        encoder_cfg.setdefault("enable_micro_diagnostics", self.enable_micro_diagnostics)
        encoder_cfg.setdefault("save_attention_maps", self.save_attention_maps)
        encoder_cfg.setdefault("save_node_similarity", self.save_node_similarity)
        encoder_cfg.setdefault("diagnostic_max_samples", self.diagnostic_max_samples)
        self.encoder = D12PixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            height=self.height,
            width=self.width,
            **encoder_cfg,
        )
        self.slot_attention = IterativeSlotAttentionD12(
            hidden_dim=self.hidden_dim,
            num_slots=self.num_slots,
            num_iterations=int(slot_iterations),
            dropout=dropout,
            residual_slot_connection=bool(residual_slot_connection),
        )
        self.motif_relation = MotifRelationTransformerD12(
            hidden_dim=self.hidden_dim,
            num_heads=int(motif_relation_heads),
            num_layers=int(motif_relation_layers),
            dropout=dropout,
        )
        self.virtual_node_gather = VirtualNodeGatherD12(
            hidden_dim=self.hidden_dim,
            global_dim=self.global_dim,
            dropout=dropout,
        )
        self.film_fusion = FiLMFusionD12(
            hidden_dim=self.hidden_dim,
            global_dim=self.global_dim,
            num_slots=self.num_slots,
            global_dropout=global_dropout,
        )
        self.class_head = ClassMotifAttentionHeadD12(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )
        self.local_head = ClassMotifAttentionHeadD12(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )
        self.supcon_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, int(supcon_projection_dim)),
        )
        self.register_buffer("border_mask", self._make_border_mask(border_width=3), persistent=False)
        self.register_buffer("pixel_positions", self._make_pixel_positions(), persistent=False)

    def set_micro_diagnostics(
        self,
        *,
        enabled: bool = True,
        save_attention_maps: Optional[bool] = None,
        save_node_similarity: Optional[bool] = None,
        diagnostic_max_samples: Optional[int] = None,
    ) -> None:
        self.enable_micro_diagnostics = bool(enabled)
        if save_attention_maps is not None:
            self.save_attention_maps = bool(save_attention_maps)
        if save_node_similarity is not None:
            self.save_node_similarity = bool(save_node_similarity)
        if diagnostic_max_samples is not None:
            self.diagnostic_max_samples = max(int(diagnostic_max_samples), 1)
        self.encoder.set_micro_diagnostics(
            enabled=self.enable_micro_diagnostics,
            save_attention_maps=self.save_attention_maps,
            save_node_similarity=self.save_node_similarity,
            diagnostic_max_samples=self.diagnostic_max_samples,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "D12GlobalLocalMotifModel":
        cfg = dict(config)
        cfg.pop("name", None)
        return cls(**cfg)

    def _make_border_mask(self, border_width: int = 3) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)

    def _make_pixel_positions(self) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, self.height)
        xs = torch.linspace(0.0, 1.0, self.width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _parse_inputs(
        self,
        batch_or_x,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            node_mask = batch.get("node_mask")
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("D12GlobalLocalMotifModel needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("D12GlobalLocalMotifModel requires edge_index and edge_attr")
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")
        if x.shape[-1] != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {x.shape[-1]}")
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E] or [B, 2, E], got {tuple(edge_index.shape)}")
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(x.shape[0], -1, -1)
        if edge_attr.ndim != 3:
            raise ValueError(f"edge_attr must be [B, E, F] or [E, F], got {tuple(edge_attr.shape)}")
        if edge_attr.shape[0] != x.shape[0]:
            raise ValueError(f"edge_attr batch {edge_attr.shape[0]} != x batch {x.shape[0]}")
        if edge_attr.shape[-1] != self.edge_dim:
            raise ValueError(f"Expected edge_dim={self.edge_dim}, got {edge_attr.shape[-1]}")
        if node_mask is None:
            node_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return x, edge_index, edge_attr, node_mask

    def _coords_from_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= 3:
            return x[:, :, 1:3]
        return self.pixel_positions.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)

    def forward(
        self,
        batch_or_x,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr, node_mask = self._parse_inputs(
            batch_or_x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask
        )
        h_pixel, encoder_diag = self.encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask,
        )
        coords = self._coords_from_x(x)
        slots_raw, slot_attn_maps, slot_centers = self.slot_attention(
            h_pixel=h_pixel,
            node_mask=node_mask,
            coords=coords,
        )
        slots_context = self.motif_relation(slots_raw)

        if self.use_global_branch:
            global_context, virtual_attention = self.virtual_node_gather(h_pixel, node_mask=node_mask)
            slots_refined, gamma, beta = self.film_fusion(slots_context, global_context)
        else:
            global_context = h_pixel.new_zeros((h_pixel.shape[0], self.global_dim))
            virtual_attention = h_pixel.new_zeros(h_pixel.shape[:2])
            gamma = h_pixel.new_zeros((h_pixel.shape[0], self.num_slots, self.hidden_dim))
            beta = h_pixel.new_zeros((h_pixel.shape[0], self.num_slots, self.hidden_dim))
            slots_refined = slots_context

        logits, class_motif_attn, class_repr = self.class_head(slots_refined)
        logits_local, class_motif_attn_local, class_repr_local = self.local_head(slots_context)
        motif_supcon = self.supcon_proj(slots_context.mean(dim=1))

        border_mask = self.border_mask.to(device=slot_attn_maps.device, dtype=slot_attn_maps.dtype)
        border_mass = (slot_attn_maps * border_mask.view(1, 1, -1)).sum(dim=2)
        slot_mass = slot_attn_maps.sum(dim=2).clamp_min(1e-8)
        border_mass_per_slot = border_mass / slot_mass
        slot_area = slot_attn_maps.mean(dim=2)
        slot_area_norm = slot_area / slot_area.sum(dim=1, keepdim=True).clamp_min(1e-8)
        slot_area_entropy = -(
            slot_area_norm * slot_area_norm.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        effective_slots = slot_area_entropy.detach().exp()
        slot_attention_peak = slot_attn_maps.detach().amax(dim=2).mean()
        slot_attention_entropy_per_slot = -(
            slot_attn_maps.detach().float()
            * slot_attn_maps.detach().float().clamp_min(1e-8).log()
        ).sum(dim=2).mean()
        class_avg = class_motif_attn.detach().float().mean(dim=0)
        class_part_similarity_disgust_angry = F.cosine_similarity(
            class_avg[1].unsqueeze(0),
            class_avg[0].unsqueeze(0),
            dim=1,
        ).squeeze(0)

        diagnostics: Dict[str, torch.Tensor] = {
            "encoder_gate_mean": encoder_diag["encoder_gate_mean"],
            "encoder_gate_std": encoder_diag["encoder_gate_std"],
            "encoder_gate_min": encoder_diag["encoder_gate_min"],
            "encoder_gate_max": encoder_diag["encoder_gate_max"],
            "scale2_edge_count": encoder_diag["scale2_edge_count"],
            "h_pixel_mean": h_pixel.detach().mean(),
            "h_pixel_std": h_pixel.detach().std(unbiased=False),
            "slot_area_entropy": slot_area_entropy.detach(),
            "logits_mean": logits.detach().mean(),
            "logits_std": logits.detach().std(unbiased=False),
            "slot_area_mean": slot_area.detach().mean(),
            "slot_area_std": slot_area.detach().std(unbiased=False),
            "slot_area_min": slot_area.detach().amin(),
            "slot_area_max": slot_area.detach().amax(),
            "slot_attention_peak": slot_attention_peak,
            "slot_attention_entropy_per_slot": slot_attention_entropy_per_slot,
            "effective_slots": effective_slots,
            "class_part_similarity_disgust_angry": class_part_similarity_disgust_angry,
            "border_mass_per_slot_mean": border_mass_per_slot.detach().mean(),
        }
        for key, value in encoder_diag.items():
            if key not in diagnostics and torch.is_tensor(value) and value.numel() == 1:
                diagnostics[key] = value.detach()

        out = {
            "logits": logits,
            "logits_local": logits_local,
            "motif_embeddings": slots_context,
            "local_raw": slots_raw,
            "local_context": slots_context,
            "local_refined": slots_refined,
            "part_masks": slot_attn_maps,
            "slot_attention": slot_attn_maps,
            "slot_centers": slot_centers,
            "part_centers": slot_centers,
            "center_of_mass": slot_centers,
            "class_motif_attn": class_motif_attn,
            "class_part_attn": class_motif_attn,
            "class_motif_attn_local": class_motif_attn_local,
            "global_context": global_context,
            "virtual_attention": virtual_attention,
            "virtual_attn": virtual_attention,
            "film_gamma": gamma,
            "film_beta": beta,
            "gamma": gamma,
            "beta": beta,
            "motif_supcon": motif_supcon,
            "supcon_features": motif_supcon,
            "local_raw_proj": motif_supcon,
            "class_repr": class_repr,
            "class_repr_local": class_repr_local,
            "h_pixel": h_pixel,
            "pixel_embeddings": h_pixel,
            "slot_area": slot_area,
            "border_mass_per_slot": border_mass_per_slot,
            "encoder_gate_mean": diagnostics["encoder_gate_mean"],
            "encoder_gate_std": diagnostics["encoder_gate_std"],
            "encoder_gate_min": diagnostics["encoder_gate_min"],
            "encoder_gate_max": diagnostics["encoder_gate_max"],
            "h_pixel_mean": diagnostics["h_pixel_mean"],
            "h_pixel_std": diagnostics["h_pixel_std"],
            "slot_area_entropy": diagnostics["slot_area_entropy"],
            "logits_mean": diagnostics["logits_mean"],
            "logits_std": diagnostics["logits_std"],
            "scale2_edge_count": diagnostics["scale2_edge_count"],
            "diagnostics": diagnostics,
        }
        for key, value in diagnostics.items():
            if key not in out and torch.is_tensor(value) and value.numel() == 1:
                out[key] = value
        return out
