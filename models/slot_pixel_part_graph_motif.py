"""D6A self-discovered hierarchical pixel-part graph motif model."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class EdgeAwarePixelMessageLayer(nn.Module):
    """Lightweight edge-gated message passing over the fixed pixel graph."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.norm_msg = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        h_src = h.index_select(dim=1, index=src)
        gate = torch.sigmoid(self.edge_gate(edge_attr.float()))
        msg = self.msg_mlp(h_src) * gate

        agg = msg.new_zeros(h.shape)
        agg.index_add_(dim=1, index=dst, source=msg)

        deg = msg.new_zeros(h.shape[1])
        deg.index_add_(dim=0, index=dst, source=torch.ones_like(dst, dtype=msg.dtype))
        agg = agg / deg.clamp_min(1.0).view(1, -1, 1)

        if node_mask is not None:
            mask = node_mask.to(dtype=h.dtype).unsqueeze(-1)
            agg = agg * mask

        h = self.norm_msg(h + self.agg_mlp(agg))
        h = self.norm_ffn(h + self.ffn(h))
        if node_mask is not None:
            h = h * node_mask.to(dtype=h.dtype).unsqueeze(-1)
        return h


class PartSelfAttentionLayer(nn.Module):
    """Transformer-style part relation layer that exposes attention weights."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, part_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(
            part_feats,
            part_feats,
            part_feats,
            need_weights=True,
            average_attn_weights=False,
        )
        h = self.norm_attn(part_feats + attn_out)
        h = self.norm_ffn(h + self.ffn(h))
        return h, attn_weights


class SlotPixelPartGraphMotif(nn.Module):
    """Pixel graph -> soft part slots -> part relation attention -> classifier."""

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        pixel_gnn_layers: int = 1,
        num_part_slots: int = 16,
        part_layers: int = 1,
        part_heads: int = 4,
        dropout: float = 0.2,
        use_part_position: bool = True,
        assignment_temperature: float = 1.0,
        return_attention: bool = True,
        use_class_part_attention: bool = False,
        height: int = 48,
        width: int = 48,
        connectivity: int = 8,
    ) -> None:
        super().__init__()
        del connectivity
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_part_slots = int(num_part_slots)
        self.height = int(height)
        self.width = int(width)
        self.use_part_position = bool(use_part_position)
        self.assignment_temperature = float(assignment_temperature)
        self.return_attention = bool(return_attention)
        self.use_class_part_attention = bool(use_class_part_attention)

        if self.num_nodes != self.height * self.width:
            raise ValueError(
                f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}"
            )

        self.input_proj = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pixel_layers = nn.ModuleList(
            [
                EdgeAwarePixelMessageLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    dropout=dropout,
                )
                for _ in range(int(pixel_gnn_layers))
            ]
        )

        self.part_queries = nn.Parameter(torch.empty(self.num_part_slots, self.hidden_dim))
        self.pixel_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pixel_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.part_layers = nn.ModuleList(
            [
                PartSelfAttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(part_heads),
                    dropout=dropout,
                )
                for _ in range(int(part_layers))
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, self.num_classes),
        )
        if self.use_class_part_attention:
            self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
            self.class_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.class_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.class_logit_head = nn.Linear(self.hidden_dim, 1)
        self.register_buffer("pixel_positions", self._make_positions(), persistent=False)
        self.register_buffer("border_mask", self._make_border_mask(border_width=3), persistent=False)
        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlotPixelPartGraphMotif":
        cfg = dict(config)
        for legacy_key in (
            "edge_hidden_dim",
            "gnn_layers",
            "use_edge_gnn",
            "temperature",
            "edge_score_weight",
            "num_edges",
            "motif_prior_path",
            "init_node_gate_from_prior",
            "prior_init_clamp_min",
            "prior_init_clamp_max",
        ):
            cfg.pop(legacy_key, None)
        return cls(**cfg)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.part_queries, mean=0.0, std=0.02)
        if self.use_class_part_attention:
            nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(
        self,
        batch_or_x,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del y
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            node_mask = batch.get("node_mask")
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("SlotPixelPartGraphMotif needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("SlotPixelPartGraphMotif requires edge_index and edge_attr")
        # Handle DataParallel: edge_index may be (B, 2, E) after scatter
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        x = x.float()
        edge_attr = edge_attr.float()
        h_pixel = self.input_proj(x)
        if node_mask is not None:
            h_pixel = h_pixel * node_mask.to(dtype=h_pixel.dtype).unsqueeze(-1)
        for layer in self.pixel_layers:
            h_pixel = layer(h_pixel, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)

        part_masks, pool_weights = self._assign_parts(h_pixel, node_mask=node_mask)
        part_feats = torch.bmm(pool_weights, self.pixel_value(h_pixel))
        part_centers = torch.bmm(pool_weights, self.pixel_positions.to(h_pixel).unsqueeze(0).expand(x.shape[0], -1, -1))
        if self.use_part_position:
            part_feats = part_feats + self.position_mlp(part_centers)

        part_context = part_feats
        part_attn = None
        for layer in self.part_layers:
            part_context, part_attn = layer(part_context)

        class_part_attn = None
        class_repr = None
        if self.use_class_part_attention:
            class_part_attn, class_repr, logits = self._class_part_attention(part_context)
        else:
            image_feat = part_context.mean(dim=1)
            logits = self.classifier(image_feat)
        slot_area = part_masks.mean(dim=2)
        border_mass = (part_masks * self.border_mask.to(part_masks).view(1, 1, -1)).mean(dim=2)
        border_mass_per_slot = self._slot_border_ratio(part_masks)

        out = {
            "logits": logits,
            "pixel_embeddings": h_pixel,
            "part_masks": part_masks,
            "part_features": part_feats,
            "part_context": part_context,
            "part_attn": part_attn if self.return_attention else None,
            "class_part_attn": class_part_attn,
            "class_repr": class_repr,
            "slot_area": slot_area,
            "border_mass": border_mass,
            "border_mass_per_slot": border_mass_per_slot,
            "part_centers": part_centers,
            "pool_weights": pool_weights,
            "diagnostics": self._diagnostics(
                part_masks=part_masks,
                slot_area=slot_area,
                border_mass=border_mass,
                border_mass_per_slot=border_mass_per_slot,
                part_attn=part_attn,
                class_part_attn=class_part_attn,
            ),
        }
        return out

    def _class_part_attention(
        self,
        part_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_part = self.class_key(part_context)
        v_part = self.class_value(part_context)
        class_logits = torch.einsum("ch,bkh->bck", self.class_queries, k_part)
        class_logits = class_logits / math.sqrt(float(self.hidden_dim))
        class_part_attn = torch.softmax(class_logits, dim=2)
        class_repr = torch.einsum("bck,bkh->bch", class_part_attn, v_part)
        logits = self.class_logit_head(class_repr).squeeze(-1)
        return class_part_attn, class_repr, logits

    def _assign_parts(
        self,
        h_pixel: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.part_queries, dim=-1, eps=1e-6)
        k = self.pixel_key(h_pixel)
        logits = torch.einsum("kh,bnh->bkn", q, k)
        logits = logits / max(self.assignment_temperature, 1e-6) / math.sqrt(float(self.hidden_dim))
        if node_mask is not None:
            valid = node_mask.bool().unsqueeze(1)
            logits = logits.masked_fill(~valid, -1e4)
        part_masks = torch.softmax(logits, dim=1)
        if node_mask is not None:
            part_masks = part_masks * node_mask.to(dtype=part_masks.dtype).unsqueeze(1)
        pool_weights = part_masks / part_masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return part_masks, pool_weights

    def _make_positions(self) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, self.height)
        xs = torch.linspace(0.0, 1.0, self.width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()

    def _make_border_mask(self, border_width: int) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)

    def _slot_border_ratio(self, part_masks: torch.Tensor) -> torch.Tensor:
        border_mask = self.border_mask.to(device=part_masks.device, dtype=part_masks.dtype)
        border_mass = (part_masks * border_mask.view(1, 1, -1)).sum(dim=2)
        slot_mass = part_masks.sum(dim=2).clamp_min(1e-6)
        return border_mass / slot_mass

    @staticmethod
    def _diagnostics(
        part_masks: torch.Tensor,
        slot_area: torch.Tensor,
        border_mass: torch.Tensor,
        border_mass_per_slot: torch.Tensor,
        part_attn: Optional[torch.Tensor],
        class_part_attn: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        m = F.normalize(part_masks.float(), dim=2, eps=1e-6)
        sim = torch.bmm(m, m.transpose(1, 2))
        k = sim.shape[1]
        off = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0))
        area_norm = slot_area.float() / slot_area.float().sum(dim=1, keepdim=True).clamp_min(1e-6)
        area_entropy = -(area_norm * area_norm.clamp_min(1e-6).log()).sum(dim=1)
        diagnostics = {
            "slot_div": off.mean().detach(),
            "slot_similarity_mean": off.mean().detach(),
            "border_mass": border_mass.mean().detach(),
            "border_mass_mean": border_mass_per_slot.mean().detach(),
            "slot_area_mean": slot_area.mean().detach(),
            "slot_area_min": slot_area.amin().detach(),
            "slot_area_max": slot_area.amax().detach(),
            "slot_area_entropy": area_entropy.mean().detach(),
        }
        if part_attn is not None:
            diagnostics["part_attn_std"] = part_attn.detach().float().std(unbiased=False)
        if class_part_attn is not None:
            class_entropy = -(
                class_part_attn.float() * class_part_attn.float().clamp_min(1e-6).log()
            ).sum(dim=2)
            diagnostics["class_part_entropy"] = class_entropy.mean().detach()
        return diagnostics
