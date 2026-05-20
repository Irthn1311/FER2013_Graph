"""D13 pixel encoders for pure GNN hierarchical reduction.

D13A intentionally keeps these encoders small: they only produce pixel-level
embeddings for a downstream learnable coarsening module and make no motif claim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class EdgeAwareMessageLayer(nn.Module):
    """Edge-aware message passing on a flat PyG-style graph."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.eps = float(eps)
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(self.edge_dim),
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must be [E, edge_dim], got {tuple(edge_attr.shape)}")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_attr rows {edge_attr.shape[0]} != edge_index edges {edge_index.shape[1]}"
            )
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            return torch.zeros_like(h), {"edge_gate_mean": h.new_tensor(0.0)}
        if int(src.max()) >= h.shape[0] or int(dst.max()) >= h.shape[0]:
            raise ValueError("edge_index contains node ids outside h")

        h_src = h.index_select(0, src)
        h_dst = h.index_select(0, dst)
        e = self.edge_mlp(edge_attr.to(dtype=h.dtype))
        gate = torch.sigmoid(self.gate_mlp(torch.cat([h_src, h_dst, e], dim=-1)))
        msg = self.message_mlp(torch.cat([h_src, e], dim=-1)) * gate

        out = msg.new_zeros(h.shape)
        out.index_add_(0, dst, msg)
        denom = gate.new_zeros(h.shape)
        denom.index_add_(0, dst, gate)
        out = out / denom.clamp_min(self.eps)
        diag = {
            "edge_gate_mean": gate.detach().mean(),
            "edge_gate_std": gate.detach().std(unbiased=False),
        }
        return out, diag


class EdgeAwarePixelBlock(nn.Module):
    """Residual edge-aware message block with norm, FFN, and dropout."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.message = EdgeAwareMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.norm_msg = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        msg, diag = self.message(h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.norm_msg(h + self.dropout(msg))
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))
        return h, diag


class GatedJumpingKnowledge(nn.Module):
    """Learn a soft layer mixture for stable shallow encoders."""

    def __init__(self, hidden_dim: int, num_states: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.num_states = int(num_states)

    def forward(self, states: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(states, dim=1)
        logits = self.score(stacked).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class EdgeAwarePixelEncoderLiteV2(nn.Module):
    """Small edge-aware pixel encoder used by D13A."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        jumping_knowledge: str = "gated",
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.jumping_knowledge = str(jumping_knowledge or "gated").lower()
        if self.jumping_knowledge not in {"none", "concat", "gated"}:
            raise ValueError("jumping_knowledge must be one of: none, concat, gated")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.layers = nn.ModuleList(
            EdgeAwarePixelBlock(self.hidden_dim, self.edge_dim, dropout=dropout)
            for _ in range(self.num_layers)
        )
        if self.jumping_knowledge == "concat":
            self.jk_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * (self.num_layers + 1), self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
        elif self.jumping_knowledge == "gated":
            self.jk_gate = GatedJumpingKnowledge(self.hidden_dim, self.num_layers + 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_layer_embeddings: bool = False,
    ) -> Dict[str, Any]:
        h = self.input_proj(x)
        states = [h]
        diagnostics: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            h, diag = layer(h, edge_index=edge_index, edge_attr=edge_attr)
            states.append(h)
            for key, value in diag.items():
                diagnostics[f"layer{idx}_{key}"] = value
        if self.jumping_knowledge == "concat":
            h = self.jk_proj(torch.cat(states, dim=-1))
        elif self.jumping_knowledge == "gated":
            h = self.jk_gate(states)
        out: Dict[str, Any] = {"h": h, "diagnostics": diagnostics}
        if return_layer_embeddings:
            out["layer_embeddings"] = states
        return out


class GINEPixelBlock(nn.Module):
    """GINE-like residual block without requiring torch_geometric internals."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(()))
        self.norm_msg = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg_input = h.index_select(0, src) + self.edge_proj(edge_attr.to(dtype=h.dtype))
        msg = self.message_mlp(F.relu(msg_input))
        agg = msg.new_zeros(h.shape)
        agg.index_add_(0, dst, msg)
        h = self.norm_msg((1.0 + self.eps) * h + self.dropout(agg))
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))
        return h


class GINEPixelEncoder(nn.Module):
    """Small GINE-style control encoder for D13A."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        jumping_knowledge: str = "gated",
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.jumping_knowledge = str(jumping_knowledge or "gated").lower()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(int(node_dim)),
            nn.Linear(int(node_dim), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.layers = nn.ModuleList(
            GINEPixelBlock(self.hidden_dim, int(edge_dim), dropout=dropout)
            for _ in range(self.num_layers)
        )
        if self.jumping_knowledge == "concat":
            self.jk_proj = nn.Linear(self.hidden_dim * (self.num_layers + 1), self.hidden_dim)
        elif self.jumping_knowledge == "gated":
            self.jk_gate = GatedJumpingKnowledge(self.hidden_dim, self.num_layers + 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_layer_embeddings: bool = False,
    ) -> Dict[str, Any]:
        h = self.input_proj(x)
        states = [h]
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
            states.append(h)
        if self.jumping_knowledge == "concat":
            h = self.jk_proj(torch.cat(states, dim=-1))
        elif self.jumping_knowledge == "gated":
            h = self.jk_gate(states)
        out: Dict[str, Any] = {"h": h, "diagnostics": {}}
        if return_layer_embeddings:
            out["layer_embeddings"] = states
        return out
