"""Clean edge-aware GNN encoder for D18."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class EdgeContextGNNLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        edge_attr_dim: int = 6,
        edge_hidden_dim: int = 32,
        dropout: float = 0.2,
        residual: bool = True,
        layer_norm: bool = True,
        aggregation: str = "mean",
        scalar_edge_gate: bool = False,
        structure_gate_cap: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.aggregation = str(aggregation)
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported D18 aggregation={aggregation!r}")
        self.residual = bool(residual)
        self.scalar_edge_gate_enabled = bool(scalar_edge_gate)
        self.structure_gate_cap = structure_gate_cap if structure_gate_cap is None else float(structure_gate_cap)
        self.last_raw_structure_gate_mean: torch.Tensor | None = None
        self.last_effective_structure_gate_mean: torch.Tensor | None = None
        self.last_gate_stats: Dict[str, torch.Tensor] = {}
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.edge_attr_dim, int(edge_hidden_dim)),
            torch.nn.LayerNorm(int(edge_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
        )
        self.gate = torch.nn.Linear(int(edge_hidden_dim), self.hidden_dim)
        self.scalar_gate = torch.nn.Linear(int(edge_hidden_dim), 1) if self.scalar_edge_gate_enabled else None
        self.message = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + int(edge_hidden_dim), self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.norm_msg = torch.nn.LayerNorm(self.hidden_dim) if layer_norm else torch.nn.Identity()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.Dropout(float(dropout)),
        )
        self.norm_ffn = torch.nn.LayerNorm(self.hidden_dim) if layer_norm else torch.nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dst_degree: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        edge_emb = self.edge_mlp(edge_attr.to(device=h.device, dtype=h.dtype))
        gate = torch.sigmoid(self.gate(edge_emb))
        msg = self.message(torch.cat([h[src], edge_emb], dim=1)) * gate
        if self.scalar_gate is not None:
            raw_scalar_gate = torch.sigmoid(self.scalar_gate(edge_emb))
            effective_scalar_gate = raw_scalar_gate
            if self.structure_gate_cap is not None and edge_type is not None:
                structure_mask = (edge_type.to(device=h.device).long() == 2).view(-1, 1)
                capped = raw_scalar_gate * float(self.structure_gate_cap)
                effective_scalar_gate = torch.where(structure_mask, capped, raw_scalar_gate)
                if bool(structure_mask.any()):
                    self.last_raw_structure_gate_mean = raw_scalar_gate[structure_mask].mean()
                    self.last_effective_structure_gate_mean = effective_scalar_gate[structure_mask].mean()
                else:
                    self.last_raw_structure_gate_mean = raw_scalar_gate.new_tensor(0.0)
                    self.last_effective_structure_gate_mean = raw_scalar_gate.new_tensor(0.0)
            else:
                self.last_raw_structure_gate_mean = None
                self.last_effective_structure_gate_mean = None
            self.last_gate_stats = {}
            if edge_type is not None:
                et = edge_type.to(device=h.device).long().view(-1, 1)
                for name, value in (("local", 0), ("knn", 1), ("structure", 2)):
                    mask = et == int(value)
                    if bool(mask.any()):
                        self.last_gate_stats[f"raw_gate_mean_{name}"] = raw_scalar_gate[mask].mean()
                        self.last_gate_stats[f"effective_gate_mean_{name}"] = effective_scalar_gate[mask].mean()
            msg = msg * effective_scalar_gate
        agg = msg.new_zeros(h.shape)
        agg.index_add_(0, dst, msg)
        if self.aggregation == "mean":
            if dst_degree is None:
                dst_degree = h.new_zeros((h.size(0), 1))
                dst_degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            agg = agg / dst_degree.clamp_min(1.0)
        h_msg = self.norm_msg(h + agg if self.residual else agg)
        return self.norm_ffn(h_msg + self.ffn(h_msg) if self.residual else self.ffn(h_msg))


class EdgeContextGNNEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        num_layers: int = 3,
        edge_attr_dim: int = 6,
        edge_hidden_dim: int = 32,
        dropout: float = 0.25,
        residual: bool = True,
        layer_norm: bool = True,
        aggregation: str = "mean",
        scalar_edge_gate: bool = False,
        structure_gate_cap: float | None = None,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                EdgeContextGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    dropout=dropout,
                    residual=residual,
                    layer_norm=layer_norm,
                    aggregation=aggregation,
                    scalar_edge_gate=scalar_edge_gate,
                    structure_gate_cap=structure_gate_cap,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.aggregation = str(aggregation)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None) -> "EdgeContextGNNEncoder":
        cfg = dict(cfg or {})
        return cls(
            hidden_dim=int(cfg.get("hidden_dim", 96)),
            num_layers=int(cfg.get("num_layers", 3)),
            edge_attr_dim=int(cfg.get("edge_attr_dim", 6)),
            edge_hidden_dim=int(cfg.get("edge_hidden_dim", 32)),
            dropout=float(cfg.get("dropout", 0.25)),
            residual=bool(cfg.get("residual", True)),
            layer_norm=bool(cfg.get("layer_norm", True)),
            aggregation=str(cfg.get("aggregation", "mean")),
            scalar_edge_gate=bool((cfg.get("scalar_edge_gate") or {}).get("enabled", cfg.get("scalar_edge_gate_enabled", False))),
            structure_gate_cap=(cfg.get("edge_gate") or cfg.get("scalar_edge_gate") or {}).get("structure_gate_cap"),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("D18 EdgeContextGNN requires edge_attr")
        dst_degree = None
        if self.layers and self.layers[0].aggregation == "mean":
            dst = edge_index[1].long()
            dst_degree = h.new_zeros((h.size(0), 1))
            dst_degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, dst_degree=dst_degree, edge_type=edge_type)
        return h

    def structure_gate_penalty(self) -> torch.Tensor | None:
        values = [layer.last_raw_structure_gate_mean for layer in self.layers if layer.last_raw_structure_gate_mean is not None]
        if not values:
            return None
        return torch.stack(values).mean()

    def gate_stats(self) -> Dict[str, torch.Tensor]:
        grouped: Dict[str, List[torch.Tensor]] = {}
        for layer in self.layers:
            for key, value in layer.last_gate_stats.items():
                grouped.setdefault(key, []).append(value)
        return {key: torch.stack(values).mean() for key, values in grouped.items() if values}

