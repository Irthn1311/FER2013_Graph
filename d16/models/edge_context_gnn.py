"""Edge-aware relation GNN for D16 A5b.

This module is deliberately small and local to D16. It borrows the useful
message-passing lesson from earlier pixel-graph models without importing the
D10 slot-attention or GraphSwin stack.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch


_GROUP_INDICES = {
    "mouth": [5, 6, 7],
    "eye": [0, 1],
    "brow": [2, 3],
    "nose_cheek": [4, 8, 9, 10],
}


def _valid_num_heads(hidden_dim: int, requested: int) -> int:
    heads = max(int(requested), 1)
    while heads > 1 and hidden_dim % heads != 0:
        heads -= 1
    return heads


class EdgeContextGNNLayer(torch.nn.Module):
    """One residual edge-gated message-passing layer."""

    def __init__(
        self,
        hidden_dim: int = 96,
        edge_attr_dim: int = 8,
        edge_hidden_dim: int = 32,
        dropout: float = 0.2,
        residual: bool = True,
        layer_norm: bool = True,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.residual = bool(residual)
        self.aggregation = str(aggregation)
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported edge_context_gnn aggregation={aggregation!r}")
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.edge_attr_dim, self.edge_hidden_dim),
            torch.nn.LayerNorm(self.edge_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
        )
        self.gate = torch.nn.Linear(self.edge_hidden_dim, self.hidden_dim)
        self.message = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + self.edge_hidden_dim, self.hidden_dim),
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
        src: torch.Tensor | None = None,
        dst: torch.Tensor | None = None,
        dst_degree: torch.Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if edge_attr is None:
            raise ValueError("edge_context_gnn requires edge_attr, but batch.edge_attr_cat is None")
        if edge_attr.dim() != 2 or edge_attr.size(1) != self.edge_attr_dim:
            raise ValueError(f"edge_attr must be [E,{self.edge_attr_dim}], got {tuple(edge_attr.shape)}")
        if src is None or dst is None:
            src, dst = edge_index[0].long(), edge_index[1].long()
        edge_emb = self.edge_mlp(edge_attr.to(device=h.device, dtype=h.dtype))
        gate = torch.sigmoid(self.gate(edge_emb))
        msg = self.message(torch.cat([h[src], edge_emb], dim=1)) * gate
        agg = msg.new_zeros(h.shape)
        agg.index_add_(0, dst, msg)
        if self.aggregation == "mean":
            if dst_degree is None:
                dst_degree = h.new_zeros((h.size(0), 1))
                dst_degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            agg = agg / dst_degree.clamp_min(1.0)
        h_msg = self.norm_msg(h + agg if self.residual else agg)
        h_out = self.norm_ffn(h_msg + self.ffn(h_msg) if self.residual else self.ffn(h_msg))
        if not collect_diagnostics:
            return h_out, {}
        diagnostics = {
            "edge_gate_mean": gate.detach().mean(),
            "edge_gate_std": gate.detach().std(unbiased=False),
            "edge_gate_min": gate.detach().min(),
            "edge_gate_max": gate.detach().max(),
            "message_norm_mean": msg.detach().norm(dim=1).mean(),
            "node_embedding_norm_mean": h_out.detach().norm(dim=1).mean(),
            "node_embedding_std_mean": h_out.detach().std(dim=0, unbiased=False).mean(),
        }
        return h_out, diagnostics


class PartGlobalContextBlock(torch.nn.Module):
    """Pool part/global context tokens, mix them, and broadcast back to nodes."""

    def __init__(
        self,
        hidden_dim: int = 96,
        part_groups: Iterable[str] | None = None,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        context_scale_init: float = 0.5,
        dropout: float = 0.2,
        prior_gate: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.part_groups = list(part_groups or ["mouth", "eye", "brow", "nose_cheek", "global"])
        gate_cfg = dict(prior_gate or {})
        self.context_prior_gate_enabled = bool(gate_cfg.get("enabled", False))
        self.context_prior_gate_learnable = bool(gate_cfg.get("learnable", True))
        heads = _valid_num_heads(self.hidden_dim, int(transformer_heads))
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.context_mixer = torch.nn.TransformerEncoder(enc_layer, num_layers=int(transformer_layers))
        self.context_update = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Dropout(float(dropout)),
        )
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.context_scale = torch.nn.Parameter(torch.tensor(float(context_scale_init), dtype=torch.float32))
        prior_gate_init = self._prior_gate_init_values(gate_cfg.get("init", 1.0))
        if self.context_prior_gate_enabled and self.context_prior_gate_learnable:
            self.context_prior_gate_logit = torch.nn.Parameter(self._logit(prior_gate_init))
            self.register_buffer("context_prior_gate_value", torch.ones(len(self.part_groups), dtype=torch.float32), persistent=False)
        else:
            self.context_prior_gate_logit = None
            gate_value = prior_gate_init if self.context_prior_gate_enabled else torch.ones(len(self.part_groups), dtype=torch.float32)
            self.register_buffer("context_prior_gate_value", gate_value, persistent=False)

    def _prior(self, part_soft: torch.Tensor, group: str) -> torch.Tensor:
        if group == "global":
            return torch.ones((part_soft.size(0),), device=part_soft.device, dtype=part_soft.dtype)
        indices = [idx for idx in _GROUP_INDICES.get(group, []) if idx < part_soft.size(1)]
        if not indices:
            return torch.zeros((part_soft.size(0),), device=part_soft.device, dtype=part_soft.dtype)
        return part_soft[:, indices].max(dim=1).values

    @staticmethod
    def _logit(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(values / (1.0 - values))

    def _prior_gate_init_values(self, init: Any) -> torch.Tensor:
        if isinstance(init, dict):
            values = [float(init.get(name, 1.0)) for name in self.part_groups]
        else:
            values = [float(init)] * len(self.part_groups)
        return torch.tensor(values, dtype=torch.float32).clamp(0.0, 1.0)

    def _prior_gate_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.context_prior_gate_enabled:
            return torch.ones(len(self.part_groups), device=device, dtype=dtype)
        if self.context_prior_gate_logit is not None:
            return torch.sigmoid(self.context_prior_gate_logit).to(device=device, dtype=dtype)
        return self.context_prior_gate_value.to(device=device, dtype=dtype)

    def _effective_priors(self, part_soft: torch.Tensor) -> torch.Tensor:
        raw = torch.stack([self._prior(part_soft, group).clamp_min(0.0) for group in self.part_groups], dim=1)
        if not self.context_prior_gate_enabled:
            return raw
        gates = self._prior_gate_values(part_soft.device, part_soft.dtype).view(1, -1)
        uniform = torch.ones_like(raw)
        return gates * raw + (1.0 - gates) * uniform

    def _forward_loop_reference(
        self,
        h: torch.Tensor,
        part_soft: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        out = h.clone()
        scale = self.context_scale.to(device=h.device, dtype=h.dtype)
        for graph_id in range(int(num_graphs)):
            mask = batch_index == graph_id
            if not bool(mask.any()):
                continue
            h_g = h[mask]
            part_g = part_soft[mask]
            priors = [self._effective_priors(part_g)[:, idx] for idx in range(len(self.part_groups))]
            tokens = []
            for prior in priors:
                denom = prior.sum().clamp_min(1e-6)
                tokens.append((h_g * prior.unsqueeze(1)).sum(dim=0) / denom)
            token_stack = torch.stack(tokens, dim=0).unsqueeze(0)
            mixed = self.context_mixer(token_stack).squeeze(0)
            non_global = [i for i, name in enumerate(self.part_groups) if name != "global"]
            if non_global:
                weights = torch.stack([priors[i] for i in non_global], dim=1)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                local_context = weights @ mixed[non_global]
            else:
                local_context = h_g.new_zeros(h_g.shape)
            if "global" in self.part_groups:
                global_context = mixed[self.part_groups.index("global")].unsqueeze(0).expand_as(local_context)
                context = 0.5 * (local_context + global_context)
            else:
                context = local_context
            update = self.context_update(torch.cat([h_g, context], dim=1))
            out[mask] = self.norm(h_g + scale * update)
        return out

    def forward(
        self,
        h: torch.Tensor,
        part_soft: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_graphs = int(num_graphs)
        scale = self.context_scale.to(device=h.device, dtype=h.dtype)
        priors = self._effective_priors(part_soft)
        token_sums = h.new_zeros((num_graphs, len(self.part_groups), self.hidden_dim))
        token_sums.index_add_(0, batch_index.long(), h.unsqueeze(1) * priors.unsqueeze(2))
        denom = h.new_zeros((num_graphs, len(self.part_groups)))
        denom.index_add_(0, batch_index.long(), priors)
        tokens = token_sums / denom.clamp_min(1e-6).unsqueeze(2)
        mixed = self.context_mixer(tokens)

        non_global = [i for i, name in enumerate(self.part_groups) if name != "global"]
        mixed_nodes = mixed[batch_index.long()]
        if non_global:
            weights = priors[:, non_global]
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            local_context = torch.sum(weights.unsqueeze(2) * mixed_nodes[:, non_global, :], dim=1)
        else:
            local_context = h.new_zeros(h.shape)
        if "global" in self.part_groups:
            global_context = mixed_nodes[:, self.part_groups.index("global"), :]
            context = 0.5 * (local_context + global_context)
        else:
            context = local_context
        update = self.context_update(torch.cat([h, context], dim=1))
        out = self.norm(h + scale * update)
        if not collect_diagnostics:
            return out, {}
        diagnostics = {
            "context_scale": scale.detach(),
            "context_prior_gate_mean": self._prior_gate_values(h.device, h.dtype).detach().mean(),
            "context_update_norm_mean": update.detach().norm(dim=1).mean(),
            "part_context_token_norm_mean": mixed.detach().norm(dim=2).mean(),
        }
        return out, diagnostics


class MultiScaleFusionBlock(torch.nn.Module):
    """Fuse selected EdgeContextGNN layer outputs back to hidden_dim."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_inputs: int = 2,
        dropout: float = 0.2,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_inputs = int(num_inputs)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * self.num_inputs, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim) if layer_norm else torch.nn.Identity(),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim) if layer_norm else torch.nn.Identity(),
        )

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != self.num_inputs:
            raise ValueError(f"MultiScaleFusionBlock expected {self.num_inputs} tensors, got {len(tensors)}")
        return self.proj(torch.cat(tensors, dim=1))


class EdgeContextGNNEncoder(torch.nn.Module):
    """A5b encoder: edge-aware local layers plus part/global context injection."""

    def __init__(
        self,
        hidden_dim: int = 96,
        edge_attr_dim: int = 8,
        num_layers: int = 3,
        edge_hidden_dim: int = 32,
        dropout: float = 0.2,
        residual: bool = True,
        layer_norm: bool = True,
        aggregation: str = "mean",
        layer_output_concat: bool = False,
        multiscale_fusion: Dict[str, Any] | None = None,
        context_injection: Dict[str, Any] | None = None,
        diagnostics: str | bool = "eval",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.num_layers = int(num_layers)
        self.layers = torch.nn.ModuleList(
            [
                EdgeContextGNNLayer(
                    hidden_dim=self.hidden_dim,
                    edge_attr_dim=self.edge_attr_dim,
                    edge_hidden_dim=int(edge_hidden_dim),
                    dropout=float(dropout),
                    residual=bool(residual),
                    layer_norm=bool(layer_norm),
                    aggregation=str(aggregation),
                )
                for _ in range(self.num_layers)
            ]
        )
        fusion_cfg = dict(multiscale_fusion or {})
        self.layer_output_concat = bool(layer_output_concat)
        self.multiscale_enabled = bool(fusion_cfg.get("enabled", self.layer_output_concat))
        self.multiscale_mode = str(fusion_cfg.get("mode", "concat_project"))
        if self.multiscale_mode != "concat_project":
            raise ValueError(f"Unsupported edge_context_gnn multiscale_fusion.mode={self.multiscale_mode!r}")
        raw_layers = fusion_cfg.get("layers") or ([1, self.num_layers] if self.layer_output_concat else [])
        self.multiscale_layers = sorted({int(x) for x in raw_layers if 1 <= int(x) <= self.num_layers})
        if self.multiscale_enabled and len(self.multiscale_layers) < 2:
            self.multiscale_layers = sorted({1, self.num_layers})
        self.multiscale_fusion = None
        if self.multiscale_enabled:
            projection_cfg = dict(fusion_cfg.get("projection", {}) or {})
            projection_type = str(projection_cfg.get("type", "mlp"))
            if projection_type != "mlp":
                raise ValueError(f"Unsupported edge_context_gnn multiscale_fusion.projection.type={projection_type!r}")
            self.multiscale_fusion = MultiScaleFusionBlock(
                hidden_dim=self.hidden_dim,
                num_inputs=len(self.multiscale_layers),
                dropout=float(projection_cfg.get("dropout", dropout)),
                layer_norm=bool(projection_cfg.get("layer_norm", True)),
            )
        context_cfg = dict(context_injection or {})
        self.context_enabled = bool(context_cfg.get("enabled", True))
        self.context_when = str(context_cfg.get("when", "final"))
        self.context_block = None
        if self.context_enabled:
            self.context_block = PartGlobalContextBlock(
                hidden_dim=self.hidden_dim,
                part_groups=context_cfg.get("part_groups") or ["mouth", "eye", "brow", "nose_cheek", "global"],
                transformer_layers=int(context_cfg.get("transformer_layers", 1)),
                transformer_heads=int(context_cfg.get("transformer_heads", 4)),
                context_scale_init=float(context_cfg.get("context_scale_init", 0.5)),
                dropout=float(context_cfg.get("dropout", dropout)),
                prior_gate=context_cfg.get("prior_gate", {}) or {},
            )
        self.diagnostics: Dict[str, torch.Tensor] = {}
        self.diagnostics_mode = diagnostics

    @classmethod
    def from_config(cls, hidden_dim: int, cfg: Dict[str, Any] | None) -> "EdgeContextGNNEncoder":
        cfg = dict(cfg or {})
        return cls(
            hidden_dim=int(hidden_dim),
            edge_attr_dim=int(cfg.get("edge_attr_dim", 8)),
            num_layers=int(cfg.get("num_layers", 3)),
            edge_hidden_dim=int(cfg.get("edge_hidden_dim", 32)),
            dropout=float(cfg.get("dropout", 0.2)),
            residual=bool(cfg.get("residual", True)),
            layer_norm=bool(cfg.get("layer_norm", True)),
            aggregation=str(cfg.get("aggregation", "mean")),
            layer_output_concat=bool(cfg.get("layer_output_concat", False)),
            multiscale_fusion=cfg.get("multiscale_fusion", {}) or {},
            context_injection=cfg.get("context_injection", {}) or {},
            diagnostics=cfg.get("diagnostics", "eval"),
        )

    def _inject_after_layer(self, layer_idx: int) -> bool:
        if not self.context_enabled:
            return False
        if self.context_when == "final":
            return layer_idx == self.num_layers - 1
        if self.context_when == "after_each":
            return True
        return False

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch_index: torch.Tensor,
        part_soft: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("EdgeContextGNNEncoder requires edge_attr_cat; enable graph.edge_features.")
        src, dst = edge_index[0].long(), edge_index[1].long()
        dst_degree = None
        if self.layers and self.layers[0].aggregation == "mean":
            dst_degree = h.new_zeros((h.size(0), 1))
            dst_degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
        mode = self.diagnostics_mode
        collect_diagnostics = bool(mode) if isinstance(mode, bool) else str(mode).lower() in {"always", "true", "1", "yes"}
        if isinstance(mode, str) and str(mode).lower() in {"eval", "validation", "val"}:
            collect_diagnostics = not self.training
        diag: Dict[str, torch.Tensor] = {}
        layer_outputs: Dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            h, layer_diag = layer(
                h,
                edge_index,
                edge_attr,
                src=src,
                dst=dst,
                dst_degree=dst_degree,
                collect_diagnostics=collect_diagnostics,
            )
            for key, value in layer_diag.items():
                diag[f"layer{idx + 1}_{key}"] = value
            if self.context_block is not None and self._inject_after_layer(idx):
                h, context_diag = self.context_block(
                    h,
                    part_soft,
                    batch_index,
                    num_graphs,
                    collect_diagnostics=collect_diagnostics,
                )
                for key, value in context_diag.items():
                    diag[key] = value
            layer_number = idx + 1
            if self.multiscale_enabled and layer_number in self.multiscale_layers:
                layer_outputs[layer_number] = h
        if self.multiscale_fusion is not None:
            selected = [layer_outputs[layer_number] for layer_number in self.multiscale_layers]
            h = self.multiscale_fusion(selected)
            if collect_diagnostics:
                diag["multiscale_fusion_enabled"] = h.new_tensor(1.0).detach()
                diag["multiscale_fusion_inputs"] = h.new_tensor(float(len(selected))).detach()
                diag["multiscale_fused_node_embedding_norm_mean"] = h.detach().norm(dim=1).mean()
                diag["multiscale_fused_node_embedding_std_mean"] = h.detach().std(dim=0, unbiased=False).mean()
        if collect_diagnostics:
            diag["final_node_embedding_norm_mean"] = h.detach().norm(dim=1).mean()
            diag["final_node_embedding_std_mean"] = h.detach().std(dim=0, unbiased=False).mean()
        self.diagnostics = diag
        return h
