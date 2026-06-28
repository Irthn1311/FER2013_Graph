"""Coarse-to-fine micro-motif support readout for D16 pixel-GNN embeddings.

Major and micro motifs are learned latent readout queries. They are debugging
and representation-learning components, not semantic detectors, evidence, or
causal explanations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

from d16.models.part_motif_query_readout import _valid_num_heads


class MicroMotifSupportReadout(torch.nn.Module):
    """A3-style major motifs plus weak gated micro-detail support motifs."""

    GROUPS: Dict[str, List[str]] = {
        "mouth": ["mouth", "left_mouth_corner", "right_mouth_corner"],
        "eye": ["left_eye", "right_eye"],
        "brow": ["left_brow", "right_brow"],
        "nose_cheek": ["nose", "left_cheek", "right_cheek"],
    }

    def __init__(
        self,
        part_names: Iterable[str],
        hidden_dim: int,
        output_dim: int,
        major_motif_counts: Dict[str, int] | None = None,
        micro_motif_counts: Dict[str, int] | None = None,
        lambda_part: float = 1.0,
        lambda_micro_part: float = 1.0,
        lambda_detail: float = 0.05,
        eps: float = 1e-6,
        gradient_x_index: int = 1,
        gradient_y_index: int = 2,
        normalize_detail_per_graph: bool = True,
        clamp_detail: float = 2.0,
        detach_detail_score: bool = True,
        use_cls_token: bool = True,
        use_token_type_embedding: bool = True,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        residual_concat: bool = True,
        micro_support_gate: bool = True,
        prior_gate: Dict[str, Any] | None = None,
        diagnostics: bool = True,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.lambda_part = float(lambda_part)
        self.lambda_micro_part = float(lambda_micro_part)
        self.lambda_detail = float(lambda_detail)
        self.eps = float(eps)
        self.gradient_x_index = int(gradient_x_index)
        self.gradient_y_index = int(gradient_y_index)
        self.normalize_detail_per_graph = bool(normalize_detail_per_graph)
        self.clamp_detail = float(clamp_detail)
        self.detach_detail_score = bool(detach_detail_score)
        self.use_cls_token = bool(use_cls_token)
        self.residual_concat = bool(residual_concat)
        self.micro_support_gate = bool(micro_support_gate)
        self.diagnostics = bool(diagnostics)
        self.part_order = ["mouth", "eye", "brow", "nose_cheek", "global"]
        gate_cfg = dict(prior_gate or {})
        self.prior_gate_enabled = bool(gate_cfg.get("enabled", False))
        self.prior_gate_learnable = bool(gate_cfg.get("learnable", True))

        major_counts = major_motif_counts or {"mouth": 3, "eye": 3, "brow": 3, "nose_cheek": 1, "global": 2}
        micro_counts = micro_motif_counts or {"mouth": 2, "eye": 2, "brow": 2, "nose_cheek": 1, "global": 1}
        self.major_motif_counts = {name: int(major_counts.get(name, 0)) for name in self.part_order}
        self.micro_motif_counts = {name: int(micro_counts.get(name, 0)) for name in self.part_order}
        missing_major = [name for name in self.part_order if self.major_motif_counts[name] <= 0]
        negative_micro = [name for name in self.part_order if self.micro_motif_counts[name] < 0]
        if missing_major:
            raise ValueError(f"micro_motif_support major_motif_counts must be positive for {missing_major}")
        if negative_micro:
            raise ValueError(f"micro_motif_support micro_motif_counts must be non-negative for {negative_micro}")

        self.group_indices = {group_name: self._indices(names) for group_name, names in self.GROUPS.items()}
        missing_groups = [name for name, indices in self.group_indices.items() if not indices]
        if missing_groups:
            raise ValueError(f"Cannot infer D16 part indices for groups: {missing_groups}; available={self.part_names}")

        self.major_parts, self.major_names = self._build_names(self.major_motif_counts, suffix="")
        self.micro_parts, self.micro_names = self._build_names(self.micro_motif_counts, suffix="_micro")
        self.num_major = len(self.major_names)
        self.num_micro = len(self.micro_names)
        if self.num_micro <= 0:
            raise ValueError("micro_motif_support requires at least one micro motif token")
        self.num_tokens = self.num_major + self.num_micro
        self.register_buffer(
            "major_part_index",
            torch.tensor([self.part_order.index(name) for name in self.major_parts], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "micro_part_index",
            torch.tensor([self.part_order.index(name) for name in self.micro_parts], dtype=torch.long),
            persistent=False,
        )

        prior_gate_init = self._prior_gate_init_values(gate_cfg.get("init", 1.0))
        if self.prior_gate_enabled and self.prior_gate_learnable:
            self.prior_gate_logit = torch.nn.Parameter(self._logit(prior_gate_init))
            self.register_buffer("prior_gate_value", torch.ones(len(self.part_order), dtype=torch.float32), persistent=False)
        else:
            self.prior_gate_logit = None
            gate_value = prior_gate_init if self.prior_gate_enabled else torch.ones(len(self.part_order), dtype=torch.float32)
            self.register_buffer("prior_gate_value", gate_value, persistent=False)

        self.major_queries = torch.nn.Parameter(torch.randn(self.num_major, self.hidden_dim) * 0.02)
        self.micro_queries = torch.nn.Parameter(torch.randn(self.num_micro, self.hidden_dim) * 0.02)
        self.major_key_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.major_value_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.micro_key_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.micro_value_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        heads = _valid_num_heads(self.hidden_dim, int(transformer_heads))
        ff_dim = max(self.hidden_dim, int(round(self.hidden_dim * float(mlp_ratio))))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(transformer_layers)),
            enable_nested_tensor=False,
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if self.use_cls_token else None
        self.token_type_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_dim))
            if bool(use_token_type_embedding)
            else None
        )
        self.micro_project = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 2),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
        )
        projection_input_dim = self.hidden_dim
        if self.residual_concat:
            projection_input_dim += len(self.part_order) * self.hidden_dim
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(projection_input_dim),
            torch.nn.Linear(projection_input_dim, self.output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.output_dim, self.output_dim),
        )

    def _build_names(self, counts: Dict[str, int], suffix: str) -> tuple[List[str], List[str]]:
        parts: List[str] = []
        names: List[str] = []
        for part in self.part_order:
            for idx in range(int(counts[part])):
                parts.append(part)
                names.append(f"{part}{suffix}_{idx}" if suffix else f"{part}_{idx}")
        return parts, names

    def _indices(self, names: Iterable[str]) -> List[int]:
        return [self.part_names.index(name) for name in names if name in self.part_names]

    @staticmethod
    def _logit(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(values / (1.0 - values))

    def _prior_gate_init_values(self, init: Any) -> torch.Tensor:
        if isinstance(init, dict):
            values = [float(init.get(name, 1.0)) for name in self.part_order]
        else:
            values = [float(init)] * len(self.part_order)
        return torch.tensor(values, dtype=torch.float32).clamp(0.0, 1.0)

    def _prior_gate_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.prior_gate_enabled:
            return torch.ones(len(self.part_order), device=device, dtype=dtype)
        if self.prior_gate_logit is not None:
            return torch.sigmoid(self.prior_gate_logit).to(device=device, dtype=dtype)
        return self.prior_gate_value.to(device=device, dtype=dtype)

    def _stack_parts(self, part_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.part_order if name not in part_embeddings]
        if missing:
            raise KeyError(f"Missing D16 part embeddings for A4 residual concat: {missing}")
        return torch.stack([part_embeddings[name] for name in self.part_order], dim=1)

    def _group_prior(self, part_soft: torch.Tensor, group_name: str) -> torch.Tensor:
        if group_name == "global":
            return torch.ones((part_soft.size(0),), device=part_soft.device, dtype=part_soft.dtype)
        indices = self.group_indices.get(group_name)
        if not indices:
            raise KeyError(f"Missing D16 part indices for group {group_name!r}")
        return part_soft[:, indices].max(dim=1).values

    def _valid_group_mask(
        self,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None,
        num_graphs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_part_groups is None:
            return torch.ones((num_graphs, len(self.part_order)), device=device, dtype=torch.bool)
        if isinstance(valid_part_groups, torch.Tensor):
            mask = valid_part_groups.to(device=device, dtype=torch.bool)
            if tuple(mask.shape) != (num_graphs, len(self.part_order)):
                raise ValueError(
                    f"valid_part_groups tensor shape {tuple(mask.shape)} does not match {(num_graphs, len(self.part_order))}"
                )
            return mask
        masks = []
        for name in self.part_order:
            value = valid_part_groups.get(name)
            if value is None:
                raise KeyError(f"Missing valid_part_groups entry for A4 readout: {name}")
            masks.append(value.to(device=device, dtype=torch.bool))
        return torch.stack(masks, dim=1)

    def _detail_score(self, x_cat: torch.Tensor | None, node_mask: torch.Tensor) -> tuple[torch.Tensor | None, bool]:
        if x_cat is None:
            return None, False
        if x_cat.dim() != 2:
            return None, False
        max_idx = max(self.gradient_x_index, self.gradient_y_index)
        if x_cat.size(1) <= max_idx or self.lambda_detail == 0.0:
            return None, False
        x_g = x_cat[node_mask]
        gx = x_g[:, self.gradient_x_index]
        gy = x_g[:, self.gradient_y_index]
        detail = torch.sqrt(gx.square() + gy.square() + self.eps)
        if self.detach_detail_score:
            detail = detail.detach()
        if self.normalize_detail_per_graph:
            mean = detail.mean()
            std = detail.std(unbiased=False).clamp_min(self.eps)
            detail = (detail - mean) / std
        if self.clamp_detail > 0:
            detail = detail.clamp(-self.clamp_detail, self.clamp_detail)
        if not torch.isfinite(detail).all().item():
            detail = torch.zeros_like(detail)
            return detail, False
        return detail, True

    def _attend_branch(
        self,
        h_g: torch.Tensor,
        part_g: torch.Tensor,
        queries: torch.Tensor,
        key_proj: torch.nn.Linear,
        value_proj: torch.nn.Linear,
        motif_parts: List[str],
        valid_groups_g: torch.Tensor,
        lambda_part: float,
        detail_score: torch.Tensor | None = None,
        lambda_detail: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = h_g.dtype
        keys = key_proj(h_g)
        values = value_proj(h_g)
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        content_scores = torch.matmul(queries.to(dtype=dtype), keys.transpose(0, 1)) * scale
        tokens = []
        entropy = h_g.new_zeros((len(motif_parts),))
        peak = h_g.new_zeros((len(motif_parts),))
        part_mass = h_g.new_zeros((len(motif_parts),))
        detail_mean = h_g.new_zeros((len(motif_parts),))
        for motif_idx, group_name in enumerate(motif_parts):
            group_idx = self.part_order.index(group_name)
            if not bool(valid_groups_g[group_idx].item()):
                tokens.append(values.new_zeros((self.hidden_dim,)))
                continue
            prior = self._group_prior(part_g, group_name).to(dtype=dtype).clamp_min(self.eps)
            prior_gate = self._prior_gate_values(h_g.device, dtype)[group_idx]
            scores = content_scores[motif_idx] + float(lambda_part) * prior_gate * torch.log(prior)
            if detail_score is not None and float(lambda_detail) != 0.0:
                scores = scores + float(lambda_detail) * detail_score.to(device=h_g.device, dtype=dtype)
            alpha = torch.softmax(scores, dim=0)
            tokens.append(torch.sum(values * alpha.unsqueeze(1), dim=0))
            safe_alpha = alpha.clamp_min(self.eps)
            entropy[motif_idx] = -(alpha * torch.log(safe_alpha)).sum()
            peak[motif_idx] = alpha.max()
            part_mass[motif_idx] = torch.sum(alpha * prior.clamp(0.0, 1.0))
            if detail_score is not None:
                detail_mean[motif_idx] = torch.sum(alpha * detail_score.to(device=h_g.device, dtype=dtype))
        return torch.stack(tokens, dim=0), entropy, peak, part_mass, detail_mean

    def _pad_by_graph(
        self,
        node_embeddings: torch.Tensor,
        part_soft: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        x_cat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_index.to(torch.long), minlength=num_graphs)
        max_nodes = int(counts.max().item()) if counts.numel() else 0
        if max_nodes <= 0:
            raise ValueError("A4 micro_motif_support received an empty batch")
        h_pad = node_embeddings.new_zeros((num_graphs, max_nodes, self.hidden_dim))
        part_pad = part_soft.new_zeros((num_graphs, max_nodes, part_soft.size(1)))
        node_valid = torch.zeros((num_graphs, max_nodes), device=node_embeddings.device, dtype=torch.bool)
        detail_pad = node_embeddings.new_zeros((num_graphs, max_nodes))
        detail_available = torch.zeros((num_graphs,), device=node_embeddings.device, dtype=torch.bool)
        for graph_id in range(num_graphs):
            mask = batch_index == graph_id
            n_nodes = int(mask.sum().item())
            if n_nodes <= 0:
                continue
            h_pad[graph_id, :n_nodes] = node_embeddings[mask]
            part_pad[graph_id, :n_nodes] = part_soft[mask]
            node_valid[graph_id, :n_nodes] = True
            detail_score, detail_ok = self._detail_score(x_cat, mask)
            if detail_score is not None:
                detail_pad[graph_id, :n_nodes] = detail_score.to(device=node_embeddings.device, dtype=node_embeddings.dtype)
            detail_available[graph_id] = bool(detail_ok)
        return h_pad, part_pad, node_valid, detail_pad, detail_available

    def _group_priors_padded(self, part_pad: torch.Tensor) -> torch.Tensor:
        priors = []
        for group_name in self.part_order:
            if group_name == "global":
                prior = torch.ones(part_pad.shape[:2], device=part_pad.device, dtype=part_pad.dtype)
            else:
                indices = self.group_indices.get(group_name)
                if not indices:
                    raise KeyError(f"Missing D16 part indices for group {group_name!r}")
                prior = part_pad[:, :, indices].amax(dim=2)
            priors.append(prior)
        return torch.stack(priors, dim=1)

    def _attend_branch_padded(
        self,
        h_pad: torch.Tensor,
        group_priors: torch.Tensor,
        node_valid: torch.Tensor,
        valid_groups: torch.Tensor,
        queries: torch.Tensor,
        key_proj: torch.nn.Linear,
        value_proj: torch.nn.Linear,
        motif_parts: List[str],
        lambda_part: float,
        detail_score: torch.Tensor | None = None,
        detail_available: torch.Tensor | None = None,
        lambda_detail: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = h_pad.size(0)
        dtype = h_pad.dtype
        keys = key_proj(h_pad)
        values = value_proj(h_pad)
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        scores = torch.einsum("kh,bnh->bkn", queries.to(device=h_pad.device, dtype=dtype), keys) * scale
        part_idx = torch.tensor(
            [self.part_order.index(part_name) for part_name in motif_parts],
            device=h_pad.device,
            dtype=torch.long,
        )
        prior = group_priors[:, part_idx, :].clamp_min(self.eps)
        prior_gate = self._prior_gate_values(h_pad.device, dtype)[part_idx].view(1, -1, 1)
        scores = scores + float(lambda_part) * prior_gate * torch.log(prior)
        if detail_score is not None and float(lambda_detail) != 0.0:
            detail = detail_score.to(device=h_pad.device, dtype=dtype)
            if detail_available is not None:
                detail = detail * detail_available.to(device=h_pad.device, dtype=dtype).view(batch_size, 1)
            scores = scores + float(lambda_detail) * detail.unsqueeze(1)
        scores = scores.masked_fill(~node_valid.unsqueeze(1), torch.finfo(dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        motif_valid = valid_groups[:, part_idx].to(dtype=dtype)
        alpha = alpha * motif_valid.unsqueeze(-1)
        tokens = torch.bmm(alpha, values)
        safe_alpha = alpha.clamp_min(self.eps)
        entropy = -(alpha * torch.log(safe_alpha)).sum(dim=-1)
        peak = alpha.max(dim=-1).values
        part_mass = torch.sum(alpha * prior.clamp(0.0, 1.0), dim=-1)
        if detail_score is None:
            detail_mean = h_pad.new_zeros((batch_size, len(motif_parts)))
        else:
            detail_mean = torch.sum(alpha * detail_score.to(device=h_pad.device, dtype=dtype).unsqueeze(1), dim=-1)
        return tokens, entropy, peak, part_mass, detail_mean

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
        part_soft: torch.Tensor,
        num_graphs: int,
        x_cat: torch.Tensor | None = None,
        part_embeddings: Dict[str, torch.Tensor] | None = None,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if node_embeddings.dim() != 2:
            raise ValueError(f"node_embeddings must be [N, H], got {tuple(node_embeddings.shape)}")
        if node_embeddings.size(1) != self.hidden_dim:
            raise ValueError(f"node embedding dim {node_embeddings.size(1)} does not match hidden_dim={self.hidden_dim}")
        if part_soft.dim() != 2:
            raise ValueError(f"part_soft must be [N, num_parts], got {tuple(part_soft.shape)}")
        if part_soft.size(0) != node_embeddings.size(0):
            raise ValueError("part_soft and node_embeddings must have the same node count")
        if x_cat is not None and x_cat.size(0) != node_embeddings.size(0):
            raise ValueError("x_cat and node_embeddings must have the same node count")

        device = node_embeddings.device
        dtype = node_embeddings.dtype
        num_graphs = int(num_graphs)
        valid_groups = self._valid_group_mask(valid_part_groups, num_graphs, device)
        h_pad, part_pad, node_valid, detail_pad, detail_available = self._pad_by_graph(
            node_embeddings,
            part_soft,
            batch_index,
            num_graphs,
            x_cat,
        )
        group_priors = self._group_priors_padded(part_pad)
        major_tokens, major_entropy, major_peak, major_mass, _ = self._attend_branch_padded(
            h_pad,
            group_priors,
            node_valid,
            valid_groups,
            self.major_queries.to(device=device, dtype=dtype),
            self.major_key_proj,
            self.major_value_proj,
            self.major_parts,
            self.lambda_part,
        )
        micro_tokens, micro_entropy, micro_peak, micro_mass, micro_detail = self._attend_branch_padded(
            h_pad,
            group_priors,
            node_valid,
            valid_groups,
            self.micro_queries.to(device=device, dtype=dtype),
            self.micro_key_proj,
            self.micro_value_proj,
            self.micro_parts,
            self.lambda_micro_part,
            detail_score=detail_pad,
            detail_available=detail_available,
            lambda_detail=self.lambda_detail,
        )

        all_tokens = torch.cat([major_tokens, micro_tokens], dim=1)
        tokens = all_tokens
        if self.token_type_embedding is not None:
            tokens = tokens + self.token_type_embedding.to(device=device, dtype=dtype)
        if self.cls_token is not None:
            cls = self.cls_token.to(device=device, dtype=dtype).expand(num_graphs, -1, -1)
            tokens_in = torch.cat([cls, tokens], dim=1)
        else:
            tokens_in = tokens
        transformed = self.encoder(tokens_in)
        transformed_all = transformed[:, 1:, :] if self.cls_token is not None else transformed
        transformed_major = transformed_all[:, : self.num_major, :]
        transformed_micro = transformed_all[:, self.num_major :, :]
        z_major = transformed_major.mean(dim=1)
        z_micro = transformed_micro.mean(dim=1)
        if self.micro_support_gate:
            gate = self.gate(torch.cat([z_major, z_micro], dim=1))
        else:
            gate = torch.ones_like(z_major)
        z_support = z_major + gate * self.micro_project(z_micro)

        if self.residual_concat:
            if part_embeddings is None:
                raise ValueError("A4 residual_concat requires part_embeddings")
            residual = self._stack_parts(part_embeddings).flatten(start_dim=1)
            fused = torch.cat([z_support, residual], dim=1)
        else:
            fused = z_support
        z_image = self.projection(fused)

        major_norm = torch.linalg.vector_norm(major_tokens, dim=-1)
        micro_norm = torch.linalg.vector_norm(micro_tokens, dim=-1)
        major_usage = major_norm / major_norm.sum(dim=1, keepdim=True).clamp_min(self.eps)
        micro_usage = micro_norm / micro_norm.sum(dim=1, keepdim=True).clamp_min(self.eps)
        major_sim = torch.matmul(
            F.normalize(major_tokens, p=2, dim=-1, eps=self.eps),
            F.normalize(major_tokens, p=2, dim=-1, eps=self.eps).transpose(1, 2),
        )
        micro_sim = torch.matmul(
            F.normalize(micro_tokens, p=2, dim=-1, eps=self.eps),
            F.normalize(micro_tokens, p=2, dim=-1, eps=self.eps).transpose(1, 2),
        )
        major_effective = 1.0 / major_usage.square().sum(dim=1).clamp_min(self.eps)
        micro_effective = 1.0 / micro_usage.square().sum(dim=1).clamp_min(self.eps)
        return {
            "z_image": z_image,
            "major_tokens": major_tokens,
            "major_transformed_tokens": transformed_major,
            "major_usage": major_usage,
            "major_attention_entropy": major_entropy,
            "major_attention_peak": major_peak,
            "major_part_mass": major_mass,
            "major_similarity": major_sim,
            "major_effective_count": major_effective,
            "major_part_index": self.major_part_index.to(device=device),
            "micro_tokens": micro_tokens,
            "micro_transformed_tokens": transformed_micro,
            "micro_usage": micro_usage,
            "micro_attention_entropy": micro_entropy,
            "micro_attention_peak": micro_peak,
            "micro_part_mass": micro_mass,
            "micro_detail_score": micro_detail,
            "micro_similarity": micro_sim,
            "micro_effective_count": micro_effective,
            "micro_part_index": self.micro_part_index.to(device=device),
            "micro_gate": gate,
            "prior_gate_values": self._prior_gate_values(device, dtype),
            "detail_available": detail_available,
        }


