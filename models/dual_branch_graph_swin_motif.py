"""D7 dual-branch pixel motif + Graph-Swin region model."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.slot_pixel_part_graph_motif import EdgeAwarePixelMessageLayer, PartSelfAttentionLayer


class SharedPixelEncoder(nn.Module):
    """Edge-aware pixel encoder shared by D6 motif and Graph-Swin branches."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        pixel_gnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(node_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pixel_layers = nn.ModuleList(
            [
                EdgeAwarePixelMessageLayer(
                    hidden_dim=int(hidden_dim),
                    edge_dim=int(edge_dim),
                    dropout=dropout,
                )
                for _ in range(int(pixel_gnn_layers))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_pixel = self.input_proj(x.float())
        if node_mask is not None:
            h_pixel = h_pixel * node_mask.to(dtype=h_pixel.dtype).unsqueeze(-1)
        for layer in self.pixel_layers:
            h_pixel = layer(
                h_pixel,
                edge_index=edge_index,
                edge_attr=edge_attr.float(),
                node_mask=node_mask,
            )
        return h_pixel


class D6BPixelMotifBranch(nn.Module):
    """D6B soft pixel-to-part motif branch fed by shared pixel embeddings."""

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        hidden_dim: int = 64,
        num_part_slots: int = 16,
        part_layers: int = 1,
        part_heads: int = 4,
        dropout: float = 0.2,
        use_part_position: bool = True,
        assignment_temperature: float = 1.0,
        return_attention: bool = True,
        height: int = 48,
        width: int = 48,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.num_part_slots = int(num_part_slots)
        self.height = int(height)
        self.width = int(width)
        self.use_part_position = bool(use_part_position)
        self.assignment_temperature = float(assignment_temperature)
        self.return_attention = bool(return_attention)

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
        self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.class_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.class_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.class_logit_head = nn.Linear(self.hidden_dim, 1)
        self.register_buffer("pixel_positions", self._make_positions(), persistent=False)
        self.register_buffer("border_mask", self._make_border_mask(border_width=3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.part_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(
        self,
        h_pixel: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        part_masks, pool_weights = self._assign_parts(h_pixel, node_mask=node_mask)
        part_features = torch.bmm(pool_weights, self.pixel_value(h_pixel))
        positions = self.pixel_positions.to(h_pixel).unsqueeze(0).expand(h_pixel.shape[0], -1, -1)
        part_centers = torch.bmm(pool_weights, positions)
        if self.use_part_position:
            part_features = part_features + self.position_mlp(part_centers)

        part_context = part_features
        part_attn = None
        for layer in self.part_layers:
            part_context, part_attn = layer(part_context)

        class_part_attn, class_repr_d6, logits_d6 = self._class_part_attention(part_context)
        slot_area = part_masks.mean(dim=2)
        border_mass = (part_masks * self.border_mask.to(part_masks).view(1, 1, -1)).mean(dim=2)
        border_mass_per_slot = self._slot_border_ratio(part_masks)
        return {
            "logits_d6": logits_d6,
            "part_masks": part_masks,
            "part_features": part_features,
            "part_context": part_context,
            "part_attn": part_attn if self.return_attention else None,
            "class_part_attn": class_part_attn,
            "class_repr_d6": class_repr_d6,
            "class_repr": class_repr_d6,
            "slot_area": slot_area,
            "border_mass": border_mass,
            "border_mass_per_slot": border_mass_per_slot,
            "part_centers": part_centers,
            "pool_weights": pool_weights,
        }

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
            logits = logits.masked_fill(~node_mask.bool().unsqueeze(1), -1e4)
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


class WindowAttentionPooling(nn.Module):
    """Light local window encoder with optional self-attention then attention pooling."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_window_mha: bool = False,
    ) -> None:
        super().__init__()
        self.use_window_mha = bool(use_window_mha)
        if self.use_window_mha:
            self.attn = nn.MultiheadAttention(
                embed_dim=int(hidden_dim),
                num_heads=int(num_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.norm_attn = nn.LayerNorm(int(hidden_dim))
            self.ffn = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.Dropout(float(dropout)),
            )
            self.norm_ffn = nn.LayerNorm(int(hidden_dim))
        self.score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_windows, tokens_per_window, hidden_dim = windows.shape
        tokens = windows.reshape(bsz * num_windows, tokens_per_window, hidden_dim)
        if self.use_window_mha:
            attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm_attn(tokens + attn_out)
            tokens = self.norm_ffn(tokens + self.ffn(tokens))
        scores = self.score(tokens).squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * tokens, dim=1)
        return pooled.reshape(bsz, num_windows, hidden_dim), attn.reshape(bsz, num_windows, tokens_per_window)


class GraphSwinBranch(nn.Module):
    """Window/shifted-window hierarchy over shared pixel embeddings."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 64,
        height: int = 48,
        width: int = 48,
        window_size: int = 6,
        shift_size: int = 3,
        window_heads: int = 4,
        use_window_mha: bool = False,
        region_merge: bool = True,
        class_head_type: str = "attn_only",
        use_region_transformer: bool = False,
        region_layers: int = 1,
        region_heads: int = 4,
        dropout: float = 0.2,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.height = int(height)
        self.width = int(width)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.region_merge = bool(region_merge)
        self.class_head_type = str(class_head_type or "attn_only")
        self.use_region_transformer = bool(use_region_transformer)
        if self.class_head_type not in {"attn_only", "attn_plus_mean"}:
            raise ValueError(f"Unsupported graph_swin.class_head_type: {self.class_head_type}")
        if self.height % self.window_size != 0 or self.width % self.window_size != 0:
            raise ValueError("GraphSwinBranch requires height/width divisible by window_size")
        self.num_win_h = self.height // self.window_size
        self.num_win_w = self.width // self.window_size
        self.num_windows = self.num_win_h * self.num_win_w

        self.window_encoder = WindowAttentionPooling(
            hidden_dim=self.hidden_dim,
            num_heads=int(window_heads),
            dropout=dropout,
            use_window_mha=bool(use_window_mha),
        )
        self.window_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.region_merge_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        if self.use_region_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(region_heads),
                dim_feedforward=self.hidden_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
            )
            self.region_transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=int(region_layers),
            )
        else:
            self.region_transformer = None
        self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.region_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.region_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.region_mean_proj = (
            nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.class_head_type == "attn_plus_mean"
            else None
        )
        self.class_logit_head = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(self, h_pixel: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, hidden_dim = h_pixel.shape
        expected_nodes = self.height * self.width
        if num_nodes != expected_nodes:
            raise ValueError(f"Expected {expected_nodes} nodes, got {num_nodes}")
        h_grid = h_pixel.reshape(bsz, self.height, self.width, hidden_dim)
        regular_windows = self._partition_windows(h_grid)
        reg_tokens, reg_attn = self.window_encoder(regular_windows)

        h_shift = torch.roll(h_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        shifted_windows = self._partition_windows(h_shift)
        shift_tokens, shift_attn = self.window_encoder(shifted_windows)

        window_tokens = self.window_fuse(torch.cat([reg_tokens, shift_tokens], dim=-1))
        region_tokens = self._merge_regions(window_tokens)
        if self.region_transformer is not None:
            region_tokens = self.region_transformer(region_tokens)
        class_region_attn, class_repr_swin, logits_swin = self._class_region_attention(region_tokens)
        return {
            "logits_swin": logits_swin,
            "regular_window_tokens": reg_tokens,
            "shifted_window_tokens": shift_tokens,
            "window_tokens": window_tokens,
            "regular_window_attn": reg_attn,
            "shifted_window_attn": shift_attn,
            "region_tokens": region_tokens,
            "class_region_attn": class_region_attn,
            "class_repr_swin": class_repr_swin,
        }

    def _partition_windows(self, h_grid: torch.Tensor) -> torch.Tensor:
        bsz, height, width, hidden_dim = h_grid.shape
        ws = self.window_size
        windows = h_grid.view(bsz, height // ws, ws, width // ws, ws, hidden_dim)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(bsz, self.num_windows, ws * ws, hidden_dim)

    def _merge_regions(self, window_tokens: torch.Tensor) -> torch.Tensor:
        bsz = window_tokens.shape[0]
        grid = window_tokens.view(bsz, self.num_win_h, self.num_win_w, self.hidden_dim)
        if not self.region_merge:
            return grid.reshape(bsz, self.num_windows, self.hidden_dim)
        if self.num_win_h % 2 != 0 or self.num_win_w % 2 != 0:
            raise ValueError("2x2 region merge requires an even window grid")
        merged = grid.view(
            bsz,
            self.num_win_h // 2,
            2,
            self.num_win_w // 2,
            2,
            self.hidden_dim,
        )
        merged = merged.permute(0, 1, 3, 2, 4, 5).contiguous()
        merged = merged.view(bsz, (self.num_win_h // 2) * (self.num_win_w // 2), self.hidden_dim * 4)
        return self.region_merge_proj(merged)

    def _class_region_attention(
        self,
        region_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_region = self.region_key(region_tokens)
        v_region = self.region_value(region_tokens)
        scores = torch.einsum("ch,brh->bcr", self.class_queries, k_region)
        scores = scores / math.sqrt(float(self.hidden_dim))
        class_region_attn = torch.softmax(scores, dim=2)
        class_repr = torch.einsum("bcr,brh->bch", class_region_attn, v_region)
        if self.region_mean_proj is not None:
            mean_repr = self.region_mean_proj(region_tokens.mean(dim=1))
            class_repr = class_repr + mean_repr.unsqueeze(1)
        logits = self.class_logit_head(class_repr).squeeze(-1)
        return class_region_attn, class_repr, logits


class FusionHead(nn.Module):
    """D7 fusion modes: swin-only, logits sum, or class-level gated fusion."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 64,
        mode: str = "gated_class_repr",
        dropout: float = 0.2,
        use_abs_diff: bool = True,
        use_product: bool = True,
        gate_type: str = "scalar_per_class",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.use_abs_diff = bool(use_abs_diff)
        self.use_product = bool(use_product)
        self.gate_type = str(gate_type)
        gate_in = self.hidden_dim * 2
        if self.use_abs_diff:
            gate_in += self.hidden_dim
        if self.use_product:
            gate_in += self.hidden_dim
        gate_out = 1 if self.gate_type == "scalar_per_class" else self.hidden_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, gate_out),
        )
        self.class_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, branch_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits_swin = branch_out.get("logits_swin")
        logits_d6 = branch_out.get("logits_d6")
        if self.mode == "swin_only":
            if logits_swin is None:
                raise KeyError("swin_only mode requires logits_swin")
            return {"logits": logits_swin, "logits_fused": logits_swin}
        if self.mode == "logits_sum":
            if logits_d6 is None or logits_swin is None:
                raise KeyError("logits_sum mode requires logits_d6 and logits_swin")
            logits = logits_d6 + logits_swin
            return {"logits": logits, "logits_fused": logits}
        if self.mode != "gated_class_repr":
            raise ValueError(f"Unsupported D7 fusion mode: {self.mode}")

        class_repr_d6 = branch_out.get("class_repr_d6")
        class_repr_swin = branch_out.get("class_repr_swin")
        if class_repr_d6 is None or class_repr_swin is None:
            raise KeyError("gated_class_repr mode requires class_repr_d6 and class_repr_swin")
        pieces = [class_repr_d6, class_repr_swin]
        if self.use_abs_diff:
            pieces.append(torch.abs(class_repr_d6 - class_repr_swin))
        if self.use_product:
            pieces.append(class_repr_d6 * class_repr_swin)
        gate = torch.sigmoid(self.gate_mlp(torch.cat(pieces, dim=-1)))
        fused = gate * class_repr_d6 + (1.0 - gate) * class_repr_swin
        logits = self.class_head(fused).squeeze(-1)
        return {
            "logits": logits,
            "logits_fused": logits,
            "fused_class_repr": fused,
            "fusion_gate": gate,
        }


class DualBranchGraphSwinMotifD7(nn.Module):
    """D7: shared pixel graph encoder + D6B motif branch + Graph-Swin branch."""

    VALID_MODES = {"swin_only", "logits_sum", "gated_class_repr"}

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
        use_class_part_attention: bool = True,
        height: int = 48,
        width: int = 48,
        connectivity: int = 8,
        mode: str = "gated_class_repr",
        graph_swin: Optional[Dict[str, Any]] = None,
        fusion: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        del connectivity, use_class_part_attention
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.height = int(height)
        self.width = int(width)
        self.mode = str(mode)
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported D7 mode: {self.mode}")
        if self.num_nodes != self.height * self.width:
            raise ValueError(
                f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}"
            )

        self.encoder = SharedPixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            pixel_gnn_layers=int(pixel_gnn_layers),
            dropout=dropout,
        )
        self.use_d6_branch = self.mode != "swin_only"
        if self.use_d6_branch:
            self.d6_branch = D6BPixelMotifBranch(
                num_classes=self.num_classes,
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                num_part_slots=int(num_part_slots),
                part_layers=int(part_layers),
                part_heads=int(part_heads),
                dropout=dropout,
                use_part_position=bool(use_part_position),
                assignment_temperature=float(assignment_temperature),
                return_attention=bool(return_attention),
                height=self.height,
                width=self.width,
            )
        else:
            self.d6_branch = None
        swin_cfg = dict(graph_swin or {})
        self.swin_branch = GraphSwinBranch(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            height=self.height,
            width=self.width,
            dropout=dropout,
            **swin_cfg,
        )
        fusion_cfg = dict(fusion or {})
        self.fusion = FusionHead(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            mode=self.mode,
            dropout=dropout,
            **fusion_cfg,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DualBranchGraphSwinMotifD7":
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
            raise KeyError("DualBranchGraphSwinMotifD7 needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("DualBranchGraphSwinMotifD7 requires edge_index and edge_attr")
        # Handle DataParallel: edge_index may be (B, 2, E) after scatter
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        h_pixel = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        out: Dict[str, torch.Tensor] = {"pixel_embeddings": h_pixel}
        if self.d6_branch is not None:
            out.update(self.d6_branch(h_pixel, node_mask=node_mask))
        out.update(self.swin_branch(h_pixel))
        out.update(self.fusion(out))
        out["diagnostics"] = self._diagnostics(out)
        return out

    @staticmethod
    def _diagnostics(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        part_masks = out.get("part_masks")
        if torch.is_tensor(part_masks):
            m = F.normalize(part_masks.float(), dim=2, eps=1e-6)
            sim = torch.bmm(m, m.transpose(1, 2))
            k = sim.shape[1]
            off = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0))
            diagnostics["slot_div"] = off.mean().detach()
        slot_area = out.get("slot_area")
        if torch.is_tensor(slot_area):
            area_norm = slot_area.float() / slot_area.float().sum(dim=1, keepdim=True).clamp_min(1e-6)
            area_entropy = -(area_norm * area_norm.clamp_min(1e-6).log()).sum(dim=1)
            diagnostics["slot_area_entropy"] = area_entropy.mean().detach()
            diagnostics["slot_area_mean"] = slot_area.float().mean().detach()
        border_mass = out.get("border_mass_per_slot", out.get("border_mass"))
        if torch.is_tensor(border_mass):
            diagnostics["border_mass_mean"] = border_mass.float().mean().detach()
        class_part_attn = out.get("class_part_attn")
        if torch.is_tensor(class_part_attn):
            entropy = -(class_part_attn.float() * class_part_attn.float().clamp_min(1e-6).log()).sum(dim=2)
            diagnostics["class_part_entropy"] = entropy.mean().detach()
        class_region_attn = out.get("class_region_attn")
        if torch.is_tensor(class_region_attn):
            attn = class_region_attn.float()
            entropy = -(attn * attn.clamp_min(1e-6).log()).sum(dim=2)
            diagnostics["swin_class_region_entropy_mean"] = entropy.mean().detach()
        region_tokens = out.get("region_tokens")
        if torch.is_tensor(region_tokens):
            diagnostics["swin_region_token_norm"] = region_tokens.float().norm(dim=-1).mean().detach()
        fusion_gate = out.get("fusion_gate")
        if torch.is_tensor(fusion_gate):
            gate = fusion_gate.detach().float()
            diagnostics["fusion_gate_mean"] = gate.mean()
            class_gate = gate.mean(dim=0)
            if class_gate.ndim == 2:
                class_gate = class_gate.mean(dim=1)
            for idx in range(min(class_gate.shape[0], 7)):
                diagnostics[f"fusion_gate_class_{idx}"] = class_gate[idx]
        return diagnostics
