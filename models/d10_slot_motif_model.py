"""D10: Iterative Slot Attention Motif Discovery for FER.

Pipeline (faithful to pixel graph → motif → emotion thesis):
    Pixel graph (N=2304, 7D node, 5D edge)
    → Deep EdgeAware GNN Encoder (multi-layer, larger hidden)
    → Iterative Slot Attention (K motifs, T iterations)
    → Motif Relation Transformer (self-attention between motifs)
    → Class-Motif Attention → emotion logits

Key innovations over D9:
    1. Correct softmax direction: over motifs (pixel competition) not over pixels
    2. Iterative refinement: T=3 rounds of attention + GRU update
    3. Deeper encoder: 3 GNN layers → larger receptive field
    4. Position-aware motif features

Reference: Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.dual_branch_graph_swin_motif import SharedPixelEncoder


class IterativeSlotAttention(nn.Module):
    """Slot Attention module with iterative refinement.

    For each iteration:
        1. Compute attention: each pixel competes to be assigned to one slot
           (softmax over slots, NOT over pixels)
        2. Aggregate pixel features weighted by attention
        3. Update slot representations via GRU
        4. Apply MLP residual

    This naturally creates non-overlapping, focused motif regions.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_slots: int = 8,
        num_iterations: int = 3,
        dropout: float = 0.1,
        eps: float = 1e-8,
        residual_slot_connection: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.eps = float(eps)
        self.residual_slot_connection = bool(residual_slot_connection)

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, self.num_slots, self.hidden_dim) * 0.02)

        # Input/slot normalization
        self.norm_inputs = nn.LayerNorm(self.hidden_dim)
        self.norm_slots = nn.LayerNorm(self.hidden_dim)

        # Projections for attention
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # GRU for slot update
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # MLP residual after GRU
        self.norm_mlp = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, N, D] pixel embeddings from GNN encoder
            node_mask: [B, N] optional mask for valid pixels

        Returns:
            slots: [B, K, D] final motif embeddings
            attn_maps: [B, K, N] final attention maps (motif assignment per pixel)
        """
        bsz, num_pixels, _ = inputs.shape
        inputs_normed = self.norm_inputs(inputs)
        k = self.key_proj(inputs_normed)  # [B, N, D]
        v = self.value_proj(inputs_normed)  # [B, N, D]

        # Initialize slots from learnable parameters
        slots = self.slot_mu.expand(bsz, -1, -1).clone()  # [B, K, D]

        attn_maps = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.query_proj(self.norm_slots(slots))  # [B, K, D]

            # Attention logits: [B, K, N]
            scale = math.sqrt(float(self.hidden_dim))
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) / scale

            # Mask invalid pixels if needed
            if node_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    ~node_mask.bool().unsqueeze(1), -1e4
                )

            # KEY DIFFERENCE from D9: softmax over SLOTS (dim=1)
            # Each pixel competes to be assigned to exactly one slot
            # This creates non-overlapping motif regions
            attn_maps = torch.softmax(attn_logits, dim=1)  # [B, K, N]

            # Mask again after softmax
            if node_mask is not None:
                attn_maps = attn_maps * node_mask.to(dtype=attn_maps.dtype).unsqueeze(1)

            # Normalize weights per slot for aggregation (weighted mean)
            attn_sum = attn_maps.sum(dim=2, keepdim=True).clamp_min(self.eps)  # [B, K, 1]
            attn_weights = attn_maps / attn_sum  # [B, K, N]

            # Aggregate pixel features per slot
            updates = torch.einsum("bkn,bnd->bkd", attn_weights, v)  # [B, K, D]

            # GRU update: slots ← GRU(updates, slots_prev)
            slots = self.gru(
                updates.reshape(-1, self.hidden_dim),
                slots_prev.reshape(-1, self.hidden_dim),
            ).reshape(bsz, self.num_slots, self.hidden_dim)

            if self.residual_slot_connection:
                slots = slots + slots_prev

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_maps


class MotifRelationTransformer(nn.Module):
    """Self-attention transformer over K motif slots to learn inter-motif relations.

    E.g., the relationship between eye motifs and mouth motifs matters for emotion.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )

    def forward(self, motif_features: torch.Tensor) -> torch.Tensor:
        """motif_features: [B, K, D] → [B, K, D] with inter-motif context."""
        return self.transformer(motif_features)


class ClassMotifAttentionHead(nn.Module):
    """Class-query attention over motif representations.

    Each emotion class has a learnable query that attends to different motifs.
    E.g., 'Happy' might attend strongly to mouth motifs,
          'Surprise' might attend to both eye and mouth motifs.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.motif_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.motif_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logit_head = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(
        self, motif_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            motif_context: [B, K, D] motif features after relation transformer

        Returns:
            logits: [B, C] classification logits
            class_motif_attn: [B, C, K] attention weights per class over motifs
            class_repr: [B, C, D] class representations
        """
        k = self.motif_key(motif_context)  # [B, K, D]
        v = self.motif_value(motif_context)  # [B, K, D]

        # Class-motif attention scores
        scores = torch.einsum("cd,bkd->bck", self.class_queries, k)
        scores = scores / math.sqrt(float(self.hidden_dim))
        class_motif_attn = torch.softmax(scores, dim=2)  # [B, C, K]

        # Class representations = weighted sum of motif values
        class_repr = torch.einsum("bck,bkd->bcd", class_motif_attn, v)  # [B, C, D]
        class_repr = self.dropout(class_repr)

        # Logits
        logits = self.logit_head(class_repr).squeeze(-1)  # [B, C]

        return logits, class_motif_attn, class_repr


class D10SlotMotifModel(nn.Module):
    """D10: Full pipeline from pixel graph to emotion via slot-attention motifs.

    Pipeline:
        1. Deep Pixel Encoder: multi-layer edge-aware GNN over pixel graph
        2. Iterative Slot Attention: discover K motifs via competitive attention
        3. Position Encoding: add spatial position info to motif representations
        4. Motif Relation Transformer: self-attention between motifs
        5. Class-Motif Attention: emotion classification from motif combinations
    """

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        pixel_gnn_layers: int = 3,
        num_motifs: int = 8,
        slot_iterations: int = 3,
        motif_relation_layers: int = 1,
        motif_relation_heads: int = 4,
        dropout: float = 0.2,
        use_position_encoding: bool = True,
        height: int = 48,
        width: int = 48,
        use_aux_classifier: bool = True,
        edge_dropout_p: float = 0.0,
        node_noise_std: float = 0.0,
        multi_scale_gnn: bool = False,
        residual_slot_connection: bool = False,
        use_slot_refinement: bool = False,
        freeze_classifier: bool = False,
        freeze_encoder: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_motifs = int(num_motifs)
        self.height = int(height)
        self.width = int(width)
        self.use_position_encoding = bool(use_position_encoding)
        self.use_aux_classifier = bool(use_aux_classifier)
        self.edge_dropout_p = float(edge_dropout_p)
        self.node_noise_std = float(node_noise_std)
        self.multi_scale_gnn = bool(multi_scale_gnn)
        self.residual_slot_connection = bool(residual_slot_connection)
        self.use_slot_refinement = bool(use_slot_refinement)
        self.freeze_classifier = bool(freeze_classifier)
        self.freeze_encoder = bool(freeze_encoder)

        if self.num_nodes != self.height * self.width:
            raise ValueError(
                f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}"
            )

        # 1. Deep Pixel Encoder (reuses proven SharedPixelEncoder from D7)
        self.encoder = SharedPixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            pixel_gnn_layers=int(pixel_gnn_layers),
            dropout=dropout,
        )

        if self.multi_scale_gnn:
            self.encoder_aux = SharedPixelEncoder(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                pixel_gnn_layers=int(pixel_gnn_layers) + 1,
                dropout=dropout,
            )
            self.combine_scale = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU()
            )

        # 2. Iterative Slot Attention Motif Discovery
        self.slot_attention = IterativeSlotAttention(
            hidden_dim=self.hidden_dim,
            num_slots=self.num_motifs,
            num_iterations=int(slot_iterations),
            dropout=dropout,
            residual_slot_connection=self.residual_slot_connection,
        )

        # 2.5 Cross-Attention Slot Refinement (Phase 3)
        if self.use_slot_refinement:
            self.slot_refinement = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.refinement_norm1 = nn.LayerNorm(self.hidden_dim)
            self.refinement_norm2 = nn.LayerNorm(self.hidden_dim)
            self.refinement_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(dropout),
            )

        # 3. Position encoding for motifs (learned from motif center coordinates)
        if self.use_position_encoding:
            self.position_mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.register_buffer(
            "pixel_positions", self._make_positions(), persistent=False
        )

        # 4. Motif Relation Transformer
        self.motif_relation = MotifRelationTransformer(
            hidden_dim=self.hidden_dim,
            num_heads=int(motif_relation_heads),
            num_layers=int(motif_relation_layers),
            dropout=dropout,
        )

        # 5. Classification Head (class-motif attention)
        self.classifier = ClassMotifAttentionHead(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )

        # Optional: auxiliary classifier (mean pooling → MLP)
        # Helps motifs carry classification signal during early training
        if self.use_aux_classifier:
            self.aux_classifier = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.num_classes),
            )

        # Border mask for diagnostics
        self.register_buffer(
            "border_mask", self._make_border_mask(border_width=3), persistent=False
        )

        if self.freeze_classifier:
            for param in self.motif_relation.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
            if self.use_aux_classifier:
                for param in self.aux_classifier.parameters():
                    param.requires_grad = False

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if self.multi_scale_gnn:
                for param in self.encoder_aux.parameters():
                    param.requires_grad = False
                for param in self.combine_scale.parameters():
                    param.requires_grad = False
            for param in self.slot_attention.parameters():
                param.requires_grad = False
            if self.use_slot_refinement:
                for param in self.slot_refinement.parameters():
                    param.requires_grad = False
                for param in self.refinement_norm1.parameters():
                    param.requires_grad = False
                for param in self.refinement_norm2.parameters():
                    param.requires_grad = False
                for param in self.refinement_mlp.parameters():
                    param.requires_grad = False
            if self.use_position_encoding:
                for param in self.pos_mlp.parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "D10SlotMotifModel":
        cfg = dict(config)
        # Remove legacy keys that don't apply
        for key in (
            "edge_hidden_dim", "gnn_layers", "use_edge_gnn", "temperature",
            "edge_score_weight", "num_edges", "motif_prior_path",
            "init_node_gate_from_prior", "prior_init_clamp_min",
            "prior_init_clamp_max", "connectivity",
        ):
            cfg.pop(key, None)
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
        # Parse batch dict or raw tensors
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            node_mask = batch.get("node_mask")
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("D10SlotMotifModel needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("D10SlotMotifModel requires edge_index and edge_attr")
        # Handle DataParallel: edge_index may be (B, 2, E) after scatter
        if edge_index.ndim == 3:
            edge_index = edge_index[0]
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        # Phase 2: Graph Augmentation
        if self.training and self.node_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.node_noise_std
        if self.training and self.edge_dropout_p > 0.0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout_p
            edge_index = edge_index[:, mask]
            if edge_attr is not None:
                if edge_attr.dim() == 2:
                    edge_attr = edge_attr[mask]
                elif edge_attr.dim() == 3:
                    edge_attr = edge_attr[:, mask, :]

        # 1. Deep pixel encoding
        h_pixel = self.encoder(
            x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask
        )

        if self.multi_scale_gnn:
            h_aux = self.encoder_aux(
                x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask
            )
            h_pixel = self.combine_scale(torch.cat([h_pixel, h_aux], dim=-1))

        # 2. Iterative slot attention motif discovery
        motif_embeddings, slot_attn_maps = self.slot_attention(
            h_pixel, node_mask=node_mask
        )
        # motif_embeddings: [B, K, D]
        # slot_attn_maps: [B, K, N] — how much each pixel belongs to each motif

        # 3. Add position information
        if self.use_position_encoding:
            positions = self.pixel_positions.to(
                device=h_pixel.device, dtype=h_pixel.dtype
            )  # [N, 2]
            # Compute motif centers as weighted mean of pixel positions
            attn_sum = slot_attn_maps.sum(dim=2, keepdim=True).clamp_min(1e-8)
            attn_weights_pos = slot_attn_maps / attn_sum  # [B, K, N]
            motif_centers = torch.einsum(
                "bkn,nd->bkd", attn_weights_pos, positions
            )  # [B, K, 2]
            motif_embeddings = motif_embeddings + self.position_mlp(motif_centers)
        else:
            motif_centers = None

        # Phase 3: Cross-Attention Slot Refinement
        # Let the motifs attend to the rich pixel features one last time
        if self.use_slot_refinement:
            norm_motifs = self.refinement_norm1(motif_embeddings)
            refined, _ = self.slot_refinement(
                query=norm_motifs,
                key=h_pixel,
                value=h_pixel,
                key_padding_mask=~node_mask.bool() if node_mask is not None else None
            )
            motif_embeddings = motif_embeddings + refined
            motif_embeddings = motif_embeddings + self.refinement_mlp(self.refinement_norm2(motif_embeddings))

        # 4. Motif relation transformer (self-attention between motifs)
        motif_context = self.motif_relation(motif_embeddings)

        # 5. Classification
        logits, class_motif_attn, class_repr = self.classifier(motif_context)

        # Reshape attention maps to spatial for visualization compatibility
        part_masks = slot_attn_maps  # [B, K, N] — compatible with D6B loss format

        # Build output dict
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "part_masks": part_masks,
            "motif_embeddings": motif_embeddings,
            "motif_context": motif_context,
            "slot_attn_maps": slot_attn_maps,
            "class_motif_attn": class_motif_attn,
            "class_repr": class_repr,
            "class_part_attn": class_motif_attn,
            "pixel_embeddings": h_pixel,
            "h_pixel": h_pixel,
        }

        if motif_centers is not None:
            out["motif_centers"] = motif_centers
            out["part_centers"] = motif_centers

        # Auxiliary classifier
        if self.use_aux_classifier and hasattr(self, "aux_classifier"):
            aux_repr = motif_context.mean(dim=1)  # [B, D]
            out["aux_logits"] = self.aux_classifier(aux_repr)

        # Diagnostics
        out["diagnostics"] = self._compute_diagnostics(out)

        return out

    def _compute_diagnostics(
        self, out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        part_masks = out.get("part_masks")

        if torch.is_tensor(part_masks):
            # Slot diversity: cosine similarity between attention maps
            m = F.normalize(part_masks.float(), dim=2, eps=1e-6)
            sim = torch.bmm(m, m.transpose(1, 2))
            k = sim.shape[1]
            off = sim.masked_select(
                ~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0)
            )
            diagnostics["slot_div"] = off.mean().detach()
            diagnostics["slot_similarity_mean"] = off.mean().detach()

            # Area per motif
            slot_area = part_masks.float().sum(dim=2)  # [B, K]
            area_norm = slot_area / slot_area.sum(dim=1, keepdim=True).clamp_min(1e-6)
            area_entropy = -(area_norm * area_norm.clamp_min(1e-6).log()).sum(dim=1)
            diagnostics["slot_area_mean"] = slot_area.mean().detach()
            diagnostics["slot_area_entropy"] = area_entropy.mean().detach()

            # Border mass
            border_mask = self.border_mask.to(
                device=part_masks.device, dtype=part_masks.dtype
            )
            border_mass = (part_masks * border_mask.view(1, 1, -1)).sum(dim=2)
            slot_mass = part_masks.sum(dim=2).clamp_min(1e-6)
            border_ratio = border_mass / slot_mass
            diagnostics["border_mass_mean"] = border_ratio.mean().detach()

            # Attention map entropy (should decrease over training)
            flat = part_masks.float()
            pixel_entropy = -(flat * flat.clamp_min(1e-8).log()).sum(dim=1)  # [B, N]
            diagnostics["pixel_assignment_entropy"] = pixel_entropy.mean().detach()

        # Class-motif attention patterns
        class_motif_attn = out.get("class_motif_attn")
        if torch.is_tensor(class_motif_attn):
            attn = class_motif_attn.float()
            entropy = -(attn * attn.clamp_min(1e-6).log()).sum(dim=2)
            diagnostics["class_motif_entropy"] = entropy.mean().detach()

        return diagnostics

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
