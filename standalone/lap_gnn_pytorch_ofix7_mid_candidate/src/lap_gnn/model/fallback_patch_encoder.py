"""Landmark-free fallback patch encoders for D16 v4."""

from __future__ import annotations

import torch

from lap_gnn.data.patch_tokenizer import grid_edge_index, image_to_patch_tokens
from lap_gnn.model.classifier import D16Classifier
from lap_gnn.model.part_aware_gnn import PartAwareGNN


class GridPatchEncoder(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 6,
        hidden_dim: int = 96,
        layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 7,
        connectivity: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.grid_size = 48 // self.patch_size
        self.token_count = self.grid_size * self.grid_size
        self.token_dim = 8
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )
        self.gnn = PartAwareGNN(hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.classifier = D16Classifier(input_dim=hidden_dim, hidden_dim=hidden_dim * 2, num_classes=num_classes, dropout=dropout)
        self.connectivity = int(connectivity)

    def _batched_edges(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base = grid_edge_index(self.grid_size, self.connectivity).to(device)
        offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1) * self.token_count
        edges = base.unsqueeze(0) + offsets
        return edges.permute(1, 0, 2).reshape(2, -1)

    def forward(self, image_48: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, _ = image_to_patch_tokens(image_48, self.patch_size)
        bsz, num_tokens, _ = tokens.shape
        h = self.embed(tokens).reshape(bsz * num_tokens, -1)
        h = self.gnn(h, self._batched_edges(bsz, h.device))
        h = h.view(bsz, num_tokens, -1)
        z = h.mean(dim=1)
        logits = self.classifier(z)
        return {
            "logits": logits,
            "z_fallback": z,
            "fallback_token_count": torch.full((bsz,), num_tokens, device=image_48.device, dtype=torch.float32),
        }


class PatchTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 6,
        hidden_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.grid_size = 48 // self.patch_size
        self.token_count = self.grid_size * self.grid_size
        self.token_dim = 8
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.token_count + 1, hidden_dim))
        block = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(heads),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(block, num_layers=int(layers))
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.classifier = D16Classifier(input_dim=hidden_dim, hidden_dim=hidden_dim * 2, num_classes=num_classes, dropout=dropout)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, image_48: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, _ = image_to_patch_tokens(image_48, self.patch_size)
        bsz, num_tokens, _ = tokens.shape
        h = self.embed(tokens)
        cls = self.cls_token.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed[:, : num_tokens + 1, :]
        h = self.encoder(h)
        z = self.norm(h[:, 0])
        logits = self.classifier(z)
        return {
            "logits": logits,
            "z_fallback": z,
            "fallback_token_count": torch.full((bsz,), num_tokens, device=image_48.device, dtype=torch.float32),
        }
