"""D16 Landmark-aware Pixel Evidence GNN v0."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from d16.data.mediapipe_priors import PART_NAMES
from d16.models.classifier import D16Classifier
from d16.models.evidence_heads import PartPooling
from d16.models.part_aware_gnn import PartAwareGNN
from d16.models.pixel_encoder import PixelEncoder


class D16Model(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        gnn_layers: int = 3,
        num_classes: int = 7,
        dropout: float = 0.1,
        part_names: List[str] | None = None,
        dual_head: bool = False,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names or PART_NAMES)
        self.dual_head = bool(dual_head)
        self.encoder = PixelEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gnn = PartAwareGNN(hidden_dim=hidden_dim, layers=gnn_layers, dropout=dropout)
        self.pooling = PartPooling(self.part_names)
        classifier_dim = hidden_dim * 5
        classifier_hidden = hidden_dim * 2
        if self.dual_head:
            self.detected_head = D16Classifier(
                input_dim=classifier_dim,
                hidden_dim=classifier_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )
            self.fallback_head = D16Classifier(
                input_dim=classifier_dim,
                hidden_dim=classifier_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )
            self.classifier = None
        else:
            self.classifier = D16Classifier(
                input_dim=classifier_dim,
                hidden_dim=classifier_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> "D16Model":
        model_cfg = config.get("model", {}) or {}
        return cls(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            gnn_layers=int(model_cfg.get("gnn_layers", 3)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            part_names=model_cfg.get("part_names") or PART_NAMES,
            dual_head=bool(model_cfg.get("dual_head", False)),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        h = self.encoder(batch.x_cat)
        h = self.gnn(h, batch.edge_index_cat)
        pooled, valid = self.pooling(
            h,
            batch.part_soft_cat,
            batch.batch_index,
            batch.num_graphs,
            batch.valid_part_mask,
        )
        z_image = torch.cat(
            [pooled["mouth"], pooled["eye"], pooled["brow"], pooled["nose_cheek"], pooled["global"]],
            dim=1,
        )
        result = {
            "z_image": z_image,
            "node_embeddings": h,
            "part_embeddings": pooled,
            "valid_part_groups": valid,
        }
        if self.dual_head:
            detected_logits = self.detected_head(z_image)
            fallback_logits = self.fallback_head(z_image)
            detected_mask = batch.landmark_missing_flag.to(device=z_image.device).long().eq(0)
            logits = torch.where(detected_mask.unsqueeze(1), detected_logits, fallback_logits)
            result.update(
                {
                    "logits": logits,
                    "logits_detected": detected_logits,
                    "logits_fallback": fallback_logits,
                    "routed_head_id": torch.where(
                        detected_mask,
                        torch.zeros_like(batch.landmark_missing_flag, device=z_image.device, dtype=torch.long),
                        torch.ones_like(batch.landmark_missing_flag, device=z_image.device, dtype=torch.long),
                    ),
                }
            )
            return result
        logits = self.classifier(z_image)
        result.update({
            "logits": logits,
        })
        return result
