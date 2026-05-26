"""D16 Landmark-aware Pixel Evidence GNN v0."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from d16.data.mediapipe_priors import PART_NAMES
from d16.models.classifier import D16Classifier
from d16.models.evidence_heads import PartPooling
from d16.models.fallback_patch_encoder import GridPatchEncoder, PatchTransformerEncoder
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
        architecture: str = "single_path",
        fallback_encoder_type: str = "grid_gnn",
        fallback_patch_size: int = 6,
        fallback_gnn_layers: int = 2,
        fallback_transformer_layers: int = 2,
        fallback_transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names or PART_NAMES)
        self.dual_head = bool(dual_head)
        self.architecture = str(architecture)
        self.use_routed_fallback_patch = self.architecture == "routed_fallback_patch"
        self.encoder = PixelEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gnn = PartAwareGNN(hidden_dim=hidden_dim, layers=gnn_layers, dropout=dropout)
        self.pooling = PartPooling(self.part_names)
        classifier_dim = hidden_dim * 5
        classifier_hidden = hidden_dim * 2
        if self.use_routed_fallback_patch:
            self.classifier = D16Classifier(
                input_dim=classifier_dim,
                hidden_dim=classifier_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )
            fallback_type = str(fallback_encoder_type)
            if fallback_type == "grid_gnn":
                self.fallback_encoder = GridPatchEncoder(
                    patch_size=int(fallback_patch_size),
                    hidden_dim=hidden_dim,
                    layers=int(fallback_gnn_layers),
                    dropout=dropout,
                    num_classes=num_classes,
                )
            elif fallback_type == "patch_transformer":
                self.fallback_encoder = PatchTransformerEncoder(
                    patch_size=int(fallback_patch_size),
                    hidden_dim=hidden_dim,
                    layers=int(fallback_transformer_layers),
                    heads=int(fallback_transformer_heads),
                    dropout=dropout,
                    num_classes=num_classes,
                )
            else:
                raise ValueError(f"Unsupported D16 fallback_encoder_type={fallback_type!r}")
            self.fallback_encoder_type = fallback_type
            self.fallback_patch_size = int(fallback_patch_size)
        elif self.dual_head:
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
        fallback_cfg = config.get("fallback_encoder", {}) or {}
        return cls(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            gnn_layers=int(model_cfg.get("gnn_layers", 3)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            part_names=model_cfg.get("part_names") or PART_NAMES,
            dual_head=bool(model_cfg.get("dual_head", False)),
            architecture=str(model_cfg.get("architecture", "dual_head" if bool(model_cfg.get("dual_head", False)) else "single_path")),
            fallback_encoder_type=str(fallback_cfg.get("type", model_cfg.get("fallback_encoder_type", "grid_gnn"))),
            fallback_patch_size=int(fallback_cfg.get("patch_size", model_cfg.get("fallback_patch_size", 6))),
            fallback_gnn_layers=int(fallback_cfg.get("gnn_layers", model_cfg.get("fallback_gnn_layers", 2))),
            fallback_transformer_layers=int(
                fallback_cfg.get("transformer_layers", model_cfg.get("fallback_transformer_layers", 2))
            ),
            fallback_transformer_heads=int(
                fallback_cfg.get("transformer_heads", model_cfg.get("fallback_transformer_heads", 4))
            ),
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
        if self.use_routed_fallback_patch:
            detected_logits = self.classifier(z_image)
            fallback_out = self.fallback_encoder(batch.image_48)
            fallback_logits = fallback_out["logits"]
            detected_mask = batch.landmark_missing_flag.to(device=z_image.device).long().eq(0)
            logits = torch.where(detected_mask.unsqueeze(1), detected_logits, fallback_logits)
            result.update(
                {
                    "logits": logits,
                    "logits_detected_path": detected_logits,
                    "logits_fallback_path": fallback_logits,
                    "routed_path_id": torch.where(
                        detected_mask,
                        torch.zeros_like(batch.landmark_missing_flag, device=z_image.device, dtype=torch.long),
                        torch.ones_like(batch.landmark_missing_flag, device=z_image.device, dtype=torch.long),
                    ),
                    "fallback_token_count": fallback_out["fallback_token_count"],
                    "fallback_encoder_type_id": torch.full(
                        (batch.num_graphs,),
                        1 if self.fallback_encoder_type == "grid_gnn" else 2,
                        device=z_image.device,
                        dtype=torch.long,
                    ),
                    "fallback_patch_size": torch.full(
                        (batch.num_graphs,),
                        self.fallback_patch_size,
                        device=z_image.device,
                        dtype=torch.float32,
                    ),
                }
            )
            return result
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
