"""D13C diagnostic image-level SupCon model.

D13C reuses the D13B diagnostic slot bottleneck and adds an image-level
projection head over the pooled slot representation. It does not apply SupCon
to individual slots, does not align slot indices, and does not use prototypes.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.d13b_motif_slot_model import D13BMotifSlotModel


class D13CSupConModel(D13BMotifSlotModel):
    """D13B slot image representation + image-level SupCon projection head."""

    def __init__(
        self,
        projection_dim: int = 64,
        projection_hidden_dim: int = 128,
        freeze_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        # D13C controls freezing itself; keep D13B init from freezing only D13A.
        kwargs["freeze_d13a_backbone"] = False
        super().__init__(**kwargs)
        self.projection_dim = int(projection_dim)
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.z_image_dim = int(self.slot_dim * 3)
        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.z_image_dim),
            nn.Linear(self.z_image_dim, self.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(self.projection_hidden_dim, self.projection_dim),
        )
        if bool(freeze_backbone):
            self.freeze_backbone_for_d13c()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "D13CSupConModel":
        cfg = dict(cfg)
        cfg.pop("name", None)
        cfg.pop("base_model", None)
        cfg.pop("init_d13b_checkpoint", None)
        cfg.pop("prototype_weight", None)
        cfg.pop("motif_level_supcon", None)
        return cls(**cfg)

    def freeze_backbone_for_d13c(self) -> None:
        """Freeze D13A + slot encoder; leave readout/classifier/projection trainable."""
        for module in (
            self.pixel_encoder,
            self.reduction,
            self.region_encoder,
            self.slot_attention,
            self.slot_self_attention,
        ):
            for param in module.parameters():
                param.requires_grad = False

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def total_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        out = super().forward(batch)
        slot_embeddings = out["slot_embeddings"]
        slot_mean = slot_embeddings.mean(dim=1)
        slot_max = slot_embeddings.max(dim=1).values
        slot_attn_pool, slot_readout_weights = self.slot_readout(slot_embeddings)
        z_image = torch.cat([slot_mean, slot_max, slot_attn_pool], dim=-1)
        logits = self.classifier(z_image)
        z_proj = F.normalize(self.projection_head(z_image), dim=1, eps=1e-8)

        aux = dict(out.get("aux", {}))
        aux.update(
            {
                "slot_readout_weights": slot_readout_weights,
                "z_image_norm_mean": z_image.detach().norm(dim=1).mean(),
                "z_image_norm_std": z_image.detach().norm(dim=1).std(unbiased=False),
                "z_proj_norm_mean": z_proj.detach().norm(dim=1).mean(),
                "z_proj_norm_std": z_proj.detach().norm(dim=1).std(unbiased=False),
                "supcon_scope": "image_level_slot_representation_only",
                "no_motif_level_supcon": True,
                "no_prototype": True,
            }
        )
        return {
            "logits": logits,
            "z_image": z_image,
            "z_proj": z_proj,
            "slot_embeddings": slot_embeddings,
            "slot_attention": out["slot_attention"],
            "aux": aux,
            "diagnostics": dict(out.get("diagnostics", {})),
        }
