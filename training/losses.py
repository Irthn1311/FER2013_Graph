"""Losses for D5 class-level graph retrieval."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights(
    class_counts,
    normalize_mean: bool = True,
    power: float = 1.0,
) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights.pow(float(power))
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-8)
    return weights


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, y.long(), weight=weight, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        
        ce_loss = F.cross_entropy(logits, y.long(), weight=weight, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class FixedMotifClassificationLoss(nn.Module):
    """Cross-entropy loss for the D5B fixed motif classifier."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_cls = self.ce(model_out["logits"], y.long())
        return {
            "loss": loss_cls,
            "loss_cls": loss_cls,
        }


class D5RetrievalLoss(nn.Module):
    """CE plus soft-subgraph regularizers for D5A retrieval."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.lambda_cls = float(cfg.get("lambda_cls", 1.0))
        self.lambda_contrast = float(cfg.get("lambda_contrast", 0.1))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 0.01))
        self.lambda_closure = float(cfg.get("lambda_closure", 0.01))
        self.lambda_area = float(cfg.get("lambda_area", 0.01))
        self.lambda_div = float(cfg.get("lambda_div", 0.0))
        self.lambda_prior = float(cfg.get("lambda_prior", 0.0))
        self.prior_loss_type = str(cfg.get("prior_loss_type", "mse")).lower()
        self.margin = float(cfg.get("margin", 0.2))
        self.target_area = float(cfg.get("target_area", 0.15))
        if self.prior_loss_type != "mse":
            raise ValueError(f"Unsupported prior_loss_type: {self.prior_loss_type}")

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        node_attn = model_out["node_attn"]
        edge_attn = model_out["edge_attn"]
        edge_index = batch["edge_index"].long()
        y = y.long()

        loss_cls = self.ce(logits, y)
        loss_contrast = self._contrast_loss(logits, y)
        loss_smooth = self._smoothness_loss(node_attn, edge_index, y)
        loss_closure = self._closure_loss(node_attn, edge_attn, edge_index, y)
        loss_area = self._area_loss(node_attn)
        loss_div = self._diversity_loss(model_out)
        loss_prior = self._prior_loss(model_out)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_contrast * loss_contrast
            + self.lambda_smooth * loss_smooth
            + self.lambda_closure * loss_closure
            + self.lambda_area * loss_area
            + self.lambda_div * loss_div
            + self.lambda_prior * loss_prior
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_contrast": loss_contrast,
            "loss_smooth": loss_smooth,
            "loss_closure": loss_closure,
            "loss_area": loss_area,
            "loss_div": loss_div,
            "loss_prior": loss_prior,
        }

    def _contrast_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz, num_classes = logits.shape
        true_score = logits[torch.arange(bsz, device=logits.device), y]
        wrong = logits.masked_fill(F.one_hot(y, num_classes=num_classes).bool(), -1e9)
        max_wrong = wrong.max(dim=1).values
        return F.relu(self.margin - true_score + max_wrong).mean()

    @staticmethod
    def _smoothness_loss(
        node_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        true_attn = node_attn[torch.arange(bsz, device=node_attn.device), y]
        src = edge_index[0]
        dst = edge_index[1]
        return (true_attn[:, src] - true_attn[:, dst]).pow(2).mean()

    @staticmethod
    def _closure_loss(
        node_attn: torch.Tensor,
        edge_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        rows = torch.arange(bsz, device=node_attn.device)
        node_true = node_attn[rows, y]
        edge_true = edge_attn[rows, y]
        src = edge_index[0]
        dst = edge_index[1]
        endpoint_min = torch.minimum(node_true[:, src], node_true[:, dst])
        return F.relu(edge_true - endpoint_min).pow(2).mean()

    def _area_loss(self, node_attn: torch.Tensor) -> torch.Tensor:
        mass = node_attn.mean(dim=-1)
        return (mass - self.target_area).pow(2).mean()

    @staticmethod
    def _diversity_loss(model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        gate = model_out.get("class_node_gate")
        if gate is None or gate.shape[0] <= 1:
            logits = model_out["logits"]
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        flat = F.normalize(gate.float(), dim=1, eps=1e-6)
        sim = flat @ flat.t()
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def _prior_loss(self, model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model_out["logits"]
        if self.lambda_prior <= 0.0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        gate = model_out.get("class_node_gate")
        prior = model_out.get("motif_node_prior")
        if gate is None:
            raise KeyError("lambda_prior > 0 requires model_out['class_node_gate']")
        if prior is None:
            raise KeyError("lambda_prior > 0 requires a loaded motif_node_prior")
        prior = prior.to(device=gate.device, dtype=gate.dtype)
        if tuple(prior.shape) != tuple(gate.shape):
            raise ValueError(f"Prior shape {tuple(prior.shape)} does not match gate shape {tuple(gate.shape)}")
        return F.mse_loss(gate, prior)


class D6HierarchicalMotifLoss(nn.Module):
    """CE plus soft-slot regularizers for D6A hierarchical motifs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.lambda_cls = float(cfg.get("lambda_cls", 1.0))
        self.lambda_slot_div = float(cfg.get("lambda_slot_div", 0.01))
        self.lambda_border = float(cfg.get("lambda_border", 0.005))
        self.lambda_slot_balance = float(cfg.get("lambda_slot_balance", 0.0))
        self.lambda_slot_smooth = float(cfg.get("lambda_slot_smooth", 0.0))
        self.border_width = int(cfg.get("border_width", 3))
        self.border_loss_type = str(cfg.get("border_loss_type", "pixel_mean")).lower()
        self.slot_balance_type = str(cfg.get("slot_balance_type", "kl_uniform")).lower()
        self.height = int(cfg.get("height", 48))
        self.width = int(cfg.get("width", 48))
        self.eps = float(cfg.get("eps", 1e-6))

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        gamma = float(cfg.get("focal_gamma", 0.0))
        if gamma > 0.0:
            self.ce = FocalLoss(class_weights, gamma=gamma, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        else:
            self.ce = WeightedCrossEntropy(class_weights, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        self.register_buffer("border_mask", self._make_border_mask(), persistent=False)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        part_masks = model_out["part_masks"]
        loss_ce = self.ce(logits, y.long())
        loss_slot_div = self._slot_diversity_loss(part_masks)
        loss_border = self._border_loss(part_masks)
        loss_slot_balance = self._slot_balance_loss(part_masks)
        loss_slot_smooth = self._slot_smoothness_loss(part_masks, batch.get("edge_index"))
        total = (
            self.lambda_cls * loss_ce
            + self.lambda_slot_div * loss_slot_div
            + self.lambda_border * loss_border
            + self.lambda_slot_balance * loss_slot_balance
            + self.lambda_slot_smooth * loss_slot_smooth
        )
        return {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_slot_div": loss_slot_div,
            "loss_border": loss_border,
            "loss_slot_balance": loss_slot_balance,
            "loss_slot_smooth": loss_slot_smooth,
        }

    def _slot_diversity_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        m = part_masks / part_masks.norm(dim=2, keepdim=True).clamp_min(self.eps)
        sim = torch.bmm(m, m.transpose(1, 2))
        k = sim.shape[1]
        off_diag = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0))
        return off_diag.mean()

    def _border_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        border_mask = self.border_mask.to(device=part_masks.device, dtype=part_masks.dtype)
        if self.border_loss_type in ("pixel_mean", "d6a_pixel_mean"):
            return (part_masks * border_mask.view(1, 1, -1)).mean()
        if self.border_loss_type in ("slot_ratio", "slot_border_ratio"):
            border_mass = (part_masks * border_mask.view(1, 1, -1)).sum(dim=2)
            slot_mass = part_masks.sum(dim=2).clamp_min(self.eps)
            return (border_mass / slot_mass).mean()
        if self.border_loss_type in ("dominant", "dominant_border"):
            dominant = part_masks.max(dim=1).values
            border_pixels = border_mask.bool()
            if not bool(border_pixels.any()):
                return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
            return dominant[:, border_pixels].mean()
        raise ValueError(f"Unsupported border_loss_type: {self.border_loss_type}")

    def _slot_balance_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        if self.lambda_slot_balance <= 0.0:
            return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
        if self.slot_balance_type != "kl_uniform":
            raise ValueError(f"Unsupported slot_balance_type: {self.slot_balance_type}")
        area = part_masks.mean(dim=2)
        area_norm = area / area.sum(dim=1, keepdim=True).clamp_min(self.eps)
        k = area_norm.shape[1]
        return (area_norm * (area_norm.clamp_min(self.eps).log() + math.log(float(k)))).sum(dim=1).mean()

    def _slot_smoothness_loss(
        self,
        part_masks: torch.Tensor,
        edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.lambda_slot_smooth <= 0.0 or edge_index is None:
            return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        return (part_masks[:, :, src] - part_masks[:, :, dst]).abs().mean()

    def _make_border_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(self.border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)


class D6ClassAttendedMotifLoss(D6HierarchicalMotifLoss):
    """D6B loss plus class-conditioned objectives for D6C-light."""

    DEFAULT_CONFUSION_PAIRS = ((2, 4), (2, 6), (2, 5), (4, 6), (0, 1))
    PAIR_DIAG_NAMES = {
        (2, 4): "fear_sad",
        (2, 6): "fear_neutral",
        (2, 5): "fear_surprise",
        (4, 6): "sad_neutral",
        (0, 1): "angry_disgust",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        super().__init__(cfg)
        self.lambda_class_border = float(cfg.get("lambda_class_border", 0.0025))
        self.lambda_class_attn_sep = float(cfg.get("lambda_class_attn_sep", 0.005))
        self.lambda_supcon = float(cfg.get("lambda_supcon", 0.03))
        self.class_attn_sep_margin = float(cfg.get("class_attn_sep_margin", 0.90))
        self.supcon_temperature = float(cfg.get("supcon_temperature", 0.2))
        self.confusion_pairs = self._parse_confusion_pairs(cfg.get("confusion_pairs", self.DEFAULT_CONFUSION_PAIRS))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        part_masks = model_out["part_masks"]
        y = y.long()

        loss_ce = self.ce(logits, y)
        loss_slot_div = self._slot_diversity_loss(part_masks)
        loss_border = self._border_loss(part_masks)
        loss_slot_balance = self._slot_balance_loss(part_masks)
        loss_slot_smooth = self._slot_smoothness_loss(part_masks, batch.get("edge_index"))
        loss_class_border, class_border_diag = self._class_attended_border_loss(model_out, y)
        loss_class_attn_sep, class_attn_diag = self._class_attention_separation_loss(model_out)
        loss_supcon, supcon_diag = self._supervised_contrastive_loss(model_out, y)

        total = (
            self.lambda_cls * loss_ce
            + self.lambda_slot_div * loss_slot_div
            + self.lambda_border * loss_border
            + self.lambda_slot_balance * loss_slot_balance
            + self.lambda_slot_smooth * loss_slot_smooth
            + self.lambda_class_border * loss_class_border
            + self.lambda_class_attn_sep * loss_class_attn_sep
            + self.lambda_supcon * loss_supcon
        )

        out = {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_slot_div": loss_slot_div,
            "loss_border": loss_border,
            "loss_slot_balance": loss_slot_balance,
            "loss_slot_smooth": loss_slot_smooth,
            "loss_class_border": loss_class_border,
            "loss_class_attn_sep": loss_class_attn_sep,
            "loss_supcon": loss_supcon,
        }
        out.update(class_border_diag)
        out.update(class_attn_diag)
        out.update(supcon_diag)
        return out

    def _class_attended_border_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        part_masks = model_out.get("part_masks")
        class_part_attn = model_out.get("class_part_attn")
        zero = logits.new_zeros(())
        if not torch.is_tensor(part_masks) or not torch.is_tensor(class_part_attn):
            return zero, {
                "diag_true_class_border_mass_mean": zero,
                "diag_true_class_border_mass_max": zero,
            }

        border_mask = self.border_mask.to(device=part_masks.device, dtype=part_masks.dtype)
        if border_mask.numel() != part_masks.shape[-1]:
            raise ValueError(
                f"Border mask has {border_mask.numel()} pixels, but part_masks has {part_masks.shape[-1]} nodes"
            )
        class_pixel_attn = torch.einsum("bck,bkn->bcn", class_part_attn, part_masks)
        rows = torch.arange(y.shape[0], device=y.device)
        true_motif = class_pixel_attn[rows, y, :]
        border_mass = (true_motif * border_mask.view(1, -1)).sum(dim=1)
        motif_mass = true_motif.sum(dim=1).clamp_min(self.eps)
        border_ratio = border_mass / motif_mass
        return border_ratio.mean(), {
            "diag_true_class_border_mass_mean": border_ratio.detach().mean(),
            "diag_true_class_border_mass_max": border_ratio.detach().max(),
        }

    def _class_attention_separation_loss(
        self,
        model_out: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        class_part_attn = model_out.get("class_part_attn")
        zero = logits.new_zeros(())
        diag: Dict[str, torch.Tensor] = {
            "diag_class_attn_sim_fear_sad": zero,
            "diag_class_attn_sim_fear_neutral": zero,
            "diag_class_attn_sim_fear_surprise": zero,
            "diag_class_attn_sim_sad_neutral": zero,
            "diag_class_attn_sim_angry_disgust": zero,
            "diag_class_part_entropy_mean": zero,
        }
        if not torch.is_tensor(class_part_attn):
            return zero, diag

        attn = class_part_attn.float()
        entropy = -(attn * attn.clamp_min(self.eps).log()).sum(dim=2)
        diag["diag_class_part_entropy_mean"] = entropy.detach().mean()
        avg_attn = attn.mean(dim=0)
        penalties = []
        for left, right in self.confusion_pairs:
            if avg_attn.shape[0] <= max(left, right):
                continue
            sim = F.cosine_similarity(avg_attn[left], avg_attn[right], dim=0, eps=self.eps)
            penalties.append(F.relu(sim - self.class_attn_sep_margin))
            pair_name = self.PAIR_DIAG_NAMES.get((left, right), self.PAIR_DIAG_NAMES.get((right, left)))
            if pair_name is not None:
                diag[f"diag_class_attn_sim_{pair_name}"] = sim.detach()

        if not penalties or self.lambda_class_attn_sep <= 0.0:
            return zero, diag
        return torch.stack(penalties).mean().to(dtype=logits.dtype), diag

    def _supervised_contrastive_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        class_repr = model_out.get("class_repr")
        zero = logits.new_zeros(())
        diag = {
            "diag_supcon_valid_anchors": zero,
            "diag_supcon_positive_pairs": zero,
        }
        if not torch.is_tensor(class_repr) or class_repr.shape[0] <= 1:
            return zero, diag
        if self.lambda_supcon <= 0.0:
            return zero, diag

        rows = torch.arange(y.shape[0], device=y.device)
        z = class_repr[rows, y, :].float()
        z = F.normalize(z, dim=-1, eps=self.eps)
        temperature = max(self.supcon_temperature, self.eps)
        sim = (z @ z.transpose(0, 1)) / temperature
        bsz = sim.shape[0]
        self_mask = torch.eye(bsz, dtype=torch.bool, device=sim.device)
        positive_mask = y.view(-1, 1).eq(y.view(1, -1)) & ~self_mask
        valid_anchor = positive_mask.any(dim=1)
        if not bool(valid_anchor.any()):
            return zero, diag

        non_self_logits = sim.masked_fill(self_mask, -float("inf"))
        max_logits = non_self_logits.max(dim=1, keepdim=True).values
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        stable_logits = sim - max_logits.detach()
        exp_logits = stable_logits.exp().masked_fill(self_mask, 0.0)
        log_prob = stable_logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps).log()
        pos_float = positive_mask.to(dtype=log_prob.dtype)
        pos_count = pos_float.sum(dim=1).clamp_min(1.0)
        per_anchor = -(log_prob * pos_float).sum(dim=1) / pos_count
        loss = per_anchor[valid_anchor].mean()
        diag["diag_supcon_valid_anchors"] = valid_anchor.float().sum().detach()
        diag["diag_supcon_positive_pairs"] = positive_mask.float().sum().detach()
        return loss.to(dtype=logits.dtype), diag

    @staticmethod
    def _parse_confusion_pairs(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
        pairs = []
        for item in value:
            if len(item) != 2:
                raise ValueError(f"Each confusion pair must have two class indices, got {item}")
            pairs.append((int(item[0]), int(item[1])))
        return tuple(pairs)


class D7DualBranchMotifLoss(D6HierarchicalMotifLoss):
    """D7 fused CE, optional branch auxiliary CE, and D6B slot regularizers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        super().__init__(cfg)
        self.lambda_aux_d6 = float(cfg.get("lambda_aux_d6", 0.2))
        self.lambda_aux_swin = float(cfg.get("lambda_aux_swin", 0.2))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        y = y.long()
        zero = logits.new_zeros(())

        loss_ce = self.ce(logits, y)
        loss_aux_d6 = self._optional_ce(model_out.get("logits_d6"), y)
        loss_aux_swin = self._optional_ce(model_out.get("logits_swin"), y)

        part_masks = model_out.get("part_masks")
        if torch.is_tensor(part_masks):
            loss_slot_div = self._slot_diversity_loss(part_masks)
            loss_border = self._border_loss(part_masks)
            loss_slot_balance = self._slot_balance_loss(part_masks)
            loss_slot_smooth = self._slot_smoothness_loss(part_masks, batch.get("edge_index"))
        else:
            loss_slot_div = zero
            loss_border = zero
            loss_slot_balance = zero
            loss_slot_smooth = zero

        total = (
            self.lambda_cls * loss_ce
            + self.lambda_aux_d6 * loss_aux_d6
            + self.lambda_aux_swin * loss_aux_swin
            + self.lambda_slot_div * loss_slot_div
            + self.lambda_border * loss_border
            + self.lambda_slot_balance * loss_slot_balance
            + self.lambda_slot_smooth * loss_slot_smooth
        )
        out = {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_aux_d6": loss_aux_d6,
            "loss_aux_swin": loss_aux_swin,
            "loss_slot_div": loss_slot_div,
            "loss_border": loss_border,
            "loss_slot_balance": loss_slot_balance,
            "loss_slot_smooth": loss_slot_smooth,
        }
        out.update(self._branch_accuracy_diagnostics(model_out, y))
        return out

    def _optional_ce(self, logits: Optional[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(logits):
            return y.new_zeros((), dtype=torch.float32)
        return self.ce(logits, y)

    @staticmethod
    def _branch_accuracy_diagnostics(
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        diag: Dict[str, torch.Tensor] = {}
        for key, out_key in (
            ("diag_logits_d6_accuracy", "logits_d6"),
            ("diag_logits_swin_accuracy", "logits_swin"),
            ("diag_logits_fused_accuracy", "logits"),
        ):
            logits = model_out.get(out_key)
            if torch.is_tensor(logits):
                diag[key] = (logits.argmax(dim=1) == y).float().mean().detach()
        return diag


class D8APrePartMotifLoss(D6HierarchicalMotifLoss):
    """D6B motif loss plus optional tiny context-alpha regularization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        super().__init__(cfg)
        self.lambda_context_alpha_l2 = float(cfg.get("context_alpha_l2_weight", cfg.get("lambda_context_alpha_l2", 0.0)))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(model_out, y, batch)
        logits = model_out["logits"]
        alpha = model_out.get("context_alpha")
        if torch.is_tensor(alpha):
            loss_context_alpha_l2 = alpha.to(device=logits.device, dtype=logits.dtype).pow(2).mean()
        else:
            loss_context_alpha_l2 = logits.new_zeros(())
        total = out["loss"] + self.lambda_context_alpha_l2 * loss_context_alpha_l2
        out["loss"] = total
        out["total_loss"] = total
        out["loss_context_alpha_l2"] = loss_context_alpha_l2
        return out


class D8BFaceAwareLoss(nn.Module):
    """CE plus light pixel-gate regularizers for D8B."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.ce_weight = float(cfg.get("ce_weight", cfg.get("lambda_cls", 1.0)))
        self.lambda_area = float(cfg.get("pixel_gate_area_weight", 0.01))
        self.target_area = float(cfg.get("pixel_gate_area_target", 0.55))
        self.lambda_border = float(cfg.get("pixel_gate_border_weight", 0.01))
        self.border_width = int(cfg.get("pixel_gate_border_width", cfg.get("border_width", 3)))
        self.lambda_smooth = float(cfg.get("pixel_gate_smooth_weight", 0.001))
        self.height = int(cfg.get("height", 48))
        self.width = int(cfg.get("width", 48))
        self.eps = float(cfg.get("eps", 1e-6))
        self.current_epoch = 0
        self._warned_missing_gate = False

        attn_cfg = dict(cfg.get("attention_regularization", {}) or {})
        self.attn_reg_enabled = bool(attn_cfg.get("enabled", False))
        self.attn_reg_map_name = str(attn_cfg.get("map_name", "pixel_gate"))
        self.attn_reg_use_upper_floor = bool(attn_cfg.get("use_upper_floor", True))
        self.attn_reg_target_upper = float(attn_cfg.get("target_upper", 0.38))
        self.attn_reg_hard_classes = tuple(int(idx) for idx in attn_cfg.get("hard_class_indices", (0, 2, 4, 5)))
        self.attn_reg_lambda_upper = float(attn_cfg.get("lambda_upper", 0.0))
        self.attn_reg_region_tau = float(attn_cfg.get("region_tau", 0.05))
        self.attn_reg_start_epoch = int(attn_cfg.get("start_epoch", 1))
        self.attn_reg_ramp_epochs = int(attn_cfg.get("ramp_epochs", 0))
        self.attn_reg_decay_enabled = bool(attn_cfg.get("decay_enabled", False))
        self.attn_reg_decay_start_epoch = int(attn_cfg.get("decay_start_epoch", 10**9))
        self.attn_reg_decay_end_epoch = int(attn_cfg.get("decay_end_epoch", self.attn_reg_decay_start_epoch))
        self.attn_reg_min_lambda_after_decay = float(attn_cfg.get("min_lambda_after_decay", 0.0))

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)
        self.register_buffer("border_mask", self._make_border_mask(), persistent=False)
        upper_mask, middle_mask, lower_mask = self._make_region_masks()
        self.register_buffer("region_upper_mask", upper_mask, persistent=False)
        self.register_buffer("region_middle_mask", middle_mask, persistent=False)
        self.register_buffer("region_lower_mask", lower_mask, persistent=False)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del batch
        logits = model_out["logits"]
        loss_ce = self.ce(logits, y.long())
        zero = logits.new_zeros(())
        pixel_gate = model_out.get("pixel_gate")
        if torch.is_tensor(pixel_gate):
            loss_gate_area = (pixel_gate.mean() - self.target_area).pow(2)
            loss_gate_border = self._border_loss(pixel_gate)
            loss_gate_smooth = self._smoothness_loss(pixel_gate)
            region_metrics = self._attention_region_metrics(pixel_gate, y.long())
        else:
            if self.attn_reg_enabled:
                raise KeyError("attention_regularization.enabled=true requires model_out['pixel_gate']")
            if not self._warned_missing_gate:
                warnings.warn("D8BFaceAwareLoss did not find model_out['pixel_gate']; gate losses are zero.")
                self._warned_missing_gate = True
            loss_gate_area = zero
            loss_gate_border = zero
            loss_gate_smooth = zero
            region_metrics = self._zero_attention_region_metrics(zero)

        current_lambda_upper = logits.new_tensor(self._lambda_upper_for_epoch(self.current_epoch))
        loss_upper_floor = region_metrics["loss_upper_floor_raw"]
        loss_upper_weighted = current_lambda_upper * loss_upper_floor

        total = (
            self.ce_weight * loss_ce
            + self.lambda_area * loss_gate_area
            + self.lambda_border * loss_gate_border
            + self.lambda_smooth * loss_gate_smooth
            + loss_upper_weighted
        )
        out = {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_gate_area": loss_gate_area,
            "loss_gate_border": loss_gate_border,
            "loss_gate_smooth": loss_gate_smooth,
            "loss_upper_floor": loss_upper_floor,
            "loss_upper_floor_weighted": loss_upper_weighted,
            "current_lambda_upper": current_lambda_upper,
        }
        out.update(region_metrics)
        return out

    def _lambda_upper_for_epoch(self, epoch: int) -> float:
        if not self.attn_reg_enabled or not self.attn_reg_use_upper_floor:
            return 0.0
        epoch = int(epoch)
        base = max(float(self.attn_reg_lambda_upper), 0.0)
        if epoch < self.attn_reg_start_epoch:
            return 0.0
        if self.attn_reg_ramp_epochs > 0:
            ramp_pos = (epoch - self.attn_reg_start_epoch + 1) / float(self.attn_reg_ramp_epochs)
            value = base * min(max(ramp_pos, 0.0), 1.0)
        else:
            value = base
        if self.attn_reg_decay_enabled and epoch >= self.attn_reg_decay_start_epoch:
            start = self.attn_reg_decay_start_epoch
            end = max(self.attn_reg_decay_end_epoch, start)
            if epoch >= end:
                return float(self.attn_reg_min_lambda_after_decay)
            decay_pos = (epoch - start) / max(float(end - start), 1.0)
            value = value + (float(self.attn_reg_min_lambda_after_decay) - value) * min(max(decay_pos, 0.0), 1.0)
        return float(value)

    def _attention_region_metrics(self, pixel_gate: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pixel_gate.shape[1] != self.height * self.width:
            raise ValueError(
                f"attention_regularization expects pixel_gate with {self.height * self.width} nodes, "
                f"got {pixel_gate.shape[1]}"
            )
        gate = pixel_gate.squeeze(-1).float().clamp_min(0.0)
        probs = gate / gate.sum(dim=1, keepdim=True).clamp_min(self.eps)
        upper_mask = self.region_upper_mask.to(device=probs.device, dtype=probs.dtype)
        middle_mask = self.region_middle_mask.to(device=probs.device, dtype=probs.dtype)
        lower_mask = self.region_lower_mask.to(device=probs.device, dtype=probs.dtype)
        upper_mass = (probs * upper_mask.view(1, -1)).sum(dim=1)
        middle_mass = (probs * middle_mask.view(1, -1)).sum(dim=1)
        lower_mass = (probs * lower_mask.view(1, -1)).sum(dim=1)

        if self.attn_reg_hard_classes:
            hard_class_tensor = torch.as_tensor(self.attn_reg_hard_classes, device=y.device, dtype=y.dtype)
            hard_mask = (y.view(-1, 1) == hard_class_tensor.view(1, -1)).any(dim=1)
        else:
            hard_mask = torch.zeros_like(y, dtype=torch.bool)

        if bool(hard_mask.any()) and self.attn_reg_use_upper_floor:
            hard_upper = upper_mass[hard_mask]
            loss_upper = F.relu(float(self.attn_reg_target_upper) - hard_upper).mean().to(dtype=pixel_gate.dtype)
            hard_upper_mean = hard_upper.mean().detach().to(dtype=pixel_gate.dtype)
        else:
            loss_upper = pixel_gate.new_zeros(())
            hard_upper_mean = pixel_gate.new_zeros(())

        return {
            "loss_upper_floor_raw": loss_upper,
            "pixel_gate_upper_mass_hard_classes": hard_upper_mean,
            "pixel_gate_hard_class_count": hard_mask.float().sum().detach().to(dtype=pixel_gate.dtype),
            "pixel_gate_upper_mass_all": upper_mass.mean().detach().to(dtype=pixel_gate.dtype),
            "pixel_gate_middle_mass_all": middle_mass.mean().detach().to(dtype=pixel_gate.dtype),
            "pixel_gate_lower_mass_all": lower_mass.mean().detach().to(dtype=pixel_gate.dtype),
        }

    @staticmethod
    def _zero_attention_region_metrics(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "loss_upper_floor_raw": zero,
            "pixel_gate_upper_mass_hard_classes": zero,
            "pixel_gate_hard_class_count": zero,
            "pixel_gate_upper_mass_all": zero,
            "pixel_gate_middle_mass_all": zero,
            "pixel_gate_lower_mass_all": zero,
        }

    def _border_loss(self, pixel_gate: torch.Tensor) -> torch.Tensor:
        border = self.border_mask.to(device=pixel_gate.device, dtype=torch.bool)
        if border.numel() != pixel_gate.shape[1] or not bool(border.any()):
            return pixel_gate.new_zeros(())
        return pixel_gate[:, border, :].mean()

    def _smoothness_loss(self, pixel_gate: torch.Tensor) -> torch.Tensor:
        if self.lambda_smooth <= 0.0:
            return pixel_gate.new_zeros(())
        if pixel_gate.shape[1] != self.height * self.width:
            return pixel_gate.new_zeros(())
        grid = pixel_gate.view(pixel_gate.shape[0], self.height, self.width, 1)
        dy = (grid[:, 1:, :, :] - grid[:, :-1, :, :]).abs().mean()
        dx = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().mean()
        return 0.5 * (dx + dy)

    def _make_border_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(self.border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)

    def _make_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = max(float(self.attn_reg_region_tau), 1e-8)
        y = torch.linspace(0.0, 1.0, self.height, dtype=torch.float32).view(self.height, 1)
        upper = torch.sigmoid((0.42 - y) / tau).expand(self.height, self.width)
        middle = (torch.sigmoid((y - 0.30) / tau) * torch.sigmoid((0.70 - y) / tau)).expand(self.height, self.width)
        lower = torch.sigmoid((y - 0.55) / tau).expand(self.height, self.width)
        return upper.reshape(-1), middle.reshape(-1), lower.reshape(-1)



class D10SlotMotifLoss(D6HierarchicalMotifLoss):
    """D10 loss: CE + slot diversity + border + balance + optional auxiliary CE.

    Reuses D6B slot regularizers. Adds auxiliary CE from pooled motif
    representations to ensure motifs carry classification signal early.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        super().__init__(cfg)
        self.lambda_aux_ce = float(cfg.get("lambda_aux_ce", 0.3))
        self.lambda_supcon = float(cfg.get("lambda_supcon", 0.0))
        self.supcon_temperature = float(cfg.get("supcon_temperature", 0.1))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(model_out, y, batch)
        y = y.long()

        # Auxiliary CE from pooled motif classifier
        aux_logits = model_out.get("aux_logits")
        if torch.is_tensor(aux_logits) and self.lambda_aux_ce > 0.0:
            loss_aux_ce = self.ce(aux_logits, y)
        else:
            loss_aux_ce = out["loss"].new_zeros(())

        total = out["loss"] + self.lambda_aux_ce * loss_aux_ce
        out["loss_aux_ce"] = loss_aux_ce

        # 3. Supervised Contrastive Loss (SupCon) on Motifs
        motif_embeddings = model_out.get("motif_embeddings")
        if torch.is_tensor(motif_embeddings) and self.lambda_supcon > 0.0 and y.shape[0] > 1:
            z = motif_embeddings.mean(dim=1)  # [B, D]
            z = F.normalize(z, dim=-1, eps=self.eps)
            temperature = max(self.supcon_temperature, self.eps)
            sim = (z @ z.transpose(0, 1)) / temperature
            bsz = sim.shape[0]
            self_mask = torch.eye(bsz, dtype=torch.bool, device=sim.device)
            positive_mask = y.view(-1, 1).eq(y.view(1, -1)) & ~self_mask
            valid_anchor = positive_mask.any(dim=1)

            if bool(valid_anchor.any()):
                non_self_logits = sim.masked_fill(self_mask, -float("inf"))
                max_logits = non_self_logits.max(dim=1, keepdim=True).values
                max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
                stable_logits = sim - max_logits.detach()
                exp_logits = stable_logits.exp().masked_fill(self_mask, 0.0)
                log_prob = stable_logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps).log()

                pos_float = positive_mask.to(dtype=log_prob.dtype)
                pos_count = pos_float.sum(dim=1).clamp_min(1.0)
                per_anchor = -(log_prob * pos_float).sum(dim=1) / pos_count
                loss_supcon = per_anchor[valid_anchor].mean()
            else:
                loss_supcon = out["loss"].new_zeros(())
        else:
            loss_supcon = out["loss"].new_zeros(())

        total = total + self.lambda_supcon * loss_supcon
        out["loss"] = total
        out["total_loss"] = total
        out["loss_supcon"] = loss_supcon

        # Add accuracy diagnostics
        logits = model_out.get("logits")
        if torch.is_tensor(logits):
            out["diag_main_accuracy"] = (logits.argmax(dim=1) == y).float().mean().detach()
        if torch.is_tensor(aux_logits):
            out["diag_aux_accuracy"] = (aux_logits.argmax(dim=1) == y).float().mean().detach()

        return out


def build_loss(config: Dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    name = str(cfg.get("name", "d5_retrieval")).lower()
    if name in ("d5_retrieval", "class_pixel_motif_retrieval"):
        return D5RetrievalLoss(cfg)
    if name in ("d6_hierarchical_motif", "d6a_hierarchical_motif"):
        return D6HierarchicalMotifLoss(cfg)
    if name in ("d6b_class_part_motif", "d6b_class_part_attention_motif"):
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        return D6HierarchicalMotifLoss(cfg)
    if name in ("d6c_class_attended_motif", "d6c_class_attended_objectives"):
        return D6ClassAttendedMotifLoss(cfg)
    if name in ("d7_dual_branch_motif", "d7_dual_branch_graph_swin_motif"):
        return D7DualBranchMotifLoss(cfg)
    if name in ("d8a_prepart_motif_loss", "d8a_prepart_motif", "graph_swin_prepart_d6b"):
        return D8APrePartMotifLoss(cfg)
    if name in ("d8b_face_aware_loss", "face_aware_graph_swin_d8b"):
        return D8BFaceAwareLoss(cfg)
    if name in ("fixed_motif_classification", "d5b_fixed_motif"):
        return FixedMotifClassificationLoss(cfg)
    if name in ("d10_slot_motif", "d10_slot_motif_loss"):
        return D10SlotMotifLoss(cfg)
    if name in ("d11_global_local", "d11_global_local_loss"):
        return D11GlobalLocalLoss(cfg)
    raise ValueError(f"Unknown loss: {name}")

class D11GlobalLocalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = dict(config)
        self.lambda_local_base = float(cfg.get('lambda_local', 0.5))
        self.lambda_spatial = float(cfg.get('lambda_spatial', 1.0))
        self.lambda_supcon_base = float(cfg.get('lambda_supcon', 0.1))
        self.lambda_div = float(cfg.get('lambda_div', 0.05))
        self.diversity_margin = float(cfg.get('diversity_margin', 0.3))
        self.warmup_epochs = float(cfg.get('warmup_epochs', 5.0))
        self.ce_warmup_epochs = float(cfg.get('ce_warmup_epochs', 0.0))
        self.supcon_decay_epochs = float(cfg.get('supcon_decay_epochs', 0.0))
        self.current_epoch = 0.0
        
        label_smoothing = float(cfg.get('label_smoothing', 0.1))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        from training.supcon_loss import SupervisedContrastiveLoss
        self.supcon = SupervisedContrastiveLoss(temperature=0.1)

    def set_epoch(self, epoch: float):
        self.current_epoch = float(epoch)

    def forward(self, model_out, y, batch):
        import torch.nn.functional as F
        y = y.long()
        logits = model_out.get('logits')
        loss_fusion_raw = self.ce(logits, y)

        # CE Scheduling (Pseudo 2-Stage)
        if self.ce_warmup_epochs > 0:
            if self.current_epoch < self.ce_warmup_epochs:
                # Trễ hoàn toàn CE ở nửa đầu (Hard Freeze) hoặc Tăng dần (Soft Warmup)
                # Dùng Linear Soft Warmup:
                ce_factor = self.current_epoch / max(1.0, self.ce_warmup_epochs)
            else:
                ce_factor = 1.0
        else:
            ce_factor = 1.0
            
        loss_fusion = loss_fusion_raw * ce_factor
        lambda_local = self.lambda_local_base * ce_factor

        logits_local = model_out.get('logits_local')
        if logits_local is not None and lambda_local > 0.0:
            loss_local = self.ce(logits_local, y)
        else:
            loss_local = logits.new_zeros(())

        center_of_mass = model_out.get('center_of_mass') # [B, K, 2]
        if center_of_mass is not None and self.lambda_spatial > 0.0:
            cx = center_of_mass[:, :, 0]
            cy = center_of_mass[:, :, 1]
            loss_sp_01_x = F.relu(cx[:, 0:2] - 0.5).mean()
            loss_sp_01_y = F.relu(cy[:, 0:2] - 0.5).mean()
            loss_sp_23_x = F.relu(0.5 - cx[:, 2:4]).mean()
            loss_sp_23_y = F.relu(cy[:, 2:4] - 0.5).mean()
            loss_sp_45_y = F.relu(0.5 - cy[:, 4:6]).mean()
            loss_spatial = loss_sp_01_x + loss_sp_01_y + loss_sp_23_x + loss_sp_23_y + loss_sp_45_y
        else:
            loss_spatial = logits.new_zeros(())

        # SupCon Scheduling
        local_raw_proj = model_out.get('local_raw_proj')
        supcon_warmup_factor = min(1.0, max(1.0, self.current_epoch) / max(1.0, self.warmup_epochs))
        
        # Optionally decay SupCon after motifs are formed
        if self.supcon_decay_epochs > 0 and self.current_epoch > self.supcon_decay_epochs:
            decay_factor = max(0.0, 1.0 - (self.current_epoch - self.supcon_decay_epochs) / 20.0)
        else:
            decay_factor = 1.0
            
        effective_lambda_supcon = self.lambda_supcon_base * supcon_warmup_factor * decay_factor
        
        loss_supcon_val = logits.new_zeros(())
        if local_raw_proj is not None and effective_lambda_supcon > 0.0:
            loss_supcon_val = self.supcon(local_raw_proj, y)
            loss_supcon = loss_supcon_val * effective_lambda_supcon
        else:
            loss_supcon = logits.new_zeros(())
            
        # Slot Diversity Loss
        local_raw = model_out.get('local_raw') # [B, K, D]
        if local_raw is not None and self.lambda_div > 0.0:
            z_norm = F.normalize(local_raw, dim=-1) # [B, K, D]
            sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2)) # [B, K, K]
            K = local_raw.size(1)
            eye = torch.eye(K, device=local_raw.device).unsqueeze(0)
            masked_sim = sim_matrix - eye * 999.0
            loss_div = F.relu(masked_sim - self.diversity_margin).mean()
        else:
            loss_div = logits.new_zeros(())

        total = loss_fusion + lambda_local * loss_local + self.lambda_spatial * loss_spatial + loss_supcon + self.lambda_div * loss_div

        out = {
            'loss': total,
            'loss_fusion': loss_fusion,
            'loss_local_aux': lambda_local * loss_local,
            'loss_spatial_prior': loss_spatial,
            'loss_supcon': loss_supcon,
            'loss_supcon_raw': loss_supcon_val,
            'effective_lambda_supcon': logits.new_tensor(effective_lambda_supcon),
            'effective_ce_factor': logits.new_tensor(ce_factor),
            'loss_div': loss_div,
            'diag_main_accuracy': (logits.argmax(dim=1) == y).float().mean().detach()
        }
        if logits_local is not None:
            out['diag_local_accuracy'] = (logits_local.argmax(dim=1) == y).float().mean().detach()
        return out
