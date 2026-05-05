"""Stage 1B pretraining for MotifDiscoveryModule anti-collapse objectives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path, save_config  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "split",
    "total_loss",
    "loss_map_diversity",
    "loss_embedding_diversity",
    "loss_coverage",
    "loss_anchor",
    "loss_border",
    "loss_soft_border",
    "loss_outer_border",
    "loss_selected_border",
    "loss_selected_outer_border",
    "loss_selected_foreground",
    "loss_selected_diversity",
    "loss_selection_entropy",
    "loss_clean_count",
    "loss_clean_mean",
    "loss_region_clean",
    "loss_region_clean_upper_component",
    "loss_region_clean_middle_component",
    "loss_region_clean_lower_component",
    "loss_teacher_align",
    "loss_aux_ce",
    "loss_entropy",
    "loss_area_high",
    "loss_area_low",
    "loss_area_balance",
    "assignment_sum_mean",
    "assignment_sum_max_abs_err",
    "mean_pairwise_map_sim",
    "max_pairwise_map_sim",
    "mean_pairwise_emb_sim",
    "max_pairwise_emb_sim",
    "mean_center_dist",
    "min_center_dist",
    "redundant_pair_count",
    "redundant_pair_ratio",
    "border_mass_mean",
    "center_mass_mean",
    "center_minus_border",
    "border_dominance",
    "soft_border_mass_mean",
    "outer_border_mass_mean",
    "selected_border_mass_mean",
    "selected_outer_border_mass_mean",
    "selected_foreground_mass_mean",
    "selection_entropy",
    "selection_entropy_soft",
    "selection_effective_count",
    "selected_effective_count",
    "selected_pairwise_map_sim",
    "selected_pairwise_emb_sim",
    "selected_top_m",
    "target_selection_entropy_min",
    "clean_score_mean",
    "clean_score_max",
    "clean_candidate_count",
    "hard_clean_candidate_count",
    "clean_border_score_mean",
    "clean_outer_score_mean",
    "clean_foreground_score_mean",
    "clean_area_low_score_mean",
    "clean_area_high_score_mean",
    "clean_count_target",
    "motif_upper_mass_mean",
    "motif_middle_mass_mean",
    "motif_lower_mass_mean",
    "upper_clean_count",
    "middle_clean_count",
    "lower_clean_count",
    "min_region_clean_count",
    "mean_region_clean_count",
    "region_clean_target",
    "region_clean_upper_weight",
    "region_clean_middle_weight",
    "region_clean_lower_weight",
    "foreground_center_mass",
    "foreground_border_mass",
    "foreground_prior_sum",
    "foreground_safe_mass",
    "effective_area_mean",
    "effective_area_min",
    "effective_area_max",
    "area_in_range_ratio",
    "area_over_max_ratio",
    "area_under_min_ratio",
    "effective_area_over_max_ratio",
    "effective_area_under_min_ratio",
    "assignment_entropy_mean",
    "effective_motif_count",
    "effective_motif_ratio",
    "coverage_cosine",
    "foreground_mass_mean",
    "motif_foreground_mass_mean",
    "motif_foreground_mass_min",
    "motif_foreground_mass_max",
    "logits_mean",
    "logits_std",
    "logits_min",
    "logits_max",
    "attention_temperature",
    "spatial_bias_strength_current",
    "current_lambda_div_map",
    "current_lambda_div_emb",
    "current_lambda_coverage",
    "current_lambda_anchor",
    "current_lambda_border",
    "current_lambda_soft_border",
    "current_lambda_outer_border",
    "current_lambda_selected_border",
    "current_lambda_selected_outer_border",
    "current_lambda_selected_foreground",
    "current_lambda_selected_diversity",
    "current_lambda_selection_entropy",
    "current_lambda_clean_count",
    "current_lambda_clean_mean",
    "current_lambda_region_clean",
    "current_lambda_teacher_align",
    "current_lambda_aux_ce",
    "current_lambda_entropy",
    "current_lambda_area",
    "coverage_boost_active",
    "soft_border_boost_active",
    "teacher_alignment_active",
    "teacher_decay_active",
    "teacher_confidence_mean",
    "teacher_pred_acc_batch",
    "teacher_saliency_border_mass",
    "teacher_saliency_center_mass",
    "filtered_teacher_saliency_border_mass",
    "filtered_teacher_saliency_center_mass",
    "teacher_motif_cosine",
    "teacher_motif_iou_top20",
    "motif_initial_center_min",
    "motif_initial_center_max",
    "center_prior_sigma",
    "center_prior_power",
    "foreground_face_safe_margin",
    "foreground_face_safe_tau",
    "q_area_in_range_component",
    "q_effective_motif_component",
    "q_coverage_component",
    "q_foreground_mass_component",
    "q_selected_foreground_component",
    "q_selection_entropy_component",
    "q_clean_candidate_component",
    "q_region_clean_mean_component",
    "q_region_clean_min_component",
    "q_teacher_motif_cosine_component",
    "q_aux_component",
    "q_center_border_component",
    "q_selected_border_penalty",
    "q_selected_outer_border_penalty",
    "q_border_dominance_penalty",
    "q_outer_border_penalty",
    "q_redundancy_penalty",
    "q_map_sim_penalty",
    "q_area_over_penalty",
    "q_area_under_penalty",
    "selection_entropy_score",
    "clean_candidate_score_for_quality",
    "region_clean_mean_score_for_quality",
    "region_clean_min_score_for_quality",
    "motif_quality_score",
    "val_motif_quality_score",
    "aux_accuracy",
    "aux_macro_f1",
    "aux_pred_count",
    "aux_confidence_mean",
    "aux_entropy_mean",
]


def _make_grid_edges(height: int = 48, width: int = 48) -> torch.Tensor:
    edges = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x + 1 < width:
                right = y * width + x + 1
                edges.append((idx, right))
                edges.append((right, idx))
            if y + 1 < height:
                down = (y + 1) * width + x
                edges.append((idx, down))
                edges.append((down, idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _synthetic_loader(batch_size: int, num_batches: int, device: torch.device) -> Iterable[Dict[str, torch.Tensor]]:
    height = width = 48
    node_dim = 7
    edge_dim = 5
    edge_index = _make_grid_edges(height, width).to(device)
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.exp(-3.0 * (xx.square() + yy.square())).flatten()
    for batch_idx in range(int(num_batches)):
        x = torch.randn(batch_size, height * width, node_dim, device=device) * 0.05
        x[..., 0] = base.unsqueeze(0) + 0.05 * torch.randn(batch_size, height * width, device=device)
        x[..., 1] = ((xx + 1.0) * 0.5).flatten().unsqueeze(0)
        x[..., 2] = ((yy + 1.0) * 0.5).flatten().unsqueeze(0)
        x[..., 5] = torch.relu(base.unsqueeze(0) + 0.03 * torch.randn(batch_size, height * width, device=device))
        x[..., 6] = torch.relu(base.roll(11).unsqueeze(0) + 0.03 * torch.randn(batch_size, height * width, device=device))
        yield {
            "graph_id": torch.arange(batch_size, device=device) + batch_idx * batch_size,
            "x": x,
            "node_features": x,
            "edge_index": edge_index,
            "edge_attr": torch.randn(batch_size, edge_index.shape[1], edge_dim, device=device),
            "node_mask": torch.ones(batch_size, height * width, dtype=torch.bool, device=device),
            "y": torch.zeros(batch_size, dtype=torch.long, device=device),
        }


def _update_paths_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)
    experiment = dict(cfg.get("experiment", {}) or {})
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    output = dict(cfg.get("output", {}) or {})
    if getattr(args, "experiment_name", None):
        experiment["name"] = str(args.experiment_name)
    if getattr(args, "output_dir", None):
        paths["resolved_output_root"] = str(args.output_dir)
        output["dir"] = str(args.output_dir)
    elif output.get("dir") and not paths.get("resolved_output_root"):
        paths["resolved_output_root"] = str(output["dir"])
    if data.get("graph_repo_path") and not paths.get("graph_repo_path"):
        paths["graph_repo_path"] = data["graph_repo_path"]
    has_batch_limit = any(
        getattr(args, attr, None) is not None
        for attr in ("num_batches", "max_train_batches", "max_val_batches")
    )
    if sys.platform.startswith("win") and has_batch_limit and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["output"] = output
    cfg["experiment"] = experiment
    return cfg


def _count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def _print_trainable_summary(model: torch.nn.Module) -> None:
    total, trainable, frozen = _count_parameters(model)
    print(f"[Params] total={total:,} trainable={trainable:,} frozen={frozen:,}")
    modules = sorted({name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad})
    print(f"[Params] trainable top-level modules: {modules}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  trainable: {name} {tuple(param.shape)}")


def _set_freeze_pixel_encoder(model: torch.nn.Module, freeze: bool) -> None:
    if hasattr(model, "freeze_pixel_encoder"):
        setattr(model, "freeze_pixel_encoder", bool(freeze))
    if hasattr(model, "set_pixel_encoder_trainable"):
        model.set_pixel_encoder_trainable(not bool(freeze))
    elif hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = not bool(freeze)
    print(f"[Freeze] pixel_encoder frozen={bool(freeze)}")


def _apply_audit_config(model: torch.nn.Module, audit_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]) -> None:
    motif = getattr(model, "motif_discovery", model)
    mapping = {
        "redundant_map_threshold": "map_sim_threshold",
        "redundant_emb_threshold": "emb_sim_threshold",
        "redundant_center_threshold_norm": "center_distance_threshold",
        "min_effective_area": "min_effective_area",
        "max_effective_area": "max_effective_area",
        "max_border_mass": "max_border_mass",
    }
    fallback = {
        "min_effective_area": loss_cfg.get("min_effective_area", loss_cfg.get("area_min", 40.0)),
        "max_effective_area": loss_cfg.get("max_effective_area", loss_cfg.get("area_max", 400.0)),
    }
    for cfg_key, attr in mapping.items():
        if hasattr(motif, attr):
            if cfg_key in audit_cfg:
                value = audit_cfg[cfg_key]
            elif cfg_key in fallback:
                value = fallback[cfg_key]
            else:
                continue
            setattr(motif, attr, float(value))
    print(
        "[Audit] "
        f"min_effective_area={getattr(motif, 'min_effective_area', None)} "
        f"max_effective_area={getattr(motif, 'max_effective_area', None)} "
        f"map_threshold={getattr(motif, 'map_sim_threshold', None)} "
        f"emb_threshold={getattr(motif, 'emb_sim_threshold', None)}"
    )


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _macro_f1_from_preds(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    scores = []
    for class_idx in range(int(num_classes)):
        pred_pos = pred == class_idx
        true_pos = target == class_idx
        tp = (pred_pos & true_pos).sum().float()
        fp = (pred_pos & ~true_pos).sum().float()
        fn = (~pred_pos & true_pos).sum().float()
        denom = 2.0 * tp + fp + fn
        scores.append(torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1e-8), tp.new_zeros(())))
    return torch.stack(scores).mean() if scores else target.new_zeros((), dtype=torch.float32)


def _class_weights_from_counts(
    class_counts: list[int] | tuple[int, ...],
    num_classes: int,
    max_class_weight: float,
) -> torch.Tensor:
    counts = torch.as_tensor(list(class_counts), dtype=torch.float32)
    if counts.numel() != int(num_classes):
        raise ValueError(f"Expected {num_classes} class counts, got {counts.numel()}")
    weights = counts.sum().clamp_min(1.0) / (float(num_classes) * counts.clamp_min(1.0))
    weights = weights / weights.mean().clamp_min(1e-8)
    if max_class_weight > 0:
        weights = weights.clamp(max=float(max_class_weight))
    return weights


def _count_train_labels_from_loader(loader_obj: Any, num_classes: int) -> list[int] | None:
    dataset = getattr(loader_obj, "dataset", None)
    graph_dataset = getattr(dataset, "dataset", None)
    if graph_dataset is None:
        return None
    counts = torch.zeros(int(num_classes), dtype=torch.long)
    try:
        for chunk_idx in range(len(graph_dataset.chunk_paths)):
            for sample in graph_dataset._get_chunk(chunk_idx):
                label = int(getattr(sample, "label"))
                if 0 <= label < int(num_classes):
                    counts[label] += 1
    except Exception as exc:
        print(f"[AuxCE] label count fallback failed: {exc}")
        return None
    return [int(v) for v in counts.tolist()]


def _build_aux_class_weights(
    aux_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_loader_obj: Any | None,
    device: torch.device,
) -> torch.Tensor | None:
    if not bool(aux_cfg.get("enabled", False)) or not bool(aux_cfg.get("use_class_weights", False)):
        return None
    num_classes = int(model_cfg.get("num_classes", aux_cfg.get("num_classes", 7)))
    counts = aux_cfg.get("class_counts")
    source = "config"
    if counts is None and train_loader_obj is not None:
        counts = _count_train_labels_from_loader(train_loader_obj, num_classes=num_classes)
        source = "train_dataset"
    if counts is None:
        print("[AuxCE] class weights disabled: no class_counts and train labels unavailable")
        return None
    weights = _class_weights_from_counts(
        class_counts=counts,
        num_classes=num_classes,
        max_class_weight=float(aux_cfg.get("max_class_weight", 3.0)),
    ).to(device=device)
    print(f"[AuxCE] class_counts_source={source} counts={list(counts)} weights={[round(float(v), 4) for v in weights.cpu()]}")
    return weights


def _scheduled_aux_lambda(aux_cfg: Dict[str, Any], epoch: int) -> float:
    if not bool(aux_cfg.get("enabled", False)):
        return 0.0
    base = float(aux_cfg.get("lambda_aux_ce", 0.0))
    start = int(aux_cfg.get("start_epoch", 6))
    ramp_epochs = max(1, int(aux_cfg.get("ramp_epochs", 5)))
    if epoch < start:
        return 0.0
    return base * min(1.0, float(epoch - start + 1) / float(ramp_epochs))


def _scheduled_teacher_align_lambda(teacher_cfg: Dict[str, Any], epoch: int) -> float:
    if not bool(teacher_cfg.get("enabled", False)):
        return 0.0
    base = float(teacher_cfg.get("lambda_teacher_align", 0.0))
    start = int(teacher_cfg.get("start_epoch", 6))
    ramp_epochs = max(1, int(teacher_cfg.get("ramp_epochs", 5)))
    if epoch < start:
        return 0.0
    ramped = base * min(1.0, float(epoch - start + 1) / float(ramp_epochs))
    if not bool(teacher_cfg.get("decay_enabled", False)):
        return ramped
    decay_start = int(teacher_cfg.get("decay_start_epoch", start + ramp_epochs))
    decay_end = int(teacher_cfg.get("decay_end_epoch", decay_start))
    min_lambda = float(teacher_cfg.get("min_lambda_after_decay", 0.0))
    if epoch < decay_start:
        return ramped
    if epoch >= decay_end:
        return min_lambda
    decay_span = max(1, decay_end - decay_start)
    progress = float(epoch - decay_start) / float(decay_span)
    return ramped + (min_lambda - ramped) * progress


def _teacher_decay_active(teacher_cfg: Dict[str, Any], epoch: int) -> bool:
    if not bool(teacher_cfg.get("enabled", False)) or not bool(teacher_cfg.get("decay_enabled", False)):
        return False
    decay_start = int(teacher_cfg.get("decay_start_epoch", 0))
    decay_end = int(teacher_cfg.get("decay_end_epoch", decay_start))
    return int(decay_start) <= int(epoch) <= int(decay_end)


def _load_teacher_alignment_model(teacher_cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module | None:
    if not bool(teacher_cfg.get("enabled", False)):
        return None
    saliency_method = str(teacher_cfg.get("saliency_method", "input_gradient")).lower()
    if saliency_method != "input_gradient":
        raise ValueError(f"Unsupported teacher_alignment.saliency_method={saliency_method!r}; only input_gradient is implemented")
    teacher_config_path = teacher_cfg.get("teacher_config") or teacher_cfg.get("model_config") or teacher_cfg.get("config")
    teacher_checkpoint_path = teacher_cfg.get("teacher_checkpoint") or teacher_cfg.get("checkpoint")
    if not teacher_config_path or not teacher_checkpoint_path:
        raise ValueError("teacher_alignment.enabled=true requires teacher_config and teacher_checkpoint")
    config_path = resolve_existing_path(teacher_config_path)
    checkpoint_path = resolve_existing_path(teacher_checkpoint_path)
    teacher_config = load_config(config_path)
    teacher = build_model(dict(teacher_config.get("model", {}) or {})).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    teacher.load_state_dict(state, strict=True)
    if bool(teacher_cfg.get("freeze_teacher", True)):
        for param in teacher.parameters():
            param.requires_grad_(False)
    if bool(teacher_cfg.get("teacher_eval_mode", True)):
        teacher.eval()
    print(f"[TeacherAlign] loaded frozen teacher config={config_path} checkpoint={checkpoint_path}")
    return teacher


def _aux_supervision_metrics(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor] | None,
    current_lambda_aux_ce: float,
    class_weights: torch.Tensor | None,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    aux_logits = out.get("aux_logits")
    if aux_logits is None or batch is None or "y" not in batch:
        device = out["motif_assignment_maps"].device
        zero = torch.zeros((), device=device)
        return {
            "loss_aux_ce": zero,
            "current_lambda_aux_ce": zero,
            "aux_accuracy": zero,
            "aux_macro_f1": zero,
            "aux_pred_count": zero,
            "aux_confidence_mean": zero,
            "aux_entropy_mean": zero,
        }
    labels = batch["y"].to(device=aux_logits.device).long()
    weight = class_weights
    if weight is not None:
        weight = weight.to(device=aux_logits.device, dtype=aux_logits.dtype)
    loss_aux_ce = F.cross_entropy(aux_logits, labels, weight=weight)
    pred = aux_logits.argmax(dim=1)
    probs = torch.softmax(aux_logits, dim=1)
    aux_accuracy = (pred == labels).float().mean()
    aux_macro_f1 = _macro_f1_from_preds(pred=pred, target=labels, num_classes=num_classes).to(device=aux_logits.device)
    return {
        "loss_aux_ce": loss_aux_ce,
        "current_lambda_aux_ce": aux_logits.new_tensor(float(current_lambda_aux_ce)),
        "aux_accuracy": aux_accuracy.detach(),
        "aux_macro_f1": aux_macro_f1.detach(),
        "aux_pred_count": aux_logits.new_tensor(float(aux_logits.shape[0])),
        "aux_confidence_mean": probs.max(dim=1).values.mean().detach(),
        "aux_entropy_mean": (-(probs * probs.clamp_min(1e-8).log()).sum(dim=1)).mean().detach(),
    }


def _extract_logits(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ("logits", "logits_fused", "logits_swin", "aux_logits"):
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return value
    available = ", ".join(sorted(out.keys()))
    raise KeyError(f"Teacher output has no logits tensor; available keys: {available}")


def _normalize_spatial_sum(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = values.flatten(1).clamp_min(0.0)
    denom = flat.sum(dim=1, keepdim=True).clamp_min(eps)
    return (flat / denom).view_as(values)


def _make_face_safe_mask(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    margin: float = 0.12,
    tau: float = 0.04,
    center_prior_sigma: float = 0.45,
    center_prior_power: float = 0.3,
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    tau_t = torch.as_tensor(max(float(tau), 1e-6), device=device, dtype=dtype)
    margin_t = torch.as_tensor(float(margin), device=device, dtype=dtype)
    safe_x = torch.sigmoid((xx - margin_t) / tau_t) * torch.sigmoid((1.0 - margin_t - xx) / tau_t)
    safe_y = torch.sigmoid((yy - margin_t) / tau_t) * torch.sigmoid((1.0 - margin_t - yy) / tau_t)
    mask = safe_x * safe_y
    sigma = max(float(center_prior_sigma), 1e-6)
    dist2 = (xx - 0.5).pow(2) + (yy - 0.5).pow(2)
    center_prior = torch.exp(-dist2 / (2.0 * sigma * sigma))
    power = float(center_prior_power)
    if power != 0.0:
        mask = mask * center_prior.clamp_min(1e-8).pow(power)
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


def _border_center_mass_from_map(values: torch.Tensor, border_width: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, height, width = values.shape
    mask = torch.zeros((height, width), device=values.device, dtype=torch.bool)
    bw = int(border_width)
    if bw > 0:
        mask[:bw, :] = True
        mask[-bw:, :] = True
        mask[:, :bw] = True
        mask[:, -bw:] = True
    flat = values.flatten(1)
    total = flat.sum(dim=1).clamp_min(1e-8)
    border = values[:, mask].sum(dim=1) / total
    center = values[:, ~mask].sum(dim=1) / total
    return border, center


def _top_iou_from_maps(a: torch.Tensor, b: torch.Tensor, q: float = 0.80) -> torch.Tensor:
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    a_thr = torch.quantile(a_flat.float(), q=float(q), dim=1, keepdim=True).to(dtype=a_flat.dtype)
    b_thr = torch.quantile(b_flat.float(), q=float(q), dim=1, keepdim=True).to(dtype=b_flat.dtype)
    a_mask = a_flat >= a_thr
    b_mask = b_flat >= b_thr
    inter = (a_mask & b_mask).sum(dim=1).float()
    union = (a_mask | b_mask).sum(dim=1).float().clamp_min(1.0)
    return inter / union


def _selected_map_from_out(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    maps = out["motif_assignment_maps"]
    if "selection_weights" in out:
        weights = out["selection_weights"]
    else:
        weights = torch.softmax(out["motif_scores"], dim=1)
    selected = (maps * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    return _normalize_spatial_sum(selected)


def _compute_teacher_alignment(
    teacher: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    current_lambda: float,
    height: int,
    width: int,
    border_width: int,
    compute_when_inactive: bool = False,
) -> Dict[str, torch.Tensor]:
    device = out["motif_assignment_maps"].device
    dtype = out["motif_assignment_maps"].dtype
    zero = out["motif_assignment_maps"].new_zeros(())
    if teacher is None or (float(current_lambda) <= 0.0 and not compute_when_inactive):
        return {
            "loss_teacher_align": zero,
            "current_lambda_teacher_align": out["motif_assignment_maps"].new_tensor(float(current_lambda)),
            "teacher_alignment_active": zero,
            "teacher_confidence_mean": zero,
            "teacher_pred_acc_batch": zero,
            "teacher_saliency_border_mass": zero,
            "teacher_saliency_center_mass": zero,
            "filtered_teacher_saliency_border_mass": zero,
            "filtered_teacher_saliency_center_mass": zero,
            "teacher_motif_cosine": zero,
            "teacher_motif_iou_top20": zero,
        }

    labels = batch.get("y")
    if not torch.is_tensor(labels):
        raise KeyError("teacher_alignment target=ground_truth requires batch['y']")
    x_teacher = batch["x"].detach().clone().requires_grad_(True)
    teacher_batch = dict(batch)
    teacher_batch["x"] = x_teacher
    teacher_batch["node_features"] = x_teacher
    with torch.enable_grad():
        teacher.zero_grad(set_to_none=True)
        teacher_out = teacher(teacher_batch)
        logits = _extract_logits(teacher_out)
        probs = torch.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        confidence = probs.max(dim=1).values
        target_mode = str(cfg.get("target", "ground_truth")).lower()
        if target_mode == "predicted":
            target_logits = logits.max(dim=1).values
        elif target_mode == "ground_truth":
            target = labels.long().clamp(min=0, max=logits.shape[1] - 1)
            target_logits = logits[torch.arange(logits.shape[0], device=device), target]
        else:
            raise ValueError(f"Unsupported teacher_alignment.target={target_mode!r}")
        grad = torch.autograd.grad(
            target_logits.sum(),
            x_teacher,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

    raw_sal = grad.detach().abs().sum(dim=-1).reshape(x_teacher.shape[0], height, width)
    raw_sal = _normalize_spatial_sum(raw_sal)
    raw_border, raw_center = _border_center_mass_from_map(raw_sal, border_width=border_width)

    face_mask = _make_face_safe_mask(
        batch_size=x_teacher.shape[0],
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        margin=float(cfg.get("face_safe_margin", 0.12)),
        tau=float(cfg.get("face_safe_tau", 0.04)),
        center_prior_sigma=float(cfg.get("center_prior_sigma", 0.45)),
        center_prior_power=float(cfg.get("center_prior_power", 0.3)),
    )
    if bool(cfg.get("use_face_safe_filter", True)):
        filtered = raw_sal.to(dtype=dtype) * face_mask
        filtered_sum = filtered.flatten(1).sum(dim=1, keepdim=True)
        if (filtered_sum <= 1e-8).any():
            fallback_mask = _make_face_safe_mask(
                batch_size=x_teacher.shape[0],
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                margin=float(cfg.get("face_safe_margin", 0.12)),
                tau=float(cfg.get("face_safe_tau", 0.04)),
                center_prior_sigma=float(cfg.get("center_prior_sigma", 0.45)),
                center_prior_power=0.0,
            )
            fallback = raw_sal.to(dtype=dtype) * fallback_mask
            fallback_sum = fallback.flatten(1).sum(dim=1, keepdim=True)
            filtered = torch.where((filtered_sum <= 1e-8).view(-1, 1, 1), fallback, filtered)
            filtered_sum = torch.where(filtered_sum <= 1e-8, fallback_sum, filtered_sum)
        valid = (filtered_sum.squeeze(1) > 1e-8).to(dtype=dtype)
        filtered = _normalize_spatial_sum(filtered)
    else:
        valid = torch.ones(x_teacher.shape[0], device=device, dtype=dtype)
        filtered = raw_sal.to(dtype=dtype)

    selected_map = _selected_map_from_out(out).to(dtype=dtype)
    flat_selected = F.normalize(selected_map.flatten(1).float(), dim=1, eps=1e-8)
    flat_teacher = F.normalize(filtered.flatten(1).float(), dim=1, eps=1e-8)
    cosine = (flat_selected * flat_teacher).sum(dim=1).to(dtype=dtype)
    per_sample = (1.0 - cosine).clamp_min(0.0)
    if bool(cfg.get("confidence_weighted", True)):
        per_sample = per_sample * confidence.detach().to(dtype=dtype)
    min_conf = float(cfg.get("min_teacher_confidence", 0.0))
    if min_conf > 0.0:
        valid = valid * (confidence.detach().to(dtype=dtype) >= min_conf).to(dtype=dtype)
    denom = valid.sum().clamp_min(1.0)
    loss_teacher_align = (per_sample * valid).sum() / denom

    filtered_border, filtered_center = _border_center_mass_from_map(filtered, border_width=border_width)
    iou20 = _top_iou_from_maps(selected_map.detach(), filtered.detach(), q=0.80)
    labels_long = labels.long().clamp(min=0, max=logits.shape[1] - 1)
    pred_acc = (pred == labels_long).float().mean()
    return {
        "loss_teacher_align": loss_teacher_align,
        "current_lambda_teacher_align": out["motif_assignment_maps"].new_tensor(float(current_lambda)),
        "teacher_alignment_active": out["motif_assignment_maps"].new_tensor(1.0),
        "teacher_confidence_mean": confidence.mean().detach(),
        "teacher_pred_acc_batch": pred_acc.detach(),
        "teacher_saliency_border_mass": raw_border.mean().detach(),
        "teacher_saliency_center_mass": raw_center.mean().detach(),
        "filtered_teacher_saliency_border_mass": filtered_border.mean().detach(),
        "filtered_teacher_saliency_center_mass": filtered_center.mean().detach(),
        "teacher_motif_cosine": cosine.mean().detach(),
        "teacher_motif_iou_top20": iou20.mean().detach(),
    }


def _quality_components(
    metrics: Dict[str, float],
    quality_cfg: Dict[str, Any],
    target_effective_motifs: float = 8.0,
) -> Dict[str, float]:
    effective_score = min(metrics.get("effective_motif_count", 0.0), target_effective_motifs) / max(target_effective_motifs, 1.0)
    selection_entropy_target = max(float(quality_cfg.get("selection_entropy_target", 1.2)), 1e-8)
    selection_entropy_score = min(metrics.get("selection_entropy", 0.0) / selection_entropy_target, 1.0)
    clean_count_target = max(float(quality_cfg.get("clean_count_target", metrics.get("clean_count_target", 3.0))), 1e-8)
    clean_candidate_score = min(metrics.get("clean_candidate_count", 0.0) / clean_count_target, 1.0)
    region_clean_target = max(float(quality_cfg.get("region_clean_target", metrics.get("region_clean_target", 0.7))), 1e-8)
    region_clean_mean_score = min(metrics.get("mean_region_clean_count", 0.0) / region_clean_target, 1.0)
    region_clean_min_score = min(metrics.get("min_region_clean_count", 0.0) / region_clean_target, 1.0)
    components = {
        "q_area_in_range_component": float(quality_cfg.get("area_in_range_weight", 2.0)) * metrics.get("area_in_range_ratio", 0.0),
        "q_effective_motif_component": float(quality_cfg.get("effective_motif_weight", 1.0)) * effective_score,
        "q_coverage_component": float(quality_cfg.get("coverage_weight", 0.8)) * metrics.get("coverage_cosine", 0.0),
        "q_foreground_mass_component": float(quality_cfg.get("foreground_mass_weight", 0.0)) * metrics.get("motif_foreground_mass_mean", 0.0),
        "q_selected_foreground_component": float(quality_cfg.get("selected_foreground_weight", 0.0)) * metrics.get("selected_foreground_mass_mean", 0.0),
        "q_selection_entropy_component": float(quality_cfg.get("selection_entropy_weight", 0.0)) * selection_entropy_score,
        "q_clean_candidate_component": float(quality_cfg.get("clean_candidate_weight", 0.0)) * clean_candidate_score,
        "q_region_clean_mean_component": float(quality_cfg.get("region_clean_mean_weight", 0.0)) * region_clean_mean_score,
        "q_region_clean_min_component": float(quality_cfg.get("region_clean_min_weight", 0.0)) * region_clean_min_score,
        "q_teacher_motif_cosine_component": float(quality_cfg.get("teacher_motif_cosine_weight", 0.0)) * metrics.get("teacher_motif_cosine", 0.0),
        "q_aux_component": float(quality_cfg.get("aux_macro_f1_weight", 0.0)) * metrics.get("aux_macro_f1", 0.0),
        "q_center_border_component": float(quality_cfg.get("center_border_weight", 0.5)) * metrics.get("center_minus_border", 0.0),
        "q_selected_border_penalty": float(quality_cfg.get("selected_border_penalty_weight", 0.0)) * metrics.get("selected_border_mass_mean", 0.0),
        "q_selected_outer_border_penalty": float(quality_cfg.get("selected_outer_border_penalty_weight", 0.0)) * metrics.get("selected_outer_border_mass_mean", 0.0),
        "q_border_dominance_penalty": float(quality_cfg.get("border_dominance_penalty_weight", 0.0)) * metrics.get("border_dominance", 0.0),
        "q_outer_border_penalty": float(quality_cfg.get("outer_border_penalty_weight", 0.0)) * metrics.get("outer_border_mass_mean", 0.0),
        "q_redundancy_penalty": float(quality_cfg.get("redundant_penalty_weight", 1.0)) * metrics.get("redundant_pair_ratio", 0.0),
        "q_map_sim_penalty": float(quality_cfg.get("map_sim_penalty_weight", 0.5)) * metrics.get("mean_pairwise_map_sim", 0.0),
        "q_area_over_penalty": float(quality_cfg.get("area_over_penalty_weight", 0.4)) * metrics.get("area_over_max_ratio", metrics.get("effective_area_over_max_ratio", 0.0)),
        "q_area_under_penalty": float(quality_cfg.get("area_under_penalty_weight", 0.4)) * metrics.get("area_under_min_ratio", metrics.get("effective_area_under_min_ratio", 0.0)),
    }
    components["motif_quality_score"] = (
        components["q_area_in_range_component"]
        + components["q_effective_motif_component"]
        + components["q_coverage_component"]
        + components["q_foreground_mass_component"]
        + components["q_selected_foreground_component"]
        + components["q_selection_entropy_component"]
        + components["q_clean_candidate_component"]
        + components["q_region_clean_mean_component"]
        + components["q_region_clean_min_component"]
        + components["q_teacher_motif_cosine_component"]
        + components["q_aux_component"]
        + components["q_center_border_component"]
        - components["q_selected_border_penalty"]
        - components["q_selected_outer_border_penalty"]
        - components["q_border_dominance_penalty"]
        - components["q_outer_border_penalty"]
        - components["q_redundancy_penalty"]
        - components["q_map_sim_penalty"]
        - components["q_area_over_penalty"]
        - components["q_area_under_penalty"]
    )
    components["selection_entropy_score"] = selection_entropy_score
    components["clean_candidate_score_for_quality"] = clean_candidate_score
    components["region_clean_mean_score_for_quality"] = region_clean_mean_score
    components["region_clean_min_score_for_quality"] = region_clean_min_score
    return components


def _scheduled_loss_weights(base: Dict[str, Any], schedule: Dict[str, Any], epoch: int) -> Dict[str, float]:
    weights = {
        "lambda_div_map": float(base.get("lambda_div_map", 0.0)),
        "lambda_div_emb": float(base.get("lambda_div_emb", 0.0)),
        "lambda_coverage": float(base.get("lambda_coverage", 0.0)),
        "lambda_anchor": float(base.get("lambda_anchor", 0.0)),
        "lambda_border": float(base.get("lambda_border", 0.0)),
        "lambda_soft_border": float(base.get("lambda_soft_border", 0.0)),
        "lambda_outer_border": float(base.get("lambda_outer_border", 0.0)),
        "lambda_selected_border": float(base.get("lambda_selected_border", 0.0)),
        "lambda_selected_outer_border": float(base.get("lambda_selected_outer_border", 0.0)),
        "lambda_selected_foreground": float(base.get("lambda_selected_foreground", 0.0)),
        "lambda_selected_diversity": float(base.get("lambda_selected_diversity", 0.0)),
        "lambda_selection_entropy": float(base.get("lambda_selection_entropy", 0.0)),
        "lambda_clean_count": float(base.get("lambda_clean_count", 0.0)),
        "lambda_clean_mean": float(base.get("lambda_clean_mean", 0.0)),
        "lambda_region_clean": float(base.get("lambda_region_clean", 0.0)),
        "lambda_entropy": float(base.get("lambda_entropy", 0.0)),
        "lambda_area": float(base.get("lambda_area", 0.0)),
    }
    if not bool(schedule.get("enabled", False)):
        return weights
    div_ramp_epochs = max(1, int(schedule.get("diversity_ramp_epochs", 10)))
    div_factor = min(1.0, 0.25 + 0.75 * max(0, epoch - 1) / float(div_ramp_epochs))
    coverage_start = int(schedule.get("coverage_start_epoch", 3))
    coverage_ramp_epochs = max(1, int(schedule.get("coverage_ramp_epochs", 7)))
    if epoch < coverage_start:
        coverage_factor = 0.0
    else:
        coverage_factor = min(1.0, float(epoch - coverage_start + 1) / float(coverage_ramp_epochs))
    boost_epoch = schedule.get("coverage_boost_epoch")
    boost_factor = float(schedule.get("coverage_boost_factor", 1.0))
    weights["lambda_div_map"] *= div_factor
    weights["lambda_div_emb"] *= div_factor
    weights["lambda_coverage"] *= coverage_factor
    if boost_epoch is not None and epoch >= int(boost_epoch):
        weights["lambda_coverage"] = float(base.get("lambda_coverage", 0.0)) * boost_factor
    soft_start = int(schedule.get("soft_border_start_epoch", 1))
    soft_ramp_epochs = max(1, int(schedule.get("soft_border_ramp_epochs", 5)))
    if epoch < soft_start:
        soft_factor = 0.0
    else:
        soft_factor = min(1.0, float(epoch - soft_start + 1) / float(soft_ramp_epochs))
    weights["lambda_soft_border"] *= soft_factor
    soft_boost_epoch = schedule.get("soft_border_boost_epoch")
    soft_boost_factor = float(schedule.get("soft_border_boost_factor", 1.0))
    if soft_boost_epoch is not None and epoch >= int(soft_boost_epoch):
        weights["lambda_soft_border"] = float(base.get("lambda_soft_border", 0.0)) * soft_boost_factor
    anchor_start = int(schedule.get("anchor_start_epoch", 1))
    anchor_ramp_epochs = max(1, int(schedule.get("anchor_ramp_epochs", 5)))
    if epoch < anchor_start:
        anchor_factor = 0.0
    else:
        anchor_factor = min(1.0, float(epoch - anchor_start + 1) / float(anchor_ramp_epochs))
    weights["lambda_anchor"] *= anchor_factor
    outer_start = int(schedule.get("outer_border_start_epoch", 1))
    outer_ramp_epochs = max(1, int(schedule.get("outer_border_ramp_epochs", 5)))
    if epoch < outer_start:
        outer_factor = 0.0
    else:
        outer_factor = min(1.0, float(epoch - outer_start + 1) / float(outer_ramp_epochs))
    weights["lambda_outer_border"] *= outer_factor
    selected_start = int(schedule.get("selected_loss_start_epoch", 1))
    selected_ramp_epochs = max(1, int(schedule.get("selected_loss_ramp_epochs", 5)))
    if epoch < selected_start:
        selected_factor = 0.0
    else:
        selected_factor = min(1.0, float(epoch - selected_start + 1) / float(selected_ramp_epochs))
    weights["lambda_selected_border"] *= selected_factor
    weights["lambda_selected_outer_border"] *= selected_factor
    weights["lambda_selected_foreground"] *= selected_factor
    weights["lambda_selected_diversity"] *= selected_factor
    entropy_start = int(schedule.get("selection_entropy_start_epoch", selected_start))
    entropy_ramp_epochs = max(1, int(schedule.get("selection_entropy_ramp_epochs", selected_ramp_epochs)))
    if epoch < entropy_start:
        entropy_factor = 0.0
    else:
        entropy_factor = min(1.0, float(epoch - entropy_start + 1) / float(entropy_ramp_epochs))
    weights["lambda_selection_entropy"] *= entropy_factor
    clean_start = int(schedule.get("clean_count_start_epoch", 1))
    clean_ramp_epochs = max(1, int(schedule.get("clean_count_ramp_epochs", 1)))
    if epoch < clean_start:
        clean_factor = 0.0
    else:
        clean_factor = min(1.0, float(epoch - clean_start + 1) / float(clean_ramp_epochs))
    weights["lambda_clean_count"] *= clean_factor
    weights["lambda_clean_mean"] *= clean_factor
    region_start = int(schedule.get("region_clean_start_epoch", 1))
    region_ramp_epochs = max(1, int(schedule.get("region_clean_ramp_epochs", 1)))
    if epoch < region_start:
        region_factor = 0.0
    else:
        region_factor = min(1.0, float(epoch - region_start + 1) / float(region_ramp_epochs))
    weights["lambda_region_clean"] *= region_factor
    return weights


def _coverage_boost_active(schedule: Dict[str, Any], epoch: int) -> bool:
    boost_epoch = schedule.get("coverage_boost_epoch")
    return bool(schedule.get("enabled", False) and boost_epoch is not None and epoch >= int(boost_epoch))


def _soft_border_boost_active(schedule: Dict[str, Any], epoch: int) -> bool:
    boost_epoch = schedule.get("soft_border_boost_epoch")
    return bool(schedule.get("enabled", False) and boost_epoch is not None and epoch >= int(boost_epoch))


def _spatial_bias_strength(model_cfg: Dict[str, Any], epoch: int) -> float:
    initial = float(model_cfg.get("spatial_bias_strength", 0.0))
    if not bool(model_cfg.get("use_spatial_query_bias", False)):
        return 0.0
    if not bool(model_cfg.get("spatial_bias_decay", False)):
        return initial
    decay_epochs = max(1, int(model_cfg.get("spatial_bias_decay_epochs", 20)))
    min_strength = float(model_cfg.get("spatial_bias_min_strength", 0.0))
    factor = max(0.0, 1.0 - float(max(0, epoch - 1)) / float(decay_epochs))
    return max(min_strength, initial * factor)


def _merge_metrics(
    loss_dict: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    target_effective_motifs: float,
    quality_cfg: Dict[str, Any],
    coverage_boost_active: bool = False,
    soft_border_boost_active: bool = False,
) -> Dict[str, float]:
    maps = out["motif_assignment_maps"]
    assignment_sum = maps.sum(dim=(2, 3))
    audit = out.get("motif_audit", {}) or {}
    values: Dict[str, float] = {
        "total_loss": _scalar(loss_dict.get("total_loss", loss_dict["loss"])),
        "loss_map_diversity": _scalar(loss_dict["loss_map_diversity"]),
        "loss_embedding_diversity": _scalar(loss_dict["loss_embedding_diversity"]),
        "loss_coverage": _scalar(loss_dict["loss_coverage"]),
        "loss_anchor": _scalar(loss_dict.get("loss_anchor", 0.0)),
        "loss_border": _scalar(loss_dict["loss_border"]),
        "loss_soft_border": _scalar(loss_dict.get("loss_soft_border", 0.0)),
        "loss_outer_border": _scalar(loss_dict.get("loss_outer_border", 0.0)),
        "loss_selected_border": _scalar(loss_dict.get("loss_selected_border", 0.0)),
        "loss_selected_outer_border": _scalar(loss_dict.get("loss_selected_outer_border", 0.0)),
        "loss_selected_foreground": _scalar(loss_dict.get("loss_selected_foreground", 0.0)),
        "loss_selected_diversity": _scalar(loss_dict.get("loss_selected_diversity", 0.0)),
        "loss_selection_entropy": _scalar(loss_dict.get("loss_selection_entropy", 0.0)),
        "loss_clean_count": _scalar(loss_dict.get("loss_clean_count", 0.0)),
        "loss_clean_mean": _scalar(loss_dict.get("loss_clean_mean", 0.0)),
        "loss_region_clean": _scalar(loss_dict.get("loss_region_clean", 0.0)),
        "loss_region_clean_upper_component": _scalar(loss_dict.get("loss_region_clean_upper_component", 0.0)),
        "loss_region_clean_middle_component": _scalar(loss_dict.get("loss_region_clean_middle_component", 0.0)),
        "loss_region_clean_lower_component": _scalar(loss_dict.get("loss_region_clean_lower_component", 0.0)),
        "loss_teacher_align": _scalar(loss_dict.get("loss_teacher_align", 0.0)),
        "loss_aux_ce": _scalar(loss_dict.get("loss_aux_ce", 0.0)),
        "loss_entropy": _scalar(loss_dict.get("loss_entropy", loss_dict["loss_entropy_control"])),
        "loss_area_high": _scalar(loss_dict.get("loss_area_high", 0.0)),
        "loss_area_low": _scalar(loss_dict.get("loss_area_low", 0.0)),
        "loss_area_balance": _scalar(loss_dict["loss_area_balance"]),
        "assignment_sum_mean": _scalar(assignment_sum.mean()),
        "assignment_sum_max_abs_err": _scalar((assignment_sum - 1.0).abs().max()),
    }
    for key in (
        "mean_pairwise_map_sim",
        "max_pairwise_map_sim",
        "mean_pairwise_emb_sim",
        "max_pairwise_emb_sim",
        "mean_center_dist",
        "min_center_dist",
        "redundant_pair_count",
        "redundant_pair_ratio",
        "border_mass_mean",
        "center_mass_mean",
        "center_minus_border",
        "border_dominance",
        "soft_border_mass_mean",
        "effective_area_mean",
        "effective_area_min",
        "effective_area_max",
        "area_in_range_ratio",
        "area_over_max_ratio",
        "area_under_min_ratio",
        "effective_area_over_max_ratio",
        "effective_area_under_min_ratio",
        "assignment_entropy_mean",
        "effective_motif_count",
        "effective_motif_ratio",
        "logits_mean",
        "logits_std",
        "logits_min",
        "logits_max",
        "attention_temperature",
        "spatial_bias_strength_current",
        "motif_initial_center_min",
        "motif_initial_center_max",
    ):
        source = audit if key in audit else loss_dict
        if key in source:
            values[key] = _scalar(source[key])
    values["coverage_cosine"] = _scalar(loss_dict["coverage_cosine"])
    values["foreground_mass_mean"] = _scalar(loss_dict["foreground_mass_mean"])
    values["foreground_center_mass"] = _scalar(loss_dict.get("foreground_center_mass", 0.0))
    values["foreground_border_mass"] = _scalar(loss_dict.get("foreground_border_mass", 0.0))
    values["foreground_prior_sum"] = _scalar(loss_dict.get("foreground_prior_sum", 0.0))
    values["foreground_safe_mass"] = _scalar(loss_dict.get("foreground_safe_mass", 0.0))
    values["outer_border_mass_mean"] = _scalar(loss_dict.get("outer_border_mass_mean", 0.0))
    values["selected_border_mass_mean"] = _scalar(loss_dict.get("selected_border_mass_mean", 0.0))
    values["selected_outer_border_mass_mean"] = _scalar(loss_dict.get("selected_outer_border_mass_mean", 0.0))
    values["selected_foreground_mass_mean"] = _scalar(loss_dict.get("selected_foreground_mass_mean", 0.0))
    values["selection_entropy"] = _scalar(loss_dict.get("selection_entropy", 0.0))
    values["selection_entropy_soft"] = _scalar(loss_dict.get("selection_entropy_soft", values["selection_entropy"]))
    values["selection_effective_count"] = _scalar(loss_dict.get("selection_effective_count", 0.0))
    values["selected_effective_count"] = _scalar(loss_dict.get("selected_effective_count", values["selection_effective_count"]))
    values["selected_pairwise_map_sim"] = _scalar(loss_dict.get("selected_pairwise_map_sim", 0.0))
    values["selected_pairwise_emb_sim"] = _scalar(loss_dict.get("selected_pairwise_emb_sim", 0.0))
    values["selected_top_m"] = _scalar(loss_dict.get("selected_top_m", 0.0))
    values["target_selection_entropy_min"] = _scalar(loss_dict.get("target_selection_entropy_min", 0.0))
    values["clean_score_mean"] = _scalar(loss_dict.get("clean_score_mean", 0.0))
    values["clean_score_max"] = _scalar(loss_dict.get("clean_score_max", 0.0))
    values["clean_candidate_count"] = _scalar(loss_dict.get("clean_candidate_count", 0.0))
    values["hard_clean_candidate_count"] = _scalar(loss_dict.get("hard_clean_candidate_count", 0.0))
    values["clean_border_score_mean"] = _scalar(loss_dict.get("clean_border_score_mean", 0.0))
    values["clean_outer_score_mean"] = _scalar(loss_dict.get("clean_outer_score_mean", 0.0))
    values["clean_foreground_score_mean"] = _scalar(loss_dict.get("clean_foreground_score_mean", 0.0))
    values["clean_area_low_score_mean"] = _scalar(loss_dict.get("clean_area_low_score_mean", 0.0))
    values["clean_area_high_score_mean"] = _scalar(loss_dict.get("clean_area_high_score_mean", 0.0))
    values["clean_count_target"] = _scalar(loss_dict.get("clean_count_target", 0.0))
    values["motif_upper_mass_mean"] = _scalar(loss_dict.get("motif_upper_mass_mean", 0.0))
    values["motif_middle_mass_mean"] = _scalar(loss_dict.get("motif_middle_mass_mean", 0.0))
    values["motif_lower_mass_mean"] = _scalar(loss_dict.get("motif_lower_mass_mean", 0.0))
    values["upper_clean_count"] = _scalar(loss_dict.get("upper_clean_count", 0.0))
    values["middle_clean_count"] = _scalar(loss_dict.get("middle_clean_count", 0.0))
    values["lower_clean_count"] = _scalar(loss_dict.get("lower_clean_count", 0.0))
    values["min_region_clean_count"] = _scalar(loss_dict.get("min_region_clean_count", 0.0))
    values["mean_region_clean_count"] = _scalar(loss_dict.get("mean_region_clean_count", 0.0))
    values["region_clean_target"] = _scalar(loss_dict.get("region_clean_target", 0.0))
    values["region_clean_upper_weight"] = _scalar(loss_dict.get("region_clean_upper_weight", 1.0))
    values["region_clean_middle_weight"] = _scalar(loss_dict.get("region_clean_middle_weight", 1.0))
    values["region_clean_lower_weight"] = _scalar(loss_dict.get("region_clean_lower_weight", 1.0))
    values["motif_foreground_mass_mean"] = _scalar(loss_dict.get("motif_foreground_mass_mean", 0.0))
    values["motif_foreground_mass_min"] = _scalar(loss_dict.get("motif_foreground_mass_min", 0.0))
    values["motif_foreground_mass_max"] = _scalar(loss_dict.get("motif_foreground_mass_max", 0.0))
    values["center_prior_sigma"] = _scalar(loss_dict.get("center_prior_sigma", 0.0))
    values["center_prior_power"] = _scalar(loss_dict.get("center_prior_power", 0.0))
    values["foreground_face_safe_margin"] = _scalar(loss_dict.get("foreground_face_safe_margin", 0.0))
    values["foreground_face_safe_tau"] = _scalar(loss_dict.get("foreground_face_safe_tau", 0.0))
    for key in (
        "current_lambda_div_map",
        "current_lambda_div_emb",
        "current_lambda_coverage",
        "current_lambda_anchor",
        "current_lambda_border",
        "current_lambda_soft_border",
        "current_lambda_outer_border",
        "current_lambda_selected_border",
        "current_lambda_selected_outer_border",
        "current_lambda_selected_foreground",
        "current_lambda_selected_diversity",
        "current_lambda_selection_entropy",
        "current_lambda_clean_count",
        "current_lambda_clean_mean",
        "current_lambda_region_clean",
        "current_lambda_teacher_align",
        "current_lambda_aux_ce",
        "current_lambda_entropy",
        "current_lambda_area",
    ):
        if key in loss_dict:
            values[key] = _scalar(loss_dict[key])
    values["aux_accuracy"] = _scalar(loss_dict.get("aux_accuracy", 0.0))
    values["aux_macro_f1"] = _scalar(loss_dict.get("aux_macro_f1", 0.0))
    values["aux_pred_count"] = _scalar(loss_dict.get("aux_pred_count", 0.0))
    values["aux_confidence_mean"] = _scalar(loss_dict.get("aux_confidence_mean", 0.0))
    values["aux_entropy_mean"] = _scalar(loss_dict.get("aux_entropy_mean", 0.0))
    values["coverage_boost_active"] = 1.0 if coverage_boost_active else 0.0
    values["soft_border_boost_active"] = 1.0 if soft_border_boost_active else 0.0
    values["teacher_alignment_active"] = _scalar(loss_dict.get("teacher_alignment_active", 0.0))
    values["teacher_decay_active"] = _scalar(loss_dict.get("teacher_decay_active", 0.0))
    values["teacher_confidence_mean"] = _scalar(loss_dict.get("teacher_confidence_mean", 0.0))
    values["teacher_pred_acc_batch"] = _scalar(loss_dict.get("teacher_pred_acc_batch", 0.0))
    values["teacher_saliency_border_mass"] = _scalar(loss_dict.get("teacher_saliency_border_mass", 0.0))
    values["teacher_saliency_center_mass"] = _scalar(loss_dict.get("teacher_saliency_center_mass", 0.0))
    values["filtered_teacher_saliency_border_mass"] = _scalar(loss_dict.get("filtered_teacher_saliency_border_mass", 0.0))
    values["filtered_teacher_saliency_center_mass"] = _scalar(loss_dict.get("filtered_teacher_saliency_center_mass", 0.0))
    values["teacher_motif_cosine"] = _scalar(loss_dict.get("teacher_motif_cosine", 0.0))
    values["teacher_motif_iou_top20"] = _scalar(loss_dict.get("teacher_motif_iou_top20", 0.0))
    values.update(_quality_components(values, quality_cfg=quality_cfg, target_effective_motifs=target_effective_motifs))
    values["val_motif_quality_score"] = values["motif_quality_score"]
    return values


def _append_history(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _average(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {key: float("nan") for key in HISTORY_FIELDS if key not in {"epoch", "split"}}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / float(len(rows)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": float(best_score),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _check_finite(loss: torch.Tensor, out: Dict[str, torch.Tensor]) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite motif loss: {float(loss.detach().cpu())}")
    for key in ("motif_embeddings", "motif_assignment_maps", "motif_scores", "selection_weights", "selected_repr", "aux_logits"):
        if key in out and not torch.isfinite(out[key]).all():
            raise FloatingPointError(f"{key} contains NaN/Inf")


def _run_epoch(
    model: torch.nn.Module,
    criterion: MotifDiscoveryStage1Loss,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    split: str,
    max_batches: int | None,
    amp: bool,
    grad_clip_norm: float,
    log_every: int,
    target_effective_motifs: float,
    quality_cfg: Dict[str, Any],
    current_lambda_aux_ce: float = 0.0,
    teacher_model: torch.nn.Module | None = None,
    teacher_alignment_cfg: Dict[str, Any] | None = None,
    current_lambda_teacher_align: float = 0.0,
    teacher_height: int = 48,
    teacher_width: int = 48,
    teacher_border_width: int = 4,
    teacher_decay_active: bool = False,
    aux_class_weights: torch.Tensor | None = None,
    aux_num_classes: int = 7,
    coverage_boost_active: bool = False,
    soft_border_boost_active: bool = False,
    synthetic: bool = False,
    feature_ablation_cfg: Dict[str, Any] | None = None,
    model_node_dim: int = 7,
    model_edge_dim: int = 5,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    if teacher_model is not None:
        teacher_model.eval()
    if getattr(model, "freeze_pixel_encoder", False) and hasattr(model, "encoder"):
        model.encoder.eval()
    rows: list[Dict[str, float]] = []
    autocast_enabled = bool(amp and device.type == "cuda")
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if not synthetic:
            batch = move_to_device(batch, device)
        batch = apply_feature_ablation(batch, feature_ablation_cfg)
        assert_feature_dims(batch, node_dim=model_node_dim, edge_dim=model_edge_dim)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                out = model(batch)
                loss_dict = criterion(out, batch)
                aux_dict = _aux_supervision_metrics(
                    out,
                    batch,
                    current_lambda_aux_ce=current_lambda_aux_ce,
                    class_weights=aux_class_weights,
                    num_classes=aux_num_classes,
                )
                loss_dict.update(aux_dict)
                teacher_dict = _compute_teacher_alignment(
                    teacher=teacher_model,
                    batch=batch,
                    out=out,
                    cfg=teacher_alignment_cfg or {},
                    current_lambda=float(current_lambda_teacher_align),
                    height=int(teacher_height),
                    width=int(teacher_width),
                    border_width=int(teacher_border_width),
                    compute_when_inactive=bool((teacher_alignment_cfg or {}).get("decay_enabled", False)),
                )
                loss_dict.update(teacher_dict)
                loss_dict["teacher_decay_active"] = out["motif_assignment_maps"].new_tensor(
                    1.0 if teacher_decay_active else 0.0
                )
                loss = loss_dict["loss"]
                if float(current_lambda_aux_ce) > 0.0:
                    loss = loss + loss_dict["current_lambda_aux_ce"] * loss_dict["loss_aux_ce"]
                    loss_dict["loss"] = loss
                    loss_dict["total_loss"] = loss
                if float(current_lambda_teacher_align) > 0.0:
                    loss = loss + loss_dict["current_lambda_teacher_align"] * loss_dict["loss_teacher_align"]
                    loss_dict["loss"] = loss
                    loss_dict["total_loss"] = loss
            _check_finite(loss, out)
            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                optimizer.step()
        metrics = _merge_metrics(
            loss_dict,
            out,
            target_effective_motifs=target_effective_motifs,
            quality_cfg=quality_cfg,
            coverage_boost_active=coverage_boost_active,
            soft_border_boost_active=soft_border_boost_active,
        )
        rows.append(metrics)
        if is_train and log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            print(
                f"[epoch {epoch:03d} {split} batch {batch_idx + 1}] "
                f"loss={metrics['total_loss']:.4f} quality={metrics['motif_quality_score']:.4f} "
                f"map_sim={metrics['mean_pairwise_map_sim']:.4f} emb_sim={metrics['mean_pairwise_emb_sim']:.4f} "
                f"redundant={metrics['redundant_pair_ratio']:.4f} effective={metrics['effective_motif_count']:.2f} "
                f"sel_border={metrics.get('selected_border_mass_mean', 0.0):.4f} "
                f"sel_fg={metrics.get('selected_foreground_mass_mean', 0.0):.4f} "
                f"sel_H={metrics.get('selection_entropy', 0.0):.4f} "
                f"clean={metrics.get('clean_candidate_count', 0.0):.2f} "
                f"region_clean={metrics.get('mean_region_clean_count', 0.0):.2f}/{metrics.get('min_region_clean_count', 0.0):.2f} "
                f"t_cos={metrics.get('teacher_motif_cosine', 0.0):.4f} "
                f"aux_f1={metrics.get('aux_macro_f1', 0.0):.4f}"
            )
    summary = _average(rows)
    print(
        f"[epoch {epoch:03d} {split}] "
        f"loss={summary['total_loss']:.4f} quality={summary['motif_quality_score']:.4f} "
        f"map_sim={summary['mean_pairwise_map_sim']:.4f} emb_sim={summary['mean_pairwise_emb_sim']:.4f} "
        f"redundant={summary['redundant_pair_ratio']:.4f} effective={summary['effective_motif_count']:.2f} "
        f"area={summary['effective_area_mean']:.1f} in_range={summary.get('area_in_range_ratio', 0.0):.4f} "
        f"area_over={summary.get('area_over_max_ratio', summary.get('effective_area_over_max_ratio', 0.0)):.4f} "
        f"logits_std={summary.get('logits_std', 0.0):.4f} coverage={summary['coverage_cosine']:.4f} "
        f"sel_border={summary.get('selected_border_mass_mean', 0.0):.4f} "
        f"sel_outer={summary.get('selected_outer_border_mass_mean', 0.0):.4f} "
        f"sel_fg={summary.get('selected_foreground_mass_mean', 0.0):.4f} "
        f"sel_H={summary.get('selection_entropy', 0.0):.4f} "
        f"sel_eff={summary.get('selection_effective_count', 0.0):.2f} "
        f"clean={summary.get('clean_candidate_count', 0.0):.2f} "
        f"hard_clean={summary.get('hard_clean_candidate_count', 0.0):.2f} "
        f"region_clean={summary.get('mean_region_clean_count', 0.0):.2f}/{summary.get('min_region_clean_count', 0.0):.2f} "
        f"t_cos={summary.get('teacher_motif_cosine', 0.0):.4f} "
        f"t_loss={summary.get('loss_teacher_align', 0.0):.4f} "
        f"aux_acc={summary.get('aux_accuracy', 0.0):.4f} aux_f1={summary.get('aux_macro_f1', 0.0):.4f}"
    )
    return summary


def run_train(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    config = _update_paths_from_args(config, args)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config=config)
    print(f"[Device] selected={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[Device] gpu={torch.cuda.get_device_name(device.index or torch.cuda.current_device())}")
    training_cfg = dict(config.get("training", {}) or {})
    audit_cfg = dict(config.get("audit", {}) or {})
    schedule_cfg = dict(config.get("schedule", {}) or {})
    quality_cfg = dict(config.get("quality_score", {}) or {})
    aux_cfg = dict(config.get("aux_supervision", {}) or {})
    teacher_alignment_cfg = dict(config.get("teacher_alignment", {}) or {})
    if args.synthetic and bool(teacher_alignment_cfg.get("enabled", False)):
        print("[TeacherAlign] synthetic mode: teacher alignment disabled for synthetic smoke.")
        teacher_alignment_cfg["enabled"] = False
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    checkpoint_dir = output_root / "checkpoints"
    history_path = output_root / "logs" / "stage1_history.csv"

    model_cfg = dict(config["model"])
    feature_ablation_cfg = dict(config.get("feature_ablation", {}) or {})
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 7)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
    )
    save_config(config, output_root)
    print(f"[Output] resolved_config={output_root / 'resolved_config.yaml'}")
    model = build_model(model_cfg).to(device)
    freeze_pixel_encoder = bool(model_cfg.get("freeze_pixel_encoder", True))
    _set_freeze_pixel_encoder(model, freeze=freeze_pixel_encoder)
    _print_trainable_summary(model)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for motif discovery Stage 1B")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(training_cfg.get("lr_motif", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    loss_cfg = dict(config.get("motif_loss", config.get("motif", {}).get("loss", {})) or {})
    loss_cfg.setdefault("height", int(model_cfg.get("image_size", 48)))
    loss_cfg.setdefault("width", int(model_cfg.get("image_size", 48)))
    criterion = MotifDiscoveryStage1Loss(loss_cfg).to(device)
    _apply_audit_config(model, audit_cfg=audit_cfg, loss_cfg=loss_cfg)
    teacher_model = _load_teacher_alignment_model(teacher_alignment_cfg, device=device)
    teacher_height = int(model_cfg.get("height", model_cfg.get("image_size", 48)))
    teacher_width = int(model_cfg.get("width", model_cfg.get("image_size", 48)))
    teacher_border_width = int(loss_cfg.get("border_width", model_cfg.get("border_width", 4)))

    epochs = int(getattr(args, "epochs", None) or training_cfg.get("epochs", 50))
    batch_size = int(config.get("data", {}).get("batch_size", 32))
    num_batches = getattr(args, "num_batches", None)
    max_train_batches = getattr(args, "max_train_batches", None)
    max_val_batches = getattr(args, "max_val_batches", None)
    if max_train_batches is None:
        max_train_batches = training_cfg.get("max_train_batches", num_batches)
    if max_val_batches is None:
        max_val_batches = training_cfg.get("max_val_batches", num_batches)
    target_effective_motifs = float(audit_cfg.get("target_effective_motifs", 8))
    train_loader_obj = None
    if args.synthetic:
        train_loader = lambda: _synthetic_loader(batch_size=batch_size, num_batches=int(num_batches or 2), device=device)
        val_loader = lambda: _synthetic_loader(batch_size=batch_size, num_batches=int(num_batches or 2), device=device)
    else:
        train_loader_obj = build_dataloader(config, split=str(args.split or "train"), shuffle=True)
        val_loader_obj = build_dataloader(config, split=str(args.val_split or "val"), shuffle=False)
        train_loader = lambda: train_loader_obj
        val_loader = lambda: val_loader_obj
    aux_num_classes = int(model_cfg.get("num_classes", aux_cfg.get("num_classes", 7)))
    aux_class_weights = _build_aux_class_weights(
        aux_cfg=aux_cfg,
        model_cfg=model_cfg,
        train_loader_obj=train_loader_obj,
        device=device,
    )

    initial_strength = _spatial_bias_strength(model_cfg, epoch=1)
    if hasattr(model, "set_spatial_bias_strength"):
        model.set_spatial_bias_strength(initial_strength)
    initial_weights = _scheduled_loss_weights(loss_cfg, schedule_cfg, epoch=1)
    criterion.set_loss_weights(initial_weights)
    with torch.no_grad():
        initial_metrics = _run_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader(),
            optimizer=None,
            device=device,
            epoch=0,
            split="initial_audit",
            max_batches=1,
            amp=False,
            grad_clip_norm=0.0,
            log_every=0,
            target_effective_motifs=target_effective_motifs,
            quality_cfg=quality_cfg,
            current_lambda_aux_ce=_scheduled_aux_lambda(aux_cfg, epoch=1),
            teacher_model=teacher_model,
            teacher_alignment_cfg=teacher_alignment_cfg,
            current_lambda_teacher_align=_scheduled_teacher_align_lambda(teacher_alignment_cfg, epoch=1),
            teacher_height=teacher_height,
            teacher_width=teacher_width,
            teacher_border_width=teacher_border_width,
            teacher_decay_active=_teacher_decay_active(teacher_alignment_cfg, epoch=1),
            aux_class_weights=aux_class_weights,
            aux_num_classes=aux_num_classes,
            coverage_boost_active=_coverage_boost_active(schedule_cfg, epoch=1),
            soft_border_boost_active=_soft_border_boost_active(schedule_cfg, epoch=1),
            synthetic=bool(args.synthetic),
            feature_ablation_cfg=feature_ablation_cfg,
            model_node_dim=int(model_cfg.get("node_dim", 7)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
    _write_json(output_root / "logs" / "initial_audit.json", initial_metrics)
    _save_checkpoint(checkpoint_dir / "initial.pth", model, optimizer, 0, -float("inf"), config, initial_metrics)
    print(f"[InitialAudit] saved={output_root / 'logs' / 'initial_audit.json'}")

    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    last_metrics: Dict[str, float] = {}
    print("[Checkpoint] save_best_metric=motif_quality_score mode=max")
    for epoch in range(1, epochs + 1):
        current_strength = _spatial_bias_strength(model_cfg, epoch=epoch)
        if hasattr(model, "set_spatial_bias_strength"):
            model.set_spatial_bias_strength(current_strength)
        current_weights = _scheduled_loss_weights(loss_cfg, schedule_cfg, epoch=epoch)
        current_lambda_aux_ce = _scheduled_aux_lambda(aux_cfg, epoch=epoch)
        current_lambda_teacher_align = _scheduled_teacher_align_lambda(teacher_alignment_cfg, epoch=epoch)
        teacher_decay_active = _teacher_decay_active(teacher_alignment_cfg, epoch=epoch)
        boost_active = _coverage_boost_active(schedule_cfg, epoch=epoch)
        soft_boost_active = _soft_border_boost_active(schedule_cfg, epoch=epoch)
        criterion.set_loss_weights(current_weights)
        print(
            f"[Schedule epoch {epoch:03d}] "
            f"spatial_bias_strength={current_strength:.4f} "
            f"lambda_div_map={current_weights['lambda_div_map']:.5f} "
            f"lambda_div_emb={current_weights['lambda_div_emb']:.5f} "
            f"lambda_coverage={current_weights['lambda_coverage']:.5f} "
            f"lambda_anchor={current_weights['lambda_anchor']:.5f} "
            f"lambda_border={current_weights['lambda_border']:.5f} "
            f"lambda_soft_border={current_weights['lambda_soft_border']:.5f} "
            f"lambda_outer_border={current_weights['lambda_outer_border']:.5f} "
            f"lambda_selected_border={current_weights['lambda_selected_border']:.5f} "
            f"lambda_selected_outer_border={current_weights['lambda_selected_outer_border']:.5f} "
            f"lambda_selected_foreground={current_weights['lambda_selected_foreground']:.5f} "
            f"lambda_selected_diversity={current_weights['lambda_selected_diversity']:.5f} "
            f"lambda_selection_entropy={current_weights['lambda_selection_entropy']:.5f} "
            f"lambda_clean_count={current_weights['lambda_clean_count']:.5f} "
            f"lambda_clean_mean={current_weights['lambda_clean_mean']:.5f} "
            f"lambda_region_clean={current_weights['lambda_region_clean']:.5f} "
            f"lambda_teacher_align={current_lambda_teacher_align:.5f} "
            f"teacher_decay_active={teacher_decay_active} "
            f"lambda_aux_ce={current_lambda_aux_ce:.5f} "
            f"lambda_entropy={current_weights['lambda_entropy']:.5f} "
            f"lambda_area={current_weights['lambda_area']:.5f} "
            f"coverage_boost_active={boost_active} "
            f"soft_border_boost_active={soft_boost_active}"
        )
        train_metrics = _run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader(),
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            split="train",
            max_batches=max_train_batches,
            amp=bool(training_cfg.get("amp", True)),
            grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
            log_every=int(training_cfg.get("log_every", 20)),
            target_effective_motifs=target_effective_motifs,
            quality_cfg=quality_cfg,
            current_lambda_aux_ce=current_lambda_aux_ce,
            teacher_model=teacher_model,
            teacher_alignment_cfg=teacher_alignment_cfg,
            current_lambda_teacher_align=current_lambda_teacher_align,
            teacher_height=teacher_height,
            teacher_width=teacher_width,
            teacher_border_width=teacher_border_width,
            teacher_decay_active=teacher_decay_active,
            aux_class_weights=aux_class_weights,
            aux_num_classes=aux_num_classes,
            coverage_boost_active=boost_active,
            soft_border_boost_active=soft_boost_active,
            synthetic=bool(args.synthetic),
            feature_ablation_cfg=feature_ablation_cfg,
            model_node_dim=int(model_cfg.get("node_dim", 7)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
        history_rows = [{**{field: "" for field in HISTORY_FIELDS}, **train_metrics, "epoch": epoch, "split": "train"}]
        val_metrics = train_metrics
        if epoch % int(training_cfg.get("val_every", 1)) == 0:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model=model,
                    criterion=criterion,
                    loader=val_loader(),
                    optimizer=None,
                    device=device,
                    epoch=epoch,
                    split="val",
                    max_batches=max_val_batches,
                    amp=False,
                    grad_clip_norm=0.0,
                    log_every=0,
                    target_effective_motifs=target_effective_motifs,
                    quality_cfg=quality_cfg,
                    current_lambda_aux_ce=current_lambda_aux_ce,
                    teacher_model=teacher_model,
                    teacher_alignment_cfg=teacher_alignment_cfg,
                    current_lambda_teacher_align=current_lambda_teacher_align,
                    teacher_height=teacher_height,
                    teacher_width=teacher_width,
                    teacher_border_width=teacher_border_width,
                    teacher_decay_active=teacher_decay_active,
                    aux_class_weights=aux_class_weights,
                    aux_num_classes=aux_num_classes,
                    coverage_boost_active=boost_active,
                    soft_border_boost_active=soft_boost_active,
                    synthetic=bool(args.synthetic),
                    feature_ablation_cfg=feature_ablation_cfg,
                    model_node_dim=int(model_cfg.get("node_dim", 7)),
                    model_edge_dim=int(model_cfg.get("edge_dim", 5)),
                )
            history_rows.append({**{field: "" for field in HISTORY_FIELDS}, **val_metrics, "epoch": epoch, "split": "val"})
        _append_history(history_path, history_rows)

        last_metrics = val_metrics
        score = float(val_metrics["motif_quality_score"])
        if score > best_score:
            old_best = best_score
            best_score = score
            best_metrics = dict(val_metrics)
            _save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, best_score, config, best_metrics)
            print(
                f"[Checkpoint] new best epoch={epoch} old_best={old_best:.6f} "
                f"new_best={best_score:.6f} area={val_metrics.get('effective_area_mean', float('nan')):.2f} "
                f"in_range={val_metrics.get('area_in_range_ratio', float('nan')):.4f} "
                f"map_sim={val_metrics.get('mean_pairwise_map_sim', float('nan')):.4f} "
                f"redundant={val_metrics.get('redundant_pair_ratio', float('nan')):.4f} "
                f"coverage={val_metrics.get('coverage_cosine', float('nan')):.4f}"
            )
        _save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, epoch, best_score, config, val_metrics)
        if epoch % int(training_cfg.get("save_every", 10)) == 0:
            _save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_score, config, val_metrics)

    print(f"[Output] history={history_path}")
    print(f"[Output] best={checkpoint_dir / 'best.pth'}")
    print(f"[Output] last={checkpoint_dir / 'last.pth'}")
    return {"best_score": best_score, **best_metrics, **{f"last_{k}": v for k, v in last_metrics.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8m_motif_discovery_stage1_k16.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", default=None)
    parser.add_argument("--persistent_workers", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Run tiny synthetic smoke training without graph repo.")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config = _update_paths_from_args(config, args)
    run_train(config, args)


if __name__ == "__main__":
    main()
