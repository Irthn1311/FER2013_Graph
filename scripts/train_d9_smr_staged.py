"""Train D9-SMR-B: staged motif rescue with a pooled classifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_path, save_config  # noqa: E402
from data.labels import EMOTION_NAMES  # noqa: E402
from evaluation.evaluator import save_confusion_matrix  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics, confusion_matrix_array  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.losses import compute_class_weights  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402
from utils.train_freeze import (  # noqa: E402
    count_trainable_params,
    freeze_by_keywords,
    set_requires_grad,
    trainable_parameter_names,
    unfreeze_by_keywords,
)

LOGGER = logging.getLogger("d9_smr_staged")

PHASES = ("warmup", "classifier", "finetune")

HISTORY_FIELDS = [
    "phase",
    "epoch",
    "train_loss",
    "train_cls_loss",
    "train_motif_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_weighted_f1",
    "train_selected_border",
    "train_selected_foreground",
    "train_map_entropy",
    "train_effective_count",
    "train_map_similarity",
    "train_redundancy",
    "train_motif_quality_score",
    "val_loss",
    "val_cls_loss",
    "val_motif_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_selected_border",
    "val_selected_foreground",
    "val_map_entropy",
    "val_effective_count",
    "val_map_similarity",
    "val_redundancy",
    "val_motif_quality_score",
    "lr",
    "epoch_seconds",
]


def _setup_logger() -> None:
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _log(message: str) -> None:
    LOGGER.info(message)


def _update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    output = dict(cfg.get("output", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    staged = dict(cfg.get("staged", {}) or {})
    if args.output_root:
        output["dir"] = str(args.output_root)
        paths["resolved_output_root"] = str(args.output_root)
    elif output.get("dir"):
        paths["resolved_output_root"] = str(output["dir"])
    if args.graph_repo_path:
        paths["graph_repo_path"] = str(args.graph_repo_path)
        data["graph_repo_path"] = str(args.graph_repo_path)
    if sys.platform.startswith("win") and args.num_workers is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    for phase in PHASES:
        phase_cfg = dict(staged.get(phase, {}) or {})
        if not phase_cfg:
            continue
        if args.epochs is not None:
            phase_cfg["epochs"] = int(args.epochs)
        if args.max_train_batches is not None:
            phase_cfg["max_train_batches"] = int(args.max_train_batches)
        if args.max_val_batches is not None:
            phase_cfg["max_val_batches"] = int(args.max_val_batches)
        staged[phase] = phase_cfg
    cfg["paths"] = paths
    cfg["output"] = output
    cfg["data"] = data
    cfg["staged"] = staged
    return cfg


def _count_train_labels(loader_obj: Any, num_classes: int) -> list[int] | None:
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
        _log(f"[ClassWeights] label count fallback failed: {exc}")
        return None
    return [int(v) for v in counts.tolist()]


def _build_class_weights(config: Dict[str, Any], train_loader, device: torch.device) -> torch.Tensor | None:
    loss_cfg = dict(config.get("loss", {}) or {})
    training_cfg = dict(config.get("training", {}) or {})
    enabled = bool(loss_cfg.get("use_class_weights", training_cfg.get("use_class_weights", training_cfg.get("class_weights", False))))
    if not enabled:
        _log("[ClassWeights] disabled")
        return None
    num_classes = int(config.get("model", {}).get("num_classes", config.get("data", {}).get("num_classes", 7)))
    counts = loss_cfg.get("class_counts") or training_cfg.get("class_counts") or _count_train_labels(train_loader, num_classes)
    if counts is None:
        _log("[ClassWeights] disabled: no counts available")
        return None
    weights = compute_class_weights(
        counts,
        normalize_mean=True,
        power=float(loss_cfg.get("class_weight_power", 1.0)),
    ).to(device=device)
    max_weight = float(loss_cfg.get("max_class_weight", 0.0) or 0.0)
    if max_weight > 0:
        weights = weights.clamp(max=max_weight)
        weights = weights / weights.mean().clamp_min(1e-8)
    _log(f"[ClassWeights] counts={list(counts)} weights={[round(float(v), 4) for v in weights.detach().cpu()]}")
    return weights


def _build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any], epochs: int, lr: float):
    name = str(training_cfg.get("scheduler", "none")).lower()
    if name in {"", "none"}:
        return None
    if name in {"warmup_cosine", "cosine_warmup"}:
        warmup = max(0, int(training_cfg.get("warmup_epochs", 1)))
        min_lr = float(training_cfg.get("min_lr", 1e-6))
        min_factor = min_lr / max(float(lr), 1e-12)

        def lr_lambda(epoch_idx: int) -> float:
            step = int(epoch_idx) + 1
            if warmup > 0 and step <= warmup:
                return max(float(step) / float(warmup), min_factor)
            span = max(1, int(epochs) - warmup)
            progress = min(max(float(step - warmup) / float(span), 0.0), 1.0)
            return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported D9-SMR scheduler={name!r}")


def _phase_config(config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    staged = dict(config.get("staged", {}) or {})
    cfg = dict(staged.get(phase, {}) or {})
    if not cfg:
        raise KeyError(f"Missing staged.{phase} config")
    return cfg


def _apply_phase_freeze(model: torch.nn.Module, phase: str, phase_cfg: Dict[str, Any]) -> Dict[str, Any]:
    freeze_cfg = phase_cfg.get("freeze", "none")
    set_requires_grad(model, True)
    frozen: list[str] = []
    unfrozen: list[str] = []
    if isinstance(freeze_cfg, str):
        mode = freeze_cfg.lower()
        if mode in {"", "none"}:
            pass
        elif mode == "freeze_motif":
            frozen.extend(freeze_by_keywords(model, ["motif_discovery"]))
        elif mode == "freeze_encoder_motif":
            frozen.extend(freeze_by_keywords(model, ["pixel_encoder", "motif_discovery"]))
        elif mode == "light_unfreeze":
            set_requires_grad(model, False)
            unfrozen.extend(unfreeze_by_keywords(model, _light_unfreeze_keywords()))
        else:
            raise ValueError(f"Unsupported freeze mode for phase={phase}: {freeze_cfg!r}")
    elif isinstance(freeze_cfg, dict):
        if bool(freeze_cfg.get("train_classifier_only", False)):
            set_requires_grad(model, False)
            unfrozen.extend(unfreeze_by_keywords(model, ["motif_relation_classifier"]))
        else:
            if bool(freeze_cfg.get("freeze_encoder", False)):
                frozen.extend(freeze_by_keywords(model, ["pixel_encoder"]))
            if bool(freeze_cfg.get("freeze_motif", False)):
                frozen.extend(freeze_by_keywords(model, ["motif_discovery"]))
            if bool(freeze_cfg.get("train_classifier", False)):
                unfrozen.extend(unfreeze_by_keywords(model, ["motif_relation_classifier"]))
            if bool(freeze_cfg.get("train_motif_queries", False)):
                unfrozen.extend(unfreeze_by_keywords(model, _motif_query_keywords()))
        frozen.extend(freeze_by_keywords(model, freeze_cfg.get("freeze_keywords", [])))
        unfrozen.extend(unfreeze_by_keywords(model, freeze_cfg.get("unfreeze_keywords", [])))
    else:
        raise TypeError(f"staged.{phase}.freeze must be string or mapping")
    counts = count_trainable_params(model)
    names = trainable_parameter_names(model, limit=80)
    _log(
        f"[Freeze phase={phase}] total={counts['total']:,} trainable={counts['trainable']:,} "
        f"frozen={counts['frozen']:,} frozen_matches={len(set(frozen))} unfrozen_matches={len(set(unfrozen))}"
    )
    _log(f"[Freeze phase={phase}] trainable_names={names}")
    return {
        "phase": phase,
        "freeze": freeze_cfg,
        "counts": counts,
        "frozen_matches": sorted(set(frozen))[:120],
        "unfrozen_matches": sorted(set(unfrozen))[:120],
        "trainable_names": names,
    }


def _motif_query_keywords() -> list[str]:
    return [
        "motif_discovery.motif_queries",
        "motif_discovery.query_proj",
        "motif_discovery.key_proj",
        "motif_discovery.value_proj",
        "motif_discovery.score_head",
    ]


def _light_unfreeze_keywords() -> list[str]:
    return ["motif_relation_classifier", *_motif_query_keywords()]


def _scalar(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu())
    if value is None:
        return float(default)
    return float(value)


def _prediction_distribution(y_pred: list[int], num_classes: int) -> list[int]:
    if not y_pred:
        return [0 for _ in range(int(num_classes))]
    return np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=int(num_classes)).tolist()


def _motif_score(metrics: Dict[str, float], prefix: str = "") -> float:
    key = (lambda name: f"{prefix}_{name}" if prefix else name)
    selected_foreground = float(metrics.get(key("selected_foreground"), 0.0))
    selected_border = float(metrics.get(key("selected_border"), 0.0))
    redundancy = float(metrics.get(key("redundancy"), 0.0))
    map_similarity = float(metrics.get(key("map_similarity"), 0.0))
    effective_count = float(metrics.get(key("effective_count"), 0.0))
    entropy_penalty = max(effective_count - 8.0, 0.0) / 8.0
    return selected_foreground - selected_border - 0.2 * redundancy - 0.1 * map_similarity - 0.1 * entropy_penalty


def _monitor_value(row: Dict[str, Any], monitor: str) -> float:
    if monitor in row:
        return float(row[monitor])
    if monitor == "motif_quality_score" and "val_motif_quality_score" in row:
        return float(row["val_motif_quality_score"])
    if monitor == "macro_f1" and "val_macro_f1" in row:
        return float(row["val_macro_f1"])
    raise KeyError(f"Monitor {monitor!r} not found in row keys")


def _extract_motif_metrics(out: Dict[str, Any], loss_dict: Dict[str, Any]) -> Dict[str, float]:
    audit = out.get("motif_audit", {}) if isinstance(out.get("motif_audit", {}), dict) else {}
    selected_border = _scalar(loss_dict.get("selected_outer_border_mass_mean", loss_dict.get("selected_border_mass_mean")))
    selected_foreground = _scalar(loss_dict.get("selected_foreground_mass_mean"), default=float("nan"))
    selection_entropy = _scalar(loss_dict.get("selection_entropy"))
    effective_count = _scalar(loss_dict.get("selection_effective_count"))
    map_similarity = _scalar(loss_dict.get("mean_pairwise_map_sim", audit.get("mean_pairwise_map_sim")))
    redundancy = _scalar(loss_dict.get("selected_pairwise_map_sim", audit.get("redundancy_rate")))
    map_entropy = _scalar(audit.get("assignment_entropy_mean"), default=_scalar(loss_dict.get("effective_area_mean")))
    clean_score = _scalar(loss_dict.get("clean_score_mean", audit.get("clean_score_mean")))
    return {
        "selected_border": selected_border,
        "selected_foreground": selected_foreground,
        "selection_entropy": selection_entropy,
        "effective_count": effective_count,
        "map_similarity": map_similarity,
        "redundancy": redundancy,
        "map_entropy": map_entropy,
        "clean_score": clean_score,
        "loss_selected_border": _scalar(loss_dict.get("loss_selected_border")),
        "loss_selected_foreground": _scalar(loss_dict.get("loss_selected_foreground")),
        "loss_selected_diversity": _scalar(loss_dict.get("loss_selected_diversity")),
        "loss_entropy": _scalar(loss_dict.get("loss_entropy")),
    }


def _run_epoch(
    *,
    phase: str,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    feature_ablation_cfg: Dict[str, Any],
    model_node_dim: int,
    model_edge_dim: int,
    class_weights: torch.Tensor | None,
    motif_criterion: MotifDiscoveryStage1Loss,
    cls_weight: float,
    motif_weight: float,
    max_batches: int | None,
    amp: bool,
    grad_clip_norm: float,
    log_interval: int,
    output_root: Path | None = None,
    epoch: int | None = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    y_true: list[int] = []
    y_pred: list[int] = []
    graph_ids: list[int] = []
    logits_rows: list[list[float]] = []
    sums = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "motif_loss": 0.0,
        "selected_border": 0.0,
        "selected_foreground": 0.0,
        "selection_entropy": 0.0,
        "effective_count": 0.0,
        "map_similarity": 0.0,
        "redundancy": 0.0,
        "map_entropy": 0.0,
        "clean_score": 0.0,
        "loss_selected_border": 0.0,
        "loss_selected_foreground": 0.0,
        "loss_selected_diversity": 0.0,
        "loss_entropy": 0.0,
    }
    count = 0
    first_logged = False
    autocast_enabled = bool(amp and device.type == "cuda")
    for batch_idx, raw_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        raw_batch = move_to_device(raw_batch, device)
        model_batch = dict(raw_batch)
        original_node_dim = int(model_batch["x"].shape[-1])
        original_edge_dim = int(model_batch["edge_attr"].shape[-1])
        model_batch = apply_feature_ablation(model_batch, feature_ablation_cfg)
        assert_feature_dims(model_batch, node_dim=model_node_dim, edge_dim=model_edge_dim)
        labels = model_batch["y"].long()
        if not first_logged:
            _log(
                f"[D9SMRBatch phase={phase}] original_node_dim={original_node_dim} "
                f"masked_node_dim={int(model_batch['x'].shape[-1])} original_edge_dim={original_edge_dim} "
                f"masked_edge_dim={int(model_batch['edge_attr'].shape[-1])} x_shape={list(model_batch['x'].shape)}"
            )
            first_logged = True
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                out = model(model_batch)
                cls_loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
                motif_loss_dict = motif_criterion(out, raw_batch)
                motif_loss = motif_loss_dict["loss"]
                loss = float(cls_weight) * cls_loss + float(motif_weight) * motif_loss
        _check_finite(loss, out)
        if is_train:
            loss.backward()
            trainable = [param for param in model.parameters() if param.requires_grad]
            if trainable:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(grad_clip_norm))
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(f"Non-finite grad norm at phase={phase} batch={batch_idx}")
            optimizer.step()
        batch_size = int(labels.shape[0])
        logits = out["logits"].detach().float()
        pred = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        if not is_train:
            graph_ids.extend(model_batch["graph_id"].detach().cpu().tolist())
            logits_rows.extend(logits.cpu().tolist())
        motif_metrics = _extract_motif_metrics(out, motif_loss_dict)
        sums["loss"] += float(loss.detach().cpu()) * batch_size
        sums["cls_loss"] += float(cls_loss.detach().cpu()) * batch_size
        sums["motif_loss"] += float(motif_loss.detach().cpu()) * batch_size
        for key, value in motif_metrics.items():
            if math.isnan(float(value)):
                continue
            sums[key] += float(value) * batch_size
        count += batch_size
        if is_train and log_interval > 0 and (batch_idx + 1) % int(log_interval) == 0:
            _log(
                f"[train phase={phase} batch={batch_idx + 1}] loss={float(loss.detach().cpu()):.6f} "
                f"cls={float(cls_loss.detach().cpu()):.6f} motif={float(motif_loss.detach().cpu()):.6f} "
                f"sel_border={motif_metrics['selected_border']:.4f} "
                f"sel_fg={motif_metrics['selected_foreground']:.4f} "
                f"entropy={motif_metrics['selection_entropy']:.4f} eff={motif_metrics['effective_count']:.4f} "
                f"pred_dist={_prediction_distribution(pred.detach().cpu().tolist(), int(out['logits'].shape[-1]))}"
            )
    prefix = "train" if is_train else "val"
    cls_metrics = compute_metrics(y_true, y_pred) if y_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    row: Dict[str, Any] = {
        f"{prefix}_loss": sums["loss"] / max(count, 1),
        f"{prefix}_cls_loss": sums["cls_loss"] / max(count, 1),
        f"{prefix}_motif_loss": sums["motif_loss"] / max(count, 1),
        f"{prefix}_accuracy": float(cls_metrics["accuracy"]),
        f"{prefix}_macro_f1": float(cls_metrics["macro_f1"]),
        f"{prefix}_weighted_f1": float(cls_metrics["weighted_f1"]),
        f"{prefix}_selected_border": sums["selected_border"] / max(count, 1),
        f"{prefix}_selected_foreground": sums["selected_foreground"] / max(count, 1),
        f"{prefix}_selection_entropy": sums["selection_entropy"] / max(count, 1),
        f"{prefix}_effective_count": sums["effective_count"] / max(count, 1),
        f"{prefix}_map_similarity": sums["map_similarity"] / max(count, 1),
        f"{prefix}_redundancy": sums["redundancy"] / max(count, 1),
        f"{prefix}_map_entropy": sums["map_entropy"] / max(count, 1),
        f"{prefix}_clean_score": sums["clean_score"] / max(count, 1),
        f"{prefix}_loss_selected_border": sums["loss_selected_border"] / max(count, 1),
        f"{prefix}_loss_selected_foreground": sums["loss_selected_foreground"] / max(count, 1),
        f"{prefix}_loss_selected_diversity": sums["loss_selected_diversity"] / max(count, 1),
        f"{prefix}_loss_entropy": sums["loss_entropy"] / max(count, 1),
        f"{prefix}_pred_distribution": _prediction_distribution(y_pred, int(model_node_dim if False else 7)),
    }
    row[f"{prefix}_motif_quality_score"] = _motif_score(row, prefix=prefix)
    if not is_train and y_true and output_root is not None and epoch is not None:
        _save_val_artifacts(
            output_root=output_root,
            phase=phase,
            epoch=int(epoch),
            y_true=y_true,
            y_pred=y_pred,
            graph_ids=graph_ids,
            logits_rows=logits_rows,
            metrics=row,
        )
    return row


def _check_finite(loss: torch.Tensor, out: Dict[str, Any]) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite D9-SMR loss: {float(loss.detach().cpu())}")
    for key in ("logits", "motif_maps", "motif_embeddings", "selection_weights"):
        value = out.get(key)
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite D9-SMR output: {key}")


def _save_val_artifacts(
    *,
    output_root: Path,
    phase: str,
    epoch: int,
    y_true: list[int],
    y_pred: list[int],
    graph_ids: list[int],
    logits_rows: list[list[float]],
    metrics: Dict[str, Any],
) -> None:
    metrics_dir = output_root / phase / "metrics"
    figures_dir = output_root / phase / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report_dict(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)
    payload = {
        "accuracy": metrics["val_accuracy"],
        "macro_f1": metrics["val_macro_f1"],
        "weighted_f1": metrics["val_weighted_f1"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": _prediction_distribution(y_pred, 7),
        "motif_metrics": {k: v for k, v in metrics.items() if k.startswith("val_") and ("motif" in k or "selected" in k or "entropy" in k or "similarity" in k or "redundancy" in k or "clean" in k)},
    }
    with (metrics_dir / f"val_metrics_epoch_{epoch:03d}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (metrics_dir / f"val_predictions_epoch_{epoch:03d}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        num_classes = len(logits_rows[0]) if logits_rows else 0
        writer.writerow(["graph_id", "y_true", "y_pred"] + [f"logit_{i}" for i in range(num_classes)])
        for gid, yt, yp, row in zip(graph_ids, y_true, y_pred, logits_rows):
            writer.writerow([gid, yt, yp] + list(row))
    save_confusion_matrix(cm, figures_dir / f"val_confusion_matrix_epoch_{epoch:03d}.png")


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    phase: str,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    monitor: str,
    mode: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    freeze_summary: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_metric_name": str(monitor),
        "best_metric_mode": str(mode),
        "freeze_summary": freeze_summary,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    _log(f"[CheckpointLoad] loaded={checkpoint_path}")
    return checkpoint


def _run_phase(
    *,
    phase: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    train_loader,
    val_loader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    output_root: Path,
) -> Dict[str, Any]:
    model_cfg = dict(config.get("model", {}) or {})
    training_cfg = dict(config.get("training", {}) or {})
    logging_cfg = dict(config.get("logging", {}) or {})
    phase_cfg = _phase_config(config, phase)
    phase_dir = output_root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    freeze_summary = _apply_phase_freeze(model, phase, phase_cfg)
    lr = float(phase_cfg.get("lr", training_cfg.get("lr", 3e-4)))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(phase_cfg.get("epochs", training_cfg.get("epochs", 1)))
    scheduler = _build_scheduler(optimizer, training_cfg, epochs=epochs, lr=lr)
    motif_cfg = dict(config.get("motif_loss", {}) or {})
    motif_cfg.setdefault("height", int(model_cfg.get("height", model_cfg.get("image_size", 48))))
    motif_cfg.setdefault("width", int(model_cfg.get("width", model_cfg.get("image_size", 48))))
    motif_criterion = MotifDiscoveryStage1Loss(motif_cfg).to(device)
    monitor = str(phase_cfg.get("monitor", "val_macro_f1"))
    mode = str(phase_cfg.get("mode", "max")).lower()
    if phase != "warmup" and (monitor != "val_macro_f1" or mode != "max"):
        raise ValueError("D9-SMR classifier/finetune phases must monitor val_macro_f1 mode=max")
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported checkpoint mode={mode!r}")
    best_metric = -float("inf") if mode == "max" else float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    history_path = phase_dir / "logs" / "history.csv"
    max_train_batches = phase_cfg.get("max_train_batches", training_cfg.get("max_train_batches"))
    max_val_batches = phase_cfg.get("max_val_batches", training_cfg.get("max_val_batches"))
    log_interval = int(logging_cfg.get("log_interval", 100))
    _log(
        f"[PhaseStart] phase={phase} epochs={epochs} lr={lr} monitor={monitor} mode={mode} "
        f"cls_weight={float(phase_cfg.get('cls_weight', 1.0))} motif_weight={float(phase_cfg.get('motif_weight', 0.0))} "
        f"max_train_batches={max_train_batches} max_val_batches={max_val_batches}"
    )
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _run_epoch(
            phase=phase,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            feature_ablation_cfg=dict(config.get("feature_ablation", {}) or {}),
            model_node_dim=int(model_cfg.get("node_dim", 3)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
            class_weights=class_weights,
            motif_criterion=motif_criterion,
            cls_weight=float(phase_cfg.get("cls_weight", 1.0)),
            motif_weight=float(phase_cfg.get("motif_weight", 0.0)),
            max_batches=max_train_batches,
            amp=bool(training_cfg.get("amp", True)),
            grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
            log_interval=log_interval,
        )
        val_metrics = _run_epoch(
            phase=phase,
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            feature_ablation_cfg=dict(config.get("feature_ablation", {}) or {}),
            model_node_dim=int(model_cfg.get("node_dim", 3)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
            class_weights=class_weights,
            motif_criterion=motif_criterion,
            cls_weight=float(phase_cfg.get("cls_weight", 1.0)),
            motif_weight=float(phase_cfg.get("motif_weight", 0.0)),
            max_batches=max_val_batches,
            amp=False,
            grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
            log_interval=0,
            output_root=output_root,
            epoch=epoch,
        )
        if scheduler is not None:
            scheduler.step()
        row = {
            "phase": phase,
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - start,
        }
        _append_history(history_path, row)
        _append_history(output_root / "staged_history.csv", row)
        current = _monitor_value(row, monitor)
        improved = current > best_metric if mode == "max" else current < best_metric
        if improved:
            best_metric = current
            best_epoch = epoch
            best_metrics = dict(row)
            _save_checkpoint(
                phase_dir / "checkpoints" / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                phase=phase,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                monitor=monitor,
                mode=mode,
                config=config,
                metrics=row,
                freeze_summary=freeze_summary,
            )
            _log(f"[Checkpoint phase={phase}] best epoch={epoch} {monitor}={best_metric:.6f}")
        _save_checkpoint(
            phase_dir / "checkpoints" / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase,
            epoch=epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            monitor=monitor,
            mode=mode,
            config=config,
            metrics=row,
            freeze_summary=freeze_summary,
        )
        _log(
            f"[epoch phase={phase} {epoch:03d}] train_loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_motif_score={row['val_motif_quality_score']:.4f} "
            f"sel_border={row['val_selected_border']:.4f} sel_fg={row['val_selected_foreground']:.4f} "
            f"best_epoch={best_epoch}"
        )
    if best_epoch < 0:
        raise RuntimeError(f"No best checkpoint produced for phase={phase}")
    best_ckpt = phase_dir / "checkpoints" / "best.pth"
    _load_model_checkpoint(model, best_ckpt, device)
    _copy_best_epoch_artifacts(phase_dir=phase_dir, best_epoch=best_epoch)
    summary = {
        "phase": phase,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "monitor": monitor,
        "mode": mode,
        "best_checkpoint": str(best_ckpt),
        "best_metrics": best_metrics,
        "freeze_summary": freeze_summary,
    }
    with (phase_dir / "metrics" / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _copy_best_epoch_artifacts(*, phase_dir: Path, best_epoch: int) -> None:
    metrics_dir = phase_dir / "metrics"
    best_json = metrics_dir / f"val_metrics_epoch_{best_epoch:03d}.json"
    if best_json.exists():
        with best_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        with (metrics_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


@torch.no_grad()
def _visualize_motif_contact_sheet(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: Path,
    max_samples: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _log(f"[Visualize] skipped: matplotlib unavailable ({exc})")
        return
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cfg = dict(config.get("feature_ablation", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    saved = 0
    for raw_batch in loader:
        raw_batch = move_to_device(raw_batch, device)
        model_batch = apply_feature_ablation(dict(raw_batch), feature_cfg)
        assert_feature_dims(
            model_batch,
            node_dim=int(model_cfg.get("node_dim", 3)),
            edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
        out = model(model_batch)
        logits = out["logits"].detach().float()
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        weights = out["selection_weights"].detach().float()
        maps = out["motif_maps"].detach().float()
        images = raw_batch["x"][..., 0].detach().float().reshape(-1, 48, 48)
        labels = raw_batch["y"].detach().long()
        graph_ids = raw_batch["graph_id"].detach().long()
        for i in range(images.shape[0]):
            if saved >= int(max_samples):
                return
            topk = torch.topk(weights[i], k=min(4, weights.shape[1])).indices.cpu().tolist()
            fig, axes = plt.subplots(1, 1 + len(topk), figsize=(3.0 * (1 + len(topk)), 3.0))
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            axes[0].imshow(images[i].cpu().numpy(), cmap="gray")
            axes[0].set_title(
                f"id={int(graph_ids[i])}\ny={EMOTION_NAMES[int(labels[i])]} pred={EMOTION_NAMES[int(pred[i])]}\nconf={float(probs[i, pred[i]].cpu()):.3f}",
                fontsize=8,
            )
            axes[0].axis("off")
            for ax, motif_idx in zip(axes[1:], topk):
                ax.imshow(images[i].cpu().numpy(), cmap="gray")
                ax.imshow(maps[i, motif_idx].cpu().numpy(), cmap="magma", alpha=0.55)
                ax.set_title(f"m{motif_idx} w={float(weights[i, motif_idx].cpu()):.3f}", fontsize=8)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / f"sample_{saved:03d}_graph_{int(graph_ids[i])}.png", dpi=140)
            plt.close(fig)
            saved += 1


def _write_summary_files(output_root: Path, phase_summaries: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
    summary = {
        "experiment": config.get("experiment", {}).get("name", "d9_smr_b"),
        "motif_score_formula": config.get("staged", {}).get("motif_score_formula"),
        "baselines": {
            "d9_rg_mr_b_original_macro_f1": 0.05529953917050691,
            "d9_pooled_mlp_no_teacher_best_macro_f1": 0.167418567328483,
            "tgms_alpha_02_macro_f1": 0.1642,
        },
        "phases": phase_summaries,
    }
    with (output_root / "staged_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with (output_root / "staged_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "phase",
            "best_epoch",
            "monitor",
            "best_metric",
            "val_macro_f1",
            "val_accuracy",
            "val_weighted_f1",
            "val_motif_quality_score",
            "val_selected_border",
            "val_selected_foreground",
            "val_map_entropy",
            "val_effective_count",
            "val_map_similarity",
            "val_redundancy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for phase in PHASES:
            item = phase_summaries.get(phase)
            if not item:
                continue
            metrics = dict(item.get("best_metrics", {}) or {})
            writer.writerow({
                "phase": phase,
                "best_epoch": item.get("best_epoch"),
                "monitor": item.get("monitor"),
                "best_metric": item.get("best_metric"),
                **{field: metrics.get(field, "") for field in fieldnames if field.startswith("val_")},
            })
    _write_report(output_root, summary)


def _write_report(output_root: Path, summary: Dict[str, Any]) -> None:
    phases = summary.get("phases", {})
    lines = [
        "# D9-SMR-B Final Report",
        "",
        "## Setup",
        "- Variant: D9-SMR-B, staged motif rescue with pooled MLP classifier.",
        "- Feature B: node indices [0, 1, 2], edge indices [0, 1, 2, 3, 4].",
        f"- Motif score formula: `{summary.get('motif_score_formula')}`.",
        "",
    ]
    for phase in PHASES:
        item = phases.get(phase)
        if not item:
            continue
        metrics = dict(item.get("best_metrics", {}) or {})
        lines.extend(
            [
                f"## Phase {phase}",
                f"- Best epoch: {item.get('best_epoch')}.",
                f"- Monitor: {item.get('monitor')} {item.get('mode')}; best metric: {_fmt(item.get('best_metric'))}.",
                f"- val_macro_f1: {_fmt(metrics.get('val_macro_f1'))}; val_accuracy: {_fmt(metrics.get('val_accuracy'))}; val_weighted_f1: {_fmt(metrics.get('val_weighted_f1'))}.",
                f"- motif_score: {_fmt(metrics.get('val_motif_quality_score'))}; selected_border: {_fmt(metrics.get('val_selected_border'))}; selected_foreground: {_fmt(metrics.get('val_selected_foreground'))}.",
                f"- map_entropy: {_fmt(metrics.get('val_map_entropy'))}; effective_count: {_fmt(metrics.get('val_effective_count'))}; map_similarity: {_fmt(metrics.get('val_map_similarity'))}; redundancy: {_fmt(metrics.get('val_redundancy'))}.",
                f"- Checkpoint: `{item.get('best_checkpoint')}`.",
                "",
            ]
        )
    finetune = phases.get("finetune", {})
    classifier = phases.get("classifier", {})
    best_macro = max(
        float((classifier.get("best_metrics", {}) or {}).get("val_macro_f1", 0.0) or 0.0),
        float((finetune.get("best_metrics", {}) or {}).get("val_macro_f1", 0.0) or 0.0),
    )
    lines.extend(
        [
            "## Comparison",
            "- D9-RG-MR-B original: macro F1 about 0.055.",
            "- D9 pooled MLP no-teacher best local baseline: macro F1 0.1674.",
            "- D9-TGMS alpha 0.2: macro F1 0.1642.",
            f"- D9-SMR-B best observed in this run: macro F1 {_fmt(best_macro)}.",
            "",
            "## Motif Deletion",
            "- Not run by the staged trainer automatically. Use `python -m scripts.evaluate_d9_motif_deletion --config configs/experiments/d9_smr_b.yaml --checkpoint <finetune_best> --split val --max_samples 300 --top_k 1,3`.",
            "",
            "## Decision Rule",
        ]
    )
    if best_macro > 0.20:
        decision = "Continue D9-SMR: classifier/finetune passed the 0.20 macro F1 gate."
    elif best_macro >= 0.16:
        decision = "Treat as research/audit direction unless motif metrics improve strongly; it has not clearly beaten the pooled no-teacher baseline."
    else:
        decision = "Stop D9 unless motif metrics show a large rescue; classification remains at or below prior pooled baselines."
    lines.extend(["- " + decision, ""])
    with (output_root / "final_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", "--env", default=None, choices=["local", "kaggle"])
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--phase", default="all", choices=["all", *PHASES])
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    return parser.parse_args()


def main() -> None:
    _setup_logger()
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    if not bool(config.get("staged", {}).get("enabled", False)):
        raise ValueError("D9-SMR requires staged.enabled=true")
    training_cfg = dict(config.get("training", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = resolve_device(args.device, config=config)
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    output_root = output_root or PROJECT_ROOT / "outputs" / "d9_smr_b"
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    _log(f"[Device] selected={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        _log(f"[Device] gpu={torch.cuda.get_device_name(device.index or torch.cuda.current_device())}")
    _log(f"[Output] run_dir={output_root}")
    _log(f"[MotifScore] {config.get('staged', {}).get('motif_score_formula')}")
    log_feature_ablation(
        dict(config.get("feature_ablation", {}) or {}),
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
    )
    train_loader = build_dataloader(config, split=str(args.split), shuffle=True)
    val_loader = build_dataloader(config, split=str(args.val_split), shuffle=False)
    class_weights = _build_class_weights(config, train_loader, device)
    model = build_model(model_cfg).to(device)
    _log(f"[Model] name={model_cfg.get('name')} params={count_trainable_params(model)}")
    if args.resume_checkpoint:
        _load_model_checkpoint(model, resolve_path(args.resume_checkpoint) or Path(args.resume_checkpoint), device)

    selected_phases = list(PHASES) if args.phase == "all" else [str(args.phase)]
    phase_summaries: Dict[str, Dict[str, Any]] = {}
    for phase in selected_phases:
        if phase == "classifier" and not args.resume_checkpoint and args.phase != "all":
            warmup_ckpt = output_root / "warmup" / "checkpoints" / "best.pth"
            if warmup_ckpt.exists():
                _load_model_checkpoint(model, warmup_ckpt, device)
        if phase == "finetune" and not args.resume_checkpoint and args.phase != "all":
            classifier_ckpt = output_root / "classifier" / "checkpoints" / "best.pth"
            if classifier_ckpt.exists():
                _load_model_checkpoint(model, classifier_ckpt, device)
        summary = _run_phase(
            phase=phase,
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
            output_root=output_root,
        )
        phase_summaries[phase] = summary
        if phase in {"warmup", "finetune"}:
            fig_dir = output_root / "figures" / f"{phase}_best_motifs"
            _visualize_motif_contact_sheet(
                model=model,
                loader=val_loader,
                device=device,
                config=config,
                output_dir=fig_dir,
                max_samples=int(config.get("logging", {}).get("visualize_samples", 12)),
            )
    _write_summary_files(output_root, phase_summaries, config)
    _log(f"[Output] staged_summary={output_root / 'staged_summary.json'}")
    _log(f"[Output] staged_summary_csv={output_root / 'staged_summary.csv'}")
    _log(f"[Output] final_report={output_root / 'final_report.md'}")


if __name__ == "__main__":
    main()
