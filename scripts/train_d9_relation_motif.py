"""Train D9-RG-MR end-to-end classification on FER pixel graphs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_path, save_config  # noqa: E402
from evaluation.evaluator import save_confusion_matrix  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics, confusion_matrix_array  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.losses import compute_class_weights  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402

LOGGER = logging.getLogger("d9_relation_motif")


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_cls_loss",
    "train_distill_loss",
    "train_motif_loss",
    "train_teacher_conf_mean",
    "train_teacher_conf_std",
    "train_accuracy",
    "train_macro_f1",
    "train_weighted_f1",
    "val_loss",
    "val_cls_loss",
    "val_distill_loss",
    "val_motif_loss",
    "val_teacher_conf_mean",
    "val_teacher_conf_std",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_selected_border_mass_mean",
    "val_selected_outer_border_mass_mean",
    "val_selected_foreground_mass_mean",
    "val_selection_entropy",
    "val_selection_effective_count",
    "lr",
    "epoch_seconds",
]


def _setup_logger() -> None:
    """Configure one stdout handler for this script, even in reused notebook kernels."""
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
    training = dict(cfg.get("training", {}) or {})
    experiment = dict(cfg.get("experiment", {}) or {})
    if getattr(args, "experiment_name", None):
        experiment["name"] = str(args.experiment_name)
        if not getattr(args, "output_dir", None):
            output["dir"] = str(Path("outputs") / str(args.experiment_name))
            paths["resolved_output_root"] = output["dir"]
    if getattr(args, "output_dir", None):
        output["dir"] = str(args.output_dir)
        paths["resolved_output_root"] = str(args.output_dir)
    elif output.get("dir"):
        paths["resolved_output_root"] = str(output["dir"])
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    if getattr(args, "teacher_probs_dir", None):
        distillation = dict(cfg.get("distillation", {}) or {})
        distillation["teacher_probs_dir"] = str(args.teacher_probs_dir)
        cfg["distillation"] = distillation
    if getattr(args, "allow_partial_teacher", False):
        distillation = dict(cfg.get("distillation", {}) or {})
        distillation["allow_partial_teacher"] = True
        cfg["distillation"] = distillation
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
    if sys.platform.startswith("win") and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["output"] = output
    cfg["data"] = data
    cfg["training"] = training
    cfg["experiment"] = experiment
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
    if not bool(loss_cfg.get("use_class_weights", training_cfg.get("class_weights", False))):
        _log("[ClassWeights] disabled")
        return None
    num_classes = int(config.get("model", {}).get("num_classes", config.get("data", {}).get("num_classes", 7)))
    counts = loss_cfg.get("class_counts") or training_cfg.get("class_counts")
    source = "config"
    if counts is None:
        counts = _count_train_labels(train_loader, num_classes=num_classes)
        source = "train_dataset"
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
    _log(f"[ClassWeights] source={source} counts={list(counts)} weights={[round(float(v), 4) for v in weights.cpu()]}")
    return weights


def _build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any]):
    name = str(training_cfg.get("scheduler", "none")).lower()
    if name in {"", "none"}:
        return None, False
    if name in {"warmup_cosine", "cosine_warmup"}:
        epochs = max(1, int(training_cfg.get("epochs", 60)))
        warmup = max(0, int(training_cfg.get("warmup_epochs", 0)))
        min_lr = float(training_cfg.get("min_lr", 1e-6))
        base_lr = float(training_cfg.get("lr", 3e-4))
        min_factor = min_lr / max(base_lr, 1e-12)

        def lr_lambda(epoch_idx: int) -> float:
            step = int(epoch_idx) + 1
            if warmup > 0 and step <= warmup:
                return max(float(step) / float(warmup), min_factor)
            span = max(1, epochs - warmup)
            progress = min(max(float(step - warmup) / float(span), 0.0), 1.0)
            return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), False
    if name in {"reducelronplateau", "reduce_lr_on_plateau"}:
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(training_cfg.get("scheduler_mode", "min")),
                factor=float(training_cfg.get("scheduler_factor", 0.5)),
                patience=int(training_cfg.get("scheduler_patience", 5)),
                min_lr=float(training_cfg.get("min_lr", 1e-6)),
            ),
            True,
        )
    raise ValueError(f"Unsupported D9 scheduler={name!r}")


def _selected_motif_metrics(out: Dict[str, torch.Tensor]) -> Dict[str, float]:
    weights = out.get("selection_weights")
    metrics: Dict[str, float] = {}
    if torch.is_tensor(weights):
        weights = weights.detach().float()
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        metrics["selection_entropy"] = float(entropy.mean().cpu())
        metrics["selection_effective_count"] = float(entropy.exp().mean().cpu())
        for src_key, out_key in (
            ("motif_border_mass", "selected_border_mass_mean"),
            ("motif_outer_border_mass", "selected_outer_border_mass_mean"),
            ("motif_foreground_mass", "selected_foreground_mass_mean"),
        ):
            value = out.get(src_key)
            if torch.is_tensor(value) and value.shape == weights.shape:
                metrics[out_key] = float((weights * value.detach().float()).sum(dim=1).mean().cpu())
    audit = out.get("motif_audit", {})
    if isinstance(audit, dict):
        for src_key, out_key in (
            ("outer_border_mass_mean", "selected_outer_border_mass_mean"),
            ("selected_foreground_mass_mean", "selected_foreground_mass_mean"),
        ):
            if out_key not in metrics and torch.is_tensor(audit.get(src_key)):
                metrics[out_key] = float(audit[src_key].detach().float().mean().cpu())
    metrics.setdefault("selected_border_mass_mean", 0.0)
    metrics.setdefault("selected_outer_border_mass_mean", 0.0)
    metrics.setdefault("selected_foreground_mass_mean", 0.0)
    metrics.setdefault("selection_entropy", 0.0)
    metrics.setdefault("selection_effective_count", 0.0)
    return metrics


def _validate_teacher_arrays(split: str, probs: np.ndarray, labels: np.ndarray, indices: np.ndarray, logits: np.ndarray | None) -> None:
    if probs.ndim != 2:
        raise RuntimeError(f"Teacher {split}_probs.npy must be [N,C], got {probs.shape}")
    if labels.shape != (probs.shape[0],):
        raise RuntimeError(f"Teacher {split}_labels.npy must be [{probs.shape[0]}], got {labels.shape}")
    if indices.shape != (probs.shape[0],):
        raise RuntimeError(f"Teacher {split}_indices.npy must be [{probs.shape[0]}], got {indices.shape}")
    if logits is not None and logits.shape != probs.shape:
        raise RuntimeError(f"Teacher {split}_logits.npy shape {logits.shape} does not match probs {probs.shape}")
    if not np.isfinite(probs).all():
        raise RuntimeError(f"Teacher {split}_probs.npy contains NaN/Inf")
    if logits is not None and not np.isfinite(logits).all():
        raise RuntimeError(f"Teacher {split}_logits.npy contains NaN/Inf")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise RuntimeError(
            f"Teacher {split}_probs.npy rows do not sum to 1.0: "
            f"min={float(row_sums.min()):.6f} max={float(row_sums.max()):.6f}"
        )
    expected = np.arange(probs.shape[0], dtype=np.int64)
    if not np.array_equal(np.sort(indices.astype(np.int64)), expected):
        raise RuntimeError(f"Teacher {split}_indices.npy must cover 0..N-1 exactly once")


def _load_distillation_assets(config: Dict[str, Any], split_names: Iterable[str]) -> Dict[str, Any] | None:
    cfg = dict(config.get("distillation", {}) or {})
    if not bool(cfg.get("enabled", False)):
        _log("[Distillation] enabled=false")
        return None
    teacher_dir = resolve_path(cfg.get("teacher_probs_dir"))
    if teacher_dir is None:
        raise RuntimeError("distillation.enabled=true requires distillation.teacher_probs_dir")
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Teacher probs dir not found: {teacher_dir}")
    temperature = float(cfg.get("temperature", 2.0))
    alpha = float(cfg.get("alpha", 0.5))
    use_logits = bool(cfg.get("use_logits", False))
    if temperature <= 0.0:
        raise ValueError(f"distillation.temperature must be > 0, got {temperature}")
    if alpha < 0.0:
        raise ValueError(f"distillation.alpha must be >= 0, got {alpha}")
    if str(cfg.get("loss_type", "kl")).lower() != "kl":
        raise ValueError("Only distillation.loss_type=kl is implemented")
    splits: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in dict.fromkeys(str(s) for s in split_names):
        probs_path = teacher_dir / f"{split}_probs.npy"
        labels_path = teacher_dir / f"{split}_labels.npy"
        indices_path = teacher_dir / f"{split}_indices.npy"
        if not (probs_path.exists() and labels_path.exists() and indices_path.exists()):
            if split == str(config.get("run", {}).get("train_split", "train")):
                raise FileNotFoundError(f"Missing teacher probs files for split={split} under {teacher_dir}")
            _log(f"[Distillation] split={split} teacher probs not found; validation distill logging disabled")
            continue
        logits_path = teacher_dir / f"{split}_logits.npy"
        probs_np = np.load(probs_path).astype(np.float32)
        labels_np = np.load(labels_path).astype(np.int64)
        indices_np = np.load(indices_path).astype(np.int64)
        logits_np = np.load(logits_path).astype(np.float32) if logits_path.exists() else None
        _validate_teacher_arrays(split, probs_np, labels_np, indices_np, logits_np)
        if use_logits and logits_np is None:
            raise FileNotFoundError(f"distillation.use_logits=true but missing {logits_path}")
        splits[split] = {
            "probs": torch.from_numpy(probs_np),
            "labels": torch.from_numpy(labels_np),
            "indices": torch.from_numpy(indices_np),
        }
        if logits_np is not None:
            splits[split]["logits"] = torch.from_numpy(logits_np)
        _log(
            f"[Distillation] loaded split={split} n={probs_np.shape[0]} "
            f"classes={probs_np.shape[1]} logits_available={logits_np is not None}"
        )
    if "train" not in splits:
        available = ", ".join(sorted(splits.keys())) or "none"
        raise RuntimeError(f"Distillation requires train teacher probs; available splits: {available}")
    if not use_logits and abs(temperature - 1.0) > 1e-8:
        _log(
            "[Distillation] use_logits=false: teacher probs are used as saved; "
            "temperature scales student logits only."
        )
    return {
        "enabled": True,
        "teacher_probs_dir": str(teacher_dir),
        "alpha": alpha,
        "temperature": temperature,
        "use_logits": use_logits,
        "confidence_weighting": bool(cfg.get("confidence_weighting", False)),
        "confidence_mode": str(cfg.get("confidence_mode", "max_prob")).lower(),
        "min_confidence": float(cfg.get("min_confidence", 0.0)),
        "allow_partial_teacher": bool(cfg.get("allow_partial_teacher", False)),
        "splits": splits,
    }


def _distillation_loss(
    *,
    logits_student: torch.Tensor,
    labels: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    state: Dict[str, Any] | None,
    split: str,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if state is None:
        return logits_student.new_zeros(()), {}
    split_data = state["splits"].get(str(split))
    if split_data is None:
        return logits_student.new_zeros(()), {}
    if "sample_idx" not in batch or not torch.is_tensor(batch["sample_idx"]):
        raise RuntimeError("distillation.enabled=true requires batch['sample_idx']")
    idx_cpu = batch["sample_idx"].detach().long().cpu()
    if idx_cpu.numel() == 0:
        raise RuntimeError("Empty sample_idx batch under distillation")
    max_idx = int(idx_cpu.max().item())
    min_idx = int(idx_cpu.min().item())
    valid_mask = (idx_cpu >= 0) & (idx_cpu < int(split_data["labels"].shape[0]))
    if (max_idx >= int(split_data["labels"].shape[0]) or min_idx < 0) and not bool(state.get("allow_partial_teacher", False)):
        raise RuntimeError(
            f"sample_idx out of range for teacher split={split}: "
            f"min={min_idx} max={max_idx} teacher_n={int(split_data['labels'].shape[0])}"
        )
    if not bool(valid_mask.all()):
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return logits_student.new_zeros(()), {}
        idx_cpu = idx_cpu[valid_mask]
        labels_for_distill = labels[valid_mask.to(device=labels.device)]
        logits_for_distill = logits_student[valid_mask.to(device=logits_student.device)]
    else:
        labels_for_distill = labels
        logits_for_distill = logits_student
    teacher_labels = split_data["labels"][idx_cpu].to(device=labels_for_distill.device, dtype=labels_for_distill.dtype)
    if not torch.equal(teacher_labels, labels_for_distill.detach()):
        bad = torch.nonzero(teacher_labels != labels_for_distill.detach(), as_tuple=False).flatten()[:8]
        details = [
            {
                "sample_idx": int(idx_cpu[i].detach().cpu()),
                "teacher_label": int(teacher_labels[i].detach().cpu()),
                "batch_label": int(labels_for_distill[i].detach().cpu()),
            }
            for i in bad.tolist()
        ]
        raise RuntimeError(f"Teacher label mismatch for split={split}: {details}")

    device = logits_student.device
    dtype = torch.float32
    temperature = float(state["temperature"])
    if bool(state["use_logits"]):
        teacher_logits = split_data["logits"][idx_cpu].to(device=device, dtype=dtype)
        teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)
    else:
        teacher_soft = split_data["probs"][idx_cpu].to(device=device, dtype=dtype)
        teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
    student_log_soft = F.log_softmax(logits_for_distill.float() / temperature, dim=1)
    per_sample = F.kl_div(student_log_soft, teacher_soft, reduction="none").sum(dim=1) * temperature * temperature
    conf = teacher_soft.max(dim=1).values
    if bool(state["confidence_weighting"]):
        if state["confidence_mode"] == "top1_margin":
            top2 = teacher_soft.topk(k=min(2, teacher_soft.shape[1]), dim=1).values
            weights = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
        elif state["confidence_mode"] == "max_prob":
            weights = conf
        else:
            raise ValueError(f"Unsupported distillation.confidence_mode={state['confidence_mode']!r}")
        if float(state["min_confidence"]) > 0.0:
            weights = weights * (conf >= float(state["min_confidence"])).to(dtype=weights.dtype)
        distill_loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        distill_loss = per_sample.mean()
    teacher_top1 = teacher_soft.argmax(dim=1)
    return distill_loss.to(dtype=logits_student.dtype), {
        "teacher_conf": conf.detach(),
        "teacher_top1": teacher_top1.detach(),
        "teacher_soft": teacher_soft.detach(),
    }


def _metric_bundle(
    *,
    y_true: list[int],
    y_pred: list[int],
    loss_sum: float,
    cls_sum: float,
    distill_sum: float,
    motif_sum: float,
    motif_metric_sums: Dict[str, float],
    teacher_conf_sum: float,
    teacher_conf_sq_sum: float,
    teacher_conf_count: int,
    count: int,
    prefix: str,
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred) if y_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    out = {
        f"{prefix}_loss": float(loss_sum / max(count, 1)),
        f"{prefix}_cls_loss": float(cls_sum / max(count, 1)),
        f"{prefix}_distill_loss": float(distill_sum / max(count, 1)),
        f"{prefix}_motif_loss": float(motif_sum / max(count, 1)),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_weighted_f1": float(metrics["weighted_f1"]),
    }
    if teacher_conf_count > 0:
        mean = float(teacher_conf_sum / max(teacher_conf_count, 1))
        variance = max(float(teacher_conf_sq_sum / max(teacher_conf_count, 1)) - mean * mean, 0.0)
        out[f"{prefix}_teacher_conf_mean"] = mean
        out[f"{prefix}_teacher_conf_std"] = math.sqrt(variance)
    else:
        out[f"{prefix}_teacher_conf_mean"] = 0.0
        out[f"{prefix}_teacher_conf_std"] = 0.0
    for key, value in motif_metric_sums.items():
        out[f"{prefix}_{key}"] = float(value / max(count, 1))
    return out


def _check_finite(loss: torch.Tensor, out: Dict[str, torch.Tensor]) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite D9 loss: {float(loss.detach().cpu())}")
    for key in ("logits", "motif_maps", "motif_embeddings", "selection_weights"):
        value = out.get(key)
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite D9 output: {key}")


def _tensor_stats(name: str, value: torch.Tensor | None) -> str:
    if not torch.is_tensor(value):
        return f"{name}=n/a"
    data = value.detach().float()
    return (
        f"{name}_mean={float(data.mean().cpu()):.6f} "
        f"{name}_std={float(data.std(unbiased=False).cpu()):.6f} "
        f"{name}_min={float(data.min().cpu()):.6f} "
        f"{name}_max={float(data.max().cpu()):.6f}"
    )


def _class_distribution(values: torch.Tensor, num_classes: int) -> list[int]:
    counts = torch.bincount(values.detach().long().cpu(), minlength=int(num_classes))
    return [int(v) for v in counts.tolist()]


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    feature_ablation_cfg: Dict[str, Any],
    model_node_dim: int,
    model_edge_dim: int,
    class_weights: torch.Tensor | None,
    motif_criterion: MotifDiscoveryStage1Loss | None,
    motif_aux_weight: float,
    distillation_state: Dict[str, Any] | None = None,
    distillation_split: str = "train",
    max_batches: int | None = None,
    amp: bool = False,
    log_interval: int = 0,
    grad_clip_norm: float = 1.0,
    prediction_csv_path: Path | None = None,
    metrics_json_path: Path | None = None,
    confusion_matrix_path: Path | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    y_true: list[int] = []
    y_pred: list[int] = []
    graph_ids: list[int] = []
    scores: list[list[float]] = []
    loss_sum = 0.0
    cls_sum = 0.0
    distill_sum = 0.0
    motif_sum = 0.0
    teacher_conf_sum = 0.0
    teacher_conf_sq_sum = 0.0
    teacher_conf_count = 0
    motif_metric_sums = {
        "selected_border_mass_mean": 0.0,
        "selected_outer_border_mass_mean": 0.0,
        "selected_foreground_mass_mean": 0.0,
        "selection_entropy": 0.0,
        "selection_effective_count": 0.0,
    }
    count = 0
    autocast_enabled = bool(amp and device.type == "cuda")
    first_logged = False
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        original_node_dim = int(batch["x"].shape[-1])
        original_edge_dim = int(batch["edge_attr"].shape[-1])
        batch = move_to_device(batch, device)
        batch = apply_feature_ablation(batch, feature_ablation_cfg)
        assert_feature_dims(batch, node_dim=model_node_dim, edge_dim=model_edge_dim)
        labels = batch["y"].long()
        if not first_logged:
            sample_idx_value = batch.get("sample_idx")
            sample_idx_available = torch.is_tensor(sample_idx_value)
            if sample_idx_available:
                sample_idx_min = int(sample_idx_value.detach().min().cpu())
                sample_idx_max = int(sample_idx_value.detach().max().cpu())
                sample_idx_text = f"sample_idx available: true min={sample_idx_min} max={sample_idx_max}"
            else:
                sample_idx_text = "sample_idx available: false"
            _log(
                "[D9Batch] "
                f"original_node_dim={original_node_dim} masked_node_dim={int(batch['x'].shape[-1])} "
                f"original_edge_dim={original_edge_dim} masked_edge_dim={int(batch['edge_attr'].shape[-1])} "
                f"x_shape={list(batch['x'].shape)} edge_attr_shape={list(batch['edge_attr'].shape)} "
                f"{sample_idx_text}"
            )
            first_logged = True
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                out = model(batch)
                cls_loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
                if motif_criterion is not None and float(motif_aux_weight) > 0.0:
                    motif_loss_dict = motif_criterion(out, batch)
                    motif_loss = motif_loss_dict["loss"]
                else:
                    motif_loss = out["logits"].new_zeros(())
                if is_train:
                    distill_loss, distill_stats = _distillation_loss(
                        logits_student=out["logits"],
                        labels=labels,
                        batch=batch,
                        state=distillation_state,
                        split=distillation_split,
                    )
                else:
                    distill_loss = out["logits"].new_zeros(())
                    distill_stats = {}
                alpha = float(distillation_state.get("alpha", 0.0)) if (is_train and distillation_state is not None) else 0.0
                loss = cls_loss + alpha * distill_loss + float(motif_aux_weight) * motif_loss
        _check_finite(loss, out)
        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=float(grad_clip_norm),
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"Non-finite D9 grad norm at batch {batch_idx}: {float(grad_norm)}")
            optimizer.step()
        batch_size = int(labels.shape[0])
        pred = out["logits"].detach().argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        if not is_train and prediction_csv_path is not None:
            if "graph_id" in batch and torch.is_tensor(batch["graph_id"]):
                graph_ids.extend(batch["graph_id"].detach().cpu().tolist())
            else:
                start_idx = len(graph_ids)
                graph_ids.extend(range(start_idx, start_idx + batch_size))
            scores.extend(out["logits"].detach().float().cpu().tolist())
        loss_sum += float(loss.detach().cpu()) * batch_size
        cls_sum += float(cls_loss.detach().cpu()) * batch_size
        distill_sum += float(distill_loss.detach().cpu()) * batch_size
        motif_sum += float(motif_loss.detach().cpu()) * batch_size
        teacher_conf = distill_stats.get("teacher_conf") if isinstance(distill_stats, dict) else None
        if torch.is_tensor(teacher_conf):
            conf_cpu = teacher_conf.detach().float().cpu()
            teacher_conf_sum += float(conf_cpu.sum())
            teacher_conf_sq_sum += float(conf_cpu.square().sum())
            teacher_conf_count += int(conf_cpu.numel())
        motif_metrics = _selected_motif_metrics(out)
        for key in motif_metric_sums:
            motif_metric_sums[key] += motif_metrics.get(key, 0.0) * batch_size
        count += batch_size
        if is_train and log_interval > 0 and (batch_idx + 1) % int(log_interval) == 0:
            weighted_motif_loss = float(motif_aux_weight) * float(motif_loss.detach().cpu())
            _log(
                f"[train batch {batch_idx + 1}] "
                f"total_loss={float(loss.detach().cpu()):.6f} "
                f"cls_loss={float(cls_loss.detach().cpu()):.6f} "
                f"distill_loss={float(distill_loss.detach().cpu()):.6f} "
                f"raw_motif_loss={float(motif_loss.detach().cpu()):.6f} "
                f"weighted_motif_loss={weighted_motif_loss:.6f} "
                f"distill_alpha={alpha:.6f} "
                f"temperature={float(distillation_state.get('temperature', 0.0)) if distillation_state else 0.0:.6f} "
                f"teacher_conf_mean={float(teacher_conf.mean().detach().cpu()) if torch.is_tensor(teacher_conf) else 0.0:.6f} "
                f"teacher_conf_std={float(teacher_conf.float().std(unbiased=False).detach().cpu()) if torch.is_tensor(teacher_conf) else 0.0:.6f} "
                f"teacher_top1_dist={_class_distribution(distill_stats['teacher_top1'], int(out['logits'].shape[-1])) if isinstance(distill_stats, dict) and torch.is_tensor(distill_stats.get('teacher_top1')) else []} "
                f"motif_aux_weight={float(motif_aux_weight):.6f} "
                f"student_pred_dist={_class_distribution(pred, int(out['logits'].shape[-1]))} "
                f"label_dist={_class_distribution(labels, int(out['logits'].shape[-1]))} "
                f"{_tensor_stats('logits', out.get('logits'))} "
                f"{_tensor_stats('motif_maps', out.get('motif_maps'))} "
                f"{_tensor_stats('selection_weights', out.get('selection_weights'))}"
            )
    metrics_out = _metric_bundle(
        y_true=y_true,
        y_pred=y_pred,
        loss_sum=loss_sum,
        cls_sum=cls_sum,
        distill_sum=distill_sum,
        motif_sum=motif_sum,
        motif_metric_sums=motif_metric_sums,
        teacher_conf_sum=teacher_conf_sum,
        teacher_conf_sq_sum=teacher_conf_sq_sum,
        teacher_conf_count=teacher_conf_count,
        count=count,
        prefix="train" if is_train else "val",
    )
    if not is_train and y_true and metrics_json_path is not None:
        report = classification_report_dict(y_true, y_pred)
        cm = confusion_matrix_array(y_true, y_pred)
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": metrics_out["val_accuracy"],
                    "macro_f1": metrics_out["val_macro_f1"],
                    "weighted_f1": metrics_out["val_weighted_f1"],
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                },
                f,
                indent=2,
            )
        if prediction_csv_path is not None:
            prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)
            num_classes = len(scores[0]) if scores else 0
            with prediction_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["graph_id", "y_true", "y_pred"] + [f"logit_{i}" for i in range(num_classes)])
                for row in zip(graph_ids, y_true, y_pred, scores):
                    gid, yt, yp, logit_row = row
                    writer.writerow([gid, yt, yp] + list(logit_row))
        if confusion_matrix_path is not None:
            save_confusion_matrix(cm, confusion_matrix_path)
    return metrics_out


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    monitor: str,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_metric_name": str(monitor),
        "best_metric_mode": str(mode),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def _evaluate_and_save(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    feature_ablation_cfg: Dict[str, Any],
    model_node_dim: int,
    model_edge_dim: int,
    output_root: Path,
    split: str,
    max_batches: int | None,
) -> Dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    graph_ids: list[int] = []
    scores: list[list[float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            batch = apply_feature_ablation(batch, feature_ablation_cfg)
            assert_feature_dims(batch, node_dim=model_node_dim, edge_dim=model_edge_dim)
            out = model(batch)
            logits = out["logits"].detach().float()
            pred = logits.argmax(dim=1)
            y_true.extend(batch["y"].detach().cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            graph_ids.extend(batch["graph_id"].detach().cpu().tolist())
            scores.extend(logits.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({**metrics, "classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
    with (metrics_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "y_true", "y_pred"] + [f"logit_{i}" for i in range(7)])
        for row in zip(graph_ids, y_true, y_pred, scores):
            gid, yt, yp, logit_row = row
            writer.writerow([gid, yt, yp] + list(logit_row))
    save_confusion_matrix(cm, figures_dir / f"{split}_confusion_matrix.png")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--teacher_probs_dir", default=None)
    parser.add_argument("--allow_partial_teacher", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--disable_shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    _setup_logger()
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    training_cfg = dict(config.get("training", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    loss_cfg = dict(config.get("loss", {}) or {})
    checkpoint_cfg = dict(config.get("checkpoint", {}) or {})
    early_cfg = dict(config.get("early_stopping", {}) or {})
    logging_cfg = dict(config.get("logging", {}) or {})
    feature_ablation_cfg = dict(config.get("feature_ablation", {}) or {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = resolve_device(args.device, config=config)
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    output_root = output_root or PROJECT_ROOT / "outputs" / str(config.get("experiment", {}).get("name", "d9_rg_mr"))
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    _log(f"[Device] selected={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        _log(f"[Device] gpu={torch.cuda.get_device_name(device.index or torch.cuda.current_device())}")
    _log(f"[Output] run_dir={output_root}")
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
    )

    train_loader = build_dataloader(config, split=str(args.split or "train"), shuffle=not bool(args.disable_shuffle))
    val_loader = build_dataloader(config, split=str(args.val_split or "val"), shuffle=False)
    distillation_state = _load_distillation_assets(
        config,
        split_names=[str(args.split or "train"), str(args.val_split or "val")],
    )
    model = build_model(model_cfg).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    _log(f"[Model] name={model_cfg.get('name')} total_params={total_params:,} trainable_params={trainable_params:,}")
    _log(
        f"[Model] node_dim={model_cfg.get('node_dim')} edge_dim={model_cfg.get('edge_dim')} "
        f"hidden_dim={model_cfg.get('hidden_dim')} num_motifs={model_cfg.get('num_motifs')} "
        f"num_classes={model_cfg.get('num_classes')}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler, scheduler_uses_monitor = _build_scheduler(optimizer, training_cfg)
    class_weights = _build_class_weights(config, train_loader, device)
    motif_aux_weight = float(loss_cfg.get("motif_aux_weight", 0.0))
    motif_criterion = None
    if motif_aux_weight > 0.0:
        motif_loss_cfg = dict(config.get("motif_loss", loss_cfg.get("motif_loss", {})) or {})
        motif_loss_cfg.setdefault("height", int(model_cfg.get("height", model_cfg.get("image_size", 48))))
        motif_loss_cfg.setdefault("width", int(model_cfg.get("width", model_cfg.get("image_size", 48))))
        motif_criterion = MotifDiscoveryStage1Loss(motif_loss_cfg).to(device)
    _log(f"[Loss] cls_loss=cross_entropy motif_aux_weight={motif_aux_weight}")
    if distillation_state is not None:
        _log(
            f"[Loss] distillation=kl alpha={distillation_state['alpha']} "
            f"temperature={distillation_state['temperature']} use_logits={distillation_state['use_logits']} "
            f"confidence_weighting={distillation_state['confidence_weighting']} "
            f"allow_partial_teacher={distillation_state['allow_partial_teacher']}"
        )

    epochs = int(training_cfg.get("epochs", 60))
    max_train_batches = training_cfg.get("max_train_batches", args.max_train_batches)
    max_val_batches = training_cfg.get("max_val_batches", args.max_val_batches)
    log_interval = int(logging_cfg.get("log_interval", training_cfg.get("log_every", 20)))
    if max_train_batches is not None and int(max_train_batches) <= 5:
        log_interval = 1
    monitor = str(checkpoint_cfg.get("monitor", checkpoint_cfg.get("save_best_metric", "val_macro_f1")))
    mode = str(checkpoint_cfg.get("mode", checkpoint_cfg.get("save_best_mode", "max"))).lower()
    if monitor != "val_macro_f1" or mode != "max":
        raise ValueError("D9 classification must checkpoint on val_macro_f1 with mode=max")
    early_monitor = str(early_cfg.get("monitor", monitor))
    early_mode = str(early_cfg.get("mode", mode)).lower()
    if early_monitor != "val_macro_f1" or early_mode != "max":
        raise ValueError("D9 classification must early-stop on val_macro_f1 with mode=max")
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 20)))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    _log(f"[Checkpoint] save_best_metric={monitor} mode={mode}")
    _log(f"[EarlyStopping] enabled={early_enabled} monitor={early_monitor} mode={early_mode} patience={patience}")
    _log(f"[Scheduler] name={training_cfg.get('scheduler', 'none')} uses_monitor={scheduler_uses_monitor}")
    if args.disable_shuffle:
        _log("[DataLoader] train shuffle disabled by --disable_shuffle")

    checkpoint_dir = output_root / "checkpoints"
    history_path = output_root / "logs" / "d9_history.csv"
    val_jsonl = output_root / "logs" / "val_metrics.jsonl"
    initial_metrics = _run_epoch(
        model=model,
        loader=val_loader,
        optimizer=None,
        device=device,
        feature_ablation_cfg=feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
        class_weights=class_weights,
        motif_criterion=motif_criterion,
        motif_aux_weight=motif_aux_weight,
        max_batches=1,
        amp=False,
        grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
    )
    _save_checkpoint(checkpoint_dir / "initial.pth", model, optimizer, scheduler, 0, -float("inf"), -1, config, initial_metrics, monitor, mode)
    _log(
        f"[Initial] val_loss={initial_metrics['val_loss']:.4f} "
        f"val_accuracy={initial_metrics['val_accuracy']:.4f} val_macro_f1={initial_metrics['val_macro_f1']:.4f}"
    )

    best_metric = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    best_metrics: Dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            feature_ablation_cfg=feature_ablation_cfg,
            model_node_dim=int(model_cfg.get("node_dim", 3)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
            class_weights=class_weights,
            motif_criterion=motif_criterion,
            motif_aux_weight=motif_aux_weight,
            distillation_state=distillation_state,
            distillation_split=str(args.split or "train"),
            max_batches=max_train_batches,
            amp=bool(training_cfg.get("amp", True)),
            log_interval=log_interval,
            grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            feature_ablation_cfg=feature_ablation_cfg,
            model_node_dim=int(model_cfg.get("node_dim", 3)),
            model_edge_dim=int(model_cfg.get("edge_dim", 5)),
            class_weights=class_weights,
            motif_criterion=motif_criterion,
            motif_aux_weight=motif_aux_weight,
            max_batches=max_val_batches,
            amp=False,
            grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
            prediction_csv_path=output_root / "metrics" / f"val_predictions_epoch_{epoch:03d}.csv",
            metrics_json_path=output_root / "metrics" / f"val_metrics_epoch_{epoch:03d}.json",
            confusion_matrix_path=output_root / "figures" / f"val_confusion_matrix_epoch_{epoch:03d}.png",
        )
        row = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - start,
        }
        if scheduler is not None:
            if scheduler_uses_monitor:
                scheduler.step(float(row["val_loss"]))
            else:
                scheduler.step()
        _append_history(history_path, row)
        _append_jsonl(val_jsonl, row)
        current = float(row[monitor])
        improved = current > best_metric + min_delta
        if improved:
            best_metric = current
            best_epoch = epoch
            stale_epochs = 0
            best_metrics = dict(row)
            _save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, scheduler, epoch, best_metric, best_epoch, config, row, monitor, mode)
            _log(f"[Checkpoint] best epoch={epoch} val_macro_f1={best_metric:.4f}")
        else:
            stale_epochs += 1
        _save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scheduler, epoch, best_metric, best_epoch, config, row, monitor, mode)
        _log(
            f"[epoch {epoch:03d}] train_loss={row['train_loss']:.4f} train_macro_f1={row['train_macro_f1']:.4f} "
            f"train_distill_loss={row['train_distill_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_accuracy={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} best_epoch={best_epoch} stale={stale_epochs}/{patience}"
        )
        if early_enabled and stale_epochs >= patience:
            _log(f"[EarlyStopping] stop epoch={epoch} stale_epochs={stale_epochs}")
            break
    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was produced")

    checkpoint = torch.load(checkpoint_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val_eval = _evaluate_and_save(
        model=model,
        loader=val_loader,
        device=device,
        feature_ablation_cfg=feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
        output_root=output_root,
        split="val",
        max_batches=max_val_batches,
    )
    with (output_root / "metrics" / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_macro_f1": best_metric, "best_metrics": best_metrics}, f, indent=2)
    _log(f"[Output] initial={checkpoint_dir / 'initial.pth'}")
    _log(f"[Output] best={checkpoint_dir / 'best.pth'}")
    _log(f"[Output] last={checkpoint_dir / 'last.pth'}")
    _log(f"[Output] history={history_path}")
    _log(f"[Output] val_metrics_jsonl={val_jsonl}")
    _log(f"[Output] val_metrics={output_root / 'metrics' / 'val_metrics.json'}")
    _log(
        f"[Summary] best_epoch={best_epoch} best_val_macro_f1={best_metric:.4f} "
        f"eval_val_macro_f1={val_eval['macro_f1']:.4f} eval_val_accuracy={val_eval['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
