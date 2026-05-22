"""Train D15 from-scratch performance runs.

D15 is the main from-scratch performance track after D13. It may reuse the
architecture family found during D13, but it must not load D13/D14 checkpoints
or warm-start weights.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, save_config
from evaluation.d13_diagnostics import compute_assignment_stats, write_confusion_matrix
from models.d13c_supcon_model import D13CSupConModel
from training.losses import FocalLoss, WeightedCrossEntropy, compute_class_weights
from training.optimizer import build_optimizer, build_scheduler, step_scheduler
from training.supcon_loss import SupervisedContrastiveLossWithStats
from training.train_d13b import (
    _append_csv,
    _amp_is_enabled,
    _autocast,
    _float,
    _make_grad_scaler,
    _metrics,
    _pred_count_row,
    _reduce_loss_value,
    _set_model_epoch,
    _slot_stats,
    _test_predictions,
)
from training.train_d14 import _apply_graph_augmentation
from training.trainer import move_to_device, set_seed


class D15ScheduledLoss(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is not None:
                class_weights = compute_class_weights(
                    counts,
                    normalize_mean=True,
                    power=float(cfg.get("class_weight_power", 0.25)),
                )
        loss_type = str(cfg.get("type", cfg.get("loss_type", "weighted_ce"))).lower()
        if loss_type in ("focal", "weighted_focal"):
            self.ce = FocalLoss(class_weights, gamma=float(cfg.get("focal_gamma", 1.5)), label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        else:
            self.ce = WeightedCrossEntropy(class_weights, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        self.supcon = SupervisedContrastiveLossWithStats(float(cfg.get("supcon_temperature", 0.1)))
        self.lambda_supcon_target = float(cfg.get("lambda_supcon_target", cfg.get("lambda_supcon", 0.0)))
        self.supcon_start_epoch = int(cfg.get("supcon_start_epoch", 35))
        self.supcon_ramp_epochs = max(int(cfg.get("supcon_ramp_epochs", 10)), 1)
        self.slot_loss_start_epoch = int(cfg.get("slot_loss_start_epoch", 10))
        self.pool_loss_early_weight_scale = float(cfg.get("pool_loss_early_weight_scale", 1.0))
        self.pool_entropy_weight = float(cfg.get("pool_entropy_weight", 0.0005))
        self.pool_balance_weight = float(cfg.get("pool_balance_weight", 0.001))
        self.pool_compactness_weight = float(cfg.get("pool_compactness_weight", 0.001))
        self.pool_area_weight = float(cfg.get("pool_area_weight", 0.0005))
        self.slot_diversity_weight = float(cfg.get("slot_diversity_weight", 0.001))
        self.slot_overlap_weight = float(cfg.get("slot_overlap_weight", 0.001))
        self.slot_entropy_weight = float(cfg.get("slot_entropy_weight", 0.0005))
        self.slot_balance_weight = float(cfg.get("slot_balance_weight", 0.0005))
        self.current_epoch = 1
        if float(cfg.get("prototype_weight", 0.0)) != 0.0:
            raise ValueError("D15 forbids prototype_weight > 0")
        if bool(cfg.get("motif_level_supcon", False)):
            raise ValueError("D15 forbids motif_level_supcon=True")

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def lambda_supcon_current(self) -> float:
        epoch = int(self.current_epoch)
        if epoch < self.supcon_start_epoch:
            return 0.0
        ramp_end = self.supcon_start_epoch + self.supcon_ramp_epochs
        if epoch < ramp_end:
            progress = (epoch - self.supcon_start_epoch + 1) / max(float(self.supcon_ramp_epochs), 1.0)
            return float(self.lambda_supcon_target * max(0.0, min(1.0, progress)))
        return float(self.lambda_supcon_target)

    def forward(self, out: Dict[str, Any], y: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aux = out.get("aux", {})
        logits = out["logits"]
        ce = self.ce(logits, y.long())
        supcon_loss, supcon_stats = self.supcon(out["z_proj"], y.long())
        pool_entropy = aux.get("entropy_loss", logits.new_tensor(0.0))
        pool_balance = aux.get("balance_loss", logits.new_tensor(0.0))
        pool_compact = aux.get("compactness_loss", logits.new_tensor(0.0))
        pool_area = aux.get("area_loss", logits.new_tensor(0.0))
        slot_div = aux.get("slot_diversity_loss", logits.new_tensor(0.0))
        slot_overlap = aux.get("slot_overlap_loss", logits.new_tensor(0.0))
        slot_entropy = aux.get("slot_entropy_loss", logits.new_tensor(0.0))
        slot_balance = aux.get("slot_balance_loss", logits.new_tensor(0.0))
        pool_scale = self.pool_loss_early_weight_scale if int(self.current_epoch) < self.slot_loss_start_epoch else 1.0
        slot_scale = 0.0 if int(self.current_epoch) < self.slot_loss_start_epoch else 1.0
        lambda_supcon = self.lambda_supcon_current()
        pool_loss = (
            self.pool_entropy_weight * pool_entropy
            + self.pool_balance_weight * pool_balance
            + self.pool_compactness_weight * pool_compact
            + self.pool_area_weight * pool_area
        ) * pool_scale
        slot_loss = (
            self.slot_diversity_weight * slot_div
            + self.slot_overlap_weight * slot_overlap
            + self.slot_entropy_weight * slot_entropy
            + self.slot_balance_weight * slot_balance
        ) * slot_scale
        total = ce + pool_loss + slot_loss + lambda_supcon * supcon_loss
        return {
            "loss": total,
            "loss_ce": ce,
            "loss_supcon": supcon_loss,
            "lambda_supcon_current": logits.new_tensor(lambda_supcon),
            "loss_pool_total": pool_loss,
            "loss_slot_total": slot_loss,
            "pool_weight_scale": logits.new_tensor(pool_scale),
            "slot_weight_scale": logits.new_tensor(slot_scale),
            "loss_pool_entropy": pool_entropy,
            "loss_pool_balance": pool_balance,
            "loss_pool_compactness": pool_compact,
            "loss_pool_area": pool_area,
            "loss_slot_diversity": slot_div,
            "loss_slot_overlap": slot_overlap,
            "loss_slot_entropy": slot_entropy,
            "loss_slot_balance": slot_balance,
            **{key: logits.new_tensor(float(value)) for key, value in supcon_stats.items()},
        }


def _finite_check(out: Dict[str, Any], loss: torch.Tensor, split: str, epoch: int, batch_idx: int) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite {split} loss at epoch={epoch} batch={batch_idx}")
    for key in ("logits", "slot_attention", "z_image", "z_proj"):
        value = out.get(key)
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {split} {key} at epoch={epoch} batch={batch_idx}")


def _supcon_stats(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    keys = [
        "loss_supcon",
        "lambda_supcon_current",
        "positive_pair_count",
        "valid_supcon_anchor_count",
        "z_norm_mean",
        "z_norm_std",
        "embedding_collapse_score",
        "has_supcon_signal",
    ]
    out = {}
    for key in keys:
        value = loss_dict.get(key)
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().mean().item())
    return out


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _stage_name(epoch: int, loss_cfg: Dict[str, Any]) -> str:
    slot_start = int(loss_cfg.get("slot_loss_start_epoch", 10))
    supcon_start = int(loss_cfg.get("supcon_start_epoch", 35))
    supcon_ramp = max(int(loss_cfg.get("supcon_ramp_epochs", 10)), 1)
    if epoch < slot_start:
        return "ce_pool_light"
    if epoch < supcon_start:
        return "slot_warmup"
    if epoch < supcon_start + supcon_ramp:
        return "supcon_ramp"
    return "supcon_full"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    target = model.module if hasattr(model, "module") else model
    return getattr(target, "_orig_mod", target)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> bool:
    if not state:
        return False
    def _byte_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu().to(torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8)

    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(_byte_tensor(state["torch_cpu"]))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_tensor(item) for item in cuda_state])
    return True


def _sampler_state_info(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    return {
        "sampler_seed_base": int(training_cfg.get("seed", config.get("run", {}).get("seed", 42))),
        "chunk_aware_sampler": bool(data_cfg.get("chunk_aware_sampler", data_cfg.get("chunk_aware_shuffle", False))),
        "shuffle_chunks": bool(data_cfg.get("shuffle_chunks", True)),
        "shuffle_within_chunk": bool(data_cfg.get("shuffle_within_chunk", True)),
        "batch_size": int(data_cfg.get("batch_size", training_cfg.get("batch_size", 0)) or 0),
    }


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_d15_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    best_value: float,
    best_epoch: int,
    early_stopping_state: Dict[str, Any],
    resume_source: str | None,
    interrupted: bool = False,
) -> None:
    payload = {
        "checkpoint_format": "d15_kaggle_safe_resume_v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_macro_f1": float(best_value),
        "best_epoch": int(best_epoch),
        "early_stopping_state": dict(early_stopping_state),
        "rng_state": _rng_state(),
        "config": config,
        "resolved_config": config,
        "run_name": str(config.get("run_name") or config.get("run", {}).get("config_name") or path.parent.parent.name),
        "from_scratch": True,
        "init_checkpoint": None,
        "loaded_pretrained": False,
        "resume_source": resume_source,
        "dataloader_sampler_state_info": _sampler_state_info(config),
        "metrics": metrics,
        "interrupted": bool(interrupted),
    }
    _atomic_torch_save(payload, path)


def _get_nested(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _nonnull(value: Any) -> bool:
    return value not in (None, "", "null", False)


def _validate_resume_config(
    checkpoint: Dict[str, Any],
    current_config: Dict[str, Any],
    current_epochs: int,
    allow_batch_size_change: bool,
) -> Dict[str, Any]:
    ckpt_config = checkpoint.get("resolved_config") or checkpoint.get("config") or {}
    failures = []
    warnings = []
    critical_paths = [
        "model.name",
        "model.base_model",
        "model.num_slots",
        "model.hidden_dim",
        "model.node_dim",
        "model.edge_dim",
        "model.pixel_layers",
        "model.region_layers",
        "model.region_encoder_layers",
        "model.pooling.grid_size",
        "model.pooling.assign_m",
        "model.pooling.neighbor_mode",
        "model.slot_iters",
        "model.slot_dim",
        "optimizer.name",
        "scheduler.name",
        "loss.name",
        "loss.type",
        "loss.lambda_supcon_target",
        "loss.supcon_start_epoch",
        "loss.supcon_ramp_epochs",
        "loss.slot_loss_start_epoch",
    ]
    for key in critical_paths:
        old = _get_nested(ckpt_config, key)
        new = _get_nested(current_config, key)
        if old != new:
            failures.append(f"{key}: checkpoint={old!r} current={new!r}")

    for cfg, label in ((ckpt_config, "checkpoint"), (current_config, "current")):
        if not bool(cfg.get("from_scratch", cfg.get("d15", {}).get("from_scratch", False))):
            failures.append(f"{label}.from_scratch is not true")
        for key in ("model.init_checkpoint", "model.init_d13b_checkpoint", "model.pretrained_checkpoint", "training.init_checkpoint"):
            if _nonnull(_get_nested(cfg, key)):
                failures.append(f"{label}.{key} is not null")

    old_batch = int(_get_nested(ckpt_config, "training.batch_size", _get_nested(ckpt_config, "data.batch_size", 0)) or 0)
    new_batch = int(_get_nested(current_config, "training.batch_size", _get_nested(current_config, "data.batch_size", 0)) or 0)
    if old_batch and new_batch and old_batch != new_batch:
        msg = f"training.batch_size: checkpoint={old_batch} current={new_batch}"
        if allow_batch_size_change:
            warnings.append("ALLOW_BATCH_SIZE_CHANGE " + msg)
        else:
            failures.append(msg)

    ckpt_epoch = int(checkpoint.get("epoch", 0) or 0)
    if int(current_epochs) < ckpt_epoch:
        failures.append(f"current max epochs {current_epochs} is smaller than checkpoint epoch {ckpt_epoch}")

    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "checkpoint_batch_size": old_batch,
        "current_batch_size": new_batch,
    }


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _load_d15_resume(
    resume_from: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    config: Dict[str, Any],
    epochs: int,
    output_root: Path,
    restore_rng: bool,
    allow_resume_from_best: bool,
    allow_batch_size_change: bool,
) -> Dict[str, Any]:
    resume_path = Path(resume_from)
    if resume_path.name.lower() == "best.pt" and not allow_resume_from_best:
        raise ValueError("Refusing to resume from best.pt without --allow_resume_from_best true")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    validation = _validate_resume_config(checkpoint, config, epochs, allow_batch_size_change)
    if not validation["ok"]:
        (output_root / "resume_config_mismatch.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        raise ValueError("D15 resume config mismatch: " + "; ".join(validation["failures"]))

    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    loaded_optimizer = checkpoint.get("optimizer_state_dict") is not None
    loaded_scheduler = checkpoint.get("scheduler_state_dict") is not None
    if not loaded_optimizer:
        raise ValueError("D15 resume checkpoint is missing optimizer_state_dict")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        if not loaded_scheduler:
            raise ValueError("D15 resume checkpoint is missing scheduler_state_dict")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is None:
            raise ValueError("D15 AMP resume checkpoint is missing scaler_state_dict")
        scaler.load_state_dict(scaler_state)
    restored_rng = _restore_rng_state(checkpoint.get("rng_state", {})) if restore_rng else False
    resumed_epoch = int(checkpoint.get("epoch", 0) or 0)
    next_epoch = resumed_epoch + 1
    early_state = dict(checkpoint.get("early_stopping_state") or {})
    best_value = float(checkpoint.get("best_val_macro_f1", early_state.get("best_val_macro_f1", -float("inf"))))
    best_epoch = int(checkpoint.get("best_epoch", early_state.get("best_epoch", -1)))
    stale = int(early_state.get("patience_counter", early_state.get("stale", 0)))
    event = {
        "resume_from": str(resume_path),
        "resumed_epoch": resumed_epoch,
        "next_epoch": next_epoch,
        "loaded_optimizer": bool(loaded_optimizer),
        "loaded_scheduler": bool(loaded_scheduler),
        "current_lr": float(optimizer.param_groups[0].get("lr", 0.0)),
        "best_val_macro_f1": best_value,
        "best_epoch": best_epoch,
        "early_stopping_counter": stale,
        "restored_rng": bool(restored_rng),
        "allow_resume_from_best": bool(allow_resume_from_best),
        "allow_batch_size_change": bool(allow_batch_size_change),
        "config_warnings": validation["warnings"],
    }
    print("[D15 Resume] " + json.dumps(event, indent=2))
    _append_jsonl(output_root / "resume_events.jsonl", event)
    return {
        "start_epoch": next_epoch,
        "global_step": int(checkpoint.get("global_step", 0) or 0),
        "best_value": best_value,
        "best_epoch": best_epoch,
        "stale": stale,
        "resume_source": str(resume_path),
        "early_stopping_state": early_state,
    }


def _run_epoch(
    model,
    criterion,
    loader,
    optimizer,
    device,
    epoch: int,
    split: str,
    max_batches: int | None,
    grad_clip: float,
    amp: bool,
    augmentation_cfg: Dict[str, Any] | None = None,
    scaler=None,
    global_step: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    _set_model_epoch(model, epoch)
    amp_enabled = _amp_is_enabled(amp, device)
    if is_train and amp_enabled and scaler is None:
        scaler = _make_grad_scaler(amp_enabled)
    totals: Dict[str, float] = {}
    slot_totals: Dict[str, float] = {}
    pool_totals: Dict[str, float] = {}
    supcon_totals: Dict[str, float] = {}
    y_true, y_pred = [], []
    count = 0
    start = time.perf_counter()
    first_batch_wait_time = None
    batch_time_total = 0.0
    cache_hit_count = 0
    cache_miss_count = 0
    chunk_load_count = 0
    chunk_eviction_count = 0
    chunk_load_time_ms_total = 0.0
    data_wait_start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch_wall_start = time.perf_counter()
        wait_time = batch_wall_start - data_wait_start
        if first_batch_wait_time is None:
            first_batch_wait_time = float(wait_time)
        cache_hit_count += int(batch.get("cache_hit", torch.empty(0)).sum().item()) if torch.is_tensor(batch.get("cache_hit")) else 0
        cache_miss_count += int(batch.get("cache_miss", torch.empty(0)).sum().item()) if torch.is_tensor(batch.get("cache_miss")) else 0
        chunk_load_count += int(batch.get("chunk_load_count", torch.empty(0)).sum().item()) if torch.is_tensor(batch.get("chunk_load_count")) else 0
        chunk_eviction_count += int(batch.get("chunk_eviction_count", torch.empty(0)).sum().item()) if torch.is_tensor(batch.get("chunk_eviction_count")) else 0
        chunk_load_time_ms_total += (
            float(batch.get("chunk_load_time_ms", torch.empty(0)).sum().item())
            if torch.is_tensor(batch.get("chunk_load_time_ms"))
            else 0.0
        )
        if is_train and augmentation_cfg:
            batch = _apply_graph_augmentation(batch, augmentation_cfg)
        batch = move_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with _autocast(amp_enabled):
                out = model(batch)
                loss_dict = criterion(out, batch["y"], batch)
                loss_dict = {key: _reduce_loss_value(value) for key, value in loss_dict.items()}
                loss = loss_dict["loss"]
        _finite_check(out, loss, split, epoch, batch_idx)
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip and float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
            global_step += 1
        pred = out["logits"].detach().argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + _float(value)
        for key, value in _slot_stats(out.get("aux", {})).items():
            slot_totals[key] = slot_totals.get(key, 0.0) + float(value)
        for key, value in compute_assignment_stats(out.get("aux", {})).items():
            pool_totals[key] = pool_totals.get(key, 0.0) + float(value)
        for key, value in _supcon_stats(loss_dict).items():
            supcon_totals[key] = supcon_totals.get(key, 0.0) + float(value)
        count += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_time_total += time.perf_counter() - batch_wall_start
        data_wait_start = time.perf_counter()
    if count == 0:
        raise RuntimeError(f"No batches processed for split={split}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(_metrics(y_true, y_pred))
    metrics["seconds"] = float(time.perf_counter() - start)
    metrics[f"{split}_epoch_time_sec"] = metrics["seconds"]
    metrics["first_batch_wait_time_sec"] = float(first_batch_wait_time or 0.0)
    metrics["avg_batch_time_ms"] = float(batch_time_total / max(count, 1) * 1000.0)
    metrics["num_batches"] = int(count)
    metrics["num_samples"] = int(len(y_true))
    total_cache = cache_hit_count + cache_miss_count
    metrics["cache_hit_count"] = int(cache_hit_count)
    metrics["cache_miss_count"] = int(cache_miss_count)
    metrics["cache_hit_rate"] = float(cache_hit_count / total_cache) if total_cache > 0 else 0.0
    metrics["chunk_load_count"] = int(chunk_load_count)
    metrics["chunk_eviction_count"] = int(chunk_eviction_count)
    metrics["avg_chunk_load_time_ms"] = float(chunk_load_time_ms_total / max(chunk_load_count, 1)) if chunk_load_count > 0 else 0.0
    return (
        metrics,
        {k: v / count for k, v in slot_totals.items()},
        {k: v / count for k, v in pool_totals.items()},
        {k: v / count for k, v in supcon_totals.items()},
        int(global_step),
    )


def _loader_dataset(loader):
    dataset = getattr(loader, "dataset", None)
    return dataset


def _append_cache_stats(output_root: Path, epoch: int, split: str, loader, metrics: Dict[str, float]) -> None:
    dataset = _loader_dataset(loader)
    inner = getattr(dataset, "dataset", None)
    row = {
        "epoch": int(epoch),
        "split": split,
        "chunk_cache_size": int(getattr(inner, "chunk_cache_size", 0) or 0),
        "num_chunks": int(getattr(dataset, "num_chunks", 0) or 0),
        "cache_hit_count": int(metrics.get("cache_hit_count", 0)),
        "cache_miss_count": int(metrics.get("cache_miss_count", 0)),
        "cache_hit_rate": float(metrics.get("cache_hit_rate", 0.0)),
        "chunk_load_count": int(metrics.get("chunk_load_count", 0)),
        "chunk_eviction_count": int(metrics.get("chunk_eviction_count", 0)),
        "avg_chunk_load_time_ms": float(metrics.get("avg_chunk_load_time_ms", 0.0)),
    }
    _append_csv(output_root / "chunk_cache_stats.csv", row)


def _from_scratch_info(model: torch.nn.Module, config: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    bad_paths = []
    for section in ("model", "training"):
        cfg = config.get(section, {}) or {}
        for key in ("init_checkpoint", "init_d13b_checkpoint", "resume_from", "checkpoint", "pretrained_checkpoint"):
            value = cfg.get(key)
            if value not in (None, "", "null", False):
                bad_paths.append(f"{section}.{key}={value}")
    if bad_paths:
        raise ValueError("D15 from-scratch forbids checkpoint fields: " + "; ".join(bad_paths))
    target = model.module if hasattr(model, "module") else model
    info = {
        "from_scratch": True,
        "init_checkpoint": None,
        "load_pretrained": False,
        "loaded_keys": 0,
        "forbidden_checkpoint_fields": [],
        "trainable_parameters": int(sum(p.numel() for p in target.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in target.parameters())),
    }
    (output_root / "d15_from_scratch_init.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def build_objects(config: Dict[str, Any], output_root: Path, device_arg: str | None = None):
    seed = int(config.get("training", {}).get("seed", config.get("run", {}).get("seed", 42)))
    set_seed(seed)
    device = resolve_device(device_arg, config)
    model = D13CSupConModel.from_config(config.get("model", {})).to(device)
    training_cfg = config.get("training", {}) or {}
    if bool(training_cfg.get("compile_model", training_cfg.get("use_compile", False))):
        if device.type != "cuda":
            print("[Compile] torch.compile skipped (only supported on CUDA)")
        elif not hasattr(torch, "compile"):
            print("[Compile] torch.compile not available in this PyTorch version")
        else:
            mode = training_cfg.get("compile_mode")
            backend = training_cfg.get("compile_backend", None)
            kwargs = {}
            if mode:
                kwargs["mode"] = str(mode)
            if backend:
                kwargs["backend"] = str(backend)
            if "compile_fullgraph" in training_cfg:
                kwargs["fullgraph"] = bool(training_cfg.get("compile_fullgraph"))
            try:
                model = torch.compile(model, **kwargs)
                print(
                    f"[Compile] torch.compile enabled mode={mode or 'default'} "
                    f"backend={backend or 'default'} fullgraph={kwargs.get('fullgraph', 'default')}"
                )
            except Exception as exc:
                print(f"[Compile] torch.compile failed: {exc}")
    init_info = _from_scratch_info(model, config, output_root)
    criterion = D15ScheduledLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    return model, criterion, optimizer, scheduler, device, init_info


def _write_report(
    output_root: Path,
    config: Dict[str, Any],
    best_epoch: int,
    best_value: float,
    test_metrics: Dict[str, float],
    slot_stats: Dict[str, float],
    pool_stats: Dict[str, float],
    supcon_stats: Dict[str, float],
    init_info: Dict[str, Any],
) -> None:
    lines = [
        "# D15 From-Scratch Report",
        "",
        "D15 is from-scratch end-to-end training, not D13 continuation. It is performance-first and does not make motif, semantic-region, causal-evidence, or full interpretability claims.",
        "",
        f"- run_name: `{config.get('run', {}).get('config_name', output_root.name)}`",
        "- from_scratch: true",
        "- init_checkpoint: null",
        "- load_pretrained: false",
        f"- loaded_keys: {init_info.get('loaded_keys', 0)}",
        f"- trainable_parameters: {init_info.get('trainable_parameters')}",
        f"- total_parameters: {init_info.get('total_parameters')}",
        f"- num_slots: {config.get('model', {}).get('num_slots')}",
        f"- lambda_supcon_target: {config.get('loss', {}).get('lambda_supcon_target', config.get('loss', {}).get('lambda_supcon', 0.0))}",
        f"- supcon_start_epoch: {config.get('loss', {}).get('supcon_start_epoch')}",
        f"- supcon_ramp_epochs: {config.get('loss', {}).get('supcon_ramp_epochs')}",
        f"- slot_loss_start_epoch: {config.get('loss', {}).get('slot_loss_start_epoch')}",
        f"- best_epoch: {best_epoch}",
        f"- best_val_macro_f1: {best_value:.6f}",
        f"- test_macro_f1: {test_metrics.get('macro_f1', 0.0):.6f}",
        f"- test_accuracy: {test_metrics.get('accuracy', 0.0):.6f}",
        f"- effective_slots: {slot_stats.get('effective_slots', 0.0):.6f}",
        f"- slot_overlap: {slot_stats.get('slot_overlap', 0.0):.6f}",
        f"- slot_entropy: {slot_stats.get('slot_entropy', 0.0):.6f}",
        f"- slot_dominance: {slot_stats.get('slot_dominance', 0.0):.6f}",
        f"- lambda_supcon_current_test: {supcon_stats.get('lambda_supcon_current', 0.0):.6f}",
        f"- positive_pair_count_test: {supcon_stats.get('positive_pair_count', 0.0):.6f}",
        "",
        "Target status is judged by D15 checker thresholds: 0.65 baseline pass, 0.68 promising, 0.70 target reached.",
        "",
    ]
    (output_root / "d15_report.md").write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def _export_test_diagnostics(model, loader, device, output_root: Path, max_batches: int | None = None) -> None:
    model.eval()
    z_images, z_projs, slot_attns = [], [], []
    graph_ids, labels, preds = [], [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        z_images.append(out["z_image"].detach().cpu().float().numpy())
        z_projs.append(out["z_proj"].detach().cpu().float().numpy())
        slot_attns.append(out["slot_attention"].detach().cpu().float().numpy())
        pred = out["logits"].argmax(dim=1)
        labels.extend(batch["y"].detach().cpu().long().tolist())
        preds.extend(pred.detach().cpu().long().tolist())
        graph_ids.extend(batch.get("graph_id", batch.get("sample_idx")).detach().cpu().long().tolist())
    if z_images:
        np.savez_compressed(
            output_root / "d15_test_embeddings.npz",
            z_image=np.concatenate(z_images, axis=0),
            z_proj=np.concatenate(z_projs, axis=0),
            slot_attention=np.concatenate(slot_attns, axis=0),
            graph_id=np.asarray(graph_ids, dtype=np.int64),
            label=np.asarray(labels, dtype=np.int64),
            pred=np.asarray(preds, dtype=np.int64),
        )


def run_train(
    config: Dict[str, Any],
    output_dir: str | Path | None = None,
    device_arg: str | None = None,
    resume_from: str | None = None,
    restore_rng: bool = True,
    max_runtime_hours: float | None = None,
    save_before_exit_minutes: float = 20.0,
    allow_resume_from_best: bool = False,
    allow_batch_size_change: bool = False,
) -> Dict[str, Any]:
    if not bool(config.get("from_scratch", config.get("d15", {}).get("from_scratch", True))):
        raise ValueError("D15 requires from_scratch=true")
    config.setdefault("model", {})
    config["model"]["init_d13b_checkpoint"] = None
    if output_dir is not None:
        config.setdefault("paths", {})["resolved_output_root"] = str(output_dir)
        config.setdefault("paths", {})["output_root"] = str(output_dir)
    output_root = Path(config.get("paths", {}).get("resolved_output_root") or output_dir or "outputs/d15_from_scratch/run")
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    (output_root / "RUN_INTERRUPTED_FOR_TIME_LIMIT").unlink(missing_ok=True)
    (output_root / "RUN_COMPLETE").unlink(missing_ok=True)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    test_loader = build_dataloader(config, "test", shuffle=False)
    model, criterion, optimizer, scheduler, device, init_info = build_objects(config, output_root, device_arg=device_arg)
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 150)))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    amp = bool(training_cfg.get("amp", False))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")
    patience = int(training_cfg.get("early_stopping_patience", 25))
    min_epochs_before_stop = int(training_cfg.get("min_epochs_before_early_stop", 0))
    monitor = str(training_cfg.get("early_stopping_metric", training_cfg.get("monitor", "val_macro_f1")))
    augmentation_cfg = dict(config.get("augmentation", {}) or {})
    amp_enabled = _amp_is_enabled(amp, device)
    scaler = _make_grad_scaler(amp_enabled)
    best_value = -float("inf")
    best_epoch = -1
    stale = 0
    global_step = 0
    start_epoch = 1
    resume_source = None
    history = []
    if resume_from:
        resume_state = _load_d15_resume(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epochs=epochs,
            output_root=output_root,
            restore_rng=restore_rng,
            allow_resume_from_best=allow_resume_from_best,
            allow_batch_size_change=allow_batch_size_change,
        )
        start_epoch = int(resume_state["start_epoch"])
        global_step = int(resume_state["global_step"])
        best_value = float(resume_state["best_value"])
        best_epoch = int(resume_state["best_epoch"])
        stale = int(resume_state["stale"])
        resume_source = str(resume_state["resume_source"])
    run_start = time.perf_counter()
    interrupted_for_time = False
    completed_training = False

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_slot, train_pool, train_supcon, global_step = _run_epoch(
            model, criterion, train_loader, optimizer, device, epoch, "train",
            max_train_batches, grad_clip, amp, augmentation_cfg, scaler=scaler,
            global_step=global_step,
        )
        val_metrics, val_slot, val_pool, val_supcon, global_step = _run_epoch(
            model, criterion, val_loader, None, device, epoch, "val",
            max_val_batches, grad_clip, amp=False, global_step=global_step,
        )
        _append_cache_stats(output_root, epoch, "train", train_loader, train_metrics)
        _append_cache_stats(output_root, epoch, "val", val_loader, val_metrics)
        if scheduler is not None:
            step_scheduler(scheduler, monitor_value=val_metrics.get("macro_f1"))
        stage = _stage_name(epoch, config.get("loss", {}) or {})
        early_stop_allowed = bool(epoch >= min_epochs_before_stop)
        row = {
            "epoch": epoch,
            "global_step": int(global_step),
            "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
            "lambda_supcon": float(train_supcon.get("lambda_supcon_current", 0.0)),
            "stage": stage,
            "early_stop_allowed": early_stop_allowed,
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["train_epoch_time_sec"] = float(train_metrics.get("seconds", 0.0))
        row["val_epoch_time_sec"] = float(val_metrics.get("seconds", 0.0))
        row["total_epoch_time_sec"] = row["train_epoch_time_sec"] + row["val_epoch_time_sec"]
        row["epoch_time_sec"] = row["total_epoch_time_sec"]
        row["ce_loss"] = float(train_metrics.get("loss_ce", 0.0))
        row["pool_loss"] = float(train_metrics.get("loss_pool_total", 0.0))
        row["slot_loss"] = float(train_metrics.get("loss_slot_total", 0.0))
        row["supcon_loss"] = float(train_metrics.get("loss_supcon", 0.0))
        row["val_macro_f1"] = float(val_metrics.get("macro_f1", 0.0))
        row.update({"from_scratch": True, "loaded_keys": 0, "trainable_parameters": init_info.get("trainable_parameters", 0)})
        for split, slot, pool, supcon, metrics in (
            ("train", train_slot, train_pool, train_supcon, train_metrics),
            ("val", val_slot, val_pool, val_supcon, val_metrics),
        ):
            _append_csv(output_root / "slot_stats.csv", {"epoch": epoch, "split": split, **slot})
            _append_csv(output_root / "pooling_stats.csv", {"epoch": epoch, "split": split, **pool})
            _append_csv(output_root / "supcon_stats.csv", {"epoch": epoch, "split": split, **supcon})
            _append_csv(output_root / "pred_count.csv", _pred_count_row(epoch, split, metrics))
        history.append(row)
        value = float(row.get(monitor, row.get("val_macro_f1", -float("inf"))))
        if value > best_value:
            best_value = value
            best_epoch = epoch
            stale = 0
            improved = True
        else:
            stale += 1
            improved = False
        row["best_val_macro_f1"] = float(best_value)
        row["best_epoch"] = int(best_epoch)
        row["patience_counter"] = int(stale)
        early_state = {
            "best_val_macro_f1": float(best_value),
            "best_epoch": int(best_epoch),
            "patience_counter": int(stale),
            "patience": int(patience),
            "min_epochs_before_early_stop": int(min_epochs_before_stop),
            "monitor": monitor,
            "mode": "max",
        }
        _append_csv(output_root / "train_log.csv", row)
        _append_csv(output_root / "val_metrics.csv", {"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if improved:
            _save_d15_checkpoint(
                output_root / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler,
                epoch, global_step, row, config, best_value, best_epoch, early_state, resume_source,
            )
        _save_d15_checkpoint(
            output_root / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler,
            epoch, global_step, row, config, best_value, best_epoch, early_state, resume_source,
        )
        _save_d15_checkpoint(
            output_root / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler,
            epoch, global_step, row, config, best_value, best_epoch, early_state, resume_source,
        )
        print(
            f"Epoch {epoch:03d}/{epochs:03d} train_loss={train_metrics['loss']:.4f} "
            f"lambda_supcon={train_supcon.get('lambda_supcon_current', 0.0):.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} best={best_value:.4f}@{best_epoch} "
            f"stage={stage} early_stop_allowed={early_stop_allowed} "
            f"patience_counter={stale} lr={row['lr']:.6g} "
            f"train_epoch_time_sec={row['train_epoch_time_sec']:.1f} "
            f"val_epoch_time_sec={row['val_epoch_time_sec']:.1f}"
        )
        if max_runtime_hours is not None:
            elapsed = time.perf_counter() - run_start
            remaining = float(max_runtime_hours) * 3600.0 - elapsed
            if remaining <= float(save_before_exit_minutes) * 60.0:
                interrupted_for_time = True
                _save_d15_checkpoint(
                    output_root / "checkpoints" / "interrupted.pt", model, optimizer, scheduler, scaler,
                    epoch, global_step, row, config, best_value, best_epoch, early_state, resume_source,
                    interrupted=True,
                )
                _save_d15_checkpoint(
                    output_root / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler,
                    epoch, global_step, row, config, best_value, best_epoch, early_state, resume_source,
                    interrupted=True,
                )
                (output_root / "RUN_INTERRUPTED_FOR_TIME_LIMIT").write_text(
                    json.dumps(
                        {
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "max_runtime_hours": float(max_runtime_hours),
                            "save_before_exit_minutes": float(save_before_exit_minutes),
                            "remaining_seconds": float(remaining),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("RUN_INTERRUPTED_FOR_TIME_LIMIT")
                break
        if early_stop_allowed and stale >= patience:
            print(f"Early stopping after {stale} stale epochs")
            completed_training = True
            break
    else:
        completed_training = True

    (output_root / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if interrupted_for_time:
        return {
            "best_epoch": best_epoch,
            "best_metric": best_value,
            "output_dir": str(output_root),
            "interrupted": True,
        }
    best_path = output_root / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_slot, test_pool, test_supcon, global_step = _run_epoch(
        model, criterion, test_loader, None, device, best_epoch, "test",
        max_test_batches, grad_clip, amp=False, global_step=global_step,
    )
    _append_cache_stats(output_root, best_epoch, "test", test_loader, test_metrics)
    print(f"test_time_sec={test_metrics.get('seconds', 0.0):.1f}")
    _append_csv(output_root / "test_metrics.csv", {"epoch": best_epoch, **{f"test_{k}": v for k, v in test_metrics.items()}})
    per_class = {"epoch": best_epoch}
    for key, value in test_metrics.items():
        if str(key).startswith("f1_"):
            per_class[key] = value
    _append_csv(output_root / "per_class_metrics.csv", per_class)
    _append_csv(output_root / "slot_stats.csv", {"epoch": best_epoch, "split": "test", **test_slot})
    _append_csv(output_root / "pooling_stats.csv", {"epoch": best_epoch, "split": "test", **test_pool})
    _append_csv(output_root / "supcon_stats.csv", {"epoch": best_epoch, "split": "test", **test_supcon})
    _append_csv(output_root / "pred_count.csv", _pred_count_row(best_epoch, "test", test_metrics))
    y_true, y_pred, graph_ids = _test_predictions(model, test_loader, device, max_test_batches)
    pd.DataFrame({"graph_id": graph_ids, "y_true": y_true, "y_pred": y_pred}).to_csv(output_root / "test_predictions.csv", index=False)
    write_confusion_matrix(y_true, y_pred, output_root / "confusion_matrix.csv")
    _export_test_diagnostics(model, test_loader, device, output_root, max_test_batches)
    _write_report(output_root, config, best_epoch, best_value, test_metrics, test_slot, test_pool, test_supcon, init_info)
    (output_root / "RUN_COMPLETE").write_text(
        json.dumps({"best_epoch": int(best_epoch), "best_val_macro_f1": float(best_value), "completed_training": bool(completed_training)}, indent=2),
        encoding="utf-8",
    )
    return {"best_epoch": best_epoch, "best_metric": best_value, "output_dir": str(output_root), "interrupted": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_epochs_override", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_test_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", default=None)
    parser.add_argument("--persistent_workers", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--chunk_aware_sampler", action="store_true", default=False)
    parser.add_argument("--chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--no_chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--restore_rng", dest="restore_rng", action="store_true")
    parser.add_argument("--no_restore_rng", dest="restore_rng", action="store_false")
    parser.set_defaults(restore_rng=True)
    parser.add_argument("--max_runtime_hours", type=float, default=None)
    parser.add_argument("--save_before_exit_minutes", type=float, default=20.0)
    parser.add_argument("--allow_resume_from_best", nargs="?", const=True, default=False, type=_str_to_bool)
    parser.add_argument("--allow_batch_size_change", nargs="?", const=True, default=False, type=_str_to_bool)
    args = parser.parse_args()
    if args.max_epochs_override is not None:
        args.epochs = int(args.max_epochs_override)
    if args.limit_train_batches is not None:
        args.max_train_batches = int(args.limit_train_batches)
    if args.limit_val_batches is not None:
        args.max_val_batches = int(args.limit_val_batches)
    if args.limit_test_batches is not None:
        args.max_test_batches = int(args.limit_test_batches)
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = args.output_dir
    run_train(
        config,
        output_dir=args.output_dir,
        device_arg=args.device,
        resume_from=args.resume_from,
        restore_rng=bool(args.restore_rng),
        max_runtime_hours=args.max_runtime_hours,
        save_before_exit_minutes=float(args.save_before_exit_minutes),
        allow_resume_from_best=bool(args.allow_resume_from_best),
        allow_batch_size_change=bool(args.allow_batch_size_change),
    )


if __name__ == "__main__":
    main()
