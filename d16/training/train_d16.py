"""Train D16 pixel-graph runs.

This runner supports CE-only and performance-oriented D16 v1 ablations. It does
not make motif, semantic-region, causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import csv
import copy
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import D16Batch, collate_d16_graphs
from d16.data.graph_cache_dataset import D16GraphCacheDataset
from d16.data.pixel_prior_dataset import D16PixelPriorDataset
from d16.losses.hard_proto_separation import (
    HardPrototypeSeparationLoss,
    build_hard_proto_separation_loss,
    hard_proto_lambda,
)
from d16.losses.main_logit_pair_margin import (
    MainLogitPairMarginLoss,
    build_main_logit_pair_margin_loss,
    main_logit_pair_margin_lambda,
)
from d16.losses.part_supcon import PartAwareSupConLoss
from d16.losses.pairwise_hard_relation import (
    PairwiseHardRelationLoss,
    build_pairwise_hard_relation_loss,
    pairwise_hard_relation_lambda,
)
from d16.models.d16_model import D16Model


class D16HybridDetectedFallbackDataset(torch.utils.data.Dataset):
    """Use one dataset for detected samples and another for fallback samples."""

    def __init__(self, detected_ds, fallback_ds) -> None:
        if len(detected_ds) != len(fallback_ds):
            raise ValueError(f"Hybrid D16 dataset length mismatch: detected={len(detected_ds)} fallback={len(fallback_ds)}")
        self.detected_ds = detected_ds
        self.fallback_ds = fallback_ds

    def __len__(self) -> int:
        return len(self.detected_ds)

    def set_epoch(self, epoch: int) -> None:
        for dataset in (self.detected_ds, self.fallback_ds):
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(int(epoch))

    def current_corruption_probability(self) -> float:
        getter = getattr(self.detected_ds, "current_corruption_probability", None)
        if callable(getter):
            return float(getter())
        return 0.0

    def __getitem__(self, index: int):
        raw_detected = None
        detector = getattr(self.detected_ds, "raw_detected", None)
        if callable(detector):
            raw_detected = bool(detector(index))
        detected_graph = self.detected_ds[index]
        if bool(detected_graph.detected.item()) or raw_detected is True:
            return detected_graph
        fallback_graph = self.fallback_ds[index]
        if int(detected_graph.sample_index.item()) != int(fallback_graph.sample_index.item()):
            raise ValueError(
                "Hybrid D16 dataset sample_index mismatch: "
                f"{int(detected_graph.sample_index.item())} != {int(fallback_graph.sample_index.item())}"
            )
        return fallback_graph


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    active_fieldnames = fieldnames
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            active_fieldnames = list(existing_header)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=active_fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in active_fieldnames})


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _wandb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    logging_cfg = cfg.get("logging", {}) or {}
    wandb_cfg = logging_cfg.get("wandb", {}) or {}
    if isinstance(wandb_cfg, bool):
        return {"enabled": bool(wandb_cfg)}
    if not isinstance(wandb_cfg, dict):
        return {"enabled": False}
    return dict(wandb_cfg)


def _wandb_run_id(output_dir: Path, wandb_module: Any) -> str:
    run_id_path = output_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        text = run_id_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    try:
        run_id = str(wandb_module.util.generate_id())
    except Exception:
        run_id = uuid.uuid4().hex[:8]
    run_id_path.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def _init_wandb(cfg: Dict[str, Any], output_dir: Path, resume_path: Path | None = None):
    wandb_cfg = _wandb_cfg(cfg)
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[D16 wandb] disabled because import failed: {exc}", flush=True)
        return None

    run_name = str(cfg.get("run_name") or output_dir.name)
    run_id = _wandb_run_id(output_dir, wandb)
    init_kwargs = {
        "project": str(wandb_cfg.get("project") or "lapgnn-d16-overfit-fix-1"),
        "name": str(wandb_cfg.get("name") or run_name),
        "id": run_id,
        "resume": str(wandb_cfg.get("resume", "allow")),
        "config": cfg,
        "dir": str(output_dir),
    }
    for key in ("entity", "group", "mode", "job_type", "notes"):
        value = wandb_cfg.get(key)
        if value not in (None, "", "null"):
            init_kwargs[key] = value
    tags = wandb_cfg.get("tags")
    if tags:
        init_kwargs["tags"] = list(tags)
    try:
        run = wandb.init(**init_kwargs)
        run.summary["output_dir"] = str(output_dir)
        run.summary["run_name"] = run_name
        run.summary["resume_path"] = None if resume_path is None else str(resume_path)
        print(f"[D16 wandb] enabled project={init_kwargs['project']} run={run_name} id={run_id}", flush=True)
        return run
    except Exception as exc:
        print(f"[D16 wandb] disabled because init failed: {exc}", flush=True)
        return None


def _wandb_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _wandb_log_epoch(wandb_run: Any, log_row: Dict[str, Any], score_status: Dict[str, Any] | None = None) -> None:
    if wandb_run is None:
        return
    metrics: Dict[str, Any] = {}

    def put(name: str, value: Any) -> None:
        number = _wandb_number(value)
        if number is not None:
            metrics[name] = number

    put("epoch", log_row.get("epoch"))
    put("train/loss", log_row.get("train_loss"))
    put("train/eval_loss", log_row.get("train_eval_loss"))
    put("train/accuracy", log_row.get("train_accuracy"))
    put("train/macro_f1", log_row.get("train_macro_f1"))
    put("val/loss", log_row.get("val_loss"))
    put("val/accuracy", log_row.get("val_accuracy"))
    put("val/macro_f1", log_row.get("val_macro_f1"))
    put("loss/ce", log_row.get("ce_loss"))
    put("loss/consistency", log_row.get("consistency_loss_total"))
    put("loss/lambda_consistency", log_row.get("lambda_consistency_current"))
    put("monitor/score", log_row.get("monitor_score"))
    put("monitor/best_score_before_epoch", log_row.get("best_monitor_score"))
    put("runtime/epoch_time_sec", log_row.get("epoch_time_sec"))
    put("runtime/memory_reserved_mb", log_row.get("memory_reserved_mb"))
    put("runtime/train_epoch_time_sec", log_row.get("train_epoch_time_sec"))
    put("runtime/val_epoch_time_sec", log_row.get("val_epoch_time_sec"))
    put("data/node_count_mean", log_row.get("node_count_mean"))
    put("data/edge_count_mean", log_row.get("edge_count_mean"))
    put("data/fallback_samples_seen", log_row.get("fallback_samples_seen"))
    if score_status:
        put("monitor/is_best", score_status.get("is_best"))
        put("monitor/best_score_current", score_status.get("best_score_current"))
        put("monitor/best_epoch_current", score_status.get("best_epoch_current"))
        put("early_stopping/without_improvement_current", score_status.get("early_stopping_without_improvement_current"))
    if not metrics:
        return
    step_value = _wandb_number(log_row.get("global_step"))
    try:
        wandb_run.log(metrics, step=None if step_value is None else int(step_value))
    except Exception as exc:
        print(f"[D16 wandb] log failed: {exc}", flush=True)


def _wandb_log_final_outputs(wandb_run: Any, output_dir: Path) -> None:
    if wandb_run is None:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None
    filenames = [
        "d16_train_summary.json",
        "train_log.csv",
        "train_metrics.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "confusion_matrix.csv",
        "last_confusion_matrix.csv",
        "best_val_loss_confusion_matrix.csv",
        "confusion_matrix.png",
        "last_confusion_matrix.png",
        "best_val_loss_confusion_matrix.png",
        "per_class_metrics.csv",
        "pred_count.csv",
    ]
    for name in filenames:
        file_path = output_dir / name
        if file_path.exists():
            try:
                wandb_run.save(str(file_path), base_path=str(output_dir))
            except Exception as exc:
                print(f"[D16 wandb] save failed for {file_path}: {exc}", flush=True)
    confusion_png_path = output_dir / "confusion_matrix.png"
    if wandb is not None and confusion_png_path.exists():
        try:
            wandb_run.log({"test/confusion_matrix_image": wandb.Image(str(confusion_png_path))})
        except Exception as exc:
            print(f"[D16 wandb] confusion matrix image failed: {exc}", flush=True)
    confusion_path = output_dir / "confusion_matrix.csv"
    if wandb is not None and confusion_path.exists():
        try:
            with confusion_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                columns = list(reader.fieldnames or [])
            if rows and columns:
                table = wandb.Table(columns=columns, data=[[row.get(col) for col in columns] for row in rows])
                wandb_run.log({"test/confusion_matrix_table": table})
        except Exception as exc:
            print(f"[D16 wandb] confusion matrix table failed: {exc}", flush=True)


def _wandb_finish(wandb_run: Any) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as exc:
        print(f"[D16 wandb] finish failed: {exc}", flush=True)


def _write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    last_exc: OSError | None = None
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    raise last_exc or OSError(f"Failed to atomically replace checkpoint: {path}")


def _dual_validation_checkpoint_cfg(training_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return opt-in checkpoint instrumentation without changing old configs."""

    cfg = dict((training_cfg or {}).get("dual_validation_checkpoints", {}) or {})
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["preserve_best_macro_alias"] = bool(cfg.get("preserve_best_macro_alias", True))
    cfg["save_validation_predictions"] = bool(cfg.get("save_validation_predictions", True))
    return cfg


def _atomic_copy_checkpoint(source: Path, destination: Path) -> None:
    """Copy a completed checkpoint atomically; this does not touch RNG state."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp_path)
    os.replace(tmp_path, destination)


def canonical_model_state_hash(checkpoint_or_state: Dict[str, Any]) -> str:
    """Hash sorted tensor keys, dtype, shape and raw bytes."""

    import hashlib

    state = checkpoint_or_state.get("model_state_dict", checkpoint_or_state)
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_validation_checkpoint_snapshot(
    output_dir: Path,
    name: str,
    row: Dict[str, Any] | None,
    per_class: List[Dict[str, Any]],
    confusion: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    enabled: bool,
) -> None:
    if not enabled or row is None:
        return
    snapshot_dir = output_dir / "validation_snapshots"
    _write_json(snapshot_dir / f"{name}_metrics.json", dict(row))
    for suffix, rows in (
        ("per_class", per_class),
        ("confusion_matrix", confusion),
        ("predictions", predictions),
    ):
        if rows:
            _write_csv_rows(snapshot_dir / f"{name}_{suffix}.csv", rows, list(rows[0].keys()))


def _model_signature(config: Dict[str, Any], input_dim: int | None = None) -> Dict[str, Any]:
    graph = config.get("graph", {}) or {}
    model = config.get("model", {}) or {}
    loss = config.get("loss", {}) or {}
    detail = graph.get("detail_features", {}) or {}
    edge = graph.get("edge_features", {}) or {}
    knn = graph.get("knn_edges", {}) or {}
    edge_gnn = model.get("edge_context_gnn", {}) or {}
    micro = model.get("micro_motif_support", {}) or {}
    context = edge_gnn.get("context_injection", {}) or {}
    multiscale = edge_gnn.get("multiscale_fusion", {}) or {}
    return {
        "run_name": config.get("run_name"),
        "seed": config.get("seed", (config.get("training", {}) or {}).get("seed")),
        "input_dim": None if input_dim is None else int(input_dim),
        "architecture": model.get("architecture", "single_path"),
        "readout_type": model.get("readout_type"),
        "gnn_type": model.get("gnn_type", "part_aware"),
        "hidden_dim": model.get("hidden_dim", 96),
        "gnn_layers": model.get("gnn_layers", 3),
        "num_classes": model.get("num_classes", 7),
        "dual_head": bool(model.get("dual_head", False)),
        "edge_attr_dim": edge_gnn.get("edge_attr_dim") if model.get("gnn_type") == "edge_context_gnn" else None,
        "graph_mode": graph.get("graph_mode", (config.get("data", {}) or {}).get("graph_mode")),
        "detail_features_enabled": bool(detail.get("enabled", False)),
        "detail_features": list(detail.get("features") or []),
        "node_features": dict(graph.get("node_features", {}) or {}),
        "prior_usage": graph.get("prior_usage"),
        "edge_features_enabled": bool(edge.get("enabled", False)),
        "edge_features": list(edge.get("features") or []),
        "knn_edges": dict(knn or {}),
        "edge_context_layer_output_concat": bool(edge_gnn.get("layer_output_concat", False)),
        "edge_context_multiscale_enabled": bool(multiscale.get("enabled", False)),
        "edge_context_multiscale_layers": list(multiscale.get("layers") or []),
        "edge_context_multiscale_mode": multiscale.get("mode"),
        "micro_prior_gate": micro.get("prior_gate", {}) or {},
        "micro_prior_usage": micro.get("prior_usage"),
        "micro_use_log_prior_bias": micro.get("use_log_prior_bias"),
        "context_prior_gate": context.get("prior_gate", {}) or {},
        "loss_mode": loss.get("mode", "ce"),
        "optimizer_type": _optimizer_name_from_config(config),
        "scheduler_type": _scheduler_type_from_config(config),
    }


def _check_resume_compatibility(
    checkpoint: Dict[str, Any],
    current_config: Dict[str, Any],
    current_input_dim: int | None,
    strict: bool,
) -> Dict[str, Any]:
    checkpoint_sig = checkpoint.get("model_signature") or _model_signature(checkpoint.get("config", {}) or {}, checkpoint.get("input_dim"))
    current_sig = _model_signature(current_config, current_input_dim)
    allowed_diff = {"run_name"}
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key, current_value in current_sig.items():
        if key in allowed_diff:
            continue
        checkpoint_value = checkpoint_sig.get(key)
        if checkpoint_value != current_value:
            mismatches[key] = {"checkpoint": checkpoint_value, "current": current_value}
    result = {
        "compatible": not mismatches,
        "mismatches": mismatches,
        "checkpoint_signature": checkpoint_sig,
        "current_signature": current_sig,
    }
    if mismatches and strict:
        raise ValueError("D16 resume config compatibility check failed: " + json.dumps(mismatches, indent=2, default=str))
    return result


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[D16 train] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _amp_enabled(training_cfg: Dict[str, Any], device: torch.device) -> bool:
    return bool(training_cfg.get("amp", training_cfg.get("mixed_precision", False))) and device.type == "cuda"


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _scheduler_cfg(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = training_cfg.get("scheduler", {}) or {}
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if not isinstance(cfg, dict):
        cfg = {}
    out = dict(cfg)
    out["type"] = str(out.get("type", "none") or "none").lower()
    return out


def _optimizer_cfg(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = training_cfg.get("optimizer", {}) or {}
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if not isinstance(cfg, dict):
        raise ValueError("training.optimizer must be a mapping or optimizer name")
    out = dict(cfg)
    out["type"] = str(out.get("type", "adamw") or "adamw").lower()
    return out


def _optimizer_type_from_config(config: Dict[str, Any]) -> str:
    return _optimizer_cfg((config.get("training", {}) or {})).get("type", "adamw")


def _optimizer_name_from_config(config: Dict[str, Any]) -> str:
    optimizer_type = _optimizer_type_from_config(config)
    return {"adamw": "AdamW", "radam": "RAdam"}.get(optimizer_type, optimizer_type)


def _resolved_optimizer_signature(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _optimizer_cfg(training_cfg)
    optimizer_type = str(cfg["type"]).lower()
    signature: Dict[str, Any] = {
        "type": optimizer_type,
        "lr": float(cfg.get("lr", training_cfg.get("lr", 3e-4))),
        "weight_decay": float(cfg.get("weight_decay", training_cfg.get("weight_decay", 1e-4))),
        "betas": [float(value) for value in cfg.get("betas", (0.9, 0.999))],
        "eps": float(cfg.get("eps", 1e-8)),
    }
    if optimizer_type == "adamw":
        signature["amsgrad"] = bool(cfg.get("amsgrad", False))
    elif optimizer_type == "radam":
        signature["decoupled_weight_decay"] = bool(cfg.get("decoupled_weight_decay", False))
    return signature


def _build_optimizer(parameters: Any, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    signature = _resolved_optimizer_signature(training_cfg)
    optimizer_type = str(signature["type"])
    common = {
        "lr": float(signature["lr"]),
        "weight_decay": float(signature["weight_decay"]),
        "betas": tuple(float(value) for value in signature["betas"]),
        "eps": float(signature["eps"]),
    }
    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, **common, amsgrad=bool(signature["amsgrad"]))
    if optimizer_type == "radam":
        if "decoupled_weight_decay" not in inspect.signature(torch.optim.RAdam).parameters:
            raise RuntimeError("BLOCKED_RADAM_WEIGHT_DECAY_SEMANTICS")
        if not bool(signature.get("decoupled_weight_decay", False)):
            raise RuntimeError("RAdam requires decoupled_weight_decay=true for the registered D16 audit")
        return torch.optim.RAdam(parameters, **common, decoupled_weight_decay=True)
    raise ValueError(f"Unsupported D16 training.optimizer.type={optimizer_type!r}")


def _scheduler_type_from_config(config: Dict[str, Any]) -> str:
    return _scheduler_cfg((config.get("training", {}) or {})).get("type", "none")


def _resolved_scheduler_signature(training_cfg: Dict[str, Any], max_epochs: int | None = None) -> Dict[str, Any]:
    cfg = _scheduler_cfg(training_cfg)
    scheduler_type = str(cfg["type"])
    signature: Dict[str, Any] = {"type": scheduler_type}
    if scheduler_type == "plateau":
        signature.update({
            "class": "ReduceLROnPlateau",
            "monitor": str(cfg.get("monitor", "val_loss")),
            "mode": str(cfg.get("mode", "min")),
            "factor": float(cfg.get("factor", 0.5)),
            "patience": int(cfg.get("patience", 3)),
            "threshold": float(cfg.get("threshold", 1e-4)),
            "cooldown": int(cfg.get("cooldown", 0)),
            "min_lr": float(cfg.get("min_lr", 0.0)),
            "step_location": "after validation/checkpoint comparison, before last.pt save",
            "step_argument": "validation metric",
        })
    elif scheduler_type == "cosine":
        signature.update({
            "class": "CosineAnnealingLR",
            "t_max": int(cfg.get("t_max", max_epochs if max_epochs is not None else 0)),
            "eta_min": float(cfg.get("eta_min", 0.0)),
            "last_epoch": -1,
            "step_location": "after validation/checkpoint comparison, before last.pt save",
            "step_argument": None,
        })
    return signature


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: Dict[str, Any],
    max_epochs: int,
    steps_per_epoch: int,
):
    scheduler_cfg = _scheduler_cfg(training_cfg)
    scheduler_type = str(scheduler_cfg.get("type", "none") or "none").lower()
    if scheduler_type in {"", "none", "null"}:
        return None, "none", scheduler_cfg
    if scheduler_type == "onecycle":
        if int(steps_per_epoch) <= 0:
            raise ValueError("OneCycleLR requires steps_per_epoch > 0")
        max_lr = float(scheduler_cfg.get("max_lr", training_cfg.get("lr", 3e-4)))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=int(max_epochs),
            steps_per_epoch=int(steps_per_epoch),
            pct_start=float(scheduler_cfg.get("pct_start", 0.2)),
            div_factor=float(scheduler_cfg.get("div_factor", 10.0)),
            final_div_factor=float(scheduler_cfg.get("final_div_factor", 10.0)),
            anneal_strategy=str(scheduler_cfg.get("anneal_strategy", "cos")),
        )
        return scheduler, scheduler_type, scheduler_cfg
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", max_epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
        return scheduler, scheduler_type, scheduler_cfg
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            cooldown=int(scheduler_cfg.get("cooldown", 0)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        return scheduler, scheduler_type, scheduler_cfg
    raise ValueError(f"Unsupported D16 scheduler.type={scheduler_type!r}")


def _scheduler_steps_per_batch(scheduler_type: str) -> bool:
    return str(scheduler_type).lower() == "onecycle"


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _step_scheduler_epoch(scheduler: Any | None, scheduler_type: str, val_row: Dict[str, Any] | None, scheduler_cfg: Dict[str, Any]) -> None:
    if scheduler is None or _scheduler_steps_per_batch(scheduler_type):
        return
    if str(scheduler_type).lower() == "plateau":
        monitor = str(scheduler_cfg.get("monitor", "val_loss"))
        score = _monitor_value(val_row, monitor)
        if score is not None and math.isfinite(float(score)):
            scheduler.step(float(score))
        return
    scheduler.step()


def _checkpoint_policy_cfg(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = training_cfg.get("checkpoint_policy", {}) or {}
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if not isinstance(cfg, dict):
        cfg = {}
    out = dict(cfg)
    out["type"] = str(out.get("type", "standard") or "standard").lower()
    return out


def _loss_guard_ok(
    val_row: Dict[str, Any] | None,
    best_guard_loss: float,
    checkpoint_policy: Dict[str, Any],
    epoch: int | None = None,
) -> bool:
    if val_row is None:
        return False
    guard_start_epoch = int(checkpoint_policy.get("guard_start_epoch", 0) or 0)
    if epoch is not None and guard_start_epoch > 0 and int(epoch) < guard_start_epoch:
        return True
    loss_metric = str(checkpoint_policy.get("loss_metric", "val_loss"))
    current_loss = _monitor_value(val_row, loss_metric)
    if current_loss is None or not math.isfinite(float(current_loss)):
        return False
    if not math.isfinite(float(best_guard_loss)):
        return True
    abs_tol = float(checkpoint_policy.get("max_loss_degrade_abs", 0.0) or 0.0)
    rel_tol = float(checkpoint_policy.get("max_loss_degrade_rel", 0.0) or 0.0)
    limit = float(best_guard_loss) + abs_tol
    if rel_tol > 0.0:
        limit = max(limit, float(best_guard_loss) * (1.0 + rel_tol))
    return float(current_loss) <= limit


def _clone_batch_with_regularized_features(
    batch: D16Batch,
    x_cat: torch.Tensor,
    edge_attr_cat: torch.Tensor | None,
    edge_index_cat: torch.Tensor | None = None,
    part_soft_cat: torch.Tensor | None = None,
    face_mask_cat: torch.Tensor | None = None,
) -> D16Batch:
    return D16Batch(
        x_cat=x_cat,
        edge_index_cat=batch.edge_index_cat if edge_index_cat is None else edge_index_cat,
        edge_attr_cat=edge_attr_cat,
        batch_index=batch.batch_index,
        ptr=batch.ptr,
        y=batch.y,
        sample_index=batch.sample_index,
        pos_cat=batch.pos_cat,
        part_soft_cat=batch.part_soft_cat if part_soft_cat is None else part_soft_cat,
        face_mask_cat=batch.face_mask_cat if face_mask_cat is None else face_mask_cat,
        valid_part_mask=batch.valid_part_mask,
        valid_anchor_mask=batch.valid_anchor_mask,
        detected=batch.detected,
        landmark_missing_flag=batch.landmark_missing_flag,
        image_48=batch.image_48,
    )


def _scheduled_graph_probability(cfg: Dict[str, Any], epoch: int, default: float = 0.0) -> float:
    probability = float(cfg.get("probability", default) or 0.0)
    for item in cfg.get("schedule") or []:
        if int(epoch) >= int(item.get("start_epoch", 1) or 1):
            probability = float(item.get("probability", probability) or 0.0)
    return min(max(probability, 0.0), 1.0)


def _node_prior_regularization_probability(graph_regularization: Dict[str, Any] | None, epoch: int) -> float:
    cfg = (graph_regularization or {}).get("node_prior_regularization", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return 0.0
    return _scheduled_graph_probability(cfg, epoch)


def _apply_node_prior_regularization(
    batch: D16Batch,
    x_cat: torch.Tensor,
    cfg: Dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(cfg.get("enabled", False)):
        return x_cat, batch.part_soft_cat, batch.face_mask_cat
    probability = _scheduled_graph_probability(cfg, epoch)
    if probability <= 0.0 or x_cat.numel() == 0:
        return x_cat, batch.part_soft_cat, batch.face_mask_cat

    x_out = x_cat.clone()
    part_out = batch.part_soft_cat.clone()
    face_out = batch.face_mask_cat.clone()

    face_index = int(cfg.get("face_index", 5))
    part_indices = [int(v) for v in (cfg.get("part_indices") or list(range(6, 13)))]
    distance_indices = [int(v) for v in (cfg.get("distance_indices") or list(range(13, 20)))]
    mode = str(cfg.get("mode", "attenuate"))
    keep = min(max(float(cfg.get("keep", 0.65) or 0.65), 0.0), 1.0)
    distance_neutral = float(cfg.get("distance_neutral", 1.0))
    face_neutral = float(cfg.get("face_neutral", 0.0))
    part_neutral = float(cfg.get("part_neutral", 0.0))
    apply_face = bool(cfg.get("apply_face", True)) and 0 <= face_index < x_out.size(1)
    apply_part = bool(cfg.get("apply_part", True))
    apply_distance = bool(cfg.get("apply_distance", True))

    def mix(values: torch.Tensor, neutral: float) -> torch.Tensor:
        neutral_tensor = torch.full_like(values, float(neutral))
        if mode == "dropout":
            return neutral_tensor
        if mode == "attenuate":
            return values * keep + neutral_tensor * (1.0 - keep)
        raise ValueError(f"Unsupported D16 node_prior_regularization mode={mode!r}")

    for graph_idx in range(batch.num_graphs):
        if float(torch.rand((), device=x_out.device).item()) >= probability:
            continue
        start = int(batch.ptr[graph_idx].item())
        end = int(batch.ptr[graph_idx + 1].item())
        if end <= start:
            continue
        node_slice = slice(start, end)
        if apply_face:
            x_out[node_slice, face_index] = mix(x_out[node_slice, face_index], face_neutral)
            face_out[node_slice] = mix(face_out[node_slice], face_neutral)
        if apply_part:
            valid_part_indices = [idx for idx in part_indices if 0 <= idx < x_out.size(1)]
            if valid_part_indices:
                x_out[node_slice, valid_part_indices] = mix(x_out[node_slice, valid_part_indices], part_neutral)
            part_out[node_slice] = mix(part_out[node_slice], part_neutral)
        if apply_distance:
            valid_distance_indices = [idx for idx in distance_indices if 0 <= idx < x_out.size(1)]
            if valid_distance_indices:
                x_out[node_slice, valid_distance_indices] = mix(x_out[node_slice, valid_distance_indices], distance_neutral)
    return x_out, part_out, face_out


def _apply_train_graph_regularization(batch: D16Batch, cfg: Dict[str, Any] | None, epoch: int = 0) -> D16Batch:
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return batch
    x_cat = batch.x_cat
    edge_index_cat = batch.edge_index_cat
    edge_attr_cat = batch.edge_attr_cat
    part_soft_cat = batch.part_soft_cat
    face_mask_cat = batch.face_mask_cat
    node_prior_cfg = cfg.get("node_prior_regularization", {}) or {}
    if bool(node_prior_cfg.get("enabled", False)):
        x_cat, part_soft_cat, face_mask_cat = _apply_node_prior_regularization(batch, x_cat, node_prior_cfg, epoch)
    noise_std = float(cfg.get("node_feature_noise_std", 0.0) or 0.0)
    node_feature_dropout = float(cfg.get("node_feature_dropout", 0.0) or 0.0)
    node_dropout = float(cfg.get("node_dropout_prob", 0.0) or 0.0)
    edge_attr_dropout = float(cfg.get("edge_attr_dropout", 0.0) or 0.0)
    edge_drop_prob = float(cfg.get("edge_dropout_prob", 0.0) or 0.0)
    if noise_std > 0.0:
        x_cat = x_cat + torch.randn_like(x_cat) * noise_std
    if node_feature_dropout > 0.0:
        x_cat = F.dropout(x_cat, p=min(max(node_feature_dropout, 0.0), 0.95), training=True)
    if node_dropout > 0.0 and x_cat.numel() > 0:
        keep = torch.rand((x_cat.size(0), 1), device=x_cat.device, dtype=x_cat.dtype) >= min(max(node_dropout, 0.0), 0.95)
        x_cat = x_cat * keep
    if edge_drop_prob > 0.0 and edge_index_cat.numel() > 0:
        keep_prob = 1.0 - min(max(edge_drop_prob, 0.0), 0.95)
        keep_edge = torch.rand((edge_index_cat.size(1),), device=edge_index_cat.device) < keep_prob
        if bool(keep_edge.any()):
            edge_index_cat = edge_index_cat[:, keep_edge]
            if edge_attr_cat is not None:
                edge_attr_cat = edge_attr_cat[keep_edge]
    if edge_attr_cat is not None and edge_attr_dropout > 0.0:
        edge_attr_cat = F.dropout(edge_attr_cat, p=min(max(edge_attr_dropout, 0.0), 0.95), training=True)
    if (
        x_cat is batch.x_cat
        and edge_attr_cat is batch.edge_attr_cat
        and edge_index_cat is batch.edge_index_cat
        and part_soft_cat is batch.part_soft_cat
        and face_mask_cat is batch.face_mask_cat
    ):
        return batch
    return _clone_batch_with_regularized_features(
        batch,
        x_cat,
        edge_attr_cat,
        edge_index_cat=edge_index_cat,
        part_soft_cat=part_soft_cat,
        face_mask_cat=face_mask_cat,
    )


def _graph_consistency_cfg(graph_regularization: Dict[str, Any] | None, epoch: int) -> tuple[Dict[str, Any], float]:
    cfg = graph_regularization or {}
    cons_cfg = cfg.get("consistency", {}) or {}
    if not bool(cons_cfg.get("enabled", False)):
        return cons_cfg, 0.0
    start_epoch = int(cons_cfg.get("start_epoch", 1) or 1)
    if int(epoch) < start_epoch:
        return cons_cfg, 0.0
    weight = float(cons_cfg.get("weight", 0.0) or 0.0)
    return cons_cfg, max(weight, 0.0)


def _logit_consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, cfg: Dict[str, Any] | None) -> torch.Tensor:
    cfg = cfg or {}
    temperature = max(float(cfg.get("temperature", 1.0) or 1.0), 1e-6)
    symmetric = bool(cfg.get("symmetric", True))
    logp_a = F.log_softmax(logits_a / temperature, dim=1)
    logp_b = F.log_softmax(logits_b / temperature, dim=1)
    prob_a = F.softmax(logits_a.detach() / temperature, dim=1)
    prob_b = F.softmax(logits_b.detach() / temperature, dim=1)
    loss_ab = F.kl_div(logp_a, prob_b, reduction="batchmean")
    if not symmetric:
        return loss_ab * (temperature * temperature)
    loss_ba = F.kl_div(logp_b, prob_a, reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temperature * temperature)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))


def _dataset_prior_corruption_probability(dataset: Any) -> float:
    getter = getattr(dataset, "current_corruption_probability", None)
    if callable(getter):
        return float(getter())
    return 0.0



def _dataset_edge_prior_regularization_probability(dataset: Any) -> float:
    getter = getattr(dataset, "current_edge_prior_regularization_probability", None)
    if callable(getter):
        return float(getter())
    return 0.0

def _loader_kwargs(data_cfg: Dict[str, Any], training_cfg: Dict[str, Any], shuffle: bool) -> Dict[str, Any]:
    num_workers = int(training_cfg.get("num_workers", data_cfg.get("num_workers", 0)) or 0)
    kwargs: Dict[str, Any] = {
        "batch_size": int(training_cfg.get("batch_size", data_cfg.get("batch_size", 8))),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", data_cfg.get("pin_memory", False))),
        "collate_fn": collate_d16_graphs,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", data_cfg.get("persistent_workers", False)))
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", data_cfg.get("prefetch_factor", 2)))
    return kwargs


def _single_dataset(
    prior_dir: str | Path,
    split: str,
    graph_mode: str,
    face_threshold: float,
    context_pixels: int,
    max_samples: int | None,
    detail_features: Dict[str, Any] | None = None,
    edge_features: Dict[str, Any] | None = None,
    anchor_nodes: Dict[str, Any] | None = None,
    node_features: Dict[str, Any] | None = None,
    knn_edges: Dict[str, Any] | None = None,
    prior_usage: str | None = None,
    prior_corruption: Dict[str, Any] | None = None,
    graph_cache_dir: str | Path | None = None,
    chunk_cache_size: int = 2,
):
    if graph_cache_dir and bool((detail_features or {}).get("enabled", False)):
        raise ValueError("D16 detail node features require graph_cache_dir=null unless a matching 37-dim cache is built.")
    if graph_cache_dir and bool((edge_features or {}).get("enabled", False)):
        raise ValueError("D16 edge features require graph_cache_dir=null unless a matching edge-attr cache is built.")
    if graph_cache_dir and bool((anchor_nodes or {}).get("enabled", False)):
        raise ValueError("D16 anchor nodes require graph_cache_dir=null unless a matching anchor-node cache is built.")
    if graph_cache_dir and bool((knn_edges or {}).get("enabled", False)):
        raise ValueError("D16 k-NN edges require graph_cache_dir=null unless a matching k-NN edge cache is built.")
    if graph_cache_dir and bool((prior_corruption or {}).get("enabled", False)):
        raise ValueError("D16 prior corruption requires graph_cache_dir=null because priors are mutated before graph building.")
    if graph_cache_dir:
        return D16GraphCacheDataset(
            graph_cache_dir,
            split=split,
            graph_mode=graph_mode,
            face_threshold=face_threshold,
            context_pixels=context_pixels,
            max_samples=max_samples,
            chunk_cache_size=chunk_cache_size,
        )
    return D16PixelPriorDataset(
        prior_dir,
        split=split,
        graph_mode=graph_mode,
        face_threshold=face_threshold,
        context_pixels=context_pixels,
        detail_features=detail_features,
        edge_features=edge_features,
        anchor_nodes=anchor_nodes,
        node_features=node_features,
        knn_edges=knn_edges,
        prior_usage=prior_usage,
        prior_corruption=prior_corruption,
        max_samples=max_samples,
    )


def build_dataset(cfg: Dict[str, Any], prior_dir: str | Path, split: str):
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    max_key = f"max_{split}_samples"
    max_samples = data_cfg.get(max_key)
    if max_samples is not None:
        max_samples = int(max_samples)
    graph_mode = graph_cfg.get("graph_mode", data_cfg.get("graph_mode", "face_plus_context"))
    face_threshold = float(graph_cfg.get("face_threshold", 0.15))
    context_pixels = int(graph_cfg.get("context_pixels", 2))
    detail_features = graph_cfg.get("detail_features", {}) or {}
    edge_features = graph_cfg.get("edge_features", {}) or {}
    anchor_nodes = graph_cfg.get("anchor_nodes", {}) or {}
    node_features = graph_cfg.get("node_features", {}) or {}
    knn_edges = graph_cfg.get("knn_edges", {}) or {}
    prior_usage = graph_cfg.get("prior_usage")
    prior_corruption = graph_cfg.get("prior_corruption", {}) or {}
    chunk_cache_size = int(data_cfg.get("graph_cache_chunk_cache_size", 2))
    if graph_mode == "hybrid_detected_face_fallback_fullmask":
        detected_ds = _single_dataset(
            prior_dir,
            split,
            "face_plus_context",
            face_threshold,
            int(graph_cfg.get("detected_context_pixels", data_cfg.get("detected_context_pixels", 2))),
            max_samples,
            detail_features=detail_features,
            edge_features=edge_features,
            anchor_nodes=anchor_nodes,
            node_features=node_features,
            knn_edges=knn_edges,
            prior_usage=prior_usage,
            prior_corruption=prior_corruption,
            graph_cache_dir=data_cfg.get("graph_cache_dir_detected"),
            chunk_cache_size=chunk_cache_size,
        )
        fallback_ds = _single_dataset(
            prior_dir,
            split,
            "full_with_mask",
            face_threshold,
            int(graph_cfg.get("fallback_context_pixels", data_cfg.get("fallback_context_pixels", 0))),
            max_samples,
            detail_features=detail_features,
            edge_features=edge_features,
            anchor_nodes=anchor_nodes,
            node_features=node_features,
            knn_edges=knn_edges,
            prior_usage=prior_usage,
            prior_corruption=prior_corruption,
            graph_cache_dir=data_cfg.get("graph_cache_dir_fallback"),
            chunk_cache_size=chunk_cache_size,
        )
        return D16HybridDetectedFallbackDataset(detected_ds, fallback_ds)
    graph_cache_dir = data_cfg.get("graph_cache_dir")
    return _single_dataset(
        prior_dir,
        split,
        graph_mode,
        face_threshold,
        context_pixels,
        max_samples,
        detail_features=detail_features,
        edge_features=edge_features,
        anchor_nodes=anchor_nodes,
        node_features=node_features,
        knn_edges=knn_edges,
        prior_usage=prior_usage,
        prior_corruption=prior_corruption,
        graph_cache_dir=graph_cache_dir,
        chunk_cache_size=chunk_cache_size,
    )


def _build_training_datasets(
    cfg: Dict[str, Any],
    prior_dir: str | Path,
    defer_test_evaluation: bool,
):
    train_ds = build_dataset(cfg, prior_dir, "train")
    val_ds = build_dataset(cfg, prior_dir, "val")
    test_ds = None if bool(defer_test_evaluation) else build_dataset(cfg, prior_dir, "test")
    return train_ds, val_ds, test_ds


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    vals = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2.0 * tp + fp + fn
        vals.append(0.0 if denom <= 0 else (2.0 * tp / denom))
    return float(np.mean(vals))


def _per_class_rows(y_true: np.ndarray, y_pred: np.ndarray, split: str, epoch: int, num_classes: int = 7) -> List[Dict[str, Any]]:
    rows = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        pred_count = int(np.sum(y_pred == cls))
        precision = 0.0 if tp + fp <= 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn <= 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "split": split,
                "epoch": epoch,
                "class_id": cls,
                "support": support,
                "pred_count": pred_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _pred_count_rows(y_pred: np.ndarray, split: str, epoch: int, num_classes: int = 7) -> List[Dict[str, Any]]:
    return [{"split": split, "epoch": epoch, "class_id": cls, "pred_count": int(np.sum(y_pred == cls))} for cls in range(num_classes)]


FER_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _write_confusion_matrix_png(
    path: Path,
    confusion_rows: List[Dict[str, Any]],
    accuracy: float | None,
    split: str,
    class_names: List[str] | None = None,
) -> None:
    if not confusion_rows:
        return
    class_names = list(class_names or FER_CLASS_NAMES)
    num_classes = len(class_names)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    ratios = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in confusion_rows:
        try:
            true_cls = int(row.get("true_class"))
            pred_cls = int(row.get("pred_class"))
        except Exception:
            continue
        if not (0 <= true_cls < num_classes and 0 <= pred_cls < num_classes):
            continue
        try:
            counts[true_cls, pred_cls] = int(float(row.get("count", 0) or 0))
        except Exception:
            counts[true_cls, pred_cls] = 0
        try:
            ratio = float(row.get("row_ratio"))
        except Exception:
            ratio = float("nan")
        ratios[true_cls, pred_cls] = ratio if math.isfinite(ratio) else 0.0

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[D16 confusion] PNG skipped because matplotlib import failed: {exc}", flush=True)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 9.5), dpi=160)
    im = ax.imshow(counts, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=13)

    title_split = str(split).strip() or "test"
    if accuracy is not None and math.isfinite(float(accuracy)):
        title = f"Confusion matrix on {title_split} set, acc: {float(accuracy) * 100:.2f}%"
    else:
        title = f"Confusion matrix on {title_split} set"
    ax.set_title(title, fontsize=24, pad=16)
    ax.set_xlabel("Pred label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, fontsize=15)
    ax.set_yticklabels(class_names, fontsize=15, rotation=90, va="center")

    threshold = float(counts.max()) / 2.0 if counts.size and counts.max() > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if counts[i, j] > threshold else "#2b2b2b"
            ax.text(
                j,
                i,
                f"{int(counts[i, j])}\n{ratios[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=15,
            )

    ax.set_ylim(num_classes - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _confusion_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    epoch: int,
    num_classes: int = 7,
    checkpoint_name: str | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for true_cls in range(num_classes):
        mask = y_true == true_cls
        support = int(mask.sum())
        for pred_cls in range(num_classes):
            count = int(np.sum(mask & (y_pred == pred_cls)))
            rows.append(
                {
                    "split": split,
                    "epoch": int(epoch),
                    "checkpoint_name": checkpoint_name,
                    "true_class": true_cls,
                    "pred_class": pred_cls,
                    "count": count,
                    "support": support,
                    "row_ratio": float(count / support) if support > 0 else float("nan"),
                }
            )
    return rows


def _detected_fallback_per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    detected: np.ndarray,
    split: str,
    epoch: int,
    num_classes: int = 7,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for flag, group in ((True, "detected"), (False, "fallback")):
        mask_group = detected == flag
        for cls in range(num_classes):
            cls_true = y_true == cls
            cls_pred = y_pred == cls
            tp = float(np.sum(mask_group & cls_true & cls_pred))
            fp = float(np.sum(mask_group & ~cls_true & cls_pred))
            fn = float(np.sum(mask_group & cls_true & ~cls_pred))
            support = int(np.sum(mask_group & cls_true))
            pred_count = int(np.sum(mask_group & cls_pred))
            precision = 0.0 if tp + fp <= 0 else tp / (tp + fp)
            recall = 0.0 if tp + fn <= 0 else tp / (tp + fn)
            f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
            rows.append(
                {
                    "split": split,
                    "epoch": int(epoch),
                    "group": group,
                    "class_id": cls,
                    "support": support,
                    "pred_count": pred_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else float("nan"),
        "macro_f1": _macro_f1(y_true, y_pred),
    }



def _label_smoothing(loss_cfg: Dict[str, Any] | None) -> float:
    value = float((loss_cfg or {}).get("label_smoothing", 0.0) or 0.0)
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"loss.label_smoothing must be in [0, 1), got {value}")
    return value


def _should_eval_epoch(epoch: int, start_epoch: int, max_epochs: int, every_n: int) -> bool:
    every_n = max(int(every_n), 1)
    return epoch == start_epoch or epoch == max_epochs or (epoch % every_n == 0)


def _monitor_value(row: Dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    metric = str(metric)
    key = metric[4:] if metric.startswith("val_") else metric
    aliases = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "macro": "macro_f1",
        "macro_f1": "macro_f1",
        "loss": "loss",
    }
    key = aliases.get(key, key)
    if key not in row:
        raise KeyError(f"D16 monitor metric {metric!r} maps to missing validation field {key!r}")
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _is_better_score(current: float | None, best: float, mode: str) -> bool:
    if current is None:
        return False
    if not math.isfinite(best):
        return True
    if str(mode) == "max":
        return current > best
    if str(mode) == "min":
        return current < best
    raise ValueError(f"Unsupported D16 monitor mode={mode!r}; expected max or min")


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> bool:
    if not state:
        return False
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state:
            torch_cpu_state = state["torch_cpu"]
            if not torch.is_tensor(torch_cpu_state):
                torch_cpu_state = torch.tensor(torch_cpu_state, dtype=torch.uint8)
            torch.set_rng_state(torch_cpu_state.detach().cpu().to(dtype=torch.uint8))
        if torch.cuda.is_available() and state.get("torch_cuda"):
            cuda_states = []
            for cuda_state in state["torch_cuda"]:
                if not torch.is_tensor(cuda_state):
                    cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                cuda_states.append(cuda_state.detach().cpu().to(dtype=torch.uint8))
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        print(f"[D16 resume] RNG restore failed: {exc}", flush=True)
        return False
    return True


@torch.no_grad()
def evaluate(
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    epoch: int,
    checkpoint_name: str | None = None,
    checkpoint_epoch: int | None = None,
    best_val_macro_f1: float | None = None,
    collect_predictions: bool = False,
    limit_batches: int | None = None,
    amp_enabled: bool = False,
    loss_cfg: Dict[str, Any] | None = None,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    model.eval()
    label_smoothing = _label_smoothing(loss_cfg)
    y_true, y_pred, detected_flags = [], [], []
    sample_indices, missing_flags = [], []
    prediction_rows: List[Dict[str, Any]] = []
    losses = []
    detected_losses = []
    fallback_losses = []
    detected_head_count = 0
    fallback_head_count = 0
    detected_path_count = 0
    fallback_path_count = 0
    fallback_token_counts = []
    node_counts, edge_counts = [], []
    consistency_losses = []
    epoch_start = time.perf_counter()
    wait_start = epoch_start
    first_batch_wait = None
    batch_wall_times = []
    batch_wait_times = []
    for batch_idx, batch in enumerate(loader, start=1):
        if limit_batches is not None and batch_idx > int(limit_batches):
            break
        batch_ready = time.perf_counter()
        wait_time = batch_ready - wait_start
        batch_wait_times.append(wait_time)
        if first_batch_wait is None:
            first_batch_wait = wait_time
        batch_start = batch_ready
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp_enabled)):
            out = model(batch)
            logits = out["logits"]
            per_sample_loss = F.cross_entropy(logits, batch.y, reduction="none", label_smoothing=label_smoothing)
            loss = per_sample_loss.mean()
        losses.append(float(loss.detach().cpu().item()))
        detected_mask = batch.landmark_missing_flag.long().eq(0)
        fallback_mask = ~detected_mask
        if int(detected_mask.sum().detach().cpu().item()) > 0:
            detected_losses.append(float(per_sample_loss[detected_mask].detach().mean().cpu().item()))
        if int(fallback_mask.sum().detach().cpu().item()) > 0:
            fallback_losses.append(float(per_sample_loss[fallback_mask].detach().mean().cpu().item()))
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        routed_heads = _routed_head_names(out, batch.detected)
        routed_paths = _routed_path_names(out, batch.detected)
        detected_head_count += sum(1 for value in routed_heads if value == "detected_head")
        fallback_head_count += sum(1 for value in routed_heads if value == "fallback_head")
        detected_path_count += sum(1 for value in routed_paths if value == "detected_face_path")
        fallback_path_count += sum(1 for value in routed_paths if value in {"fallback_grid_path", "fallback_transformer_path"})
        if isinstance(out.get("fallback_token_count"), torch.Tensor):
            fallback_token_counts.extend(out["fallback_token_count"].detach().cpu().numpy().astype(float).tolist())
        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()
        y_cpu = batch.y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        sample_cpu = batch.sample_index.detach().cpu()
        detected_cpu = batch.landmark_missing_flag.detach().cpu().long().eq(0)
        missing_cpu = batch.landmark_missing_flag.detach().cpu()
        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        detected_flags.extend(detected_cpu.numpy().astype(bool).tolist())
        sample_indices.extend(sample_cpu.numpy().tolist())
        missing_flags.extend(missing_cpu.numpy().tolist())
        if collect_predictions:
            for i in range(int(y_cpu.numel())):
                item: Dict[str, Any] = {
                    "split": split,
                    "epoch": int(epoch),
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": checkpoint_epoch,
                    "sample_index": int(sample_cpu[i].item()),
                    "y_true": int(y_cpu[i].item()),
                    "y_pred": int(pred_cpu[i].item()),
                    "correct": int(y_cpu[i].item() == pred_cpu[i].item()),
                    "detected": int(bool(detected_cpu[i].item())),
                    "landmark_missing_flag": int(missing_cpu[i].item()),
                    "routed_head": routed_heads[i],
                    "routed_path": routed_paths[i],
                }
                for cls in range(logits_cpu.size(1)):
                    item[f"logit_{cls}"] = float(logits_cpu[i, cls].item())
                    item[f"prob_{cls}"] = float(probs_cpu[i, cls].item())
                prediction_rows.append(item)
        counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()
        node_counts.extend(counts.tolist())
        edge_counts.append(int(batch.edge_index_cat.size(1)) / max(batch.num_graphs, 1))
        batch_end = time.perf_counter()
        batch_wall_times.append(batch_end - batch_start)
        wait_start = batch_end
    total_time = time.perf_counter() - epoch_start
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    detected_np = np.asarray(detected_flags, dtype=bool)
    metric = _metrics(y_true_np, y_pred_np)
    row = {
        "split": split,
        "epoch": int(epoch),
        "checkpoint_name": checkpoint_name,
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": metric["accuracy"],
        "macro_f1": metric["macro_f1"],
        "node_count_mean": float(np.mean(node_counts)) if node_counts else float("nan"),
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "predicted_classes": int(len(set(y_pred))),
        "total": int(len(y_true)),
        "detected_head_count": int(detected_head_count),
        "fallback_head_count": int(fallback_head_count),
        "detected_path_count": int(detected_path_count),
        "fallback_path_count": int(fallback_path_count),
        "fallback_token_count_mean": float(np.mean(fallback_token_counts)) if fallback_token_counts else float("nan"),
        "detected_loss_mean": float(np.mean(detected_losses)) if detected_losses else float("nan"),
        "fallback_loss_mean": float(np.mean(fallback_losses)) if fallback_losses else float("nan"),
        f"{split}_epoch_time_sec": total_time,
        f"{split}_first_batch_wait_time_sec": float(first_batch_wait or 0.0),
        f"{split}_avg_batch_time_ms": float(np.mean(batch_wall_times) * 1000.0) if batch_wall_times else float("nan"),
        f"{split}_avg_batch_wait_time_ms": float(np.mean(batch_wait_times) * 1000.0) if batch_wait_times else float("nan"),
        f"{split}_num_batches": int(len(batch_wall_times)),
    }
    per_class = _per_class_rows(y_true_np, y_pred_np, split, int(epoch))
    pred_count = _pred_count_rows(y_pred_np, split, int(epoch))
    confusion = _confusion_rows(y_true_np, y_pred_np, split, int(epoch), checkpoint_name=checkpoint_name)
    group_per_class = _detected_fallback_per_class_rows(y_true_np, y_pred_np, detected_np, split, int(epoch))
    fallback_rows = []
    for detected_value, name in ((True, "detected"), (False, "fallback")):
        mask = detected_np == detected_value
        if mask.any():
            sub = _metrics(y_true_np[mask], y_pred_np[mask])
            total = int(mask.sum())
        else:
            sub = {"accuracy": float("nan"), "macro_f1": float("nan")}
            total = 0
        fallback_rows.append(
            {
                "split": split,
                "epoch": int(epoch),
                "group": name,
                "total": total,
                "accuracy": sub["accuracy"],
                "macro_f1": sub["macro_f1"],
            }
        )
    return row, per_class, pred_count, fallback_rows, confusion, prediction_rows, group_per_class


def save_checkpoint(
    path: Path,
    model: D16Model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    config: Dict[str, Any],
    global_step: int = 0,
    input_dim: int | None = None,
    best_epoch: int = 0,
    epochs_without_improvement: int = 0,
    resume_source: str | None = None,
    best_monitor_metric: str = "val_macro_f1",
    best_monitor_mode: str = "max",
    best_monitor_score: float | None = None,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    scheduler_type: str = "none",
    best_early_metric: str | None = None,
    best_early_mode: str | None = None,
    best_early_score: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if best_monitor_score is None:
        best_monitor_score = best_val_macro_f1
    payload = {
        "checkpoint_format": "d16_resume_v2",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scheduler_type": str(scheduler_type or "none"),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_metric": float(best_monitor_score),
        "best_monitor_metric": str(best_monitor_metric),
        "best_monitor_mode": str(best_monitor_mode),
        "best_monitor_score": float(best_monitor_score),
        "best_epoch": int(best_epoch),
        "best_checkpoint_path": str(path.parent / "best.pt"),
        "early_stopping_state": {
            "epochs_without_improvement": int(epochs_without_improvement),
            "epochs_since_improvement": int(epochs_without_improvement),
            "best_val_macro_f1": float(best_val_macro_f1),
            "monitor_name": str(best_monitor_metric),
            "monitor_mode": str(best_monitor_mode),
            "min_delta": 0.0,
            "best_monitor_metric": str(best_monitor_metric),
            "best_monitor_mode": str(best_monitor_mode),
            "best_monitor_score": float(best_monitor_score),
            "best_epoch": int(best_epoch),
            "best_early_metric": str(best_early_metric or best_monitor_metric),
            "best_early_mode": str(best_early_mode or best_monitor_mode),
            "best_early_score": float(best_monitor_score if best_early_score is None else best_early_score),
        },
        "rng_state": _rng_state(),
        "config": copy.deepcopy(config),
        "resolved_config": copy.deepcopy(config),
        "run_name": config.get("run_name"),
        "input_dim": None if input_dim is None else int(input_dim),
        "model_signature": _model_signature(config, input_dim),
        "from_scratch": bool(config.get("from_scratch", True)),
        "init_checkpoint": config.get("init_checkpoint"),
        "resume_source": resume_source,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.name == "last.pt" and path.exists():
        prev_path = path.with_name("last_prev.pt")
        try:
            os.replace(path, prev_path)
        except OSError:
            pass
    _atomic_torch_save(payload, path)


def load_checkpoint(path: Path, model: D16Model, device: torch.device, strict: bool = True) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=bool(strict))
    return checkpoint


def resume_training(
    resume_from: Path,
    model: D16Model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: Path,
    restore_rng: bool = True,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    scheduler_type: str = "none",
    current_config: Dict[str, Any] | None = None,
    current_input_dim: int | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(resume_from, model, device, strict=bool(strict))
    compatibility = _check_resume_compatibility(checkpoint, current_config or {}, current_input_dim, strict=bool(strict))
    if checkpoint.get("optimizer_state_dict") is None:
        if strict:
            raise ValueError(f"D16 strict resume checkpoint is missing optimizer_state_dict: {resume_from}")
        optimizer_restored = False
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_restored = True
    checkpoint_scheduler_state = checkpoint.get("scheduler_state_dict")
    checkpoint_scheduler_type = str(checkpoint.get("scheduler_type", "none") or "none")
    requested_scheduler_type = str(scheduler_type or "none")
    if scheduler is not None and checkpoint_scheduler_state is not None:
        scheduler.load_state_dict(checkpoint_scheduler_state)
        scheduler_restored = True
    elif scheduler is None:
        scheduler_restored = checkpoint_scheduler_state is None and checkpoint_scheduler_type == "none"
    else:
        scheduler_restored = False
        if strict and requested_scheduler_type != "none":
            raise ValueError(f"D16 strict resume checkpoint is missing scheduler_state_dict: {resume_from}")
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scaler_restored = True
    else:
        scaler_restored = False
        if strict and scaler is not None and checkpoint.get("scaler_state_dict") is None:
            raise ValueError(f"D16 strict resume checkpoint is missing scaler_state_dict: {resume_from}")
    restored_rng = _restore_rng_state(checkpoint.get("rng_state", {})) if restore_rng else False
    resumed_epoch = int(checkpoint.get("epoch", 0) or 0)
    global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)) or 0)
    best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -math.inf))
    best_monitor_score = float(checkpoint.get("best_monitor_score", best_val_macro_f1))
    best_monitor_metric = str(checkpoint.get("best_monitor_metric", "val_macro_f1"))
    best_monitor_mode = str(checkpoint.get("best_monitor_mode", "max"))
    best_epoch = int(checkpoint.get("best_epoch", 0) or 0)
    if best_epoch <= 0:
        best_epoch = resumed_epoch if math.isfinite(best_monitor_score) else 0
    early_state = checkpoint.get("early_stopping_state", {}) or {}
    epochs_without_improvement = int(early_state.get("epochs_without_improvement", 0) or 0)
    best_early_score = float(early_state.get("best_early_score", best_monitor_score))
    best_early_metric = str(early_state.get("best_early_metric", best_monitor_metric))
    best_early_mode = str(early_state.get("best_early_mode", best_monitor_mode))
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "RESUME_ENABLED": True,
        "resume_from": str(resume_from),
        "resume_path": str(resume_from),
        "resumed_epoch": resumed_epoch,
        "start_epoch": resumed_epoch + 1,
        "global_step": global_step,
        "optimizer_restored": bool(optimizer_restored),
        "scheduler_restored": bool(scheduler_restored),
        "scaler_restored": bool(scaler_restored),
        "early_stop_state_restored": bool(early_state),
        "rng_restored": bool(restored_rng),
        "best_val_macro_f1": best_val_macro_f1,
        "best_monitor_metric": best_monitor_metric,
        "best_monitor_mode": best_monitor_mode,
        "best_metric": best_monitor_score,
        "best_monitor_score": best_monitor_score,
        "best_epoch": best_epoch,
        "scheduler_type": scheduler_type,
        "epochs_without_improvement": epochs_without_improvement,
        "best_early_metric": best_early_metric,
        "best_early_mode": best_early_mode,
        "best_early_score": best_early_score,
        "checkpoint_format": checkpoint.get("checkpoint_format"),
        "resume_strict": bool(strict),
        "config_compatibility": compatibility,
        "warning": None if checkpoint.get("rng_state") else "rng_state_missing_in_checkpoint",
    }
    _append_jsonl(output_dir / "resume_events.jsonl", event)
    _write_json(output_dir / "resume_info.json", event)
    print("[D16 resume] " + json.dumps(event, indent=2, default=str), flush=True)
    return {
        "start_epoch": resumed_epoch + 1,
        "global_step": global_step,
        "best_val_macro_f1": best_val_macro_f1,
        "best_monitor_metric": best_monitor_metric,
        "best_monitor_mode": best_monitor_mode,
        "best_monitor_score": best_monitor_score,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "best_early_metric": best_early_metric,
        "best_early_mode": best_early_mode,
        "best_early_score": best_early_score,
        "resume_source": str(resume_from),
    }


def _supcon_lambda(loss_cfg: Dict[str, Any], epoch: int) -> float:
    target = float(loss_cfg.get("lambda_part_supcon", loss_cfg.get("part_supcon_weight", 0.0)) or 0.0)
    start = int(loss_cfg.get("supcon_start_epoch", 1) or 1)
    ramp = int(loss_cfg.get("supcon_ramp_epochs", 0) or 0)
    if epoch < start or target <= 0.0:
        return 0.0
    if ramp <= 0:
        return target
    progress = min(max((epoch - start + 1) / float(ramp), 0.0), 1.0)
    return float(target * progress)


def _infer_class_weights_from_prior_dir(prior_dir: Path, num_classes: int) -> list[float]:
    train_dir = Path(prior_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Cannot infer class weights; missing prior train dir: {train_dir}")
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for path in sorted(train_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            label = int(np.asarray(data["label"]).item())
        if 0 <= label < int(num_classes):
            counts[label] += 1
    if int(counts.sum()) <= 0 or np.any(counts <= 0):
        raise ValueError(f"Cannot infer class weights from {train_dir}; class counts={counts.tolist()}")
    weights = counts.sum() / (float(num_classes) * counts.astype(np.float64))
    weights = weights / weights.mean()
    return [float(x) for x in weights.tolist()]


def _mean_for_mask(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().detach().cpu().item()) <= 0:
        return float("nan")
    return float(values[mask].detach().mean().cpu().item())


def _routed_head_names(out: Dict[str, Any], detected: torch.Tensor) -> List[str]:
    routed = out.get("routed_head_id")
    if isinstance(routed, torch.Tensor):
        ids = routed.detach().cpu().numpy().astype(int).tolist()
        return ["detected_head" if int(value) == 0 else "fallback_head" for value in ids]
    return ["single_head" for _ in range(int(detected.numel()))]


def _routed_path_names(out: Dict[str, Any], detected: torch.Tensor) -> List[str]:
    routed = out.get("routed_path_id")
    if isinstance(routed, torch.Tensor):
        ids = routed.detach().cpu().numpy().astype(int).tolist()
        fallback_type_id = out.get("fallback_encoder_type_id")
        fallback_name = "fallback_grid_path"
        if isinstance(fallback_type_id, torch.Tensor) and fallback_type_id.numel() > 0:
            fallback_name = "fallback_transformer_path" if int(fallback_type_id.detach().cpu().view(-1)[0].item()) == 2 else "fallback_grid_path"
        return ["detected_face_path" if int(value) == 0 else fallback_name for value in ids]
    return ["single_path" for _ in range(int(detected.numel()))]


def _weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, batch: D16Batch, loss_cfg: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
    weights = torch.ones_like(labels, dtype=logits.dtype)
    class_weights = loss_cfg.get("class_weights")
    if class_weights:
        class_weight_tensor = torch.as_tensor(class_weights, dtype=logits.dtype, device=labels.device)
        if class_weight_tensor.numel() != logits.size(-1):
            raise ValueError(f"class_weights length must match num_classes={logits.size(-1)}, got {class_weight_tensor.numel()}")
        weights = weights * class_weight_tensor[labels]
    fallback_weight = float(loss_cfg.get("fallback_weight", 1.0) or 1.0)
    if bool(loss_cfg.get("fallback_weighted", False)) and fallback_weight != 1.0:
        weights = torch.where(batch.detected.bool(), weights, weights * fallback_weight)
    per_sample = F.cross_entropy(logits, labels, reduction="none", label_smoothing=_label_smoothing(loss_cfg))
    denom = weights.sum().clamp_min(1e-8)
    loss = (per_sample * weights).sum() / denom
    detected_mask = batch.landmark_missing_flag.long().eq(0)
    fallback_mask = ~detected_mask
    return loss, {
        "ce_loss": float(loss.detach().cpu().item()),
        "sample_weight_mean": float(weights.detach().mean().cpu().item()),
        "fallback_weight": fallback_weight,
        "fallback_samples": int(fallback_mask.sum().detach().cpu().item()),
        "detected_head_count": int(detected_mask.sum().detach().cpu().item()),
        "fallback_head_count": int(fallback_mask.sum().detach().cpu().item()),
        "detected_loss_mean": _mean_for_mask(per_sample, detected_mask),
        "fallback_loss_mean": _mean_for_mask(per_sample, fallback_mask),
    }


def attach_hard_proto_loss_if_needed(model: D16Model, loss_cfg: Dict[str, Any], embedding_dim: int) -> HardPrototypeSeparationLoss | None:
    hard_proto_loss = build_hard_proto_separation_loss(loss_cfg, embedding_dim=embedding_dim)
    if hard_proto_loss is None:
        return None
    model.add_module("hard_proto_sep_loss", hard_proto_loss)
    return hard_proto_loss


def attach_pairwise_hard_relation_loss_if_needed(
    model: D16Model,
    loss_cfg: Dict[str, Any],
    embedding_dim: int,
) -> PairwiseHardRelationLoss | None:
    pairwise_loss = build_pairwise_hard_relation_loss(loss_cfg, embedding_dim=embedding_dim)
    if pairwise_loss is None:
        return None
    model.add_module("pairwise_hard_relation_loss", pairwise_loss)
    return pairwise_loss


def train_one_epoch(
    model: D16Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    progress_interval: int = 0,
    loss_cfg: Dict[str, Any] | None = None,
    supcon_loss_fn: PartAwareSupConLoss | None = None,
    hard_proto_loss_fn: HardPrototypeSeparationLoss | None = None,
    pairwise_loss_fn: PairwiseHardRelationLoss | None = None,
    main_logit_pair_margin_loss_fn: MainLogitPairMarginLoss | None = None,
    limit_batches: int | None = None,
    amp_enabled: bool = False,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    scheduler_type: str = "none",
    graph_regularization: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model.train()
    loss_cfg = loss_cfg or {}
    losses = []
    ce_losses = []
    supcon_losses = []
    supcon_part_sums = {name: 0.0 for name in ("mouth", "eye", "brow", "nose_cheek")}
    supcon_valid_pairs = 0.0
    supcon_skipped_parts = 0.0
    supcon_no_positive_parts = 0.0
    hard_proto_losses = []
    hard_proto_ce_losses = []
    hard_proto_margin_losses = []
    hard_proto_sample_counts = []
    hard_proto_pos_sims = []
    hard_proto_neg_sims = []
    pairwise_losses = []
    pairwise_fear_sad_losses = []
    pairwise_sad_neutral_losses = []
    pairwise_fear_sad_counts = []
    pairwise_sad_neutral_counts = []
    pairwise_fear_sad_accs = []
    pairwise_sad_neutral_accs = []
    pair_margin_losses = []
    pair_margin_fear_sad_losses = []
    pair_margin_sad_neutral_losses = []
    pair_margin_neutral_sad_losses = []
    pair_margin_fear_sad_counts = []
    pair_margin_sad_neutral_counts = []
    pair_margin_neutral_sad_counts = []
    pair_margin_fear_sad_violations = []
    pair_margin_sad_neutral_violations = []
    pair_margin_neutral_sad_violations = []
    pair_margin_fear_sad_satisfied = []
    pair_margin_sad_neutral_satisfied = []
    pair_margin_neutral_sad_satisfied = []
    sample_weight_means = []
    fallback_sample_counts = []
    detected_head_counts = []
    fallback_head_counts = []
    detected_path_counts = []
    fallback_path_counts = []
    fallback_token_counts = []
    detected_loss_means = []
    fallback_loss_means = []
    node_counts, edge_counts = [], []
    consistency_losses = []
    epoch_start = time.perf_counter()
    wait_start = epoch_start
    first_batch_wait = None
    batch_wall_times = []
    batch_wait_times = []
    total_batches = len(loader) if limit_batches is None else min(len(loader), int(limit_batches))
    consistency_cfg, consistency_weight = _graph_consistency_cfg(graph_regularization, epoch)
    for batch_idx, batch in enumerate(loader, start=1):
        if limit_batches is not None and batch_idx > int(limit_batches):
            break
        batch_ready = time.perf_counter()
        wait_time = batch_ready - wait_start
        batch_wait_times.append(wait_time)
        if first_batch_wait is None:
            first_batch_wait = wait_time
        batch_start = batch_ready
        batch: D16Batch = batch.to(device)
        base_batch = batch
        consistency_batch = None
        if consistency_weight > 0.0:
            if bool(consistency_cfg.get("clean_anchor", False)):
                batch = base_batch
                consistency_batch = _apply_train_graph_regularization(base_batch, graph_regularization, epoch=epoch)
            else:
                batch = _apply_train_graph_regularization(base_batch, graph_regularization, epoch=epoch)
                consistency_batch = _apply_train_graph_regularization(base_batch, graph_regularization, epoch=epoch)
        else:
            batch = _apply_train_graph_regularization(base_batch, graph_regularization, epoch=epoch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp_enabled)):
            out = model(batch)
            ce_loss, ce_stats = _weighted_ce_loss(out["logits"], batch.y, batch, loss_cfg)
            lambda_supcon = _supcon_lambda(loss_cfg, epoch)
            supcon_loss = batch.y.new_tensor(0.0, dtype=torch.float32)
            supcon_stats: Dict[str, torch.Tensor] = {}
            if lambda_supcon > 0.0 and supcon_loss_fn is not None:
                supcon_stats = supcon_loss_fn(
                    out["part_embeddings"],
                    out["valid_part_groups"],
                    batch.y,
                    detected=batch.detected,
                    skip_fallback=bool(loss_cfg.get("supcon_skip_fallback", True)),
                )
                supcon_loss = supcon_stats["loss_part_supcon"]
            lambda_hard_proto = hard_proto_lambda(loss_cfg, epoch)
            hard_proto_loss = batch.y.new_tensor(0.0, dtype=torch.float32)
            hard_proto_stats: Dict[str, torch.Tensor] = {}
            if lambda_hard_proto > 0.0 and hard_proto_loss_fn is not None:
                if "z_image" not in out:
                    raise KeyError("D16 hard prototype separation requires out['z_image']")
                hard_proto_stats = hard_proto_loss_fn(out["z_image"], batch.y)
                hard_proto_loss = hard_proto_stats["loss_hard_proto_sep"]
            lambda_pair = pairwise_hard_relation_lambda(loss_cfg, epoch)
            pairwise_loss = batch.y.new_tensor(0.0, dtype=torch.float32)
            pairwise_stats: Dict[str, torch.Tensor] = {}
            if lambda_pair > 0.0 and pairwise_loss_fn is not None:
                if "z_image" not in out:
                    raise KeyError("D16 pairwise hard relation requires out['z_image']")
                pairwise_stats = pairwise_loss_fn(out["z_image"], batch.y)
                pairwise_loss = pairwise_stats["loss_pairwise_hard_relation"]
            lambda_pair_margin = main_logit_pair_margin_lambda(loss_cfg, epoch)
            pair_margin_loss = batch.y.new_tensor(0.0, dtype=torch.float32)
            pair_margin_stats: Dict[str, torch.Tensor] = {}
            if lambda_pair_margin > 0.0 and main_logit_pair_margin_loss_fn is not None:
                pair_margin_stats = main_logit_pair_margin_loss_fn(out["logits"], batch.y)
                pair_margin_loss = pair_margin_stats["main_logit_pair_margin_loss"]
            consistency_loss = batch.y.new_tensor(0.0, dtype=torch.float32)
            if consistency_batch is not None:
                out_consistency = model(consistency_batch)
                consistency_loss = _logit_consistency_loss(out["logits"], out_consistency["logits"], consistency_cfg)
            main_ce_weight = float(loss_cfg.get("main_ce_weight", 1.0) or 1.0)
            loss = (
                main_ce_weight * ce_loss
                + float(lambda_supcon) * supcon_loss
                + float(lambda_hard_proto) * hard_proto_loss
                + float(lambda_pair) * pairwise_loss
                + float(lambda_pair_margin) * pair_margin_loss
                + float(consistency_weight) * consistency_loss
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"D16 loss is not finite: {float(loss.detach().cpu().item())}")
        if scaler is not None and bool(amp_enabled):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if scheduler is not None and _scheduler_steps_per_batch(scheduler_type):
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
        ce_losses.append(ce_stats["ce_loss"])
        consistency_losses.append(float(consistency_loss.detach().cpu().item()))
        supcon_losses.append(float(supcon_loss.detach().cpu().item()))
        hard_proto_losses.append(float(hard_proto_loss.detach().cpu().item()))
        if hard_proto_stats:
            hard_proto_ce_losses.append(float(hard_proto_stats["loss_proto_ce"].detach().cpu().item()))
            hard_proto_margin_losses.append(float(hard_proto_stats["loss_proto_margin"].detach().cpu().item()))
            hard_proto_sample_counts.append(float(hard_proto_stats["hard_proto_sample_count"].detach().cpu().item()))
            pos_sim = hard_proto_stats.get("hard_proto_positive_sim_mean")
            neg_sim = hard_proto_stats.get("hard_proto_max_negative_sim_mean")
            if isinstance(pos_sim, torch.Tensor) and torch.isfinite(pos_sim).all():
                hard_proto_pos_sims.append(float(pos_sim.detach().cpu().item()))
            if isinstance(neg_sim, torch.Tensor) and torch.isfinite(neg_sim).all():
                hard_proto_neg_sims.append(float(neg_sim.detach().cpu().item()))
        else:
            hard_proto_ce_losses.append(0.0)
            hard_proto_margin_losses.append(0.0)
            hard_proto_sample_counts.append(0.0)
        pairwise_losses.append(float(pairwise_loss.detach().cpu().item()))
        if pairwise_stats:
            pairwise_fear_sad_losses.append(float(pairwise_stats["loss_pairwise_fear_sad"].detach().cpu().item()))
            pairwise_sad_neutral_losses.append(float(pairwise_stats["loss_pairwise_sad_neutral"].detach().cpu().item()))
            pairwise_fear_sad_counts.append(float(pairwise_stats["pair_count_fear_sad"].detach().cpu().item()))
            pairwise_sad_neutral_counts.append(float(pairwise_stats["pair_count_sad_neutral"].detach().cpu().item()))
            fs_acc = pairwise_stats.get("pair_acc_fear_sad")
            sn_acc = pairwise_stats.get("pair_acc_sad_neutral")
            if isinstance(fs_acc, torch.Tensor) and torch.isfinite(fs_acc).all():
                pairwise_fear_sad_accs.append(float(fs_acc.detach().cpu().item()))
            if isinstance(sn_acc, torch.Tensor) and torch.isfinite(sn_acc).all():
                pairwise_sad_neutral_accs.append(float(sn_acc.detach().cpu().item()))
        else:
            pairwise_fear_sad_losses.append(0.0)
            pairwise_sad_neutral_losses.append(0.0)
            pairwise_fear_sad_counts.append(0.0)
            pairwise_sad_neutral_counts.append(0.0)
        pair_margin_losses.append(float(pair_margin_loss.detach().cpu().item()))
        if pair_margin_stats:
            for key, target in (
                ("fear_sad", (pair_margin_fear_sad_losses, pair_margin_fear_sad_counts, pair_margin_fear_sad_violations, pair_margin_fear_sad_satisfied)),
                ("sad_neutral", (pair_margin_sad_neutral_losses, pair_margin_sad_neutral_counts, pair_margin_sad_neutral_violations, pair_margin_sad_neutral_satisfied)),
                ("neutral_sad", (pair_margin_neutral_sad_losses, pair_margin_neutral_sad_counts, pair_margin_neutral_sad_violations, pair_margin_neutral_sad_satisfied)),
            ):
                losses_list, counts_list, violations_list, satisfied_list = target
                losses_list.append(float(pair_margin_stats[f"pair_margin_loss_{key}"].detach().cpu().item()))
                counts_list.append(float(pair_margin_stats[f"pair_margin_count_{key}"].detach().cpu().item()))
                violation = pair_margin_stats.get(f"mean_margin_violation_{key}")
                satisfied = pair_margin_stats.get(f"pair_margin_satisfied_ratio_{key}")
                if isinstance(violation, torch.Tensor) and torch.isfinite(violation).all():
                    violations_list.append(float(violation.detach().cpu().item()))
                if isinstance(satisfied, torch.Tensor) and torch.isfinite(satisfied).all():
                    satisfied_list.append(float(satisfied.detach().cpu().item()))
        else:
            pair_margin_fear_sad_losses.append(0.0)
            pair_margin_sad_neutral_losses.append(0.0)
            pair_margin_neutral_sad_losses.append(0.0)
            pair_margin_fear_sad_counts.append(0.0)
            pair_margin_sad_neutral_counts.append(0.0)
            pair_margin_neutral_sad_counts.append(0.0)
        sample_weight_means.append(ce_stats["sample_weight_mean"])
        fallback_sample_counts.append(ce_stats["fallback_samples"])
        detected_head_counts.append(ce_stats["detected_head_count"])
        fallback_head_counts.append(ce_stats["fallback_head_count"])
        routed_paths = _routed_path_names(out, batch.detected)
        detected_path_counts.append(sum(1 for value in routed_paths if value == "detected_face_path"))
        fallback_path_counts.append(sum(1 for value in routed_paths if value in {"fallback_grid_path", "fallback_transformer_path"}))
        if isinstance(out.get("fallback_token_count"), torch.Tensor):
            fallback_token_counts.extend(out["fallback_token_count"].detach().cpu().numpy().astype(float).tolist())
        if math.isfinite(float(ce_stats["detected_loss_mean"])):
            detected_loss_means.append(float(ce_stats["detected_loss_mean"]))
        if math.isfinite(float(ce_stats["fallback_loss_mean"])):
            fallback_loss_means.append(float(ce_stats["fallback_loss_mean"]))
        if supcon_stats:
            supcon_valid_pairs += float(supcon_stats["part_supcon_positive_pair_count"].detach().cpu().item())
            supcon_skipped_parts += float(supcon_stats["part_supcon_skipped_parts"].detach().cpu().item())
            supcon_no_positive_parts += float(supcon_stats["part_supcon_no_positive_parts"].detach().cpu().item())
            for name in supcon_part_sums:
                key = f"loss_part_supcon_{name}"
                if key in supcon_stats:
                    supcon_part_sums[name] += float(supcon_stats[key].detach().cpu().item())
        counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()
        node_counts.extend(counts.tolist())
        edge_counts.append(int(batch.edge_index_cat.size(1)) / max(batch.num_graphs, 1))
        batch_end = time.perf_counter()
        batch_wall_times.append(batch_end - batch_start)
        wait_start = batch_end
        if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches):
            elapsed = batch_end - epoch_start
            print(
                json.dumps(
                    {
                        "event": "d16_train_progress",
                        "epoch": int(epoch),
                        "batch": int(batch_idx),
                        "total_batches": int(total_batches),
                        "elapsed_sec": float(elapsed),
                        "last_batch_time_sec": float(batch_wall_times[-1]),
                        "last_batch_wait_sec": float(batch_wait_times[-1]),
                        "avg_loss_so_far": float(np.mean(losses)) if losses else float("nan"),
                        "lambda_part_supcon_current": float(lambda_supcon),
                        "supcon_loss_so_far": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
                        "lambda_hard_proto_current": float(lambda_hard_proto),
                        "hard_proto_loss_so_far": float(np.mean(hard_proto_losses)) if hard_proto_losses else 0.0,
                        "lambda_pair_current": float(lambda_pair),
                        "pairwise_loss_so_far": float(np.mean(pairwise_losses)) if pairwise_losses else 0.0,
                        "lambda_pair_margin_current": float(lambda_pair_margin),
                        "pair_margin_loss_so_far": float(np.mean(pair_margin_losses)) if pair_margin_losses else 0.0,
                        "consistency_loss_so_far": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
                        "lambda_consistency_current": float(consistency_weight),
                    }
                ),
                flush=True,
            )
    total_time = time.perf_counter() - epoch_start
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "node_count_mean": float(np.mean(node_counts)) if node_counts else float("nan"),
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "train_epoch_time_sec": total_time,
        "train_first_batch_wait_time_sec": float(first_batch_wait or 0.0),
        "train_avg_batch_time_ms": float(np.mean(batch_wall_times) * 1000.0) if batch_wall_times else float("nan"),
        "train_avg_batch_wait_time_ms": float(np.mean(batch_wait_times) * 1000.0) if batch_wait_times else float("nan"),
        "train_num_batches": int(len(batch_wall_times)),
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else float("nan"),
        "consistency_loss_total": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
        "lambda_consistency_current": float(consistency_weight),
        "graph_aug_node_dropout_prob": float((graph_regularization or {}).get("node_dropout_prob", 0.0) or 0.0),
        "graph_aug_edge_dropout_prob": float((graph_regularization or {}).get("edge_dropout_prob", 0.0) or 0.0),
        "graph_aug_node_feature_noise_std": float((graph_regularization or {}).get("node_feature_noise_std", 0.0) or 0.0),
        "node_prior_regularization_probability": float(_node_prior_regularization_probability(graph_regularization, epoch)),
        "supcon_loss_total": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
        "supcon_loss_mouth": float(supcon_part_sums["mouth"] / max(len(batch_wall_times), 1)),
        "supcon_loss_eye": float(supcon_part_sums["eye"] / max(len(batch_wall_times), 1)),
        "supcon_loss_brow": float(supcon_part_sums["brow"] / max(len(batch_wall_times), 1)),
        "supcon_loss_nose_cheek": float(supcon_part_sums["nose_cheek"] / max(len(batch_wall_times), 1)),
        "supcon_valid_pairs": float(supcon_valid_pairs),
        "supcon_skipped_parts": float(supcon_skipped_parts),
        "supcon_no_positive_pairs": float(supcon_no_positive_parts),
        "lambda_part_supcon_current": float(_supcon_lambda(loss_cfg, epoch)),
        "hard_proto_loss_total": float(np.mean(hard_proto_losses)) if hard_proto_losses else 0.0,
        "hard_proto_loss_ce": float(np.mean(hard_proto_ce_losses)) if hard_proto_ce_losses else 0.0,
        "hard_proto_loss_margin": float(np.mean(hard_proto_margin_losses)) if hard_proto_margin_losses else 0.0,
        "hard_proto_sample_count_mean": float(np.mean(hard_proto_sample_counts)) if hard_proto_sample_counts else 0.0,
        "hard_proto_positive_sim_mean": float(np.mean(hard_proto_pos_sims)) if hard_proto_pos_sims else float("nan"),
        "hard_proto_max_negative_sim_mean": float(np.mean(hard_proto_neg_sims)) if hard_proto_neg_sims else float("nan"),
        "lambda_hard_proto_current": float(hard_proto_lambda(loss_cfg, epoch)),
        "pairwise_loss_total": float(np.mean(pairwise_losses)) if pairwise_losses else 0.0,
        "pairwise_loss_fear_sad": float(np.mean(pairwise_fear_sad_losses)) if pairwise_fear_sad_losses else 0.0,
        "pairwise_loss_sad_neutral": float(np.mean(pairwise_sad_neutral_losses)) if pairwise_sad_neutral_losses else 0.0,
        "lambda_pair_current": float(pairwise_hard_relation_lambda(loss_cfg, epoch)),
        "pair_count_fear_sad": (
            float(np.mean(pairwise_fear_sad_counts))
            if pairwise_loss_fn is not None and pairwise_fear_sad_counts
            else float(np.mean(pair_margin_fear_sad_counts)) if pair_margin_fear_sad_counts else 0.0
        ),
        "pair_count_sad_neutral": (
            float(np.mean(pairwise_sad_neutral_counts))
            if pairwise_loss_fn is not None and pairwise_sad_neutral_counts
            else float(np.mean(pair_margin_sad_neutral_counts)) if pair_margin_sad_neutral_counts else 0.0
        ),
        "pair_count_neutral_sad": float(np.mean(pair_margin_neutral_sad_counts)) if pair_margin_neutral_sad_counts else 0.0,
        "pair_acc_fear_sad_train": float(np.mean(pairwise_fear_sad_accs)) if pairwise_fear_sad_accs else float("nan"),
        "pair_acc_sad_neutral_train": float(np.mean(pairwise_sad_neutral_accs)) if pairwise_sad_neutral_accs else float("nan"),
        "pair_margin_loss_total": float(np.mean(pair_margin_losses)) if pair_margin_losses else 0.0,
        "pair_margin_loss_fear_sad": float(np.mean(pair_margin_fear_sad_losses)) if pair_margin_fear_sad_losses else 0.0,
        "pair_margin_loss_sad_neutral": float(np.mean(pair_margin_sad_neutral_losses)) if pair_margin_sad_neutral_losses else 0.0,
        "pair_margin_loss_neutral_sad": float(np.mean(pair_margin_neutral_sad_losses)) if pair_margin_neutral_sad_losses else 0.0,
        "lambda_pair_margin_current": float(main_logit_pair_margin_lambda(loss_cfg, epoch)),
        "pair_count_fear_sad_margin": float(np.mean(pair_margin_fear_sad_counts)) if pair_margin_fear_sad_counts else 0.0,
        "pair_count_sad_neutral_margin": float(np.mean(pair_margin_sad_neutral_counts)) if pair_margin_sad_neutral_counts else 0.0,
        "pair_count_neutral_sad_margin": float(np.mean(pair_margin_neutral_sad_counts)) if pair_margin_neutral_sad_counts else 0.0,
        "mean_margin_violation_fear_sad": float(np.mean(pair_margin_fear_sad_violations)) if pair_margin_fear_sad_violations else float("nan"),
        "mean_margin_violation_sad_neutral": float(np.mean(pair_margin_sad_neutral_violations)) if pair_margin_sad_neutral_violations else float("nan"),
        "mean_margin_violation_neutral_sad": float(np.mean(pair_margin_neutral_sad_violations)) if pair_margin_neutral_sad_violations else float("nan"),
        "pair_margin_satisfied_fear_sad": float(np.mean(pair_margin_fear_sad_satisfied)) if pair_margin_fear_sad_satisfied else float("nan"),
        "pair_margin_satisfied_sad_neutral": float(np.mean(pair_margin_sad_neutral_satisfied)) if pair_margin_sad_neutral_satisfied else float("nan"),
        "pair_margin_satisfied_neutral_sad": float(np.mean(pair_margin_neutral_sad_satisfied)) if pair_margin_neutral_sad_satisfied else float("nan"),
        "sample_weight_mean": float(np.mean(sample_weight_means)) if sample_weight_means else 1.0,
        "fallback_samples_seen": int(np.sum(fallback_sample_counts)) if fallback_sample_counts else 0,
        "detected_head_count": int(np.sum(detected_head_counts)) if detected_head_counts else 0,
        "fallback_head_count": int(np.sum(fallback_head_counts)) if fallback_head_counts else 0,
        "detected_path_count": int(np.sum(detected_path_counts)) if detected_path_counts else 0,
        "fallback_path_count": int(np.sum(fallback_path_counts)) if fallback_path_counts else 0,
        "fallback_token_count_mean": float(np.mean(fallback_token_counts)) if fallback_token_counts else float("nan"),
        "detected_loss_mean": float(np.mean(detected_loss_means)) if detected_loss_means else float("nan"),
        "fallback_loss_mean": float(np.mean(fallback_loss_means)) if fallback_loss_means else float("nan"),
    }


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "nan"


def _write_report(output_dir: Path, best_val_macro_f1: float, best_epoch: int, checker_decision: str | None = None) -> None:
    train_log = output_dir / "train_log.csv"
    val_metrics = output_dir / "val_metrics.csv"
    test_metrics = output_dir / "test_metrics.csv"
    last_test_metrics = output_dir / "last_test_metrics.csv"
    test_row = _read_first_csv_row(test_metrics)
    last_row = _read_first_csv_row(last_test_metrics)
    best_macro = test_row.get("macro_f1")
    last_macro = last_row.get("macro_f1")
    best_acc = test_row.get("accuracy")
    last_acc = last_row.get("accuracy")
    diff_macro = _float_or_nan(best_macro) - _float_or_nan(last_macro)
    diff_acc = _float_or_nan(best_acc) - _float_or_nan(last_acc)
    lines = [
        "# D16 v0 Small Train Report",
        "",
        "No full D16 training was launched. This report uses the best validation checkpoint for final test.",
        "",
        "## Best checkpoint test",
        "- final_test_checkpoint: `best.pt`",
        f"- best_val_macro_f1: {best_val_macro_f1:.6f}",
        f"- best_epoch: {best_epoch}",
        f"- test_accuracy: {_format_metric(best_acc)}",
        f"- test_macro_f1: {_format_metric(best_macro)}",
        "",
        "## Last checkpoint test",
        "- checkpoint: `last.pt`",
        f"- test_accuracy: {_format_metric(last_acc)}",
        f"- test_macro_f1: {_format_metric(last_macro)}",
        "",
        "## Difference best minus last",
        f"- accuracy_delta: {_format_metric(diff_acc)}",
        f"- macro_f1_delta: {_format_metric(diff_macro)}",
        "",
        "## Artifacts",
        f"- checker_decision: {checker_decision or 'pending'}",
        f"- train_log: `{train_log}`",
        f"- val_metrics: `{val_metrics}`",
        f"- test_metrics: `{test_metrics}`",
        f"- last_test_metrics: `{last_test_metrics}`",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
    ]
    (output_dir / "D16_V0_SMALL_TRAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "d16_report.md").write_text("\n".join(lines), encoding="utf-8")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_first_csv_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


@torch.no_grad()
def _collect_part_attention_diagnostics(
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint_name: str,
    checkpoint_epoch: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(getattr(model, "readout_type", "concat")) != "part_attention":
        return [], []
    part_names = list(getattr(model, "readout_part_order", []))
    if not part_names:
        return [], []

    model.eval()
    part_count = len(part_names)
    sums = np.zeros(part_count, dtype=np.float64)
    counts = np.zeros(part_count, dtype=np.int64)
    class_sums = np.zeros((7, part_count), dtype=np.float64)
    class_counts = np.zeros((7, part_count), dtype=np.int64)

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        weights = out.get("part_attention_weights") if isinstance(out, dict) else None
        if not isinstance(weights, torch.Tensor):
            return [], []
        if weights.dim() != 2 or int(weights.size(1)) != part_count:
            raise ValueError(f"part_attention_weights shape {tuple(weights.shape)} does not match part_count={part_count}")
        if not torch.isfinite(weights).all().item():
            raise FloatingPointError("part_attention_weights contains NaN or inf")
        w_np = weights.detach().cpu().numpy().astype(np.float64)
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        sums += w_np.sum(axis=0)
        counts += w_np.shape[0]
        for cls in range(7):
            mask = y_np == cls
            if mask.any():
                class_sums[cls] += w_np[mask].sum(axis=0)
                class_counts[cls] += int(mask.sum())

    summary_rows = [
        {
            "split": split,
            "checkpoint_name": checkpoint_name,
            "checkpoint_epoch": int(checkpoint_epoch),
            "part_index": idx,
            "part_name": name,
            "attention_mean": float(sums[idx] / counts[idx]) if counts[idx] > 0 else float("nan"),
            "samples": int(counts[idx]),
        }
        for idx, name in enumerate(part_names)
    ]
    by_class_rows: List[Dict[str, Any]] = []
    for cls in range(7):
        for idx, name in enumerate(part_names):
            by_class_rows.append(
                {
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "class_id": cls,
                    "part_index": idx,
                    "part_name": name,
                    "attention_mean": float(class_sums[cls, idx] / class_counts[cls, idx])
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "samples": int(class_counts[cls, idx]),
                }
            )
    return summary_rows, by_class_rows


@torch.no_grad()
def _collect_part_token_transformer_diagnostics(
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint_name: str,
    checkpoint_epoch: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(getattr(model, "readout_type", "concat")) != "part_token_transformer":
        return [], []
    part_names = list(getattr(model, "readout_part_order", []))
    if not part_names:
        return [], []

    model.eval()
    part_count = len(part_names)
    orig_sums = np.zeros(part_count, dtype=np.float64)
    trans_sums = np.zeros(part_count, dtype=np.float64)
    counts = np.zeros(part_count, dtype=np.int64)
    class_orig_sums = np.zeros((7, part_count), dtype=np.float64)
    class_trans_sums = np.zeros((7, part_count), dtype=np.float64)
    class_counts = np.zeros((7, part_count), dtype=np.int64)

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        original = out.get("part_token_original_tokens") if isinstance(out, dict) else None
        transformed = out.get("part_token_transformed_tokens") if isinstance(out, dict) else None
        valid_mask = out.get("part_token_valid_mask") if isinstance(out, dict) else None
        if not isinstance(original, torch.Tensor) or not isinstance(transformed, torch.Tensor):
            return [], []
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.ones(original.shape[:2], device=original.device, dtype=torch.bool)
        if original.dim() != 3 or transformed.dim() != 3 or int(original.size(1)) != part_count:
            raise ValueError(
                "part-token diagnostics expected original/transformed tokens [B, P, H], "
                f"got original={tuple(original.shape)} transformed={tuple(transformed.shape)} part_count={part_count}"
            )
        if tuple(valid_mask.shape) != tuple(original.shape[:2]):
            raise ValueError(f"part-token valid mask shape {tuple(valid_mask.shape)} does not match {tuple(original.shape[:2])}")
        if not torch.isfinite(original).all().item() or not torch.isfinite(transformed).all().item():
            raise FloatingPointError("part-token diagnostics contain NaN or inf")

        orig_norm = torch.linalg.vector_norm(original, dim=-1).detach().cpu().numpy().astype(np.float64)
        trans_norm = torch.linalg.vector_norm(transformed, dim=-1).detach().cpu().numpy().astype(np.float64)
        valid_np = valid_mask.detach().cpu().numpy().astype(bool)
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        for idx in range(part_count):
            mask = valid_np[:, idx]
            if mask.any():
                orig_sums[idx] += orig_norm[mask, idx].sum()
                trans_sums[idx] += trans_norm[mask, idx].sum()
                counts[idx] += int(mask.sum())
            for cls in range(7):
                class_mask = (y_np == cls) & mask
                if class_mask.any():
                    class_orig_sums[cls, idx] += orig_norm[class_mask, idx].sum()
                    class_trans_sums[cls, idx] += trans_norm[class_mask, idx].sum()
                    class_counts[cls, idx] += int(class_mask.sum())

    summary_rows = [
        {
            "split": split,
            "checkpoint_name": checkpoint_name,
            "checkpoint_epoch": int(checkpoint_epoch),
            "part_index": idx,
            "part_name": name,
            "token_norm_mean": float(orig_sums[idx] / counts[idx]) if counts[idx] > 0 else float("nan"),
            "transformed_token_norm_mean": float(trans_sums[idx] / counts[idx]) if counts[idx] > 0 else float("nan"),
            "valid_samples": int(counts[idx]),
        }
        for idx, name in enumerate(part_names)
    ]
    by_class_rows: List[Dict[str, Any]] = []
    for cls in range(7):
        for idx, name in enumerate(part_names):
            by_class_rows.append(
                {
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "class_id": cls,
                    "part_index": idx,
                    "part_name": name,
                    "token_norm_mean": float(class_orig_sums[cls, idx] / class_counts[cls, idx])
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "transformed_token_norm_mean": float(class_trans_sums[cls, idx] / class_counts[cls, idx])
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "valid_samples": int(class_counts[cls, idx]),
                }
            )
    return summary_rows, by_class_rows


@torch.no_grad()
def _collect_part_motif_query_diagnostics(
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint_name: str,
    checkpoint_epoch: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(getattr(model, "readout_type", "concat")) != "part_motif_query":
        return [], [], []

    model.eval()
    motif_names: List[str] = []
    motif_parts: List[str] = []
    usage_sums = entropy_sums = peak_sums = mass_sums = None
    token_norm_sums = trans_norm_sums = None
    counts = None
    class_usage_sums = class_entropy_sums = class_peak_sums = class_mass_sums = None
    class_counts = None
    sim_sum = None
    sim_count = 0
    effective_values: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        usage = out.get("part_motif_usage") if isinstance(out, dict) else None
        entropy = out.get("part_motif_attention_entropy") if isinstance(out, dict) else None
        peak = out.get("part_motif_attention_peak") if isinstance(out, dict) else None
        mass = out.get("part_motif_part_mass") if isinstance(out, dict) else None
        tokens = out.get("part_motif_tokens") if isinstance(out, dict) else None
        transformed = out.get("part_motif_transformed_tokens") if isinstance(out, dict) else None
        sim = out.get("part_motif_similarity") if isinstance(out, dict) else None
        effective = out.get("part_motif_effective_count") if isinstance(out, dict) else None
        if not all(isinstance(x, torch.Tensor) for x in (usage, entropy, peak, mass, tokens, transformed, sim, effective)):
            return [], [], []
        if usage.dim() != 2 or tokens.dim() != 3 or transformed.dim() != 3:
            raise ValueError(
                f"part motif diagnostics expected usage [B,K], tokens [B,K,H]; got {tuple(usage.shape)}, {tuple(tokens.shape)}"
            )
        if not all(torch.isfinite(x).all().item() for x in (usage, entropy, peak, mass, tokens, transformed, sim, effective)):
            raise FloatingPointError("part motif diagnostics contain NaN or inf")
        if not motif_names:
            motif_names = list(out.get("part_motif_names") or [f"motif_{idx}" for idx in range(int(usage.size(1)))])
            motif_parts = list(out.get("part_motif_parts") or ["unknown" for _ in motif_names])
            k = len(motif_names)
            usage_sums = np.zeros(k, dtype=np.float64)
            entropy_sums = np.zeros(k, dtype=np.float64)
            peak_sums = np.zeros(k, dtype=np.float64)
            mass_sums = np.zeros(k, dtype=np.float64)
            token_norm_sums = np.zeros(k, dtype=np.float64)
            trans_norm_sums = np.zeros(k, dtype=np.float64)
            counts = np.zeros(k, dtype=np.int64)
            class_usage_sums = np.zeros((7, k), dtype=np.float64)
            class_entropy_sums = np.zeros((7, k), dtype=np.float64)
            class_peak_sums = np.zeros((7, k), dtype=np.float64)
            class_mass_sums = np.zeros((7, k), dtype=np.float64)
            class_counts = np.zeros((7, k), dtype=np.int64)
            sim_sum = np.zeros((k, k), dtype=np.float64)
        if int(usage.size(1)) != len(motif_names):
            raise ValueError(f"part motif count changed within eval: {int(usage.size(1))} != {len(motif_names)}")

        usage_np = usage.detach().cpu().numpy().astype(np.float64)
        entropy_np = entropy.detach().cpu().numpy().astype(np.float64)
        peak_np = peak.detach().cpu().numpy().astype(np.float64)
        mass_np = mass.detach().cpu().numpy().astype(np.float64)
        token_norm_np = torch.linalg.vector_norm(tokens, dim=-1).detach().cpu().numpy().astype(np.float64)
        trans_norm_np = torch.linalg.vector_norm(transformed, dim=-1).detach().cpu().numpy().astype(np.float64)
        sim_np = sim.detach().cpu().numpy().astype(np.float64)
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        usage_sums += usage_np.sum(axis=0)
        entropy_sums += entropy_np.sum(axis=0)
        peak_sums += peak_np.sum(axis=0)
        mass_sums += mass_np.sum(axis=0)
        token_norm_sums += token_norm_np.sum(axis=0)
        trans_norm_sums += trans_norm_np.sum(axis=0)
        counts += usage_np.shape[0]
        sim_sum += sim_np.sum(axis=0)
        sim_count += sim_np.shape[0]
        effective_values.extend(effective.detach().cpu().numpy().astype(float).tolist())
        for cls in range(7):
            cls_mask = y_np == cls
            if cls_mask.any():
                class_usage_sums[cls] += usage_np[cls_mask].sum(axis=0)
                class_entropy_sums[cls] += entropy_np[cls_mask].sum(axis=0)
                class_peak_sums[cls] += peak_np[cls_mask].sum(axis=0)
                class_mass_sums[cls] += mass_np[cls_mask].sum(axis=0)
                class_counts[cls] += int(cls_mask.sum())

    if counts is None or sim_sum is None:
        return [], [], []
    effective_mean = float(np.mean(effective_values)) if effective_values else float("nan")
    sim_mean = sim_sum / max(sim_count, 1)
    offdiag = sim_mean[~np.eye(sim_mean.shape[0], dtype=bool)]
    offdiag_mean = float(np.mean(offdiag)) if offdiag.size else float("nan")
    summary_rows = []
    for idx, name in enumerate(motif_names):
        denom = max(int(counts[idx]), 1)
        summary_rows.append(
            {
                "split": split,
                "checkpoint_name": checkpoint_name,
                "checkpoint_epoch": int(checkpoint_epoch),
                "motif_index": idx,
                "motif_name": name,
                "part_name": motif_parts[idx],
                "motif_usage_mean": float(usage_sums[idx] / denom),
                "motif_attention_entropy_mean": float(entropy_sums[idx] / denom),
                "motif_attention_peak_mean": float(peak_sums[idx] / denom),
                "motif_part_mass_mean": float(mass_sums[idx] / denom),
                "motif_token_norm_mean": float(token_norm_sums[idx] / denom),
                "motif_transformed_token_norm_mean": float(trans_norm_sums[idx] / denom),
                "samples": int(counts[idx]),
                "effective_motif_count_mean": effective_mean,
                "avg_offdiag_similarity_mean": offdiag_mean,
            }
        )

    by_class_rows: List[Dict[str, Any]] = []
    for cls in range(7):
        for idx, name in enumerate(motif_names):
            denom = max(int(class_counts[cls, idx]), 1)
            by_class_rows.append(
                {
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "class_id": cls,
                    "motif_index": idx,
                    "motif_name": name,
                    "part_name": motif_parts[idx],
                    "motif_usage_mean": float(class_usage_sums[cls, idx] / denom)
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "motif_attention_entropy_mean": float(class_entropy_sums[cls, idx] / denom)
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "motif_attention_peak_mean": float(class_peak_sums[cls, idx] / denom)
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "motif_part_mass_mean": float(class_mass_sums[cls, idx] / denom)
                    if class_counts[cls, idx] > 0
                    else float("nan"),
                    "samples": int(class_counts[cls, idx]),
                }
            )

    similarity_rows: List[Dict[str, Any]] = []
    for i, name_i in enumerate(motif_names):
        for j, name_j in enumerate(motif_names):
            similarity_rows.append(
                {
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "motif_i": i,
                    "motif_i_name": name_i,
                    "motif_i_part": motif_parts[i],
                    "motif_j": j,
                    "motif_j_name": name_j,
                    "motif_j_part": motif_parts[j],
                    "cosine_mean": float(sim_mean[i, j]),
                    "samples": int(sim_count),
                    "avg_offdiag_similarity_mean": offdiag_mean,
                    "effective_motif_count_mean": effective_mean,
                }
            )
    return summary_rows, by_class_rows, similarity_rows


@torch.no_grad()
def _collect_micro_motif_support_diagnostics(
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint_name: str,
    checkpoint_epoch: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(getattr(model, "readout_type", "concat")) != "micro_motif_support":
        return [], [], []

    model.eval()
    branch_state: Dict[str, Dict[str, Any]] = {}
    gate_sum = 0.0
    gate_count = 0
    detail_available_sum = 0
    detail_available_count = 0

    def _ensure_branch(name: str, motif_names: List[str], motif_parts: List[str]) -> Dict[str, Any]:
        if name in branch_state:
            return branch_state[name]
        k = len(motif_names)
        branch_state[name] = {
            "motif_names": motif_names,
            "motif_parts": motif_parts,
            "usage_sums": np.zeros(k, dtype=np.float64),
            "entropy_sums": np.zeros(k, dtype=np.float64),
            "peak_sums": np.zeros(k, dtype=np.float64),
            "mass_sums": np.zeros(k, dtype=np.float64),
            "detail_sums": np.zeros(k, dtype=np.float64),
            "token_norm_sums": np.zeros(k, dtype=np.float64),
            "trans_norm_sums": np.zeros(k, dtype=np.float64),
            "counts": np.zeros(k, dtype=np.int64),
            "class_usage_sums": np.zeros((7, k), dtype=np.float64),
            "class_entropy_sums": np.zeros((7, k), dtype=np.float64),
            "class_peak_sums": np.zeros((7, k), dtype=np.float64),
            "class_mass_sums": np.zeros((7, k), dtype=np.float64),
            "class_detail_sums": np.zeros((7, k), dtype=np.float64),
            "class_counts": np.zeros((7, k), dtype=np.int64),
            "sim_sum": np.zeros((k, k), dtype=np.float64),
            "sim_count": 0,
            "effective_values": [],
        }
        return branch_state[name]

    def _accumulate(
        state: Dict[str, Any],
        usage: torch.Tensor,
        entropy: torch.Tensor,
        peak: torch.Tensor,
        mass: torch.Tensor,
        detail: torch.Tensor | None,
        tokens: torch.Tensor,
        transformed: torch.Tensor,
        sim: torch.Tensor,
        effective: torch.Tensor,
        y_np: np.ndarray,
    ) -> None:
        if not all(torch.isfinite(x).all().item() for x in (usage, entropy, peak, mass, tokens, transformed, sim, effective)):
            raise FloatingPointError("micro motif support diagnostics contain NaN or inf")
        if detail is not None and not torch.isfinite(detail).all().item():
            raise FloatingPointError("micro motif detail diagnostics contain NaN or inf")
        usage_np = usage.detach().cpu().numpy().astype(np.float64)
        entropy_np = entropy.detach().cpu().numpy().astype(np.float64)
        peak_np = peak.detach().cpu().numpy().astype(np.float64)
        mass_np = mass.detach().cpu().numpy().astype(np.float64)
        detail_np = (
            detail.detach().cpu().numpy().astype(np.float64)
            if isinstance(detail, torch.Tensor)
            else np.full_like(usage_np, np.nan, dtype=np.float64)
        )
        token_norm_np = torch.linalg.vector_norm(tokens, dim=-1).detach().cpu().numpy().astype(np.float64)
        trans_norm_np = torch.linalg.vector_norm(transformed, dim=-1).detach().cpu().numpy().astype(np.float64)
        sim_np = sim.detach().cpu().numpy().astype(np.float64)
        state["usage_sums"] += usage_np.sum(axis=0)
        state["entropy_sums"] += entropy_np.sum(axis=0)
        state["peak_sums"] += peak_np.sum(axis=0)
        state["mass_sums"] += mass_np.sum(axis=0)
        state["detail_sums"] += np.nan_to_num(detail_np, nan=0.0).sum(axis=0)
        state["token_norm_sums"] += token_norm_np.sum(axis=0)
        state["trans_norm_sums"] += trans_norm_np.sum(axis=0)
        state["counts"] += usage_np.shape[0]
        state["sim_sum"] += sim_np.sum(axis=0)
        state["sim_count"] += sim_np.shape[0]
        state["effective_values"].extend(effective.detach().cpu().numpy().astype(float).tolist())
        for cls in range(7):
            cls_mask = y_np == cls
            if cls_mask.any():
                state["class_usage_sums"][cls] += usage_np[cls_mask].sum(axis=0)
                state["class_entropy_sums"][cls] += entropy_np[cls_mask].sum(axis=0)
                state["class_peak_sums"][cls] += peak_np[cls_mask].sum(axis=0)
                state["class_mass_sums"][cls] += mass_np[cls_mask].sum(axis=0)
                state["class_detail_sums"][cls] += np.nan_to_num(detail_np[cls_mask], nan=0.0).sum(axis=0)
                state["class_counts"][cls] += int(cls_mask.sum())

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        gate = out.get("micro_support_gate") if isinstance(out, dict) else None
        if isinstance(gate, torch.Tensor):
            if not torch.isfinite(gate).all().item():
                raise FloatingPointError("micro support gate contains NaN or inf")
            gate_sum += float(gate.detach().mean().cpu().item()) * int(gate.size(0))
            gate_count += int(gate.size(0))
        detail_available = out.get("micro_detail_available") if isinstance(out, dict) else None
        if isinstance(detail_available, torch.Tensor):
            detail_available_sum += int(detail_available.detach().long().sum().cpu().item())
            detail_available_count += int(detail_available.numel())

        branch_specs = [
            (
                "major",
                "micro_major_motif",
                out.get("micro_major_motif_names") or [],
                out.get("micro_major_motif_parts") or [],
                None,
            ),
            (
                "micro",
                "micro_motif",
                out.get("micro_motif_names") or [],
                out.get("micro_motif_parts") or [],
                out.get("micro_motif_detail_score"),
            ),
        ]
        for branch_name, prefix, names, parts, detail in branch_specs:
            usage = out.get(f"{prefix}_usage")
            entropy = out.get(f"{prefix}_attention_entropy")
            peak = out.get(f"{prefix}_attention_peak")
            mass = out.get(f"{prefix}_part_mass")
            tokens = out.get(f"{prefix}_tokens")
            transformed = out.get(f"{prefix}_transformed_tokens")
            sim = out.get(f"{prefix}_similarity")
            effective = out.get(f"{prefix}_effective_count")
            if not all(isinstance(x, torch.Tensor) for x in (usage, entropy, peak, mass, tokens, transformed, sim, effective)):
                return [], [], []
            if usage.dim() != 2 or tokens.dim() != 3 or transformed.dim() != 3:
                raise ValueError(
                    f"{branch_name} micro support diagnostics expected usage [B,K], tokens [B,K,H]; "
                    f"got {tuple(usage.shape)}, {tuple(tokens.shape)}"
                )
            motif_names = list(names or [f"{branch_name}_{idx}" for idx in range(int(usage.size(1)))])
            motif_parts = list(parts or ["unknown" for _ in motif_names])
            state = _ensure_branch(branch_name, motif_names, motif_parts)
            if int(usage.size(1)) != len(state["motif_names"]):
                raise ValueError(f"{branch_name} motif count changed within eval")
            _accumulate(state, usage, entropy, peak, mass, detail, tokens, transformed, sim, effective, y_np)

    if not branch_state:
        return [], [], []

    micro_gate_mean = gate_sum / max(gate_count, 1) if gate_count else float("nan")
    detail_available_ratio = detail_available_sum / max(detail_available_count, 1) if detail_available_count else float("nan")
    summary_rows: List[Dict[str, Any]] = []
    by_class_rows: List[Dict[str, Any]] = []
    similarity_rows: List[Dict[str, Any]] = []
    for branch_name, state in branch_state.items():
        counts = state["counts"]
        sim_count = int(state["sim_count"])
        sim_mean = state["sim_sum"] / max(sim_count, 1)
        offdiag = sim_mean[~np.eye(sim_mean.shape[0], dtype=bool)]
        offdiag_mean = float(np.mean(offdiag)) if offdiag.size else float("nan")
        effective_values = state["effective_values"]
        effective_mean = float(np.mean(effective_values)) if effective_values else float("nan")
        for idx, name in enumerate(state["motif_names"]):
            denom = max(int(counts[idx]), 1)
            summary_rows.append(
                {
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "branch": branch_name,
                    "motif_index": idx,
                    "motif_name": name,
                    "part_name": state["motif_parts"][idx],
                    "motif_usage_mean": float(state["usage_sums"][idx] / denom),
                    "motif_attention_entropy_mean": float(state["entropy_sums"][idx] / denom),
                    "motif_attention_peak_mean": float(state["peak_sums"][idx] / denom),
                    "motif_part_mass_mean": float(state["mass_sums"][idx] / denom),
                    "micro_detail_score_mean": float(state["detail_sums"][idx] / denom)
                    if branch_name == "micro"
                    else float("nan"),
                    "motif_token_norm_mean": float(state["token_norm_sums"][idx] / denom),
                    "motif_transformed_token_norm_mean": float(state["trans_norm_sums"][idx] / denom),
                    "samples": int(counts[idx]),
                    "effective_motif_count_mean": effective_mean,
                    "avg_offdiag_similarity_mean": offdiag_mean,
                    "micro_gate_mean": micro_gate_mean,
                    "detail_available_ratio": detail_available_ratio,
                }
            )
        for cls in range(7):
            for idx, name in enumerate(state["motif_names"]):
                class_counts = state["class_counts"]
                denom = max(int(class_counts[cls, idx]), 1)
                by_class_rows.append(
                    {
                        "split": split,
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_epoch": int(checkpoint_epoch),
                        "class_id": cls,
                        "branch": branch_name,
                        "motif_index": idx,
                        "motif_name": name,
                        "part_name": state["motif_parts"][idx],
                        "motif_usage_mean": float(state["class_usage_sums"][cls, idx] / denom)
                        if class_counts[cls, idx] > 0
                        else float("nan"),
                        "motif_attention_entropy_mean": float(state["class_entropy_sums"][cls, idx] / denom)
                        if class_counts[cls, idx] > 0
                        else float("nan"),
                        "motif_attention_peak_mean": float(state["class_peak_sums"][cls, idx] / denom)
                        if class_counts[cls, idx] > 0
                        else float("nan"),
                        "motif_part_mass_mean": float(state["class_mass_sums"][cls, idx] / denom)
                        if class_counts[cls, idx] > 0
                        else float("nan"),
                        "micro_detail_score_mean": float(state["class_detail_sums"][cls, idx] / denom)
                        if branch_name == "micro" and class_counts[cls, idx] > 0
                        else float("nan"),
                        "micro_gate_mean": micro_gate_mean,
                        "samples": int(class_counts[cls, idx]),
                    }
                )
        for i, name_i in enumerate(state["motif_names"]):
            for j, name_j in enumerate(state["motif_names"]):
                similarity_rows.append(
                    {
                        "split": split,
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_epoch": int(checkpoint_epoch),
                        "branch": branch_name,
                        "motif_i": i,
                        "motif_i_name": name_i,
                        "motif_i_part": state["motif_parts"][i],
                        "motif_j": j,
                        "motif_j_name": name_j,
                        "motif_j_part": state["motif_parts"][j],
                        "cosine_mean": float(sim_mean[i, j]),
                        "samples": int(sim_count),
                        "avg_offdiag_similarity_mean": offdiag_mean,
                        "effective_motif_count_mean": effective_mean,
                    }
                )
    return summary_rows, by_class_rows, similarity_rows


def _checkpoint_epoch(checkpoint: Dict[str, Any], fallback: int) -> int:
    try:
        return int(checkpoint.get("epoch", fallback))
    except Exception:
        return int(fallback)


def _checkpoint_best_val(checkpoint: Dict[str, Any], fallback: float) -> float:
    try:
        return float(checkpoint.get("best_val_macro_f1", fallback))
    except Exception:
        return float(fallback)


def _checkpoint_best_monitor(checkpoint: Dict[str, Any], fallback: float) -> float:
    try:
        return float(checkpoint.get("best_monitor_score", checkpoint.get("best_val_macro_f1", fallback)))
    except Exception:
        return float(fallback)


def _write_eval_outputs(
    output_dir: Path,
    model: D16Model,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint_name: str,
    checkpoint_epoch: int,
    best_val_macro_f1: float,
    metric_fields: List[str],
    per_class_fields: List[str],
    pred_fields: List[str],
    fallback_fields: List[str],
    prefix: str = "",
    loss_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row, per_class, pred_count, fallback, confusion, predictions, group_per_class = evaluate(
        model,
        loader,
        device,
        split,
        checkpoint_epoch,
        checkpoint_name=checkpoint_name,
        checkpoint_epoch=checkpoint_epoch,
        best_val_macro_f1=best_val_macro_f1,
        collect_predictions=True,
        loss_cfg=loss_cfg,
    )
    confusion_fields = ["split", "epoch", "checkpoint_name", "true_class", "pred_class", "count", "support", "row_ratio"]
    group_per_class_fields = ["split", "epoch", "group", "class_id", "support", "pred_count", "precision", "recall", "f1"]
    prediction_fields = [
        "split",
        "epoch",
        "checkpoint_name",
        "checkpoint_epoch",
        "sample_index",
        "y_true",
        "y_pred",
        "correct",
        "detected",
        "landmark_missing_flag",
        "routed_head",
        "routed_path",
    ] + [f"logit_{cls}" for cls in range(7)] + [f"prob_{cls}" for cls in range(7)]
    _write_csv_rows(output_dir / f"{prefix}{split}_metrics.csv", [row], metric_fields)
    _write_csv_rows(output_dir / f"{prefix}per_class_metrics.csv", per_class, per_class_fields)
    _write_csv_rows(output_dir / f"{prefix}pred_count.csv", pred_count, pred_fields)
    _write_csv_rows(output_dir / f"{prefix}detected_vs_fallback_metrics.csv", fallback, fallback_fields)
    _write_csv_rows(output_dir / f"{prefix}detected_fallback_per_class_metrics.csv", group_per_class, group_per_class_fields)
    _write_csv_rows(output_dir / f"{prefix}confusion_matrix.csv", confusion, confusion_fields)
    _write_confusion_matrix_png(output_dir / f"{prefix}confusion_matrix.png", confusion, row.get("accuracy"), split)
    _write_csv_rows(output_dir / f"{prefix}predictions.csv", predictions, prediction_fields)
    try:
        attention_summary, attention_by_class = _collect_part_attention_diagnostics(
            model,
            loader,
            device,
            split,
            checkpoint_name,
            checkpoint_epoch,
        )
        if attention_summary:
            _write_csv_rows(
                output_dir / f"{prefix}part_attention_summary.csv",
                attention_summary,
                ["split", "checkpoint_name", "checkpoint_epoch", "part_index", "part_name", "attention_mean", "samples"],
            )
        if attention_by_class:
            _write_csv_rows(
                output_dir / f"{prefix}part_attention_by_class.csv",
                attention_by_class,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "class_id",
                    "part_index",
                    "part_name",
                    "attention_mean",
                    "samples",
                ],
            )
    except Exception as exc:
        _write_json(output_dir / f"{prefix}part_attention_diagnostics_error.json", {"error": str(exc)})
    try:
        token_summary, token_by_class = _collect_part_token_transformer_diagnostics(
            model,
            loader,
            device,
            split,
            checkpoint_name,
            checkpoint_epoch,
        )
        if token_summary:
            _write_csv_rows(
                output_dir / f"{prefix}part_token_transformer_summary.csv",
                token_summary,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "part_index",
                    "part_name",
                    "token_norm_mean",
                    "transformed_token_norm_mean",
                    "valid_samples",
                ],
            )
        if token_by_class:
            _write_csv_rows(
                output_dir / f"{prefix}part_token_transformer_by_class.csv",
                token_by_class,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "class_id",
                    "part_index",
                    "part_name",
                    "token_norm_mean",
                    "transformed_token_norm_mean",
                    "valid_samples",
                ],
            )
    except Exception as exc:
        _write_json(output_dir / f"{prefix}part_token_transformer_diagnostics_error.json", {"error": str(exc)})
    try:
        motif_summary, motif_by_class, motif_similarity = _collect_part_motif_query_diagnostics(
            model,
            loader,
            device,
            split,
            checkpoint_name,
            checkpoint_epoch,
        )
        if motif_summary:
            _write_csv_rows(
                output_dir / f"{prefix}part_motif_summary.csv",
                motif_summary,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "motif_index",
                    "motif_name",
                    "part_name",
                    "motif_usage_mean",
                    "motif_attention_entropy_mean",
                    "motif_attention_peak_mean",
                    "motif_part_mass_mean",
                    "motif_token_norm_mean",
                    "motif_transformed_token_norm_mean",
                    "samples",
                    "effective_motif_count_mean",
                    "avg_offdiag_similarity_mean",
                ],
            )
        if motif_by_class:
            _write_csv_rows(
                output_dir / f"{prefix}part_motif_by_class.csv",
                motif_by_class,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "class_id",
                    "motif_index",
                    "motif_name",
                    "part_name",
                    "motif_usage_mean",
                    "motif_attention_entropy_mean",
                    "motif_attention_peak_mean",
                    "motif_part_mass_mean",
                    "samples",
                ],
            )
        if motif_similarity:
            _write_csv_rows(
                output_dir / f"{prefix}part_motif_similarity.csv",
                motif_similarity,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "motif_i",
                    "motif_i_name",
                    "motif_i_part",
                    "motif_j",
                    "motif_j_name",
                    "motif_j_part",
                    "cosine_mean",
                    "samples",
                    "avg_offdiag_similarity_mean",
                    "effective_motif_count_mean",
                ],
            )
    except Exception as exc:
        _write_json(output_dir / f"{prefix}part_motif_diagnostics_error.json", {"error": str(exc)})
    try:
        micro_summary, micro_by_class, micro_similarity = _collect_micro_motif_support_diagnostics(
            model,
            loader,
            device,
            split,
            checkpoint_name,
            checkpoint_epoch,
        )
        if micro_summary:
            _write_csv_rows(
                output_dir / f"{prefix}micro_motif_summary.csv",
                micro_summary,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "branch",
                    "motif_index",
                    "motif_name",
                    "part_name",
                    "motif_usage_mean",
                    "motif_attention_entropy_mean",
                    "motif_attention_peak_mean",
                    "motif_part_mass_mean",
                    "micro_detail_score_mean",
                    "motif_token_norm_mean",
                    "motif_transformed_token_norm_mean",
                    "samples",
                    "effective_motif_count_mean",
                    "avg_offdiag_similarity_mean",
                    "micro_gate_mean",
                    "detail_available_ratio",
                ],
            )
        if micro_by_class:
            _write_csv_rows(
                output_dir / f"{prefix}micro_motif_by_class.csv",
                micro_by_class,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "class_id",
                    "branch",
                    "motif_index",
                    "motif_name",
                    "part_name",
                    "motif_usage_mean",
                    "motif_attention_entropy_mean",
                    "motif_attention_peak_mean",
                    "motif_part_mass_mean",
                    "micro_detail_score_mean",
                    "micro_gate_mean",
                    "samples",
                ],
            )
        if micro_similarity:
            _write_csv_rows(
                output_dir / f"{prefix}micro_motif_similarity.csv",
                micro_similarity,
                [
                    "split",
                    "checkpoint_name",
                    "checkpoint_epoch",
                    "branch",
                    "motif_i",
                    "motif_i_name",
                    "motif_i_part",
                    "motif_j",
                    "motif_j_name",
                    "motif_j_part",
                    "cosine_mean",
                    "samples",
                    "avg_offdiag_similarity_mean",
                    "effective_motif_count_mean",
                ],
            )
    except Exception as exc:
        _write_json(output_dir / f"{prefix}micro_motif_diagnostics_error.json", {"error": str(exc)})
    return row


def evaluate_checkpoints(
    cfg: Dict[str, Any],
    prior_dir: Path,
    output_dir: Path,
    device: torch.device,
    checkpoint_path: Path | None = None,
    also_eval_last: bool = True,
    also_eval_best_val_loss: bool = False,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    test_ds = build_dataset(cfg, prior_dir, "test")
    test_loader = DataLoader(test_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))
    first_batch = next(iter(DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=first_batch.x_cat.size(1)).to(device)
    hard_proto_loss_fn = attach_hard_proto_loss_if_needed(
        model,
        cfg.get("loss", {}) or {},
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss_fn is not None:
        hard_proto_loss_fn.to(device)
    pairwise_loss_fn = attach_pairwise_hard_relation_loss_if_needed(
        model,
        cfg.get("loss", {}) or {},
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if pairwise_loss_fn is not None:
        pairwise_loss_fn.to(device)

    best_path = checkpoint_path or (output_dir / "checkpoints" / "best.pt")
    best_checkpoint = load_checkpoint(best_path, model, device)
    best_epoch = _checkpoint_epoch(best_checkpoint, 0)
    best_val_macro_f1 = _checkpoint_best_val(best_checkpoint, float("nan"))
    best_monitor_score = _checkpoint_best_monitor(best_checkpoint, best_val_macro_f1)
    best_monitor_metric = str(best_checkpoint.get("best_monitor_metric", "val_macro_f1"))
    metric_fields = [
        "split",
        "epoch",
        "checkpoint_name",
        "checkpoint_epoch",
        "best_val_macro_f1",
        "loss",
        "accuracy",
        "macro_f1",
        "node_count_mean",
        "edge_count_mean",
        "predicted_classes",
        "total",
        "detected_head_count",
        "fallback_head_count",
        "detected_path_count",
        "fallback_path_count",
        "fallback_token_count_mean",
        "detected_loss_mean",
        "fallback_loss_mean",
    ]
    per_class_fields = ["split", "epoch", "class_id", "support", "pred_count", "precision", "recall", "f1"]
    pred_fields = ["split", "epoch", "class_id", "pred_count"]
    fallback_fields = ["split", "epoch", "group", "total", "accuracy", "macro_f1"]

    best_row = _write_eval_outputs(
        output_dir,
        model,
        test_loader,
        device,
        "test",
        best_path.name,
        best_epoch,
        best_val_macro_f1,
        metric_fields,
        per_class_fields,
        pred_fields,
        fallback_fields,
        loss_cfg=cfg.get("loss", {}) or {},
    )

    best_val_loss_row: Dict[str, Any] | None = None
    best_val_loss_path = output_dir / "checkpoints" / "best_val_loss.pt"
    if also_eval_best_val_loss and best_val_loss_path.exists():
        val_loss_checkpoint = load_checkpoint(best_val_loss_path, model, device)
        val_loss_epoch = _checkpoint_epoch(val_loss_checkpoint, best_epoch)
        val_loss_best_val = _checkpoint_best_val(val_loss_checkpoint, best_val_macro_f1)
        best_val_loss_row = _write_eval_outputs(
            output_dir,
            model,
            test_loader,
            device,
            "test",
            best_val_loss_path.name,
            val_loss_epoch,
            val_loss_best_val,
            metric_fields,
            per_class_fields,
            pred_fields,
            fallback_fields,
            prefix="best_val_loss_",
            loss_cfg=cfg.get("loss", {}) or {},
        )

    last_row: Dict[str, Any] | None = None
    if also_eval_last:
        last_path = output_dir / "checkpoints" / "last.pt"
        if last_path.exists():
            last_checkpoint = load_checkpoint(last_path, model, device)
            last_epoch = _checkpoint_epoch(last_checkpoint, best_epoch)
            last_best_val = _checkpoint_best_val(last_checkpoint, best_val_macro_f1)
            last_row = _write_eval_outputs(
                output_dir,
                model,
                test_loader,
                device,
                "test",
                last_path.name,
                last_epoch,
                last_best_val,
                metric_fields,
                per_class_fields,
                pred_fields,
                fallback_fields,
                prefix="last_",
                loss_cfg=cfg.get("loss", {}) or {},
            )

    summary = {
        "final_test_checkpoint": best_path.name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_monitor_metric": best_monitor_metric,
        "best_monitor_score": best_monitor_score,
        "test_accuracy": best_row["accuracy"],
        "test_macro_f1": best_row["macro_f1"],
        "last_test_accuracy": None if last_row is None else last_row["accuracy"],
        "last_test_macro_f1": None if last_row is None else last_row["macro_f1"],
        "best_val_loss_test_accuracy": None if best_val_loss_row is None else best_val_loss_row["accuracy"],
        "best_val_loss_test_macro_f1": None if best_val_loss_row is None else best_val_loss_row["macro_f1"],
        "test_samples": len(test_ds),
    }
    _write_report(output_dir, best_val_macro_f1, best_epoch)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--graph_cache_dir", default=None)
    parser.add_argument("--graph_cache_dir_detected", default=None)
    parser.add_argument("--graph_cache_dir_fallback", default=None)
    parser.add_argument("--disable_graph_cache", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_epochs_override", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--supcon_start_epoch_override", type=int, default=None)
    parser.add_argument("--supcon_ramp_epochs_override", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--also_eval_last", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--resume_auto", type=_str_bool, nargs="?", const=True, default=False)
    parser.add_argument("--resume_strict", type=_str_bool, nargs="?", const=True, default=True)
    parser.add_argument("--restore_rng", dest="restore_rng", action="store_true")
    parser.add_argument("--no_restore_rng", dest="restore_rng", action="store_false")
    parser.set_defaults(restore_rng=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", cfg.get("training", {}).get("seed"))
    if seed is not None:
        set_seed(int(seed))
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    loss_mode = str((cfg.get("loss", {}) or {}).get("mode", "ce"))
    if loss_mode not in {
        "ce",
        "ce_only",
        "ce_hard_proto_sep",
        "ce_pairwise_hard_relation",
        "ce_main_logit_pair_margin",
        "ce_part_supcon",
        "fallback_weighted_ce",
        "class_weighted_ce",
    }:
        raise ValueError(
            "D16 trainer supports CE, class_weighted_ce, fallback_weighted_ce, "
            f"ce_hard_proto_sep, ce_pairwise_hard_relation, ce_main_logit_pair_margin, "
            f"and ce_part_supcon modes, got loss.mode={loss_mode!r}"
        )
    loss_cfg = cfg.setdefault("loss", {})
    if args.supcon_start_epoch_override is not None:
        loss_cfg["supcon_start_epoch"] = int(args.supcon_start_epoch_override)
    if args.supcon_ramp_epochs_override is not None:
        loss_cfg["supcon_ramp_epochs"] = int(args.supcon_ramp_epochs_override)
    if args.prior_dir:
        data_cfg["prior_dir"] = args.prior_dir
    if args.graph_cache_dir:
        data_cfg["graph_cache_dir"] = args.graph_cache_dir
    if args.graph_cache_dir_detected:
        data_cfg["graph_cache_dir_detected"] = args.graph_cache_dir_detected
    if args.graph_cache_dir_fallback:
        data_cfg["graph_cache_dir_fallback"] = args.graph_cache_dir_fallback
    if args.disable_graph_cache:
        data_cfg["graph_cache_dir"] = None
        data_cfg["graph_cache_dir_detected"] = None
        data_cfg["graph_cache_dir_fallback"] = None
    prior_dir = Path(data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best"))
    output_dir = Path(args.output_dir or Path("outputs/d16_runs") / str(cfg.get("run_name", "d16_v0_small")))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume_from) if args.resume_from else None
    if bool(args.resume_auto) and resume_path is None:
        auto_path = output_dir / "checkpoints" / "last.pt"
        if auto_path.exists():
            resume_path = auto_path
    init_checkpoint = cfg.get("init_checkpoint", training_cfg.get("init_checkpoint"))
    if resume_path is not None and init_checkpoint not in (None, "", "null"):
        raise ValueError("D16 resume_from/resume_auto cannot be combined with init_checkpoint; resume must restore full training state.")
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)
    if args.max_train_samples is not None:
        data_cfg["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        data_cfg["max_val_samples"] = int(args.max_val_samples)
    if args.max_test_samples is not None:
        data_cfg["max_test_samples"] = int(args.max_test_samples)
    if loss_mode == "class_weighted_ce" and bool(loss_cfg.get("class_weights_auto", True)) and not loss_cfg.get("class_weights"):
        num_classes = int(cfg.get("model", {}).get("num_classes", 7))
        loss_cfg["class_weights"] = _infer_class_weights_from_prior_dir(prior_dir, num_classes)
    max_epochs = int(args.max_epochs_override or args.max_epochs or training_cfg.get("max_epochs", 30))
    eval_train_metrics = bool(training_cfg.get("eval_train_metrics", False))
    eval_train_limit_batches_cfg = training_cfg.get("eval_train_limit_batches")
    eval_train_limit_batches = (
        args.limit_train_batches
        if eval_train_limit_batches_cfg is None
        else int(eval_train_limit_batches_cfg)
    )
    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", training_cfg.get("early_stopping", False)))
    monitor_metric = str(
        training_cfg.get(
            "checkpoint_monitor",
            training_cfg.get("monitor_metric", training_cfg.get("metric_for_best_model", early_cfg.get("metric", "val_macro_f1"))),
        )
    )
    early_metric = str(early_cfg.get("metric", monitor_metric))
    early_mode = str(early_cfg.get("mode", "max"))
    monitor_mode = str(training_cfg.get("checkpoint_monitor_mode", training_cfg.get("monitor_mode", "max")))
    checkpoint_policy = _checkpoint_policy_cfg(training_cfg)
    checkpoint_policy_type = str(checkpoint_policy.get("type", "standard"))
    if checkpoint_policy_type != "standard":
        monitor_metric = str(checkpoint_policy.get("primary_metric", monitor_metric))
        monitor_mode = str(checkpoint_policy.get("primary_mode", monitor_mode))
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"D16 monitor mode must be max or min, got {monitor_mode!r}")
    if early_mode not in {"max", "min"}:
        raise ValueError(f"D16 early stopping mode must be max or min, got {early_mode!r}")
    early_patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 999)))
    early_min_epochs = int(early_cfg.get("min_epochs_before_stop", 0))
    eval_every_epoch = bool(training_cfg.get("eval_every_epoch", True))
    eval_every_n_epochs = int(training_cfg.get("eval_every_n_epochs", 1 if eval_every_epoch else max_epochs))
    epochs_without_improvement = 0
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    amp_enabled = _amp_enabled(training_cfg, device)
    if "drop_edge_p" in training_cfg:
        drop_edge_p = float(training_cfg.get("drop_edge_p", 0.0) or 0.0)
        graph_reg = dict(training_cfg.get("graph_regularization", {}) or {})
        graph_reg["edge_dropout_prob"] = drop_edge_p
        graph_reg["enabled"] = bool(graph_reg.get("enabled", False) or drop_edge_p > 0.0)
        training_cfg["graph_regularization"] = graph_reg

    _write_json(output_dir / "resolved_config.json", cfg)
    Path(output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    wandb_run = _init_wandb(cfg, output_dir, resume_path)

    if args.eval_only:
        summary = evaluate_checkpoints(
            cfg,
            prior_dir,
            output_dir,
            device,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            also_eval_last=bool(args.also_eval_last),
        )
        existing_summary = {}
        summary_path = output_dir / "d16_train_summary.json"
        if summary_path.exists():
            try:
                existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                existing_summary = {}
        existing_summary.update(summary)
        existing_summary.update({"output_dir": str(output_dir), "prior_dir": str(prior_dir), "device": str(device)})
        _write_json(summary_path, existing_summary)
        if wandb_run is not None:
            wandb_run.summary.update(existing_summary)
        _wandb_log_final_outputs(wandb_run, output_dir)
        _wandb_finish(wandb_run)
        print(json.dumps(existing_summary, indent=2), flush=True)
        return

    defer_test_evaluation = bool(training_cfg.get("defer_test_evaluation", False))
    train_ds, val_ds, test_ds = _build_training_datasets(cfg, prior_dir, defer_test_evaluation)
    train_loader = DataLoader(train_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=True))
    val_loader = DataLoader(val_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))

    first_batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    input_dim = int(first_batch.x_cat.size(1))
    edge_dim = None if first_batch.edge_attr_cat is None else int(first_batch.edge_attr_cat.size(1))
    feature_schema = {
        "node_dim": input_dim,
        "edge_dim": edge_dim,
        "node_feature_names": list(first_batch.node_feature_names or []),
        "edge_feature_names": list(first_batch.edge_feature_names or []),
        "graph_prior_usage": (cfg.get("graph", {}) or {}).get("prior_usage"),
        "graph_node_features": (cfg.get("graph", {}) or {}).get("node_features", {}) or {},
        "graph_edge_features": (cfg.get("graph", {}) or {}).get("edge_features", {}) or {},
        "anchor_nodes": (cfg.get("graph", {}) or {}).get("anchor_nodes", {}) or {},
        "knn_edges": (cfg.get("graph", {}) or {}).get("knn_edges", {}) or {},
    }
    _write_json(output_dir / "feature_schema.json", feature_schema)
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    loss_cfg = cfg.get("loss", {}) or {}
    hard_proto_loss_fn = attach_hard_proto_loss_if_needed(
        model,
        loss_cfg,
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss_fn is not None:
        hard_proto_loss_fn.to(device)
    pairwise_loss_fn = attach_pairwise_hard_relation_loss_if_needed(
        model,
        loss_cfg,
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if pairwise_loss_fn is not None:
        pairwise_loss_fn.to(device)
    main_logit_pair_margin_loss_fn = build_main_logit_pair_margin_loss(loss_cfg)
    if main_logit_pair_margin_loss_fn is not None:
        main_logit_pair_margin_loss_fn.to(device)
    optimizer = _build_optimizer(model.parameters(), training_cfg)
    optimizer_signature = _resolved_optimizer_signature(training_cfg)
    train_steps_per_epoch = len(train_loader) if args.limit_train_batches is None else min(len(train_loader), int(args.limit_train_batches))
    scheduler, scheduler_type, scheduler_cfg = _build_scheduler(optimizer, training_cfg, max_epochs, train_steps_per_epoch)
    scaler = _make_grad_scaler(bool(amp_enabled))
    supcon_loss_fn = None
    if loss_mode == "ce_part_supcon":
        supcon_loss_fn = PartAwareSupConLoss(temperature=float(loss_cfg.get("supcon_temperature", 0.1))).to(device)

    best_val_macro_f1 = -math.inf
    best_monitor_score = -math.inf if monitor_mode == "max" else math.inf
    best_early_score = -math.inf if early_mode == "max" else math.inf
    best_guard_loss = math.inf
    save_best_val_loss_diagnostic = bool(training_cfg.get("save_best_val_loss_diagnostic", False))
    dual_checkpoint_cfg = _dual_validation_checkpoint_cfg(training_cfg)
    dual_checkpoints_enabled = bool(dual_checkpoint_cfg.get("enabled", False))
    best_val_accuracy_score = -math.inf
    best_val_accuracy_epoch = 0
    best_val_loss_score = math.inf
    best_val_loss_epoch = 0
    best_epoch = 0
    start_epoch = 1
    global_step = 0
    resume_source = None
    train_fields = [
        "epoch",
        "global_step",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_macro_f1",
        "val_accuracy",
        "monitor_metric",
        "monitor_score",
        "best_monitor_score",
        "early_metric",
        "early_score",
        "best_early_score",
        "checkpoint_policy",
        "checkpoint_loss_guard_ok",
        "lr",
        "scheduler_type",
        "node_count_mean",
        "edge_count_mean",
        "train_epoch_time_sec",
        "train_first_batch_wait_time_sec",
        "train_avg_batch_time_ms",
        "train_avg_batch_wait_time_ms",
        "train_num_batches",
        "train_eval_epoch_time_sec",
        "train_eval_first_batch_wait_time_sec",
        "train_eval_avg_batch_time_ms",
        "train_eval_avg_batch_wait_time_ms",
        "train_eval_num_batches",
        "val_epoch_time_sec",
        "val_first_batch_wait_time_sec",
        "val_avg_batch_time_ms",
        "val_avg_batch_wait_time_ms",
        "val_num_batches",
        "epoch_time_sec",
        "evaluated",
        "memory_reserved_mb",
        "ce_loss",
        "consistency_loss_total",
        "lambda_consistency_current",
        "graph_aug_node_dropout_prob",
        "graph_aug_edge_dropout_prob",
        "graph_aug_node_feature_noise_std",
        "node_prior_regularization_probability",
        "prior_corruption_probability",
        "edge_prior_regularization_probability",
        "supcon_loss_total",
        "supcon_loss_mouth",
        "supcon_loss_eye",
        "supcon_loss_brow",
        "supcon_loss_nose_cheek",
        "supcon_valid_pairs",
        "supcon_skipped_parts",
        "supcon_no_positive_pairs",
        "lambda_part_supcon_current",
        "hard_proto_loss_total",
        "hard_proto_loss_ce",
        "hard_proto_loss_margin",
        "hard_proto_sample_count_mean",
        "hard_proto_positive_sim_mean",
        "hard_proto_max_negative_sim_mean",
        "lambda_hard_proto_current",
        "pairwise_loss_total",
        "pairwise_loss_fear_sad",
        "pairwise_loss_sad_neutral",
        "lambda_pair_current",
        "pair_count_fear_sad",
        "pair_count_sad_neutral",
        "pair_count_neutral_sad",
        "pair_acc_fear_sad_train",
        "pair_acc_sad_neutral_train",
        "pair_margin_loss_total",
        "pair_margin_loss_fear_sad",
        "pair_margin_loss_sad_neutral",
        "pair_margin_loss_neutral_sad",
        "lambda_pair_margin_current",
        "pair_count_fear_sad_margin",
        "pair_count_sad_neutral_margin",
        "pair_count_neutral_sad_margin",
        "mean_margin_violation_fear_sad",
        "mean_margin_violation_sad_neutral",
        "mean_margin_violation_neutral_sad",
        "pair_margin_satisfied_fear_sad",
        "pair_margin_satisfied_sad_neutral",
        "pair_margin_satisfied_neutral_sad",
        "sample_weight_mean",
        "fallback_samples_seen",
        "detected_head_count",
        "fallback_head_count",
        "detected_path_count",
        "fallback_path_count",
        "fallback_token_count_mean",
        "detected_loss_mean",
        "fallback_loss_mean",
    ]
    metric_fields = [
        "split",
        "epoch",
        "checkpoint_name",
        "checkpoint_epoch",
        "best_val_macro_f1",
        "loss",
        "accuracy",
        "macro_f1",
        "node_count_mean",
        "edge_count_mean",
        "predicted_classes",
        "total",
        "detected_head_count",
        "fallback_head_count",
        "detected_path_count",
        "fallback_path_count",
        "fallback_token_count_mean",
        "detected_loss_mean",
        "fallback_loss_mean",
        "train_epoch_time_sec",
        "train_first_batch_wait_time_sec",
        "train_avg_batch_time_ms",
        "train_avg_batch_wait_time_ms",
        "train_num_batches",
        "val_epoch_time_sec",
        "val_first_batch_wait_time_sec",
        "val_avg_batch_time_ms",
        "val_avg_batch_wait_time_ms",
        "val_num_batches",
        "test_epoch_time_sec",
        "test_first_batch_wait_time_sec",
        "test_avg_batch_time_ms",
        "test_avg_batch_wait_time_ms",
        "test_num_batches",
    ]
    per_class_fields = ["split", "epoch", "class_id", "support", "pred_count", "precision", "recall", "f1"]
    pred_fields = ["split", "epoch", "class_id", "pred_count"]
    fallback_fields = ["split", "epoch", "group", "total", "accuracy", "macro_f1"]

    if resume_path is not None:
        resume_state = resume_training(
            resume_path,
            model,
            optimizer,
            device,
            output_dir,
            restore_rng=bool(args.restore_rng),
            scaler=scaler,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            current_config=cfg,
            current_input_dim=input_dim,
            strict=bool(args.resume_strict),
        )
        start_epoch = int(resume_state["start_epoch"])
        global_step = int(resume_state.get("global_step", 0) or 0)
        best_val_macro_f1 = float(resume_state["best_val_macro_f1"])
        best_monitor_score = float(resume_state.get("best_monitor_score", best_val_macro_f1))
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        best_early_score = float(resume_state.get("best_early_score", best_early_score))
        resume_source = str(resume_state["resume_source"])

    if start_epoch > max_epochs:
        raise ValueError(f"D16 resume start_epoch={start_epoch} exceeds max_epochs={max_epochs}")

    for epoch in range(start_epoch, max_epochs + 1):
        start = time.time()
        progress_interval = int(training_cfg.get("progress_interval_batches", data_cfg.get("progress_interval_batches", 500)) or 0)
        _set_dataset_epoch(train_ds, epoch)
        prior_corruption_probability = _dataset_prior_corruption_probability(train_ds)
        edge_prior_regularization_probability = _dataset_edge_prior_regularization_probability(train_ds)
        print(json.dumps({"event": "d16_epoch_start", "epoch": int(epoch), "max_epochs": int(max_epochs), "prior_corruption_probability": prior_corruption_probability, "edge_prior_regularization_probability": edge_prior_regularization_probability}), flush=True)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            progress_interval,
            loss_cfg=loss_cfg,
            supcon_loss_fn=supcon_loss_fn,
            hard_proto_loss_fn=hard_proto_loss_fn,
            pairwise_loss_fn=pairwise_loss_fn,
            main_logit_pair_margin_loss_fn=main_logit_pair_margin_loss_fn,
            limit_batches=args.limit_train_batches,
            amp_enabled=amp_enabled,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            graph_regularization=training_cfg.get("graph_regularization", {}) or {},
        )
        global_step += int(train_stats.get("train_num_batches", 0) or 0)
        should_eval = _should_eval_epoch(epoch, start_epoch, max_epochs, eval_every_n_epochs)
        train_eval_row: Dict[str, Any] | None = None
        val_row: Dict[str, Any] | None = None
        val_per_class: List[Dict[str, Any]] = []
        val_pred_count: List[Dict[str, Any]] = []
        val_fallback: List[Dict[str, Any]] = []
        val_confusion: List[Dict[str, Any]] = []
        val_predictions: List[Dict[str, Any]] = []
        if should_eval:
            if eval_train_metrics:
                train_eval_row, _, _, _, _, _, _ = evaluate(
                    model,
                    train_loader,
                    device,
                    "train",
                    epoch,
                    limit_batches=eval_train_limit_batches,
                    amp_enabled=amp_enabled,
                    loss_cfg=loss_cfg,
                )
            val_row, val_per_class, val_pred_count, val_fallback, val_confusion, val_predictions, _ = evaluate(
                model,
                val_loader,
                device,
                "val",
                epoch,
                collect_predictions=bool(
                    dual_checkpoints_enabled
                    and dual_checkpoint_cfg.get("save_validation_predictions", True)
                ),
                limit_batches=args.limit_val_batches,
                amp_enabled=amp_enabled,
                loss_cfg=loss_cfg,
            )
        epoch_time = float(time.time() - start)
        memory_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else float("nan")
        log_row = {
            "epoch": epoch,
            "global_step": int(global_step),
            "train_loss": train_stats["train_loss"],
            "train_eval_loss": None if train_eval_row is None else train_eval_row["loss"],
            "train_accuracy": None if train_eval_row is None else train_eval_row["accuracy"],
            "train_macro_f1": None if train_eval_row is None else train_eval_row["macro_f1"],
            "val_loss": None if val_row is None else val_row["loss"],
            "val_macro_f1": None if val_row is None else val_row["macro_f1"],
            "val_accuracy": None if val_row is None else val_row["accuracy"],
            "monitor_metric": monitor_metric,
            "monitor_score": _monitor_value(val_row, monitor_metric),
            "best_monitor_score": best_monitor_score,
            "early_metric": early_metric,
            "early_score": _monitor_value(val_row, early_metric),
            "best_early_score": best_early_score,
            "checkpoint_policy": checkpoint_policy_type,
            "checkpoint_loss_guard_ok": None,
            "lr": _current_lr(optimizer),
            "scheduler_type": scheduler_type,
            "node_count_mean": train_stats["node_count_mean"],
            "edge_count_mean": train_stats["edge_count_mean"],
            "train_epoch_time_sec": train_stats["train_epoch_time_sec"],
            "train_first_batch_wait_time_sec": train_stats["train_first_batch_wait_time_sec"],
            "train_avg_batch_time_ms": train_stats["train_avg_batch_time_ms"],
            "train_avg_batch_wait_time_ms": train_stats["train_avg_batch_wait_time_ms"],
            "train_num_batches": train_stats["train_num_batches"],
            "train_eval_epoch_time_sec": None if train_eval_row is None else train_eval_row.get("train_epoch_time_sec"),
            "train_eval_first_batch_wait_time_sec": None if train_eval_row is None else train_eval_row.get("train_first_batch_wait_time_sec"),
            "train_eval_avg_batch_time_ms": None if train_eval_row is None else train_eval_row.get("train_avg_batch_time_ms"),
            "train_eval_avg_batch_wait_time_ms": None if train_eval_row is None else train_eval_row.get("train_avg_batch_wait_time_ms"),
            "train_eval_num_batches": None if train_eval_row is None else train_eval_row.get("train_num_batches"),
            "val_epoch_time_sec": None if val_row is None else val_row.get("val_epoch_time_sec"),
            "val_first_batch_wait_time_sec": None if val_row is None else val_row.get("val_first_batch_wait_time_sec"),
            "val_avg_batch_time_ms": None if val_row is None else val_row.get("val_avg_batch_time_ms"),
            "val_avg_batch_wait_time_ms": None if val_row is None else val_row.get("val_avg_batch_wait_time_ms"),
            "val_num_batches": None if val_row is None else val_row.get("val_num_batches"),
            "epoch_time_sec": epoch_time,
            "evaluated": int(should_eval),
            "memory_reserved_mb": memory_reserved,
            "ce_loss": train_stats["ce_loss"],
            "consistency_loss_total": train_stats["consistency_loss_total"],
            "lambda_consistency_current": train_stats["lambda_consistency_current"],
            "graph_aug_node_dropout_prob": train_stats["graph_aug_node_dropout_prob"],
            "graph_aug_edge_dropout_prob": train_stats["graph_aug_edge_dropout_prob"],
            "graph_aug_node_feature_noise_std": train_stats["graph_aug_node_feature_noise_std"],
            "node_prior_regularization_probability": train_stats["node_prior_regularization_probability"],
            "prior_corruption_probability": prior_corruption_probability,
            "edge_prior_regularization_probability": edge_prior_regularization_probability,
            "supcon_loss_total": train_stats["supcon_loss_total"],
            "supcon_loss_mouth": train_stats["supcon_loss_mouth"],
            "supcon_loss_eye": train_stats["supcon_loss_eye"],
            "supcon_loss_brow": train_stats["supcon_loss_brow"],
            "supcon_loss_nose_cheek": train_stats["supcon_loss_nose_cheek"],
            "supcon_valid_pairs": train_stats["supcon_valid_pairs"],
            "supcon_skipped_parts": train_stats["supcon_skipped_parts"],
            "supcon_no_positive_pairs": train_stats["supcon_no_positive_pairs"],
            "lambda_part_supcon_current": train_stats["lambda_part_supcon_current"],
            "hard_proto_loss_total": train_stats["hard_proto_loss_total"],
            "hard_proto_loss_ce": train_stats["hard_proto_loss_ce"],
            "hard_proto_loss_margin": train_stats["hard_proto_loss_margin"],
            "hard_proto_sample_count_mean": train_stats["hard_proto_sample_count_mean"],
            "hard_proto_positive_sim_mean": train_stats["hard_proto_positive_sim_mean"],
            "hard_proto_max_negative_sim_mean": train_stats["hard_proto_max_negative_sim_mean"],
            "lambda_hard_proto_current": train_stats["lambda_hard_proto_current"],
            "pairwise_loss_total": train_stats["pairwise_loss_total"],
            "pairwise_loss_fear_sad": train_stats["pairwise_loss_fear_sad"],
            "pairwise_loss_sad_neutral": train_stats["pairwise_loss_sad_neutral"],
            "lambda_pair_current": train_stats["lambda_pair_current"],
            "pair_count_fear_sad": train_stats["pair_count_fear_sad"],
            "pair_count_sad_neutral": train_stats["pair_count_sad_neutral"],
            "pair_count_neutral_sad": train_stats["pair_count_neutral_sad"],
            "pair_acc_fear_sad_train": train_stats["pair_acc_fear_sad_train"],
            "pair_acc_sad_neutral_train": train_stats["pair_acc_sad_neutral_train"],
            "pair_margin_loss_total": train_stats["pair_margin_loss_total"],
            "pair_margin_loss_fear_sad": train_stats["pair_margin_loss_fear_sad"],
            "pair_margin_loss_sad_neutral": train_stats["pair_margin_loss_sad_neutral"],
            "pair_margin_loss_neutral_sad": train_stats["pair_margin_loss_neutral_sad"],
            "lambda_pair_margin_current": train_stats["lambda_pair_margin_current"],
            "pair_count_fear_sad_margin": train_stats["pair_count_fear_sad_margin"],
            "pair_count_sad_neutral_margin": train_stats["pair_count_sad_neutral_margin"],
            "pair_count_neutral_sad_margin": train_stats["pair_count_neutral_sad_margin"],
            "mean_margin_violation_fear_sad": train_stats["mean_margin_violation_fear_sad"],
            "mean_margin_violation_sad_neutral": train_stats["mean_margin_violation_sad_neutral"],
            "mean_margin_violation_neutral_sad": train_stats["mean_margin_violation_neutral_sad"],
            "pair_margin_satisfied_fear_sad": train_stats["pair_margin_satisfied_fear_sad"],
            "pair_margin_satisfied_sad_neutral": train_stats["pair_margin_satisfied_sad_neutral"],
            "pair_margin_satisfied_neutral_sad": train_stats["pair_margin_satisfied_neutral_sad"],
            "sample_weight_mean": train_stats["sample_weight_mean"],
            "fallback_samples_seen": train_stats["fallback_samples_seen"],
            "detected_head_count": train_stats["detected_head_count"],
            "fallback_head_count": train_stats["fallback_head_count"],
            "detected_path_count": train_stats["detected_path_count"],
            "fallback_path_count": train_stats["fallback_path_count"],
            "fallback_token_count_mean": train_stats["fallback_token_count_mean"],
            "detected_loss_mean": train_stats["detected_loss_mean"],
            "fallback_loss_mean": train_stats["fallback_loss_mean"],
        }
        if train_eval_row is not None:
            _append_csv(output_dir / "train_metrics.csv", train_eval_row, metric_fields)
        if val_row is not None:
            _append_csv(output_dir / "val_metrics.csv", val_row, metric_fields)
            for row in val_per_class:
                _append_csv(output_dir / "per_class_metrics.csv", row, per_class_fields)
            for row in val_pred_count:
                _append_csv(output_dir / "pred_count.csv", row, pred_fields)
            for row in val_fallback:
                _append_csv(output_dir / "detected_vs_fallback_metrics.csv", row, fallback_fields)

        previous_best_monitor_score = float(best_monitor_score)
        previous_best_epoch = int(best_epoch)
        previous_epochs_without_improvement = int(epochs_without_improvement)
        current_val_macro_f1 = None if val_row is None else float(val_row["macro_f1"])
        current_val_loss = None if val_row is None else float(val_row["loss"])
        current_monitor_score = _monitor_value(val_row, monitor_metric)
        current_early_score = _monitor_value(val_row, early_metric)
        if current_val_loss is not None and math.isfinite(float(current_val_loss)):
            best_guard_loss = min(best_guard_loss, float(current_val_loss))
        loss_guard_ok = _loss_guard_ok(val_row, best_guard_loss, checkpoint_policy, epoch=epoch) if checkpoint_policy_type == "hybrid_val_acc_loss_guard" else True
        log_row["checkpoint_loss_guard_ok"] = None if val_row is None else int(bool(loss_guard_ok))
        early_improved = _is_better_score(current_early_score, best_early_score, early_mode)
        if early_improved:
            best_early_score = float(current_early_score)
            epochs_without_improvement = 0
        elif val_row is not None:
            epochs_without_improvement += 1
        if save_best_val_loss_diagnostic and current_val_loss is not None and math.isfinite(current_val_loss) and current_val_loss < best_val_loss_score:
            best_val_loss_score = float(current_val_loss)
            best_val_loss_epoch = epoch
            save_checkpoint(
                output_dir / "checkpoints" / "best_val_loss.pt",
                model,
                optimizer,
                epoch,
                float(val_row["macro_f1"]),
                cfg,
                global_step=global_step,
                input_dim=input_dim,
                best_epoch=best_val_loss_epoch,
                epochs_without_improvement=epochs_without_improvement,
                resume_source=resume_source,
                best_monitor_metric="val_loss",
                best_monitor_mode="min",
                best_monitor_score=best_val_loss_score,
                scaler=scaler,
                scheduler=scheduler,
                scheduler_type=scheduler_type,
                best_early_metric=early_metric,
                best_early_mode=early_mode,
                best_early_score=best_early_score,
            )
        improved = _is_better_score(current_monitor_score, best_monitor_score, monitor_mode) and bool(loss_guard_ok)
        if improved:
            best_monitor_score = float(current_monitor_score)
            best_val_macro_f1 = float(val_row["macro_f1"])
            best_epoch = epoch
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_macro_f1,
                cfg,
                global_step=global_step,
                input_dim=input_dim,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                resume_source=resume_source,
                best_monitor_metric=monitor_metric,
                best_monitor_mode=monitor_mode,
                best_monitor_score=best_monitor_score,
                scaler=scaler,
                scheduler=scheduler,
                scheduler_type=scheduler_type,
                best_early_metric=early_metric,
                best_early_mode=early_mode,
                best_early_score=best_early_score,
            )
            if (
                dual_checkpoints_enabled
                and bool(dual_checkpoint_cfg.get("preserve_best_macro_alias", True))
                and monitor_metric == "val_macro_f1"
            ):
                _atomic_copy_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    output_dir / "checkpoints" / "best_val_macro_f1.pt",
                )
                _write_validation_checkpoint_snapshot(
                    output_dir,
                    "best_val_macro_f1",
                    val_row,
                    val_per_class,
                    val_confusion,
                    val_predictions,
                    enabled=True,
                )
        current_val_accuracy = None if val_row is None else float(val_row["accuracy"])
        accuracy_improved = (
            dual_checkpoints_enabled
            and current_val_accuracy is not None
            and math.isfinite(current_val_accuracy)
            and current_val_accuracy > best_val_accuracy_score
        )
        if accuracy_improved:
            best_val_accuracy_score = float(current_val_accuracy)
            best_val_accuracy_epoch = int(epoch)
            save_checkpoint(
                output_dir / "checkpoints" / "best_val_accuracy.pt",
                model,
                optimizer,
                epoch,
                float(val_row["macro_f1"]),
                cfg,
                global_step=global_step,
                input_dim=input_dim,
                best_epoch=best_val_accuracy_epoch,
                epochs_without_improvement=epochs_without_improvement,
                resume_source=resume_source,
                best_monitor_metric="val_accuracy",
                best_monitor_mode="max",
                best_monitor_score=best_val_accuracy_score,
                scaler=scaler,
                scheduler=scheduler,
                scheduler_type=scheduler_type,
                best_early_metric=early_metric,
                best_early_mode=early_mode,
                best_early_score=best_early_score,
            )
            _write_validation_checkpoint_snapshot(
                output_dir,
                "best_val_accuracy",
                val_row,
                val_per_class,
                val_confusion,
                val_predictions,
                enabled=bool(dual_checkpoint_cfg.get("save_validation_predictions", True)),
            )
        _step_scheduler_epoch(scheduler, scheduler_type, val_row, scheduler_cfg)
        log_row["lr"] = _current_lr(optimizer)
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_macro_f1,
            cfg,
            global_step=global_step,
            input_dim=input_dim,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            resume_source=resume_source,
            best_monitor_metric=monitor_metric,
            best_monitor_mode=monitor_mode,
            best_monitor_score=best_monitor_score,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            best_early_metric=early_metric,
            best_early_mode=early_mode,
            best_early_score=best_early_score,
        )
        if dual_checkpoints_enabled:
            _write_validation_checkpoint_snapshot(
                output_dir,
                "last",
                val_row,
                val_per_class,
                val_confusion,
                val_predictions,
                enabled=bool(dual_checkpoint_cfg.get("save_validation_predictions", True)),
            )
        log_row["best_monitor_score"] = best_monitor_score
        log_row["best_early_score"] = best_early_score
        log_row["early_score"] = current_early_score
        _append_csv(output_dir / "train_log.csv", log_row, train_fields)
        score_status = {
            "event": "d16_epoch_score_status",
            "epoch": int(epoch),
            "evaluated": bool(val_row is not None),
            "metric": monitor_metric,
            "early_metric": early_metric,
            "checkpoint_policy": checkpoint_policy_type,
            "checkpoint_loss_guard_ok": None if val_row is None else bool(loss_guard_ok),
            "is_best": bool(improved),
            "current_score": current_monitor_score,
            "current_val_macro_f1": current_val_macro_f1,
            "current_early_score": current_early_score,
            "best_early_score_current": None if not math.isfinite(float(best_early_score)) else float(best_early_score),
            "display_score": float(best_monitor_score) if improved else current_monitor_score,
            "best_score_before": None if not math.isfinite(previous_best_monitor_score) else previous_best_monitor_score,
            "best_epoch_before": previous_best_epoch,
            "best_score_current": None if not math.isfinite(float(best_monitor_score)) else float(best_monitor_score),
            "best_val_macro_f1_current": None if not math.isfinite(float(best_val_macro_f1)) else float(best_val_macro_f1),
            "best_epoch_current": int(best_epoch),
            "early_stopping_enabled": bool(early_enabled),
            "early_stopping_without_improvement_before": previous_epochs_without_improvement,
            "early_stopping_without_improvement_current": int(epochs_without_improvement),
            "early_stopping_patience": int(early_patience),
            "early_stopping_min_epochs": int(early_min_epochs),
        }
        print(json.dumps(score_status), flush=True)
        _wandb_log_epoch(wandb_run, log_row, score_status)
        print(json.dumps(log_row, indent=2), flush=True)
        if early_enabled:
            if epoch >= early_min_epochs and epochs_without_improvement >= early_patience:
                print(
                    json.dumps(
                        {
                            "early_stopped": True,
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_macro_f1": best_val_macro_f1,
                            "best_monitor_metric": monitor_metric,
                            "best_monitor_score": best_monitor_score,
                            "epochs_without_improvement": epochs_without_improvement,
                            "patience": early_patience,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                break

    eval_summary: Dict[str, Any] = {}
    if not defer_test_evaluation:
        eval_summary = evaluate_checkpoints(
            cfg,
            prior_dir,
            output_dir,
            device,
            also_eval_last=True,
            also_eval_best_val_loss=save_best_val_loss_diagnostic,
        )

    summary = {
        "output_dir": str(output_dir),
        "prior_dir": str(prior_dir),
        "device": str(device),
        "max_epochs": max_epochs,
        "best_val_macro_f1": best_val_macro_f1,
        "best_monitor_metric": monitor_metric,
        "best_monitor_mode": monitor_mode,
        "best_monitor_score": best_monitor_score,
        "best_epoch": best_epoch,
        "early_stopping_metric": early_metric,
        "early_stopping_mode": early_mode,
        "best_early_score": best_early_score,
        "checkpoint_policy": checkpoint_policy_type,
        "optimizer_signature": optimizer_signature,
        "scheduler_type": scheduler_type,
        "scheduler_signature": _resolved_scheduler_signature(training_cfg, max_epochs),
        "global_step": int(global_step),
        "amp": bool(amp_enabled),
        "best_val_loss_score": None if not math.isfinite(best_val_loss_score) else float(best_val_loss_score),
        "best_val_loss_epoch": int(best_val_loss_epoch),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "wandb_enabled": bool(wandb_run is not None),
        "test_evaluation_deferred": bool(defer_test_evaluation),
        "official_test_data_accessed": False if defer_test_evaluation else True,
    }
    if defer_test_evaluation:
        summary["test_samples"] = None
    else:
        summary.update({
            "final_test_checkpoint": eval_summary["final_test_checkpoint"],
            "test_accuracy": eval_summary["test_accuracy"],
            "test_macro_f1": eval_summary["test_macro_f1"],
            "last_test_accuracy": eval_summary["last_test_accuracy"],
            "last_test_macro_f1": eval_summary["last_test_macro_f1"],
            "best_val_loss_test_accuracy": eval_summary.get("best_val_loss_test_accuracy"),
            "best_val_loss_test_macro_f1": eval_summary.get("best_val_loss_test_macro_f1"),
            "test_samples": len(test_ds) if test_ds is not None else None,
        })
    _write_json(output_dir / "d16_train_summary.json", summary)
    if defer_test_evaluation:
        (output_dir / "d16_training_only_report.md").write_text(
            "# D16 Training-Only Report\n\n"
            "Official holdout evaluation was deferred by preregistration. "
            "This run contains training and validation artifacts only.\n",
            encoding="utf-8",
        )
    else:
        _write_report(output_dir, best_val_macro_f1, best_epoch)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    _wandb_log_final_outputs(wandb_run, output_dir)
    _wandb_finish(wandb_run)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
