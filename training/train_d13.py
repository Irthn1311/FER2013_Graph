"""Train D13A pure GNN hierarchical reduction baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, save_config
from data.labels import EMOTION_NAMES
from evaluation.d13_diagnostics import (
    compute_assignment_stats,
    compute_encoder_oversmoothing_score,
    compute_per_class_f1,
    compute_pred_count,
    write_confusion_matrix,
    write_d13_report,
)
from models.d13_hierarchical_reduction_model import D13HierarchicalReductionModel
from training.losses import WeightedCrossEntropy, compute_class_weights
from training.optimizer import build_optimizer, build_scheduler, step_scheduler
from training.trainer import move_to_device, set_seed


def _float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class D13ReductionLoss(torch.nn.Module):
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
        self.ce = WeightedCrossEntropy(class_weights, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        self.pool_entropy_weight = float(cfg.get("pool_entropy_weight", 0.0005))
        self.pool_balance_weight = float(cfg.get("pool_balance_weight", 0.001))
        self.pool_compactness_weight = float(cfg.get("pool_compactness_weight", 0.001))
        self.pool_area_weight = float(cfg.get("pool_area_weight", 0.0005))
        self.supcon_weight = float(cfg.get("supcon_weight", 0.0))
        if self.supcon_weight != 0.0:
            raise ValueError("D13A requires supcon_weight=0.0")

    def forward(self, out: Dict[str, Any], y: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aux = out.get("aux", {})
        logits = out["logits"]
        loss_ce = self.ce(logits, y.long())
        entropy_loss = aux.get("entropy_loss", logits.new_tensor(0.0))
        balance_loss = aux.get("balance_loss", logits.new_tensor(0.0))
        compactness_loss = aux.get("compactness_loss", logits.new_tensor(0.0))
        area_loss = aux.get("area_loss", logits.new_tensor(0.0))
        total = (
            loss_ce
            + self.pool_entropy_weight * entropy_loss
            + self.pool_balance_weight * balance_loss
            + self.pool_compactness_weight * compactness_loss
            + self.pool_area_weight * area_loss
        )
        return {
            "loss": total,
            "loss_ce": loss_ce,
            "loss_pool_entropy": entropy_loss,
            "loss_pool_balance": balance_loss,
            "loss_pool_compactness": compactness_loss,
            "loss_pool_area": area_loss,
        }


def _metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(yt, yp)) if len(yt) else 0.0,
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)) if len(yt) else 0.0,
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)) if len(yt) else 0.0,
    }
    out.update(compute_per_class_f1(yt, yp))
    out.update(compute_pred_count(yp))
    return out


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _pred_count_row(epoch: int, split: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"epoch": int(epoch), "split": str(split)}
    total = 0
    for idx, name in enumerate(EMOTION_NAMES):
        key = f"pred_count_{idx}_{name}"
        value = int(metrics.get(key, 0))
        row[key] = value
        total += value
    row["pred_total"] = int(total)
    row["pred_max_ratio"] = float(max((row[f"pred_count_{idx}_{name}"] for idx, name in enumerate(EMOTION_NAMES)), default=0) / max(total, 1))
    return row


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def build_objects(config: Dict[str, Any], device_arg: str | None = None):
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(device_arg, config)
    model = D13HierarchicalReductionModel.from_config(config.get("model", {})).to(device)
    criterion = D13ReductionLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    return model, criterion, optimizer, scheduler, device


def _set_model_epoch(model: torch.nn.Module, epoch: int) -> None:
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_epoch"):
        target.set_epoch(epoch)


def _run_epoch(
    model: torch.nn.Module,
    criterion: D13ReductionLoss,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    split: str,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
    amp: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    is_train = optimizer is not None
    _set_model_epoch(model, epoch)
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp) and device.type == "cuda")
    totals: Dict[str, float] = {}
    pool_totals: Dict[str, float] = {}
    encoder_totals: Dict[str, float] = {}
    y_true, y_pred = [], []
    count = 0
    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        if "edge_attr" not in batch:
            raise KeyError("D13 batch is missing edge_attr")
        if "x" not in batch or "edge_index" not in batch or "y" not in batch:
            raise KeyError("D13 batch requires x, edge_index, edge_attr, y")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
                out = model(batch)
                loss_dict = criterion(out, batch["y"], batch)
                loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {split} loss at epoch={epoch} batch={batch_idx}")
        logits = out["logits"]
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite {split} logits at epoch={epoch} batch={batch_idx}")
        if is_train:
            scaler.scale(loss).backward()
            if grad_clip is not None and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        pred = logits.detach().argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + _float(value)
        for key, value in compute_assignment_stats(out.get("aux", {})).items():
            pool_totals[key] = pool_totals.get(key, 0.0) + float(value)
        for key, value in compute_encoder_oversmoothing_score(out.get("pixel_layer_embeddings")).items():
            encoder_totals[key] = encoder_totals.get(key, 0.0) + float(value)
        count += 1

    if count == 0:
        raise RuntimeError(f"No batches processed for split={split}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(_metrics(y_true, y_pred))
    metrics["seconds"] = float(time.perf_counter() - start)
    metrics["num_batches"] = int(count)
    metrics["num_samples"] = int(len(y_true))
    pool_stats = {key: value / count for key, value in pool_totals.items()}
    encoder_stats = {key: value / count for key, value in encoder_totals.items()}
    return metrics, pool_stats, encoder_stats


@torch.no_grad()
def run_test_predictions(model: torch.nn.Module, loader, device: torch.device, max_batches: int | None = None):
    model.eval()
    y_true, y_pred, graph_ids = [], [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        pred = out["logits"].argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        if "graph_id" in batch:
            graph_ids.extend(batch["graph_id"].detach().cpu().tolist())
        else:
            graph_ids.extend(list(range(len(y_pred) - pred.numel(), len(y_pred))))
    return y_true, y_pred, graph_ids


def run_train(config: Dict[str, Any], output_dir: str | Path | None = None, device_arg: str | None = None) -> Dict[str, Any]:
    if output_dir is not None:
        config.setdefault("paths", {})["resolved_output_root"] = str(output_dir)
        config.setdefault("paths", {})["output_root"] = str(output_dir)
    output_root = Path(config.get("paths", {}).get("resolved_output_root") or output_dir or "outputs/d13_hierarchical_reduction/run")
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    test_loader = build_dataloader(config, "test", shuffle=False)
    model, criterion, optimizer, scheduler, device = build_objects(config, device_arg=device_arg)
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 50)))
    grad_clip = float(training_cfg.get("grad_clip", training_cfg.get("grad_clip_norm", 1.0)))
    amp = bool(training_cfg.get("amp", False))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")
    patience = int(config.get("early_stopping", {}).get("patience", training_cfg.get("early_stopping_patience", 15)))
    monitor = str(training_cfg.get("early_stopping_metric", training_cfg.get("monitor", "val_macro_f1")))
    best_value = -float("inf")
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics, train_pool, train_enc = _run_epoch(
            model, criterion, train_loader, optimizer, device, epoch, "train", max_train_batches, grad_clip, amp
        )
        val_metrics, val_pool, val_enc = _run_epoch(
            model, criterion, val_loader, None, device, epoch, "val", max_val_batches, grad_clip, amp=False
        )
        if scheduler is not None:
            step_scheduler(scheduler, monitor_value=val_metrics.get("loss"))
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        _append_csv(output_root / "train_log.csv", row)
        _append_csv(output_root / "val_metrics.csv", {"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        _append_csv(output_root / "pooling_stats.csv", {"epoch": epoch, "split": "train", **train_pool})
        _append_csv(output_root / "pooling_stats.csv", {"epoch": epoch, "split": "val", **val_pool})
        _append_csv(output_root / "encoder_diagnostics.csv", {"epoch": epoch, "split": "train", **train_enc})
        _append_csv(output_root / "encoder_diagnostics.csv", {"epoch": epoch, "split": "val", **val_enc})
        _append_csv(output_root / "pred_count.csv", _pred_count_row(epoch, "train", train_metrics))
        _append_csv(output_root / "pred_count.csv", _pred_count_row(epoch, "val", val_metrics))
        history.append(row)
        checkpoint_value = float(row.get(monitor, row.get("val_macro_f1", -float("inf"))))
        if checkpoint_value > best_value:
            best_value = checkpoint_value
            best_epoch = epoch
            stale = 0
            _save_checkpoint(output_root / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, row, config)
        else:
            stale += 1
        _save_checkpoint(output_root / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, row, config)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best={best_value:.4f}@{best_epoch}"
        )
        if stale >= patience:
            print(f"Early stopping after {stale} stale epochs")
            break

    (output_root / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_path = output_root / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_pool, test_enc = _run_epoch(
        model, criterion, test_loader, None, device, best_epoch, "test", max_test_batches, grad_clip, amp=False
    )
    _append_csv(output_root / "test_metrics.csv", {"epoch": best_epoch, **{f"test_{k}": v for k, v in test_metrics.items()}})
    _append_csv(output_root / "pooling_stats.csv", {"epoch": best_epoch, "split": "test", **test_pool})
    _append_csv(output_root / "encoder_diagnostics.csv", {"epoch": best_epoch, "split": "test", **test_enc})
    _append_csv(output_root / "pred_count.csv", _pred_count_row(best_epoch, "test", test_metrics))
    y_true, y_pred, graph_ids = run_test_predictions(model, test_loader, device, max_test_batches)
    pd.DataFrame({"graph_id": graph_ids, "y_true": y_true, "y_pred": y_pred}).to_csv(output_root / "test_predictions.csv", index=False)
    write_confusion_matrix(y_true, y_pred, output_root / "confusion_matrix.csv")
    warnings = []
    if test_pool.get("empty_region_ratio", 0.0) > 0.25:
        warnings.append("High empty-region ratio on test split.")
    if not np.isfinite(test_metrics.get("loss", np.nan)):
        warnings.append("Non-finite final test loss.")
    decision = "D13A_FAIL_TRAINING_UNSTABLE" if warnings and "loss" in warnings[0].lower() else None
    write_d13_report(output_root, config, final_test={f"test_{k}": v for k, v in test_metrics.items()}, decision=decision, warnings=warnings)
    return {"best_epoch": best_epoch, "best_metric": best_value, "output_dir": str(output_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = args.output_dir
    run_train(config, output_dir=args.output_dir, device_arg=args.device)


if __name__ == "__main__":
    main()
