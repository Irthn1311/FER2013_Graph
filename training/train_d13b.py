"""Train D13B diagnostic slot bottleneck models.

D13B is diagnostic only: no SupCon, no prototype learning, and no motif claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, save_config
from data.labels import EMOTION_NAMES
from evaluation.d13_diagnostics import compute_assignment_stats, compute_per_class_f1, compute_pred_count, write_confusion_matrix
from models.d13b_motif_slot_model import D13BMotifSlotModel
from training.losses import WeightedCrossEntropy, compute_class_weights
from training.optimizer import build_optimizer, build_scheduler, step_scheduler
from training.trainer import move_to_device, set_seed


def _float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class D13BDiagnosticLoss(torch.nn.Module):
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
        self.slot_diversity_weight = float(cfg.get("slot_diversity_weight", 0.001))
        self.slot_overlap_weight = float(cfg.get("slot_overlap_weight", 0.001))
        self.slot_entropy_weight = float(cfg.get("slot_entropy_weight", 0.0005))
        self.slot_balance_weight = float(cfg.get("slot_balance_weight", 0.0005))
        self.supcon_weight = float(cfg.get("supcon_weight", 0.0))
        if self.supcon_weight != 0.0:
            raise ValueError("D13B diagnostic requires supcon_weight=0.0")

    def forward(self, out: Dict[str, Any], y: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aux = out.get("aux", {})
        logits = out["logits"]
        ce = self.ce(logits, y.long())
        pool_entropy = aux.get("entropy_loss", logits.new_tensor(0.0))
        pool_balance = aux.get("balance_loss", logits.new_tensor(0.0))
        pool_compact = aux.get("compactness_loss", logits.new_tensor(0.0))
        pool_area = aux.get("area_loss", logits.new_tensor(0.0))
        slot_div = aux.get("slot_diversity_loss", logits.new_tensor(0.0))
        slot_overlap = aux.get("slot_overlap_loss", logits.new_tensor(0.0))
        slot_entropy = aux.get("slot_entropy_loss", logits.new_tensor(0.0))
        slot_balance = aux.get("slot_balance_loss", logits.new_tensor(0.0))
        total = (
            ce
            + self.pool_entropy_weight * pool_entropy
            + self.pool_balance_weight * pool_balance
            + self.pool_compactness_weight * pool_compact
            + self.pool_area_weight * pool_area
            + self.slot_diversity_weight * slot_div
            + self.slot_overlap_weight * slot_overlap
            + self.slot_entropy_weight * slot_entropy
            + self.slot_balance_weight * slot_balance
        )
        return {
            "loss": total,
            "loss_ce": ce,
            "loss_pool_entropy": pool_entropy,
            "loss_pool_balance": pool_balance,
            "loss_pool_compactness": pool_compact,
            "loss_pool_area": pool_area,
            "loss_slot_diversity": slot_div,
            "loss_slot_overlap": slot_overlap,
            "loss_slot_entropy": slot_entropy,
            "loss_slot_balance": slot_balance,
        }


def _metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    out = {
        "accuracy": float(accuracy_score(yt, yp)) if len(yt) else 0.0,
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)) if len(yt) else 0.0,
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)) if len(yt) else 0.0,
    }
    out.update(compute_per_class_f1(yt, yp))
    out.update(compute_pred_count(yp))
    return out


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


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


def _slot_stats(aux: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "slot_entropy",
        "effective_slots",
        "slot_overlap",
        "slot_dominance",
        "slot_area_mean",
        "slot_area_min",
        "slot_area_max",
        "slot_center_std",
    ]
    out = {}
    for key in keys:
        value = aux.get(key)
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().mean().item())
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def _save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
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


def _load_init_d13a(model: torch.nn.Module, checkpoint: str | None, device: torch.device, output_root: Path) -> Dict[str, Any]:
    if not checkpoint:
        return {"enabled": False, "loaded_d13a_keys": 0, "missing_keys": [], "unexpected_keys": [], "init_checkpoint_path": ""}
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"init_d13a_checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    loadable = {}
    skipped = []
    for key, value in source.items():
        if key.startswith("classifier."):
            skipped.append(key)
            continue
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            loadable[key] = value
        else:
            skipped.append(key)
    missing = [key for key in target.keys() if key not in loadable]
    model.load_state_dict(loadable, strict=False)
    info = {
        "enabled": True,
        "init_checkpoint_path": str(path),
        "loaded_d13a_keys": int(len(loadable)),
        "missing_keys": missing[:200],
        "unexpected_keys": skipped[:200],
        "num_missing_keys": int(len(missing)),
        "num_unexpected_or_shape_mismatch_keys": int(len(skipped)),
    }
    (output_root / "init_d13a_load.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"Initialized D13B from D13A checkpoint {path}: "
        f"loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)}"
    )
    return info


def _init_wandb(config: Dict[str, Any], output_root: Path):
    logging_cfg = config.get("logging", {}) or {}
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed or importable.") from exc
    run_cfg = config.get("run", {}) or {}
    run_name = logging_cfg.get("run_name") or f"{run_cfg.get('config_name', 'd13b')}_{output_root.name}"
    wandb.init(
        project=logging_cfg.get("project") or "FER-GRAPH-D13B",
        entity=logging_cfg.get("entity") or None,
        name=run_name,
        config=config,
        dir=str(output_root),
    )
    print(
        f"[W&B] enabled project={logging_cfg.get('project') or 'FER-GRAPH-D13B'} "
        f"entity={logging_cfg.get('entity') or 'default'} run={run_name}"
    )
    return wandb


def _wandb_log(wandb_obj: Any, metrics: Dict[str, Any], epoch: int | None = None) -> None:
    if wandb_obj is None:
        return
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool, str)):
            payload[key] = value
    if epoch is not None:
        payload.setdefault("epoch", int(epoch))
    if payload:
        wandb_obj.log(payload)


def _amp_is_enabled(amp: bool, device: torch.device) -> bool:
    return bool(amp) and device.type == "cuda"


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _set_model_epoch(model: torch.nn.Module, epoch: int) -> None:
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_epoch"):
        target.set_epoch(epoch)


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
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    _set_model_epoch(model, epoch)
    amp_enabled = _amp_is_enabled(amp, device)
    scaler = _make_grad_scaler(amp_enabled)
    totals: Dict[str, float] = {}
    slot_totals: Dict[str, float] = {}
    pool_totals: Dict[str, float] = {}
    y_true, y_pred = [], []
    count = 0
    start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        for key in ("x", "edge_index", "edge_attr", "y"):
            if key not in batch:
                raise KeyError(f"D13B batch requires field {key!r}")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with _autocast(amp_enabled):
                out = model(batch)
                loss_dict = criterion(out, batch["y"], batch)
                loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {split} loss at epoch={epoch} batch={batch_idx}")
        if not torch.isfinite(out["logits"]).all():
            raise FloatingPointError(f"Non-finite {split} logits at epoch={epoch} batch={batch_idx}")
        if not torch.isfinite(out["slot_attention"]).all():
            raise FloatingPointError(f"Non-finite {split} slot attention at epoch={epoch} batch={batch_idx}")
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
        pred = out["logits"].detach().argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + _float(value)
        for key, value in _slot_stats(out.get("aux", {})).items():
            slot_totals[key] = slot_totals.get(key, 0.0) + float(value)
        for key, value in compute_assignment_stats(out.get("aux", {})).items():
            pool_totals[key] = pool_totals.get(key, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError(f"No batches processed for split={split}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(_metrics(y_true, y_pred))
    metrics["seconds"] = float(time.perf_counter() - start)
    metrics["num_batches"] = int(count)
    metrics["num_samples"] = int(len(y_true))
    return metrics, {k: v / count for k, v in slot_totals.items()}, {k: v / count for k, v in pool_totals.items()}


@torch.no_grad()
def _test_predictions(model, loader, device, max_batches: int | None = None):
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


def _write_report(output_root: Path, config: Dict[str, Any], best_epoch: int, best_value: float, test_metrics: Dict[str, float], slot_stats: Dict[str, float], pool_stats: Dict[str, float]) -> None:
    lines = [
        "# D13B Diagnostic Report",
        "",
        "D13B diagnostic only. No SupCon, no prototype learning, no motif claim, and no semantic-region claim.",
        "",
        f"- run_name: `{config.get('run', {}).get('config_name', output_root.name)}`",
        f"- best_epoch: {best_epoch}",
        f"- best_val_macro_f1: {best_value:.6f}",
        f"- test_macro_f1: {test_metrics.get('macro_f1', 0.0):.6f}",
        f"- test_accuracy: {test_metrics.get('accuracy', 0.0):.6f}",
        f"- effective_slots: {slot_stats.get('effective_slots', 0.0):.6f}",
        f"- slot_overlap: {slot_stats.get('slot_overlap', 0.0):.6f}",
        f"- slot_entropy: {slot_stats.get('slot_entropy', 0.0):.6f}",
        f"- slot_dominance: {slot_stats.get('slot_dominance', 0.0):.6f}",
        f"- effective_regions: {pool_stats.get('effective_regions', 0.0):.6f}",
        f"- empty_region_ratio: {pool_stats.get('empty_region_ratio', 0.0):.6f}",
        "",
        "Region nodes and slot nodes are soft bottleneck diagnostics, not semantic regions or motifs.",
        "",
    ]
    (output_root / "d13b_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_objects(config: Dict[str, Any], output_root: Path, device_arg: str | None = None):
    seed = int(config.get("training", {}).get("seed", config.get("run", {}).get("seed", 42)))
    set_seed(seed)
    device = resolve_device(device_arg, config)
    model = D13BMotifSlotModel.from_config(config.get("model", {})).to(device)
    init_info = _load_init_d13a(
        model,
        config.get("model", {}).get("init_d13a_checkpoint") or config.get("training", {}).get("init_d13a_checkpoint"),
        device,
        output_root,
    )
    criterion = D13BDiagnosticLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    return model, criterion, optimizer, scheduler, device, init_info


def run_train(config: Dict[str, Any], output_dir: str | Path | None = None, device_arg: str | None = None) -> Dict[str, Any]:
    if output_dir is not None:
        config.setdefault("paths", {})["resolved_output_root"] = str(output_dir)
        config.setdefault("paths", {})["output_root"] = str(output_dir)
    output_root = Path(config.get("paths", {}).get("resolved_output_root") or output_dir or "outputs/d13b_diagnostic/run")
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    test_loader = build_dataloader(config, "test", shuffle=False)
    model, criterion, optimizer, scheduler, device, init_info = build_objects(config, output_root, device_arg=device_arg)
    wandb_obj = _init_wandb(config, output_root)
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 50)))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    amp = bool(training_cfg.get("amp", False))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")
    patience = int(training_cfg.get("early_stopping_patience", 12))
    monitor = str(training_cfg.get("early_stopping_metric", training_cfg.get("monitor", "val_macro_f1")))
    best_value = -float("inf")
    best_epoch = -1
    stale = 0
    history = []
    try:
        for epoch in range(1, epochs + 1):
            train_metrics, train_slot, train_pool = _run_epoch(model, criterion, train_loader, optimizer, device, epoch, "train", max_train_batches, grad_clip, amp)
            val_metrics, val_slot, val_pool = _run_epoch(model, criterion, val_loader, None, device, epoch, "val", max_val_batches, grad_clip, amp=False)
            if scheduler is not None:
                step_scheduler(scheduler, monitor_value=val_metrics.get("loss"))
            row = {"epoch": epoch, "lr": float(optimizer.param_groups[0].get("lr", 0.0))}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row.update({"loaded_d13a_keys": init_info.get("loaded_d13a_keys", 0)})
            _append_csv(output_root / "train_log.csv", row)
            _append_csv(output_root / "val_metrics.csv", {"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
            _append_csv(output_root / "slot_stats.csv", {"epoch": epoch, "split": "train", **train_slot})
            _append_csv(output_root / "slot_stats.csv", {"epoch": epoch, "split": "val", **val_slot})
            _append_csv(output_root / "pooling_stats.csv", {"epoch": epoch, "split": "train", **train_pool})
            _append_csv(output_root / "pooling_stats.csv", {"epoch": epoch, "split": "val", **val_pool})
            _append_csv(output_root / "pred_count.csv", _pred_count_row(epoch, "train", train_metrics))
            _append_csv(output_root / "pred_count.csv", _pred_count_row(epoch, "val", val_metrics))
            _wandb_log(
                wandb_obj,
                {
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    **{f"slot/train_{k}": v for k, v in train_slot.items()},
                    **{f"slot/val_{k}": v for k, v in val_slot.items()},
                    **{f"pool/train_{k}": v for k, v in train_pool.items()},
                    **{f"pool/val_{k}": v for k, v in val_pool.items()},
                    "lr": row["lr"],
                    "best/val_macro_f1": best_value if best_epoch > 0 else 0.0,
                    "best/epoch": best_epoch,
                },
                epoch=epoch,
            )
            history.append(row)
            value = float(row.get(monitor, row.get("val_macro_f1", -float("inf"))))
            if value > best_value:
                best_value = value
                best_epoch = epoch
                stale = 0
                _save_checkpoint(output_root / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, row, config)
            else:
                stale += 1
            _save_checkpoint(output_root / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, row, config)
            print(
                f"Epoch {epoch:03d}/{epochs:03d} train_loss={train_metrics['loss']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"eff_slots={val_slot.get('effective_slots', 0.0):.2f} "
                f"overlap={val_slot.get('slot_overlap', 0.0):.3f} best={best_value:.4f}@{best_epoch}"
            )
            if stale >= patience:
                print(f"Early stopping after {stale} stale epochs")
                break

        (output_root / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        best_path = output_root / "checkpoints" / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, test_slot, test_pool = _run_epoch(model, criterion, test_loader, None, device, best_epoch, "test", max_test_batches, grad_clip, amp=False)
        _append_csv(output_root / "test_metrics.csv", {"epoch": best_epoch, **{f"test_{k}": v for k, v in test_metrics.items()}})
        _append_csv(output_root / "slot_stats.csv", {"epoch": best_epoch, "split": "test", **test_slot})
        _append_csv(output_root / "pooling_stats.csv", {"epoch": best_epoch, "split": "test", **test_pool})
        _append_csv(output_root / "pred_count.csv", _pred_count_row(best_epoch, "test", test_metrics))
        _wandb_log(
            wandb_obj,
            {
                **{f"test/{k}": v for k, v in test_metrics.items()},
                **{f"slot/test_{k}": v for k, v in test_slot.items()},
                **{f"pool/test_{k}": v for k, v in test_pool.items()},
                "best/val_macro_f1": best_value,
                "best/epoch": best_epoch,
            },
            epoch=best_epoch,
        )
        y_true, y_pred, graph_ids = _test_predictions(model, test_loader, device, max_test_batches)
        pd.DataFrame({"graph_id": graph_ids, "y_true": y_true, "y_pred": y_pred}).to_csv(output_root / "test_predictions.csv", index=False)
        write_confusion_matrix(y_true, y_pred, output_root / "confusion_matrix.csv")
        _write_report(output_root, config, best_epoch, best_value, test_metrics, test_slot, test_pool)
        return {"best_epoch": best_epoch, "best_metric": best_value, "output_dir": str(output_root)}
    finally:
        if wandb_obj is not None:
            wandb_obj.finish()


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
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = args.output_dir
    run_train(config, output_dir=args.output_dir, device_arg=args.device)


if __name__ == "__main__":
    main()
