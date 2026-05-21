"""Train D13C diagnostic image-level SupCon runs.

D13C is diagnostic only. It reuses D13B slot candidates, applies SupCon only to
the pooled image-level slot representation, and makes no motif, semantic-region,
prototype, or causal claim.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, save_config
from evaluation.d13_diagnostics import compute_assignment_stats, write_confusion_matrix
from models.d13c_supcon_model import D13CSupConModel
from training.losses import WeightedCrossEntropy, compute_class_weights
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
    _save_checkpoint,
    _set_model_epoch,
    _slot_stats,
    _test_predictions,
)
from training.trainer import move_to_device, set_seed


D13B_M16_TEST_MACRO_F1 = 0.6187
D13B_M16_TEST_ACC = 0.6328


class D13CDiagnosticLoss(torch.nn.Module):
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
        self.supcon = SupervisedContrastiveLossWithStats(float(cfg.get("supcon_temperature", 0.1)))
        self.lambda_supcon = float(cfg.get("lambda_supcon", cfg.get("supcon_weight", 0.0)))
        self.pool_entropy_weight = float(cfg.get("pool_entropy_weight", 0.0005))
        self.pool_balance_weight = float(cfg.get("pool_balance_weight", 0.001))
        self.pool_compactness_weight = float(cfg.get("pool_compactness_weight", 0.001))
        self.pool_area_weight = float(cfg.get("pool_area_weight", 0.0005))
        self.slot_diversity_weight = float(cfg.get("slot_diversity_weight", 0.001))
        self.slot_overlap_weight = float(cfg.get("slot_overlap_weight", 0.001))
        self.slot_entropy_weight = float(cfg.get("slot_entropy_weight", 0.0005))
        self.slot_balance_weight = float(cfg.get("slot_balance_weight", 0.0005))
        if float(cfg.get("prototype_weight", 0.0)) != 0.0:
            raise ValueError("D13C diagnostic forbids prototype_weight > 0")
        if bool(cfg.get("motif_level_supcon", False)):
            raise ValueError("D13C diagnostic forbids motif_level_supcon=True")

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
        total = (
            ce
            + self.lambda_supcon * supcon_loss
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
            "loss_supcon": supcon_loss,
            "lambda_supcon": logits.new_tensor(self.lambda_supcon),
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


def _load_init_d13b(model: torch.nn.Module, checkpoint: str | None, device: torch.device, output_root: Path) -> Dict[str, Any]:
    if not checkpoint:
        return {"enabled": False, "loaded_d13b_keys": 0, "missing_keys": [], "unexpected_keys": [], "init_checkpoint_path": ""}
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"init_d13b_checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    loadable = {}
    skipped = []
    for key, value in source.items():
        clean_key = key[7:] if str(key).startswith("module.") else key
        if clean_key in target and tuple(target[clean_key].shape) == tuple(value.shape):
            loadable[clean_key] = value
        else:
            skipped.append(key)
    missing = [key for key in target.keys() if key not in loadable]
    model.load_state_dict(loadable, strict=False)
    info = {
        "enabled": True,
        "init_checkpoint_path": str(path),
        "loaded_d13b_keys": int(len(loadable)),
        "missing_keys": missing[:200],
        "unexpected_keys": skipped[:200],
        "num_missing_keys": int(len(missing)),
        "num_unexpected_or_shape_mismatch_keys": int(len(skipped)),
    }
    (output_root / "init_d13b_load.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Initialized D13C from D13B checkpoint {path}: loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)}")
    return info


def _load_model_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)


def _supcon_stats(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    keys = [
        "loss_supcon",
        "lambda_supcon",
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


def _finite_check(out: Dict[str, Any], loss: torch.Tensor, split: str, epoch: int, batch_idx: int) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite {split} loss at epoch={epoch} batch={batch_idx}")
    for key in ("logits", "slot_attention", "z_image", "z_proj"):
        value = out.get(key)
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {split} {key} at epoch={epoch} batch={batch_idx}")


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
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    _set_model_epoch(model, epoch)
    amp_enabled = _amp_is_enabled(amp, device)
    scaler = _make_grad_scaler(amp_enabled)
    totals: Dict[str, float] = {}
    slot_totals: Dict[str, float] = {}
    pool_totals: Dict[str, float] = {}
    supcon_totals: Dict[str, float] = {}
    y_true, y_pred = [], []
    count = 0
    start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
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
    if count == 0:
        raise RuntimeError(f"No batches processed for split={split}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(_metrics(y_true, y_pred))
    metrics["seconds"] = float(time.perf_counter() - start)
    metrics["num_batches"] = int(count)
    metrics["num_samples"] = int(len(y_true))
    return (
        metrics,
        {k: v / count for k, v in slot_totals.items()},
        {k: v / count for k, v in pool_totals.items()},
        {k: v / count for k, v in supcon_totals.items()},
    )


def build_objects(config: Dict[str, Any], output_root: Path, device_arg: str | None = None):
    seed = int(config.get("training", {}).get("seed", config.get("run", {}).get("seed", 42)))
    set_seed(seed)
    device = resolve_device(device_arg, config)
    model = D13CSupConModel.from_config(config.get("model", {})).to(device)
    init_info = _load_init_d13b(
        model,
        config.get("model", {}).get("init_d13b_checkpoint") or config.get("training", {}).get("init_d13b_checkpoint"),
        device,
        output_root,
    )
    if bool(config.get("training", {}).get("multi_gpu", False)) and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Multi-GPU] DataParallel enabled: {torch.cuda.device_count()} GPUs")
    criterion = D13CDiagnosticLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    target = model.module if hasattr(model, "module") else model
    trainable = target.trainable_parameter_count() if hasattr(target, "trainable_parameter_count") else sum(p.numel() for p in target.parameters() if p.requires_grad)
    total = target.total_parameter_count() if hasattr(target, "total_parameter_count") else sum(p.numel() for p in target.parameters())
    init_info["trainable_parameters"] = int(trainable)
    init_info["total_parameters"] = int(total)
    init_info["freeze_backbone"] = bool(config.get("model", {}).get("freeze_backbone", False))
    (output_root / "d13c_model_init.json").write_text(json.dumps(init_info, indent=2), encoding="utf-8")
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
        "# D13C Diagnostic Report",
        "",
        "D13C diagnostic only. No CNN teacher, no prototype, no motif-level SupCon, no full D13C, no SupCon full, no motif claim, no semantic-region claim, and no causal-evidence claim.",
        "",
        f"- run_name: `{config.get('run', {}).get('config_name', output_root.name)}`",
        f"- base_model: `{config.get('model', {}).get('base_model', '')}`",
        f"- init_d13b_checkpoint: `{config.get('model', {}).get('init_d13b_checkpoint', '')}`",
        f"- freeze_backbone: {config.get('model', {}).get('freeze_backbone', False)}",
        f"- trainable_parameters: {init_info.get('trainable_parameters')}",
        f"- lambda_supcon: {config.get('loss', {}).get('lambda_supcon', 0.0)}",
        f"- supcon_temperature: {config.get('loss', {}).get('supcon_temperature', 0.1)}",
        f"- projection_dim: {config.get('model', {}).get('projection_dim', 64)}",
        f"- best_epoch: {best_epoch}",
        f"- best_val_macro_f1: {best_value:.6f}",
        f"- test_macro_f1: {test_metrics.get('macro_f1', 0.0):.6f}",
        f"- test_accuracy: {test_metrics.get('accuracy', 0.0):.6f}",
        f"- D13B M16 reference macro_f1: {D13B_M16_TEST_MACRO_F1:.4f}",
        f"- D13B M16 reference acc: {D13B_M16_TEST_ACC:.4f}",
        f"- effective_slots: {slot_stats.get('effective_slots', 0.0):.6f}",
        f"- slot_overlap: {slot_stats.get('slot_overlap', 0.0):.6f}",
        f"- slot_entropy: {slot_stats.get('slot_entropy', 0.0):.6f}",
        f"- slot_dominance: {slot_stats.get('slot_dominance', 0.0):.6f}",
        f"- supcon_loss: {supcon_stats.get('loss_supcon', 0.0):.6f}",
        f"- positive_pair_count: {supcon_stats.get('positive_pair_count', 0.0):.6f}",
        f"- valid_supcon_anchor_count: {supcon_stats.get('valid_supcon_anchor_count', 0.0):.6f}",
        f"- embedding_collapse_score: {supcon_stats.get('embedding_collapse_score', 0.0):.6f}",
        "",
        "Slot outputs are slot candidates / motif candidate diagnostics only.",
        "Post-D13C visual slot audit is required before any downstream decision.",
        "",
    ]
    (output_root / "d13c_report.md").write_text("\n".join(lines), encoding="utf-8")


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
        if "graph_id" in batch:
            graph_ids.extend(batch["graph_id"].detach().cpu().long().tolist())
        else:
            graph_ids.extend(list(range(len(graph_ids), len(graph_ids) + int(pred.numel()))))
    if not z_images:
        return
    np.savez_compressed(
        output_root / "d13c_test_diagnostic_embeddings.npz",
        z_image=np.concatenate(z_images, axis=0),
        z_proj=np.concatenate(z_projs, axis=0),
        slot_attention=np.concatenate(slot_attns, axis=0),
        graph_id=np.asarray(graph_ids, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        pred=np.asarray(preds, dtype=np.int64),
    )
    metadata = {
        "file": "d13c_test_diagnostic_embeddings.npz",
        "contains": ["z_image", "z_proj", "slot_attention", "graph_id", "label", "pred"],
        "slot_pixel_maps": "not exported by D13C train; use post-visual audit staging with assignment projection",
        "diagnostic_only": True,
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
        "no_causal_claim": True,
    }
    (output_root / "d13c_test_diagnostic_embeddings_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_train(config: Dict[str, Any], output_dir: str | Path | None = None, device_arg: str | None = None) -> Dict[str, Any]:
    if output_dir is not None:
        config.setdefault("paths", {})["resolved_output_root"] = str(output_dir)
        config.setdefault("paths", {})["output_root"] = str(output_dir)
    output_root = Path(config.get("paths", {}).get("resolved_output_root") or output_dir or "outputs/d13c_diagnostic/run")
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    test_loader = build_dataloader(config, "test", shuffle=False)
    model, criterion, optimizer, scheduler, device, init_info = build_objects(config, output_root, device_arg=device_arg)
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 30)))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    amp = bool(training_cfg.get("amp", False))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")
    patience = int(training_cfg.get("early_stopping_patience", 10))
    monitor = str(training_cfg.get("early_stopping_metric", training_cfg.get("monitor", "val_macro_f1")))
    best_value = -float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics, train_slot, train_pool, train_supcon = _run_epoch(model, criterion, train_loader, optimizer, device, epoch, "train", max_train_batches, grad_clip, amp)
        val_metrics, val_slot, val_pool, val_supcon = _run_epoch(model, criterion, val_loader, None, device, epoch, "val", max_val_batches, grad_clip, amp=False)
        if scheduler is not None:
            step_scheduler(scheduler, monitor_value=val_metrics.get("macro_f1"))
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0].get("lr", 0.0))}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({"loaded_d13b_keys": init_info.get("loaded_d13b_keys", 0), "trainable_parameters": init_info.get("trainable_parameters", 0)})
        _append_csv(output_root / "train_log.csv", row)
        _append_csv(output_root / "val_metrics.csv", {"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
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
            _save_checkpoint(output_root / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, row, config)
        else:
            stale += 1
        _save_checkpoint(output_root / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, row, config)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_supcon={train_supcon.get('loss_supcon', 0.0):.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"pos_pairs={train_supcon.get('positive_pair_count', 0.0):.1f} best={best_value:.4f}@{best_epoch}"
        )
        if stale >= patience:
            print(f"Early stopping after {stale} stale epochs")
            break

    (output_root / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_path = output_root / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        _load_model_state(model, ckpt["model_state_dict"])
    test_metrics, test_slot, test_pool, test_supcon = _run_epoch(model, criterion, test_loader, None, device, best_epoch, "test", max_test_batches, grad_clip, amp=False)
    _append_csv(output_root / "test_metrics.csv", {"epoch": best_epoch, **{f"test_{k}": v for k, v in test_metrics.items()}})
    _append_csv(output_root / "slot_stats.csv", {"epoch": best_epoch, "split": "test", **test_slot})
    _append_csv(output_root / "pooling_stats.csv", {"epoch": best_epoch, "split": "test", **test_pool})
    _append_csv(output_root / "supcon_stats.csv", {"epoch": best_epoch, "split": "test", **test_supcon})
    _append_csv(output_root / "pred_count.csv", _pred_count_row(best_epoch, "test", test_metrics))
    y_true, y_pred, graph_ids = _test_predictions(model, test_loader, device, max_test_batches)
    pd.DataFrame({"graph_id": graph_ids, "y_true": y_true, "y_pred": y_pred}).to_csv(output_root / "test_predictions.csv", index=False)
    write_confusion_matrix(y_true, y_pred, output_root / "confusion_matrix.csv")
    _export_test_diagnostics(model, test_loader, device, output_root, max_test_batches)
    _write_report(output_root, config, best_epoch, best_value, test_metrics, test_slot, test_pool, test_supcon, init_info)
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
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = args.output_dir
    run_train(config, output_dir=args.output_dir, device_arg=args.device)


if __name__ == "__main__":
    main()
