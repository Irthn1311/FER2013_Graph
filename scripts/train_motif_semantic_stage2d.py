"""Stage 2D light semantic unfreeze for D8M frozen motif discovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_path, save_config  # noqa: E402
from evaluation.evaluator import save_confusion_matrix  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics, confusion_matrix_array  # noqa: E402
from models.registry import build_model  # noqa: E402
from scripts.train_motif_semantic_stage2c import _build_train_loader, _class_weights  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.motif_graph_builder import build_motif_graph  # noqa: E402
from utils.motif_stage1_loader import load_frozen_motif_model  # noqa: E402


MOTIF_METRIC_FIELDS = [
    "selected_border_mass_mean",
    "selected_outer_border_mass_mean",
    "selected_foreground_mass_mean",
    "selection_entropy",
    "selection_effective_count",
    "upper_clean_count",
    "middle_clean_count",
    "lower_clean_count",
    "min_region_clean_count",
    "mean_region_clean_count",
]

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_ce_loss",
    "train_stage1_reg_loss",
    "train_acc",
    "train_macro_f1",
    "train_weighted_f1",
    "val_loss",
    "val_ce_loss",
    "val_stage1_reg_loss",
    "val_acc",
    "val_macro_f1",
    "val_weighted_f1",
    "lr_stage2",
    "lr_stage1",
    *[f"train_{name}" for name in MOTIF_METRIC_FIELDS],
    *[f"val_{name}" for name in MOTIF_METRIC_FIELDS],
]


def _update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    output = dict(cfg.get("output", {}) or {})
    training = dict(cfg.get("training", {}) or {})
    if getattr(args, "output_dir", None):
        paths["resolved_output_root"] = str(args.output_dir)
        output["dir"] = str(args.output_dir)
    elif output.get("dir"):
        paths["resolved_output_root"] = str(output["dir"])
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    if getattr(args, "epochs", None) is not None:
        training["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
    if sys.platform.startswith("win") and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["output"] = output
    cfg["training"] = training
    return cfg


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(str(pattern).lower() in lowered for pattern in patterns)


def _apply_partial_unfreeze(stage1_model: torch.nn.Module, cfg: Dict[str, Any]) -> list[str]:
    if not bool(cfg.get("enabled", True)):
        for param in stage1_model.parameters():
            param.requires_grad = False
        return []
    unfreeze_patterns = cfg.get("unfreeze_patterns") or [
        "motif_queries",
        "query_embed",
        "motif_query",
        "motif_score",
        "score_head",
        "motif_projection",
        "motif_mlp",
        "assignment",
        "logit_scale",
    ]
    freeze_exclude_patterns = cfg.get("freeze_exclude_patterns") or [
        "pixel_encoder",
        "shared_pixel_encoder",
        "graph_swin",
        "backbone",
    ]
    trainable_names: list[str] = []
    for name, param in stage1_model.named_parameters():
        param.requires_grad = False
        if _matches_any(name, freeze_exclude_patterns):
            continue
        if _matches_any(name, unfreeze_patterns):
            param.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        available = [name for name, _ in stage1_model.named_parameters()]
        raise RuntimeError(
            "No Stage1 motif parameters matched stage1_finetune.unfreeze_patterns. "
            f"Available parameter names include: {available[:40]}"
        )
    return trainable_names


def _set_train_modes(stage1_model: torch.nn.Module, projector: torch.nn.Module) -> None:
    stage1_model.train()
    if hasattr(stage1_model, "encoder"):
        stage1_model.encoder.eval()
    projector.train()


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def _metric_bundle(
    y_true: list[int],
    y_pred: list[int],
    loss_sum: float,
    ce_sum: float,
    reg_sum: float,
    motif_sums: Dict[str, float],
    count: int,
    prefix: str,
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred) if y_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    out = {
        f"{prefix}_loss": float(loss_sum / max(count, 1)),
        f"{prefix}_ce_loss": float(ce_sum / max(count, 1)),
        f"{prefix}_stage1_reg_loss": float(reg_sum / max(count, 1)),
        f"{prefix}_acc": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_weighted_f1": float(metrics["weighted_f1"]),
    }
    for key in MOTIF_METRIC_FIELDS:
        out[f"{prefix}_{key}"] = float(motif_sums.get(key, 0.0) / max(count, 1))
    return out


def _check_finite(loss: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite Stage2D loss: {float(loss.detach().cpu())}")
    for key in ("logits", "z", "hidden"):
        if not torch.isfinite(outputs[key]).all():
            raise FloatingPointError(f"Non-finite Stage2D output: {key}")


def _run_epoch(
    *,
    stage1_model: torch.nn.Module,
    projector: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    stage2_cfg: Dict[str, Any],
    criterion_stage1: MotifDiscoveryStage1Loss,
    class_weights: torch.Tensor | None,
    lambda_stage1_reg: float,
    max_batches: int | None,
    amp: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        _set_train_modes(stage1_model, projector)
    else:
        stage1_model.eval()
        projector.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum = 0.0
    ce_sum = 0.0
    reg_sum = 0.0
    motif_sums = {key: 0.0 for key in MOTIF_METRIC_FIELDS}
    count = 0
    autocast_enabled = bool(amp and device.type == "cuda")
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        labels = batch["y"].long()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                stage1_outputs = stage1_model(batch)
                motif_graph = build_motif_graph(stage1_outputs, stage2_cfg, detach=not is_train)
                outputs = projector(motif_graph["node_features"], selected_weights=motif_graph["selected_weights"])
                ce_loss = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
                stage1_loss = criterion_stage1(stage1_outputs, batch)
                reg_loss = stage1_loss["loss"]
                loss = ce_loss + float(lambda_stage1_reg) * reg_loss
        _check_finite(loss, outputs)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in list(projector.parameters()) + list(stage1_model.parameters()) if param.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        ce_sum += float(ce_loss.detach().cpu()) * batch_size
        reg_sum += float(reg_loss.detach().cpu()) * batch_size
        for key in MOTIF_METRIC_FIELDS:
            motif_sums[key] += _scalar(stage1_loss.get(key, 0.0)) * batch_size
        count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(outputs["logits"].detach().argmax(dim=1).cpu().tolist())
    return _metric_bundle(y_true, y_pred, loss_sum, ce_sum, reg_sum, motif_sums, count, "train" if is_train else "val")


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def _save_checkpoint(path: Path, stage1_model: torch.nn.Module, projector: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, config: Dict[str, Any], metrics: Dict[str, Any], trainable_stage1_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "stage1_state_dict": stage1_model.state_dict(),
            "model_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
            "trainable_stage1_names": trainable_stage1_names,
        },
        path,
    )


def _evaluate_and_save(projector, stage1_model, loader, device, stage2_cfg, criterion_stage1, class_weights, lambda_stage1_reg, output_root: Path, split: str, max_batches: int | None) -> Dict[str, Any]:
    metrics = _run_epoch(
        stage1_model=stage1_model,
        projector=projector,
        loader=loader,
        optimizer=None,
        device=device,
        stage2_cfg=stage2_cfg,
        criterion_stage1=criterion_stage1,
        class_weights=class_weights,
        lambda_stage1_reg=lambda_stage1_reg,
        max_batches=max_batches,
        amp=False,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            stage1_outputs = stage1_model(batch)
            motif_graph = build_motif_graph(stage1_outputs, stage2_cfg, detach=True)
            outputs = projector(motif_graph["node_features"], selected_weights=motif_graph["selected_weights"])
            y_true.extend(batch["y"].long().cpu().tolist())
            y_pred.extend(outputs["logits"].argmax(dim=1).cpu().tolist())
    report = classification_report_dict(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({**metrics, "classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
    save_confusion_matrix(cm, figures_dir / f"{split}_confusion_matrix.png")
    return metrics


def _count_grads(model: torch.nn.Module) -> int:
    return sum(1 for param in model.parameters() if param.grad is not None)


def _build_stage1_criterion(stage1_config_path: str | Path, device: torch.device) -> MotifDiscoveryStage1Loss:
    stage1_config = load_config(stage1_config_path)
    model_cfg = dict(stage1_config.get("model", {}) or {})
    loss_cfg = dict(stage1_config.get("motif_loss", stage1_config.get("motif", {}).get("loss", {})) or {})
    loss_cfg.setdefault("height", int(model_cfg.get("height", model_cfg.get("image_size", 48))))
    loss_cfg.setdefault("width", int(model_cfg.get("width", model_cfg.get("image_size", 48))))
    return MotifDiscoveryStage1Loss(loss_cfg).to(device)


def _infer_projector(stage1_model, loader, device, stage2_cfg, model_cfg):
    first_batch = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        first_outputs = stage1_model(first_batch)
        first_graph = build_motif_graph(first_outputs, stage2_cfg, detach=True)
    cfg = dict(model_cfg)
    cfg["input_dim"] = int(first_graph["node_features"].shape[-1])
    return build_model(cfg).to(device), first_batch, first_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    training_cfg = dict(config.get("training", {}) or {})
    stage1_cfg = dict(config.get("stage1", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    finetune_cfg = dict(config.get("stage1_finetune", {}) or {})
    reg_cfg = dict(config.get("stage1_regularization", {}) or {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = resolve_device(args.device, config=config)
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    output_root = output_root or PROJECT_ROOT / "outputs" / str(config.get("experiment", {}).get("name", "stage2d"))
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)

    stage1_model = load_frozen_motif_model(stage1_cfg["config"], stage1_cfg["checkpoint"], device)
    trainable_stage1_names = _apply_partial_unfreeze(stage1_model, finetune_cfg)
    train_loader = _build_train_loader(config, training_cfg)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    projector, first_batch, first_graph = _infer_projector(stage1_model, train_loader, device, stage2_cfg, config.get("model", {}) or {})
    criterion_stage1 = _build_stage1_criterion(stage1_cfg["config"], device)
    lambda_stage1_reg = float(reg_cfg.get("lambda_stage1_reg", 0.0)) if bool(reg_cfg.get("enabled", False)) else 0.0
    class_weights = _class_weights(training_cfg, device)
    use_sampler = bool(training_cfg.get("use_weighted_sampler", training_cfg.get("weighted_sampler", False)))
    if use_sampler and class_weights is not None:
        raise ValueError("Stage2D forbids using weighted sampler and CE class weights together")

    stage1_total = sum(param.numel() for param in stage1_model.parameters())
    stage1_trainable = sum(param.numel() for param in stage1_model.parameters() if param.requires_grad)
    stage2_trainable = sum(param.numel() for param in projector.parameters() if param.requires_grad)
    print(f"[Stage2D] stage1_total_params={stage1_total:,}")
    print(f"[Stage2D] stage1_trainable_params={stage1_trainable:,}")
    print(f"[Stage2D] stage1_trainable_names={trainable_stage1_names}")
    print(f"[Stage2D] stage2_trainable_params={stage2_trainable:,}")
    print(f"[Stage2D] node_feature_dim={int(first_graph['node_features'].shape[-1])} pooled_dim={int(first_graph['node_features'].shape[-1]) * 3}")
    print(f"[Stage2D] lambda_stage1_reg={lambda_stage1_reg}")
    if class_weights is not None:
        print(f"[Stage2D] class_weights={[round(float(v), 4) for v in class_weights.detach().cpu()]}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [param for param in projector.parameters() if param.requires_grad],
                "lr": float(training_cfg.get("lr", 1e-3)),
                "weight_decay": float(training_cfg.get("weight_decay", 1e-4)),
            },
            {
                "params": [param for param in stage1_model.parameters() if param.requires_grad],
                "lr": float(finetune_cfg.get("lr_stage1", 1e-5)),
                "weight_decay": float(finetune_cfg.get("weight_decay_stage1", 1e-5)),
            },
        ]
    )

    if int(training_cfg.get("epochs", 10)) <= 1 and args.num_batches == 1:
        _set_train_modes(stage1_model, projector)
        batch = first_batch
        labels = batch["y"].long()
        optimizer.zero_grad(set_to_none=True)
        stage1_outputs = stage1_model(batch)
        motif_graph = build_motif_graph(stage1_outputs, stage2_cfg, detach=False)
        outputs = projector(motif_graph["node_features"], selected_weights=motif_graph["selected_weights"])
        ce_loss = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
        stage1_loss = criterion_stage1(stage1_outputs, batch)
        loss = ce_loss + lambda_stage1_reg * stage1_loss["loss"]
        _check_finite(loss, outputs)
        loss.backward()
        print(f"[Debug] node_features={tuple(motif_graph['node_features'].shape)} pooled={tuple(outputs['pooled_repr'].shape)} hidden={tuple(outputs['hidden'].shape)} z={tuple(outputs['z'].shape)} logits={tuple(outputs['logits'].shape)}")
        print(f"[Debug] ce_loss={float(ce_loss.detach().cpu()):.6f} stage1_reg={float(stage1_loss['loss'].detach().cpu()):.6f} total={float(loss.detach().cpu()):.6f}")
        print(f"[Debug] stage1_grad_count_after_backward={_count_grads(stage1_model)}")
        print(f"[Debug] stage2_grad_count_after_backward={_count_grads(projector)}")
        return

    history_path = output_root / "logs" / "history.csv"
    checkpoint_dir = output_root / "checkpoints"
    best_metric = -float("inf")
    best_metrics: Dict[str, Any] = {}
    epochs = int(training_cfg.get("epochs", 10))
    max_batches = args.num_batches
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            stage1_model=stage1_model,
            projector=projector,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            stage2_cfg=stage2_cfg,
            criterion_stage1=criterion_stage1,
            class_weights=class_weights,
            lambda_stage1_reg=lambda_stage1_reg,
            max_batches=max_batches,
            amp=bool(training_cfg.get("amp", True)),
        )
        val_metrics = _run_epoch(
            stage1_model=stage1_model,
            projector=projector,
            loader=val_loader,
            optimizer=None,
            device=device,
            stage2_cfg=stage2_cfg,
            criterion_stage1=criterion_stage1,
            class_weights=class_weights,
            lambda_stage1_reg=lambda_stage1_reg,
            max_batches=max_batches,
            amp=False,
        )
        row = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "lr_stage2": optimizer.param_groups[0]["lr"],
            "lr_stage1": optimizer.param_groups[1]["lr"],
        }
        _append_history(history_path, row)
        metric_name = str(training_cfg.get("save_best_metric", "val_macro_f1"))
        if metric_name not in row:
            raise KeyError(f"Configured save_best_metric={metric_name!r} missing from Stage2D metrics")
        current_metric = float(row[metric_name])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dict(row)
            _save_checkpoint(checkpoint_dir / "best.pth", stage1_model, projector, optimizer, epoch, best_metric, config, best_metrics, trainable_stage1_names)
            print(f"[Checkpoint] best epoch={epoch} {metric_name}={best_metric:.4f}")
        _save_checkpoint(checkpoint_dir / "last.pth", stage1_model, projector, optimizer, epoch, best_metric, config, row, trainable_stage1_names)
        print(
            f"[epoch {epoch:03d}] train_ce={row['train_ce_loss']:.4f} val_ce={row['val_ce_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
            f"sel_border={row['val_selected_border_mass_mean']:.4f} "
            f"sel_outer={row['val_selected_outer_border_mass_mean']:.4f} "
            f"sel_fg={row['val_selected_foreground_mass_mean']:.4f} "
            f"sel_H={row['val_selection_entropy']:.4f} "
            f"region_clean={row['val_mean_region_clean_count']:.2f}/{row['val_min_region_clean_count']:.2f}"
        )
    if not math.isfinite(best_metric):
        raise RuntimeError("No finite best metric produced")

    best_ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device)
    stage1_model.load_state_dict(best_ckpt["stage1_state_dict"])
    projector.load_state_dict(best_ckpt["model_state_dict"])
    val_eval = _evaluate_and_save(projector, stage1_model, val_loader, device, stage2_cfg, criterion_stage1, class_weights, lambda_stage1_reg, output_root, "val", max_batches)
    print(f"[Output] history={history_path}")
    print(f"[Output] best={checkpoint_dir / 'best.pth'}")
    print(f"[Output] last={checkpoint_dir / 'last.pth'}")
    print(f"[Output] val_metrics={output_root / 'metrics' / 'val_metrics.json'}")
    print(f"[Summary] best_val_macro_f1={best_metric:.4f} eval_val_macro_f1={val_eval['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
