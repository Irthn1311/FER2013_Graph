"""Train D9-RG-MR end-to-end classification on FER pixel graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
from training.losses import compute_class_weights  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_cls_loss",
    "train_motif_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_weighted_f1",
    "val_loss",
    "val_cls_loss",
    "val_motif_loss",
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
        print(f"[ClassWeights] label count fallback failed: {exc}")
        return None
    return [int(v) for v in counts.tolist()]


def _build_class_weights(config: Dict[str, Any], train_loader, device: torch.device) -> torch.Tensor | None:
    loss_cfg = dict(config.get("loss", {}) or {})
    training_cfg = dict(config.get("training", {}) or {})
    if not bool(loss_cfg.get("use_class_weights", training_cfg.get("class_weights", False))):
        print("[ClassWeights] disabled")
        return None
    num_classes = int(config.get("model", {}).get("num_classes", config.get("data", {}).get("num_classes", 7)))
    counts = loss_cfg.get("class_counts") or training_cfg.get("class_counts")
    source = "config"
    if counts is None:
        counts = _count_train_labels(train_loader, num_classes=num_classes)
        source = "train_dataset"
    if counts is None:
        print("[ClassWeights] disabled: no counts available")
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
    print(f"[ClassWeights] source={source} counts={list(counts)} weights={[round(float(v), 4) for v in weights.cpu()]}")
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


def _metric_bundle(
    *,
    y_true: list[int],
    y_pred: list[int],
    loss_sum: float,
    cls_sum: float,
    motif_sum: float,
    motif_metric_sums: Dict[str, float],
    count: int,
    prefix: str,
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred) if y_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    out = {
        f"{prefix}_loss": float(loss_sum / max(count, 1)),
        f"{prefix}_cls_loss": float(cls_sum / max(count, 1)),
        f"{prefix}_motif_loss": float(motif_sum / max(count, 1)),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_weighted_f1": float(metrics["weighted_f1"]),
    }
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
    max_batches: int | None,
    amp: bool,
    log_interval: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum = 0.0
    cls_sum = 0.0
    motif_sum = 0.0
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
            print(
                "[D9Batch] "
                f"original_node_dim={original_node_dim} masked_node_dim={int(batch['x'].shape[-1])} "
                f"original_edge_dim={original_edge_dim} masked_edge_dim={int(batch['edge_attr'].shape[-1])} "
                f"x_shape={list(batch['x'].shape)} edge_attr_shape={list(batch['edge_attr'].shape)}"
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
                loss = cls_loss + float(motif_aux_weight) * motif_loss
        _check_finite(loss, out)
        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=1.0,
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"Non-finite D9 grad norm at batch {batch_idx}: {float(grad_norm)}")
            optimizer.step()
        batch_size = int(labels.shape[0])
        pred = out["logits"].detach().argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        loss_sum += float(loss.detach().cpu()) * batch_size
        cls_sum += float(cls_loss.detach().cpu()) * batch_size
        motif_sum += float(motif_loss.detach().cpu()) * batch_size
        motif_metrics = _selected_motif_metrics(out)
        for key in motif_metric_sums:
            motif_metric_sums[key] += motif_metrics.get(key, 0.0) * batch_size
        count += batch_size
        if is_train and log_interval > 0 and (batch_idx + 1) % int(log_interval) == 0:
            print(
                f"[train batch {batch_idx + 1}] loss={float(loss.detach().cpu()):.4f} "
                f"cls={float(cls_loss.detach().cpu()):.4f} motif={float(motif_loss.detach().cpu()):.4f} "
                f"logits_shape={tuple(out['logits'].shape)} motifs={tuple(out['motif_maps'].shape)}"
            )
    return _metric_bundle(
        y_true=y_true,
        y_pred=y_pred,
        loss_sum=loss_sum,
        cls_sum=cls_sum,
        motif_sum=motif_sum,
        motif_metric_sums=motif_metric_sums,
        count=count,
        prefix="train" if is_train else "val",
    )


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
    return parser.parse_args()


def main() -> None:
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
    print(f"[Device] selected={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[Device] gpu={torch.cuda.get_device_name(device.index or torch.cuda.current_device())}")
    print(f"[Output] run_dir={output_root}")
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
    )

    train_loader = build_dataloader(config, split=str(args.split or "train"), shuffle=True)
    val_loader = build_dataloader(config, split=str(args.val_split or "val"), shuffle=False)
    model = build_model(model_cfg).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"[Model] name={model_cfg.get('name')} total_params={total_params:,} trainable_params={trainable_params:,}")
    print(
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
    print(f"[Loss] cls_loss=cross_entropy motif_aux_weight={motif_aux_weight}")

    epochs = int(training_cfg.get("epochs", 60))
    max_train_batches = training_cfg.get("max_train_batches", args.max_train_batches)
    max_val_batches = training_cfg.get("max_val_batches", args.max_val_batches)
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
    print(f"[Checkpoint] save_best_metric={monitor} mode={mode}")
    print(f"[EarlyStopping] enabled={early_enabled} monitor={early_monitor} mode={early_mode} patience={patience}")
    print(f"[Scheduler] name={training_cfg.get('scheduler', 'none')} uses_monitor={scheduler_uses_monitor}")

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
    )
    _save_checkpoint(checkpoint_dir / "initial.pth", model, optimizer, scheduler, 0, -float("inf"), -1, config, initial_metrics, monitor, mode)
    print(
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
            max_batches=max_train_batches,
            amp=bool(training_cfg.get("amp", True)),
            log_interval=int(logging_cfg.get("log_interval", training_cfg.get("log_every", 20))),
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
            print(f"[Checkpoint] best epoch={epoch} val_macro_f1={best_metric:.4f}")
        else:
            stale_epochs += 1
        _save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scheduler, epoch, best_metric, best_epoch, config, row, monitor, mode)
        print(
            f"[epoch {epoch:03d}] train_loss={row['train_loss']:.4f} train_macro_f1={row['train_macro_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_accuracy={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} best_epoch={best_epoch} stale={stale_epochs}/{patience}"
        )
        if early_enabled and stale_epochs >= patience:
            print(f"[EarlyStopping] stop epoch={epoch} stale_epochs={stale_epochs}")
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
    print(f"[Output] initial={checkpoint_dir / 'initial.pth'}")
    print(f"[Output] best={checkpoint_dir / 'best.pth'}")
    print(f"[Output] last={checkpoint_dir / 'last.pth'}")
    print(f"[Output] history={history_path}")
    print(f"[Output] val_metrics_jsonl={val_jsonl}")
    print(f"[Output] val_metrics={output_root / 'metrics' / 'val_metrics.json'}")
    print(
        f"[Summary] best_epoch={best_epoch} best_val_macro_f1={best_metric:.4f} "
        f"eval_val_macro_f1={val_eval['macro_f1']:.4f} eval_val_accuracy={val_eval['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
