"""Train Stage 2A classifier over frozen Stage 1 motif graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

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
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.motif_graph_builder import build_motif_graph  # noqa: E402
from utils.motif_stage1_loader import load_frozen_motif_model  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "train_macro_f1",
    "train_weighted_f1",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "val_weighted_f1",
    "lr",
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
    if sys.platform.startswith("win") and getattr(args, "num_batches", None) is not None and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["output"] = output
    cfg["training"] = training
    return cfg


def _class_weights(training_cfg: Dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not bool(training_cfg.get("class_weights", False)):
        return None
    counts = training_cfg.get("class_counts")
    if not counts:
        return None
    counts_t = torch.tensor([max(float(x), 1.0) for x in counts], dtype=torch.float32, device=device)
    weights = counts_t.sum() / (counts_t.numel() * counts_t)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.clamp(max=float(training_cfg.get("max_class_weight", 3.0)))


def _metric_bundle(y_true: list[int], y_pred: list[int], loss_sum: float, count: int, prefix: str) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred) if y_true else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        f"{prefix}_loss": float(loss_sum / max(count, 1)),
        f"{prefix}_acc": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_weighted_f1": float(metrics["weighted_f1"]),
    }


def _check_finite(loss: torch.Tensor, logits: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite Stage 2 loss: {float(loss.detach().cpu())}")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite Stage 2 logits")


def _run_epoch(
    *,
    stage1_model: torch.nn.Module,
    classifier: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    stage2_cfg: Dict[str, Any],
    class_weights: torch.Tensor | None,
    max_batches: int | None,
    amp: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    classifier.train(mode=is_train)
    stage1_model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum = 0.0
    count = 0
    autocast_enabled = bool(amp and device.type == "cuda")
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        labels = batch["y"].long()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            stage1_outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(batch).items()}
            motif_graph = build_motif_graph(stage1_outputs, stage2_cfg)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = classifier(
                    motif_graph["node_features"],
                    motif_edge_features=motif_graph["edge_features"],
                    selected_weights=motif_graph["selected_weights"],
                )
                loss = F.cross_entropy(logits, labels, weight=class_weights)
        _check_finite(loss, logits)
        if is_train:
            loss.backward()
            grad_clip = float(getattr(_run_epoch, "grad_clip_norm", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            optimizer.step()
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return _metric_bundle(y_true, y_pred, loss_sum, count, "train" if is_train else "val")


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def _save_checkpoint(path: Path, classifier: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def _evaluate_and_save(classifier, stage1_model, loader, device, stage2_cfg, class_weights, output_root: Path, split: str, max_batches: int | None) -> Dict[str, Any]:
    classifier.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            labels = batch["y"].long()
            stage1_outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(batch).items()}
            motif_graph = build_motif_graph(stage1_outputs, stage2_cfg)
            logits = classifier(motif_graph["node_features"], motif_graph["edge_features"], motif_graph["selected_weights"])
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            _check_finite(loss, logits)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            loss_sum += float(loss.cpu()) * int(labels.shape[0])
            count += int(labels.shape[0])
    metrics = _metric_bundle(y_true, y_pred, loss_sum, count, split)
    report = classification_report_dict(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({**metrics, "classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2)
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
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        raise NotImplementedError("Stage 2A synthetic mode is not implemented; use --num_batches for local smoke.")
    config = _update_config(load_config(args.config, environment=args.environment), args)
    training_cfg = dict(config.get("training", {}) or {})
    stage1_cfg = dict(config.get("stage1", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = resolve_device(args.device, config=config)
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    if output_root is None:
        output_root = PROJECT_ROOT / "outputs" / str(config.get("experiment", {}).get("name", "stage2a"))
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)

    stage1_model = load_frozen_motif_model(stage1_cfg["config"], stage1_cfg["checkpoint"], device)
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    first_batch = move_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        first_outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(first_batch).items()}
        first_graph = build_motif_graph(first_outputs, stage2_cfg)
    model_cfg = dict(config.get("model", {}) or {})
    model_cfg["input_dim"] = int(first_graph["node_features"].shape[-1])
    classifier = build_model(model_cfg).to(device)
    stage2_trainable = sum(param.numel() for param in classifier.parameters() if param.requires_grad)
    print(f"[Stage2] node_feature_dim={model_cfg['input_dim']}")
    print(f"[Stage2] trainable_stage2_params={stage2_trainable:,}")
    if stage2_trainable <= 0:
        raise RuntimeError("Stage 2 classifier has no trainable parameters")
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=float(training_cfg.get("lr", 1e-3)), weight_decay=float(training_cfg.get("weight_decay", 1e-4)))
    class_weights = _class_weights(training_cfg, device)
    if class_weights is not None:
        print(f"[Stage2] class_weights={[round(float(v), 4) for v in class_weights.detach().cpu()]}")
    setattr(_run_epoch, "grad_clip_norm", float(training_cfg.get("grad_clip_norm", 1.0)))

    history_path = output_root / "logs" / "history.csv"
    checkpoint_dir = output_root / "checkpoints"
    best_metric = -float("inf")
    best_metrics: Dict[str, Any] = {}
    bad_epochs = 0
    epochs = int(training_cfg.get("epochs", 60))
    max_batches = args.num_batches
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            stage1_model=stage1_model,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            stage2_cfg=stage2_cfg,
            class_weights=class_weights,
            max_batches=max_batches,
            amp=bool(training_cfg.get("amp", True)),
        )
        val_metrics = _run_epoch(
            stage1_model=stage1_model,
            classifier=classifier,
            loader=val_loader,
            optimizer=None,
            device=device,
            stage2_cfg=stage2_cfg,
            class_weights=class_weights,
            max_batches=max_batches,
            amp=False,
        )
        row = {**train_metrics, **val_metrics, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        _append_history(history_path, row)
        metric_name = str(training_cfg.get("save_best_metric", "val_macro_f1"))
        if metric_name not in row:
            raise KeyError(f"Configured save_best_metric={metric_name!r} missing from Stage2 metrics")
        current_metric = float(row[metric_name])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dict(row)
            bad_epochs = 0
            _save_checkpoint(checkpoint_dir / "best.pth", classifier, optimizer, epoch, best_metric, config, best_metrics)
            print(f"[Checkpoint] best epoch={epoch} {metric_name}={best_metric:.4f}")
        else:
            bad_epochs += 1
        _save_checkpoint(checkpoint_dir / "last.pth", classifier, optimizer, epoch, best_metric, config, row)
        print(
            f"[epoch {epoch:03d}] train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )
        patience = int(training_cfg.get("early_stopping_patience", 0) or 0)
        if patience > 0 and bad_epochs >= patience:
            print(f"[EarlyStopping] patience={patience} reached at epoch={epoch}")
            break
    if not math.isfinite(best_metric):
        raise RuntimeError("No finite best metric produced")

    best_ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device)
    classifier.load_state_dict(best_ckpt["model_state_dict"])
    val_eval = _evaluate_and_save(classifier, stage1_model, val_loader, device, stage2_cfg, class_weights, output_root, "val", max_batches)
    print(f"[Output] history={history_path}")
    print(f"[Output] best={checkpoint_dir / 'best.pth'}")
    print(f"[Output] last={checkpoint_dir / 'last.pth'}")
    print(f"[Output] val_metrics={output_root / 'metrics' / 'val_metrics.json'}")
    print(f"[Summary] best_val_macro_f1={best_metric:.4f} eval_val_macro_f1={val_eval['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
