"""Train D16 v0 small runs.

This runner is intentionally scoped to D16 v0 CE-only/small-train validation.
It does not enable part-aware SupCon and does not make motif, semantic-region,
causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
from d16.data.pixel_prior_dataset import D16PixelPriorDataset
from d16.models.d16_model import D16Model


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[D16 train] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


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


def build_dataset(cfg: Dict[str, Any], prior_dir: str | Path, split: str) -> D16PixelPriorDataset:
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    max_key = f"max_{split}_samples"
    max_samples = data_cfg.get(max_key)
    if max_samples is not None:
        max_samples = int(max_samples)
    return D16PixelPriorDataset(
        prior_dir,
        split=split,
        graph_mode=graph_cfg.get("graph_mode", data_cfg.get("graph_mode", "face_plus_context")),
        face_threshold=float(graph_cfg.get("face_threshold", 0.15)),
        context_pixels=int(graph_cfg.get("context_pixels", 2)),
        max_samples=max_samples,
    )


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else float("nan"),
        "macro_f1": _macro_f1(y_true, y_pred),
    }


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
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    y_true, y_pred, detected_flags = [], [], []
    losses = []
    node_counts, edge_counts = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = out["logits"]
        loss = F.cross_entropy(logits, batch.y)
        losses.append(float(loss.detach().cpu().item()))
        pred = logits.argmax(dim=1)
        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        detected_flags.extend(batch.detected.detach().cpu().numpy().astype(bool).tolist())
        counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()
        node_counts.extend(counts.tolist())
        edge_counts.append(int(batch.edge_index_cat.size(1)) / max(batch.num_graphs, 1))
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
    }
    per_class = _per_class_rows(y_true_np, y_pred_np, split, int(epoch))
    pred_count = _pred_count_rows(y_pred_np, split, int(epoch))
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
    return row, per_class, pred_count, fallback_rows


def save_checkpoint(path: Path, model: D16Model, optimizer: torch.optim.Optimizer, epoch: int, best_val_macro_f1: float, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": float(best_val_macro_f1),
            "config": config,
            "from_scratch": bool(config.get("from_scratch", True)),
            "init_checkpoint": config.get("init_checkpoint"),
        },
        path,
    )


def load_checkpoint(path: Path, model: D16Model, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def train_one_epoch(model: D16Model, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, Any]:
    model.train()
    losses = []
    node_counts, edge_counts = [], []
    for batch in loader:
        batch: D16Batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = F.cross_entropy(out["logits"], batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"D16 loss is not finite: {float(loss.detach().cpu().item())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        counts = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()
        node_counts.extend(counts.tolist())
        edge_counts.append(int(batch.edge_index_cat.size(1)) / max(batch.num_graphs, 1))
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "node_count_mean": float(np.mean(node_counts)) if node_counts else float("nan"),
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
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
) -> Dict[str, Any]:
    row, per_class, pred_count, fallback = evaluate(
        model,
        loader,
        device,
        split,
        checkpoint_epoch,
        checkpoint_name=checkpoint_name,
        checkpoint_epoch=checkpoint_epoch,
        best_val_macro_f1=best_val_macro_f1,
    )
    _write_csv_rows(output_dir / f"{prefix}{split}_metrics.csv", [row], metric_fields)
    _write_csv_rows(output_dir / f"{prefix}per_class_metrics.csv", per_class, per_class_fields)
    _write_csv_rows(output_dir / f"{prefix}pred_count.csv", pred_count, pred_fields)
    _write_csv_rows(output_dir / f"{prefix}detected_vs_fallback_metrics.csv", fallback, fallback_fields)
    return row


def evaluate_checkpoints(
    cfg: Dict[str, Any],
    prior_dir: Path,
    output_dir: Path,
    device: torch.device,
    checkpoint_path: Path | None = None,
    also_eval_last: bool = True,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    test_ds = build_dataset(cfg, prior_dir, "test")
    test_loader = DataLoader(test_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))
    first_batch = next(iter(DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=first_batch.x_cat.size(1)).to(device)

    best_path = checkpoint_path or (output_dir / "checkpoints" / "best.pt")
    best_checkpoint = load_checkpoint(best_path, model, device)
    best_epoch = _checkpoint_epoch(best_checkpoint, 0)
    best_val_macro_f1 = _checkpoint_best_val(best_checkpoint, float("nan"))
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
            )

    summary = {
        "final_test_checkpoint": best_path.name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": best_row["accuracy"],
        "test_macro_f1": best_row["macro_f1"],
        "last_test_accuracy": None if last_row is None else last_row["accuracy"],
        "last_test_macro_f1": None if last_row is None else last_row["macro_f1"],
        "test_samples": len(test_ds),
    }
    _write_report(output_dir, best_val_macro_f1, best_epoch)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--also_eval_last", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    loss_mode = str((cfg.get("loss", {}) or {}).get("mode", "ce"))
    if loss_mode not in {"ce", "ce_only"}:
        raise ValueError(f"D16 trainer currently supports CE-only modes, got loss.mode={loss_mode!r}")
    if args.prior_dir:
        data_cfg["prior_dir"] = args.prior_dir
    prior_dir = Path(data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best"))
    output_dir = Path(args.output_dir or Path("outputs/d16_runs") / str(cfg.get("run_name", "d16_v0_small")))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
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
    max_epochs = int(args.max_epochs or training_cfg.get("max_epochs", 30))
    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", training_cfg.get("early_stopping", False)))
    early_metric = str(early_cfg.get("metric", "val_macro_f1"))
    early_mode = str(early_cfg.get("mode", "max"))
    early_patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 999)))
    early_min_epochs = int(early_cfg.get("min_epochs_before_stop", 0))
    epochs_without_improvement = 0
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    _write_json(output_dir / "resolved_config.json", cfg)
    Path(output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

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
        print(json.dumps(existing_summary, indent=2), flush=True)
        return

    train_ds = build_dataset(cfg, prior_dir, "train")
    val_ds = build_dataset(cfg, prior_dir, "val")
    test_ds = build_dataset(cfg, prior_dir, "test")
    train_loader = DataLoader(train_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=True))
    val_loader = DataLoader(val_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))
    test_loader = DataLoader(test_ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))

    first_batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=first_batch.x_cat.size(1)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    best_val_macro_f1 = -math.inf
    best_epoch = 0
    train_fields = [
        "epoch",
        "train_loss",
        "val_macro_f1",
        "val_accuracy",
        "node_count_mean",
        "edge_count_mean",
        "epoch_time_sec",
        "memory_reserved_mb",
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
    ]
    per_class_fields = ["split", "epoch", "class_id", "support", "pred_count", "precision", "recall", "f1"]
    pred_fields = ["split", "epoch", "class_id", "pred_count"]
    fallback_fields = ["split", "epoch", "group", "total", "accuracy", "macro_f1"]

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_row, val_per_class, val_pred_count, val_fallback = evaluate(model, val_loader, device, "val", epoch)
        epoch_time = float(time.time() - start)
        memory_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else float("nan")
        log_row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_macro_f1": val_row["macro_f1"],
            "val_accuracy": val_row["accuracy"],
            "node_count_mean": train_stats["node_count_mean"],
            "edge_count_mean": train_stats["edge_count_mean"],
            "epoch_time_sec": epoch_time,
            "memory_reserved_mb": memory_reserved,
        }
        _append_csv(output_dir / "train_log.csv", log_row, train_fields)
        _append_csv(output_dir / "val_metrics.csv", val_row, metric_fields)
        for row in val_per_class:
            _append_csv(output_dir / "per_class_metrics.csv", row, per_class_fields)
        for row in val_pred_count:
            _append_csv(output_dir / "pred_count.csv", row, pred_fields)
        for row in val_fallback:
            _append_csv(output_dir / "detected_vs_fallback_metrics.csv", row, fallback_fields)

        improved = float(val_row["macro_f1"]) > best_val_macro_f1
        if improved:
            best_val_macro_f1 = float(val_row["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val_macro_f1, cfg)
        else:
            epochs_without_improvement += 1
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val_macro_f1, cfg)
        print(json.dumps(log_row, indent=2), flush=True)
        if early_enabled:
            if early_metric != "val_macro_f1" or early_mode != "max":
                raise ValueError("D16 early stopping currently supports metric=val_macro_f1 and mode=max")
            if epoch >= early_min_epochs and epochs_without_improvement >= early_patience:
                print(
                    json.dumps(
                        {
                            "early_stopped": True,
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_macro_f1": best_val_macro_f1,
                            "patience": early_patience,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                break

    eval_summary = evaluate_checkpoints(cfg, prior_dir, output_dir, device, also_eval_last=True)

    summary = {
        "output_dir": str(output_dir),
        "prior_dir": str(prior_dir),
        "device": str(device),
        "max_epochs": max_epochs,
        "final_test_checkpoint": eval_summary["final_test_checkpoint"],
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "test_accuracy": eval_summary["test_accuracy"],
        "test_macro_f1": eval_summary["test_macro_f1"],
        "last_test_accuracy": eval_summary["last_test_accuracy"],
        "last_test_macro_f1": eval_summary["last_test_macro_f1"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    _write_json(output_dir / "d16_train_summary.json", summary)
    _write_report(output_dir, best_val_macro_f1, best_epoch)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
