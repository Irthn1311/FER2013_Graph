"""Train D16 pixel-graph runs.

This runner supports CE-only and performance-oriented D16 v1 ablations. It does
not make motif, semantic-region, causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
from d16.data.graph_cache_dataset import D16GraphCacheDataset
from d16.data.pixel_prior_dataset import D16PixelPriorDataset
from d16.losses.part_supcon import PartAwareSupConLoss
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

    def __getitem__(self, index: int):
        detected_graph = self.detected_ds[index]
        if bool(detected_graph.detected.item()):
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


def _single_dataset(
    prior_dir: str | Path,
    split: str,
    graph_mode: str,
    face_threshold: float,
    context_pixels: int,
    max_samples: int | None,
    graph_cache_dir: str | Path | None = None,
    chunk_cache_size: int = 2,
):
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
    chunk_cache_size = int(data_cfg.get("graph_cache_chunk_cache_size", 2))
    if graph_mode == "hybrid_detected_face_fallback_fullmask":
        detected_ds = _single_dataset(
            prior_dir,
            split,
            "face_plus_context",
            face_threshold,
            int(graph_cfg.get("detected_context_pixels", data_cfg.get("detected_context_pixels", 2))),
            max_samples,
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
        graph_cache_dir=graph_cache_dir,
        chunk_cache_size=chunk_cache_size,
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


def _should_eval_epoch(epoch: int, start_epoch: int, max_epochs: int, every_n: int) -> bool:
    every_n = max(int(every_n), 1)
    return epoch == start_epoch or epoch == max_epochs or (epoch % every_n == 0)


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
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
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
    y_true, y_pred, detected_flags = [], [], []
    sample_indices, missing_flags = [], []
    prediction_rows: List[Dict[str, Any]] = []
    losses = []
    node_counts, edge_counts = [], []
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
        out = model(batch)
        logits = out["logits"]
        loss = F.cross_entropy(logits, batch.y)
        losses.append(float(loss.detach().cpu().item()))
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()
        y_cpu = batch.y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        sample_cpu = batch.sample_index.detach().cpu()
        detected_cpu = batch.detected.detach().cpu()
        missing_cpu = batch.landmark_missing_flag.detach().cpu()
        y_true.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        detected_flags.extend(batch.detected.detach().cpu().numpy().astype(bool).tolist())
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
    best_epoch: int = 0,
    epochs_without_improvement: int = 0,
    resume_source: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format": "d16_resume_v1",
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_epoch": int(best_epoch),
            "early_stopping_state": {
                "epochs_without_improvement": int(epochs_without_improvement),
                "best_val_macro_f1": float(best_val_macro_f1),
                "best_epoch": int(best_epoch),
            },
            "rng_state": _rng_state(),
            "config": config,
            "from_scratch": bool(config.get("from_scratch", True)),
            "init_checkpoint": config.get("init_checkpoint"),
            "resume_source": resume_source,
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


def resume_training(
    resume_from: Path,
    model: D16Model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: Path,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(resume_from, model, device)
    if checkpoint.get("optimizer_state_dict") is None:
        raise ValueError(f"D16 resume checkpoint is missing optimizer_state_dict: {resume_from}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_rng = _restore_rng_state(checkpoint.get("rng_state", {})) if restore_rng else False
    resumed_epoch = int(checkpoint.get("epoch", 0) or 0)
    best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -math.inf))
    best_epoch = int(checkpoint.get("best_epoch", 0) or 0)
    if best_epoch <= 0:
        best_epoch = resumed_epoch if math.isfinite(best_val_macro_f1) else 0
    early_state = checkpoint.get("early_stopping_state", {}) or {}
    epochs_without_improvement = int(early_state.get("epochs_without_improvement", 0) or 0)
    event = {
        "resume_from": str(resume_from),
        "resumed_epoch": resumed_epoch,
        "next_epoch": resumed_epoch + 1,
        "loaded_optimizer": True,
        "restored_rng": bool(restored_rng),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "checkpoint_format": checkpoint.get("checkpoint_format"),
        "warning": None if checkpoint.get("rng_state") else "rng_state_missing_in_checkpoint",
    }
    _append_jsonl(output_dir / "resume_events.jsonl", event)
    print("[D16 resume] " + json.dumps(event, indent=2, default=str), flush=True)
    return {
        "start_epoch": resumed_epoch + 1,
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
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
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    denom = weights.sum().clamp_min(1e-8)
    loss = (per_sample * weights).sum() / denom
    fallback_mask = ~batch.detected.bool()
    return loss, {
        "ce_loss": float(loss.detach().cpu().item()),
        "sample_weight_mean": float(weights.detach().mean().cpu().item()),
        "fallback_weight": fallback_weight,
        "fallback_samples": int(fallback_mask.sum().detach().cpu().item()),
    }


def train_one_epoch(
    model: D16Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    progress_interval: int = 0,
    loss_cfg: Dict[str, Any] | None = None,
    supcon_loss_fn: PartAwareSupConLoss | None = None,
    limit_batches: int | None = None,
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
    sample_weight_means = []
    fallback_sample_counts = []
    node_counts, edge_counts = [], []
    epoch_start = time.perf_counter()
    wait_start = epoch_start
    first_batch_wait = None
    batch_wall_times = []
    batch_wait_times = []
    total_batches = len(loader) if limit_batches is None else min(len(loader), int(limit_batches))
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
        optimizer.zero_grad(set_to_none=True)
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
        loss = ce_loss + float(lambda_supcon) * supcon_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"D16 loss is not finite: {float(loss.detach().cpu().item())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        ce_losses.append(ce_stats["ce_loss"])
        supcon_losses.append(float(supcon_loss.detach().cpu().item()))
        sample_weight_means.append(ce_stats["sample_weight_mean"])
        fallback_sample_counts.append(ce_stats["fallback_samples"])
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
        "supcon_loss_total": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
        "supcon_loss_mouth": float(supcon_part_sums["mouth"] / max(len(batch_wall_times), 1)),
        "supcon_loss_eye": float(supcon_part_sums["eye"] / max(len(batch_wall_times), 1)),
        "supcon_loss_brow": float(supcon_part_sums["brow"] / max(len(batch_wall_times), 1)),
        "supcon_loss_nose_cheek": float(supcon_part_sums["nose_cheek"] / max(len(batch_wall_times), 1)),
        "supcon_valid_pairs": float(supcon_valid_pairs),
        "supcon_skipped_parts": float(supcon_skipped_parts),
        "supcon_no_positive_pairs": float(supcon_no_positive_parts),
        "lambda_part_supcon_current": float(_supcon_lambda(loss_cfg, epoch)),
        "sample_weight_mean": float(np.mean(sample_weight_means)) if sample_weight_means else 1.0,
        "fallback_samples_seen": int(np.sum(fallback_sample_counts)) if fallback_sample_counts else 0,
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
    ] + [f"logit_{cls}" for cls in range(7)] + [f"prob_{cls}" for cls in range(7)]
    _write_csv_rows(output_dir / f"{prefix}{split}_metrics.csv", [row], metric_fields)
    _write_csv_rows(output_dir / f"{prefix}per_class_metrics.csv", per_class, per_class_fields)
    _write_csv_rows(output_dir / f"{prefix}pred_count.csv", pred_count, pred_fields)
    _write_csv_rows(output_dir / f"{prefix}detected_vs_fallback_metrics.csv", fallback, fallback_fields)
    _write_csv_rows(output_dir / f"{prefix}detected_fallback_per_class_metrics.csv", group_per_class, group_per_class_fields)
    _write_csv_rows(output_dir / f"{prefix}confusion_matrix.csv", confusion, confusion_fields)
    _write_csv_rows(output_dir / f"{prefix}predictions.csv", predictions, prediction_fields)
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
    parser.add_argument("--restore_rng", dest="restore_rng", action="store_true")
    parser.add_argument("--no_restore_rng", dest="restore_rng", action="store_false")
    parser.set_defaults(restore_rng=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    loss_mode = str((cfg.get("loss", {}) or {}).get("mode", "ce"))
    if loss_mode not in {"ce", "ce_only", "ce_part_supcon", "fallback_weighted_ce", "class_weighted_ce"}:
        raise ValueError(
            "D16 trainer supports CE, class_weighted_ce, fallback_weighted_ce, "
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
    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", training_cfg.get("early_stopping", False)))
    early_metric = str(early_cfg.get("metric", "val_macro_f1"))
    early_mode = str(early_cfg.get("mode", "max"))
    early_patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 999)))
    early_min_epochs = int(early_cfg.get("min_epochs_before_stop", 0))
    eval_every_epoch = bool(training_cfg.get("eval_every_epoch", True))
    eval_every_n_epochs = int(training_cfg.get("eval_every_n_epochs", 1 if eval_every_epoch else max_epochs))
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
    loss_cfg = cfg.get("loss", {}) or {}
    supcon_loss_fn = None
    if loss_mode == "ce_part_supcon":
        supcon_loss_fn = PartAwareSupConLoss(temperature=float(loss_cfg.get("supcon_temperature", 0.1))).to(device)

    best_val_macro_f1 = -math.inf
    best_epoch = 0
    start_epoch = 1
    resume_source = None
    train_fields = [
        "epoch",
        "train_loss",
        "val_macro_f1",
        "val_accuracy",
        "node_count_mean",
        "edge_count_mean",
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
        "epoch_time_sec",
        "evaluated",
        "memory_reserved_mb",
        "ce_loss",
        "supcon_loss_total",
        "supcon_loss_mouth",
        "supcon_loss_eye",
        "supcon_loss_brow",
        "supcon_loss_nose_cheek",
        "supcon_valid_pairs",
        "supcon_skipped_parts",
        "supcon_no_positive_pairs",
        "lambda_part_supcon_current",
        "sample_weight_mean",
        "fallback_samples_seen",
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

    if args.resume_from:
        resume_state = resume_training(
            Path(args.resume_from),
            model,
            optimizer,
            device,
            output_dir,
            restore_rng=bool(args.restore_rng),
        )
        start_epoch = int(resume_state["start_epoch"])
        best_val_macro_f1 = float(resume_state["best_val_macro_f1"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        resume_source = str(resume_state["resume_source"])

    if start_epoch > max_epochs:
        raise ValueError(f"D16 resume start_epoch={start_epoch} exceeds max_epochs={max_epochs}")

    for epoch in range(start_epoch, max_epochs + 1):
        start = time.time()
        progress_interval = int(training_cfg.get("progress_interval_batches", data_cfg.get("progress_interval_batches", 500)) or 0)
        print(json.dumps({"event": "d16_epoch_start", "epoch": int(epoch), "max_epochs": int(max_epochs)}), flush=True)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            progress_interval,
            loss_cfg=loss_cfg,
            supcon_loss_fn=supcon_loss_fn,
            limit_batches=args.limit_train_batches,
        )
        should_eval = _should_eval_epoch(epoch, start_epoch, max_epochs, eval_every_n_epochs)
        val_row: Dict[str, Any] | None = None
        val_per_class: List[Dict[str, Any]] = []
        val_pred_count: List[Dict[str, Any]] = []
        val_fallback: List[Dict[str, Any]] = []
        if should_eval:
            val_row, val_per_class, val_pred_count, val_fallback, _, _, _ = evaluate(
                model,
                val_loader,
                device,
                "val",
                epoch,
                limit_batches=args.limit_val_batches,
            )
        epoch_time = float(time.time() - start)
        memory_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else float("nan")
        log_row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_macro_f1": None if val_row is None else val_row["macro_f1"],
            "val_accuracy": None if val_row is None else val_row["accuracy"],
            "node_count_mean": train_stats["node_count_mean"],
            "edge_count_mean": train_stats["edge_count_mean"],
            "train_epoch_time_sec": train_stats["train_epoch_time_sec"],
            "train_first_batch_wait_time_sec": train_stats["train_first_batch_wait_time_sec"],
            "train_avg_batch_time_ms": train_stats["train_avg_batch_time_ms"],
            "train_avg_batch_wait_time_ms": train_stats["train_avg_batch_wait_time_ms"],
            "train_num_batches": train_stats["train_num_batches"],
            "val_epoch_time_sec": None if val_row is None else val_row.get("val_epoch_time_sec"),
            "val_first_batch_wait_time_sec": None if val_row is None else val_row.get("val_first_batch_wait_time_sec"),
            "val_avg_batch_time_ms": None if val_row is None else val_row.get("val_avg_batch_time_ms"),
            "val_avg_batch_wait_time_ms": None if val_row is None else val_row.get("val_avg_batch_wait_time_ms"),
            "val_num_batches": None if val_row is None else val_row.get("val_num_batches"),
            "epoch_time_sec": epoch_time,
            "evaluated": int(should_eval),
            "memory_reserved_mb": memory_reserved,
            "ce_loss": train_stats["ce_loss"],
            "supcon_loss_total": train_stats["supcon_loss_total"],
            "supcon_loss_mouth": train_stats["supcon_loss_mouth"],
            "supcon_loss_eye": train_stats["supcon_loss_eye"],
            "supcon_loss_brow": train_stats["supcon_loss_brow"],
            "supcon_loss_nose_cheek": train_stats["supcon_loss_nose_cheek"],
            "supcon_valid_pairs": train_stats["supcon_valid_pairs"],
            "supcon_skipped_parts": train_stats["supcon_skipped_parts"],
            "supcon_no_positive_pairs": train_stats["supcon_no_positive_pairs"],
            "lambda_part_supcon_current": train_stats["lambda_part_supcon_current"],
            "sample_weight_mean": train_stats["sample_weight_mean"],
            "fallback_samples_seen": train_stats["fallback_samples_seen"],
        }
        _append_csv(output_dir / "train_log.csv", log_row, train_fields)
        if val_row is not None:
            _append_csv(output_dir / "val_metrics.csv", val_row, metric_fields)
            for row in val_per_class:
                _append_csv(output_dir / "per_class_metrics.csv", row, per_class_fields)
            for row in val_pred_count:
                _append_csv(output_dir / "pred_count.csv", row, pred_fields)
            for row in val_fallback:
                _append_csv(output_dir / "detected_vs_fallback_metrics.csv", row, fallback_fields)

        previous_best_val_macro_f1 = float(best_val_macro_f1)
        previous_best_epoch = int(best_epoch)
        previous_epochs_without_improvement = int(epochs_without_improvement)
        current_val_macro_f1 = None if val_row is None else float(val_row["macro_f1"])
        improved = val_row is not None and float(val_row["macro_f1"]) > best_val_macro_f1
        if improved:
            best_val_macro_f1 = float(val_row["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_macro_f1,
                cfg,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                resume_source=resume_source,
            )
        else:
            if val_row is not None:
                epochs_without_improvement += 1
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_macro_f1,
            cfg,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            resume_source=resume_source,
        )
        score_status = {
            "event": "d16_epoch_score_status",
            "epoch": int(epoch),
            "evaluated": bool(val_row is not None),
            "metric": "val_macro_f1",
            "is_best": bool(improved),
            "current_score": current_val_macro_f1,
            "display_score": float(best_val_macro_f1) if improved else current_val_macro_f1,
            "best_score_before": None if not math.isfinite(previous_best_val_macro_f1) else previous_best_val_macro_f1,
            "best_epoch_before": previous_best_epoch,
            "best_score_current": None if not math.isfinite(float(best_val_macro_f1)) else float(best_val_macro_f1),
            "best_epoch_current": int(best_epoch),
            "early_stopping_enabled": bool(early_enabled),
            "early_stopping_without_improvement_before": previous_epochs_without_improvement,
            "early_stopping_without_improvement_current": int(epochs_without_improvement),
            "early_stopping_patience": int(early_patience),
            "early_stopping_min_epochs": int(early_min_epochs),
        }
        print(json.dumps(score_status), flush=True)
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
                            "epochs_without_improvement": epochs_without_improvement,
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
