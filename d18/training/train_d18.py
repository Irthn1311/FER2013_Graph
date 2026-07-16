"""Train/evaluate D18 Structure-Guided Pixel Graph models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.collate import D18Batch, collate_d18_graphs
from d18.data.structure_dataset import StructurePixelDataset
from d18.models.structure_gnn import StructureGNN


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def scientific_resume_signature(cfg: Dict[str, Any]) -> str:
    """Hash scientific state while excluding machine-specific paths/metadata."""
    graph_cfg = json.loads(json.dumps(cfg.get("graph", {}) or {}))
    graph_cfg.pop("cache", None)
    training_cfg = json.loads(json.dumps(cfg.get("training", {}) or {}))
    training_cfg.pop("resume_from", None)
    payload = {
        "signature_version": "d18_scientific_resume_v1",
        "seed": int(training_cfg.get("seed", cfg.get("seed", 42)) or 42),
        "graph": graph_cfg,
        "model": cfg.get("model", {}) or {},
        "training": training_cfg,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_resume_signature(cfg: Dict[str, Any]) -> str:
    """Bind a resumable checkpoint to both scientific state and run identity."""
    payload = {
        "signature_version": "d18_run_resume_v2",
        "scientific_signature": scientific_resume_signature(cfg),
        "run_name": str(cfg.get("run_name", "")),
        "output_dir": str(cfg.get("output_dir", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict[str, Any], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def restore_rng(payload: Dict[str, Any]) -> None:
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    if payload.get("numpy_random_state") is not None:
        np.random.set_state(payload["numpy_random_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    elif payload.get("rng_state") is not None:
        torch.set_rng_state(payload["rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def resolve_device(text: str | None) -> torch.device:
    requested = text or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: Dict[str, Any], split: str, max_samples: int | None = None) -> StructurePixelDataset:
    data_cfg = cfg.get("data", {}) or {}
    return StructurePixelDataset(
        prior_dir=data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"),
        split=split,
        graph=cfg.get("graph", {}) or {},
        max_samples=max_samples,
    )


def loader_kwargs(cfg: Dict[str, Any], shuffle: bool) -> Dict[str, Any]:
    train_cfg = cfg.get("training", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    num_workers = int(train_cfg.get("num_workers", data_cfg.get("num_workers", 0)) or 0)
    kwargs: Dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", data_cfg.get("batch_size", 16)) or 16),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "collate_fn": collate_d18_graphs,
        "pin_memory": bool(train_cfg.get("pin_memory", data_cfg.get("pin_memory", False))),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", data_cfg.get("persistent_workers", False)))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", data_cfg.get("prefetch_factor", 2)) or 2)
    return kwargs


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 7) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_predictions(y_true: List[int], y_pred: List[int], loss_sum: float, count: int) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))
    f1s = []
    for i in range(len(CLASS_NAMES)):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
    return {
        "loss": float(loss_sum / max(count, 1)),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
        "per_class_f1": f1s,
    }


def detected_fallback_metrics(y_true: List[int], y_pred: List[int], detected: List[bool]) -> List[Dict[str, Any]]:
    rows = []
    arr_d = np.asarray(detected, dtype=bool)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    for name, mask in [("detected", arr_d), ("fallback", ~arr_d)]:
        if not bool(mask.any()):
            rows.append({"group": name, "total": 0, "accuracy": math.nan, "macro_f1": math.nan})
            continue
        m = metrics_from_predictions(y_t[mask].tolist(), y_p[mask].tolist(), 0.0, 1)
        rows.append({"group": name, "total": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return rows


def _filter_batch_edges(batch: D18Batch, keep: torch.Tensor) -> D18Batch:
    return D18Batch(
        x_cat=batch.x_cat,
        edge_index_cat=batch.edge_index_cat[:, keep],
        edge_attr_cat=batch.edge_attr_cat[keep],
        batch_index=batch.batch_index,
        ptr=batch.ptr,
        y=batch.y,
        sample_index=batch.sample_index,
        pos_cat=batch.pos_cat,
        detected=batch.detected,
        landmark_missing_flag=batch.landmark_missing_flag,
        image_48=batch.image_48,
        edge_type_cat=batch.edge_type_cat[keep],
        structure_relation_id_cat=batch.structure_relation_id_cat[keep],
        local_edge_count=batch.local_edge_count,
        knn_edge_count=batch.knn_edge_count,
        structure_edge_count=batch.structure_edge_count,
        total_edge_count=batch.total_edge_count,
        structure_edge_count_before_purification=batch.structure_edge_count_before_purification,
        structure_edge_count_after_purification=batch.structure_edge_count_after_purification,
        purification_compatibility_kept_mean=batch.purification_compatibility_kept_mean,
        purification_compatibility_dropped_mean=batch.purification_compatibility_dropped_mean,
        node_feature_names=batch.node_feature_names,
        edge_feature_names=batch.edge_feature_names,
    )


def apply_dropedge(batch: D18Batch, p: float) -> D18Batch:
    p = float(p)
    if p <= 0.0 or batch.edge_index_cat.size(1) <= 1:
        return batch
    keep = torch.rand((batch.edge_index_cat.size(1),), device=batch.edge_index_cat.device) >= p
    if not bool(keep.any()):
        keep[int(torch.randint(0, keep.numel(), (1,), device=keep.device).item())] = True
    return _filter_batch_edges(batch, keep)


def apply_graph_regularization(batch: D18Batch, train_cfg: Dict[str, Any]) -> tuple[D18Batch, Dict[str, Any]]:
    reg_cfg = dict((train_cfg.get("graph_regularization") or {}))
    mix_cfg = dict((train_cfg.get("structure_mode_mix") or {}))
    edge_type = batch.edge_type_cat.long()
    keep = torch.ones((edge_type.numel(),), dtype=torch.bool, device=edge_type.device)
    before_structure = int((edge_type == 2).sum().item())
    before_local = int((edge_type == 0).sum().item())
    before_knn = int((edge_type == 1).sum().item())
    forced_samples = 0
    if bool(mix_cfg.get("enabled", False)) and float(mix_cfg.get("p_forced_structure", 0.0) or 0.0) > 0.0 and before_structure > 0:
        p_forced = float(mix_cfg.get("p_forced_structure", 0.0) or 0.0)
        sample_force = torch.rand((batch.num_graphs,), device=edge_type.device) < p_forced
        forced_samples = int(sample_force.sum().item())
        if forced_samples > 0:
            src = batch.edge_index_cat[0].long()
            graph_for_edge = torch.bucketize(src, batch.ptr[1:].to(src.device), right=True)
            forced_edge = sample_force[graph_for_edge]
            keep = keep & ~((edge_type == 2) & forced_edge)
    if not bool(mix_cfg.get("enabled", False)) and forced_samples != 0:
        raise RuntimeError("structure mode mix is disabled but forced samples were generated")
    probs = {
        0: float(reg_cfg.get("drop_local_edge_p", 0.0) or 0.0),
        1: float(reg_cfg.get("drop_knn_edge_p", 0.0) or 0.0),
        2: float(reg_cfg.get("drop_structure_edge_p", 0.0) or 0.0),
    }
    for etype, prob in probs.items():
        if prob <= 0.0:
            continue
        mask = (edge_type == int(etype)) & keep
        if bool(mask.any()):
            keep[mask] = torch.rand((int(mask.sum().item()),), device=edge_type.device) >= prob
    if not bool(keep.any()):
        keep[int(torch.randint(0, keep.numel(), (1,), device=keep.device).item())] = True
    after_type = edge_type[keep]
    stats = {
        "structure_edges_before_drop": before_structure,
        "structure_edges_after_drop": int((after_type == 2).sum().item()),
        "local_edges_after_drop": int((after_type == 0).sum().item()),
        "knn_edges_after_drop": int((after_type == 1).sum().item()),
        "structure_drop_fraction_observed": float(1.0 - (int((after_type == 2).sum().item()) / max(before_structure, 1))),
        "structure_mode_forced_sample_count": forced_samples,
        "structure_mode_official_sample_count": int(batch.num_graphs - forced_samples),
        "structure_mode_total_sample_count": int(batch.num_graphs),
        "structure_mode_forced_sample_pct": float(forced_samples / max(batch.num_graphs, 1)),
        "structure_edges_train_mean": float(before_structure / max(batch.num_graphs, 1)),
    }
    return _filter_batch_edges(batch, keep), stats


def structure_gate_penalty_lambda(cfg: Dict[str, Any]) -> float:
    model_cfg = ((cfg.get("model") or {}).get("edge_context_gnn") or {})
    scalar_cfg = dict(model_cfg.get("scalar_edge_gate") or {})
    edge_gate_cfg = dict(model_cfg.get("edge_gate") or {})
    train_reg = dict((cfg.get("training") or {}).get("graph_regularization") or {})
    return float(
        train_reg.get(
            "structure_gate_penalty_lambda",
            edge_gate_cfg.get("structure_gate_penalty_lambda", scalar_cfg.get("structure_gate_penalty_lambda", 0.0)),
        )
        or 0.0
    )


def train_one_epoch(
    model: StructureGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: torch.nn.Module,
    drop_edge_p: float,
    graph_regularization_cfg: Dict[str, Any],
    gate_penalty_lambda: float,
    progress_interval: int,
    epoch: int,
) -> Dict[str, Any]:
    model.train()
    start = time.perf_counter()
    last_end = start
    loss_sum = 0.0
    count = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    node_counts: List[float] = []
    edge_counts: List[float] = []
    purify_before_counts: List[float] = []
    purify_after_counts: List[float] = []
    purify_kept_means: List[float] = []
    purify_dropped_means: List[float] = []
    wait_times: List[float] = []
    batch_times: List[float] = []
    structure_before_drop: List[float] = []
    structure_after_drop: List[float] = []
    local_after_drop: List[float] = []
    knn_after_drop: List[float] = []
    structure_drop_fraction: List[float] = []
    structure_mode_forced_pct: List[float] = []
    structure_mode_forced_count = 0
    structure_mode_official_count = 0
    structure_mode_total_count = 0
    gate_metric_values: Dict[str, List[float]] = {}
    iterator = iter(loader)
    total_batches = len(loader)
    for batch_idx in range(1, total_batches + 1):
        fetch_start = time.perf_counter()
        batch = next(iterator)
        fetch_done = time.perf_counter()
        wait_time = fetch_done - last_end
        batch = batch.to(device)
        train_batch = apply_dropedge(batch, drop_edge_p)
        train_batch, reg_stats = apply_graph_regularization(train_batch, graph_regularization_cfg)
        optimizer.zero_grad(set_to_none=True)
        out = model(train_batch)
        logits = out["logits"]
        for key, value in dict(out.get("gate_stats", {})).items():
            gate_metric_values.setdefault(key, []).append(float(value.detach().cpu().item()))
        loss = loss_fn(logits, train_batch.y)
        if gate_penalty_lambda > 0.0 and out.get("structure_gate_penalty") is not None:
            loss = loss + float(gate_penalty_lambda) * out["structure_gate_penalty"]
        loss.backward()
        optimizer.step()
        batch_done = time.perf_counter()
        bs = int(train_batch.y.numel())
        loss_sum += float(loss.detach().item()) * bs
        count += bs
        y_true.extend(train_batch.y.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
        node_counts.extend(((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()).tolist())
        edge_counts.extend(batch.total_edge_count.detach().cpu().numpy().tolist())
        structure_before_drop.append(float(reg_stats.get("structure_edges_before_drop", 0)) / max(train_batch.num_graphs, 1))
        structure_after_drop.append(float(reg_stats.get("structure_edges_after_drop", 0)) / max(train_batch.num_graphs, 1))
        local_after_drop.append(float(reg_stats.get("local_edges_after_drop", 0)) / max(train_batch.num_graphs, 1))
        knn_after_drop.append(float(reg_stats.get("knn_edges_after_drop", 0)) / max(train_batch.num_graphs, 1))
        structure_drop_fraction.append(float(reg_stats.get("structure_drop_fraction_observed", 0.0)))
        structure_mode_forced_pct.append(float(reg_stats.get("structure_mode_forced_sample_pct", 0.0)))
        structure_mode_forced_count += int(reg_stats.get("structure_mode_forced_sample_count", 0))
        structure_mode_official_count += int(reg_stats.get("structure_mode_official_sample_count", train_batch.num_graphs))
        structure_mode_total_count += int(reg_stats.get("structure_mode_total_sample_count", train_batch.num_graphs))
        wait_times.append(wait_time)
        batch_times.append(batch_done - fetch_done)
        last_end = batch_done
        if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches):
            elapsed = batch_done - start
            print(
                json.dumps(
                    {
                        "event": "d18_train_progress",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": total_batches,
                        "elapsed_sec": elapsed,
                        "last_batch_time_sec": batch_times[-1],
                        "last_batch_wait_sec": wait_times[-1],
                        "avg_loss_so_far": loss_sum / max(count, 1),
                        "structure_drop_fraction_observed_so_far": float(np.mean(structure_drop_fraction)) if structure_drop_fraction else 0.0,
                        "structure_mode_official_samples_so_far": structure_mode_official_count,
                        "structure_mode_forced_samples_so_far": structure_mode_forced_count,
                        "structure_mode_forced_ratio_so_far": float(structure_mode_forced_count / max(structure_mode_total_count, 1)),
                    }
                ),
                flush=True,
            )
    train_metrics = metrics_from_predictions(y_true, y_pred, loss_sum, count)
    epoch_time = time.perf_counter() - start
    return {
        "train_loss": loss_sum / max(count, 1),
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "node_count_mean": float(np.mean(node_counts)) if node_counts else math.nan,
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else math.nan,
        "train_epoch_time_sec": epoch_time,
        "train_first_batch_wait_time_sec": wait_times[0] if wait_times else math.nan,
        "train_avg_batch_time_ms": float(np.mean(batch_times) * 1000.0) if batch_times else math.nan,
        "train_avg_batch_wait_time_ms": float(np.mean(wait_times) * 1000.0) if wait_times else math.nan,
        "train_num_batches": total_batches,
        "structure_edges_before_drop_mean": float(np.mean(structure_before_drop)) if structure_before_drop else 0.0,
        "structure_edges_after_drop_mean": float(np.mean(structure_after_drop)) if structure_after_drop else 0.0,
        "structure_drop_fraction_observed": float(np.mean(structure_drop_fraction)) if structure_drop_fraction else 0.0,
        "local_edges_after_drop_mean": float(np.mean(local_after_drop)) if local_after_drop else 0.0,
        "knn_edges_after_drop_mean": float(np.mean(knn_after_drop)) if knn_after_drop else 0.0,
        "structure_mode_official_sample_count": structure_mode_official_count,
        "structure_mode_forced_sample_count": structure_mode_forced_count,
        "structure_mode_total_sample_count": structure_mode_total_count,
        "structure_mode_forced_sample_pct": float(structure_mode_forced_count / max(structure_mode_total_count, 1)),
        **{key: float(np.mean(values)) for key, values in gate_metric_values.items() if values},
    }


@torch.no_grad()
def evaluate(model: StructureGNN, loader: DataLoader, device: torch.device, loss_fn: torch.nn.Module) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    start = time.perf_counter()
    last_end = start
    y_true: List[int] = []
    y_pred: List[int] = []
    detected: List[bool] = []
    sample_index: List[int] = []
    loss_sum = 0.0
    count = 0
    node_counts: List[float] = []
    local_counts: List[float] = []
    knn_counts: List[float] = []
    structure_counts: List[float] = []
    edge_counts: List[float] = []
    purify_before_counts: List[float] = []
    purify_after_counts: List[float] = []
    purify_kept_means: List[float] = []
    purify_dropped_means: List[float] = []
    wait_times: List[float] = []
    batch_times: List[float] = []
    structure_before_drop: List[float] = []
    structure_after_drop: List[float] = []
    local_after_drop: List[float] = []
    knn_after_drop: List[float] = []
    structure_drop_fraction: List[float] = []
    structure_mode_forced_pct: List[float] = []
    edge_feature_names: List[str] = []
    iterator = iter(loader)
    total_batches = len(loader)
    for _ in range(total_batches):
        batch = next(iterator)
        fetch_done = time.perf_counter()
        wait_times.append(fetch_done - last_end)
        if not edge_feature_names:
            edge_feature_names = list(batch.edge_feature_names)
        batch = batch.to(device)
        logits = model(batch)["logits"]
        loss = loss_fn(logits, batch.y)
        pred = logits.argmax(dim=1)
        batch_done = time.perf_counter()
        batch_times.append(batch_done - fetch_done)
        last_end = batch_done
        bs = int(batch.y.numel())
        loss_sum += float(loss.item()) * bs
        count += bs
        y_true.extend(batch.y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        detected.extend(batch.detected.detach().cpu().tolist())
        sample_index.extend(batch.sample_index.detach().cpu().tolist())
        node_counts.extend(((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()).tolist())
        local_counts.extend(batch.local_edge_count.detach().cpu().numpy().tolist())
        knn_counts.extend(batch.knn_edge_count.detach().cpu().numpy().tolist())
        structure_counts.extend(batch.structure_edge_count.detach().cpu().numpy().tolist())
        edge_counts.extend(batch.total_edge_count.detach().cpu().numpy().tolist())
        purify_before_counts.extend(batch.structure_edge_count_before_purification.detach().cpu().numpy().tolist())
        purify_after_counts.extend(batch.structure_edge_count_after_purification.detach().cpu().numpy().tolist())
        purify_kept_means.extend(batch.purification_compatibility_kept_mean.detach().cpu().numpy().tolist())
        purify_dropped_means.extend(batch.purification_compatibility_dropped_mean.detach().cpu().numpy().tolist())
    row = metrics_from_predictions(y_true, y_pred, loss_sum, count)
    row["edge_feature_names"] = edge_feature_names
    row.update(
        {
            "node_count_mean": float(np.mean(node_counts)) if node_counts else math.nan,
            "local_edge_count_mean": float(np.mean(local_counts)) if local_counts else math.nan,
            "knn_edge_count_mean": float(np.mean(knn_counts)) if knn_counts else math.nan,
            "structure_edge_count_mean": float(np.mean(structure_counts)) if structure_counts else math.nan,
            "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else math.nan,
            "structure_edges_before_purification_mean": float(np.mean(purify_before_counts)) if purify_before_counts else math.nan,
            "structure_edges_after_purification_mean": float(np.mean(purify_after_counts)) if purify_after_counts else math.nan,
            "purification_keep_ratio_observed": float(np.mean(purify_after_counts) / max(np.mean(purify_before_counts), 1.0)) if purify_before_counts and purify_after_counts else math.nan,
            "compatibility_mean_kept": float(np.nanmean(purify_kept_means)) if purify_kept_means else math.nan,
            "compatibility_mean_dropped": float(np.nanmean(purify_dropped_means)) if purify_dropped_means else math.nan,
            "eval_epoch_time_sec": time.perf_counter() - start,
            "eval_first_batch_wait_time_sec": wait_times[0] if wait_times else math.nan,
            "eval_avg_batch_time_ms": float(np.mean(batch_times) * 1000.0) if batch_times else math.nan,
            "eval_avg_batch_wait_time_ms": float(np.mean(wait_times) * 1000.0) if wait_times else math.nan,
            "eval_num_batches": total_batches,
        }
    )
    detail = {"y_true": y_true, "y_pred": y_pred, "detected": detected, "sample_index": sample_index}
    return row, detail


def save_checkpoint(
    path: Path,
    model: StructureGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    best_score: float,
    best_epoch: int,
    best_val_loss: float,
    best_val_loss_epoch: int,
    epochs_without_improvement: int,
    global_step: int,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_loss_epoch": int(best_val_loss_epoch),
            "epochs_without_improvement": int(epochs_without_improvement),
            "global_step": int(global_step),
            "config": cfg,
            "resume_signature": scientific_resume_signature(cfg),
            "run_resume_signature": run_resume_signature(cfg),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: StructureGNN,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    device: torch.device | str = "cpu",
    restore_random_state: bool = False,
    expected_resume_signature: str | None = None,
    expected_run_resume_signature: str | None = None,
    strict_signature: bool = False,
) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    checkpoint_signature = payload.get("resume_signature")
    if checkpoint_signature is None and isinstance(payload.get("config"), dict):
        checkpoint_signature = scientific_resume_signature(payload["config"])
    if expected_resume_signature is not None and checkpoint_signature != expected_resume_signature:
        message = (
            f"Resume signature mismatch for {path}: checkpoint={checkpoint_signature!r}, "
            f"current={expected_resume_signature!r}. Refusing cross-config resume."
        )
        if strict_signature:
            raise RuntimeError(message)
        print(f"[D18 resume warning] {message}", flush=True)
    checkpoint_run_signature = payload.get("run_resume_signature")
    if (
        expected_run_resume_signature is not None
        and checkpoint_run_signature != expected_run_resume_signature
    ):
        message = (
            f"Run resume signature mismatch for {path}: checkpoint={checkpoint_run_signature!r}, "
            f"current={expected_run_resume_signature!r}. Refusing cross-run resume."
        )
        if strict_signature:
            raise RuntimeError(message)
        print(f"[D18 resume warning] {message}", flush=True)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_random_state:
        restore_rng(payload)
    return payload


def write_eval_outputs(output_dir: Path, prefix: str, row: Dict[str, Any], detail: Dict[str, Any]) -> None:
    cm = row["confusion_matrix"]
    with (output_dir / f"{prefix}confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [int(x) for x in cm[i].tolist()])
    with (output_dir / f"{prefix}per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "support", "f1"])
        writer.writeheader()
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow({"class_id": i, "class_name": name, "support": int(cm[i].sum()), "f1": float(row["per_class_f1"][i])})
    with (output_dir / f"{prefix}detected_vs_fallback_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "total", "accuracy", "macro_f1"])
        writer.writeheader()
        for r in detected_fallback_metrics(detail["y_true"], detail["y_pred"], detail["detected"]):
            writer.writerow(r)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"D18 confusion matrix, acc: {row['accuracy']*100:.2f}%")
        ax.set_xlabel("Pred label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(cm.shape[0]):
            denom = max(int(cm[i].sum()), 1)
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]}\n{cm[i,j]/denom*100:.1f}%", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[D18] skipped confusion png: {exc}", flush=True)


def write_graph_schema(output_dir: Path, cfg: Dict[str, Any], eval_row: Dict[str, Any]) -> None:
    graph_cfg = cfg.get("graph", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    structure_cfg = graph_cfg.get("structure_edges", {}) or {}
    purification_cfg = structure_cfg.get("purification", {}) or {}
    reg_cfg = train_cfg.get("graph_regularization", {}) or {}
    mix_cfg = train_cfg.get("structure_mode_mix", {}) or {}
    gnn_cfg = ((cfg.get("model", {}) or {}).get("edge_context_gnn", {}) or {})
    scalar_cfg = gnn_cfg.get("scalar_edge_gate", {}) or {}
    payload = {
        "node_support_mode": graph_cfg.get("node_support_mode", "stratified_detail_knn"),
        "target_node_count": int(graph_cfg.get("target_node_count", 1800)),
        "actual_node_count_mean": eval_row.get("node_count_mean"),
        "local_edge_count_mean": eval_row.get("local_edge_count_mean"),
        "knn_edge_count_mean": eval_row.get("knn_edge_count_mean"),
        "structure_edge_count_mean": eval_row.get("structure_edge_count_mean"),
        "total_edge_count_mean": eval_row.get("edge_count_mean"),
        "structure_edges_before_purification_mean": eval_row.get("structure_edges_before_purification_mean"),
        "structure_edges_after_purification_mean": eval_row.get("structure_edges_after_purification_mean"),
        "purification_keep_ratio_observed": eval_row.get("purification_keep_ratio_observed"),
        "compatibility_mean_kept": eval_row.get("compatibility_mean_kept"),
        "compatibility_mean_dropped": eval_row.get("compatibility_mean_dropped"),
        "graph_cache": graph_cfg.get("cache", {}),
        "edge_type_available": True,
        "structure_relation_available": True,
        "drop_structure_edge_p": float(reg_cfg.get("drop_structure_edge_p", 0.0) or 0.0),
        "drop_knn_edge_p": float(reg_cfg.get("drop_knn_edge_p", 0.0) or 0.0),
        "drop_local_edge_p": float(reg_cfg.get("drop_local_edge_p", 0.0) or 0.0),
        "structure_mode_mix_enabled": bool(mix_cfg.get("enabled", False)),
        "structure_mode_mix_p": float(mix_cfg.get("p_forced_structure", 0.0) or 0.0),
        "structure_mode_mix_p_forced_structure": float(mix_cfg.get("p_forced_structure", 0.0) or 0.0),
        "structure_purification_enabled": bool(purification_cfg.get("enabled", False)),
        "structure_purification_keep_ratio": float(purification_cfg.get("keep_ratio", 1.0) or 1.0),
        "structure_gate_cap": scalar_cfg.get("structure_gate_cap"),
        "structure_gate_penalty_lambda": structure_gate_penalty_lambda(cfg),
        "node_feature_names": [
            "intensity",
            "gx",
            "gy",
            "x_norm",
            "y_norm",
            "grad_mag",
            "local_mean_3x3",
            "local_std_3x3",
            "laplacian_abs",
            "center_surround",
        ],
        "edge_feature_names": eval_row.get("edge_feature_names") or ([]),
        "uses_face_mask_for_node_selection": False,
        "uses_part_prior_for_node_selection": False,
        "uses_structure_prior_for_edge_topology": bool((graph_cfg.get("structure_edges") or {}).get("enabled", True)),
        "uses_anchor_nodes": False,
        "uses_log_prior_bias": False,
        "uses_node_prior_features": False,
    }
    write_json(output_dir / "graph_schema.json", payload)
    write_json(output_dir / "feature_schema.json", {"node_feature_names": payload["node_feature_names"], "edge_feature_names": payload["edge_feature_names"]})


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = read_config(args.config)
    train_cfg = cfg.get("training", {}) or {}
    seed = int(train_cfg.get("seed", cfg.get("seed", 42)) or 42)
    set_seed(seed)
    device = resolve_device(args.device or train_cfg.get("device"))
    output_dir = Path(args.output_dir or cfg.get("output_dir") or Path("outputs/d18_runs/ofix16") / str(cfg.get("run_name", Path(args.config).stem)))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", cfg)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    max_samples = args.max_samples
    train_ds = build_dataset(cfg, "train", max_samples=max_samples)
    val_ds = build_dataset(cfg, "val", max_samples=max_samples)
    test_ds = build_dataset(cfg, "test", max_samples=max_samples)
    train_loader = DataLoader(train_ds, **loader_kwargs(cfg, shuffle=True))
    val_loader = DataLoader(val_ds, **loader_kwargs(cfg, shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(cfg, shuffle=False))

    first_batch = next(iter(DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_d18_graphs)))
    model = StructureGNN.from_config(cfg, input_dim=int(first_batch.x_cat.size(1)), edge_attr_dim=int(first_batch.edge_attr_cat.size(1))).to(device)
    resume_signature = scientific_resume_signature(cfg)
    strict_run_signature = run_resume_signature(cfg)
    graph_reg_cfg = train_cfg.get("graph_regularization", {}) or {}
    mode_mix_cfg = train_cfg.get("structure_mode_mix", {}) or {}
    effective_config = {
        "event": "d18_effective_training_config",
        "run_id": cfg.get("run_name", output_dir.name),
        "seed": seed,
        "cell": "C2" if bool(mode_mix_cfg.get("enabled", False)) else "C0",
        "node_dim": int(first_batch.x_cat.size(1)),
        "edge_dim": int(first_batch.edge_attr_cat.size(1)),
        "node_feature_names": list(first_batch.node_feature_names),
        "edge_feature_names": list(first_batch.edge_feature_names),
        "node_count_mean": float((first_batch.ptr[1:] - first_batch.ptr[:-1]).float().mean().item()),
        "local_edge_count_mean": float(first_batch.local_edge_count.float().mean().item()),
        "knn_edge_count_mean": float(first_batch.knn_edge_count.float().mean().item()),
        "structure_edge_count_mean": float(first_batch.structure_edge_count.float().mean().item()),
        "drop_edge_p": float(train_cfg.get("drop_edge_p", 0.0) or 0.0),
        "drop_local_edge_p": float(graph_reg_cfg.get("drop_local_edge_p", 0.0) or 0.0),
        "drop_knn_edge_p": float(graph_reg_cfg.get("drop_knn_edge_p", 0.0) or 0.0),
        "drop_structure_edge_p": float(graph_reg_cfg.get("drop_structure_edge_p", 0.0) or 0.0),
        "structure_mode_mix_enabled": bool(mode_mix_cfg.get("enabled", False)),
        "structure_mode_probabilities": {
            "p_forced_structure": float(mode_mix_cfg.get("p_forced_structure", 0.0) or 0.0),
            "p_zero_structure": float(mode_mix_cfg.get("p_zero_structure", 0.0) or 0.0),
        },
        "checkpoint_monitor": train_cfg.get("checkpoint_monitor", "val_macro_f1"),
        "checkpoint_monitor_mode": train_cfg.get("checkpoint_monitor_mode", "max"),
        "output_dir": str(output_dir),
        "resume_signature": resume_signature,
        "run_resume_signature": strict_run_signature,
        "seed_policy": {
            "configured_seed": seed,
            "python_random_seed": seed,
            "numpy_seed": seed,
            "torch_cpu_seed": seed,
            "torch_cuda_seed": seed if torch.cuda.is_available() else None,
            "dataloader_generator": "global_torch_rng_seeded_before_loader_construction",
            "worker_seed_policy": "pytorch_default_worker_seed_from_dataloader_base_seed",
            "mode_mix_rng": "global_torch_rng",
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }
    write_json(output_dir / "effective_training_config.json", effective_config)
    print(json.dumps(effective_config), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-3)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float((train_cfg.get("scheduler", {}) or {}).get("factor", 0.5)),
        patience=int((train_cfg.get("scheduler", {}) or {}).get("patience", 5)),
        min_lr=float((train_cfg.get("scheduler", {}) or {}).get("min_lr", 3e-5)),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    best_score = -math.inf
    best_epoch = 0
    best_val_loss = math.inf
    best_val_loss_epoch = 0
    patience = int((train_cfg.get("early_stopping", {}) or {}).get("patience", 15))
    min_epochs = int((train_cfg.get("early_stopping", {}) or {}).get("min_epochs_before_stop", 30))
    epochs_wo = 0
    max_epochs = int(train_cfg.get("max_epochs", 90))
    drop_edge_p = float(train_cfg.get("drop_edge_p", 0.0) or 0.0)
    graph_regularization_cfg = train_cfg
    gate_penalty_lambda = structure_gate_penalty_lambda(cfg)
    global_step = 0
    start_epoch = 1
    resume_from = args.resume_from or train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            if args.resume_strict:
                raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
            print(f"[D18 resume] checkpoint not found, starting fresh: {resume_path}", flush=True)
        else:
            payload = load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                restore_random_state=True,
                expected_resume_signature=resume_signature,
                expected_run_resume_signature=strict_run_signature,
                strict_signature=bool(args.resume_strict),
            )
            start_epoch = int(payload.get("epoch", 0)) + 1
            best_score = float(payload.get("best_score", best_score))
            best_epoch = int(payload.get("best_epoch", payload.get("epoch", 0)))
            best_val_loss = float(payload.get("best_val_loss", best_val_loss))
            best_val_loss_epoch = int(payload.get("best_val_loss_epoch", 0))
            epochs_wo = int(payload.get("epochs_without_improvement", 0))
            global_step = int(payload.get("global_step", 0))
            info = {
                "event": "d18_resume_loaded",
                "resume_from": str(resume_path),
                "start_epoch": start_epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_loss_epoch": best_val_loss_epoch,
                "epochs_without_improvement": epochs_wo,
                "global_step": global_step,
            }
            write_json(output_dir / "resume_info.json", info)
            append_jsonl(output_dir / "resume_events.jsonl", info)
            print(json.dumps(info), flush=True)

    fields = [
        "epoch",
        "global_step",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "best_val_macro_f1",
        "best_epoch",
        "lr",
        "node_count_mean",
        "local_edge_count_mean",
        "knn_edge_count_mean",
        "structure_edge_count_mean",
        "edge_count_mean",
        "structure_edges_before_purification_mean",
        "structure_edges_after_purification_mean",
        "purification_keep_ratio_observed",
        "compatibility_mean_kept",
        "compatibility_mean_dropped",
        "drop_edge_p",
        "drop_structure_edge_p",
        "drop_knn_edge_p",
        "drop_local_edge_p",
        "structure_edges_before_drop_mean",
        "structure_edges_after_drop_mean",
        "structure_drop_fraction_observed",
        "local_edges_after_drop_mean",
        "knn_edges_after_drop_mean",
        "structure_mode_forced_sample_pct",
        "structure_mode_official_sample_count",
        "structure_mode_forced_sample_count",
        "structure_mode_total_sample_count",
        "structure_gate_penalty_lambda",
        "raw_gate_mean_local",
        "raw_gate_mean_knn",
        "raw_gate_mean_structure",
        "effective_gate_mean_local",
        "effective_gate_mean_knn",
        "effective_gate_mean_structure",
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
        "memory_reserved_mb",
        "cache_enabled",
        "cache_dir",
    ]
    graph_cache = ((cfg.get("graph", {}) or {}).get("cache", {}) or {})
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn,
            drop_edge_p=drop_edge_p,
            graph_regularization_cfg=graph_regularization_cfg,
            gate_penalty_lambda=gate_penalty_lambda,
            progress_interval=int(train_cfg.get("progress_interval_batches", 500) or 500),
            epoch=epoch,
        )
        global_step += int(train_stats.get("train_num_batches", len(train_loader)))
        val_row, _ = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(float(val_row["loss"]))
        improved = float(val_row["macro_f1"]) > best_score
        if improved:
            best_score = float(val_row["macro_f1"])
            best_epoch = epoch
            epochs_wo = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, best_score, best_epoch, best_val_loss, best_val_loss_epoch, epochs_wo, global_step, cfg)
        else:
            epochs_wo += 1
        if float(val_row["loss"]) < best_val_loss:
            best_val_loss = float(val_row["loss"])
            best_val_loss_epoch = epoch
            save_checkpoint(output_dir / "checkpoints" / "best_val_loss.pt", model, optimizer, scheduler, epoch, best_score, best_epoch, best_val_loss, best_val_loss_epoch, epochs_wo, global_step, cfg)
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, best_score, best_epoch, best_val_loss, best_val_loss_epoch, epochs_wo, global_step, cfg)
        epoch_time = time.perf_counter() - epoch_start
        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_stats["train_loss"],
            "train_eval_loss": train_stats["train_loss"],
            "train_accuracy": train_stats["train_accuracy"],
            "train_macro_f1": train_stats["train_macro_f1"],
            "val_loss": val_row["loss"],
            "val_accuracy": val_row["accuracy"],
            "val_macro_f1": val_row["macro_f1"],
            "best_val_macro_f1": best_score,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "node_count_mean": val_row["node_count_mean"],
            "local_edge_count_mean": val_row["local_edge_count_mean"],
            "knn_edge_count_mean": val_row["knn_edge_count_mean"],
            "structure_edge_count_mean": val_row["structure_edge_count_mean"],
            "edge_count_mean": val_row["edge_count_mean"],
            "structure_edges_before_purification_mean": val_row.get("structure_edges_before_purification_mean"),
            "structure_edges_after_purification_mean": val_row.get("structure_edges_after_purification_mean"),
            "purification_keep_ratio_observed": val_row.get("purification_keep_ratio_observed"),
            "compatibility_mean_kept": val_row.get("compatibility_mean_kept"),
            "compatibility_mean_dropped": val_row.get("compatibility_mean_dropped"),
            "drop_edge_p": drop_edge_p,
            "drop_structure_edge_p": float(((train_cfg.get("graph_regularization") or {}).get("drop_structure_edge_p", 0.0)) or 0.0),
            "drop_knn_edge_p": float(((train_cfg.get("graph_regularization") or {}).get("drop_knn_edge_p", 0.0)) or 0.0),
            "drop_local_edge_p": float(((train_cfg.get("graph_regularization") or {}).get("drop_local_edge_p", 0.0)) or 0.0),
            "structure_edges_before_drop_mean": train_stats.get("structure_edges_before_drop_mean"),
            "structure_edges_after_drop_mean": train_stats.get("structure_edges_after_drop_mean"),
            "structure_drop_fraction_observed": train_stats.get("structure_drop_fraction_observed"),
            "local_edges_after_drop_mean": train_stats.get("local_edges_after_drop_mean"),
            "knn_edges_after_drop_mean": train_stats.get("knn_edges_after_drop_mean"),
            "structure_mode_forced_sample_pct": train_stats.get("structure_mode_forced_sample_pct"),
            "structure_mode_official_sample_count": train_stats.get("structure_mode_official_sample_count"),
            "structure_mode_forced_sample_count": train_stats.get("structure_mode_forced_sample_count"),
            "structure_mode_total_sample_count": train_stats.get("structure_mode_total_sample_count"),
            "structure_gate_penalty_lambda": gate_penalty_lambda,
            "raw_gate_mean_local": train_stats.get("raw_gate_mean_local"),
            "raw_gate_mean_knn": train_stats.get("raw_gate_mean_knn"),
            "raw_gate_mean_structure": train_stats.get("raw_gate_mean_structure"),
            "effective_gate_mean_local": train_stats.get("effective_gate_mean_local"),
            "effective_gate_mean_knn": train_stats.get("effective_gate_mean_knn"),
            "effective_gate_mean_structure": train_stats.get("effective_gate_mean_structure"),
            "train_epoch_time_sec": train_stats["train_epoch_time_sec"],
            "train_first_batch_wait_time_sec": train_stats["train_first_batch_wait_time_sec"],
            "train_avg_batch_time_ms": train_stats["train_avg_batch_time_ms"],
            "train_avg_batch_wait_time_ms": train_stats["train_avg_batch_wait_time_ms"],
            "train_num_batches": train_stats["train_num_batches"],
            "val_epoch_time_sec": val_row["eval_epoch_time_sec"],
            "val_first_batch_wait_time_sec": val_row["eval_first_batch_wait_time_sec"],
            "val_avg_batch_time_ms": val_row["eval_avg_batch_time_ms"],
            "val_avg_batch_wait_time_ms": val_row["eval_avg_batch_wait_time_ms"],
            "val_num_batches": val_row["eval_num_batches"],
            "epoch_time_sec": epoch_time,
            "memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024**2)) if device.type == "cuda" else 0.0,
            "cache_enabled": bool(graph_cache.get("enabled", False)),
            "cache_dir": graph_cache.get("dir"),
        }
        append_csv(output_dir / "train_log.csv", log_row, fields)
        print(json.dumps({"event": "d18_epoch", **log_row}), flush=True)
        if epoch >= min_epochs and epochs_wo >= patience:
            stop_info = {"early_stopped": True, "epoch": epoch, "best_epoch": best_epoch, "epochs_without_improvement": epochs_wo, "patience": patience}
            append_jsonl(output_dir / "resume_events.jsonl", {"event": "d18_early_stop", **stop_info})
            print(json.dumps(stop_info), flush=True)
            break

    best_ckpt = output_dir / "checkpoints" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = output_dir / "checkpoints" / "last.pt"
    load_checkpoint(best_ckpt, model, optimizer=None, device=device)
    test_row, test_detail = evaluate(model, test_loader, device, loss_fn)
    write_eval_outputs(output_dir, "", test_row, test_detail)
    load_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer=None, device=device)
    last_row, last_detail = evaluate(model, test_loader, device, loss_fn)
    write_eval_outputs(output_dir, "last_", last_row, last_detail)
    write_graph_schema(output_dir, cfg, test_row)
    summary = {
        "output_dir": str(output_dir),
        "run_name": cfg.get("run_name", output_dir.name),
        "device": str(device),
        "max_epochs": max_epochs,
        "final_test_checkpoint": best_ckpt.name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "test_accuracy": test_row["accuracy"],
        "test_macro_f1": test_row["macro_f1"],
        "last_test_accuracy": last_row["accuracy"],
        "last_test_macro_f1": last_row["macro_f1"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "graph_cache": graph_cache,
    }
    write_json(output_dir / "d18_train_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D18 Structure-Guided Pixel GNN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--resume_strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_train(parse_args())


if __name__ == "__main__":
    main()
