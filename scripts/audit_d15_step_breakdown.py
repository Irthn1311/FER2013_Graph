"""Profile D15 per-step timing after the chunk-aware speedfix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, save_config
from training.train_d14 import _apply_graph_augmentation
from training.train_d15 import build_objects
from training.trainer import move_to_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mem(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"max_memory_allocated_gb": 0.0, "max_memory_reserved_gb": 0.0}
    gib = 1024 ** 3
    return {
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / gib,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / gib,
    }


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / max(len(vals), 1)


def _pct(part: float, total: float) -> float:
    return float(part / total * 100.0) if total > 0 else 0.0


def _edge_recompute_expected(batch: Dict[str, torch.Tensor], augmentation_cfg: Dict[str, Any]) -> bool:
    return bool(
        augmentation_cfg
        and augmentation_cfg.get("enabled", False)
        and "edge_attr" in batch
        and "x" in batch
        and torch.is_tensor(batch["x"])
        and batch["x"].ndim == 3
        and int(batch["x"].shape[1]) == 48 * 48
    )


def _train_profile(
    model,
    criterion,
    optimizer,
    loader,
    device: torch.device,
    augmentation_cfg: Dict[str, Any],
    warmup_batches: int,
    profile_batches: int,
    csv_path: Path,
) -> List[Dict[str, Any]]:
    model.train()
    rows: List[Dict[str, Any]] = []
    total_needed = int(warmup_batches) + int(profile_batches)
    data_wait_start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > total_needed:
            break
        _sync(device)
        got_batch = time.perf_counter()
        data_wait_ms = (got_batch - data_wait_start) * 1000.0
        do_record = batch_idx > warmup_batches

        edge_recompute = _edge_recompute_expected(batch, augmentation_cfg)
        _sync(device)
        t0 = time.perf_counter()
        batch = _apply_graph_augmentation(batch, augmentation_cfg)
        _sync(device)
        augmentation_ms = (time.perf_counter() - t0) * 1000.0

        _sync(device)
        t0 = time.perf_counter()
        batch = move_to_device(batch, device)
        _sync(device)
        h2d_ms = (time.perf_counter() - t0) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        step_start = time.perf_counter()
        t0 = step_start
        out = model(batch)
        _sync(device)
        forward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        loss_dict = criterion(out, batch["y"], batch)
        loss = loss_dict["loss"]
        _sync(device)
        loss_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        optimizer_ms = (time.perf_counter() - t0) * 1000.0

        total_step_ms = data_wait_ms + augmentation_ms + h2d_ms + forward_ms + loss_ms + backward_ms + optimizer_ms
        if do_record:
            row = {
                "phase": "train",
                "batch_idx": batch_idx,
                "data_wait_time_ms": data_wait_ms,
                "augmentation_time_ms": augmentation_ms,
                "h2d_transfer_time_ms": h2d_ms,
                "forward_time_ms": forward_ms,
                "loss_time_ms": loss_ms,
                "backward_time_ms": backward_ms,
                "optimizer_step_time_ms": optimizer_ms,
                "total_step_time_ms": total_step_ms,
                "val_forward_time_ms": 0.0,
                "edge_attr_compute_time_ms": augmentation_ms if edge_recompute else 0.0,
                "edge_attr_recompute_count": 1 if edge_recompute else 0,
                "chunk_load_time_ms": float(batch.get("chunk_load_time_ms", torch.zeros(1)).sum().detach().cpu().item()) if "chunk_load_time_ms" in batch else 0.0,
                "graph_getitem_time_ms": "",
                "collate_time_ms": "",
                "loss_finite": bool(torch.isfinite(loss).detach().cpu().item()),
            }
            rows.append(row)
            _append_csv(csv_path, row)
        data_wait_start = time.perf_counter()
    return rows


@torch.no_grad()
def _val_profile(model, loader, device: torch.device, warmup_batches: int, profile_batches: int, csv_path: Path) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    total_needed = int(warmup_batches) + int(profile_batches)
    data_wait_start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > total_needed:
            break
        _sync(device)
        got_batch = time.perf_counter()
        data_wait_ms = (got_batch - data_wait_start) * 1000.0
        do_record = batch_idx > warmup_batches

        _sync(device)
        t0 = time.perf_counter()
        batch = move_to_device(batch, device)
        _sync(device)
        h2d_ms = (time.perf_counter() - t0) * 1000.0

        _sync(device)
        t0 = time.perf_counter()
        _ = model(batch)
        _sync(device)
        forward_ms = (time.perf_counter() - t0) * 1000.0
        total_step_ms = data_wait_ms + h2d_ms + forward_ms
        if do_record:
            row = {
                "phase": "val",
                "batch_idx": batch_idx,
                "data_wait_time_ms": data_wait_ms,
                "augmentation_time_ms": 0.0,
                "h2d_transfer_time_ms": h2d_ms,
                "forward_time_ms": 0.0,
                "loss_time_ms": 0.0,
                "backward_time_ms": 0.0,
                "optimizer_step_time_ms": 0.0,
                "total_step_time_ms": total_step_ms,
                "val_forward_time_ms": forward_ms,
                "edge_attr_compute_time_ms": 0.0,
                "edge_attr_recompute_count": 0,
                "chunk_load_time_ms": float(batch.get("chunk_load_time_ms", torch.zeros(1)).sum().detach().cpu().item()) if "chunk_load_time_ms" in batch else 0.0,
                "graph_getitem_time_ms": "",
                "collate_time_ms": "",
                "loss_finite": "",
            }
            rows.append(row)
            _append_csv(csv_path, row)
        data_wait_start = time.perf_counter()
    return rows


def _write_outputs(output_dir: Path, config_path: str, train_rows: List[Dict[str, Any]], val_rows: List[Dict[str, Any]], device: torch.device) -> None:
    train_total = _mean(train_rows, "total_step_time_ms")
    data_wait = _mean(train_rows, "data_wait_time_ms")
    forward = _mean(train_rows, "forward_time_ms")
    loss = _mean(train_rows, "loss_time_ms")
    backward = _mean(train_rows, "backward_time_ms")
    optimizer = _mean(train_rows, "optimizer_step_time_ms")
    augmentation = _mean(train_rows, "augmentation_time_ms")
    h2d = _mean(train_rows, "h2d_transfer_time_ms")
    val_forward = _mean(val_rows, "val_forward_time_ms")
    summary = {
        "config": config_path,
        "train_profile_batches": len(train_rows),
        "val_profile_batches": len(val_rows),
        "avg_total_step_time_ms": train_total,
        "avg_data_wait_time_ms": data_wait,
        "avg_data_wait_percent": _pct(data_wait, train_total),
        "avg_h2d_transfer_time_ms": h2d,
        "avg_h2d_percent": _pct(h2d, train_total),
        "avg_augmentation_time_ms": augmentation,
        "avg_augmentation_percent": _pct(augmentation, train_total),
        "avg_forward_time_ms": forward,
        "avg_forward_percent": _pct(forward, train_total),
        "avg_loss_time_ms": loss,
        "avg_loss_percent": _pct(loss, train_total),
        "avg_backward_time_ms": backward,
        "avg_backward_percent": _pct(backward, train_total),
        "avg_optimizer_step_time_ms": optimizer,
        "avg_optimizer_percent": _pct(optimizer, train_total),
        "avg_val_forward_time_ms": val_forward,
        "edge_attr_recompute_count": int(sum(int(r.get("edge_attr_recompute_count", 0)) for r in train_rows)),
        "avg_edge_attr_compute_time_ms": _mean(train_rows, "edge_attr_compute_time_ms"),
        **_mem(device),
    }
    (output_dir / "d15_step_breakdown_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bottleneck = "data loading" if summary["avg_data_wait_percent"] > 20.0 else "GPU compute"
    lines = [
        "# D15 Step Breakdown Report",
        "",
        f"- config: `{config_path}`",
        f"- train_profile_batches: {len(train_rows)}",
        f"- val_profile_batches: {len(val_rows)}",
        f"- bottleneck_after_speedfix: {bottleneck}",
        f"- data_wait_percent: {summary['avg_data_wait_percent']:.2f}%",
        f"- forward_percent: {summary['avg_forward_percent']:.2f}%",
        f"- backward_percent: {summary['avg_backward_percent']:.2f}%",
        f"- optimizer_percent: {summary['avg_optimizer_percent']:.2f}%",
        f"- augmentation_percent: {summary['avg_augmentation_percent']:.2f}%",
        f"- avg_val_forward_time_ms: {summary['avg_val_forward_time_ms']:.3f}",
        f"- max_memory_reserved_gb: {summary['max_memory_reserved_gb']:.3f}",
        "",
        "Interpretation:",
        f"- Data wait is {'still high' if summary['avg_data_wait_percent'] > 20.0 else 'not the dominant bottleneck'} after chunk-aware sampling.",
        f"- Edge attr recompute count during profiled train batches: {summary['edge_attr_recompute_count']}.",
    ]
    (output_dir / "d15_step_breakdown_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--num_warmup_batches", type=int, default=5)
    parser.add_argument("--num_profile_batches", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("paths", {})["resolved_output_root"] = str(output_dir)
    config.setdefault("paths", {})["output_root"] = str(output_dir)
    save_config(config, output_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    model, criterion, optimizer, _scheduler, device, _init_info = build_objects(config, output_dir, device_arg=args.device)
    csv_path = output_dir / "d15_step_breakdown.csv"
    if csv_path.exists():
        csv_path.unlink()
    train_rows = _train_profile(
        model, criterion, optimizer, train_loader, device, dict(config.get("augmentation", {}) or {}),
        args.num_warmup_batches, args.num_profile_batches, csv_path,
    )
    val_rows = _val_profile(model, val_loader, device, args.num_warmup_batches, args.num_profile_batches, csv_path)
    _write_outputs(output_dir, args.config, train_rows, val_rows, device)
    print(json.dumps({"output_dir": str(output_dir), "train_rows": len(train_rows), "val_rows": len(val_rows)}, indent=2))


if __name__ == "__main__":
    main()
