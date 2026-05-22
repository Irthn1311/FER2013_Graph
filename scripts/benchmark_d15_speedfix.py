"""Benchmark D15 speedfix configs for two train+val epochs.

This script is intentionally speed-only: it does not save checkpoints, does not
run final test evaluation, and does not make score claims.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

BASELINE_EPOCH_MIN = 24.0
B32_REFERENCE_CONFIG = "d15_m8_basic_speedfix_chunkaware_b32_w2_cache8"
B32_REFERENCE_EPOCH_SEC = 447.19666433149996


def _cfg(config: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return (config.get(section, {}) or {}).get(key, default)


def _cuda_mem(device: Any) -> Dict[str, float]:
    import torch

    if device.type != "cuda":
        return {"max_memory_allocated_gb": 0.0, "max_memory_reserved_gb": 0.0}
    gib = 1024 ** 3
    return {
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / gib,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / gib,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "OK" and float(r.get("avg_total_epoch_time_sec") or 0) > 0]
    best = min(ok_rows, key=lambda r: float(r["avg_total_epoch_time_sec"])) if ok_rows else None
    lines = [
        "# D15 Speedfix Sweep Report",
        "",
        f"- original baseline estimated: {BASELINE_EPOCH_MIN:.1f} min/epoch",
        f"- recommended config: `{best['config_name']}`" if best else "- recommended config: unavailable",
        "",
        "| config | batch | workers | cache | chunk_aware | epoch_min | speedup_vs_b32 | speedup_vs_24min | memory_gb | status |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {config_name} | {batch_size} | {num_workers} | {chunk_cache_size} | {chunk_aware_sampler} | "
            "{epoch_min} | {speedup_b32} | {speedup} | {memory} | {status} |".format(
                config_name=row.get("config_name", "-"),
                batch_size=row.get("batch_size", "-"),
                num_workers=row.get("num_workers", "-"),
                chunk_cache_size=row.get("chunk_cache_size", "-"),
                chunk_aware_sampler=row.get("chunk_aware_sampler", "-"),
                epoch_min=_format(float(row.get("avg_total_epoch_time_sec") or 0) / 60.0),
                speedup_b32=_format(row.get("speedup_vs_b32")),
                speedup=_format(row.get("speedup_vs_original_24min", row.get("speedup_factor"))),
                memory=_format(row.get("max_memory_reserved_gb")),
                status=row.get("status", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_b32_reference_sec(output_dir: Path) -> float:
    candidates = [
        output_dir / "d15_speedfix_sweep.csv",
        output_dir.parent / "speedfix_sweep" / "d15_speedfix_sweep.csv",
        PROJECT_ROOT / "outputs" / "d15_speed_debug" / "speedfix_sweep" / "d15_speedfix_sweep.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("config_name") == B32_REFERENCE_CONFIG and row.get("status") == "OK":
                    try:
                        value = float(row.get("avg_total_epoch_time_sec") or 0)
                    except ValueError:
                        value = 0.0
                    if value > 0:
                        return value
    return B32_REFERENCE_EPOCH_SEC


def _run_one(config_path: Path, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from common import apply_cli_overrides, build_dataloader, load_config, save_config
    from training.optimizer import step_scheduler
    from training.train_d14 import _apply_graph_augmentation
    from training.train_d15 import _append_cache_stats, _append_csv, _run_epoch, build_objects

    config = load_config(config_path, environment=args.environment)
    config = apply_cli_overrides(config, args)
    config_name = str(config.get("run", {}).get("config_name") or config_path.stem)
    run_dir = output_dir / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("paths", {})["resolved_output_root"] = str(run_dir)
    config.setdefault("paths", {})["output_root"] = str(run_dir)
    save_config(config, run_dir)

    training_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}
    epochs = min(int(training_cfg.get("epochs", training_cfg.get("max_epochs", 2)) or 2), int(args.epochs))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    amp = bool(training_cfg.get("amp", False))
    augmentation_cfg = dict(config.get("augmentation", {}) or {})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    model, criterion, optimizer, scheduler, device, _init_info = build_objects(config, run_dir, device_arg=args.device)

    epoch_rows = []
    for epoch in range(1, epochs + 1):
        train_metrics, _train_slot, _train_pool, _train_supcon = _run_epoch(
            model, criterion, train_loader, optimizer, device, epoch, "train",
            max_train_batches, grad_clip, amp, augmentation_cfg,
        )
        val_metrics, _val_slot, _val_pool, _val_supcon = _run_epoch(
            model, criterion, val_loader, None, device, epoch, "val",
            max_val_batches, grad_clip, amp=False,
        )
        if scheduler is not None:
            step_scheduler(scheduler, monitor_value=val_metrics.get("macro_f1"))
        _append_cache_stats(run_dir, epoch, "train", train_loader, train_metrics)
        _append_cache_stats(run_dir, epoch, "val", val_loader, val_metrics)
        row = {
            "epoch": epoch,
            "train_epoch_time_sec": float(train_metrics["seconds"]),
            "val_epoch_time_sec": float(val_metrics["seconds"]),
            "total_epoch_time_sec": float(train_metrics["seconds"] + val_metrics["seconds"]),
            "first_batch_wait_time_sec": float(train_metrics.get("first_batch_wait_time_sec", 0.0)),
            "avg_batch_time_ms": float(train_metrics.get("avg_batch_time_ms", 0.0)),
            "cache_hit_count": int(train_metrics.get("cache_hit_count", 0)),
            "cache_miss_count": int(train_metrics.get("cache_miss_count", 0)),
            "cache_hit_rate": float(train_metrics.get("cache_hit_rate", 0.0)),
        }
        epoch_rows.append(row)
        _append_csv(run_dir / "speed_epoch_log.csv", row)

    mem = _cuda_mem(device)
    avg_total = sum(float(r["total_epoch_time_sec"]) for r in epoch_rows) / max(len(epoch_rows), 1)
    avg_train = sum(float(r["train_epoch_time_sec"]) for r in epoch_rows) / max(len(epoch_rows), 1)
    avg_val = sum(float(r["val_epoch_time_sec"]) for r in epoch_rows) / max(len(epoch_rows), 1)
    speedup = (BASELINE_EPOCH_MIN * 60.0 / avg_total) if avg_total > 0 else 0.0
    b32_ref_sec = _load_b32_reference_sec(output_dir)
    speedup_vs_b32 = (b32_ref_sec / avg_total) if avg_total > 0 else 0.0
    avg_cache_hit_rate = sum(float(r["cache_hit_rate"]) for r in epoch_rows) / max(len(epoch_rows), 1)
    return {
        "config_name": config_name,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "status": "OK",
        "batch_size": int(data_cfg.get("batch_size", training_cfg.get("batch_size", 0)) or 0),
        "num_workers": int(data_cfg.get("num_workers", training_cfg.get("num_workers", 0)) or 0),
        "pin_memory": bool(data_cfg.get("pin_memory", training_cfg.get("pin_memory", False))),
        "persistent_workers": bool(data_cfg.get("persistent_workers", training_cfg.get("persistent_workers", False))),
        "prefetch_factor": data_cfg.get("prefetch_factor", training_cfg.get("prefetch_factor")),
        "chunk_cache_size": int(data_cfg.get("chunk_cache_size", 0) or 0),
        "chunk_aware_sampler": bool(data_cfg.get("chunk_aware_sampler", data_cfg.get("chunk_aware_shuffle", False))),
        "shuffle_chunks": bool(data_cfg.get("shuffle_chunks", True)),
        "shuffle_within_chunk": bool(data_cfg.get("shuffle_within_chunk", True)),
        "avg_train_epoch_time_sec": avg_train,
        "avg_val_epoch_time_sec": avg_val,
        "avg_total_epoch_time_sec": avg_total,
        "speedup_factor": speedup,
        "speedup_vs_original_24min": speedup,
        "speedup_vs_b32": speedup_vs_b32,
        "first_batch_wait_time_sec": epoch_rows[0].get("first_batch_wait_time_sec", 0.0),
        "avg_batch_time_ms": sum(float(r["avg_batch_time_ms"]) for r in epoch_rows) / max(len(epoch_rows), 1),
        "cache_hit_rate": avg_cache_hit_rate,
        **mem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2-epoch D15 speedfix benchmark sweep.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d15_speed_debug/speedfix_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for cfg in args.configs:
        config_path = Path(cfg)
        print(f"[D15 speedfix] RUN {config_path}")
        try:
            row = _run_one(config_path, output_dir, args)
        except RuntimeError as exc:
            text = str(exc)
            status = "OOM" if "out of memory" in text.lower() else "FAIL"
            row = {"config_name": config_path.stem, "config_path": str(config_path), "status": status, "error": text}
            print(f"[D15 speedfix] {status}: {text}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            row = {"config_name": config_path.stem, "config_path": str(config_path), "status": "FAIL", "error": str(exc)}
            print(f"[D15 speedfix] FAIL: {exc}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        _write_csv(output_dir / "d15_speedfix_sweep.csv", rows)
        _write_report(output_dir / "d15_speedfix_sweep_report.md", rows)

    print(f"[D15 speedfix] wrote {output_dir / 'd15_speedfix_sweep.csv'}")
    print(f"[D15 speedfix] wrote {output_dir / 'd15_speedfix_sweep_report.md'}")


if __name__ == "__main__":
    main()
