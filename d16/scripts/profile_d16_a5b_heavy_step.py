"""Profile one-step throughput for D16R-A5b heavy edge-context GNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.training.train_d16 import _loader_kwargs, _weighted_ce_loss, build_dataset, load_config, resolve_device, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(rows: List[Dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row.get(key, float("nan"))))]
    return float(np.mean(values)) if values else float("nan")


def _p50(rows: List[Dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row.get(key, float("nan"))))]
    return float(np.percentile(values, 50)) if values else float("nan")


def _p90(rows: List[Dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row.get(key, float("nan"))))]
    return float(np.percentile(values, 90)) if values else float("nan")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _amp_enabled(cfg: Dict[str, Any], override: str) -> bool:
    if override == "on":
        return True
    if override == "off":
        return False
    training = cfg.get("training", {}) or {}
    return bool(training.get("amp", training.get("mixed_precision", False)))


def profile(
    config_path: Path,
    prior_dir: Path,
    output_dir: Path,
    device_name: str,
    num_warmup_batches: int,
    num_profile_batches: int,
    batch_size: int | None,
    num_workers: int | None,
    amp_override: str,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = cfg.get("seed", (cfg.get("training", {}) or {}).get("seed"))
    if seed is not None:
        set_seed(int(seed))
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    if batch_size is not None:
        training_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        training_cfg["num_workers"] = int(num_workers)
    # Keep the profiler bounded and deterministic-ish without changing the run config file.
    max_samples = int(training_cfg.get("batch_size", data_cfg.get("batch_size", 16))) * (
        int(num_warmup_batches) + int(num_profile_batches) + 2
    )
    data_cfg["max_train_samples"] = max(data_cfg.get("max_train_samples") or 0, max_samples)

    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = _amp_enabled(cfg, amp_override) and device.type == "cuda"

    ds = build_dataset(cfg, prior_dir, "train")
    loader = DataLoader(ds, **_loader_kwargs(data_cfg, training_cfg, shuffle=False))
    first_batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=first_batch.x_cat.size(1)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    loss_cfg = cfg.get("loss", {}) or {}
    rows: List[Dict[str, Any]] = []
    model.train()
    total_needed = int(num_warmup_batches) + int(num_profile_batches)
    wait_start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > total_needed:
            break
        batch_ready = time.perf_counter()
        dataloader_wait_ms = (batch_ready - wait_start) * 1000.0

        move_start = time.perf_counter()
        batch = batch.to(device)
        _sync(device)
        move_ms = (time.perf_counter() - move_start) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out = model(batch)
            loss, _ = _weighted_ce_loss(out["logits"], batch.y, batch, loss_cfg)
        _sync(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0

        backward_start = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_ms = (time.perf_counter() - backward_start) * 1000.0

        optimizer_start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        _sync(device)
        optimizer_step_ms = (time.perf_counter() - optimizer_start) * 1000.0

        batch_end = time.perf_counter()
        wait_start = batch_end
        row = {
            "batch": int(batch_idx),
            "phase": "warmup" if batch_idx <= int(num_warmup_batches) else "profile",
            "dataloader_wait_ms": float(dataloader_wait_ms),
            "batch_to_device_ms": float(move_ms),
            "forward_ms": float(forward_ms),
            "backward_ms": float(backward_ms),
            "optimizer_step_ms": float(optimizer_step_ms),
            "total_batch_ms": float((batch_end - batch_ready) * 1000.0),
            "node_count_mean": float(((batch.ptr[1:] - batch.ptr[:-1]).float().mean()).detach().cpu().item()),
            "edge_count_mean": float(batch.edge_index_cat.size(1) / max(batch.num_graphs, 1)),
            "edge_attr_dim": 0 if batch.edge_attr_cat is None else int(batch.edge_attr_cat.size(1)),
            "loss": float(loss.detach().float().cpu().item()),
            "memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else float("nan"),
            "memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024**2)) if device.type == "cuda" else float("nan"),
            "amp": bool(amp_enabled),
            "edge_attr_build_ms": float("nan"),
            "edge_mlp_ms": float("nan"),
            "aggregation_ms": float("nan"),
            "context_injection_ms": float("nan"),
            "readout_ms": float("nan"),
        }
        rows.append(row)
    profile_rows = [row for row in rows if row["phase"] == "profile"]
    if not profile_rows:
        raise RuntimeError("No profile rows collected")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "a5b_heavy_profile.csv", rows)
    summary = {
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "device": str(device),
        "amp": bool(amp_enabled),
        "num_warmup_batches": int(num_warmup_batches),
        "num_profile_batches": len(profile_rows),
        "batch_size": int(training_cfg.get("batch_size", data_cfg.get("batch_size", 16))),
        "num_workers": int(training_cfg.get("num_workers", data_cfg.get("num_workers", 0)) or 0),
        "total_batch_ms_mean": _mean(profile_rows, "total_batch_ms"),
        "total_batch_ms_p50": _p50(profile_rows, "total_batch_ms"),
        "total_batch_ms_p90": _p90(profile_rows, "total_batch_ms"),
        "dataloader_wait_ms_mean": _mean(profile_rows, "dataloader_wait_ms"),
        "forward_ms_mean": _mean(profile_rows, "forward_ms"),
        "backward_ms_mean": _mean(profile_rows, "backward_ms"),
        "optimizer_step_ms_mean": _mean(profile_rows, "optimizer_step_ms"),
        "node_count_mean": _mean(profile_rows, "node_count_mean"),
        "edge_count_mean": _mean(profile_rows, "edge_count_mean"),
        "edge_attr_dim": int(profile_rows[-1]["edge_attr_dim"]),
        "memory_allocated_mb": _mean(profile_rows, "memory_allocated_mb"),
        "memory_reserved_mb": _mean(profile_rows, "memory_reserved_mb"),
        "estimated_train_epoch_sec": _mean(profile_rows, "total_batch_ms") * 1795.0 / 1000.0,
    }
    (output_dir / "a5b_heavy_profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# A5b Heavy Step Profile",
        "",
        f"- device: `{summary['device']}`",
        f"- amp: `{summary['amp']}`",
        f"- batch_size: `{summary['batch_size']}`",
        f"- num_workers: `{summary['num_workers']}`",
        f"- profile_batches: `{summary['num_profile_batches']}`",
        f"- total_batch_ms_mean: `{summary['total_batch_ms_mean']:.3f}`",
        f"- dataloader_wait_ms_mean: `{summary['dataloader_wait_ms_mean']:.3f}`",
        f"- forward_ms_mean: `{summary['forward_ms_mean']:.3f}`",
        f"- backward_ms_mean: `{summary['backward_ms_mean']:.3f}`",
        f"- optimizer_step_ms_mean: `{summary['optimizer_step_ms_mean']:.3f}`",
        f"- node_count_mean: `{summary['node_count_mean']:.3f}`",
        f"- edge_count_mean: `{summary['edge_count_mean']:.3f}`",
        f"- edge_attr_dim: `{summary['edge_attr_dim']}`",
        f"- estimated_train_epoch_sec: `{summary['estimated_train_epoch_sec']:.3f}`",
        "",
        "Uninstrumented internal columns are written as NaN unless a lower-level timer is enabled.",
    ]
    (output_dir / "A5B_HEAVY_PROFILE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_warmup_batches", type=int, default=5)
    parser.add_argument("--num_profile_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()
    profile(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        device_name=str(args.device),
        num_warmup_batches=int(args.num_warmup_batches),
        num_profile_batches=int(args.num_profile_batches),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp_override=str(args.amp),
    )


if __name__ == "__main__":
    main()
