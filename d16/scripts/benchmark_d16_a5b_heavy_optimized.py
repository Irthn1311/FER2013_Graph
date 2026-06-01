"""Benchmark optimized D16R-A5b heavy implementation without full training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.scripts.profile_d16_a5b_heavy_step import profile


ORIGINAL_KAGGLE_BATCH_MS = 334.86507721113935
ORIGINAL_KAGGLE_TRAIN_EPOCH_SEC = 676.1465993239999
ORIGINAL_KAGGLE_VAL_EPOCH_SEC = 48.52240916300025


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decision(total_batch_ms: float) -> str:
    if total_batch_ms <= 180.0:
        return "HEAVY_OPT_SPEED_OK_FULL_RUN"
    if total_batch_ms <= 230.0:
        return "HEAVY_OPT_BORDERLINE_BUT_USABLE"
    if total_batch_ms > 250.0:
        return "HEAVY_OPT_STILL_TOO_SLOW_CONSIDER_LITE"
    return "HEAVY_OPT_BORDERLINE_MEASURE_MORE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_warmup_batches", type=int, default=5)
    parser.add_argument("--num_benchmark_batches", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = profile(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=output_dir,
        device_name=str(args.device),
        num_warmup_batches=int(args.num_warmup_batches),
        num_profile_batches=int(args.num_benchmark_batches),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp_override=str(args.amp),
    )
    source_csv = output_dir / "a5b_heavy_profile.csv"
    bench_csv = output_dir / "a5b_heavy_optimized_benchmark.csv"
    if source_csv.exists():
        shutil.copyfile(source_csv, bench_csv)
    rows = _read_csv(bench_csv)

    total_batch_ms = float(summary["total_batch_ms_mean"])
    estimated_train_epoch_sec = float(summary["estimated_train_epoch_sec"])
    estimated_total_epoch_sec = estimated_train_epoch_sec + ORIGINAL_KAGGLE_VAL_EPOCH_SEC
    speedup = ORIGINAL_KAGGLE_BATCH_MS / total_batch_ms if total_batch_ms > 0 else float("nan")
    speedup_pct = (speedup - 1.0) * 100.0
    optimized_summary = {
        **summary,
        "original_kaggle_total_batch_ms": ORIGINAL_KAGGLE_BATCH_MS,
        "original_kaggle_train_epoch_sec": ORIGINAL_KAGGLE_TRAIN_EPOCH_SEC,
        "original_kaggle_val_epoch_sec": ORIGINAL_KAGGLE_VAL_EPOCH_SEC,
        "estimated_total_epoch_sec_with_val": estimated_total_epoch_sec,
        "speedup_vs_original_334ms": speedup,
        "speedup_percent_vs_original_334ms": speedup_pct,
        "decision": _decision(total_batch_ms),
    }
    (output_dir / "a5b_heavy_optimized_benchmark_summary.json").write_text(
        json.dumps(optimized_summary, indent=2),
        encoding="utf-8",
    )
    # Add a one-row decision CSV for easy Kaggle scanning.
    _write_csv(output_dir / "a5b_heavy_optimized_benchmark_decision.csv", [optimized_summary])
    lines = [
        "# A5b Heavy Optimized Benchmark",
        "",
        f"- decision: `{optimized_summary['decision']}`",
        f"- amp: `{optimized_summary['amp']}`",
        f"- total_batch_ms_mean: `{total_batch_ms:.3f}`",
        f"- forward_ms_mean: `{float(summary['forward_ms_mean']):.3f}`",
        f"- backward_ms_mean: `{float(summary['backward_ms_mean']):.3f}`",
        f"- dataloader_wait_ms_mean: `{float(summary['dataloader_wait_ms_mean']):.3f}`",
        f"- estimated_train_epoch_sec: `{estimated_train_epoch_sec:.3f}`",
        f"- estimated_total_epoch_sec_with_val: `{estimated_total_epoch_sec:.3f}`",
        f"- original_kaggle_total_batch_ms: `{ORIGINAL_KAGGLE_BATCH_MS:.3f}`",
        f"- speedup_vs_original_334ms: `{speedup:.3f}x`",
        f"- speedup_percent_vs_original_334ms: `{speedup_pct:.2f}%`",
        f"- rows: `{len(rows)}`",
        "",
        "Full run gate: run A5b-heavy only if the decision is `HEAVY_OPT_SPEED_OK_FULL_RUN` or the user explicitly approves.",
    ]
    (output_dir / "A5B_HEAVY_OPTIMIZED_BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(optimized_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
