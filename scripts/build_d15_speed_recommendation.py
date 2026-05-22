"""Build the final D15 speedfix recommendation report from benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List

BASELINE_EPOCH_MIN = 24.0


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}f}"


def _find_sweep(input_dir: Path) -> Path | None:
    direct = input_dir / "speedfix_sweep" / "d15_speedfix_sweep.csv"
    if direct.exists():
        return direct
    matches = sorted(input_dir.rglob("d15_speedfix_sweep.csv"))
    return matches[0] if matches else None


def _decision(best: Dict[str, Any] | None) -> str:
    if not best:
        return "D15_SPEED_BLOCKED_NEEDS_DATA_PIPELINE_REWRITE"
    epoch_min = _float(best, "avg_total_epoch_time_sec") / 60.0
    if epoch_min <= 12.0:
        return "D15_SPEED_OK_RUN_150_WITH_CHUNKAWARE"
    if epoch_min <= 18.0:
        return "D15_SPEED_OK_RUN_150_WITH_CHUNKAWARE"
    return "D15_SPEED_STILL_TOO_SLOW_USE_SCREENING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/d15_speed_debug")
    parser.add_argument("--output_dir", default="outputs/d15_speed_debug")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = _find_sweep(input_dir)
    rows = _read_rows(sweep_path) if sweep_path else []
    ok_rows = [r for r in rows if r.get("status") == "OK" and _float(r, "avg_total_epoch_time_sec") > 0]
    best = min(ok_rows, key=lambda r: _float(r, "avg_total_epoch_time_sec")) if ok_rows else None
    compile_decision = "DO_NOT_USE_TORCH_COMPILE"
    final_decision = _decision(best)

    lines = [
        "# D15 Speedfix Final Report",
        "",
        "## 1. Problem",
        "Original D15 m8_basic:",
        "- train chunks=58",
        "- chunk_cache_size=4",
        "- chunk_aware_sampler=False",
        "- batch_size=16",
        "- num_workers=2",
        "- pin_memory=False",
        "- persistent_workers=False",
        "- epoch time ~= 24 min",
        "",
        "## 2. Root Cause Analysis",
        "- Cache thrashing is likely when `chunk_cache_size=4` is combined with global sample shuffle across 58 train chunks.",
        "- Batch size 16 creates about 1795 train batches per epoch, so Python/DataLoader overhead is paid many times.",
        "- `pin_memory=False` and `persistent_workers=False` leave useful CUDA input-pipeline settings off for Kaggle.",
        "- Code inspection found D15 builds train/val/test loaders once in `run_train`; final test runs only after training. The duplicate log symptom is consistent with notebook/stdout replay or duplicated execution, not a second in-code `setup_data()` path.",
        "- The speedfix accepts both `chunk_aware_sampler` and the older `chunk_aware_shuffle` alias and logs the resolved sampler settings.",
        "",
        "## 3. Speedfix Results",
        "| config | batch_size | workers | cache_size | chunk_aware | epoch_time_min | speedup | memory | status |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        epoch_min = _float(row, "avg_total_epoch_time_sec") / 60.0
        lines.append(
            f"| {row.get('config_name', '-')} | {row.get('batch_size', '-')} | {row.get('num_workers', '-')} | "
            f"{row.get('chunk_cache_size', '-')} | {row.get('chunk_aware_sampler', '-')} | "
            f"{_fmt(epoch_min)} | {_fmt(row.get('speedup_factor'))} | {_fmt(row.get('max_memory_reserved_gb'))} GB | {row.get('status', '-')} |"
        )
    if not rows:
        lines.append("| no sweep results found | - | - | - | - | - | - | - | FAIL |")

    lines.extend(["", "## 4. Recommended D15 Runtime Config"])
    if best:
        lines.extend(
            [
                "```yaml",
                f"batch_size: {best.get('batch_size')}",
                f"num_workers: {best.get('num_workers')}",
                f"pin_memory: {str(best.get('pin_memory')).lower()}",
                f"persistent_workers: {str(best.get('persistent_workers')).lower()}",
                f"prefetch_factor: {best.get('prefetch_factor')}",
                f"chunk_cache_size: {best.get('chunk_cache_size')}",
                "chunk_aware_sampler: true",
                "shuffle_chunks: true",
                "shuffle_within_chunk: true",
                "```",
            ]
        )
    else:
        lines.append("No successful speedfix run is available yet.")

    lines.extend(["", "## 5. Expected 150 Epoch Runtime"])
    if best:
        epoch_min = _float(best, "avg_total_epoch_time_sec") / 60.0
        lines.append(f"- expected_runtime_hours: {_fmt(epoch_min * 150.0 / 60.0)}")
    else:
        lines.append("- expected_runtime_hours: unknown")

    lines.extend(
        [
            "",
            "## 6. torch.compile Decision",
            compile_decision,
            "",
            "Torch compile remains optional after the eager data-pipeline sweep. Use it only if a separate compile sanity benchmark shows >=1.10 speedup, no NaN, small logit/loss diffs, and acceptable warmup overhead.",
            "",
            "## 7. Final Decision",
            final_decision,
            "",
        ]
    )
    out = output_dir / "D15_SPEEDFIX_FINAL_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
