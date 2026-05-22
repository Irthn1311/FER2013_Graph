"""Run D15 augmentation speed overhead benchmark."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    "configs/d15_speed/d15_m8_basic_speedfix_chunkaware_b32_w2_cache8_no_aug.yaml",
    "configs/d15_speed/d15_m8_basic_speedfix_chunkaware_b32_w2_cache8_standard_aug.yaml",
    "configs/d15_speed/d15_m8_basic_speedfix_chunkaware_b32_w2_cache8_strong_aug.yaml",
]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _aug_name(config_name: str) -> str:
    if "no_aug" in config_name:
        return "no_aug"
    if "strong_aug" in config_name:
        return "strong_aug"
    if "standard_aug" in config_name:
        return "standard_aug"
    return "configured_aug"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output_dir", default="outputs/d15_speed_debug/augmentation_overhead")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sweep_dir = output_dir / "speed_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/benchmark_d15_speedfix.py",
        "--configs",
        *args.configs,
        "--output_dir",
        str(sweep_dir),
        "--device",
        args.device,
    ]
    if args.environment:
        cmd += ["--environment", args.environment]
    if args.graph_repo_path:
        cmd += ["--graph_repo_path", args.graph_repo_path]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    rows = _read_csv(sweep_dir / "d15_speedfix_sweep.csv")
    no_aug = next((r for r in rows if _aug_name(r.get("config_name", "")) == "no_aug" and r.get("status") == "OK"), None)
    no_aug_sec = _float(no_aug or {}, "avg_total_epoch_time_sec")
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        epoch_sec = _float(row, "avg_total_epoch_time_sec")
        slowdown = (epoch_sec / no_aug_sec - 1.0) if no_aug_sec > 0 and epoch_sec > 0 else 0.0
        aug_name = _aug_name(row.get("config_name", ""))
        out_rows.append(
            {
                "augmentation": aug_name,
                "config_name": row.get("config_name"),
                "status": row.get("status"),
                "epoch_time_min": epoch_sec / 60.0 if epoch_sec > 0 else "",
                "avg_step_time_ms": row.get("avg_batch_time_ms", ""),
                "augmentation_time_ms": "",
                "edge_attr_recompute_count": "enabled_per_train_batch" if aug_name != "no_aug" else 0,
                "slowdown_vs_no_aug": slowdown,
                "max_memory_reserved_gb": row.get("max_memory_reserved_gb", ""),
            }
        )
    _write_csv(output_dir / "augmentation_overhead.csv", out_rows)
    strong = next((r for r in out_rows if r["augmentation"] == "strong_aug"), None)
    standard = next((r for r in out_rows if r["augmentation"] == "standard_aug"), None)
    strong_slow = _float(strong or {}, "slowdown_vs_no_aug")
    standard_slow = _float(standard or {}, "slowdown_vs_no_aug")
    decision = "USE_STANDARD_AUG"
    if standard_slow > 0.10:
        decision = "USE_NO_AUG_OR_LIGHT_AUG"
    if strong_slow > 0.30:
        strong_decision = "DO_NOT_USE_STRONG_AUG_FOR_FIRST_150"
    else:
        strong_decision = "STRONG_AUG_SPEED_OK"
    lines = [
        "# D15 Augmentation Overhead Report",
        "",
        "| augmentation | epoch_min | slowdown_vs_no_aug | memory_gb | status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in out_rows:
        lines.append(
            f"| {row['augmentation']} | {row['epoch_time_min']} | {row['slowdown_vs_no_aug']} | "
            f"{row['max_memory_reserved_gb']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            f"- standard_aug_decision: {decision}",
            f"- strong_aug_decision: {strong_decision}",
        ]
    )
    (output_dir / "augmentation_overhead_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output_dir": str(output_dir), "decision": decision, "strong_decision": strong_decision})


if __name__ == "__main__":
    main()
