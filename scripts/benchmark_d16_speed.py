"""Benchmark D16 runtime knobs without claiming model quality."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SPECS = [
    ("face_b8_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 8, 2),
    ("face_b16_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 16, 2),
    ("face_b24_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 24, 2),
    ("face_b32_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 32, 2),
    ("face_b32_w4", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 32, 4),
    ("face_b48_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 48, 2),
    ("face_b64_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 64, 2),
    ("face_b64_w4", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 64, 4),
    ("face_b96_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 96, 2),
    ("face_b128_w2", "configs/d16/d16_v0_face_plus_context_ce_full.yaml", 128, 2),
    ("fullmask_b8_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 8, 2),
    ("fullmask_b16_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 16, 2),
    ("fullmask_b24_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 24, 2),
    ("fullmask_b32_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 32, 2),
    ("fullmask_b32_w4", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 32, 4),
    ("fullmask_b48_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 48, 2),
    ("fullmask_b64_w2", "configs/d16/d16_v0_full_with_mask_ce_full.yaml", 64, 2),
]


def _run(cmd: List[str], cwd: Path) -> int:
    print("$", " ".join(str(item) for item in cmd), flush=True)
    result = subprocess.run([str(item) for item in cmd], cwd=str(cwd), text=True)
    return int(result.returncode)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _avg(rows: Iterable[Dict[str, str]], key: str) -> float | None:
    vals = [_float(row.get(key)) for row in rows]
    vals = [val for val in vals if val is not None]
    return None if not vals else float(sum(vals) / len(vals))


def _max(rows: Iterable[Dict[str, str]], key: str) -> float | None:
    vals = [_float(row.get(key)) for row in rows]
    vals = [val for val in vals if val is not None]
    return None if not vals else float(max(vals))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "config",
        "run_name",
        "graph_mode",
        "batch_size",
        "num_workers",
        "status",
        "epochs",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
        "avg_epoch_time_sec",
        "avg_epoch_time_min",
        "max_memory_reserved_mb",
        "node_count_mean",
        "edge_count_mean",
        "best_val_macro_f1",
        "test_macro_f1_best",
        "test_accuracy_best",
        "last_test_macro_f1",
        "last_test_accuracy",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    baselines = {}
    for row in ok_rows:
        graph_mode = row.get("graph_mode")
        if row.get("batch_size") == 8 and row.get("num_workers") == 2:
            baselines[graph_mode] = row
    for row in ok_rows:
        baseline = baselines.get(row.get("graph_mode"))
        if baseline is None:
            row["macro_f1_delta_vs_b8"] = None
            row["accuracy_delta_vs_b8"] = None
            row["speedup_vs_b8"] = None
            continue
        row["macro_f1_delta_vs_b8"] = None
        row["accuracy_delta_vs_b8"] = None
        if row.get("test_macro_f1_best") is not None and baseline.get("test_macro_f1_best") is not None:
            row["macro_f1_delta_vs_b8"] = float(row["test_macro_f1_best"]) - float(baseline["test_macro_f1_best"])
        if row.get("test_accuracy_best") is not None and baseline.get("test_accuracy_best") is not None:
            row["accuracy_delta_vs_b8"] = float(row["test_accuracy_best"]) - float(baseline["test_accuracy_best"])
        if row.get("avg_epoch_time_sec") and baseline.get("avg_epoch_time_sec"):
            row["speedup_vs_b8"] = float(baseline["avg_epoch_time_sec"]) / float(row["avg_epoch_time_sec"])
        else:
            row["speedup_vs_b8"] = None
    best_speed = sorted(ok_rows, key=lambda row: row.get("avg_epoch_time_sec") or 10**12)
    stable_rows = [
        row
        for row in ok_rows
        if row.get("macro_f1_delta_vs_b8") is None or row.get("macro_f1_delta_vs_b8") >= -0.02
    ]
    best_stable = sorted(stable_rows, key=lambda row: row.get("avg_epoch_time_sec") or 10**12)
    best_memory = sorted(ok_rows, key=lambda row: row.get("max_memory_reserved_mb") or 10**12)
    lines = [
        "# D16 Speed Benchmark Report",
        "",
        "This is a runtime benchmark only. It does not claim model quality, motif evidence, semantic regions, or causal evidence.",
        "",
        "## Results",
        "| name | graph_mode | batch | workers | status | epoch_min | speedup_vs_b8 | memory_mb | test_macro_f1 | delta_macro_vs_b8 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {graph_mode} | {batch_size} | {num_workers} | {status} | {epoch_min} | {speedup} | {memory} | {macro} | {delta} |".format(
                name=row.get("name"),
                graph_mode=row.get("graph_mode"),
                batch_size=row.get("batch_size"),
                num_workers=row.get("num_workers"),
                status=row.get("status"),
                epoch_min="" if row.get("avg_epoch_time_min") is None else f"{row.get('avg_epoch_time_min'):.3f}",
                speedup="" if row.get("speedup_vs_b8") is None else f"{row.get('speedup_vs_b8'):.2f}x",
                memory="" if row.get("max_memory_reserved_mb") is None else f"{row.get('max_memory_reserved_mb'):.1f}",
                macro="" if row.get("test_macro_f1_best") is None else f"{float(row.get('test_macro_f1_best')):.4f}",
                delta="" if row.get("macro_f1_delta_vs_b8") is None else f"{row.get('macro_f1_delta_vs_b8'):.4f}",
            )
        )
    lines.extend(["", "## Recommendation"])
    if best_speed:
        row = best_speed[0]
        lines.append(
            f"- fastest measured config: `{row['name']}` with batch_size={row['batch_size']}, num_workers={row['num_workers']}, avg_epoch_time_min={row['avg_epoch_time_min']:.3f}"
        )
    if best_stable:
        row = best_stable[0]
        lines.append(
            f"- fastest config within the short-probe macro-F1 guard: `{row['name']}` with delta_macro_vs_b8={row.get('macro_f1_delta_vs_b8')}"
        )
    if best_memory:
        row = best_memory[0]
        lines.append(
            f"- lowest measured memory config: `{row['name']}` with max_memory_reserved_mb={row['max_memory_reserved_mb']:.1f}"
        )
    lines.append("- Quality guard is only a short-probe filter. For final choice, prefer the fastest config that leaves comfortable GPU memory headroom and does not show clear metric degradation versus the b8/w2 baseline.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_spec(text: str) -> tuple[str, str, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("spec must be name,config,batch_size,num_workers")
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_speed_benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--max_val_samples", type=int, default=512)
    parser.add_argument("--max_test_samples", type=int, default=512)
    parser.add_argument("--spec", action="append", type=_parse_spec, help="name,config,batch_size,num_workers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = args.spec or DEFAULT_SPECS
    rows: List[Dict[str, Any]] = []

    for name, config, batch_size, num_workers in specs:
        config_path = Path(config)
        run_dir = output_dir / "runs" / name
        cmd = [
            sys.executable,
            "d16/training/train_d16.py",
            "--config",
            str(config_path),
            "--prior_dir",
            str(args.prior_dir),
            "--output_dir",
            str(run_dir),
            "--device",
            str(args.device),
            "--max_epochs",
            str(args.epochs),
            "--batch_size",
            str(batch_size),
            "--num_workers",
            str(num_workers),
            "--max_train_samples",
            str(args.max_train_samples),
            "--max_val_samples",
            str(args.max_val_samples),
            "--max_test_samples",
            str(args.max_test_samples),
        ]
        status = "ok" if _run(cmd, PROJECT_ROOT) == 0 else "failed"
        train_rows = _read_csv_rows(run_dir / "train_log.csv")
        summary = _read_json(run_dir / "d16_train_summary.json")
        resolved = _read_json(run_dir / "resolved_config.json")
        graph_mode = ((resolved.get("graph") or {}).get("graph_mode") or (resolved.get("data") or {}).get("graph_mode"))
        epoch_time = _avg(train_rows, "epoch_time_sec")
        row = {
            "name": name,
            "config": str(config_path),
            "run_name": resolved.get("run_name", config_path.stem),
            "graph_mode": graph_mode,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "status": status,
            "epochs": args.epochs,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "avg_epoch_time_sec": epoch_time,
            "avg_epoch_time_min": None if epoch_time is None else epoch_time / 60.0,
            "max_memory_reserved_mb": _max(train_rows, "memory_reserved_mb"),
            "node_count_mean": _avg(train_rows, "node_count_mean"),
            "edge_count_mean": _avg(train_rows, "edge_count_mean"),
            "best_val_macro_f1": summary.get("best_val_macro_f1"),
            "test_macro_f1_best": summary.get("test_macro_f1"),
            "test_accuracy_best": summary.get("test_accuracy"),
            "last_test_macro_f1": summary.get("last_test_macro_f1"),
            "last_test_accuracy": summary.get("last_test_accuracy"),
            "output_dir": str(run_dir),
        }
        rows.append(row)
        _write_csv(output_dir / "d16_speed_benchmark.csv", rows)
        _write_report(output_dir / "D16_SPEED_BENCHMARK_REPORT.md", rows)

    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
