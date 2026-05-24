"""Collect D16 v1 results and compare against D15 plus D16 v0 controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").iloc[-1])


def _weighted_f1(per_class: pd.DataFrame) -> float:
    if per_class.empty or not {"support", "f1"}.issubset(per_class.columns):
        return float("nan")
    support = pd.to_numeric(per_class["support"], errors="coerce").fillna(0)
    f1 = pd.to_numeric(per_class["f1"], errors="coerce").fillna(0)
    return float((support * f1).sum() / max(float(support.sum()), 1.0))


def _run_row(path: Path, name: str, kind: str) -> Dict[str, Any]:
    train = _read_csv(path / "train_log.csv")
    test = _read_csv(path / "test_metrics.csv")
    per = _read_csv(path / "per_class_metrics.csv")
    fallback = _read_csv(path / "detected_vs_fallback_metrics.csv")
    check = _read_json(path / "d16_v1_check_summary.json")
    detected = fallback[fallback["group"] == "detected"] if not fallback.empty and "group" in fallback else pd.DataFrame()
    fb = fallback[fallback["group"] == "fallback"] if not fallback.empty and "group" in fallback else pd.DataFrame()
    return {
        "run_name": name,
        "kind": kind,
        "output_dir": str(path),
        "epoch_count": int(train["epoch"].nunique()) if not train.empty and "epoch" in train else 0,
        "best_val_macro_f1": float(pd.to_numeric(train.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce").max()) if not train.empty else float("nan"),
        "test_accuracy": _metric(test, "accuracy"),
        "test_macro_f1": _metric(test, "macro_f1"),
        "test_weighted_f1": _weighted_f1(per),
        "detected_macro_f1": _metric(detected, "macro_f1"),
        "fallback_macro_f1": _metric(fb, "macro_f1"),
        "fallback_accuracy": _metric(fb, "accuracy"),
        "supcon_loss_total_final": _metric(train, "supcon_loss_total"),
        "lambda_part_supcon_final": _metric(train, "lambda_part_supcon_current"),
        "checker_decision": check.get("decision", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v1_results")
    parser.add_argument("--d15_baseline_acc", type=float, default=0.645026)
    parser.add_argument("--d15_baseline_macro_f1", type=float, default=0.622471)
    parser.add_argument("--d15_baseline_weighted_f1", type=float, default=0.641866)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_row(Path(path), name, "d16_v1")
        for path, name in zip(args.runs, args.names)
    ]
    controls = [
        {
            "run_name": "D15_m8_basic",
            "kind": "baseline",
            "output_dir": "",
            "epoch_count": "",
            "best_val_macro_f1": "",
            "test_accuracy": args.d15_baseline_acc,
            "test_macro_f1": args.d15_baseline_macro_f1,
            "test_weighted_f1": args.d15_baseline_weighted_f1,
            "detected_macro_f1": "",
            "fallback_macro_f1": "",
            "fallback_accuracy": "",
            "supcon_loss_total_final": "",
            "lambda_part_supcon_final": "",
            "checker_decision": "baseline",
        },
        _run_row(Path("outputs/d16_v0_face_plus_context_ce_full"), "d16_v0_face_plus_context_ce", "d16_v0_control"),
        _run_row(Path("outputs/d16_v0_full_with_mask_ce_full"), "d16_v0_full_with_mask_ce", "d16_v0_control"),
    ]
    all_rows = controls + rows
    fields = [
        "run_name",
        "kind",
        "output_dir",
        "epoch_count",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "detected_macro_f1",
        "fallback_macro_f1",
        "fallback_accuracy",
        "supcon_loss_total_final",
        "lambda_part_supcon_final",
        "checker_decision",
    ]
    _write_csv(out / "d16_v1_summary.csv", all_rows, fields)
    per_rows: List[Dict[str, Any]] = []
    fallback_rows: List[Dict[str, Any]] = []
    supcon_rows: List[Dict[str, Any]] = []
    for path, name in zip(args.runs, args.names):
        per = _read_csv(Path(path) / "per_class_metrics.csv")
        for row in per.to_dict("records"):
            row["run_name"] = name
            per_rows.append(row)
        fb = _read_csv(Path(path) / "detected_vs_fallback_metrics.csv")
        for row in fb.to_dict("records"):
            row["run_name"] = name
            fallback_rows.append(row)
        train = _read_csv(Path(path) / "train_log.csv")
        for _, row in train.iterrows():
            supcon_rows.append({
                "run_name": name,
                "epoch": row.get("epoch"),
                "supcon_loss_total": row.get("supcon_loss_total"),
                "lambda_part_supcon_current": row.get("lambda_part_supcon_current"),
                "supcon_valid_pairs": row.get("supcon_valid_pairs"),
                "supcon_no_positive_pairs": row.get("supcon_no_positive_pairs"),
            })
    _write_csv(out / "d16_v1_per_class.csv", per_rows, sorted(set().union(*(row.keys() for row in per_rows))) if per_rows else ["run_name"])
    _write_csv(out / "d16_v1_detected_fallback.csv", fallback_rows, sorted(set().union(*(row.keys() for row in fallback_rows))) if fallback_rows else ["run_name"])
    _write_csv(out / "d16_v1_supcon_stats.csv", supcon_rows, ["run_name", "epoch", "supcon_loss_total", "lambda_part_supcon_current", "supcon_valid_pairs", "supcon_no_positive_pairs"])
    best = max(rows, key=lambda row: float(row["test_macro_f1"]) if str(row["test_macro_f1"]) != "nan" else -1.0, default=None)
    report = [
        "# D16 V1 Results Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## Baselines",
        f"- D15 macro-F1: {args.d15_baseline_macro_f1:.6f}, accuracy: {args.d15_baseline_acc:.6f}",
        "- D16 v0 controls are included in `d16_v1_summary.csv` when artifacts are present.",
        "",
        "## Best V1 Run",
        f"- {best['run_name'] if best else 'none'}",
        "",
        "## Success Gates",
        "- Beats D15 if macro-F1 > 0.622471 and accuracy >= 0.640 with no hard collapse.",
        "- Better than v0 if macro-F1 > 0.615703, fallback macro-F1 improves by >= 0.05 with limited overall drop, or hard classes improve clearly.",
    ]
    (out / "D16_V1_RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "rows": len(rows), "best_run": best["run_name"] if best else None}, indent=2))


if __name__ == "__main__":
    main()
