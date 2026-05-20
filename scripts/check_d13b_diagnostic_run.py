"""Check a D13B diagnostic run for training, prediction, and slot collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite_frame(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all()) if not numeric.empty else True


def _last_or_mean(df: pd.DataFrame, col: str, split: str | None = None) -> Dict[str, float]:
    if df.empty or col not in df:
        return {f"{col}_mean": float("nan"), f"{col}_last": float("nan"), f"{col}_min": float("nan")}
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    if vals.empty:
        return {f"{col}_mean": float("nan"), f"{col}_last": float("nan"), f"{col}_min": float("nan")}
    return {f"{col}_mean": float(vals.mean()), f"{col}_last": float(vals.iloc[-1]), f"{col}_min": float(vals.min())}


def _pred_summary(pred: pd.DataFrame) -> Dict[str, float]:
    if pred.empty:
        return {"pred_max_ratio": float("nan")}
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    row = work.iloc[-1].to_dict()
    if "pred_max_ratio" in row:
        return {"pred_max_ratio": float(row["pred_max_ratio"])}
    count_cols = [c for c in work.columns if c.startswith("pred_count_")]
    counts = [float(row.get(c, 0.0)) for c in count_cols]
    total = sum(counts)
    return {"pred_max_ratio": float(max(counts) / max(total, 1.0)) if counts else float("nan")}


def check_run(output_dir: Path) -> Dict[str, Any]:
    train = _read_csv(output_dir / "train_log.csv")
    val = _read_csv(output_dir / "val_metrics.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    slots = _read_csv(output_dir / "slot_stats.csv")
    pooling = _read_csv(output_dir / "pooling_stats.csv")
    checkpoint_exists = (output_dir / "checkpoints" / "best.pt").exists()
    report_exists = (output_dir / "d13b_report.md").exists()
    warnings: List[str] = []
    failures: List[str] = []

    if train.empty:
        failures.append("missing train_log.csv")
    if val.empty:
        failures.append("missing val_metrics.csv")
    if slots.empty:
        failures.append("missing slot_stats.csv")
    if not checkpoint_exists:
        failures.append("missing best checkpoint")
    if not _finite_frame(train) or not _finite_frame(val) or not _finite_frame(slots):
        failures.append("non-finite numeric values found")

    best_val = float("nan")
    best_epoch = -1
    if not val.empty:
        metric_col = "val_macro_f1" if "val_macro_f1" in val else "macro_f1"
        if metric_col in val:
            idx = pd.to_numeric(val[metric_col], errors="coerce").idxmax()
            best_val = float(val.loc[idx, metric_col])
            best_epoch = int(val.loc[idx, "epoch"]) if "epoch" in val else int(idx)
    test_macro = float("nan")
    test_acc = float("nan")
    if not test.empty:
        row = test.iloc[-1]
        test_macro = float(row.get("test_macro_f1", row.get("macro_f1", np.nan)))
        test_acc = float(row.get("test_accuracy", row.get("accuracy", np.nan)))

    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "epochs_found": int(train["epoch"].nunique()) if "epoch" in train else int(len(train)),
        "best_val_macro_f1": best_val,
        "test_macro_f1": test_macro,
        "test_accuracy": test_acc,
        "best_epoch": best_epoch,
        "checkpoint_exists": checkpoint_exists,
        "report_exists": report_exists,
    }
    summary.update(_pred_summary(pred))
    for col in ["effective_slots", "slot_overlap", "slot_entropy", "slot_dominance"]:
        summary.update(_last_or_mean(slots, col, split="test"))
    for col in ["effective_regions", "empty_region_ratio"]:
        summary.update(_last_or_mean(pooling, col, split="test"))

    num_slots = None
    init_json = output_dir / "resolved_config.yaml"
    # Keep checker dependency-free from YAML; infer from effective slot threshold if config is absent.
    if np.isfinite(summary.get("effective_slots_last", np.nan)):
        num_slots = 16 if summary["effective_slots_last"] > 12 else 8
    threshold_slots = 0.5 * float(num_slots or 8)

    if summary.get("pred_max_ratio", 0.0) >= 0.9:
        failures.append("prediction hard collapse")
    elif summary.get("pred_max_ratio", 0.0) >= 0.75:
        warnings.append("prediction distribution is biased")
    if summary.get("effective_slots_last", 999.0) < threshold_slots:
        failures.append("effective slots below half of configured slots")
    elif summary.get("effective_slots_last", 999.0) < threshold_slots + 1.0:
        warnings.append("effective slots are borderline")
    if summary.get("slot_overlap_last", 0.0) > 0.85:
        failures.append("slot overlap very high")
    elif summary.get("slot_overlap_last", 0.0) > 0.70:
        warnings.append("slot overlap high")
    if np.isfinite(best_val) and best_val < 0.35 and summary["epochs_found"] >= 45:
        failures.append("best val macro-F1 below 0.35 after near-full run")
    if np.isfinite(test_macro) and test_macro < 0.54:
        warnings.append("test macro-F1 is well below D13A K144 reference")
    if summary.get("slot_entropy_last", 0.0) > 0.90:
        warnings.append("slot entropy very high/diffuse")
    if summary.get("slot_dominance_last", 0.0) > 0.75:
        warnings.append("slot dominance high")
    if not report_exists:
        warnings.append("missing d13b_report.md")

    if failures:
        if any("non-finite" in f or "missing train" in f for f in failures):
            decision = "D13B_DIAGNOSTIC_FAIL_TRAINING"
        else:
            decision = "D13B_DIAGNOSTIC_FAIL_COLLAPSE"
    elif warnings:
        decision = "D13B_DIAGNOSTIC_WARN_REVIEW"
    else:
        decision = "D13B_DIAGNOSTIC_PASS"
    summary["warnings"] = warnings
    summary["failures"] = failures
    summary["decision"] = decision
    return summary


def _write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    warning_lines = [f"- {w}" for w in summary.get("warnings", [])] or ["- none"]
    failure_lines = [f"- {f}" for f in summary.get("failures", [])] or ["- none"]
    lines = [
        "# D13B Diagnostic Check Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- epochs_found: {summary.get('epochs_found')}",
        f"- best_val_macro_f1: {summary.get('best_val_macro_f1')}",
        f"- test_macro_f1: {summary.get('test_macro_f1')}",
        f"- pred_max_ratio: {summary.get('pred_max_ratio')}",
        f"- effective_slots_last: {summary.get('effective_slots_last')}",
        f"- slot_overlap_last: {summary.get('slot_overlap_last')}",
        f"- slot_entropy_last: {summary.get('slot_entropy_last')}",
        f"- slot_dominance_last: {summary.get('slot_dominance_last')}",
        f"- effective_regions_last: {summary.get('effective_regions_last')}",
        f"- empty_region_ratio_last: {summary.get('empty_region_ratio_last')}",
        "",
        "## Warnings",
        *warning_lines,
        "",
        "## Failures",
        *failure_lines,
        "",
        "No motif or semantic-region claim is made.",
        "",
    ]
    (output_dir / "d13b_diagnostic_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = check_run(output_dir)
    (output_dir / "d13b_diagnostic_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
