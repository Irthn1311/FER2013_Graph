"""Check whether a D13A debug or full run is ready for Kaggle/full follow-up."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.labels import EMOTION_NAMES

PASS = "D13_DEBUG_PASS_READY_FOR_KAGGLE"
WARN = "D13_DEBUG_WARN_REVIEW_BEFORE_KAGGLE"
FAIL = "D13_DEBUG_FAIL_DO_NOT_RUN_FULL"


def _finite_series(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return bool(np.isfinite(values.to_numpy(dtype=float)).all())


def _read_csv(path: Path, required: bool, errors: List[str], warnings: List[str]) -> pd.DataFrame | None:
    if not path.exists():
        msg = f"Missing required file: {path.name}" if required else f"Optional file missing: {path.name}"
        (errors if required else warnings).append(msg)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Could not read {path.name}: {exc}")
        return None
    if df.empty:
        errors.append(f"Empty CSV: {path.name}")
        return None
    return df


def _best_existing_checkpoint(output_dir: Path) -> Tuple[bool, str | None]:
    candidates = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "checkpoints" / "best.pth",
        output_dir / "best.pt",
        output_dir / "best.pth",
    ]
    for path in candidates:
        if path.exists():
            return True, str(path)
    found = sorted((output_dir / "checkpoints").glob("*best*")) if (output_dir / "checkpoints").exists() else []
    if found:
        return True, str(found[0])
    return False, None


def _loss_trend(train_log: pd.DataFrame, warnings: List[str], errors: List[str]) -> Tuple[float | None, float | None, str]:
    if "train_loss" not in train_log:
        errors.append("train_log.csv does not contain train_loss")
        return None, None, "missing"
    values = pd.to_numeric(train_log["train_loss"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append("train_loss contains NaN or Inf")
        return None, None, "non_finite"
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    trend = "flat"
    if last < first * 0.98:
        trend = "down"
    elif last > first * 1.25:
        trend = "up_warning"
        warnings.append(f"train_loss increased from {first:.6f} to {last:.6f}")
    return first, last, trend


def _val_summary(val_metrics: pd.DataFrame, warnings: List[str], errors: List[str]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {"best_val_macro_f1": None, "best_val_accuracy": None}
    if "val_macro_f1" not in val_metrics:
        errors.append("val_metrics.csv missing val_macro_f1")
    else:
        vals = pd.to_numeric(val_metrics["val_macro_f1"], errors="coerce")
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            errors.append("val_macro_f1 contains NaN or Inf")
        else:
            out["best_val_macro_f1"] = float(vals.max())
    if "val_accuracy" not in val_metrics:
        errors.append("val_metrics.csv missing val_accuracy")
    else:
        vals = pd.to_numeric(val_metrics["val_accuracy"], errors="coerce")
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            errors.append("val_accuracy contains NaN or Inf")
        else:
            out["best_val_accuracy"] = float(vals.max())
    return out


def _pred_count_summary(
    output_dir: Path,
    train_log: pd.DataFrame | None,
    val_metrics: pd.DataFrame | None,
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    pred_path = output_dir / "pred_count.csv"
    pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
    source = "pred_count.csv" if pred_df is not None and not pred_df.empty else "val_metrics.csv/train_log.csv"
    row = None
    if pred_df is not None and not pred_df.empty:
        val_rows = pred_df[pred_df.get("split", "") == "val"] if "split" in pred_df else pred_df
        row = val_rows.iloc[-1] if not val_rows.empty else pred_df.iloc[-1]
    elif val_metrics is not None:
        row = val_metrics.iloc[-1]
    elif train_log is not None:
        row = train_log.iloc[-1]
    else:
        errors.append("No source available for pred_count")
        return {"source": source, "counts": {}, "max_ratio": None, "collapsed_class": None}

    counts: Dict[str, int] = {}
    for idx, name in enumerate(EMOTION_NAMES):
        candidates = [
            f"pred_count_{idx}_{name}",
            f"val_pred_count_{idx}_{name}",
            f"train_pred_count_{idx}_{name}",
            f"test_pred_count_{idx}_{name}",
        ]
        value = 0
        for key in candidates:
            if key in row:
                value = int(float(row[key]))
                break
        counts[f"{idx}_{name}"] = value
    total = sum(counts.values())
    if total <= 0:
        errors.append("pred_count total is zero or unavailable")
        return {"source": source, "counts": counts, "max_ratio": None, "collapsed_class": None}
    max_key = max(counts, key=counts.get)
    max_ratio = counts[max_key] / total
    if max_ratio > 0.95:
        errors.append(f"Prediction collapse: {max_key} has {max_ratio:.3f} of predictions")
    elif max_ratio > 0.90:
        warnings.append(f"Prediction near-collapse: {max_key} has {max_ratio:.3f} of predictions")
    return {
        "source": source,
        "counts": counts,
        "total": int(total),
        "max_ratio": float(max_ratio),
        "collapsed_class": max_key if max_ratio > 0.90 else None,
    }


def _pooling_summary(pooling_stats: pd.DataFrame, k_regions: int, warnings: List[str], errors: List[str]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {
        "effective_regions_mean": None,
        "effective_regions_min": None,
        "effective_regions_max": None,
        "empty_region_ratio_mean": None,
        "empty_region_ratio_max": None,
    }
    loss_cols = [c for c in pooling_stats.columns if "loss" in c.lower()]
    for col in loss_cols:
        if not _finite_series(pooling_stats[col]):
            errors.append(f"pooling loss column {col} contains NaN or Inf")
    if "effective_regions" not in pooling_stats:
        errors.append("pooling_stats.csv missing effective_regions")
    else:
        vals = pd.to_numeric(pooling_stats["effective_regions"], errors="coerce")
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            errors.append("effective_regions contains NaN or Inf")
        else:
            out["effective_regions_mean"] = float(vals.mean())
            out["effective_regions_min"] = float(vals.min())
            out["effective_regions_max"] = float(vals.max())
            if out["effective_regions_mean"] < 0.5 * float(k_regions):
                errors.append(
                    f"effective_regions mean {out['effective_regions_mean']:.3f} < threshold {0.5 * float(k_regions):.3f}"
                )
    if "empty_region_ratio" not in pooling_stats:
        errors.append("pooling_stats.csv missing empty_region_ratio")
    else:
        vals = pd.to_numeric(pooling_stats["empty_region_ratio"], errors="coerce")
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            errors.append("empty_region_ratio contains NaN or Inf")
        else:
            out["empty_region_ratio_mean"] = float(vals.mean())
            out["empty_region_ratio_max"] = float(vals.max())
            if out["empty_region_ratio_mean"] > 0.4:
                errors.append(f"empty_region_ratio mean {out['empty_region_ratio_mean']:.3f} > 0.400")
            elif out["empty_region_ratio_max"] > 0.4:
                warnings.append(f"empty_region_ratio max {out['empty_region_ratio_max']:.3f} > 0.400")
    return out


def check_run(output_dir: str | Path, k_regions: int = 144) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    errors: List[str] = []
    warnings: List[str] = []
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")
    train_log = _read_csv(output_dir / "train_log.csv", True, errors, warnings)
    val_metrics = _read_csv(output_dir / "val_metrics.csv", True, errors, warnings)
    pooling_stats = _read_csv(output_dir / "pooling_stats.csv", True, errors, warnings)

    epochs_found = 0
    if train_log is not None and "epoch" in train_log:
        epochs_found = int(pd.to_numeric(train_log["epoch"], errors="coerce").dropna().nunique())
        if epochs_found < 3:
            errors.append(f"Only {epochs_found} epochs found; need at least 3")
    elif train_log is not None:
        errors.append("train_log.csv missing epoch column")

    first_loss, last_loss, loss_trend = (None, None, "missing")
    if train_log is not None:
        for col in [c for c in train_log.columns if "loss" in c.lower()]:
            if not _finite_series(train_log[col]):
                errors.append(f"train_log loss column {col} contains NaN or Inf")
        first_loss, last_loss, loss_trend = _loss_trend(train_log, warnings, errors)

    val_out = _val_summary(val_metrics, warnings, errors) if val_metrics is not None else {
        "best_val_macro_f1": None,
        "best_val_accuracy": None,
    }
    pred_out = _pred_count_summary(output_dir, train_log, val_metrics, warnings, errors)
    pool_out = _pooling_summary(pooling_stats, k_regions, warnings, errors) if pooling_stats is not None else {}
    checkpoint_exists, checkpoint_path = _best_existing_checkpoint(output_dir)
    if not checkpoint_exists:
        errors.append("No best checkpoint found under checkpoints/best.pt or checkpoints/best.pth")
    report_path = output_dir / "d13a_report.md"
    report_exists = report_path.exists()
    if not report_exists:
        errors.append("Missing d13a_report.md")

    decision = FAIL if errors else (WARN if warnings else PASS)
    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "epochs_found": int(epochs_found),
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "loss_trend": loss_trend,
        **val_out,
        "pred_count_summary": pred_out,
        **pool_out,
        "checkpoint_exists": bool(checkpoint_exists),
        "checkpoint_path": checkpoint_path,
        "report_exists": bool(report_exists),
        "warnings": warnings,
        "errors": errors,
        "final_decision": decision,
    }
    return summary


def write_reports(output_dir: str | Path, summary: Dict[str, Any]) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "d13_debug_check_summary.json"
    md_path = output_dir / "d13_debug_check_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "# D13 Run Check Report",
        "",
        f"- final_decision: {summary['final_decision']}",
        f"- epochs_found: {summary['epochs_found']}",
        f"- first_train_loss: {summary.get('first_train_loss')}",
        f"- last_train_loss: {summary.get('last_train_loss')}",
        f"- loss_trend: {summary.get('loss_trend')}",
        f"- best_val_macro_f1: {summary.get('best_val_macro_f1')}",
        f"- best_val_accuracy: {summary.get('best_val_accuracy')}",
        f"- effective_regions_mean: {summary.get('effective_regions_mean')}",
        f"- effective_regions_min: {summary.get('effective_regions_min')}",
        f"- effective_regions_max: {summary.get('effective_regions_max')}",
        f"- empty_region_ratio_mean: {summary.get('empty_region_ratio_mean')}",
        f"- empty_region_ratio_max: {summary.get('empty_region_ratio_max')}",
        f"- checkpoint_exists: {summary.get('checkpoint_exists')}",
        f"- report_exists: {summary.get('report_exists')}",
        "",
        "## Pred Count Summary",
        "```json",
        json.dumps(summary.get("pred_count_summary", {}), indent=2),
        "```",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {w}" for w in summary.get("warnings", [])] or ["- none"])
    lines.extend(["", "## Errors"])
    lines.extend([f"- {e}" for e in summary.get("errors", [])] or ["- none"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_regions", type=int, default=144)
    args = parser.parse_args()
    summary = check_run(args.output_dir, k_regions=args.k_regions)
    md_path, json_path = write_reports(args.output_dir, summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote: {md_path}")
    print(f"wrote: {json_path}")
    if summary["final_decision"] == FAIL:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

