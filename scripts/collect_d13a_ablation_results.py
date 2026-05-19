"""Collect D13A ablation outputs into CSV summaries and a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
BASELINE_EDGEAWARE_K144_TEST_MACRO_F1 = 0.5343026827428065


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _best_val(val_df: pd.DataFrame | None) -> Dict[str, Any]:
    if val_df is None or "val_macro_f1" not in val_df:
        return {"best_epoch": None, "best_val_macro_f1": None, "best_val_accuracy": None}
    idx = pd.to_numeric(val_df["val_macro_f1"], errors="coerce").idxmax()
    row = val_df.loc[idx]
    return {
        "best_epoch": int(row.get("epoch")) if not pd.isna(row.get("epoch")) else None,
        "best_val_macro_f1": _float(row.get("val_macro_f1")),
        "best_val_accuracy": _float(row.get("val_accuracy")),
    }


def _runtime_minutes(train_df: pd.DataFrame | None) -> float | None:
    if train_df is None:
        return None
    total = 0.0
    found = False
    for col in train_df.columns:
        if col.endswith("_seconds") or col in {"train_seconds", "val_seconds"}:
            values = pd.to_numeric(train_df[col], errors="coerce").dropna()
            if not values.empty:
                total += float(values.sum())
                found = True
    return total / 60.0 if found else None


def _pooling_row(pool_df: pd.DataFrame | None) -> Dict[str, Any]:
    empty = {
        "effective_regions_mean_test_or_last": None,
        "empty_region_ratio_mean_test_or_last": None,
        "assignment_entropy_mean_test_or_last": None,
        "assignment_temperature": None,
    }
    if pool_df is None:
        return empty
    sub = pool_df[pool_df["split"] == "test"] if "split" in pool_df else pool_df
    if sub.empty:
        sub = pool_df[pool_df["split"] == "val"] if "split" in pool_df else pool_df
    if sub.empty:
        sub = pool_df
    out = dict(empty)
    if "effective_regions" in sub:
        out["effective_regions_mean_test_or_last"] = _float(pd.to_numeric(sub["effective_regions"], errors="coerce").mean())
    if "empty_region_ratio" in sub:
        out["empty_region_ratio_mean_test_or_last"] = _float(pd.to_numeric(sub["empty_region_ratio"], errors="coerce").mean())
    if "assignment_entropy" in sub:
        out["assignment_entropy_mean_test_or_last"] = _float(pd.to_numeric(sub["assignment_entropy"], errors="coerce").mean())
    if "assignment_temperature" in sub:
        out["assignment_temperature"] = _float(pd.to_numeric(sub["assignment_temperature"], errors="coerce").mean())
    return out


def _pred_row(pred_df: pd.DataFrame | None) -> Dict[str, Any]:
    out = {"max_pred_class": None, "max_pred_ratio": None, "pred_total": None}
    if pred_df is None:
        return out
    sub = pred_df[pred_df["split"] == "test"] if "split" in pred_df else pred_df
    if sub.empty:
        sub = pred_df[pred_df["split"] == "val"] if "split" in pred_df else pred_df
    if sub.empty:
        return out
    row = sub.iloc[-1]
    counts = {}
    for idx, name in enumerate(EMOTION_NAMES):
        key = f"pred_count_{idx}_{name}"
        counts[name] = int(float(row.get(key, 0)))
    total = sum(counts.values())
    if total <= 0:
        return out
    max_class = max(counts, key=counts.get)
    out.update(
        {
            "max_pred_class": max_class,
            "max_pred_ratio": counts[max_class] / total,
            "pred_total": total,
        }
    )
    for name, count in counts.items():
        out[f"pred_{name}"] = count
    return out


def _recommendation(row: Dict[str, Any], baseline_macro_f1: float) -> str:
    max_pred_ratio = row.get("max_pred_ratio")
    empty_ratio = row.get("empty_region_ratio_mean_test_or_last")
    effective = row.get("effective_regions_mean_test_or_last")
    entropy = row.get("assignment_entropy_mean_test_or_last")
    test_macro = row.get("test_macro_f1")
    healthy = (
        max_pred_ratio is not None
        and max_pred_ratio <= 0.9
        and empty_ratio is not None
        and empty_ratio <= 0.4
        and effective is not None
        and effective >= 0.5 * 144
    )
    if not healthy:
        return "INVALID_COLLAPSE"
    if test_macro is not None and test_macro >= baseline_macro_f1 + 0.01:
        return "STRONG_CANDIDATE"
    if test_macro is not None and test_macro >= baseline_macro_f1 - 0.005 and ("k64" in str(row.get("run_name", ""))):
        return "EFFICIENT_CANDIDATE"
    if entropy is not None and entropy < 0.80 and test_macro is not None and test_macro >= baseline_macro_f1 - 0.005:
        return "BETTER_HARDENING_CANDIDATE"
    return "NEEDS_REVIEW"


def summarize_run(run_dir: Path, baseline_macro_f1: float) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = _read_yaml(run_dir / "resolved_config.yaml")
    train_df = _read_csv(run_dir / "train_log.csv")
    val_df = _read_csv(run_dir / "val_metrics.csv")
    test_df = _read_csv(run_dir / "test_metrics.csv")
    pool_df = _read_csv(run_dir / "pooling_stats.csv")
    pred_df = _read_csv(run_dir / "pred_count.csv")
    check = _read_json(run_dir / "d13_debug_check_summary.json")

    best = _best_val(val_df)
    last_train_loss = None
    first_train_loss = None
    loss_drop = None
    if train_df is not None and "train_loss" in train_df:
        values = pd.to_numeric(train_df["train_loss"], errors="coerce").dropna()
        if not values.empty:
            first_train_loss = float(values.iloc[0])
            last_train_loss = float(values.iloc[-1])
            loss_drop = first_train_loss - last_train_loss
    test_row = test_df.iloc[-1] if test_df is not None else {}
    pool = _pooling_row(pool_df)
    pred = _pred_row(pred_df)
    run_name = run_dir.name
    config_name = cfg.get("run", {}).get("config_name")
    config_path = cfg.get("run", {}).get("config_path") or config_name
    summary = {
        "run_name": run_name,
        "config_path": config_path,
        "checker_decision": check.get("final_decision"),
        **best,
        "test_macro_f1": _float(test_row.get("test_macro_f1")),
        "test_accuracy": _float(test_row.get("test_accuracy")),
        "test_weighted_f1": _float(test_row.get("test_weighted_f1")),
        "last_train_loss": last_train_loss,
        "train_loss_first": first_train_loss,
        "train_loss_last": last_train_loss,
        "loss_drop": loss_drop,
        **pool,
        **{k: pred.get(k) for k in ("max_pred_class", "max_pred_ratio")},
        "checkpoint_exists": (run_dir / "checkpoints" / "best.pt").exists(),
        "report_exists": (run_dir / "d13a_report.md").exists(),
        "runtime_minutes": _runtime_minutes(train_df),
    }
    summary["recommendation"] = _recommendation(summary, baseline_macro_f1)
    pooling_summary = {
        "run_name": run_name,
        **pool,
    }
    pred_summary = {
        "run_name": run_name,
        **pred,
    }
    return summary, pooling_summary, pred_summary


def collect(root_dir: str | Path, output_dir: str | Path, baseline_macro_f1: float) -> Dict[str, Any]:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in root_dir.iterdir() if p.is_dir()) if root_dir.exists() else []
    summaries: List[Dict[str, Any]] = []
    pooling_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary, pooling, pred = summarize_run(run_dir, baseline_macro_f1)
        summaries.append(summary)
        pooling_rows.append(pooling)
        pred_rows.append(pred)

    summary_df = pd.DataFrame(summaries)
    pooling_df = pd.DataFrame(pooling_rows)
    pred_df = pd.DataFrame(pred_rows)
    summary_csv = output_dir / "d13a_ablation_summary.csv"
    pooling_csv = output_dir / "d13a_ablation_pooling_summary.csv"
    pred_csv = output_dir / "d13a_ablation_pred_count_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    pooling_df.to_csv(pooling_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    report_path = output_dir / "d13a_ablation_report.md"
    final_decision = "D13A_ABLATION_NEEDS_REVIEW"
    if not summary_df.empty and "STRONG_CANDIDATE" in set(summary_df.get("recommendation", [])):
        final_decision = "D13A_FINAL_CANDIDATE_SELECTED"

    lines = [
        "# D13A Ablation Report",
        "",
        f"- root_dir: {root_dir}",
        f"- baseline_edgeaware_k144_test_macro_f1: {baseline_macro_f1:.6f}",
        f"- final_decision: {final_decision}",
        "",
        "## Ranking By Test Macro-F1",
        "```text",
        (summary_df.sort_values("test_macro_f1", ascending=False).to_string(index=False) if not summary_df.empty and "test_macro_f1" in summary_df else "No runs found."),
        "```",
        "",
        "## Ranking By Best Val Macro-F1",
        "```text",
        (summary_df.sort_values("best_val_macro_f1", ascending=False).to_string(index=False) if not summary_df.empty and "best_val_macro_f1" in summary_df else "No runs found."),
        "```",
        "",
        "## Pooling Health Comparison",
        "```text",
        (pooling_df.to_string(index=False) if not pooling_df.empty else "No pooling stats found."),
        "```",
        "",
        "## Warning And Collapse Section",
    ]
    if summary_df.empty:
        lines.append("- No run directories found.")
    else:
        for _, row in summary_df.iterrows():
            notes = []
            if row.get("max_pred_ratio") is not None and row.get("max_pred_ratio") > 0.9:
                notes.append("prediction collapse")
            if row.get("empty_region_ratio_mean_test_or_last") is not None and row.get("empty_region_ratio_mean_test_or_last") > 0.4:
                notes.append("empty region ratio high")
            if not bool(row.get("checkpoint_exists")):
                notes.append("missing best checkpoint")
            if notes:
                lines.append(f"- {row.get('run_name')}: {', '.join(notes)}")
        if lines[-1] == "## Warning And Collapse Section":
            lines.append("- none")
    lines.extend(
        [
            "",
            "## Recommendation",
            "- STRONG_CANDIDATE means test macro-F1 beats baseline by at least +0.01 with healthy pooling.",
            "- EFFICIENT_CANDIDATE means K64 is near baseline and may be cheaper.",
            "- BETTER_HARDENING_CANDIDATE means lower assignment entropy did not hurt performance while pooling stayed healthy.",
            "- Do not open D13B from this report without reviewing the actual figures and per-class behavior.",
            "",
            "No motif claim is made for D13A ablations.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_csv": str(summary_csv),
        "pooling_csv": str(pooling_csv),
        "pred_count_csv": str(pred_csv),
        "report": str(report_path),
        "num_runs": len(summaries),
        "final_decision": final_decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_macro_f1", type=float, default=BASELINE_EDGEAWARE_K144_TEST_MACRO_F1)
    args = parser.parse_args()
    result = collect(args.root_dir, args.output_dir, baseline_macro_f1=args.baseline_macro_f1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

