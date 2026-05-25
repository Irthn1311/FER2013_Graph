"""Collect D16 v3 fallback-specific dual-head results.

This collector compares performance artifacts only. It does not make motif,
semantic-region, causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


HARD_CLASS_IDS = [0, 2, 4]
D15 = {"accuracy": 0.645026, "macro_f1": 0.622471, "weighted_f1": 0.641866}
V1_BEST = {"accuracy": 0.639175, "macro_f1": 0.632938, "fallback_macro_f1": 0.409767, "hard_f1": 0.510704}
V2_BEST = {"accuracy": 0.635274, "macro_f1": 0.623511, "fallback_macro_f1": 0.456697, "hard_f1": 0.507355}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _weighted_f1(per_class: pd.DataFrame) -> float:
    if per_class.empty or not {"support", "f1"}.issubset(per_class.columns):
        return float("nan")
    support = pd.to_numeric(per_class["support"], errors="coerce").fillna(0)
    f1 = pd.to_numeric(per_class["f1"], errors="coerce").fillna(0)
    return float((support * f1).sum() / max(float(support.sum()), 1.0))


def _class_f1(per_class: pd.DataFrame, class_id: int) -> float:
    if per_class.empty or not {"class_id", "f1"}.issubset(per_class.columns):
        return float("nan")
    subset = per_class[pd.to_numeric(per_class["class_id"], errors="coerce") == int(class_id)]
    return _metric(subset, "f1")


def _hard_f1(per_class: pd.DataFrame) -> float:
    vals = [_class_f1(per_class, class_id) for class_id in HARD_CLASS_IDS]
    vals = [value for value in vals if math.isfinite(value)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _collapse(pred_count: pd.DataFrame) -> Dict[str, Any]:
    if pred_count.empty or "pred_count" not in pred_count.columns:
        return {"predicted_classes": 0, "collapse_risk": "MISSING_PRED_COUNT", "collapse_risk_order": 4}
    counts = pd.to_numeric(pred_count["pred_count"], errors="coerce").fillna(0)
    predicted_classes = int((counts > 0).sum())
    total = float(counts.sum())
    max_ratio = float(counts.max() / total) if total > 0 else float("nan")
    if predicted_classes <= 2:
        risk, order = "HARD_COLLAPSE", 4
    elif predicted_classes < 7 or (math.isfinite(max_ratio) and max_ratio > 0.50):
        risk, order = "COLLAPSE_RISK", 3
    elif math.isfinite(max_ratio) and max_ratio > 0.35:
        risk, order = "MILD_PRED_BIAS", 1
    else:
        risk, order = "NO_COLLAPSE", 0
    return {"predicted_classes": predicted_classes, "max_pred_ratio": max_ratio, "collapse_risk": risk, "collapse_risk_order": order}


def _head_counts(predictions: pd.DataFrame) -> Dict[str, int]:
    if predictions.empty or "routed_head" not in predictions.columns:
        return {"detected_head_count": 0, "fallback_head_count": 0, "single_head_count": 0}
    heads = predictions["routed_head"].astype(str)
    return {
        "detected_head_count": int((heads == "detected_head").sum()),
        "fallback_head_count": int((heads == "fallback_head").sum()),
        "single_head_count": int((heads == "single_head").sum()),
    }


def _run_row(path: Path, name: str) -> Dict[str, Any]:
    train = _read_csv(path / "train_log.csv")
    test = _read_csv(path / "test_metrics.csv")
    per = _read_csv(path / "per_class_metrics.csv")
    fallback = _read_csv(path / "detected_vs_fallback_metrics.csv")
    pred_count = _read_csv(path / "pred_count.csv")
    predictions = _read_csv(path / "predictions.csv")
    check = _read_json(path / "d16_v3_check_summary.json") or _read_json(path / "d16_v1_check_summary.json")
    summary = _read_json(path / "d16_train_summary.json")
    detected = fallback[fallback["group"].astype(str) == "detected"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    fb = fallback[fallback["group"].astype(str) == "fallback"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    row = {
        "run_name": name,
        "output_dir": str(path),
        "epoch_count": int(train["epoch"].nunique()) if not train.empty and "epoch" in train.columns else 0,
        "best_epoch": summary.get("best_epoch", ""),
        "best_val_macro_f1": summary.get("best_val_macro_f1", float("nan")),
        "test_accuracy": _metric(test, "accuracy"),
        "test_macro_f1": _metric(test, "macro_f1"),
        "test_weighted_f1": _weighted_f1(per),
        "hard_F1": _hard_f1(per),
        "detected_macro_f1": _metric(detected, "macro_f1"),
        "detected_accuracy": _metric(detected, "accuracy"),
        "fallback_macro_f1": _metric(fb, "macro_f1"),
        "fallback_accuracy": _metric(fb, "accuracy"),
        "detected_loss_mean_final": _metric(train, "detected_loss_mean"),
        "fallback_loss_mean_final": _metric(train, "fallback_loss_mean"),
        "train_loss_final": _metric(train, "train_loss"),
        "lr": "",
        "epoch_time_sec_final": _metric(train, "epoch_time_sec"),
        "checker_decision": check.get("decision", ""),
    }
    row["detected_fallback_gap"] = (
        row["detected_macro_f1"] - row["fallback_macro_f1"]
        if math.isfinite(row["detected_macro_f1"]) and math.isfinite(row["fallback_macro_f1"])
        else float("nan")
    )
    row.update(_collapse(pred_count))
    row.update(_head_counts(predictions))
    row["delta_vs_d15_macro_f1"] = row["test_macro_f1"] - D15["macro_f1"] if math.isfinite(row["test_macro_f1"]) else float("nan")
    row["delta_vs_v1_best_macro_f1"] = row["test_macro_f1"] - V1_BEST["macro_f1"] if math.isfinite(row["test_macro_f1"]) else float("nan")
    row["delta_vs_v2_best_macro_f1"] = row["test_macro_f1"] - V2_BEST["macro_f1"] if math.isfinite(row["test_macro_f1"]) else float("nan")
    return row


def _decision(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "D16_V3_INVALID"
    if any(str(row.get("checker_decision", "")).startswith("D16_V3_RUN_FAIL") for row in rows):
        return "D16_V3_INVALID"
    repeats = [row for row in rows if "repeat_v1" in str(row.get("run_name", ""))]
    repeat_macros = [float(row["test_macro_f1"]) for row in repeats if math.isfinite(float(row["test_macro_f1"]))]
    if repeat_macros and any(value < 0.628 or value > 0.636 for value in repeat_macros):
        return "D16_V3_SEED_VARIANCE_HIGH"
    candidates = [row for row in rows if "dual_head" in str(row.get("run_name", ""))]
    if not candidates:
        return "D16_V3_INVALID"
    best = max(
        candidates,
        key=lambda row: (
            float(row["test_macro_f1"]) if math.isfinite(float(row["test_macro_f1"])) else -999.0,
            float(row["test_accuracy"]) if math.isfinite(float(row["test_accuracy"])) else -999.0,
        ),
    )
    no_collapse = str(best.get("collapse_risk")) == "NO_COLLAPSE"
    if (
        float(best["test_macro_f1"]) > V1_BEST["macro_f1"]
        and float(best["test_accuracy"]) >= V1_BEST["accuracy"] - 0.005
        and float(best["fallback_macro_f1"]) >= V1_BEST["fallback_macro_f1"]
        and no_collapse
    ):
        return "D16_V3_DUAL_HEAD_NEW_BEST"
    if float(best["fallback_macro_f1"]) > V2_BEST["fallback_macro_f1"] and float(best["test_macro_f1"]) <= V1_BEST["macro_f1"]:
        return "D16_V3_DUAL_HEAD_IMPROVES_FALLBACK_ONLY"
    return "D16_V3_NO_GAIN_OVER_V1"


def _markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v3_results")
    args = parser.parse_args()

    if args.runs:
        names = args.names or [Path(path).name for path in args.runs]
        if len(names) != len(args.runs):
            raise ValueError("--runs and --names must have the same length")
        run_pairs = list(zip([Path(path) for path in args.runs], names))
    else:
        root = Path("outputs/d16_runs/v3")
        run_pairs = [(path, path.name) for path in sorted(root.glob("*")) if path.is_dir()]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_run_row(path, name) for path, name in run_pairs]
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["test_macro_f1"]) if math.isfinite(float(row["test_macro_f1"])) else -999.0,
            float(row["test_accuracy"]) if math.isfinite(float(row["test_accuracy"])) else -999.0,
            float(row["fallback_macro_f1"]) if math.isfinite(float(row["fallback_macro_f1"])) else -999.0,
            float(row["detected_macro_f1"]) if math.isfinite(float(row["detected_macro_f1"])) else -999.0,
            float(row["hard_F1"]) if math.isfinite(float(row["hard_F1"])) else -999.0,
            -abs(float(row["detected_fallback_gap"])) if math.isfinite(float(row["detected_fallback_gap"])) else -999.0,
            -int(row.get("collapse_risk_order", 9)),
        ),
        reverse=True,
    )
    decision = _decision(rows)

    summary_fields = [
        "run_name",
        "checker_decision",
        "test_macro_f1",
        "test_accuracy",
        "test_weighted_f1",
        "hard_F1",
        "fallback_macro_f1",
        "detected_macro_f1",
        "detected_fallback_gap",
        "detected_head_count",
        "fallback_head_count",
        "collapse_risk",
        "delta_vs_v1_best_macro_f1",
        "delta_vs_v2_best_macro_f1",
    ]
    _write_csv(out / "d16_v3_summary.csv", rows, summary_fields)
    _write_csv(out / "d16_v3_ranked_summary.csv", ranked, summary_fields)

    per_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    learning_rows: List[Dict[str, Any]] = []
    for path, name in run_pairs:
        per = _read_csv(path / "per_class_metrics.csv")
        for row in per.to_dict("records"):
            row["run_name"] = name
            per_rows.append(row)
        group = _read_csv(path / "detected_vs_fallback_metrics.csv")
        for row in group.to_dict("records"):
            row["run_name"] = name
            group_rows.append(row)
        conf = _read_csv(path / "confusion_matrix.csv")
        for row in conf.to_dict("records"):
            row["run_name"] = name
            confusion_rows.append(row)
        train = _read_csv(path / "train_log.csv")
        for row in train.to_dict("records"):
            row["run_name"] = name
            learning_rows.append(row)
        preds = _read_csv(path / "predictions.csv")
        counts = _head_counts(preds)
        counts["run_name"] = name
        head_rows.append(counts)

    _write_csv(out / "d16_v3_per_class.csv", per_rows)
    _write_csv(out / "d16_v3_detected_fallback.csv", group_rows)
    _write_csv(out / "d16_v3_head_routing.csv", head_rows)
    _write_csv(out / "d16_v3_confusion_compare.csv", confusion_rows)
    _write_csv(out / "d16_v3_learning_dynamics.csv", learning_rows)

    report = [
        "# D16 V3 Results Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        f"Final decision: `{decision}`",
        "",
        "## Baselines",
        "",
        f"- D15 baseline: accuracy `{D15['accuracy']:.6f}`, macro-F1 `{D15['macro_f1']:.6f}`, weighted-F1 `{D15['weighted_f1']:.6f}`",
        f"- D16 v1 best: accuracy `{V1_BEST['accuracy']:.6f}`, macro-F1 `{V1_BEST['macro_f1']:.6f}`, fallback macro-F1 `{V1_BEST['fallback_macro_f1']:.6f}`",
        f"- D16 v2 best: accuracy `{V2_BEST['accuracy']:.6f}`, macro-F1 `{V2_BEST['macro_f1']:.6f}`, fallback macro-F1 `{V2_BEST['fallback_macro_f1']:.6f}`",
        "",
        "## Ranked Runs",
        "",
    ]
    report.extend(_markdown_table(ranked, ["run_name", "test_macro_f1", "test_accuracy", "fallback_macro_f1", "detected_macro_f1", "hard_F1", "checker_decision"]))
    report.extend(["", "## Output Files", "", "- `d16_v3_summary.csv`", "- `d16_v3_ranked_summary.csv`", "- `d16_v3_per_class.csv`", "- `d16_v3_detected_fallback.csv`", "- `d16_v3_head_routing.csv`", "- `d16_v3_confusion_compare.csv`", "- `d16_v3_learning_dynamics.csv`"])
    (out / "D16_V3_RESULTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "runs": len(rows), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
