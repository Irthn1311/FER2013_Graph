"""Collect D16 v4 fallback patch encoder results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd


D15 = {"accuracy": 0.645026, "macro_f1": 0.622471}
V1 = {"accuracy": 0.639175, "macro_f1": 0.632938, "fallback_macro_f1": 0.409767}
V2 = {"macro_f1": 0.623511, "fallback_macro_f1": 0.456697}
V3 = {"macro_f1": 0.617991}
HARD = [0, 2, 4]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def mean_metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def class_f1(per: pd.DataFrame, cls: int) -> float:
    if per.empty:
        return float("nan")
    sub = per[pd.to_numeric(per.get("class_id"), errors="coerce") == int(cls)]
    return metric(sub, "f1")


def weighted_f1(per: pd.DataFrame) -> float:
    if per.empty or not {"support", "f1"}.issubset(per.columns):
        return float("nan")
    support = pd.to_numeric(per["support"], errors="coerce").fillna(0)
    f1 = pd.to_numeric(per["f1"], errors="coerce").fillna(0)
    return float((support * f1).sum() / max(float(support.sum()), 1.0))


def hard_f1(per: pd.DataFrame) -> float:
    vals = [class_f1(per, cls) for cls in HARD]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def pred_stats(pred: pd.DataFrame) -> Dict[str, Any]:
    if pred.empty or "pred_count" not in pred.columns:
        return {"predicted_classes": 0, "collapse_risk": "MISSING_PRED_COUNT", "collapse_order": 9}
    counts = pd.to_numeric(pred["pred_count"], errors="coerce").fillna(0)
    total = float(counts.sum())
    predicted = int((counts > 0).sum())
    max_ratio = float(counts.max() / total) if total > 0 else float("nan")
    if predicted <= 2 or (math.isfinite(max_ratio) and max_ratio > 0.5):
        risk, order = "COLLAPSE_RISK", 3
    elif predicted < 7:
        risk, order = "HARD_CLASS_SUPPRESSION", 2
    elif math.isfinite(max_ratio) and max_ratio > 0.35:
        risk, order = "MILD_PRED_BIAS", 1
    else:
        risk, order = "NO_COLLAPSE", 0
    return {"predicted_classes": predicted, "pred_max_ratio": max_ratio, "collapse_risk": risk, "collapse_order": order}


def run_row(path: Path, name: str) -> Dict[str, Any]:
    train = read_csv(path / "train_log.csv")
    test = read_csv(path / "test_metrics.csv")
    per = read_csv(path / "per_class_metrics.csv")
    fallback = read_csv(path / "detected_vs_fallback_metrics.csv")
    pred = read_csv(path / "pred_count.csv")
    predictions = read_csv(path / "predictions.csv")
    check = read_json(path / "d16_v4_check_summary.json")
    summary = read_json(path / "d16_train_summary.json")
    det = fallback[fallback["group"].astype(str) == "detected"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    fb = fallback[fallback["group"].astype(str) == "fallback"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    paths = predictions["routed_path"].astype(str) if not predictions.empty and "routed_path" in predictions.columns else pd.Series(dtype=str)
    row = {
        "run_name": name,
        "output_dir": str(path),
        "best_epoch": summary.get("best_epoch", ""),
        "best_val_macro_f1": summary.get("best_val_macro_f1", float("nan")),
        "test_accuracy": metric(test, "accuracy"),
        "test_macro_f1": metric(test, "macro_f1"),
        "test_weighted_f1": weighted_f1(per),
        "hard_F1": hard_f1(per),
        "detected_accuracy": metric(det, "accuracy"),
        "detected_macro_f1": metric(det, "macro_f1"),
        "fallback_accuracy": metric(fb, "accuracy"),
        "fallback_macro_f1": metric(fb, "macro_f1"),
        "detected_path_count": int((paths == "detected_face_path").sum()),
        "fallback_path_count": int(paths.isin(["fallback_grid_path", "fallback_transformer_path"]).sum()),
        "fallback_token_count_mean": metric(train, "fallback_token_count_mean"),
        "epoch_time_mean": mean_metric(train, "epoch_time_sec"),
        "memory_reserved_mb": metric(train, "memory_reserved_mb"),
        "checker_decision": check.get("decision", ""),
    }
    row["detected_fallback_gap"] = row["detected_macro_f1"] - row["fallback_macro_f1"] if math.isfinite(row["detected_macro_f1"]) and math.isfinite(row["fallback_macro_f1"]) else float("nan")
    row["delta_vs_D15_macro_f1"] = row["test_macro_f1"] - D15["macro_f1"]
    row["delta_vs_v1_macro_f1"] = row["test_macro_f1"] - V1["macro_f1"]
    row["delta_vs_v2_macro_f1"] = row["test_macro_f1"] - V2["macro_f1"]
    row["delta_vs_v3_macro_f1"] = row["test_macro_f1"] - V3["macro_f1"]
    row.update(pred_stats(pred))
    if row["test_macro_f1"] > V1["macro_f1"] and row["test_accuracy"] >= 0.634 and row["fallback_macro_f1"] >= V1["fallback_macro_f1"] and row["collapse_risk"] == "NO_COLLAPSE":
        decision = "D16_V4_PATCH_FALLBACK_NEW_BEST"
    elif row["fallback_macro_f1"] > V2["fallback_macro_f1"] and row["test_macro_f1"] <= V1["macro_f1"]:
        decision = "D16_V4_PATCH_FALLBACK_IMPROVES_FALLBACK_ONLY"
    elif str(row["checker_decision"]).startswith("D16_V4_RUN_FAIL"):
        decision = "D16_V4_INVALID"
    else:
        decision = "D16_V4_NO_GAIN_OVER_V1"
    row["run_decision"] = decision
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v4_results")
    args = parser.parse_args()
    if args.runs:
        names = args.names or [Path(p).name for p in args.runs]
        pairs = list(zip([Path(p) for p in args.runs], names))
    else:
        root = Path("outputs/d16_runs/v4")
        pairs = [(p, p.name) for p in sorted(root.glob("*")) if p.is_dir()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_row(path, name) for path, name in pairs]
    ranked = sorted(rows, key=lambda r: (r["test_macro_f1"], r["test_accuracy"], r["fallback_macro_f1"], r["detected_macro_f1"], r["hard_F1"], -abs(r["detected_fallback_gap"]) if math.isfinite(r["detected_fallback_gap"]) else -999, -r["collapse_order"]), reverse=True)
    fields = ["run_name", "test_macro_f1", "test_accuracy", "test_weighted_f1", "fallback_macro_f1", "detected_macro_f1", "hard_F1", "detected_fallback_gap", "detected_path_count", "fallback_path_count", "fallback_token_count_mean", "epoch_time_mean", "memory_reserved_mb", "collapse_risk", "checker_decision", "run_decision", "delta_vs_D15_macro_f1", "delta_vs_v1_macro_f1", "delta_vs_v2_macro_f1", "delta_vs_v3_macro_f1"]
    write_csv(out / "d16_v4_summary.csv", rows, fields)
    write_csv(out / "d16_v4_ranked_summary.csv", ranked, fields)
    for filename, source in [
        ("d16_v4_per_class.csv", "per_class_metrics.csv"),
        ("d16_v4_detected_fallback.csv", "detected_vs_fallback_metrics.csv"),
        ("d16_v4_confusion_compare.csv", "confusion_matrix.csv"),
        ("d16_v4_learning_dynamics.csv", "train_log.csv"),
    ]:
        merged = []
        for path, name in pairs:
            df = read_csv(path / source)
            for row in df.to_dict("records"):
                row["run_name"] = name
                merged.append(row)
        write_csv(out / filename, merged)
    write_csv(out / "d16_v4_routing_compare.csv", [{"run_name": r["run_name"], "detected_path_count": r["detected_path_count"], "fallback_path_count": r["fallback_path_count"], "fallback_token_count_mean": r["fallback_token_count_mean"]} for r in rows])
    write_csv(out / "d16_v4_runtime_compare.csv", [{"run_name": r["run_name"], "epoch_time_mean": r["epoch_time_mean"], "memory_reserved_mb": r["memory_reserved_mb"], "fallback_token_count_mean": r["fallback_token_count_mean"]} for r in rows])
    if any(r["run_decision"] == "D16_V4_PATCH_FALLBACK_NEW_BEST" for r in rows):
        final = "D16_V4_PATCH_FALLBACK_NEW_BEST"
    elif any(r["run_decision"] == "D16_V4_PATCH_FALLBACK_IMPROVES_FALLBACK_ONLY" for r in rows):
        final = "D16_V4_PATCH_FALLBACK_IMPROVES_FALLBACK_ONLY"
    elif any(str(r["run_name"]).find("grid") >= 0 for r in ranked[:1]):
        final = "D16_V4_GRID_BETTER_THAN_TRANSFORMER" if ranked else "D16_V4_INVALID"
    elif ranked:
        final = "D16_V4_TRANSFORMER_BETTER_THAN_GRID"
    else:
        final = "D16_V4_INVALID"
    lines = [
        "# D16 V4 Results Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## Baselines",
        f"- D15: accuracy `{D15['accuracy']:.6f}`, macro-F1 `{D15['macro_f1']:.6f}`",
        f"- D16 v1 best: accuracy `{V1['accuracy']:.6f}`, macro-F1 `{V1['macro_f1']:.6f}`, fallback macro-F1 `{V1['fallback_macro_f1']:.6f}`",
        f"- D16 v2 best: macro-F1 `{V2['macro_f1']:.6f}`, fallback macro-F1 `{V2['fallback_macro_f1']:.6f}`",
        f"- D16 v3 best: macro-F1 `{V3['macro_f1']:.6f}`",
        "",
        "## Ranked Runs",
        "| run_name | test_macro_f1 | test_accuracy | fallback_macro_f1 | detected_macro_f1 | hard_F1 | decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in ranked:
        lines.append(f"| {r['run_name']} | {r['test_macro_f1']:.6f} | {r['test_accuracy']:.6f} | {r['fallback_macro_f1']:.6f} | {r['detected_macro_f1']:.6f} | {r['hard_F1']:.6f} | {r['run_decision']} |")
    lines.extend(["", "## Final Decision", f"`{final}`"])
    (out / "D16_V4_RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(rows), "output_dir": str(out), "final_decision": final}, indent=2))


if __name__ == "__main__":
    main()
