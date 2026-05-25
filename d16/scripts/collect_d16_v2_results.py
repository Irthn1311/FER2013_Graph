"""Collect D16 v2 fallback-aware refinement results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
HARD_CLASS_IDS = [0, 2, 4]
V0_FACE_MACRO_F1 = 0.615703
V1_BEST_MACRO_F1 = 0.632938
V1_BEST_ACCURACY = 0.639175
V1_BEST_FALLBACK_MACRO_F1 = 0.409767
V1_SUPCON_L002_MACRO_F1 = 0.618280
V1_HYBRID_CE_MACRO_F1 = 0.618734


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float("nan")
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


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


def _hard_macro_f1(per_class: pd.DataFrame) -> float:
    values = [_class_f1(per_class, class_id) for class_id in HARD_CLASS_IDS]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else float("nan")


def _pred_count_stats(pred_count: pd.DataFrame) -> Dict[str, Any]:
    if pred_count.empty or not {"class_id", "pred_count"}.issubset(pred_count.columns):
        return {
            "predicted_classes": 0,
            "max_pred_ratio": float("nan"),
            "class1_pred_count": 0,
            "class2_pred_count": 0,
            "class4_pred_count": 0,
            "collapse_risk": "MISSING_PRED_COUNT",
            "collapse_risk_order": 4,
        }
    rows = pred_count.copy()
    rows["class_id"] = pd.to_numeric(rows["class_id"], errors="coerce").astype("Int64")
    rows["pred_count"] = pd.to_numeric(rows["pred_count"], errors="coerce").fillna(0).astype(int)
    counts = {int(row["class_id"]): int(row["pred_count"]) for _, row in rows.dropna(subset=["class_id"]).iterrows()}
    total = int(sum(counts.values()))
    predicted_classes = int(sum(1 for value in counts.values() if value > 0))
    max_pred_ratio = max(counts.values()) / total if total > 0 and counts else float("nan")
    class1 = int(counts.get(1, 0))
    class2 = int(counts.get(2, 0))
    class4 = int(counts.get(4, 0))
    if predicted_classes < 7 or (math.isfinite(max_pred_ratio) and max_pred_ratio > 0.50):
        risk = "COLLAPSE_RISK"
        order = 3
    elif (total > 0 and class1 / total < 0.005) or (total > 0 and (class2 / total < 0.05 or class4 / total < 0.05)):
        risk = "HARD_CLASS_SUPPRESSION"
        order = 2
    elif math.isfinite(max_pred_ratio) and max_pred_ratio > 0.35:
        risk = "MILD_PRED_BIAS"
        order = 1
    else:
        risk = "NO_COLLAPSE"
        order = 0
    return {
        "predicted_classes": predicted_classes,
        "max_pred_ratio": float(max_pred_ratio),
        "class1_pred_count": class1,
        "class2_pred_count": class2,
        "class4_pred_count": class4,
        "collapse_risk": risk,
        "collapse_risk_order": order,
    }


def _run_row(path: Path, name: str, kind: str) -> Dict[str, Any]:
    train = _read_csv(path / "train_log.csv")
    test = _read_csv(path / "test_metrics.csv")
    per = _read_csv(path / "per_class_metrics.csv")
    fallback = _read_csv(path / "detected_vs_fallback_metrics.csv")
    pred_count = _read_csv(path / "pred_count.csv")
    check = _read_json(path / "d16_v1_check_summary.json")
    summary = _read_json(path / "d16_train_summary.json")
    detected = fallback[fallback["group"] == "detected"] if not fallback.empty and "group" in fallback else pd.DataFrame()
    fb = fallback[fallback["group"] == "fallback"] if not fallback.empty and "group" in fallback else pd.DataFrame()
    detected_macro = _metric(detected, "macro_f1")
    fallback_macro = _metric(fb, "macro_f1")
    row = {
        "run_name": name,
        "kind": kind,
        "output_dir": str(path),
        "epoch_count": int(train["epoch"].nunique()) if not train.empty and "epoch" in train else 0,
        "best_epoch": summary.get("best_epoch", ""),
        "best_val_macro_f1": summary.get("best_val_macro_f1", float("nan")),
        "test_accuracy": _metric(test, "accuracy"),
        "test_macro_f1": _metric(test, "macro_f1"),
        "test_weighted_f1": _weighted_f1(per),
        "hard_class_macro_f1": _hard_macro_f1(per),
        "angry_f1": _class_f1(per, 0),
        "fear_f1": _class_f1(per, 2),
        "sad_f1": _class_f1(per, 4),
        "detected_macro_f1": detected_macro,
        "fallback_macro_f1": fallback_macro,
        "fallback_accuracy": _metric(fb, "accuracy"),
        "detected_fallback_gap_macro_f1": detected_macro - fallback_macro if math.isfinite(detected_macro) and math.isfinite(fallback_macro) else float("nan"),
        "supcon_loss_total_final": _metric(train, "supcon_loss_total"),
        "lambda_part_supcon_final": _metric(train, "lambda_part_supcon_current"),
        "sample_weight_mean_final": _metric(train, "sample_weight_mean"),
        "checker_decision": check.get("decision", ""),
        "delta_vs_d15_macro_f1": float("nan"),
        "delta_vs_v1_best_macro_f1": float("nan"),
        "delta_vs_v1_best_fallback_macro_f1": float("nan"),
    }
    row.update(_pred_count_stats(pred_count))
    return row


def _direction(run_name: str) -> str:
    if "hybrid" in run_name and "supcon" in run_name:
        return "hybrid_supcon"
    if "hybrid" in run_name:
        return "hybrid"
    if "class_weighted" in run_name:
        return "fallback_class_weighted"
    if "supcon" in run_name:
        return "fallback_supcon"
    if "fallback" in run_name:
        return "fallback_weight"
    return "other"


def _decision(v2_rows: List[Dict[str, Any]]) -> str:
    if not v2_rows:
        return "D16_V2_NO_GAIN_OVER_V1"
    best = max(v2_rows, key=lambda row: (_safe_float(row["test_macro_f1"]), _safe_float(row["test_accuracy"])))
    best_macro = _safe_float(best["test_macro_f1"])
    best_acc = _safe_float(best["test_accuracy"])
    best_fallback = _safe_float(best["fallback_macro_f1"])
    no_collapse = str(best.get("collapse_risk")) == "NO_COLLAPSE"
    new_best = (
        best_macro > V1_BEST_MACRO_F1
        and best_acc >= V1_BEST_ACCURACY - 0.005
        and best_fallback >= V1_BEST_FALLBACK_MACRO_F1 - 0.02
        and no_collapse
    )
    if new_best:
        return "D16_V2_NEW_BEST"
    if any("supcon" in str(row["run_name"]) and _safe_float(row["test_macro_f1"]) > V1_BEST_MACRO_F1 for row in v2_rows):
        return "D16_V2_SUPCON_ADDS_GAIN"
    if any("hybrid" in str(row["run_name"]) and _safe_float(row["test_macro_f1"]) > V1_BEST_MACRO_F1 for row in v2_rows):
        return "D16_V2_HYBRID_ADDS_GAIN"
    if any(
        _safe_float(row["fallback_macro_f1"]) >= V1_BEST_FALLBACK_MACRO_F1 - 0.02
        and _safe_float(row["test_macro_f1"]) >= V1_BEST_MACRO_F1 - 0.005
        for row in v2_rows
    ):
        return "D16_V2_FALLBACK_WEIGHT_CONFIRMED"
    return "D16_V2_NO_GAIN_OVER_V1"


def _markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], max_rows: int = 12) -> List[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in list(rows)[:max_rows]:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v2_results")
    parser.add_argument("--d15_baseline_acc", type=float, default=0.645026)
    parser.add_argument("--d15_baseline_macro_f1", type=float, default=0.622471)
    parser.add_argument("--d15_baseline_weighted_f1", type=float, default=0.641866)
    args = parser.parse_args()
    if len(args.runs) != len(args.names):
        raise ValueError("--runs and --names must have the same length")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_run_row(Path(path), name, "d16_v2") for path, name in zip(args.runs, args.names)]
    for row in rows:
        row["direction_group"] = _direction(str(row["run_name"]))
        row["delta_vs_d15_macro_f1"] = _safe_float(row["test_macro_f1"]) - args.d15_baseline_macro_f1
        row["delta_vs_v1_best_macro_f1"] = _safe_float(row["test_macro_f1"]) - V1_BEST_MACRO_F1
        row["delta_vs_v1_best_fallback_macro_f1"] = _safe_float(row["fallback_macro_f1"]) - V1_BEST_FALLBACK_MACRO_F1
    controls = [
        {
            "run_name": "D15_m8_basic",
            "kind": "baseline",
            "test_accuracy": args.d15_baseline_acc,
            "test_macro_f1": args.d15_baseline_macro_f1,
            "test_weighted_f1": args.d15_baseline_weighted_f1,
            "checker_decision": "baseline",
        },
        _run_row(Path("outputs/d16_v0_face_plus_context_ce_full"), "d16_v0_face_plus_context_ce", "d16_v0_control"),
        _run_row(Path("outputs/d16_runs/v1/d16_v1_face_plus_context_fallback_weighted_ce"), "d16_v1_fallback_weighted_ce", "d16_v1_control"),
        _run_row(Path("outputs/d16_runs/v1/d16_v1_face_plus_context_part_supcon_l002"), "d16_v1_face_supcon_l002", "d16_v1_control"),
        _run_row(Path("outputs/d16_runs/v1/d16_v1_hybrid_detected_face_fallback_fullmask_ce"), "d16_v1_hybrid_ce", "d16_v1_control"),
    ]
    fields = [
        "run_name",
        "kind",
        "direction_group",
        "output_dir",
        "epoch_count",
        "best_epoch",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "hard_class_macro_f1",
        "angry_f1",
        "fear_f1",
        "sad_f1",
        "detected_macro_f1",
        "fallback_macro_f1",
        "fallback_accuracy",
        "detected_fallback_gap_macro_f1",
        "predicted_classes",
        "max_pred_ratio",
        "class1_pred_count",
        "class2_pred_count",
        "class4_pred_count",
        "collapse_risk",
        "supcon_loss_total_final",
        "lambda_part_supcon_final",
        "sample_weight_mean_final",
        "checker_decision",
        "delta_vs_d15_macro_f1",
        "delta_vs_v1_best_macro_f1",
        "delta_vs_v1_best_fallback_macro_f1",
    ]
    _write_csv(out / "d16_v2_summary.csv", controls + rows, fields)
    ranked = sorted(
        rows,
        key=lambda row: (
            -_safe_float(row.get("test_macro_f1")),
            -_safe_float(row.get("test_accuracy")),
            -_safe_float(row.get("fallback_macro_f1")),
            -_safe_float(row.get("hard_class_macro_f1")),
            _safe_float(row.get("detected_fallback_gap_macro_f1")),
            int(row.get("collapse_risk_order", 99) or 99),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    rank_fields = [
        "rank",
        "run_name",
        "direction_group",
        "test_macro_f1",
        "test_accuracy",
        "fallback_macro_f1",
        "hard_class_macro_f1",
        "detected_fallback_gap_macro_f1",
        "collapse_risk",
        "delta_vs_v1_best_macro_f1",
    ]
    _write_csv(out / "d16_v2_ranked_summary.csv", ranked, rank_fields)
    per_rows: List[Dict[str, Any]] = []
    fallback_rows: List[Dict[str, Any]] = []
    for path, name in zip(args.runs, args.names):
        run_path = Path(path)
        per = _read_csv(run_path / "per_class_metrics.csv")
        for row in per.to_dict("records"):
            row["run_name"] = name
            row["class_name"] = CLASS_NAMES.get(int(row.get("class_id", -1)), "")
            per_rows.append(row)
        fb = _read_csv(run_path / "detected_vs_fallback_metrics.csv")
        for row in fb.to_dict("records"):
            row["run_name"] = name
            fallback_rows.append(row)
    _write_csv(out / "d16_v2_per_class.csv", per_rows, sorted(set().union(*(row.keys() for row in per_rows))) if per_rows else ["run_name"])
    _write_csv(out / "d16_v2_detected_fallback.csv", fallback_rows, sorted(set().union(*(row.keys() for row in fallback_rows))) if fallback_rows else ["run_name"])
    final_decision = _decision(rows)
    best = ranked[0] if ranked else None
    report = [
        "# D16 V2 Results Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## Baselines",
        f"- D15: accuracy `{args.d15_baseline_acc:.6f}`, macro-F1 `{args.d15_baseline_macro_f1:.6f}`",
        f"- D16 v0 face_plus_context CE macro-F1: `{V0_FACE_MACRO_F1:.6f}`",
        f"- D16 v1 fallback_weighted_ce macro-F1: `{V1_BEST_MACRO_F1:.6f}`, accuracy `{V1_BEST_ACCURACY:.6f}`, fallback macro-F1 `{V1_BEST_FALLBACK_MACRO_F1:.6f}`",
        f"- D16 v1 SupCon l002 macro-F1: `{V1_SUPCON_L002_MACRO_F1:.6f}`",
        f"- D16 v1 hybrid CE macro-F1: `{V1_HYBRID_CE_MACRO_F1:.6f}`",
        "",
        "## Ranking Criteria",
        "Runs are ranked by test macro-F1, test accuracy, fallback macro-F1, hard-class macro-F1 for Angry/Fear/Sad, smaller detected-fallback gap, then lower collapse risk.",
        "",
        "## Ranked Summary",
        *_markdown_table(ranked, rank_fields),
        "",
        "## Best V2 Run",
        f"- Best run: `{best['run_name'] if best else 'none'}`",
        f"- test_macro_f1: `{_safe_float(best.get('test_macro_f1')):.6f}`" if best else "- test_macro_f1: `nan`",
        f"- test_accuracy: `{_safe_float(best.get('test_accuracy')):.6f}`" if best else "- test_accuracy: `nan`",
        f"- fallback_macro_f1: `{_safe_float(best.get('fallback_macro_f1')):.6f}`" if best else "- fallback_macro_f1: `nan`",
        "",
        "## Decision",
        f"- Final decision: `{final_decision}`",
        "",
        "## Success Gates",
        "- Better than v1 best if macro-F1 > 0.632938, accuracy >= 0.634175, fallback macro-F1 >= 0.389767, and no collapse.",
        "- Strong v2 if macro-F1 >= 0.645, accuracy >= 0.650, and fallback macro-F1 >= 0.43.",
    ]
    (out / "D16_V2_RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "rows": len(rows), "best_run": best["run_name"] if best else None, "final_decision": final_decision}, indent=2))


if __name__ == "__main__":
    main()
