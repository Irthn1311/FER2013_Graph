"""Collect and compare the two D16 pixel-prior rescue anchor runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

ANCHORS = {
    "D15 baseline": {"test_accuracy": 0.645026, "test_macro_f1": 0.622471, "fallback_macro_f1": None},
    "D16 v1 original best observed": {"test_accuracy": 0.639175, "test_macro_f1": 0.632938, "fallback_macro_f1": 0.409767},
    "D16 v4 grid8 old prior": {"test_accuracy": 0.624965, "test_macro_f1": 0.618746, "fallback_macro_f1": 0.366175},
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def latest_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        return {}
    return rows[-1]


def group_row(rows: List[Dict[str, str]], group: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("group")) == group:
            return row
    return {}


def run_name_from_dir(run_dir: Path, summary: Dict[str, Any]) -> str:
    return str(summary.get("run_name") or run_dir.name)


def collect_run(run_dir: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    summary_json = read_json(run_dir / "d16_train_summary.json")
    run_name = run_name_from_dir(run_dir, summary_json)

    required = [
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "train_log.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "per_class_metrics.csv",
        "last_per_class_metrics.csv",
        "pred_count.csv",
        "last_pred_count.csv",
        "detected_vs_fallback_metrics.csv",
        "last_detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "last_detected_fallback_per_class_metrics.csv",
        "confusion_matrix.csv",
        "last_confusion_matrix.csv",
        "predictions.csv",
        "last_predictions.csv",
        "resolved_config.json",
        "resolved_config.yaml",
        "d16_train_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        warnings.append(f"{run_name}: missing files: {', '.join(missing)}")

    test = latest_row(read_rows(run_dir / "test_metrics.csv"))
    last_test = latest_row(read_rows(run_dir / "last_test_metrics.csv"))
    val_rows = read_rows(run_dir / "val_metrics.csv")
    best_val = max((as_float(r.get("macro_f1")) for r in val_rows), default=float("nan"))
    best_epoch = as_int(summary_json.get("best_epoch") or test.get("checkpoint_epoch") or test.get("epoch"))
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    detected = group_row(groups, "detected")
    fallback = group_row(groups, "fallback")
    pred_count = read_rows(run_dir / "pred_count.csv")
    predicted_classes = sum(1 for row in pred_count if as_int(row.get("pred_count")) > 0)

    row = {
        "run_name": run_name,
        "test_accuracy": as_float(summary_json.get("test_accuracy", test.get("accuracy"))),
        "test_macro_f1": as_float(summary_json.get("test_macro_f1", test.get("macro_f1"))),
        "test_weighted_f1": as_float(test.get("weighted_f1")),
        "best_val_macro_f1": as_float(summary_json.get("best_val_macro_f1", best_val)),
        "best_epoch": best_epoch,
        "last_test_accuracy": as_float(summary_json.get("last_test_accuracy", last_test.get("accuracy"))),
        "last_test_macro_f1": as_float(summary_json.get("last_test_macro_f1", last_test.get("macro_f1"))),
        "detected_accuracy": as_float(detected.get("accuracy")),
        "detected_macro_f1": as_float(detected.get("macro_f1")),
        "fallback_accuracy": as_float(fallback.get("accuracy")),
        "fallback_macro_f1": as_float(fallback.get("macro_f1")),
        "predicted_classes": predicted_classes or as_int(test.get("predicted_classes")),
        "total": as_int(test.get("total") or summary_json.get("test_samples")),
        "output_dir": str(run_dir),
        "missing_files": ";".join(missing),
    }

    per_class_rows: List[Dict[str, Any]] = []
    pred_by_class = {as_int(r.get("class_id")): as_int(r.get("pred_count")) for r in pred_count}
    for pc in read_rows(run_dir / "per_class_metrics.csv"):
        cid = as_int(pc.get("class_id"))
        per_class_rows.append(
            {
                "run_name": run_name,
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "support": as_int(pc.get("support")),
                "pred_count": pred_by_class.get(cid, as_int(pc.get("pred_count"))),
                "precision": as_float(pc.get("precision")),
                "recall": as_float(pc.get("recall")),
                "f1": as_float(pc.get("f1")),
            }
        )

    group_rows: List[Dict[str, Any]] = []
    for gr in groups:
        group_rows.append(
            {
                "run_name": run_name,
                "group": gr.get("group", ""),
                "total": as_int(gr.get("total")),
                "accuracy": as_float(gr.get("accuracy")),
                "macro_f1": as_float(gr.get("macro_f1")),
            }
        )
    return row, group_rows, per_class_rows, warnings


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    value = as_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def decision(summary_rows: List[Dict[str, Any]], warnings: List[str]) -> str:
    if warnings or not summary_rows:
        return "RUN_FAILED_NEEDS_DEBUG"
    best = max(summary_rows, key=lambda r: as_float(r.get("test_accuracy")))
    acc = as_float(best.get("test_accuracy"))
    macro = as_float(best.get("test_macro_f1"))
    d15_acc = ANCHORS["D15 baseline"]["test_accuracy"]
    if acc >= 0.660:
        return "KEEP_PIXEL_RESCUE_AS_NEW_PRIOR_BASELINE"
    if acc >= 0.650:
        return "KEEP_PIXEL_RESCUE_AS_NEW_PRIOR_BASELINE"
    if acc >= d15_acc:
        return "KEEP_PIXEL_RESCUE_AS_NEW_PRIOR_BASELINE"
    if acc < 0.645 and math.isfinite(macro) and macro > float(ANCHORS["D15 baseline"]["test_macro_f1"]):
        return "KEEP_RESCUE_BUT_NEED_DETECTED_BRANCH_UPGRADE"
    if 0.62 <= acc < 0.645:
        return "RESCUE_NOT_ENOUGH_MOVE_TO_DETECTED_UPGRADE"
    return "RESCUE_NOT_ENOUGH_MOVE_TO_DETECTED_UPGRADE"


def markdown_table(rows: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---" for _ in fields]) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(path: Path, summary_rows: List[Dict[str, Any]], group_rows: List[Dict[str, Any]], per_class_rows: List[Dict[str, Any]], warnings: List[str]) -> str:
    dec = decision(summary_rows, warnings)
    best_rows = sorted(summary_rows, key=lambda r: as_float(r.get("test_accuracy")), reverse=True)
    anchor_rows = [
        {
            "run": name,
            "acc": vals["test_accuracy"],
            "macro_f1": vals["test_macro_f1"],
            "fallback_macro_f1": vals["fallback_macro_f1"] if vals["fallback_macro_f1"] is not None else "",
        }
        for name, vals in ANCHORS.items()
    ]
    rescue_anchor_rows = [
        {"run": r["run_name"], "acc": r["test_accuracy"], "macro_f1": r["test_macro_f1"], "fallback_macro_f1": r["fallback_macro_f1"]}
        for r in best_rows
    ]

    lines = [
        "# D16 Rescue Train Compare",
        "",
        "## Rescue Prior Status",
        "- prior_dir: `outputs/d16_mediapipe_pixel_priors_best_retry_rescue`",
        "- fallback reduction train: `3582 -> 1156` (`12.48% -> 4.03%`)",
        "- fallback reduction val: `422 -> 161` (`11.76% -> 4.49%`)",
        "- fallback reduction test: `446 -> 158` (`12.43% -> 4.40%`)",
        "- checker: `PASS`",
        "- dataset smoke: `PASS`",
        "- scope: D16 MediaPipe pixel priors only; no region masks and no model architecture changes",
        "",
        "## Best Checkpoint Comparison",
        *markdown_table(
            best_rows,
            [
                "run_name",
                "test_accuracy",
                "test_macro_f1",
                "detected_accuracy",
                "detected_macro_f1",
                "fallback_accuracy",
                "fallback_macro_f1",
                "best_epoch",
                "predicted_classes",
            ],
        ),
        "",
        "## Anchor Comparison",
        *markdown_table(anchor_rows + rescue_anchor_rows, ["run", "acc", "macro_f1", "fallback_macro_f1"]),
        "",
        "## Accuracy-First Interpretation",
        "The primary target is test accuracy on the path toward 0.70+. Macro-F1 remains secondary and is used to catch class collapse or hard-class regressions.",
        "",
    ]
    if best_rows:
        best = best_rows[0]
        acc = as_float(best["test_accuracy"])
        macro = as_float(best["test_macro_f1"])
        if acc >= 0.650:
            lines.append(f"- `STRONG_SIGNAL`: best rescue accuracy is `{acc:.6f}`, above the 0.650 gate.")
        elif acc >= float(ANCHORS["D15 baseline"]["test_accuracy"]):
            lines.append(f"- `USEFUL_SIGNAL`: best rescue accuracy `{acc:.6f}` beats the D15 accuracy anchor.")
        elif macro > float(ANCHORS["D15 baseline"]["test_macro_f1"]):
            lines.append(f"- Macro-F1 improved to `{macro:.6f}`, but accuracy `{acc:.6f}` does not clear the D15 accuracy anchor, so this is not enough for the 0.70+ goal.")
        else:
            lines.append(f"- Best rescue accuracy `{acc:.6f}` does not beat D15 accuracy, so fallback rescue alone is not enough.")
    lines.extend(
        [
            "",
            "## Fallback Interpretation",
            "- Fallback count is now much smaller, so fallback_macro_f1 has higher variance and should not be over-interpreted alone.",
            "- Detected metrics are now more important for the 0.70+ accuracy route because most test samples are detected after rescue.",
            "",
            "## Group Metrics",
            *markdown_table(group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"]),
            "",
            "## Hard Class F1",
        ]
    )
    hard = [r for r in per_class_rows if r["class_id"] in (0, 2, 4, 6)]
    lines.extend(markdown_table(hard, ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"]))
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return dec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/rescue_train_compare")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for run_dir_text in args.run_dirs:
        row, groups, per_class, run_warnings = collect_run(Path(run_dir_text))
        summary_rows.append(row)
        group_rows.extend(groups)
        per_class_rows.extend(per_class)
        warnings.extend(run_warnings)

    write_csv(
        output_dir / "d16_rescue_runs_summary.csv",
        summary_rows,
        [
            "run_name",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "best_val_macro_f1",
            "best_epoch",
            "last_test_accuracy",
            "last_test_macro_f1",
            "detected_accuracy",
            "detected_macro_f1",
            "fallback_accuracy",
            "fallback_macro_f1",
            "predicted_classes",
            "total",
            "output_dir",
            "missing_files",
        ],
    )
    write_csv(output_dir / "d16_rescue_group_metrics.csv", group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"])
    write_csv(output_dir / "d16_rescue_per_class_metrics.csv", per_class_rows, ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"])
    dec = write_report(output_dir / "D16_RESCUE_TRAIN_COMPARE.md", summary_rows, group_rows, per_class_rows, warnings)
    print(json.dumps({"output_dir": str(output_dir), "decision": dec, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
