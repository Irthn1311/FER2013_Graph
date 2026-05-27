"""Collect D16R main-branch results against fixed accuracy-first anchors."""

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
HARD_CLASS_IDS = {0, 2, 4, 6}
D15_ACC = 0.645026
D15_MACRO = 0.622471

ANCHORS = [
    {
        "run_name": "D15 baseline",
        "test_accuracy": D15_ACC,
        "test_macro_f1": D15_MACRO,
        "detected_accuracy": "",
        "detected_macro_f1": "",
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue",
        "test_accuracy": 0.633881,
        "test_macro_f1": 0.623164,
        "detected_accuracy": 0.647042,
        "detected_macro_f1": 0.635443,
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "D16 v1 original best observed",
        "test_accuracy": 0.639175,
        "test_macro_f1": 0.632938,
        "detected_accuracy": "",
        "detected_macro_f1": "",
        "predicted_classes": 7,
        "source": "anchor",
    },
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def latest(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return rows[-1] if rows else {}


def group_row(rows: List[Dict[str, str]], group: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("group")) == group:
            return row
    return {}


def collect_run(run_dir: Path) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not run_dir.exists():
        return None, [], [], [f"missing run_dir: {run_dir}"]
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    pred_count = read_rows(run_dir / "pred_count.csv")
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    run_name = str(summary.get("run_name") or run_dir.name)
    required = [
        "checkpoints/best.pt",
        "test_metrics.csv",
        "per_class_metrics.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "d16_train_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        warnings.append(f"{run_name}: missing files: {', '.join(missing)}")
    detected = group_row(groups, "detected")
    fallback = group_row(groups, "fallback")
    predicted_classes = as_int(test.get("predicted_classes"))
    if pred_count:
        predicted_classes = sum(1 for row in pred_count if as_int(row.get("pred_count")) > 0)
    row = {
        "run_name": run_name,
        "test_accuracy": as_float(summary.get("test_accuracy", test.get("accuracy"))),
        "test_macro_f1": as_float(summary.get("test_macro_f1", test.get("macro_f1"))),
        "best_val_macro_f1": as_float(summary.get("best_val_macro_f1")),
        "best_epoch": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch") or test.get("epoch")),
        "detected_accuracy": as_float(detected.get("accuracy")),
        "detected_macro_f1": as_float(detected.get("macro_f1")),
        "fallback_accuracy": as_float(fallback.get("accuracy")),
        "fallback_macro_f1": as_float(fallback.get("macro_f1")),
        "predicted_classes": predicted_classes,
        "total": as_int(test.get("total") or summary.get("test_samples")),
        "output_dir": str(run_dir),
        "missing_files": ";".join(missing),
        "source": "run",
    }
    group_rows = [
        {
            "run_name": run_name,
            "group": item.get("group", ""),
            "total": as_int(item.get("total")),
            "accuracy": as_float(item.get("accuracy")),
            "macro_f1": as_float(item.get("macro_f1")),
        }
        for item in groups
    ]
    pred_by_class = {as_int(item.get("class_id")): as_int(item.get("pred_count")) for item in pred_count}
    hard_rows: List[Dict[str, Any]] = []
    for item in per_class:
        cid = as_int(item.get("class_id"))
        if cid not in HARD_CLASS_IDS:
            continue
        hard_rows.append(
            {
                "run_name": run_name,
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "support": as_int(item.get("support")),
                "pred_count": pred_by_class.get(cid, as_int(item.get("pred_count"))),
                "precision": as_float(item.get("precision")),
                "recall": as_float(item.get("recall")),
                "f1": as_float(item.get("f1")),
            }
        )
    return row, group_rows, hard_rows, warnings


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    value = as_float(value)
    return "" if not math.isfinite(value) else f"{value:.6f}"


def md_table(rows: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def decision(run_rows: List[Dict[str, Any]], warnings: List[str]) -> str:
    valid_rows = [row for row in run_rows if math.isfinite(as_float(row.get("test_accuracy")))]
    if not valid_rows:
        return "RUN_NOT_AVAILABLE"
    best = max(valid_rows, key=lambda row: as_float(row.get("test_accuracy")))
    acc = as_float(best.get("test_accuracy"))
    macro = as_float(best.get("test_macro_f1"))
    predicted_classes = as_int(best.get("predicted_classes"))
    if predicted_classes < 7:
        return "REJECT_RUN_COLLAPSE"
    if warnings and best.get("missing_files"):
        return "RUN_FAILED_NEEDS_DEBUG"
    if acc >= 0.660:
        return "STRONG_MAIN_BRANCH_SIGNAL"
    if acc >= 0.650:
        return "KEEP_PART_ATTENTION_AS_D16R_A1"
    if acc > D15_ACC:
        return "BEATS_D15_ACCURACY_KEEP_AND_REPEAT"
    if acc <= D15_ACC and math.isfinite(macro) and macro > D15_MACRO:
        return "USEFUL_BALANCE_BUT_NOT_ACCURACY_ROUTE"
    if 0.63 <= acc <= 0.64:
        return "PART_ATTENTION_NOT_ENOUGH_TRY_DETECTED_TRANSFORMER_OR_ENSEMBLE"
    return "PART_ATTENTION_NOT_ENOUGH_TRY_DETECTED_TRANSFORMER_OR_ENSEMBLE"


def write_report(
    output_dir: Path,
    run_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    hard_rows: List[Dict[str, Any]],
    warnings: List[str],
) -> str:
    dec = decision(run_rows, warnings)
    accuracy_rows = sorted(ANCHORS + run_rows, key=lambda row: as_float(row.get("test_accuracy")), reverse=True)
    macro_rows = sorted(ANCHORS + run_rows, key=lambda row: as_float(row.get("test_macro_f1")), reverse=True)
    pred_rows = [
        {
            "run_name": row.get("run_name"),
            "predicted_classes": row.get("predicted_classes"),
            "total": row.get("total", ""),
            "source": row.get("source", ""),
        }
        for row in run_rows
    ]
    lines = [
        "# D16R Main Branch Compare",
        "",
        "D16R-A1 evaluates learned part weighting/readout on the MediaPipe pixel prior rescue path. It does not add region masks, fallback rescue, SupCon, multi-seed runs, or ensemble logic.",
        "",
        "## Accuracy-First Table",
        *md_table(
            accuracy_rows,
            [
                "run_name",
                "test_accuracy",
                "test_macro_f1",
                "detected_accuracy",
                "detected_macro_f1",
                "predicted_classes",
                "source",
            ],
        ),
        "",
        "## Macro-F1 Secondary Table",
        *md_table(macro_rows, ["run_name", "test_macro_f1", "test_accuracy", "source"]),
        "",
        "## Detected vs Fallback Group Table",
        *md_table(group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"]),
        "",
        "## Hard Classes",
        *md_table(hard_rows, ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Predicted Class Count / No Collapse",
        *md_table(pred_rows, ["run_name", "predicted_classes", "total", "source"]),
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in warnings]])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("D16R_MAIN_BRANCH_COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    return dec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for text in args.run_dirs:
        row, groups, hard, run_warnings = collect_run(Path(text))
        if row is not None:
            run_rows.append(row)
        group_rows.extend(groups)
        hard_rows.extend(hard)
        warnings.extend(run_warnings)

    write_csv(
        output_dir / "d16r_main_branch_summary.csv",
        run_rows,
        [
            "run_name",
            "test_accuracy",
            "test_macro_f1",
            "best_val_macro_f1",
            "best_epoch",
            "detected_accuracy",
            "detected_macro_f1",
            "fallback_accuracy",
            "fallback_macro_f1",
            "predicted_classes",
            "total",
            "output_dir",
            "missing_files",
            "source",
        ],
    )
    write_csv(
        output_dir / "d16r_main_branch_hard_class.csv",
        hard_rows,
        ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"],
    )
    write_csv(output_dir / "d16r_main_branch_group_metrics.csv", group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"])
    dec = write_report(output_dir, run_rows, group_rows, hard_rows, warnings)
    print(json.dumps({"output_dir": str(output_dir), "decision": dec, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
