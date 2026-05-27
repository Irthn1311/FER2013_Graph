"""Check D16 main-branch run artifacts.

This checker is accuracy-first and keeps part-attention diagnostics optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


HARD_CLASSES = {0: "Angry", 2: "Fear", 4: "Sad", 6: "Neutral"}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def finite(value: Any) -> bool:
    value = as_float(value)
    return bool(math.isfinite(value))


def attention_status(run_dir: Path) -> Dict[str, Any]:
    summary = read_rows(run_dir / "part_attention_summary.csv")
    by_class = read_rows(run_dir / "part_attention_by_class.csv")
    if not summary and not by_class:
        return {"present": False, "status": "NOT_AVAILABLE", "failures": []}
    failures: List[str] = []
    if len(summary) != 5:
        failures.append(f"part_attention_summary.csv row_count={len(summary)} expected=5")
    if by_class and len(by_class) != 35:
        failures.append(f"part_attention_by_class.csv row_count={len(by_class)} expected=35")
    for name, rows in (("part_attention_summary.csv", summary), ("part_attention_by_class.csv", by_class)):
        for idx, row in enumerate(rows):
            val = as_float(row.get("attention_mean"))
            samples = as_int(row.get("samples"))
            if samples > 0 and not math.isfinite(val):
                failures.append(f"{name} row {idx} attention_mean is non-finite")
            if math.isfinite(val) and not (0.0 <= val <= 1.0):
                failures.append(f"{name} row {idx} attention_mean out of [0,1]: {val}")
    return {
        "present": True,
        "status": "PASS" if not failures else "FAIL",
        "summary_rows": len(summary),
        "by_class_rows": len(by_class),
        "failures": failures,
    }


def check_run(run_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
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
    for rel in required:
        if not (run_dir / rel).exists():
            failures.append(f"missing {rel}")

    test = latest(read_rows(run_dir / "test_metrics.csv"))
    pred_count = read_rows(run_dir / "pred_count.csv")
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    group_per_class = read_rows(run_dir / "detected_fallback_per_class_metrics.csv")

    test_accuracy = as_float(test.get("accuracy"))
    test_macro_f1 = as_float(test.get("macro_f1"))
    if not finite(test_accuracy):
        failures.append("test_accuracy missing or non-finite")
    if not finite(test_macro_f1):
        failures.append("test_macro_f1 missing or non-finite")

    predicted_classes = as_int(test.get("predicted_classes"))
    if pred_count:
        predicted_classes = sum(1 for row in pred_count if as_int(row.get("pred_count")) > 0)
    if predicted_classes < 7:
        failures.append(f"predicted_classes={predicted_classes} expected=7")

    if not groups:
        failures.append("missing detected/fallback group metrics")
    else:
        group_names = {str(row.get("group")) for row in groups}
        if "detected" not in group_names or "fallback" not in group_names:
            failures.append(f"detected/fallback groups incomplete: {sorted(group_names)}")
    if not group_per_class:
        failures.append("missing detected_fallback_per_class_metrics.csv rows")

    hard_seen = {as_int(row.get("class_id")) for row in per_class if as_int(row.get("class_id")) in HARD_CLASSES}
    missing_hard = sorted(set(HARD_CLASSES) - hard_seen)
    if missing_hard:
        failures.append(f"missing hard class metrics: {missing_hard}")
    for row in per_class:
        if as_int(row.get("class_id")) in HARD_CLASSES and not finite(row.get("f1")):
            failures.append(f"hard class {row.get('class_id')} f1 non-finite")

    attention = attention_status(run_dir)
    if attention["status"] == "FAIL":
        failures.extend(attention["failures"])
    elif not attention["present"]:
        warnings.append("part attention diagnostics not found")

    if predicted_classes < 7:
        decision = "REJECT_RUN_COLLAPSE"
    elif failures:
        decision = "D16_MAIN_BRANCH_CHECK_FAIL"
    else:
        decision = "D16_MAIN_BRANCH_CHECK_PASS"

    return {
        "run_dir": str(run_dir),
        "decision": decision,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "predicted_classes": predicted_classes,
        "attention": attention,
        "failures": failures,
        "warnings": warnings,
    }


def write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "d16_main_branch_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# D16 Main Branch Run Check",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- decision: `{summary['decision']}`",
        f"- test_accuracy: `{as_float(summary.get('test_accuracy')):.6f}`",
        f"- test_macro_f1: `{as_float(summary.get('test_macro_f1')):.6f}`",
        f"- predicted_classes: `{summary.get('predicted_classes')}`",
        f"- attention_diagnostics: `{summary.get('attention', {}).get('status')}`",
        "",
        "## Failures",
    ]
    failures = summary.get("failures") or []
    lines.extend([f"- {item}" for item in failures] if failures else ["- none"])
    warnings = summary.get("warnings") or []
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    output_dir.joinpath("D16_MAIN_BRANCH_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = check_run(Path(args.run_dir))
    write_report(Path(args.output_dir), summary)
    print(json.dumps(summary, indent=2))
    if summary["decision"] in {"D16_MAIN_BRANCH_CHECK_FAIL", "REJECT_RUN_COLLAPSE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
