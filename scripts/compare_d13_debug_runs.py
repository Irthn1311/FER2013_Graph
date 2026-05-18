"""Compare two D13A debug run checks and recommend Kaggle run order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_d13_debug_run import FAIL, PASS, WARN, check_run


def _load_summary(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "d13_debug_check_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return check_run(run_dir)


def _collapse_warning(summary: Dict[str, Any]) -> bool:
    pred = summary.get("pred_count_summary", {}) or {}
    ratio = pred.get("max_ratio")
    return ratio is not None and float(ratio) > 0.90


def _row(name: str, run_dir: str | Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    warnings = summary.get("warnings", []) or []
    errors = summary.get("errors", []) or []
    return {
        "name": name,
        "run_dir": str(run_dir),
        "final_decision": summary.get("final_decision"),
        "best_val_macro_f1": summary.get("best_val_macro_f1"),
        "best_val_accuracy": summary.get("best_val_accuracy"),
        "last_train_loss": summary.get("last_train_loss"),
        "loss_trend": summary.get("loss_trend"),
        "pred_count_collapse_warning": _collapse_warning(summary),
        "pred_count_max_ratio": (summary.get("pred_count_summary", {}) or {}).get("max_ratio"),
        "effective_regions_mean": summary.get("effective_regions_mean"),
        "empty_region_ratio_mean": summary.get("empty_region_ratio_mean"),
        "runtime_seconds": _runtime_seconds(run_dir),
        "checkpoint_exists": summary.get("checkpoint_exists"),
        "num_warnings": len(warnings),
        "num_errors": len(errors),
    }


def _runtime_seconds(run_dir: str | Path) -> float | None:
    train_log = Path(run_dir) / "train_log.csv"
    if not train_log.exists():
        return None
    try:
        df = pd.read_csv(train_log)
    except Exception:
        return None
    cols = [c for c in df.columns if c.endswith("_seconds") or c in {"train_seconds", "val_seconds"}]
    if not cols:
        return None
    total = 0.0
    found = False
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.empty:
            total += float(values.sum())
            found = True
    return total if found else None


def _comparison_decision(a: Dict[str, Any], b: Dict[str, Any], name_a: str, name_b: str) -> str:
    da = str(a.get("final_decision"))
    db = str(b.get("final_decision"))
    if da == FAIL or db == FAIL:
        return "DO_NOT_RUN_FAILED_CONFIG_FULL"
    if da == PASS and db == PASS:
        return "BOTH_READY_FOR_KAGGLE"
    if da == PASS and db != PASS:
        return "RUN_EDGEAWARE_FULL_FIRST" if "edge" in name_a.lower() else f"RUN_{name_a.upper()}_FULL_FIRST"
    if db == PASS and da != PASS:
        return "RUN_GINE_FULL_FIRST" if "gine" in name_b.lower() else f"RUN_{name_b.upper()}_FULL_FIRST"
    if da == WARN and db == WARN:
        return "REVIEW_BEFORE_FULL"
    return "REVIEW_BEFORE_FULL"


def compare_runs(run_a: str | Path, name_a: str, run_b: str | Path, name_b: str, output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_a = _load_summary(run_a)
    summary_b = _load_summary(run_b)
    rows = [_row(name_a, run_a, summary_a), _row(name_b, run_b, summary_b)]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "d13_debug_compare_summary.csv"
    df.to_csv(csv_path, index=False)
    decision = _comparison_decision(summary_a, summary_b, name_a, name_b)
    report_path = output_dir / "d13_debug_compare_report.md"
    lines = [
        "# D13 Debug Compare Report",
        "",
        f"- final_decision: {decision}",
        f"- run_a: {name_a} ({run_a})",
        f"- run_b: {name_b} ({run_b})",
        "",
        "## Summary",
        "```text",
        df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "- This comparison only checks training-loop health and collapse risk.",
        "- Do not choose the final encoder from a 5 epoch debug run.",
        "- Use the decision only to choose Kaggle full-train order.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "final_decision": decision,
        "summary_csv": str(csv_path),
        "report": str(report_path),
        "runs": rows,
    }
    print(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--name_a", required=True)
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--name_b", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    compare_runs(args.run_a, args.name_a, args.run_b, args.name_b, args.output_dir)


if __name__ == "__main__":
    main()
