"""Collect D13B diagnostic run outputs into summary tables and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


D13A_K144_TEST_MACRO_F1 = 0.5829
D13A_K144_TEST_ACC = 0.6166


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _last(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _best_val(val: pd.DataFrame) -> Dict[str, float]:
    if val.empty:
        return {"best_epoch": -1, "best_val_macro_f1": float("nan"), "best_val_accuracy": float("nan")}
    metric = "val_macro_f1" if "val_macro_f1" in val else "macro_f1"
    if metric not in val:
        return {"best_epoch": -1, "best_val_macro_f1": float("nan"), "best_val_accuracy": float("nan")}
    idx = pd.to_numeric(val[metric], errors="coerce").idxmax()
    row = val.loc[idx]
    return {
        "best_epoch": int(row.get("epoch", idx)),
        "best_val_macro_f1": float(row.get(metric, np.nan)),
        "best_val_accuracy": float(row.get("val_accuracy", row.get("accuracy", np.nan))),
    }


def _pred_row(pred: pd.DataFrame) -> Dict[str, Any]:
    if pred.empty:
        return {"pred_max_ratio": float("nan")}
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    return work.iloc[-1].to_dict()


def _load_check(run_dir: Path) -> str:
    path = run_dir / "d13b_diagnostic_check_summary.json"
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("decision", ""))
    except Exception:
        return ""


def collect(root_dir: Path) -> Dict[str, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    run_dirs = [
        p
        for p in root_dir.iterdir()
        if p.is_dir()
        and (p / "train_log.csv").exists()
        and (p / "val_metrics.csv").exists()
        and (p / "test_metrics.csv").exists()
    ]
    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        train = _read_csv(run_dir / "train_log.csv")
        val = _read_csv(run_dir / "val_metrics.csv")
        test = _read_csv(run_dir / "test_metrics.csv")
        slots = _read_csv(run_dir / "slot_stats.csv")
        pred = _read_csv(run_dir / "pred_count.csv")
        best = _best_val(val)
        test_row = test.iloc[-1].to_dict() if not test.empty else {}
        pred_summary = _pred_row(pred)
        row = {
            "run_name": run_name,
            "checker_decision": _load_check(run_dir),
            **best,
            "test_macro_f1": float(test_row.get("test_macro_f1", test_row.get("macro_f1", np.nan))),
            "test_accuracy": float(test_row.get("test_accuracy", test_row.get("accuracy", np.nan))),
            "test_weighted_f1": float(test_row.get("test_weighted_f1", test_row.get("weighted_f1", np.nan))),
            "last_train_loss": _last(train, "train_loss"),
            "checkpoint_exists": (run_dir / "checkpoints" / "best.pt").exists(),
            "report_exists": (run_dir / "d13b_report.md").exists(),
            "pred_max_ratio": float(pred_summary.get("pred_max_ratio", np.nan)),
            "effective_slots_test": _last(slots, "effective_slots", split="test"),
            "slot_overlap_test": _last(slots, "slot_overlap", split="test"),
            "slot_entropy_test": _last(slots, "slot_entropy", split="test"),
            "slot_dominance_test": _last(slots, "slot_dominance", split="test"),
        }
        rows.append(row)
        slot_rows.append({"run_name": run_name, **{k: row[k] for k in row if k.startswith("slot_") or k.startswith("effective_slots")}})
        pred_rows.append({"run_name": run_name, **pred_summary})
        pcs = {"run_name": run_name}
        for key, value in test_row.items():
            if "f1_" in str(key).lower() or str(key).startswith("test_f1"):
                pcs[key] = value
        per_class_rows.append(pcs)
    return {
        "summary": pd.DataFrame(rows),
        "slot": pd.DataFrame(slot_rows),
        "pred": pd.DataFrame(pred_rows),
        "per_class": pd.DataFrame(per_class_rows),
    }


def _write_md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def _decision(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "D13B_DIAGNOSTIC_INCONCLUSIVE"
    if (summary.get("checker_decision", pd.Series()).astype(str).str.contains("FAIL")).any():
        return "D13B_DIAGNOSTIC_COLLAPSE_STOP"
    best = summary.sort_values("test_macro_f1", ascending=False).head(1)
    if not best.empty and float(best.iloc[0].get("test_macro_f1", 0.0)) >= 0.54:
        return "D13B_DIAGNOSTIC_BASELINE_PASS_READY_FOR_VISUAL_SLOT_AUDIT"
    return "D13B_DIAGNOSTIC_NEEDS_SLOT_REG_TUNING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = collect(root_dir)
    tables["summary"].to_csv(output_dir / "d13b_diagnostic_summary.csv", index=False)
    tables["slot"].to_csv(output_dir / "d13b_slot_health_summary.csv", index=False)
    tables["pred"].to_csv(output_dir / "d13b_pred_summary.csv", index=False)
    tables["per_class"].to_csv(output_dir / "d13b_per_class_summary.csv", index=False)
    decision = _decision(tables["summary"])
    ranked = tables["summary"].sort_values("test_macro_f1", ascending=False) if not tables["summary"].empty else tables["summary"]
    lines = [
        "# D13B Diagnostic Summary Report",
        "",
        "## 1. Context",
        "- D13B diagnostic only.",
        "- Base D13A is K144 visual base with caution.",
        "- No SupCon, no motif claim, no semantic-region claim.",
        "",
        "## 2. Accuracy Ranking",
        f"D13A K144 reference: test_macro_f1 = {D13A_K144_TEST_MACRO_F1:.4f}; test_acc = {D13A_K144_TEST_ACC:.4f}.",
        "",
        _write_md_table(ranked),
        "",
        "## 3. Slot Health",
        _write_md_table(tables["slot"]),
        "",
        "## 4. Collapse Check",
        _write_md_table(tables["pred"]),
        "",
        "## 5. Variant Analysis",
        "Compare M8 vs M16, no-reg vs regularized, strong reg, deep readout, deep region, K256 score control, and seed2 stability from the tables above.",
        "",
        "## 6. Recommendation",
        decision,
        "",
        "Do not open D13C or SupCon from this collector. Slot outputs are diagnostic candidates only.",
        "",
    ]
    (output_dir / "d13b_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "num_runs": int(len(tables["summary"]))}, indent=2))


if __name__ == "__main__":
    main()
