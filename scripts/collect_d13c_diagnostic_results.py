"""Collect D13C diagnostic outputs into summary tables and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.check_d13c_diagnostic_run import D13B_M16_TEST_ACC, D13B_M16_TEST_MACRO_F1, check_run


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _last(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _mean(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _pred_row(pred: pd.DataFrame) -> Dict[str, Any]:
    if pred.empty:
        return {"pred_max_ratio": float("nan")}
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    return work.iloc[-1].to_dict()


def _run_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return [
        path
        for path in sorted(root_dir.iterdir())
        if path.is_dir()
        and path.name != "summary"
        and (path / "train_log.csv").exists()
        and (path / "val_metrics.csv").exists()
    ]


def collect(root_dir: Path) -> Dict[str, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    supcon_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for run_dir in _run_dirs(root_dir):
        check = check_run(run_dir)
        train = _read_csv(run_dir / "train_log.csv")
        test = _read_csv(run_dir / "test_metrics.csv")
        slots = _read_csv(run_dir / "slot_stats.csv")
        supcon = _read_csv(run_dir / "supcon_stats.csv")
        pred = _read_csv(run_dir / "pred_count.csv")
        pred_summary = _pred_row(pred)
        row = {
            "run_name": run_dir.name,
            "checker_decision": check.get("decision", ""),
            "best_epoch": check.get("best_epoch", -1),
            "best_val_macro_f1": check.get("best_val_macro_f1", np.nan),
            "test_macro_f1": check.get("test_macro_f1", np.nan),
            "test_acc": check.get("test_acc", np.nan),
            "test_weighted_f1": check.get("test_weighted_f1", np.nan),
            "delta_macro_f1_vs_d13b_m16": float(check.get("test_macro_f1", np.nan)) - D13B_M16_TEST_MACRO_F1,
            "last_train_loss": _last(train, "train_loss"),
            "checkpoint_exists": bool(check.get("checkpoint_exists", False)),
            "pred_max_ratio": check.get("pred_max_ratio", np.nan),
            "effective_slots_test": check.get("effective_slots", np.nan),
            "slot_overlap_test": check.get("slot_overlap", np.nan),
            "slot_entropy_test": check.get("slot_entropy", np.nan),
            "slot_dominance_test": check.get("slot_dominance", np.nan),
            "supcon_loss_mean_train": check.get("supcon_loss_mean", np.nan),
            "positive_pair_count_mean_train": check.get("positive_pair_count_mean", np.nan),
            "embedding_collapse_score_test": check.get("embedding_collapse_score", np.nan),
        }
        summary_rows.append(row)
        supcon_rows.append(
            {
                "run_name": run_dir.name,
                "loss_supcon_train_mean": _mean(supcon, "loss_supcon", split="train"),
                "loss_supcon_train_last": _last(supcon, "loss_supcon", split="train"),
                "positive_pair_count_mean": _mean(supcon, "positive_pair_count", split="train"),
                "positive_pair_count_test": _last(supcon, "positive_pair_count", split="test"),
                "valid_supcon_anchor_count_mean": _mean(supcon, "valid_supcon_anchor_count", split="train"),
                "z_norm_mean_test": _last(supcon, "z_norm_mean", split="test"),
                "z_norm_std_test": _last(supcon, "z_norm_std", split="test"),
                "embedding_collapse_score_test": _last(supcon, "embedding_collapse_score", split="test"),
            }
        )
        slot_rows.append(
            {
                "run_name": run_dir.name,
                "effective_slots_test": _last(slots, "effective_slots", split="test"),
                "slot_overlap_test": _last(slots, "slot_overlap", split="test"),
                "slot_entropy_test": _last(slots, "slot_entropy", split="test"),
                "slot_dominance_test": _last(slots, "slot_dominance", split="test"),
                "slot_area_mean_test": _last(slots, "slot_area_mean", split="test"),
                "slot_center_std_test": _last(slots, "slot_center_std", split="test"),
            }
        )
        pred_rows.append({"run_name": run_dir.name, **pred_summary})
        per_class = {"run_name": run_dir.name}
        test_row = test.iloc[-1].to_dict() if not test.empty else {}
        for key, value in test_row.items():
            if str(key).startswith("test_f1_") or str(key).startswith("f1_"):
                per_class[key] = value
        per_class_rows.append(per_class)
    return {
        "summary": pd.DataFrame(summary_rows),
        "supcon": pd.DataFrame(supcon_rows),
        "slot": pd.DataFrame(slot_rows),
        "pred": pd.DataFrame(pred_rows),
        "per_class": pd.DataFrame(per_class_rows),
    }


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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
        return "D13C_INCONCLUSIVE"
    if summary["checker_decision"].astype(str).str.contains("FAIL_COLLAPSE", case=False, regex=False).any():
        return "D13C_COLLAPSE_STOP"
    if summary["checker_decision"].astype(str).str.contains("FAIL", case=False, regex=False).any():
        return "D13C_INCONCLUSIVE"
    ranked = summary.sort_values("test_macro_f1", ascending=False)
    best = ranked.iloc[0]
    if float(best.get("test_macro_f1", 0.0)) >= D13B_M16_TEST_MACRO_F1 + 0.001:
        return "D13C_DIAGNOSTIC_PASS_READY_FOR_POST_VISUAL_SLOT_AUDIT"
    if "d13c_m16_ce_continue" in set(summary["run_name"].astype(str)) and ranked.iloc[0]["run_name"] == "d13c_m16_ce_continue":
        return "D13C_SUPCON_NOT_HELPFUL_KEEP_D13B_FINAL"
    if float(best.get("test_macro_f1", 0.0)) >= D13B_M16_TEST_MACRO_F1 - 0.005:
        return "D13C_NEEDS_LAMBDA_TUNING"
    return "D13C_INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = collect(root_dir)
    tables["summary"].to_csv(output_dir / "d13c_diagnostic_summary.csv", index=False)
    tables["supcon"].to_csv(output_dir / "d13c_supcon_summary.csv", index=False)
    tables["slot"].to_csv(output_dir / "d13c_slot_health_summary.csv", index=False)
    tables["pred"].to_csv(output_dir / "d13c_pred_summary.csv", index=False)
    tables["per_class"].to_csv(output_dir / "d13c_per_class_summary.csv", index=False)

    summary = tables["summary"].sort_values("test_macro_f1", ascending=False) if not tables["summary"].empty else tables["summary"]
    decision = _decision(tables["summary"])
    lines = [
        "# D13C Diagnostic Summary Report",
        "",
        "## 1. Context",
        "- D13C diagnostic only.",
        "- Base D13B = M16 deep readout.",
        "- No prototype, no motif-level SupCon, no motif claim.",
        f"- D13B M16 reference: test_macro_f1 = {D13B_M16_TEST_MACRO_F1:.4f}; test_acc = {D13B_M16_TEST_ACC:.4f}.",
        "",
        "## 2. Accuracy Ranking",
        _md_table(summary),
        "",
        "## 3. SupCon Effect",
        "Compare CE-only continuation vs lambda variants: l001, l002, l005, l010, freeze, proj128, and M8 control.",
        "",
        _md_table(tables["supcon"]),
        "",
        "## 4. Representation Health",
        "Use positive pairs, SupCon loss, z-norm, and embedding collapse score above. Missing positive-pair signal blocks SupCon conclusions.",
        "",
        "## 5. Slot Health",
        _md_table(tables["slot"]),
        "",
        "## 6. Prediction Collapse",
        _md_table(tables["pred"]),
        "",
        "## 7. Per-class Behavior",
        "Hard classes to inspect: Angry, Disgust, Fear, Sad.",
        "",
        _md_table(tables["per_class"]),
        "",
        "## 8. Recommendation",
        decision,
        "",
        "Do not output OPEN_D13C_FULL, OPEN_SUPCON_FULL, or MOTIF_DISCOVERED from this collector.",
        "",
    ]
    (output_dir / "d13c_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "num_runs": int(len(tables["summary"]))}, indent=2))


if __name__ == "__main__":
    main()
