"""Collect D15 from-scratch outputs and compare to checkpoint baselines."""

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

from scripts.check_d15_from_scratch_run import check_run


D13C_M8_ACC = 0.6481
D13C_M8_MACRO = 0.6364
D13C_M16_ACC = 0.6420
D13C_M16_MACRO = 0.6277


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _run_dirs(root_dir: Path) -> List[Path]:
    skip = {"summary", "smoke", "temp", "tmp"}
    if not root_dir.exists():
        return []
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir() and p.name not in skip and not p.name.startswith(".")]


def _last(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _slope_last(df: pd.DataFrame, col: str, n: int = 10) -> float:
    if df.empty or col not in df:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna().tail(int(n)).to_numpy(dtype=float)
    if vals.size < 2:
        return float("nan")
    x = np.arange(vals.size, dtype=float)
    return float(np.polyfit(x, vals, 1)[0])


def _variant_type(run_name: str) -> str:
    name = run_name.lower()
    if "curriculum" in name:
        return "curriculum"
    if "aug" in name:
        return "augmentation"
    if "focal" in name or "class_weight" in name:
        return "hard_class_loss"
    if "deeper" in name:
        return "capacity"
    if "m16" in name:
        return "m16_basic"
    if "m8" in name:
        return "m8_basic"
    return "other"


def collect(root_dir: Path) -> Dict[str, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    dynamics_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    for run_dir in _run_dirs(root_dir):
        if not (run_dir / "train_log.csv").exists():
            continue
        check = check_run(run_dir)
        train = _read_csv(run_dir / "train_log.csv")
        val = _read_csv(run_dir / "val_metrics.csv")
        test = _read_csv(run_dir / "test_metrics.csv")
        pred = _read_csv(run_dir / "pred_count.csv")
        slots = _read_csv(run_dir / "slot_stats.csv")
        pooling = _read_csv(run_dir / "pooling_stats.csv")
        supcon = _read_csv(run_dir / "supcon_stats.csv")
        row = {
            "run_name": check["run_name"],
            "track": "D15_from_scratch",
            "variant_type": _variant_type(check["run_name"]),
            "from_scratch": check["from_scratch"],
            "loaded_keys": check["loaded_keys"],
            "best_epoch": check["best_epoch"],
            "best_val_macro_f1": check["best_val_macro_f1"],
            "test_macro_f1": check["test_macro_f1"],
            "test_acc": check["test_acc"],
            "test_weighted_f1": check["test_weighted_f1"],
            "hard_class_macro": check["hard_class_macro"],
            "delta_acc_vs_d13c_m8_checkpoint": check["test_acc"] - D13C_M8_ACC,
            "delta_macro_vs_d13c_m8_checkpoint": check["test_macro_f1"] - D13C_M8_MACRO,
            "delta_acc_vs_d13c_m16_checkpoint": check["test_acc"] - D13C_M16_ACC,
            "delta_macro_vs_d13c_m16_checkpoint": check["test_macro_f1"] - D13C_M16_MACRO,
            "pred_max_ratio": check["pred_max_ratio"],
            "classes_predicted_count": check["classes_predicted_count"],
            "effective_slots": check["effective_slots"],
            "slot_overlap": check["slot_overlap"],
            "checker_decision": check["decision"],
        }
        summary_rows.append(row)
        dynamics_rows.append(
            {
                "run_name": check["run_name"],
                "train_loss_first": _last(train.head(1), "train_loss"),
                "train_loss_last": _last(train, "train_loss"),
                "train_loss_slope_last10": _slope_last(train, "train_loss"),
                "val_macro_best": check["best_val_macro_f1"],
                "val_macro_last": _last(val, "val_macro_f1"),
                "val_macro_slope_last10": _slope_last(val, "val_macro_f1"),
                "lambda_supcon_last_train": _last(supcon, "lambda_supcon_current", "train"),
                "positive_pair_count_last_train": _last(supcon, "positive_pair_count", "train"),
            }
        )
        pc = {"run_name": check["run_name"]}
        test_row = test.iloc[-1].to_dict() if not test.empty else {}
        for key, value in test_row.items():
            if str(key).startswith("test_f1_") or str(key).startswith("f1_"):
                pc[str(key).replace("test_", "")] = value
        per_class_rows.append(pc)
        if not pred.empty:
            work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
            if not work.empty:
                pred_rows.append({"run_name": check["run_name"], **work.iloc[-1].to_dict()})
        slot_rows.append(
            {
                "run_name": check["run_name"],
                "effective_slots_test": _last(slots, "effective_slots", "test"),
                "slot_overlap_test": _last(slots, "slot_overlap", "test"),
                "slot_entropy_test": _last(slots, "slot_entropy", "test"),
                "slot_dominance_test": _last(slots, "slot_dominance", "test"),
                "assignment_entropy_test": _last(pooling, "assignment_entropy", "test"),
                "effective_regions_test": _last(pooling, "effective_regions", "test"),
            }
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "dynamics": pd.DataFrame(dynamics_rows),
        "per_class": pd.DataFrame(per_class_rows),
        "pred": pd.DataFrame(pred_rows),
        "slot": pd.DataFrame(slot_rows),
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


def _read_d14_refs() -> pd.DataFrame:
    path = Path("outputs/d14_performance/summary/d14_performance_summary.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _decision(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "D15_NO_RESULTS_YET"
    best = summary.sort_values(["test_acc", "test_macro_f1"], ascending=False).iloc[0]
    acc = float(best.get("test_acc", 0.0))
    if acc >= 0.70:
        return "D15_SCRATCH_TARGET_REACHED_0P70"
    if acc >= 0.68:
        return "D15_SCRATCH_PROMISING"
    if acc >= 0.65:
        return "D15_SCRATCH_PASS_BASELINE"
    return "D15_SCRATCH_UNDER_TARGET"


def write_report(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    summary = tables["summary"].sort_values(["test_acc", "test_macro_f1"], ascending=False) if not tables["summary"].empty else tables["summary"]
    d14 = _read_d14_refs()
    decision = _decision(summary)
    lines = [
        "# D15 From-Scratch Summary Report",
        "",
        "## 1. Context",
        "D15 is the main from-scratch end-to-end performance track. It does not load D13/D14 checkpoints and is not a continuation run.",
        "",
        "## 2. From-Scratch Verification",
        "Runs are invalid if `from_scratch` is not true, checkpoint fields are present, or `loaded_keys > 0`.",
        _md_table(summary[["run_name", "from_scratch", "loaded_keys", "checker_decision"]] if not summary.empty else summary),
        "",
        "## 3. Accuracy Ranking",
        _md_table(summary),
        "",
        "## 4. M8 vs M16",
        "Compare basic/curriculum/augmentation rows by slot count. D15 must be compared separately from checkpoint-based D14/D13C results.",
        "",
        "## 5. Curriculum Effect",
        "Compare `curriculum` rows against `m8_basic` and `m16_basic` rows.",
        "",
        "## 6. Augmentation Effect",
        "Compare `augmentation` rows against matching basic rows.",
        "",
        "## 7. Class-Weight/Focal Effect",
        "Inspect hard-class macro and per-class table for the hard-class-loss variant.",
        "",
        "## 8. Capacity Effect",
        "Compare the deeper-readout row against M8 basic/curriculum.",
        "",
        "## 9. Comparison To D13C Checkpoint Results",
        f"- D13C M8 checkpoint: acc = {D13C_M8_ACC:.4f}; macro-F1 = {D13C_M8_MACRO:.4f}.",
        f"- D13C M16 l005 checkpoint: acc = {D13C_M16_ACC:.4f}; macro-F1 = {D13C_M16_MACRO:.4f}.",
        "D15 rows include deltas against both checkpoint baselines.",
        "",
        "## 10. D14 Minimal References",
        _md_table(d14) if not d14.empty else "D14 summary not found yet. Run minimal D14 references and collector when available.",
        "",
        "## 11. Target Status",
        f"`{decision}`",
        "",
        "## 12. Next Action",
        "If D15 remains below 0.65 while D14 improves, from-scratch optimization is the bottleneck. If D15 nears D14, scale the best D15 recipe. If both tracks stall below 0.70, shift architecture/data/training recipe.",
        "",
        "No motif, semantic-region, causal-evidence, or full interpretability claim is made.",
    ]
    (output_dir / "d15_from_scratch_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="outputs/d15_from_scratch")
    parser.add_argument("--output_dir", default="outputs/d15_from_scratch/summary")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = collect(Path(args.root_dir))
    tables["summary"].to_csv(output_dir / "d15_from_scratch_summary.csv", index=False)
    tables["dynamics"].to_csv(output_dir / "d15_learning_dynamics.csv", index=False)
    tables["per_class"].to_csv(output_dir / "d15_per_class_summary.csv", index=False)
    tables["pred"].to_csv(output_dir / "d15_pred_summary.csv", index=False)
    tables["slot"].to_csv(output_dir / "d15_slot_health_summary.csv", index=False)
    write_report(output_dir, tables)
    print(json.dumps({"output_dir": str(output_dir), "runs": int(len(tables["summary"]))}, indent=2))


if __name__ == "__main__":
    main()
