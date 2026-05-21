"""Collect D14 performance sweep outputs."""

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

from scripts.check_d14_performance_run import D13C_M16_ACC, D13C_M16_MACRO, D13C_M8_ACC, D13C_M8_MACRO, check_run


HARD_KEYS = ["f1_0_Angry", "f1_1_Disgust", "f1_2_Fear", "f1_4_Sad"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _run_dirs(root_dir: Path) -> List[Path]:
    skip = {"summary", "smoke", "temp", "tmp"}
    if not root_dir.exists():
        return []
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir() and p.name not in skip and not p.name.startswith(".")]


def _ensemble_summary(run_dir: Path) -> Dict[str, Any]:
    df = _read_csv(run_dir / "ensemble_metrics.csv")
    if df.empty:
        return {}
    best = df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0].to_dict()
    macro = float(best.get("macro_f1", np.nan))
    acc = float(best.get("accuracy", np.nan))
    decision = "D14_REACHED_0P70_TARGET" if acc >= 0.70 else ("D14_ENSEMBLE_PROMISING" if acc > D13C_M8_ACC or macro > D13C_M8_MACRO else "D14_NO_GAIN_NEED_ARCHITECTURE_SHIFT")
    return {
        "run_name": run_dir.name,
        "variant_type": "ensemble_eval",
        "test_macro_f1": macro,
        "test_acc": acc,
        "test_weighted_f1": float(best.get("weighted_f1", np.nan)),
        "hard_class_macro": np.nan,
        "delta_macro_vs_m8_ref": macro - D13C_M8_MACRO if np.isfinite(macro) else np.nan,
        "delta_acc_vs_m8_ref": acc - D13C_M8_ACC if np.isfinite(acc) else np.nan,
        "checker_decision": decision,
        "best_method": best.get("method", ""),
        "pred_max_ratio": np.nan,
        "effective_slots": np.nan,
        "slot_overlap": np.nan,
    }


def _variant_type(name: str) -> str:
    if "ensemble" in name:
        return "ensemble_eval"
    if "aug" in name:
        return "augmentation"
    if "extend" in name:
        return "extend_training"
    if "deeper" in name:
        return "capacity"
    if "focal" in name or "class_weight" in name:
        return "hard_class_loss"
    return "other"


def collect(root_dir: Path) -> Dict[str, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for run_dir in _run_dirs(root_dir):
        ens = _ensemble_summary(run_dir)
        if ens:
            summary_rows.append(ens)
            pc = _read_csv(run_dir / "ensemble_per_class_metrics.csv")
            if not pc.empty:
                row = {"run_name": run_dir.name, **pc.iloc[0].to_dict()}
                per_class_rows.append(row)
            pred = _read_csv(run_dir / "ensemble_pred_count.csv")
            if not pred.empty:
                pred_rows.append({"run_name": run_dir.name, **pred.iloc[-1].to_dict()})
            continue
        if not (run_dir / "test_metrics.csv").exists():
            continue
        check = check_run(run_dir)
        row = {
            "run_name": check["run_name"],
            "variant_type": _variant_type(check["run_name"]),
            "test_macro_f1": check["test_macro_f1"],
            "test_acc": check["test_acc"],
            "test_weighted_f1": check["test_weighted_f1"],
            "hard_class_macro": check["hard_class_macro"],
            "delta_macro_vs_m8_ref": check["test_macro_f1"] - D13C_M8_MACRO,
            "delta_acc_vs_m8_ref": check["test_acc"] - D13C_M8_ACC,
            "delta_macro_vs_m16_ref": check["test_macro_f1"] - D13C_M16_MACRO,
            "delta_acc_vs_m16_ref": check["test_acc"] - D13C_M16_ACC,
            "base_reference": check["base_reference"],
            "checker_decision": check["decision"],
            "pred_max_ratio": check["pred_max_ratio"],
            "effective_slots": check["effective_slots"],
            "slot_overlap": check["slot_overlap"],
        }
        summary_rows.append(row)
        test = _read_csv(run_dir / "test_metrics.csv")
        if not test.empty:
            pc = {"run_name": check["run_name"]}
            for key, value in test.iloc[-1].to_dict().items():
                if str(key).startswith("test_f1_") or str(key).startswith("f1_"):
                    pc[str(key).replace("test_", "")] = value
            per_class_rows.append(pc)
            hard_rows.append({"run_name": check["run_name"], "hard_class_macro": check["hard_class_macro"], **{k: pc.get(k, np.nan) for k in HARD_KEYS}})
        pred = _read_csv(run_dir / "pred_count.csv")
        if not pred.empty:
            work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
            if not work.empty:
                pred_rows.append({"run_name": check["run_name"], **work.iloc[-1].to_dict()})
    return {
        "summary": pd.DataFrame(summary_rows),
        "per_class": pd.DataFrame(per_class_rows),
        "hard": pd.DataFrame(hard_rows),
        "pred": pd.DataFrame(pred_rows),
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
        return "D14_NO_GAIN_NEED_ARCHITECTURE_SHIFT"
    best = summary.sort_values(["test_acc", "test_macro_f1"], ascending=False).iloc[0]
    if float(best.get("test_acc", 0.0)) >= 0.70:
        return "D14_REACHED_0P70_TARGET"
    if str(best.get("variant_type", "")) == "ensemble_eval" and float(best.get("test_acc", 0.0)) > D13C_M8_ACC:
        return "D14_ENSEMBLE_PROMISING"
    if "m8" in str(best.get("run_name", "")).lower() and float(best.get("test_acc", 0.0)) > D13C_M8_ACC:
        return "D14_M8_BRANCH_PROMISING"
    if float(best.get("test_acc", 0.0)) > D13C_M8_ACC or float(best.get("test_macro_f1", 0.0)) > D13C_M8_MACRO:
        return "D14_IMPROVED_BUT_BELOW_0P70"
    return "D14_NO_GAIN_NEED_ARCHITECTURE_SHIFT"


def write_report(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    summary = tables["summary"].sort_values(["test_acc", "test_macro_f1"], ascending=False) if not tables["summary"].empty else tables["summary"]
    decision = _decision(summary)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# D14 Performance Sweep Report",
        "",
        "D14 is performance-first. Evidence gates are not used here; slot statistics are logged only to catch collapse. No motif, semantic-region, causal-evidence, or full interpretability claim is made.",
        "",
        "## References",
        f"- D13C M8 control: macro-F1 = {D13C_M8_MACRO:.4f}; acc = {D13C_M8_ACC:.4f}.",
        f"- D13C M16 l005: macro-F1 = {D13C_M16_MACRO:.4f}; acc = {D13C_M16_ACC:.4f}.",
        "",
        "## Overall Ranking",
        _md_table(summary),
        "",
        "## Questions",
        f"1. Any run above 0.65 acc: {'yes' if (not summary.empty and (summary['test_acc'] > 0.65).any()) else 'no or not available yet'}.",
        f"2. Any run near/reaching 0.70 acc: {'yes' if (not summary.empty and (summary['test_acc'] >= 0.69).any()) else 'no or not available yet'}.",
        "3. M8 vs M16: compare `delta_acc_vs_m8_ref`, `delta_acc_vs_m16_ref`, and top ranked run after all outputs are present.",
        "4. Augmentation: compare `augmentation` rows against extend-training rows.",
        "5. Extend training: compare `extend_training` rows against D13C references.",
        "6. Hard-class loss: inspect `d14_hard_class_summary.csv` and hard-class macro.",
        "7. Ensemble: inspect `ensemble_eval` rows; a strong ensemble should move accuracy toward 0.67-0.70 before longer architecture work.",
        "8. Next step to 0.70+: favor whichever axis produces a real gain without collapse, otherwise shift architecture/capacity rather than making evidence claims.",
        "",
        "## Best Run",
        json.dumps(best, indent=2),
        "",
        "## Decision",
        f"`{decision}`",
        "",
    ]
    (output_dir / "d14_performance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="outputs/d14_performance")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = collect(root_dir)
    tables["summary"].to_csv(output_dir / "d14_performance_summary.csv", index=False)
    tables["per_class"].to_csv(output_dir / "d14_per_class_summary.csv", index=False)
    tables["hard"].to_csv(output_dir / "d14_hard_class_summary.csv", index=False)
    tables["pred"].to_csv(output_dir / "d14_pred_summary.csv", index=False)
    write_report(output_dir, tables)
    print(json.dumps({"output_dir": str(output_dir), "runs": int(len(tables["summary"]))}, indent=2))


if __name__ == "__main__":
    main()
