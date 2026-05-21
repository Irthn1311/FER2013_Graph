"""Check one D14 performance run for score gain and collapse risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


D13C_M8_MACRO = 0.6364
D13C_M8_ACC = 0.6481
D13C_M16_MACRO = 0.6277
D13C_M16_ACC = 0.6420
HARD_CLASSES = ["0_Angry", "1_Disgust", "2_Fear", "4_Sad"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _config(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _last(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _pred_ratio(pred: pd.DataFrame) -> float:
    if pred.empty:
        return float("nan")
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    row = work.iloc[-1].to_dict()
    if "pred_max_ratio" in row:
        return float(row["pred_max_ratio"])
    cols = [c for c in work.columns if c.startswith("pred_count_")]
    vals = [float(row.get(c, 0.0)) for c in cols]
    return float(max(vals) / max(sum(vals), 1.0)) if vals else float("nan")


def _hard_macro(row: Dict[str, Any]) -> float:
    vals = []
    for suffix in HARD_CLASSES:
        for prefix in ("test_f1_", "f1_"):
            key = prefix + suffix
            if key in row and pd.notna(row[key]):
                vals.append(float(row[key]))
                break
    return float(np.mean(vals)) if vals else float("nan")


def _base_for_run(run_name: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    text = (run_name + " " + str(cfg.get("model", {}).get("base_model", ""))).lower()
    if "m8" in text:
        return {"macro": D13C_M8_MACRO, "acc": D13C_M8_ACC, "name": "d13c_m8_supcon_l002_control"}
    return {"macro": D13C_M16_MACRO, "acc": D13C_M16_ACC, "name": "d13c_m16_supcon_l005"}


def check_run(output_dir: Path) -> Dict[str, Any]:
    train = _read_csv(output_dir / "train_log.csv")
    val = _read_csv(output_dir / "val_metrics.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    slots = _read_csv(output_dir / "slot_stats.csv")
    supcon = _read_csv(output_dir / "supcon_stats.csv")
    cfg = _config(output_dir)
    run_name = str(cfg.get("run", {}).get("config_name") or output_dir.name)
    base = _base_for_run(run_name, cfg)
    failures: List[str] = []
    warnings: List[str] = []
    for name, df in [("train_log.csv", train), ("val_metrics.csv", val), ("test_metrics.csv", test)]:
        if df.empty:
            failures.append(f"missing {name}")
    if not (output_dir / "checkpoints" / "best.pt").exists() and not (output_dir / "ensemble_metrics.csv").exists():
        failures.append("missing best checkpoint")

    test_row = test.iloc[-1].to_dict() if not test.empty else {}
    test_macro = float(test_row.get("test_macro_f1", test_row.get("macro_f1", np.nan)))
    test_acc = float(test_row.get("test_accuracy", test_row.get("accuracy", np.nan)))
    hard_macro = _hard_macro(test_row)
    pred_max_ratio = _pred_ratio(pred)
    effective_slots = _last(slots, "effective_slots", "test")
    slot_overlap = _last(slots, "slot_overlap", "test")
    collapse_score = _last(supcon, "embedding_collapse_score", "test")
    delta_macro = test_macro - float(base["macro"]) if np.isfinite(test_macro) else float("nan")
    delta_acc = test_acc - float(base["acc"]) if np.isfinite(test_acc) else float("nan")

    if np.isfinite(pred_max_ratio) and pred_max_ratio >= 0.9:
        failures.append("prediction hard collapse")
    elif np.isfinite(pred_max_ratio) and pred_max_ratio >= 0.75:
        warnings.append("prediction distribution collapse risk")
    if np.isfinite(effective_slots) and effective_slots < 4:
        failures.append("slot collapse risk: effective_slots < 4")
    if np.isfinite(slot_overlap) and slot_overlap > 0.85:
        failures.append("slot overlap too high")
    if np.isfinite(collapse_score) and collapse_score > 0.95:
        failures.append("embedding collapse score too high")

    if failures and any("collapse" in f for f in failures):
        decision = "D14_PERFORMANCE_FAIL_COLLAPSE"
    elif failures:
        decision = "D14_PERFORMANCE_NO_GAIN"
    elif np.isfinite(test_acc) and test_acc >= 0.70 and np.isfinite(delta_macro) and delta_macro > 0:
        decision = "D14_PERFORMANCE_PASS"
    elif np.isfinite(test_acc) and test_acc > 0.65 and np.isfinite(delta_macro) and delta_macro > 0:
        decision = "D14_PERFORMANCE_PROMISING"
    else:
        decision = "D14_PERFORMANCE_NO_GAIN"

    return {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "base_reference": base["name"],
        "base_macro_f1": base["macro"],
        "base_acc": base["acc"],
        "test_macro_f1": test_macro,
        "test_acc": test_acc,
        "test_weighted_f1": float(test_row.get("test_weighted_f1", test_row.get("weighted_f1", np.nan))),
        "hard_class_macro": hard_macro,
        "delta_macro_vs_base": delta_macro,
        "delta_acc_vs_base": delta_acc,
        "pred_max_ratio": pred_max_ratio,
        "effective_slots": effective_slots,
        "slot_overlap": slot_overlap,
        "embedding_collapse_score": collapse_score,
        "warnings": warnings,
        "failures": failures,
        "decision": decision,
        "performance_first": True,
        "no_evidence_gate": True,
    }


def write_outputs(output_dir: Path, summary: Dict[str, Any]) -> None:
    (output_dir / "d14_performance_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    warnings = [f"- {w}" for w in summary.get("warnings", [])] or ["- none"]
    failures = [f"- {f}" for f in summary.get("failures", [])] or ["- none"]
    lines = [
        "# D14 Performance Check Report",
        "",
        "D14 is performance-first. Slot stats are monitored for collapse only; this is not an evidence gate.",
        "",
        f"- decision: `{summary['decision']}`",
        f"- run_name: `{summary['run_name']}`",
        f"- test_macro_f1: {summary.get('test_macro_f1')}",
        f"- test_acc: {summary.get('test_acc')}",
        f"- hard_class_macro: {summary.get('hard_class_macro')}",
        f"- delta_macro_vs_base: {summary.get('delta_macro_vs_base')}",
        f"- delta_acc_vs_base: {summary.get('delta_acc_vs_base')}",
        f"- pred_max_ratio: {summary.get('pred_max_ratio')}",
        f"- effective_slots: {summary.get('effective_slots')}",
        f"- slot_overlap: {summary.get('slot_overlap')}",
        "",
        "## Warnings",
        *warnings,
        "",
        "## Failures",
        *failures,
        "",
        "No motif, semantic-region, causal-evidence, or full interpretability claim is made.",
    ]
    (output_dir / "d14_performance_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = check_run(output_dir)
    write_outputs(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
