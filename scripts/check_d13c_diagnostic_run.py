"""Check a D13C diagnostic run for SupCon signal and collapse risks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


D13B_M16_TEST_MACRO_F1 = 0.6187
D13B_M16_TEST_ACC = 0.6328


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite_frame(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all()) if not numeric.empty else True


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


def _min(df: pd.DataFrame, col: str, split: str | None = None) -> float:
    if df.empty or col not in df:
        return float("nan")
    work = df
    if split is not None and "split" in work:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.min()) if not vals.empty else float("nan")


def _pred_max_ratio(pred: pd.DataFrame) -> float:
    if pred.empty:
        return float("nan")
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    row = work.iloc[-1].to_dict()
    if "pred_max_ratio" in row:
        return float(row["pred_max_ratio"])
    count_cols = [c for c in work.columns if c.startswith("pred_count_")]
    counts = [float(row.get(c, 0.0)) for c in count_cols]
    return float(max(counts) / max(sum(counts), 1.0)) if counts else float("nan")


def _best_val(val: pd.DataFrame) -> Dict[str, Any]:
    if val.empty:
        return {"best_val_macro_f1": float("nan"), "best_epoch": -1}
    metric = "val_macro_f1" if "val_macro_f1" in val else "macro_f1"
    if metric not in val:
        return {"best_val_macro_f1": float("nan"), "best_epoch": -1}
    vals = pd.to_numeric(val[metric], errors="coerce")
    idx = vals.idxmax()
    return {
        "best_val_macro_f1": float(vals.loc[idx]),
        "best_epoch": int(val.loc[idx, "epoch"]) if "epoch" in val else int(idx),
    }


def _config(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def check_run(output_dir: Path) -> Dict[str, Any]:
    train = _read_csv(output_dir / "train_log.csv")
    val = _read_csv(output_dir / "val_metrics.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    slots = _read_csv(output_dir / "slot_stats.csv")
    pooling = _read_csv(output_dir / "pooling_stats.csv")
    supcon = _read_csv(output_dir / "supcon_stats.csv")
    report_exists = (output_dir / "d13c_report.md").exists()
    checkpoint_exists = (output_dir / "checkpoints" / "best.pt").exists()
    cfg = _config(output_dir)
    num_slots = int(cfg.get("model", {}).get("num_slots", 16))
    lambda_supcon = float(cfg.get("loss", {}).get("lambda_supcon", 0.0))

    failures: List[str] = []
    warnings: List[str] = []
    if train.empty:
        failures.append("missing train_log.csv")
    if val.empty:
        failures.append("missing val_metrics.csv")
    if test.empty:
        failures.append("missing test_metrics.csv")
    if slots.empty:
        failures.append("missing slot_stats.csv")
    if supcon.empty:
        failures.append("missing supcon_stats.csv")
    if not checkpoint_exists:
        failures.append("missing best checkpoint")
    for name, df in [("train", train), ("val", val), ("test", test), ("slot", slots), ("pooling", pooling), ("supcon", supcon)]:
        if not df.empty and not _finite_frame(df):
            failures.append(f"non-finite numeric values in {name}")

    best = _best_val(val)
    test_row = test.iloc[-1].to_dict() if not test.empty else {}
    test_macro = float(test_row.get("test_macro_f1", test_row.get("macro_f1", np.nan)))
    test_acc = float(test_row.get("test_accuracy", test_row.get("accuracy", np.nan)))
    test_weighted = float(test_row.get("test_weighted_f1", test_row.get("weighted_f1", np.nan)))
    pred_ratio = _pred_max_ratio(pred)
    effective_slots = _last(slots, "effective_slots", split="test")
    slot_overlap = _last(slots, "slot_overlap", split="test")
    slot_entropy = _last(slots, "slot_entropy", split="test")
    slot_dominance = _last(slots, "slot_dominance", split="test")
    supcon_loss_mean = _mean(supcon, "loss_supcon", split="train")
    supcon_loss_last = _last(supcon, "loss_supcon", split="train")
    positive_pair_mean = _mean(supcon, "positive_pair_count", split="train")
    positive_pair_min = _min(supcon, "positive_pair_count", split="train")
    valid_anchor_mean = _mean(supcon, "valid_supcon_anchor_count", split="train")
    collapse_score = _last(supcon, "embedding_collapse_score", split="test")

    if np.isfinite(pred_ratio) and pred_ratio >= 0.9:
        failures.append("prediction hard collapse")
    elif np.isfinite(pred_ratio) and pred_ratio >= 0.75:
        warnings.append("prediction distribution biased")
    if np.isfinite(effective_slots) and effective_slots < 0.5 * num_slots:
        failures.append("effective slots below half of configured slots")
    if np.isfinite(test_macro) and test_macro < 0.58:
        failures.append("test macro-F1 below D13C fail floor")
    elif np.isfinite(test_macro) and test_macro < D13B_M16_TEST_MACRO_F1 - 0.01:
        warnings.append("test macro-F1 drops more than 0.01 vs D13B M16")
    if lambda_supcon > 0 and (not np.isfinite(positive_pair_mean) or positive_pair_mean <= 0):
        warnings.append("SupCon has no positive-pair signal")
    if lambda_supcon > 0 and np.isfinite(positive_pair_min) and positive_pair_min <= 0:
        warnings.append("some train batches have no SupCon positive pairs")
    if np.isfinite(slot_overlap) and slot_overlap > 0.85:
        failures.append("slot overlap very high")
    elif np.isfinite(slot_overlap) and slot_overlap > 0.70:
        warnings.append("slot overlap high")
    if np.isfinite(collapse_score) and collapse_score >= 0.95:
        failures.append("embedding collapse severe")
    elif np.isfinite(collapse_score) and collapse_score >= 0.85:
        warnings.append("embedding collapse moderate")
    if not report_exists:
        warnings.append("missing d13c_report.md")

    if failures:
        if any("non-finite" in item or "missing train" in item or "missing best" in item for item in failures):
            decision = "D13C_DIAGNOSTIC_FAIL_TRAINING"
        elif any("collapse" in item or "slots" in item for item in failures):
            decision = "D13C_DIAGNOSTIC_FAIL_COLLAPSE"
        else:
            decision = "D13C_DIAGNOSTIC_FAIL_TRAINING"
    elif lambda_supcon > 0 and (not np.isfinite(positive_pair_mean) or positive_pair_mean <= 0):
        decision = "D13C_DIAGNOSTIC_NO_SUPCON_SIGNAL"
    elif warnings:
        decision = "D13C_DIAGNOSTIC_WARN_REVIEW"
    elif np.isfinite(test_macro) and test_macro >= D13B_M16_TEST_MACRO_F1 - 0.005:
        decision = "D13C_DIAGNOSTIC_PASS"
    else:
        decision = "D13C_DIAGNOSTIC_WARN_REVIEW"

    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": output_dir.name,
        "best_val_macro_f1": best["best_val_macro_f1"],
        "best_epoch": best["best_epoch"],
        "test_macro_f1": test_macro,
        "test_acc": test_acc,
        "test_weighted_f1": test_weighted,
        "d13b_m16_ref_macro_f1": D13B_M16_TEST_MACRO_F1,
        "d13b_m16_ref_acc": D13B_M16_TEST_ACC,
        "pred_max_ratio": pred_ratio,
        "effective_slots": effective_slots,
        "slot_overlap": slot_overlap,
        "slot_entropy": slot_entropy,
        "slot_dominance": slot_dominance,
        "supcon_loss_mean": supcon_loss_mean,
        "supcon_loss_last": supcon_loss_last,
        "positive_pair_count_mean": positive_pair_mean,
        "positive_pair_count_min": positive_pair_min,
        "valid_supcon_anchor_count_mean": valid_anchor_mean,
        "embedding_collapse_score": collapse_score,
        "checkpoint_exists": checkpoint_exists,
        "report_exists": report_exists,
        "warnings": warnings,
        "failures": failures,
        "decision": decision,
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
        "no_causal_claim": True,
    }
    return summary


def _write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    warning_lines = [f"- {w}" for w in summary.get("warnings", [])] or ["- none"]
    failure_lines = [f"- {f}" for f in summary.get("failures", [])] or ["- none"]
    lines = [
        "# D13C Diagnostic Check Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- best_val_macro_f1: {summary.get('best_val_macro_f1')}",
        f"- test_macro_f1: {summary.get('test_macro_f1')}",
        f"- test_acc: {summary.get('test_acc')}",
        f"- pred_max_ratio: {summary.get('pred_max_ratio')}",
        f"- effective_slots: {summary.get('effective_slots')}",
        f"- slot_overlap: {summary.get('slot_overlap')}",
        f"- slot_entropy: {summary.get('slot_entropy')}",
        f"- slot_dominance: {summary.get('slot_dominance')}",
        f"- supcon_loss_mean: {summary.get('supcon_loss_mean')}",
        f"- positive_pair_count_mean: {summary.get('positive_pair_count_mean')}",
        f"- embedding_collapse_score: {summary.get('embedding_collapse_score')}",
        "",
        "## Warnings",
        *warning_lines,
        "",
        "## Failures",
        *failure_lines,
        "",
        "D13C diagnostic only. No full D13C, no full SupCon, no prototype, no motif discovery, no semantic-region claim, and no causal-evidence claim.",
        "",
    ]
    (output_dir / "d13c_diagnostic_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = check_run(output_dir)
    (output_dir / "d13c_diagnostic_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
