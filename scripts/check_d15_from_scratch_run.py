"""Check a D15 from-scratch run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite_frame(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all()) if not numeric.empty else True


def _config(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _init_info(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "d15_from_scratch_init.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


def _pred_stats(pred: pd.DataFrame) -> Dict[str, Any]:
    if pred.empty:
        return {"pred_max_ratio": float("nan"), "classes_predicted_count": 0}
    work = pred[pred["split"].astype(str) == "test"] if "split" in pred else pred.tail(1)
    if work.empty:
        work = pred.tail(1)
    row = work.iloc[-1].to_dict()
    count_cols = [c for c in work.columns if c.startswith("pred_count_")]
    counts = [int(row.get(c, 0)) for c in count_cols]
    max_ratio = float(row.get("pred_max_ratio", max(counts) / max(sum(counts), 1) if counts else np.nan))
    return {
        "pred_max_ratio": max_ratio,
        "classes_predicted_count": int(sum(1 for v in counts if v > 0)),
    }


def _hard_macro(row: Dict[str, Any]) -> float:
    vals = []
    for suffix in ("0_Angry", "1_Disgust", "2_Fear", "4_Sad"):
        for prefix in ("test_f1_", "f1_"):
            key = prefix + suffix
            if key in row and pd.notna(row[key]):
                vals.append(float(row[key]))
                break
    return float(np.mean(vals)) if vals else float("nan")


def _checkpoint_fields(cfg: Dict[str, Any]) -> List[str]:
    bad = []
    for section in ("model", "training"):
        sec = cfg.get(section, {}) or {}
        for key in ("init_checkpoint", "init_d13b_checkpoint", "resume_from", "checkpoint", "pretrained_checkpoint"):
            value = sec.get(key)
            if value not in (None, "", "null", False):
                bad.append(f"{section}.{key}={value}")
    return bad


def check_run(output_dir: Path) -> Dict[str, Any]:
    cfg = _config(output_dir)
    init = _init_info(output_dir)
    train = _read_csv(output_dir / "train_log.csv")
    val = _read_csv(output_dir / "val_metrics.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    slots = _read_csv(output_dir / "slot_stats.csv")
    pooling = _read_csv(output_dir / "pooling_stats.csv")
    supcon = _read_csv(output_dir / "supcon_stats.csv")
    run_name = str(cfg.get("run", {}).get("config_name") or output_dir.name)
    failures: List[str] = []
    warnings: List[str] = []

    bad_checkpoint_fields = _checkpoint_fields(cfg)
    if bad_checkpoint_fields:
        failures.append("checkpoint fields present: " + "; ".join(bad_checkpoint_fields))
    if not bool(cfg.get("from_scratch", cfg.get("d15", {}).get("from_scratch", False))):
        failures.append("from_scratch flag is not true")
    if init.get("loaded_keys", 0) not in (0, "0"):
        failures.append("loaded_keys > 0")
    if init and not bool(init.get("from_scratch", False)):
        failures.append("d15 init info does not confirm from_scratch")
    if not (output_dir / "checkpoints" / "best.pt").exists():
        failures.append("missing best checkpoint")
    for name, df in [("train_log.csv", train), ("val_metrics.csv", val), ("test_metrics.csv", test), ("slot_stats.csv", slots), ("supcon_stats.csv", supcon)]:
        if df.empty:
            failures.append(f"missing {name}")
        elif not _finite_frame(df):
            failures.append(f"non-finite numeric values in {name}")

    best = _best_val(val)
    test_row = test.iloc[-1].to_dict() if not test.empty else {}
    test_macro = float(test_row.get("test_macro_f1", test_row.get("macro_f1", np.nan)))
    test_acc = float(test_row.get("test_accuracy", test_row.get("accuracy", np.nan)))
    test_weighted = float(test_row.get("test_weighted_f1", test_row.get("weighted_f1", np.nan)))
    pred_stats = _pred_stats(pred)
    effective_slots = _last(slots, "effective_slots", "test")
    slot_overlap = _last(slots, "slot_overlap", "test")
    assignment_entropy = _last(pooling, "assignment_entropy", "test")
    effective_regions = _last(pooling, "effective_regions", "test")
    positive_pairs = _last(supcon, "positive_pair_count", "train")
    lambda_last = _last(supcon, "lambda_supcon_current", "train")
    hard_macro = _hard_macro(test_row)

    if np.isfinite(pred_stats["pred_max_ratio"]) and pred_stats["pred_max_ratio"] >= 0.9:
        failures.append("prediction hard collapse")
    elif np.isfinite(pred_stats["pred_max_ratio"]) and pred_stats["pred_max_ratio"] >= 0.75:
        warnings.append("prediction distribution collapse risk")
    if int(pred_stats.get("classes_predicted_count", 0)) < 4:
        failures.append("fewer than four predicted classes")
    if np.isfinite(effective_slots) and effective_slots < 4:
        failures.append("slot collapse risk")
    if np.isfinite(slot_overlap) and slot_overlap > 0.85:
        failures.append("slot overlap too high")

    if any("checkpoint" in f or "loaded_keys" in f or "from_scratch" in f for f in failures):
        decision = "D15_SCRATCH_INVALID_USED_CHECKPOINT"
    elif any("collapse" in f or "non-finite" in f or "classes" in f for f in failures):
        decision = "D15_SCRATCH_FAIL_COLLAPSE"
    elif np.isfinite(test_acc) and test_acc >= 0.70:
        decision = "D15_SCRATCH_TARGET_REACHED_0P70"
    elif np.isfinite(test_acc) and test_acc >= 0.68:
        decision = "D15_SCRATCH_PROMISING"
    elif np.isfinite(test_acc) and test_acc >= 0.65:
        decision = "D15_SCRATCH_PASS_BASELINE"
    else:
        decision = "D15_SCRATCH_UNDER_TARGET"

    return {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "from_scratch": bool(cfg.get("from_scratch", cfg.get("d15", {}).get("from_scratch", False))),
        "init_checkpoint": None,
        "loaded_keys": int(init.get("loaded_keys", -1)) if init else -1,
        "best_val_macro_f1": best["best_val_macro_f1"],
        "best_epoch": best["best_epoch"],
        "test_macro_f1": test_macro,
        "test_acc": test_acc,
        "test_weighted_f1": test_weighted,
        "hard_class_macro": hard_macro,
        "pred_max_ratio": pred_stats["pred_max_ratio"],
        "classes_predicted_count": pred_stats["classes_predicted_count"],
        "effective_slots": effective_slots,
        "slot_overlap": slot_overlap,
        "assignment_entropy": assignment_entropy,
        "effective_regions": effective_regions,
        "positive_pair_count_last_train": positive_pairs,
        "lambda_supcon_current_last_train": lambda_last,
        "checkpoint_exists": (output_dir / "checkpoints" / "best.pt").exists(),
        "warnings": warnings,
        "failures": failures,
        "decision": decision,
    }


def write_outputs(output_dir: Path, summary: Dict[str, Any]) -> None:
    (output_dir / "d15_from_scratch_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    warnings = [f"- {w}" for w in summary.get("warnings", [])] or ["- none"]
    failures = [f"- {f}" for f in summary.get("failures", [])] or ["- none"]
    lines = [
        "# D15 From-Scratch Check Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- from_scratch: {summary.get('from_scratch')}",
        f"- init_checkpoint: {summary.get('init_checkpoint')}",
        f"- loaded_keys: {summary.get('loaded_keys')}",
        f"- best_val_macro_f1: {summary.get('best_val_macro_f1')}",
        f"- test_macro_f1: {summary.get('test_macro_f1')}",
        f"- test_acc: {summary.get('test_acc')}",
        f"- hard_class_macro: {summary.get('hard_class_macro')}",
        f"- pred_max_ratio: {summary.get('pred_max_ratio')}",
        f"- classes_predicted_count: {summary.get('classes_predicted_count')}",
        f"- effective_slots: {summary.get('effective_slots')}",
        f"- slot_overlap: {summary.get('slot_overlap')}",
        "",
        "## Warnings",
        *warnings,
        "",
        "## Failures",
        *failures,
        "",
        "D15 is performance-first. No motif, semantic-region, causal-evidence, or full interpretability claim is made.",
    ]
    (output_dir / "d15_from_scratch_check_report.md").write_text("\n".join(lines), encoding="utf-8")


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
