"""Deep read-only analysis for D13A ablation outputs.

This script analyzes learning dynamics, reduction health, prediction
distribution, and per-class trade-offs. It does not train or modify model
artifacts and makes no motif or semantic-region claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - figures are optional in limited envs
    plt = None


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
HARD_CLASSES = ["Angry", "Disgust", "Fear", "Sad"]
BASELINE_RUN_NAME = "d13a_edgeaware_lite_localpool_k144_outputs"


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def slope_last(df: Optional[pd.DataFrame], metric: str, n: int) -> Optional[float]:
    if df is None or metric not in df or "epoch" not in df:
        return None
    sub = df[["epoch", metric]].dropna().tail(n)
    if len(sub) < 2:
        return None
    x = sub["epoch"].to_numpy(dtype=float)
    y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def std_last(df: Optional[pd.DataFrame], metric: str, n: int) -> Optional[float]:
    if df is None or metric not in df:
        return None
    vals = pd.to_numeric(df[metric], errors="coerce").dropna().tail(n)
    if len(vals) == 0:
        return None
    return float(vals.std(ddof=0))


def value_at_epoch(df: Optional[pd.DataFrame], metric: str, epoch: int) -> Optional[float]:
    if df is None or metric not in df or "epoch" not in df:
        return None
    rows = df[pd.to_numeric(df["epoch"], errors="coerce") == int(epoch)]
    if rows.empty:
        return None
    return as_float(rows.iloc[-1].get(metric))


def last_value(df: Optional[pd.DataFrame], metric: str) -> Optional[float]:
    if df is None or metric not in df:
        return None
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def best_row(df: Optional[pd.DataFrame], metric: str) -> Tuple[Optional[pd.Series], Optional[int], Optional[float]]:
    if df is None or metric not in df:
        return None, None, None
    vals = pd.to_numeric(df[metric], errors="coerce")
    if vals.dropna().empty:
        return None, None, None
    idx = vals.idxmax()
    row = df.loc[idx]
    return row, as_int(row.get("epoch")), as_float(row.get(metric))


def pred_entropy(counts: List[int]) -> Optional[float]:
    total = float(sum(counts))
    if total <= 0:
        return None
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    if len(p) == 0:
        return None
    return float(-(p * np.log(p)).sum() / math.log(len(EMOTION_NAMES)))


def infer_run_type(run_name: str, cfg: Dict[str, Any]) -> str:
    n = run_name.lower()
    if "gine" in n:
        return "gine_control"
    if "k64" in n:
        return "k_sweep_k64"
    if "k256" in n:
        return "k_sweep_k256"
    if "temp07" in n:
        return "temperature_temp07"
    if "temp05" in n:
        return "temperature_temp05"
    if "anneal" in n:
        return "temperature_anneal"
    if "no_aux" in n:
        return "pool_loss_no_aux"
    if "compact_balance_x2" in n:
        return "pool_loss_compact_balance_x2"
    if "seed2" in n:
        return "seed2"
    if "seed3" in n:
        return "seed3"
    if "lr1e4" in n:
        return "lr1e4"
    return "baseline_k144" if "k144" in n else "unknown"


def find_run_dirs(root_dir: Path, baseline_dir: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    if root_dir.exists():
        candidates.extend(p for p in sorted(root_dir.iterdir()) if p.is_dir())
    else:
        parent = root_dir.parent
        if parent.exists():
            candidates.extend(
                p for p in sorted(parent.iterdir())
                if p.is_dir() and p.name.startswith("d13a_") and p.name.endswith("_outputs")
            )
    if baseline_dir is not None and baseline_dir.exists():
        candidates.append(baseline_dir)
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if (p / "train_log.csv").exists() or (p / "val_metrics.csv").exists() or (p / "test_metrics.csv").exists():
            out.append(p)
    return out


def load_run(run_dir: Path) -> Dict[str, Any]:
    cfg = read_yaml(run_dir / "resolved_config.yaml")
    model_cfg = cfg.get("model", {})
    pool_cfg = model_cfg.get("pooling", {})
    loss_cfg = cfg.get("loss", {})
    training_cfg = cfg.get("training", {})
    optimizer_cfg = cfg.get("optimizer", {})
    grid_size = as_int(pool_cfg.get("grid_size"))
    k_regions = grid_size * grid_size if grid_size else None
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "cfg": cfg,
        "train": read_csv(run_dir / "train_log.csv"),
        "val": read_csv(run_dir / "val_metrics.csv"),
        "test": read_csv(run_dir / "test_metrics.csv"),
        "pool": read_csv(run_dir / "pooling_stats.csv"),
        "pred": read_csv(run_dir / "pred_count.csv"),
        "cm": read_csv(run_dir / "confusion_matrix.csv"),
        "check": read_json(run_dir / "d13_debug_check_summary.json"),
        "config_type": infer_run_type(run_dir.name, cfg),
        "grid_size": grid_size,
        "k_regions": k_regions,
        "assignment_temperature_config": pool_cfg.get("assignment_temperature"),
        "assignment_temperature_start": pool_cfg.get("assignment_temperature_start"),
        "assignment_temperature_end": pool_cfg.get("assignment_temperature_end"),
        "assignment_temperature_anneal_epochs": pool_cfg.get("assignment_temperature_anneal_epochs"),
        "pool_entropy_weight": loss_cfg.get("pool_entropy_weight"),
        "pool_balance_weight": loss_cfg.get("pool_balance_weight"),
        "pool_compactness_weight": loss_cfg.get("pool_compactness_weight"),
        "pool_area_weight": loss_cfg.get("pool_area_weight"),
        "seed": training_cfg.get("seed"),
        "lr": optimizer_cfg.get("lr"),
        "max_epochs_config": training_cfg.get("epochs", training_cfg.get("max_epochs")),
        "checkpoint_best_exists": (run_dir / "checkpoints" / "best.pt").exists() or (run_dir / "checkpoints" / "best.pth").exists(),
        "checkpoint_last_exists": (run_dir / "checkpoints" / "last.pt").exists() or (run_dir / "checkpoints" / "last.pth").exists(),
        "report_exists": (run_dir / "d13a_report.md").exists(),
    }


def learning_dynamics(run: Dict[str, Any]) -> Dict[str, Any]:
    train = run["train"]
    val = run["val"]
    best, best_epoch, best_val_macro = best_row(val, "val_macro_f1")
    max_epoch = as_int(train["epoch"].max()) if train is not None and "epoch" in train else as_int(run.get("max_epochs_config"))
    train_first = as_float(train.iloc[0].get("train_loss")) if train is not None and "train_loss" in train else None
    train_last = last_value(train, "train_loss")
    train_min = as_float(pd.to_numeric(train["train_loss"], errors="coerce").min()) if train is not None and "train_loss" in train else None
    val_last = last_value(val, "val_macro_f1")
    val_acc_row, _, val_acc_best = best_row(val, "val_accuracy")
    val_acc_last = last_value(val, "val_accuracy")
    train_slope5 = slope_last(train, "train_loss", 5)
    train_slope10 = slope_last(train, "train_loss", 10)
    val_slope5 = slope_last(val, "val_macro_f1", 5)
    val_slope10 = slope_last(val, "val_macro_f1", 10)
    val_acc_slope10 = slope_last(val, "val_accuracy", 10)
    volatility10 = std_last(val, "val_macro_f1", 10)
    best_at_final = bool(best_epoch is not None and max_epoch is not None and best_epoch == max_epoch)
    still_learning = bool(best_at_final or (val_slope10 is not None and val_slope10 > 0.0))
    train_still_decreasing = bool(train_slope10 is not None and train_slope10 < -0.001)
    likely_undertrained = bool(still_learning and train_last is not None and train_last > 0.9 and train_still_decreasing)
    likely_plateaued = bool(
        not still_learning
        and (val_slope10 is not None and abs(val_slope10) <= 0.0005)
        and (train_slope10 is not None and abs(train_slope10) <= 0.002)
    )
    notes = []
    if best_at_final:
        notes.append("best_at_final_epoch")
    if still_learning:
        notes.append("still_learning_signal")
    if train_still_decreasing:
        notes.append("train_loss_still_decreasing")
    if likely_undertrained:
        notes.append("likely_undertrained")
    if likely_plateaued:
        notes.append("likely_plateaued")
    return {
        "run_name": run["run_name"],
        "config_type": run["config_type"],
        "max_epoch": max_epoch,
        "best_epoch": best_epoch,
        "best_epoch_ratio": best_epoch / max_epoch if best_epoch and max_epoch else None,
        "best_at_final_epoch": best_at_final,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "min_train_loss": train_min,
        "train_loss_drop": train_first - train_last if train_first is not None and train_last is not None else None,
        "train_loss_drop_pct": (train_first - train_last) / train_first if train_first and train_last is not None else None,
        "train_loss_slope_last5": train_slope5,
        "train_loss_slope_last10": train_slope10,
        "train_loss_still_decreasing": train_still_decreasing,
        "val_macro_best": best_val_macro,
        "val_macro_last": val_last,
        "delta_final_vs_best": val_last - best_val_macro if val_last is not None and best_val_macro is not None else None,
        "val_macro_slope_last5": val_slope5,
        "val_macro_slope_last10": val_slope10,
        "val_macro_volatility_last10": volatility10,
        "val_acc_best": val_acc_best,
        "val_acc_last": val_acc_last,
        "val_acc_slope_last10": val_acc_slope10,
        "still_learning_signal": still_learning,
        "likely_undertrained": likely_undertrained,
        "likely_plateaued": likely_plateaued,
        "notes": "; ".join(notes),
    }


def epoch50_analysis(run: Dict[str, Any], dyn: Dict[str, Any]) -> Dict[str, Any]:
    train = run["train"]
    val = run["val"]
    best_epoch = dyn["best_epoch"]
    max_epoch = dyn["max_epoch"]
    best_at_final = dyn["best_at_final_epoch"]
    slope10 = dyn["val_macro_slope_last10"]
    volatility = dyn["val_macro_volatility_last10"]
    recommendation = "REVIEW_MANUALLY"
    if best_at_final and slope10 is not None and slope10 > 0.001:
        recommendation = "EXTEND_TO_100"
    elif best_epoch is not None and max_epoch and best_epoch >= 0.9 * max_epoch and slope10 is not None and slope10 > 0:
        recommendation = "EXTEND_TO_75"
    elif dyn["likely_plateaued"]:
        recommendation = "KEEP_50_ENOUGH"
    elif volatility is not None and volatility > 0.01:
        recommendation = "REVIEW_MANUALLY"
    return {
        "run_name": run["run_name"],
        "best_epoch": best_epoch,
        "max_epoch": max_epoch,
        "best_at_final_epoch": best_at_final,
        "val_macro_epoch_40": value_at_epoch(val, "val_macro_f1", 40),
        "val_macro_epoch_45": value_at_epoch(val, "val_macro_f1", 45),
        "val_macro_epoch_50": value_at_epoch(val, "val_macro_f1", 50),
        "delta_40_to_50": (
            value_at_epoch(val, "val_macro_f1", 50) - value_at_epoch(val, "val_macro_f1", 40)
            if value_at_epoch(val, "val_macro_f1", 50) is not None and value_at_epoch(val, "val_macro_f1", 40) is not None
            else None
        ),
        "delta_45_to_50": (
            value_at_epoch(val, "val_macro_f1", 50) - value_at_epoch(val, "val_macro_f1", 45)
            if value_at_epoch(val, "val_macro_f1", 50) is not None and value_at_epoch(val, "val_macro_f1", 45) is not None
            else None
        ),
        "train_loss_epoch_40": value_at_epoch(train, "train_loss", 40),
        "train_loss_epoch_50": value_at_epoch(train, "train_loss", 50),
        "loss_delta_40_to_50": (
            value_at_epoch(train, "train_loss", 50) - value_at_epoch(train, "train_loss", 40)
            if value_at_epoch(train, "train_loss", 50) is not None and value_at_epoch(train, "train_loss", 40) is not None
            else None
        ),
        "val_macro_slope_last10": slope10,
        "recommendation": recommendation,
    }


def pooling_dynamics(run: Dict[str, Any]) -> Dict[str, Any]:
    pool = run["pool"]
    k = run["k_regions"] or 144
    out = {
        "run_name": run["run_name"],
        "config_type": run["config_type"],
        "k_regions": run["k_regions"],
        "grid_size": run["grid_size"],
    }
    if pool is None:
        out.update({"pool_status": "MISSING_POOLING_STATS"})
        return out
    sub = pool[pool["split"] == "test"] if "split" in pool else pd.DataFrame()
    dynamic = pool[pool["split"].isin(["train", "val"])] if "split" in pool else pool
    last_split = sub if not sub.empty else (pool[pool["split"] == "val"] if "split" in pool else pool)
    if last_split.empty:
        last_split = pool
    metrics = [
        "effective_regions",
        "empty_region_ratio",
        "assignment_entropy",
        "region_area_min",
        "region_area_mean",
        "region_area_max",
        "region_area_std",
        "balance_loss",
        "compactness_loss",
        "entropy_loss",
        "area_loss",
        "assignment_temperature",
    ]
    for m in metrics:
        if m in pool:
            vals = pd.to_numeric(last_split[m], errors="coerce").dropna()
            all_vals = pd.to_numeric(pool[m], errors="coerce").dropna()
            out[f"{m}_mean"] = float(vals.mean()) if not vals.empty else None
            out[f"{m}_last"] = float(vals.iloc[-1]) if not vals.empty else None
            out[f"{m}_min"] = float(all_vals.min()) if not all_vals.empty else None
            out[f"{m}_max"] = float(all_vals.max()) if not all_vals.empty else None
            out[f"{m}_slope_last10"] = slope_last(dynamic, m, 10)
    eff = out.get("effective_regions_mean")
    empty = out.get("empty_region_ratio_mean")
    entropy = out.get("assignment_entropy_mean")
    entropy_slope = out.get("assignment_entropy_slope_last10")
    pool_healthy = bool(eff is not None and eff >= 0.5 * k and empty is not None and empty <= 0.4)
    too_soft = bool(entropy is not None and entropy > 0.95 and (entropy_slope is None or abs(entropy_slope) < 0.0005))
    too_hard = bool(entropy is not None and entropy < 0.45 and (eff is None or eff < 0.75 * k or (empty is not None and empty > 0.05)))
    collapse = bool(eff is not None and eff < 0.5 * k or (empty is not None and empty > 0.4))
    stable = bool(pool_healthy and empty is not None and empty <= 0.05)
    flags = []
    if pool_healthy:
        flags.append("POOL_HEALTHY")
    if too_soft:
        flags.append("POOL_TOO_SOFT")
    if too_hard:
        flags.append("POOL_TOO_HARD")
    if collapse:
        flags.append("POOL_COLLAPSE")
    if stable:
        flags.append("POOL_STABLE")
    out.update(
        {
            "pool_healthy": pool_healthy,
            "pool_too_soft": too_soft,
            "pool_too_hard": too_hard,
            "pool_collapse": collapse,
            "pool_stable": stable,
            "pool_status": "; ".join(flags) if flags else "POOL_REVIEW",
        }
    )
    return out


def pred_dynamics(run: Dict[str, Any]) -> Dict[str, Any]:
    pred = run["pred"]
    out = {"run_name": run["run_name"], "config_type": run["config_type"]}
    if pred is None:
        out["pred_status"] = "MISSING_PRED_COUNT"
        return out
    sub = pred[pred["split"] == "test"] if "split" in pred else pd.DataFrame()
    if sub.empty:
        sub = pred[pred["split"] == "val"] if "split" in pred else pred
    row = sub.iloc[-1] if not sub.empty else pred.iloc[-1]
    counts = []
    for i, name in enumerate(EMOTION_NAMES):
        value = as_int(row.get(f"pred_count_{i}_{name}"), 0) or 0
        counts.append(value)
        out[f"pred_count_{name}"] = value
    total = sum(counts)
    max_idx = int(np.argmax(counts)) if counts else 0
    min_idx = int(np.argmin(counts)) if counts else 0
    max_ratio = counts[max_idx] / total if total else None
    out.update(
        {
            "pred_total": total,
            "max_pred_class": EMOTION_NAMES[max_idx],
            "max_pred_ratio": max_ratio,
            "min_pred_class": EMOTION_NAMES[min_idx],
            "min_pred_count": counts[min_idx] if counts else None,
            "classes_predicted_count": int(sum(c > 0 for c in counts)),
            "disgust_pred_count": counts[1] if len(counts) > 1 else None,
            "happy_pred_ratio": counts[3] / total if total else None,
            "neutral_pred_ratio": counts[6] / total if total else None,
            "pred_entropy": pred_entropy(counts),
        }
    )
    if max_ratio is None:
        status = "PRED_REVIEW"
    elif max_ratio < 0.5:
        status = "NO_COLLAPSE"
    elif max_ratio < 0.75:
        status = "MILD_BIAS"
    elif max_ratio < 0.9:
        status = "COLLAPSE_RISK"
    else:
        status = "HARD_COLLAPSE"
    out["pred_status"] = status
    return out


def per_class_summary(run: Dict[str, Any], baseline_f1: Dict[str, Optional[float]]) -> Dict[str, Any]:
    test = run["test"]
    out = {"run_name": run["run_name"], "config_type": run["config_type"]}
    row = test.iloc[-1] if test is not None else {}
    hard_values = []
    for i, name in enumerate(EMOTION_NAMES):
        val = as_float(row.get(f"test_f1_{i}_{name}"))
        out[f"f1_{name}"] = val
        base = baseline_f1.get(name)
        out[f"delta_f1_{name}"] = val - base if val is not None and base is not None else None
        if name in HARD_CLASSES and val is not None:
            hard_values.append(val)
    out["hard_class_macro_f1"] = float(np.mean(hard_values)) if hard_values else None
    out["easy_class_warning"] = bool(
        out.get("f1_Happy") is not None
        and out.get("f1_Happy") > 0.75
        and out.get("hard_class_macro_f1") is not None
        and out["hard_class_macro_f1"] < 0.40
    )
    return out


def deep_summary(run: Dict[str, Any], dyn: Dict[str, Any], pool: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    val_row, best_epoch, best_val = best_row(run["val"], "val_macro_f1")
    test_row = run["test"].iloc[-1] if run["test"] is not None else {}
    return {
        "run_name": run["run_name"],
        "config_type": run["config_type"],
        "grid_size": run["grid_size"],
        "k_regions": run["k_regions"],
        "assignment_temperature_config": run["assignment_temperature_config"],
        "assignment_temperature_start": run["assignment_temperature_start"],
        "assignment_temperature_end": run["assignment_temperature_end"],
        "pool_entropy_weight": run["pool_entropy_weight"],
        "pool_balance_weight": run["pool_balance_weight"],
        "pool_compactness_weight": run["pool_compactness_weight"],
        "pool_area_weight": run["pool_area_weight"],
        "seed": run["seed"],
        "lr": run["lr"],
        "max_epochs": dyn["max_epoch"],
        "best_epoch": best_epoch,
        "best_epoch_ratio": dyn["best_epoch_ratio"],
        "best_at_final_epoch": dyn["best_at_final_epoch"],
        "best_val_macro_f1": best_val,
        "best_val_acc": as_float(val_row.get("val_accuracy")) if val_row is not None else None,
        "final_val_macro_f1": dyn["val_macro_last"],
        "delta_final_vs_best": dyn["delta_final_vs_best"],
        "test_macro_f1": as_float(test_row.get("test_macro_f1")),
        "test_acc": as_float(test_row.get("test_accuracy")),
        "test_weighted_f1": as_float(test_row.get("test_weighted_f1")),
        "last_epoch_train_loss": dyn["train_loss_last"],
        "checkpoint_exists": run["checkpoint_best_exists"],
        "checker_decision": run["check"].get("final_decision"),
        "still_learning_signal": dyn["still_learning_signal"],
        "likely_undertrained": dyn["likely_undertrained"],
        "pooling_status": pool.get("pool_status"),
        "pred_status": pred.get("pred_status"),
        "max_pred_class": pred.get("max_pred_class"),
        "max_pred_ratio": pred.get("max_pred_ratio"),
    }


def make_recommendations(
    summary_df: pd.DataFrame,
    dyn_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    baseline_test_macro: Optional[float],
) -> pd.DataFrame:
    rows = []
    metric_rank_df = summary_df.copy()
    for metric in ["best_val_macro_f1", "test_macro_f1", "test_acc", "test_weighted_f1"]:
        metric_rank_df[f"rank_{metric}"] = metric_rank_df[metric].rank(ascending=False, method="min")
    metric_rank_df["score_rank"] = metric_rank_df[[f"rank_{m}" for m in ["best_val_macro_f1", "test_macro_f1", "test_acc", "test_weighted_f1"]]].mean(axis=1)
    for _, row in metric_rank_df.iterrows():
        run_name = row["run_name"]
        dyn = dyn_df[dyn_df["run_name"] == run_name].iloc[0].to_dict()
        epoch = epoch_df[epoch_df["run_name"] == run_name].iloc[0].to_dict()
        pool = pool_df[pool_df["run_name"] == run_name].iloc[0].to_dict()
        pred = pred_df[pred_df["run_name"] == run_name].iloc[0].to_dict()
        pc = per_class_df[per_class_df["run_name"] == run_name].iloc[0].to_dict()
        test_macro = row.get("test_macro_f1")
        if pred.get("pred_status") in {"HARD_COLLAPSE", "COLLAPSE_RISK"} or bool(pool.get("pool_collapse")):
            tag = "INVALID_COLLAPSE"
            rec = "Do not use for D13B; inspect collapse first."
        elif bool(dyn.get("still_learning_signal")) and epoch.get("recommendation") in {"EXTEND_TO_75", "EXTEND_TO_100"}:
            tag = "PROMISING_NEEDS_EXTENDED_TRAIN"
            rec = f"{epoch.get('recommendation').replace('_', ' ').title()} before judging final capacity."
        elif baseline_test_macro is not None and test_macro is not None and test_macro >= baseline_test_macro + 0.01 and "POOL_HEALTHY" in str(pool.get("pool_status")) and pred.get("pred_status") == "NO_COLLAPSE":
            tag = "STRONG_FINAL_CANDIDATE"
            rec = "Use as D13A-final candidate after visual pooling audit."
        elif "k64" in run_name.lower() and baseline_test_macro is not None and test_macro is not None and test_macro >= baseline_test_macro - 0.01:
            tag = "EFFICIENT_CANDIDATE"
            rec = "Keep as efficient K64 control; compare compute and class trade-off."
        elif "temp" in run_name.lower() or "anneal" in run_name.lower():
            tag = "HARDENING_CANDIDATE"
            rec = "Review entropy/per-class trade-off before using hardening."
        elif "seed" in run_name.lower():
            tag = "STABILITY_CONTROL"
            rec = "Use for multi-seed stability estimate, not as final by itself."
        elif test_macro is not None and baseline_test_macro is not None and test_macro < baseline_test_macro - 0.03 and not dyn.get("still_learning_signal"):
            tag = "WEAK_RUN"
            rec = "Do not prioritize unless it answers a control question."
        else:
            tag = "REVIEW_MANUALLY"
            rec = "Review learning curves, per-class F1, and pooling figures."
        hard_delta_cols = [f"delta_f1_{c}" for c in HARD_CLASSES if f"delta_f1_{c}" in pc and pc.get(f"delta_f1_{c}") is not None]
        hard_delta = float(np.mean([pc[c] for c in hard_delta_cols])) if hard_delta_cols else None
        rows.append(
            {
                "run_name": run_name,
                "score_rank": row["score_rank"],
                "learning_status": "STILL_LEARNING" if dyn.get("still_learning_signal") else ("PLATEAUED" if dyn.get("likely_plateaued") else "REVIEW"),
                "pooling_status": pool.get("pool_status"),
                "pred_status": pred.get("pred_status"),
                "per_class_status": "HARD_CLASSES_IMPROVED" if hard_delta is not None and hard_delta > 0 else "NO_HARD_CLASS_GAIN",
                "final_tag": tag,
                "recommendation": rec,
            }
        )
    return pd.DataFrame(rows).sort_values(["score_rank", "run_name"])


def save_figures(runs: List[Dict[str, Any]], output_dir: Path, per_class_df: pd.DataFrame, pred_df: pd.DataFrame, summary_df: pd.DataFrame) -> List[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    if plt is None:
        return made

    def line_plot(filename: str, metric: str, source: str, title: str) -> None:
        plt.figure(figsize=(12, 6))
        for run in runs:
            df = run[source]
            if df is None or metric not in df or "epoch" not in df:
                continue
            if source == "pool" and "split" in df:
                df = df[df["split"] == "val"]
            plt.plot(df["epoch"], df[metric], label=run["run_name"].replace("d13a_edgeaware_lite_localpool_", ""))
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        path = fig_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))

    line_plot("val_macro_f1_curves_all.png", "val_macro_f1", "val", "Validation macro-F1 curves")
    line_plot("train_loss_curves_all.png", "train_loss", "train", "Training loss curves")
    line_plot("effective_regions_curves_all.png", "effective_regions", "pool", "Effective regions curves (val)")
    line_plot("assignment_entropy_curves_all.png", "assignment_entropy", "pool", "Assignment entropy curves (val)")

    if not summary_df.empty:
        plt.figure(figsize=(12, 5))
        plot_df = summary_df.sort_values("test_macro_f1", ascending=False)
        plt.bar(plot_df["run_name"], plot_df["test_macro_f1"])
        plt.xticks(rotation=75, ha="right", fontsize=7)
        plt.ylabel("test_macro_f1")
        plt.title("D13A ablation test macro-F1")
        plt.tight_layout()
        path = fig_dir / "test_macro_f1_bar.png"
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))

        plt.figure(figsize=(12, 5))
        plot_df = summary_df.sort_values("best_epoch", ascending=False)
        plt.bar(plot_df["run_name"], plot_df["best_epoch"])
        plt.xticks(rotation=75, ha="right", fontsize=7)
        plt.ylabel("best epoch")
        plt.title("Best epoch by run")
        plt.tight_layout()
        path = fig_dir / "best_epoch_bar.png"
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))

    if not per_class_df.empty:
        cols = [f"f1_{c}" for c in EMOTION_NAMES]
        heat = per_class_df.set_index("run_name")[cols].astype(float)
        plt.figure(figsize=(9, max(4, 0.35 * len(heat))))
        plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(label="F1")
        plt.xticks(range(len(cols)), EMOTION_NAMES, rotation=45, ha="right")
        plt.yticks(range(len(heat)), heat.index, fontsize=7)
        plt.title("Per-class F1 heatmap")
        plt.tight_layout()
        path = fig_dir / "per_class_f1_heatmap.png"
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))

    if not pred_df.empty:
        cols = [f"pred_count_{c}" for c in EMOTION_NAMES if f"pred_count_{c}" in pred_df]
        heat = pred_df.set_index("run_name")[cols].astype(float)
        plt.figure(figsize=(9, max(4, 0.35 * len(heat))))
        plt.imshow(heat.to_numpy(), aspect="auto", cmap="magma")
        plt.colorbar(label="count")
        plt.xticks(range(len(cols)), [c.replace("pred_count_", "") for c in cols], rotation=45, ha="right")
        plt.yticks(range(len(heat)), heat.index, fontsize=7)
        plt.title("Pred count heatmap")
        plt.tight_layout()
        path = fig_dir / "pred_count_heatmap.png"
        plt.savefig(path, dpi=150)
        plt.close()
        made.append(str(path))
    return made


def md_table(df: pd.DataFrame, cols: List[str], n: Optional[int] = None) -> str:
    if df.empty:
        return "No data."
    available = [c for c in cols if c in df.columns]
    if not available:
        return "No requested columns available."
    use = df[available].copy()
    if n is not None:
        use = use.head(n)
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def group_section(summary_df: pd.DataFrame, pool_df: pd.DataFrame, per_class_df: pd.DataFrame) -> str:
    lines = []
    groups = {
        "K sweep": ["baseline_k144", "k_sweep_k64", "k_sweep_k256"],
        "Temperature": ["baseline_k144", "temperature_temp07", "temperature_temp05", "temperature_anneal"],
        "Pool losses": ["baseline_k144", "pool_loss_no_aux", "pool_loss_compact_balance_x2"],
        "Seed stability": ["baseline_k144", "seed2", "seed3"],
        "LR sensitivity": ["baseline_k144", "lr1e4"],
    }
    for title, types in groups.items():
        sub = summary_df[summary_df["config_type"].isin(types)].copy()
        lines.extend([f"### {title}", ""])
        if sub.empty:
            lines.extend(["No matching runs found.", ""])
            continue
        cols = ["run_name", "config_type", "best_val_macro_f1", "test_macro_f1", "test_acc", "best_epoch", "still_learning_signal", "pooling_status", "pred_status"]
        lines.extend([md_table(sub.sort_values("test_macro_f1", ascending=False), cols), ""])
        if title == "K sweep":
            lines.append("Interpretation: compare K64/K144/K256 as an information bottleneck test. K64 is efficient only if macro-F1 stays near baseline; K256 is useful only if extra regions improve class balance or macro-F1.")
        elif title == "Temperature":
            lines.append("Interpretation: hardening is useful only if entropy drops without empty-region growth, class collapse, or hard-class F1 loss.")
        elif title == "Pool losses":
            lines.append("Interpretation: no_aux tests whether auxiliary losses are necessary for stable coarsening; compact_balance_x2 tests stronger geometric regularization.")
        elif title == "Seed stability":
            vals = sub["test_macro_f1"].dropna()
            lines.append(f"Seed stability estimate: mean={vals.mean():.4f}, std={vals.std(ddof=0):.4f} over available baseline/seed runs." if len(vals) else "No seed metrics available.")
        elif title == "LR sensitivity":
            lines.append("Interpretation: lr1e4 should not be rejected only for lower score if it is still improving at epoch 50; it may simply learn slower.")
        lines.append("")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    dyn_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    figures: List[str],
) -> str:
    report_path = output_dir / "d13a_ablation_deep_analysis_report.md"
    top_test = summary_df.sort_values("test_macro_f1", ascending=False)
    top_val = summary_df.sort_values("best_val_macro_f1", ascending=False)
    best_final_count = int(summary_df["best_at_final_epoch"].fillna(False).sum()) if "best_at_final_epoch" in summary_df else 0
    extend = epoch_df[epoch_df["recommendation"].isin(["EXTEND_TO_75", "EXTEND_TO_100"])] if not epoch_df.empty else pd.DataFrame()
    has_extend_uncertainty = best_final_count > 0 or not extend.empty
    final_decision = "D13A_NEEDS_EXTENDED_TRAIN_BEFORE_D13B" if has_extend_uncertainty else "D13A_NEEDS_MORE_ABLATION"
    strong = rec_df[rec_df["final_tag"] == "STRONG_FINAL_CANDIDATE"] if not rec_df.empty else pd.DataFrame()
    if not strong.empty and not has_extend_uncertainty:
        final_decision = "D13A_FINAL_SELECTED_READY_FOR_D13B"
    if summary_df.empty:
        final_decision = "D13A_ABLATION_INCONCLUSIVE"

    lines = [
        "# D13A Ablation Deep Analysis Report",
        "",
        "## 1. Context",
        "D13A is a pure GNN hierarchical reduction baseline. It uses no CNN teacher, no D12 full architecture, no SupCon, and no motif slots.",
        "Region nodes in this report are a soft learnable bottleneck for graph coarsening. They are not semantic facial regions and this report makes no motif claim.",
        "The goal here is to analyze learning dynamics, reduction behavior, prediction distribution, and per-class trade-offs, not only final score.",
        "",
        "## 2. Overall Ranking",
        "### Top by test macro-F1",
        md_table(top_test, ["run_name", "config_type", "test_macro_f1", "test_acc", "test_weighted_f1", "best_epoch", "best_at_final_epoch"], n=10),
        "",
        "### Top by best val macro-F1",
        md_table(top_val, ["run_name", "config_type", "best_val_macro_f1", "best_val_acc", "test_macro_f1", "best_epoch", "best_at_final_epoch"], n=10),
        "",
        f"Warning: {best_final_count} run(s) reached best validation macro-F1 at the final logged epoch. Those runs should not be treated as capacity-limited.",
        "",
        "## 3. Learning Dynamics",
        md_table(dyn_df.sort_values("val_macro_best", ascending=False), ["run_name", "max_epoch", "best_epoch", "best_at_final_epoch", "train_loss_first", "train_loss_last", "val_macro_best", "val_macro_last", "val_macro_slope_last10", "still_learning_signal", "likely_undertrained", "likely_plateaued"], n=20),
        "",
        "Runs with positive late validation slope or best-at-final behavior should be extended before judging the final D13A candidate.",
        "",
        "## 4. Epoch-50 Best Issue",
        md_table(epoch_df.sort_values(["recommendation", "run_name"]), ["run_name", "best_epoch", "max_epoch", "best_at_final_epoch", "val_macro_epoch_40", "val_macro_epoch_45", "val_macro_epoch_50", "delta_40_to_50", "val_macro_slope_last10", "recommendation"], n=30),
        "",
        "A best epoch at 50 means the training horizon may be too short or the scheduler may be too conservative. Do not call these runs worse only from the 50-epoch endpoint.",
        "",
        "## 5. Reduction / Pooling Health",
        md_table(pool_df.sort_values("effective_regions_mean", ascending=False), ["run_name", "k_regions", "effective_regions_mean", "empty_region_ratio_mean", "assignment_entropy_mean", "assignment_temperature_mean", "pool_status"], n=30),
        "",
        "Effective regions and empty-region ratio indicate whether the soft bottleneck stayed usable. They do not imply semantic region discovery.",
        "",
        "## 6. Prediction Distribution and Collapse",
        md_table(pred_df.sort_values("max_pred_ratio"), ["run_name", "max_pred_class", "max_pred_ratio", "classes_predicted_count", "disgust_pred_count", "happy_pred_ratio", "neutral_pred_ratio", "pred_entropy", "pred_status"], n=30),
        "",
        "Macro-F1 gains are less trustworthy when prediction mass collapses into one class. Runs that improve Disgust/Fear/Sad/Angry are especially valuable.",
        "",
        "## 7. Per-Class Behavior",
        "### Per-class trade-off",
        md_table(per_class_df.sort_values("hard_class_macro_f1", ascending=False), ["run_name", "hard_class_macro_f1", "f1_Angry", "f1_Disgust", "f1_Fear", "f1_Happy", "f1_Sad", "f1_Surprise", "f1_Neutral"], n=30),
        "",
        "Hard-class improvements matter even when accuracy does not move much; weighted-F1 can hide class imbalance.",
        "",
        "## 8. Ablation Group Conclusions",
        group_section(summary_df, pool_df, per_class_df),
        "",
        "## 9. Candidate Selection",
        md_table(rec_df, ["run_name", "score_rank", "learning_status", "pooling_status", "pred_status", "per_class_status", "final_tag", "recommendation"], n=30),
        "",
        "The current recommendation table is a prioritization aid, not a D13B trigger. D13B should wait until extended training resolves final-epoch uncertainty.",
        "",
        "## 10. Next Actions",
    ]
    if not extend.empty:
        lines.append("Extend these runs first:")
        for _, row in extend.iterrows():
            lines.append(f"- {row['run_name']}: {row['recommendation']}")
    else:
        lines.append("- No clear extend candidate detected from epoch-50 rules; review curves manually.")
    extend100 = epoch_df[epoch_df["recommendation"] == "EXTEND_TO_100"] if not epoch_df.empty else pd.DataFrame()
    if extend100.empty:
        lines.append("- Strict EXTEND_TO_100 candidates: none under the best-at-final-epoch rule.")
        budget100 = epoch_df.sort_values(["val_macro_slope_last10", "delta_40_to_50"], ascending=False).head(3) if not epoch_df.empty else pd.DataFrame()
        if not budget100.empty:
            names = ", ".join(str(x) for x in budget100["run_name"].tolist())
            lines.append(f"- If using a 100-epoch budget anyway, prioritize by late learning slope: {names}.")
    else:
        names = ", ".join(str(x) for x in extend100["run_name"].tolist())
        lines.append(f"- Strict EXTEND_TO_100 candidates: {names}.")
    lines.extend(
        [
            "- Keep K144 as the reference unless K64 is near baseline with much lower compute or K256 clearly improves macro-F1.",
            "- Review temperature runs for entropy reduction without prediction/pooling collapse.",
            "- Use seed2/seed3 to decide whether D13A needs multi-seed reporting.",
            "- Run visual pooling audit for any final candidate before opening D13B.",
            "- Do not open D13B until extended-run uncertainty is resolved.",
            "",
            "## 11. Final Decision",
            final_decision,
            "",
            "## Figures",
        ]
    )
    if figures:
        for fig in figures:
            lines.append(f"- `{Path(fig).relative_to(output_dir)}`")
    else:
        lines.append("- No figures generated.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final_decision


def analyze(root_dir: Path, output_dir: Path, baseline_dir: Optional[Path]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = find_run_dirs(root_dir, baseline_dir)
    runs = [load_run(p) for p in run_dirs]
    baseline_run = next((r for r in runs if r["run_name"] == BASELINE_RUN_NAME), None)
    if baseline_run is None:
        baseline_run = next((r for r in runs if r["config_type"] == "baseline_k144"), None)
    baseline_test = baseline_run["test"].iloc[-1] if baseline_run is not None and baseline_run["test"] is not None else {}
    baseline_test_macro = as_float(baseline_test.get("test_macro_f1"))
    baseline_f1 = {}
    for i, name in enumerate(EMOTION_NAMES):
        baseline_f1[name] = as_float(baseline_test.get(f"test_f1_{i}_{name}"))

    dyn_rows, epoch_rows, pool_rows, pred_rows, per_class_rows, summary_rows = [], [], [], [], [], []
    for run in runs:
        dyn = learning_dynamics(run)
        pool = pooling_dynamics(run)
        pred = pred_dynamics(run)
        epoch = epoch50_analysis(run, dyn)
        pc = per_class_summary(run, baseline_f1)
        summary = deep_summary(run, dyn, pool, pred)
        dyn_rows.append(dyn)
        epoch_rows.append(epoch)
        pool_rows.append(pool)
        pred_rows.append(pred)
        per_class_rows.append(pc)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    dyn_df = pd.DataFrame(dyn_rows)
    epoch_df = pd.DataFrame(epoch_rows)
    pool_df = pd.DataFrame(pool_rows)
    pred_df = pd.DataFrame(pred_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    if not summary_df.empty:
        for metric in ["best_val_macro_f1", "test_macro_f1", "test_acc", "test_weighted_f1"]:
            summary_df[f"rank_{metric}"] = summary_df[metric].rank(ascending=False, method="min")
        summary_df["average_metric_rank"] = summary_df[[f"rank_{m}" for m in ["best_val_macro_f1", "test_macro_f1", "test_acc", "test_weighted_f1"]]].mean(axis=1)
        summary_df = summary_df.sort_values("average_metric_rank")

    rec_df = make_recommendations(summary_df, dyn_df, epoch_df, pool_df, pred_df, per_class_df, baseline_test_macro)

    summary_df.to_csv(output_dir / "d13a_ablation_deep_summary.csv", index=False)
    dyn_df.to_csv(output_dir / "d13a_ablation_learning_dynamics.csv", index=False)
    epoch_df.to_csv(output_dir / "d13a_ablation_epoch50_analysis.csv", index=False)
    pool_df.to_csv(output_dir / "d13a_ablation_pooling_dynamics.csv", index=False)
    pred_df.to_csv(output_dir / "d13a_ablation_pred_dynamics.csv", index=False)
    per_class_df.to_csv(output_dir / "d13a_ablation_per_class_summary.csv", index=False)
    rec_df.to_csv(output_dir / "d13a_ablation_recommendation_table.csv", index=False)
    figures = save_figures(runs, output_dir, per_class_df, pred_df, summary_df)
    final_decision = write_report(output_dir, summary_df, dyn_df, epoch_df, pool_df, pred_df, per_class_df, rec_df, figures)
    return {
        "num_runs": len(runs),
        "runs": [r["run_name"] for r in runs],
        "output_dir": str(output_dir),
        "baseline_test_macro_f1": baseline_test_macro,
        "final_decision": final_decision,
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="outputs/d13_hierarchical_reduction/ablations")
    parser.add_argument("--baseline_dir", default="outputs/d13_hierarchical_reduction/d13a_edgeaware_lite_localpool_k144_outputs")
    parser.add_argument("--output_dir", default="outputs/d13_hierarchical_reduction/ablation_deep_analysis")
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    result = analyze(Path(args.root_dir), Path(args.output_dir), baseline_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
