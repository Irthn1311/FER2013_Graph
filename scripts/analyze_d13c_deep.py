"""Deep artifact analysis for D13C diagnostic runs.

This is an offline collector only: it reads existing run artifacts and writes
summary tables, figures, and a diagnostic recommendation report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is expected in this repo env.
    yaml = None


D13B_M16_REFERENCE = {"test_macro_f1": 0.6187, "test_acc": 0.6328}
D13B_M8_REFERENCE = {"test_macro_f1": 0.6171, "test_acc": 0.6344}
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
HARD_CLASSES = ["Angry", "Disgust", "Fear", "Sad"]
SKIP_DIRS = {"summary", "deep_analysis", "smoke", "zip", "temp", "tmp"}


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def read_csv(path: Path, warnings: List[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"missing {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"failed to read {path.name}: {exc}")
        return pd.DataFrame()


def read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing {path.name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to read {path.name}: {exc}")
        return {}


def read_yaml(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing {path.name}")
        return {}
    if yaml is None:
        warnings.append("PyYAML unavailable; resolved_config.yaml not parsed")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        warnings.append(f"failed to read {path.name}: {exc}")
        return {}


def last_value(df: pd.DataFrame, col: str, split: Optional[str] = None) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    work = df
    if split is not None and "split" in work.columns:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def first_value(df: pd.DataFrame, col: str, split: Optional[str] = None) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    work = df
    if split is not None and "split" in work.columns:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty else np.nan


def mean_value(df: pd.DataFrame, col: str, split: Optional[str] = None) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    work = df
    if split is not None and "split" in work.columns:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def min_value(df: pd.DataFrame, col: str, split: Optional[str] = None) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    work = df
    if split is not None and "split" in work.columns:
        work = work[work["split"].astype(str) == split]
    vals = pd.to_numeric(work[col], errors="coerce").dropna()
    return float(vals.min()) if not vals.empty else np.nan


def slope_last(values: Iterable[Any], n: int) -> float:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.nan
    vals = vals[-n:] if len(vals) >= n else vals
    x = np.arange(len(vals), dtype=float)
    return float(np.polyfit(x, vals, 1)[0])


def volatility_last(values: Iterable[Any], n: int) -> float:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.nan
    vals = vals[-n:] if len(vals) >= n else vals
    return float(np.std(vals))


def infer_lambda(run_name: str, cfg: Dict[str, Any]) -> float:
    loss = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    if "lambda_supcon" in loss:
        return as_float(loss.get("lambda_supcon"), 0.0)
    if "ce_continue" in run_name:
        return 0.0
    match = re.search(r"_l(\d{3})", run_name)
    if match:
        return int(match.group(1)) / 1000.0
    return np.nan


def infer_variant(run_name: str) -> str:
    if "ce_continue" in run_name:
        return "ce_continue"
    if "freeze_backbone" in run_name:
        return "freeze_backbone"
    if "proj128" in run_name:
        return "proj128"
    if "m8" in run_name:
        return "m8_control"
    if "supcon" in run_name:
        return "supcon_lambda"
    return "unknown"


def infer_base_model(run_name: str, cfg: Dict[str, Any]) -> str:
    model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    base = str(model.get("base_model", ""))
    if "m8" in run_name or "m8" in base:
        return "m8"
    return "m16"


def run_dirs(root: Path) -> List[Path]:
    return [
        p
        for p in sorted(root.iterdir())
        if p.is_dir() and p.name.lower() not in SKIP_DIRS and not p.name.lower().startswith(("zip", "temp"))
    ]


def best_val(val: pd.DataFrame) -> Tuple[float, float, int]:
    if val.empty or "val_macro_f1" not in val.columns:
        return np.nan, np.nan, -1
    vals = pd.to_numeric(val["val_macro_f1"], errors="coerce")
    if vals.dropna().empty:
        return np.nan, np.nan, -1
    idx = vals.idxmax()
    best_acc = as_float(val.loc[idx].get("val_accuracy", np.nan))
    epoch = int(as_float(val.loc[idx].get("epoch", idx), idx))
    return float(vals.loc[idx]), best_acc, epoch


def pred_status(max_ratio: float, classes_predicted: float) -> str:
    if np.isnan(max_ratio):
        return "MISSING"
    if max_ratio > 0.9:
        return "HARD_COLLAPSE"
    if max_ratio > 0.75:
        return "COLLAPSE_RISK"
    if max_ratio >= 0.5:
        return "MILD_BIAS"
    if classes_predicted >= 7:
        return "NO_COLLAPSE"
    return "MILD_BIAS"


def slot_status(row: Dict[str, Any]) -> str:
    eff = as_float(row.get("effective_slots_test", row.get("effective_slots_last")))
    overlap = as_float(row.get("slot_overlap_test", row.get("slot_overlap_last")))
    dominance = as_float(row.get("slot_dominance_test", row.get("slot_dominance_last")))
    slots = as_float(row.get("num_slots"), 16.0)
    if np.isnan(eff) or np.isnan(overlap) or np.isnan(dominance):
        return "MISSING"
    if eff < 0.5 * slots or overlap > 0.85 or dominance > 0.35:
        return "SLOT_COLLAPSE_RISK"
    if eff < 0.9 * slots or overlap > 0.70 or dominance > 0.20:
        return "SLOT_DEGRADED"
    return "SLOT_HEALTHY"


def class_f1_from_test(test: pd.DataFrame) -> Dict[str, float]:
    if test.empty:
        return {name: np.nan for name in CLASS_NAMES}
    row = test.iloc[-1].to_dict()
    out: Dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        out[name] = as_float(row.get(f"test_f1_{idx}_{name}", row.get(f"f1_{idx}_{name}", np.nan)))
    return out


def load_reference_per_class(root: Path, run_name: str) -> Dict[str, float]:
    path = root / "outputs" / "d13b_diagnostic" / run_name / "test_metrics.csv"
    if not path.exists():
        return {name: np.nan for name in CLASS_NAMES}
    try:
        return class_f1_from_test(pd.read_csv(path))
    except Exception:
        return {name: np.nan for name in CLASS_NAMES}


def md_table(df: pd.DataFrame, cols: Optional[List[str]] = None, n: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df[cols].copy() if cols else df.copy()
    use = use.head(n)
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def artifact_warnings(run_dir: Path, warnings: List[str]) -> str:
    expected = [
        "train_log.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "pred_count.csv",
        "slot_stats.csv",
        "pooling_stats.csv",
        "supcon_stats.csv",
        "confusion_matrix.csv",
        "d13c_report.md",
        "d13c_diagnostic_check_summary.json",
        "d13c_diagnostic_check_report.md",
        "resolved_config.yaml",
    ]
    for name in expected:
        if not (run_dir / name).exists():
            warnings.append(f"missing {name}")
    if not (run_dir / "checkpoints" / "best.pt").exists():
        warnings.append("missing checkpoints/best.pt")
    if not (run_dir / "checkpoints" / "last.pt").exists():
        warnings.append("missing checkpoints/last.pt")
    if not ((run_dir / "per_class_metrics.csv").exists() or (run_dir / "per_class_f1.csv").exists()):
        if not (run_dir / "summary" / "d13c_per_class_summary.csv").exists():
            warnings.append("missing per_class_metrics.csv/per_class_f1.csv")
    return "; ".join(sorted(set(warnings)))


def collect(root: Path, project_root: Path) -> Dict[str, pd.DataFrame]:
    m16_ref_class = load_reference_per_class(project_root, "d13b_k144_m16_deep_readout")
    m8_ref_class = load_reference_per_class(project_root, "d13b_k144_m8_deep_region")
    runs: List[Dict[str, Any]] = []
    learning_rows: List[Dict[str, Any]] = []
    supcon_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    raw: Dict[str, Dict[str, pd.DataFrame]] = {}
    warnings_by_run: Dict[str, str] = {}

    for run_dir in run_dirs(root):
        warnings: List[str] = []
        cfg = read_yaml(run_dir / "resolved_config.yaml", warnings)
        checker = read_json(run_dir / "d13c_diagnostic_check_summary.json", warnings)
        train = read_csv(run_dir / "train_log.csv", warnings)
        val = read_csv(run_dir / "val_metrics.csv", warnings)
        test = read_csv(run_dir / "test_metrics.csv", warnings)
        pred = read_csv(run_dir / "pred_count.csv", warnings)
        slots = read_csv(run_dir / "slot_stats.csv", warnings)
        supcon = read_csv(run_dir / "supcon_stats.csv", warnings)
        raw[run_dir.name] = {"train": train, "val": val, "test": test, "pred": pred, "slots": slots, "supcon": supcon}

        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        lambda_supcon = infer_lambda(run_dir.name, cfg)
        projection_dim = int(as_float(model_cfg.get("projection_dim", 64), 64))
        freeze_backbone = as_bool(model_cfg.get("freeze_backbone", "freeze_backbone" in run_dir.name))
        base_model = infer_base_model(run_dir.name, cfg)
        max_epochs = int(as_float(train_cfg.get("max_epochs", train_cfg.get("epochs", len(val))), len(val)))
        best_macro, best_acc, best_epoch = best_val(val)
        if not np.isfinite(best_macro):
            best_macro = as_float(checker.get("best_val_macro_f1"))
            best_epoch = int(as_float(checker.get("best_epoch"), -1))
        test_row = test.iloc[-1].to_dict() if not test.empty else {}
        test_macro = as_float(test_row.get("test_macro_f1", checker.get("test_macro_f1")))
        test_acc = as_float(test_row.get("test_accuracy", checker.get("test_acc")))
        test_weighted = as_float(test_row.get("test_weighted_f1", checker.get("test_weighted_f1")))
        checker_decision = str(checker.get("decision", "MISSING_CHECKER"))
        checkpoint_exists = (run_dir / "checkpoints" / "best.pt").exists()

        runs.append(
            {
                "run_name": run_dir.name,
                "variant_type": infer_variant(run_dir.name),
                "lambda_supcon": lambda_supcon,
                "projection_dim": projection_dim,
                "freeze_backbone": freeze_backbone,
                "base_model": base_model,
                "max_epochs": max_epochs,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro,
                "best_val_accuracy": best_acc,
                "test_macro_f1": test_macro,
                "test_accuracy": test_acc,
                "test_weighted_f1": test_weighted,
                "checkpoint_exists": checkpoint_exists,
                "checker_decision": checker_decision,
            }
        )

        train_loss = pd.to_numeric(train.get("train_loss", pd.Series(dtype=float)), errors="coerce")
        val_macro = pd.to_numeric(val.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce")
        train_loss_first = float(train_loss.dropna().iloc[0]) if not train_loss.dropna().empty else np.nan
        train_loss_last = float(train_loss.dropna().iloc[-1]) if not train_loss.dropna().empty else np.nan
        val_macro_last = float(val_macro.dropna().iloc[-1]) if not val_macro.dropna().empty else np.nan
        val_slope5 = slope_last(val_macro, 5)
        val_slope10 = slope_last(val_macro, 10)
        train_slope5 = slope_last(train_loss, 5)
        train_slope10 = slope_last(train_loss, 10)
        val_vol10 = volatility_last(val_macro, 10)
        final_epoch = int(as_float(val.iloc[-1].get("epoch", len(val)) if not val.empty else len(train), len(val)))
        best_at_final = bool(best_epoch >= final_epoch - 1) if best_epoch > 0 else False
        still_learning = bool((np.isfinite(val_slope5) and val_slope5 > 0.001) or best_at_final)
        plateau = bool(np.isfinite(val_slope10) and abs(val_slope10) < 0.001 and np.isfinite(val_vol10) and val_vol10 < 0.012)
        undertrained = bool(best_at_final and np.isfinite(val_slope5) and val_slope5 >= -0.0005)
        overfit = bool(np.isfinite(train_slope10) and train_slope10 < 0 and np.isfinite(val_slope10) and val_slope10 < -0.001)
        if undertrained:
            decision = "EXTEND_TO_50/75"
            learning_status = "still_learning"
        elif overfit:
            decision = "STOP_VARIANT"
            learning_status = "overfit_or_degrading"
        elif plateau:
            decision = "KEEP_EPOCHS_ENOUGH"
            learning_status = "plateaued"
        elif still_learning:
            decision = "EXTEND_TO_50/75"
            learning_status = "still_learning"
        else:
            decision = "REVIEW_MANUALLY"
            learning_status = "mixed"
        learning_rows.append(
            {
                "run_name": run_dir.name,
                "train_loss_first": train_loss_first,
                "train_loss_last": train_loss_last,
                "train_loss_drop": train_loss_first - train_loss_last if np.isfinite(train_loss_first + train_loss_last) else np.nan,
                "train_loss_slope_last5": train_slope5,
                "train_loss_slope_last10": train_slope10,
                "val_macro_best": best_macro,
                "val_macro_last": val_macro_last,
                "val_macro_slope_last5": val_slope5,
                "val_macro_slope_last10": val_slope10,
                "val_macro_volatility_last10": val_vol10,
                "best_epoch": best_epoch,
                "best_at_final_epoch": best_at_final,
                "likely_undertrained": undertrained,
                "likely_plateaued": plateau,
                "still_learning_signal": still_learning,
                "likely_overfit": overfit,
                "learning_status": learning_status,
                "decision_recommendation": decision,
            }
        )

        train_sup = supcon[supcon["split"].astype(str) == "train"] if not supcon.empty and "split" in supcon.columns else supcon
        sup_first = first_value(train_sup, "loss_supcon")
        sup_last = last_value(train_sup, "loss_supcon")
        collapse_mean = mean_value(train_sup, "embedding_collapse_score")
        active_rate = mean_value(train_sup, "has_supcon_signal")
        supcon_rows.append(
            {
                "run_name": run_dir.name,
                "lambda_supcon": lambda_supcon,
                "projection_dim": projection_dim,
                "freeze_backbone": freeze_backbone,
                "base_model": base_model,
                "supcon_loss_first": sup_first,
                "supcon_loss_last": sup_last,
                "supcon_loss_min": min_value(train_sup, "loss_supcon"),
                "supcon_loss_slope_last10": slope_last(train_sup.get("loss_supcon", []), 10),
                "positive_pair_count_mean": mean_value(train_sup, "positive_pair_count"),
                "positive_pair_count_min": min_value(train_sup, "positive_pair_count"),
                "valid_anchor_mean": mean_value(train_sup, "valid_supcon_anchor_count"),
                "valid_anchor_min": min_value(train_sup, "valid_supcon_anchor_count"),
                "z_norm_mean_last": last_value(train_sup, "z_norm_mean"),
                "z_norm_std_last": last_value(train_sup, "z_norm_std"),
                "embedding_collapse_score_mean": collapse_mean,
                "embedding_collapse_score_last": last_value(train_sup, "embedding_collapse_score"),
                "supcon_active_rate": active_rate,
                "supcon_loss_drop": sup_first - sup_last if np.isfinite(sup_first + sup_last) else np.nan,
            }
        )

        test_slots = slots[slots["split"].astype(str) == "test"] if not slots.empty and "split" in slots.columns else slots.tail(1)
        all_slots = slots if not slots.empty else pd.DataFrame()
        slot_row = {
            "run_name": run_dir.name,
            "base_model": base_model,
            "num_slots": int(as_float(model_cfg.get("num_slots", 16), 16)),
            "effective_slots_test": last_value(test_slots, "effective_slots"),
            "effective_slots_last": last_value(all_slots, "effective_slots"),
            "effective_slots_mean": mean_value(all_slots, "effective_slots"),
            "slot_overlap_test": last_value(test_slots, "slot_overlap"),
            "slot_overlap_last": last_value(all_slots, "slot_overlap"),
            "slot_overlap_mean": mean_value(all_slots, "slot_overlap"),
            "slot_entropy_test": last_value(test_slots, "slot_entropy"),
            "slot_entropy_last": last_value(all_slots, "slot_entropy"),
            "slot_entropy_mean": mean_value(all_slots, "slot_entropy"),
            "slot_dominance_test": last_value(test_slots, "slot_dominance"),
            "slot_dominance_last": last_value(all_slots, "slot_dominance"),
            "slot_dominance_mean": mean_value(all_slots, "slot_dominance"),
            "slot_area_mean_test": last_value(test_slots, "slot_area_mean"),
            "slot_center_std_test": last_value(test_slots, "slot_center_std"),
        }
        slot_row["slot_health_status"] = slot_status(slot_row)
        slot_rows.append(slot_row)

        test_pred = pred[pred["split"].astype(str) == "test"] if not pred.empty and "split" in pred.columns else pred.tail(1)
        prow = test_pred.iloc[-1].to_dict() if not test_pred.empty else {}
        counts: List[float] = []
        pred_out: Dict[str, Any] = {"run_name": run_dir.name, "base_model": base_model}
        for idx, name in enumerate(CLASS_NAMES):
            val_count = as_float(prow.get(f"pred_count_{idx}_{name}", 0.0), 0.0)
            pred_out[f"pred_count_{name}"] = val_count
            counts.append(val_count)
        total = as_float(prow.get("pred_total", sum(counts)), sum(counts))
        max_ratio = as_float(prow.get("pred_max_ratio", max(counts) / total if total > 0 else np.nan))
        probs = np.array(counts, dtype=float) / total if total > 0 else np.array([])
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum() / np.log(len(CLASS_NAMES))) if total > 0 else np.nan
        classes_pred = float(sum(c > 0 for c in counts))
        pred_out.update(
            {
                "pred_total": total,
                "pred_max_ratio": max_ratio,
                "classes_predicted_count": classes_pred,
                "pred_entropy": entropy,
                "pred_status": pred_status(max_ratio, classes_pred),
            }
        )
        pred_rows.append(pred_out)

        f1s = class_f1_from_test(test)
        ref = m8_ref_class if base_model == "m8" else m16_ref_class
        pc_row: Dict[str, Any] = {"run_name": run_dir.name, "base_model": base_model}
        for name in CLASS_NAMES:
            pc_row[name] = f1s[name]
            pc_row[f"delta_{name}_vs_d13b_ref"] = f1s[name] - ref.get(name, np.nan)
        pc_row["hard_class_macro"] = float(np.nanmean([f1s[name] for name in HARD_CLASSES]))
        pc_row["easy_class_macro"] = float(np.nanmean([f1s[name] for name in ["Happy", "Surprise", "Neutral"]]))
        per_class_rows.append(pc_row)
        warnings_by_run[run_dir.name] = artifact_warnings(run_dir, warnings)

    summary = pd.DataFrame(runs)
    if summary.empty:
        return {name: pd.DataFrame() for name in ["summary", "learning", "supcon", "slot", "pred", "per_class", "vs_ref", "recommendations", "warnings"]}

    ce_macro = as_float(summary.loc[summary["run_name"] == "d13c_m16_ce_continue", "test_macro_f1"].iloc[0]) if (summary["run_name"] == "d13c_m16_ce_continue").any() else np.nan
    ce_acc = as_float(summary.loc[summary["run_name"] == "d13c_m16_ce_continue", "test_accuracy"].iloc[0]) if (summary["run_name"] == "d13c_m16_ce_continue").any() else np.nan
    summary["delta_macro_vs_d13b_m16"] = summary["test_macro_f1"] - D13B_M16_REFERENCE["test_macro_f1"]
    summary["delta_acc_vs_d13b_m16"] = summary["test_accuracy"] - D13B_M16_REFERENCE["test_acc"]
    summary["delta_macro_vs_ce_continue"] = summary["test_macro_f1"] - ce_macro
    summary["delta_acc_vs_ce_continue"] = summary["test_accuracy"] - ce_acc
    summary = summary[
        [
            "run_name",
            "variant_type",
            "lambda_supcon",
            "projection_dim",
            "freeze_backbone",
            "base_model",
            "best_epoch",
            "best_val_macro_f1",
            "best_val_accuracy",
            "test_macro_f1",
            "test_accuracy",
            "test_weighted_f1",
            "delta_macro_vs_d13b_m16",
            "delta_acc_vs_d13b_m16",
            "delta_macro_vs_ce_continue",
            "delta_acc_vs_ce_continue",
            "checkpoint_exists",
            "checker_decision",
        ]
    ].sort_values("test_macro_f1", ascending=False)

    learning = pd.DataFrame(learning_rows)
    supcon = pd.DataFrame(supcon_rows)
    slot = pd.DataFrame(slot_rows)
    pred_df = pd.DataFrame(pred_rows)
    per_class = pd.DataFrame(per_class_rows)
    ce_classes = per_class[per_class["run_name"] == "d13c_m16_ce_continue"]
    if not ce_classes.empty:
        ce_row = ce_classes.iloc[0].to_dict()
        for name in CLASS_NAMES:
            per_class[f"delta_{name}_vs_ce_continue"] = per_class[name] - as_float(ce_row.get(name))
        per_class["delta_hard_class_macro_vs_ce_continue"] = per_class["hard_class_macro"] - as_float(ce_row.get("hard_class_macro"))
    else:
        for name in CLASS_NAMES:
            per_class[f"delta_{name}_vs_ce_continue"] = np.nan
        per_class["delta_hard_class_macro_vs_ce_continue"] = np.nan
    per_class["per_class_status"] = np.where(
        per_class["delta_hard_class_macro_vs_ce_continue"] >= 0.002,
        "HARD_CLASSES_IMPROVED_VS_CE",
        np.where(per_class["delta_hard_class_macro_vs_ce_continue"] <= -0.01, "HARD_CLASSES_DEGRADED_VS_CE", "HARD_CLASSES_SIMILAR_TO_CE"),
    )
    per_class = per_class.sort_values("hard_class_macro", ascending=False)

    l002_macro = as_float(summary.loc[summary["run_name"] == "d13c_m16_supcon_l002", "test_macro_f1"].iloc[0]) if (summary["run_name"] == "d13c_m16_supcon_l002").any() else np.nan
    status_rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        run_name = str(row["run_name"])
        slot_status_value = str(slot.loc[slot["run_name"] == run_name, "slot_health_status"].iloc[0])
        pred_status_value = str(pred_df.loc[pred_df["run_name"] == run_name, "pred_status"].iloc[0])
        per_status = str(per_class.loc[per_class["run_name"] == run_name, "per_class_status"].iloc[0])
        learn_status = str(learning.loc[learning["run_name"] == run_name, "learning_status"].iloc[0])
        delta_ce = as_float(row["delta_macro_vs_ce_continue"])
        delta_ref = as_float(row["delta_macro_vs_d13b_m16"])
        if pred_status_value in {"HARD_COLLAPSE", "COLLAPSE_RISK"}:
            final_tag = "COLLAPSE_INVALID"
        elif slot_status_value != "SLOT_HEALTHY":
            final_tag = "SLOT_DEGRADED_REVIEW"
        elif row["variant_type"] == "ce_continue":
            final_tag = "CE_CONTINUATION_ONLY"
        elif row["variant_type"] == "m8_control":
            final_tag = "M8_CONTROL_USEFUL"
        elif delta_ce <= 0:
            final_tag = "SUPCON_NOT_HELPFUL"
        elif learn_status == "still_learning":
            final_tag = "PROMISING_NEEDS_EXTEND"
        elif delta_ce > 0.002:
            final_tag = "SUPCON_HELPFUL"
        elif delta_ref > 0.001 and delta_ce > 0:
            final_tag = "STRONG_D13C_CANDIDATE"
        else:
            final_tag = "REVIEW_MANUALLY"
        if final_tag in {"SUPCON_HELPFUL", "STRONG_D13C_CANDIDATE", "PROMISING_NEEDS_EXTEND"}:
            rec = "post-D13C visual slot audit required before any downstream choice"
        elif final_tag == "CE_CONTINUATION_ONLY":
            rec = "treat gain as extra fine-tuning, not SupCon evidence"
        elif final_tag == "M8_CONTROL_USEFUL":
            rec = "use as compact control, not replacement unless visual/compactness is prioritized"
        elif final_tag == "SUPCON_NOT_HELPFUL":
            rec = "do not select this SupCon setting"
        elif final_tag == "SLOT_DEGRADED_REVIEW":
            rec = "manual review required before candidate selection"
        else:
            rec = "review manually"
        if run_name == "d13c_m16_supcon_l002_proj128" and np.isfinite(l002_macro) and as_float(row["test_macro_f1"]) > l002_macro + 0.002:
            sup_status = "PROJ128_HELPFUL"
        elif row["variant_type"] == "freeze_backbone" and np.isfinite(l002_macro) and as_float(row["test_macro_f1"]) < l002_macro - 0.002:
            sup_status = "FREEZE_NOT_ENOUGH"
        elif row["variant_type"] == "supcon_lambda" and delta_ce > 0.002:
            sup_status = "SUPCON_HELPFUL"
        elif row["variant_type"] in {"supcon_lambda", "proj128", "freeze_backbone", "m8_control"} and delta_ce <= 0:
            sup_status = "SUPCON_NOT_HELPFUL"
        elif as_float(row["lambda_supcon"]) >= 0.10 and delta_ce < -0.002:
            sup_status = "SUPCON_TOO_STRONG"
        elif row["variant_type"] == "ce_continue":
            sup_status = "NO_SUPCON"
        else:
            sup_status = "NO_CLEAR_SUPCON_SIGNAL"
        status_rows.append(
            {
                "run_name": run_name,
                "score_status": "ABOVE_D13B_M16" if delta_ref > 0 else "BELOW_D13B_M16",
                "supcon_status": sup_status,
                "slot_health_status": slot_status_value,
                "pred_status": pred_status_value,
                "per_class_status": per_status,
                "learning_status": learn_status,
                "final_tag": final_tag,
                "recommendation": rec,
            }
        )

    recommendations = pd.DataFrame(status_rows)
    vs_ref = summary[
        [
            "run_name",
            "base_model",
            "test_macro_f1",
            "test_accuracy",
            "delta_macro_vs_d13b_m16",
            "delta_acc_vs_d13b_m16",
            "delta_macro_vs_ce_continue",
            "delta_acc_vs_ce_continue",
        ]
    ].copy()
    vs_ref["d13b_m16_reference_macro_f1"] = D13B_M16_REFERENCE["test_macro_f1"]
    vs_ref["d13b_m16_reference_acc"] = D13B_M16_REFERENCE["test_acc"]
    vs_ref["d13b_m8_reference_macro_f1"] = D13B_M8_REFERENCE["test_macro_f1"]
    vs_ref["d13b_m8_reference_acc"] = D13B_M8_REFERENCE["test_acc"]
    vs_ref["delta_macro_vs_own_d13b_ref"] = np.where(
        vs_ref["base_model"] == "m8",
        vs_ref["test_macro_f1"] - D13B_M8_REFERENCE["test_macro_f1"],
        vs_ref["test_macro_f1"] - D13B_M16_REFERENCE["test_macro_f1"],
    )
    vs_ref["delta_acc_vs_own_d13b_ref"] = np.where(
        vs_ref["base_model"] == "m8",
        vs_ref["test_accuracy"] - D13B_M8_REFERENCE["test_acc"],
        vs_ref["test_accuracy"] - D13B_M16_REFERENCE["test_acc"],
    )
    warnings_df = pd.DataFrame([{"run_name": k, "artifact_warnings": v} for k, v in warnings_by_run.items()])

    return {
        "summary": summary,
        "learning": learning.sort_values("val_macro_best", ascending=False),
        "supcon": supcon,
        "slot": slot,
        "pred": pred_df,
        "per_class": per_class,
        "vs_ref": vs_ref,
        "recommendations": recommendations,
        "warnings": warnings_df,
        "raw": raw,  # type: ignore[dict-item]
    }


def save_figures(tables: Dict[str, Any], output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw: Dict[str, Dict[str, pd.DataFrame]] = tables["raw"]
    summary: pd.DataFrame = tables["summary"]
    per_class: pd.DataFrame = tables["per_class"]
    slot: pd.DataFrame = tables["slot"]
    pred: pd.DataFrame = tables["pred"]

    def line_plot(filename: str, split_name: str, col: str, title: str, ylabel: str) -> None:
        plt.figure(figsize=(11, 6))
        for run_name, frames in raw.items():
            df = frames[split_name]
            if df.empty or col not in df.columns:
                continue
            x = pd.to_numeric(df.get("epoch", pd.Series(range(1, len(df) + 1))), errors="coerce")
            y = pd.to_numeric(df[col], errors="coerce")
            plt.plot(x, y, label=run_name)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=150)
        plt.close()

    line_plot("val_macro_curves_all.png", "val", "val_macro_f1", "Validation macro-F1", "val_macro_f1")
    line_plot("train_loss_curves_all.png", "train", "train_loss", "Training loss", "train_loss")

    plt.figure(figsize=(11, 6))
    for run_name, frames in raw.items():
        df = frames["train"]
        if df.empty:
            continue
        x = pd.to_numeric(df.get("epoch", pd.Series(range(1, len(df) + 1))), errors="coerce")
        if "train_loss_ce" in df.columns:
            plt.plot(x, pd.to_numeric(df["train_loss_ce"], errors="coerce"), label=f"{run_name} CE", linestyle="-")
        if "train_loss_supcon" in df.columns:
            plt.plot(x, pd.to_numeric(df["train_loss_supcon"], errors="coerce"), label=f"{run_name} SupCon", linestyle="--")
    plt.title("CE vs SupCon train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(figures / "ce_vs_supcon_loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    for run_name, frames in raw.items():
        df = frames["supcon"]
        if df.empty or "loss_supcon" not in df.columns:
            continue
        work = df[df["split"].astype(str) == "train"] if "split" in df.columns else df
        x = pd.to_numeric(work.get("epoch", pd.Series(range(1, len(work) + 1))), errors="coerce")
        plt.plot(x, pd.to_numeric(work["loss_supcon"], errors="coerce"), label=run_name)
    plt.title("SupCon loss curves")
    plt.xlabel("epoch")
    plt.ylabel("loss_supcon")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "supcon_loss_curves.png", dpi=150)
    plt.close()

    bar_df = summary.sort_values("test_macro_f1", ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(bar_df["run_name"], bar_df["test_macro_f1"])
    plt.axvline(D13B_M16_REFERENCE["test_macro_f1"], linestyle="--", color="black", linewidth=1, label="D13B M16 ref")
    plt.xlabel("test_macro_f1")
    plt.title("D13C test macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "test_macro_bar.png", dpi=150)
    plt.close()

    heat_cols = HARD_CLASSES
    heat = per_class.set_index("run_name")[heat_cols].apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(8, max(4, 0.45 * len(heat))))
    plt.imshow(heat.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="F1")
    plt.xticks(range(len(heat_cols)), heat_cols)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.title("Hard-class F1")
    plt.tight_layout()
    plt.savefig(figures / "hard_class_f1_heatmap.png", dpi=150)
    plt.close()

    slot_plot = slot.set_index("run_name")[["effective_slots_test", "slot_entropy_test", "slot_overlap_test", "slot_dominance_test"]]
    slot_plot.plot(kind="bar", figsize=(11, 5))
    plt.title("Slot health summary")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(figures / "slot_health_bar.png", dpi=150)
    plt.close()

    pred_cols = [f"pred_count_{name}" for name in CLASS_NAMES]
    pred_heat = pred.set_index("run_name")[pred_cols].apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(10, max(4, 0.45 * len(pred_heat))))
    plt.imshow(pred_heat.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="test predicted count")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(pred_heat.index)), pred_heat.index)
    plt.title("Prediction count heatmap")
    plt.tight_layout()
    plt.savefig(figures / "pred_count_heatmap.png", dpi=150)
    plt.close()


def final_decision(summary: pd.DataFrame, recs: pd.DataFrame, learning: pd.DataFrame, per_class: pd.DataFrame) -> str:
    if summary.empty:
        return "D13C_INCONCLUSIVE"
    if recs["pred_status"].isin(["COLLAPSE_RISK", "HARD_COLLAPSE"]).any():
        return "D13C_COLLAPSE_STOP"
    ce_macro = as_float(summary.loc[summary["run_name"] == "d13c_m16_ce_continue", "test_macro_f1"].iloc[0]) if (summary["run_name"] == "d13c_m16_ce_continue").any() else np.nan
    sup = summary[(summary["variant_type"] != "ce_continue") & (summary["base_model"] == "m16")]
    best_sup_delta_ce = as_float(sup["test_macro_f1"].max() - ce_macro) if not sup.empty and np.isfinite(ce_macro) else np.nan
    best_delta_ref = as_float(summary["delta_macro_vs_d13b_m16"].max())
    still_count = int((learning["learning_status"] == "still_learning").sum()) if not learning.empty else 0
    if still_count >= 4:
        return "D13C_NEEDS_EXTENDED_TRAINING"
    if np.isfinite(best_sup_delta_ce) and best_sup_delta_ce > 0.002:
        best_run = str(sup.sort_values("test_macro_f1", ascending=False).iloc[0]["run_name"])
        pc = per_class[per_class["run_name"] == best_run]
        hard_ok = pc.empty or as_float(pc.iloc[0].get("delta_hard_class_macro_vs_ce_continue", 0.0)) > -0.01
        slot_ok = recs.loc[recs["run_name"] == best_run, "slot_health_status"].iloc[0] == "SLOT_HEALTHY"
        pred_ok = recs.loc[recs["run_name"] == best_run, "pred_status"].iloc[0] == "NO_COLLAPSE"
        if hard_ok and slot_ok and pred_ok:
            return "D13C_DIAGNOSTIC_PASS_READY_FOR_POST_VISUAL_SLOT_AUDIT"
        return "D13C_INCONCLUSIVE"
    if np.isfinite(best_delta_ref) and best_delta_ref > 0.001 and (not np.isfinite(best_sup_delta_ce) or best_sup_delta_ce <= 0.002):
        return "D13C_SUPCON_NOT_HELPFUL_KEEP_D13B_FINAL"
    if np.isfinite(best_sup_delta_ce) and 0 < best_sup_delta_ce <= 0.002:
        return "D13C_NEEDS_LAMBDA_TUNING"
    return "D13C_SUPCON_NOT_HELPFUL_KEEP_D13B_FINAL"


def write_report(tables: Dict[str, Any], output_dir: Path) -> str:
    summary: pd.DataFrame = tables["summary"]
    learning: pd.DataFrame = tables["learning"]
    supcon: pd.DataFrame = tables["supcon"]
    slot: pd.DataFrame = tables["slot"]
    pred: pd.DataFrame = tables["pred"]
    per_class: pd.DataFrame = tables["per_class"]
    vs_ref: pd.DataFrame = tables["vs_ref"]
    recs: pd.DataFrame = tables["recommendations"]
    warnings: pd.DataFrame = tables["warnings"]
    decision = final_decision(summary, recs, learning, per_class)

    top_test = summary.sort_values("test_macro_f1", ascending=False)
    top_val = summary.sort_values("best_val_macro_f1", ascending=False)
    top_hard = per_class.sort_values("hard_class_macro", ascending=False)
    best_sup = summary[summary["variant_type"].isin(["supcon_lambda", "freeze_backbone", "proj128"])].sort_values("test_macro_f1", ascending=False)
    best_sup_name = str(best_sup.iloc[0]["run_name"]) if not best_sup.empty else "none"
    ce_row = summary[summary["run_name"] == "d13c_m16_ce_continue"]
    ce_note = "CE-only continuation is missing."
    if not ce_row.empty:
        ce = ce_row.iloc[0]
        ce_note = (
            f"CE-only continuation reaches test_macro_f1={ce['test_macro_f1']:.4f} "
            f"and delta_vs_D13B_M16={ce['delta_macro_vs_d13b_m16']:+.4f}. "
            "Any SupCon gain must be judged against this fine-tune continuation baseline."
        )

    lines = [
        "# D13C Diagnostic Deep Analysis Report",
        "",
        "## 1. Context",
        "- D13C is diagnostic only.",
        "- Base reference is D13B M16 deep readout: test_macro_f1 = 0.6187, test_acc = 0.6328.",
        "- D13C uses image-level SupCon on pooled slot representation.",
        "- No prototype, no motif-level SupCon, no full D13C, no motif claim, no semantic-region claim, and no causal-evidence claim.",
        "",
        "## 2. Overall Ranking",
        "Top by test macro-F1:",
        "",
        md_table(top_test, ["run_name", "variant_type", "test_macro_f1", "test_accuracy", "delta_macro_vs_d13b_m16", "delta_macro_vs_ce_continue"], 8),
        "",
        "Top by best validation macro-F1:",
        "",
        md_table(top_val, ["run_name", "best_epoch", "best_val_macro_f1", "best_val_accuracy", "test_macro_f1"], 8),
        "",
        "Top by hard-class macro (Angry, Disgust, Fear, Sad):",
        "",
        md_table(top_hard, ["run_name", "hard_class_macro", "Angry", "Disgust", "Fear", "Sad", "delta_hard_class_macro_vs_ce_continue"], 8),
        "",
        "## 3. Fair Comparison Against References",
        ce_note,
        "",
        md_table(vs_ref.sort_values("test_macro_f1", ascending=False), ["run_name", "base_model", "test_macro_f1", "test_accuracy", "delta_macro_vs_d13b_m16", "delta_macro_vs_ce_continue", "delta_macro_vs_own_d13b_ref"], 8),
        "",
        "Interpretation rule: if CE-only continuation is comparable to or better than SupCon, the improvement is extra fine-tuning evidence, not SupCon evidence.",
        "",
        "## 4. Learning Dynamics",
        md_table(learning.sort_values("val_macro_best", ascending=False), ["run_name", "val_macro_best", "val_macro_last", "val_macro_slope_last5", "val_macro_volatility_last10", "best_epoch", "best_at_final_epoch", "learning_status", "decision_recommendation"], 8),
        "",
        "Runs marked still_learning can justify an extended diagnostic rerun later, but this report does not train further.",
        "",
        "## 5. SupCon Effect",
        f"Best SupCon-family run by test macro-F1: `{best_sup_name}`.",
        "",
        md_table(supcon.merge(summary[["run_name", "test_macro_f1", "delta_macro_vs_ce_continue"]], on="run_name", how="left").sort_values("test_macro_f1", ascending=False), ["run_name", "lambda_supcon", "projection_dim", "freeze_backbone", "supcon_loss_first", "supcon_loss_last", "supcon_loss_slope_last10", "positive_pair_count_mean", "valid_anchor_mean", "embedding_collapse_score_last", "supcon_active_rate", "test_macro_f1", "delta_macro_vs_ce_continue"], 8),
        "",
        "- Lambda sweep should be read against CE-only, not only against D13B.",
        "- Freeze-backbone and projection-128 conclusions are local to this diagnostic setup.",
        "- M8 control is a compact-control readout, not evidence to replace M16 unless its accuracy and audit quality justify that trade.",
        "",
        "## 6. Slot Health",
        md_table(slot.sort_values("run_name"), ["run_name", "effective_slots_test", "slot_overlap_test", "slot_entropy_test", "slot_dominance_test", "slot_health_status"], 8),
        "",
        "Slot health is a diagnostic sanity check only. It does not establish motif or semantic-region validity.",
        "",
        "## 7. Prediction Collapse",
        md_table(pred.sort_values("pred_max_ratio", ascending=False), ["run_name", "pred_total", "pred_max_ratio", "classes_predicted_count", "pred_entropy", "pred_status"], 8),
        "",
        "## 8. Per-class Behavior",
        md_table(per_class.sort_values("hard_class_macro", ascending=False), ["run_name", "hard_class_macro", "Angry", "Disgust", "Fear", "Sad", "Happy", "Neutral", "per_class_status"], 8),
        "",
        "Hard-class gains are necessary to trust a macro-F1 improvement. Accuracy-only gains from easier classes should not drive candidate selection.",
        "",
        "## 9. Candidate Selection",
        md_table(recs.merge(summary[["run_name", "test_macro_f1", "delta_macro_vs_ce_continue"]], on="run_name", how="left").sort_values("test_macro_f1", ascending=False), ["run_name", "test_macro_f1", "delta_macro_vs_ce_continue", "score_status", "supcon_status", "slot_health_status", "pred_status", "per_class_status", "learning_status", "final_tag"], 8),
        "",
        "Candidate rule: choose a D13C candidate only if it beats CE-only or D13B clearly, avoids prediction collapse, keeps slot health sane, and does not harm hard classes.",
        "",
        "## 10. Required Next Step",
        "If a D13C run is selected, post-D13C visual slot audit is mandatory before any downstream decision. If SupCon is not helpful versus CE-only, keep D13B final.",
        "",
        "## 11. Final Decision",
        decision,
        "",
        "## Artifact Warnings",
        md_table(warnings, ["run_name", "artifact_warnings"], 20),
        "",
    ]
    (output_dir / "d13c_deep_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="outputs/d13c_diagnostic")
    parser.add_argument("--output_dir", default="outputs/d13c_diagnostic/deep_analysis")
    args = parser.parse_args()
    root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    tables = collect(root, project_root)

    tables["summary"].to_csv(output_dir / "d13c_deep_summary.csv", index=False)
    tables["learning"].to_csv(output_dir / "d13c_learning_dynamics.csv", index=False)
    tables["supcon"].to_csv(output_dir / "d13c_supcon_effect_summary.csv", index=False)
    tables["slot"].to_csv(output_dir / "d13c_slot_health_summary.csv", index=False)
    tables["pred"].to_csv(output_dir / "d13c_pred_count_summary.csv", index=False)
    tables["per_class"].to_csv(output_dir / "d13c_per_class_summary.csv", index=False)
    tables["vs_ref"].to_csv(output_dir / "d13c_vs_d13b_reference.csv", index=False)
    tables["recommendations"].to_csv(output_dir / "d13c_recommendation_table.csv", index=False)
    save_figures(tables, output_dir)
    decision = write_report(tables, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "runs": int(len(tables["summary"])), "final_decision": decision}, indent=2))


if __name__ == "__main__":
    main()
