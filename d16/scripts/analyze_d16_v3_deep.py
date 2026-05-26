"""Deep read-only analysis for completed D16 v3 runs.

This script reads existing artifacts only. It does not train, mutate model code,
or make motif, semantic-region, causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import yaml


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
HARD_CLASS_IDS = [0, 2, 4]
FOCUS_CONFUSIONS = [(2, 4), (4, 6), (6, 4), (0, 4), (1, 0), (1, 4), (1, 6)]
FORBIDDEN_TOKENS = [
    "MOTIF_DISCOVERED",
    "SEMANTIC_REGION_DISCOVERED",
    "CAUSAL_EVIDENCE_CONFIRMED",
    "FULL_INTERPRETABILITY_CLAIM",
]

D15 = {
    "name": "D15_m8_basic_150",
    "accuracy": 0.645026,
    "macro_f1": 0.622471,
    "weighted_f1": 0.641866,
    "per_class_f1": {
        0: 0.558704,
        1: 0.593220,
        2: 0.465016,
        3: 0.844974,
        4: 0.497946,
        5: 0.772050,
        6: 0.625387,
    },
}
V1_BEST = {
    "name": "fallback_weighted_ce",
    "path": Path("outputs/d16_runs/v1/d16_v1_face_plus_context_fallback_weighted_ce"),
    "accuracy": 0.639175,
    "macro_f1": 0.632938,
    "fallback_macro_f1": 0.409767,
    "hard_f1": 0.510704,
}
V2_BEST = {
    "name": "w20_supcon_l002",
    "path": Path("outputs/d16_runs/v2/d16_v2_face_plus_context_fallback_w20_supcon_l002"),
    "accuracy": 0.635274,
    "macro_f1": 0.623511,
    "fallback_macro_f1": 0.456697,
    "hard_f1": 0.507355,
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fields = keys or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result


def finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def metric(df: pd.DataFrame, column: str, default: float = float("nan")) -> float:
    if df.empty or column not in df.columns:
        return default
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else default


def first_metric(df: pd.DataFrame, column: str, default: float = float("nan")) -> float:
    if df.empty or column not in df.columns:
        return default
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else default


def weighted_f1(per_class: pd.DataFrame) -> float:
    if per_class.empty or not {"support", "f1"}.issubset(per_class.columns):
        return float("nan")
    support = pd.to_numeric(per_class["support"], errors="coerce").fillna(0)
    f1 = pd.to_numeric(per_class["f1"], errors="coerce").fillna(0)
    return float((support * f1).sum() / max(float(support.sum()), 1.0))


def class_f1(per_class: pd.DataFrame, class_id: int) -> float:
    if per_class.empty or not {"class_id", "f1"}.issubset(per_class.columns):
        return float("nan")
    cls = pd.to_numeric(per_class["class_id"], errors="coerce")
    subset = per_class[cls == int(class_id)]
    return metric(subset, "f1")


def hard_f1(per_class: pd.DataFrame) -> float:
    values = [class_f1(per_class, class_id) for class_id in HARD_CLASS_IDS]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else float("nan")


def slope_last(values: Sequence[float], n: int = 10) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if len(values) < 2:
        return float("nan")
    tail = values[-n:]
    return float((tail[-1] - tail[0]) / max(len(tail) - 1, 1))


def pred_distribution(pred_count: pd.DataFrame) -> Dict[str, Any]:
    if pred_count.empty or "pred_count" not in pred_count.columns:
        return {
            "predicted_classes": 0,
            "pred_total": 0,
            "pred_max_class": "",
            "pred_max_ratio": float("nan"),
            "class1_pred_ratio": float("nan"),
            "class2_pred_ratio": float("nan"),
            "class4_pred_ratio": float("nan"),
            "class6_pred_ratio": float("nan"),
            "collapse_risk": "MISSING_PRED_COUNT",
            "collapse_risk_order": 5,
        }
    rows = pred_count.copy()
    rows["class_id"] = pd.to_numeric(rows.get("class_id"), errors="coerce")
    rows["pred_count"] = pd.to_numeric(rows["pred_count"], errors="coerce").fillna(0)
    counts = {int(row["class_id"]): int(row["pred_count"]) for _, row in rows.dropna(subset=["class_id"]).iterrows()}
    total = int(sum(counts.values()))
    predicted_classes = int(sum(1 for value in counts.values() if value > 0))
    max_class = max(counts, key=lambda class_id: counts[class_id]) if counts else None
    max_ratio = counts[max_class] / total if max_class is not None and total > 0 else float("nan")
    c1 = counts.get(1, 0) / total if total > 0 else float("nan")
    c2 = counts.get(2, 0) / total if total > 0 else float("nan")
    c4 = counts.get(4, 0) / total if total > 0 else float("nan")
    c6 = counts.get(6, 0) / total if total > 0 else float("nan")
    if predicted_classes <= 2:
        risk, order = "COLLAPSE_RISK", 5
    elif predicted_classes < 7:
        risk, order = "HARD_CLASS_SUPPRESSION", 4
    elif total > 0 and counts.get(1, 0) == 0:
        risk, order = "CLASS_1_SUPPRESSION", 3
    elif (math.isfinite(c2) and c2 < 0.04) or (math.isfinite(c4) and c4 < 0.04):
        risk, order = "HARD_CLASS_SUPPRESSION", 4
    elif math.isfinite(max_ratio) and max_ratio > 0.45:
        risk, order = "COLLAPSE_RISK", 5
    elif math.isfinite(max_ratio) and max_ratio > 0.35:
        risk, order = "MILD_PRED_BIAS", 2
    else:
        risk, order = "NO_COLLAPSE", 0
    return {
        "predicted_classes": predicted_classes,
        "pred_total": total,
        "pred_max_class": "" if max_class is None else f"{max_class}:{CLASS_NAMES.get(max_class, max_class)}",
        "pred_max_ratio": float(max_ratio),
        "class1_pred_ratio": float(c1),
        "class2_pred_ratio": float(c2),
        "class4_pred_ratio": float(c4),
        "class6_pred_ratio": float(c6),
        "collapse_risk": risk,
        "collapse_risk_order": order,
    }


def confusion_focus(confusion: pd.DataFrame, run_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if confusion.empty or not {"true_class", "pred_class", "count"}.issubset(confusion.columns):
        return rows
    data = confusion.copy()
    data["true_class"] = pd.to_numeric(data["true_class"], errors="coerce")
    data["pred_class"] = pd.to_numeric(data["pred_class"], errors="coerce")
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    if "row_ratio" in data.columns:
        data["row_ratio"] = pd.to_numeric(data["row_ratio"], errors="coerce")
    for true_cls, pred_cls in FOCUS_CONFUSIONS:
        subset = data[(data["true_class"] == true_cls) & (data["pred_class"] == pred_cls)]
        count = int(subset["count"].iloc[-1]) if not subset.empty else 0
        ratio = float(subset["row_ratio"].iloc[-1]) if not subset.empty and "row_ratio" in subset.columns else float("nan")
        rows.append(
            {
                "run_name": run_name,
                "true_class": true_cls,
                "true_name": CLASS_NAMES.get(true_cls, str(true_cls)),
                "pred_class": pred_cls,
                "pred_name": CLASS_NAMES.get(pred_cls, str(pred_cls)),
                "count": count,
                "row_ratio": ratio,
            }
        )
    top = data[data["true_class"] != data["pred_class"]].sort_values(["count"], ascending=False).head(10)
    for _, row in top.iterrows():
        rows.append(
            {
                "run_name": run_name,
                "true_class": int(row["true_class"]),
                "true_name": CLASS_NAMES.get(int(row["true_class"]), str(int(row["true_class"]))),
                "pred_class": int(row["pred_class"]),
                "pred_name": CLASS_NAMES.get(int(row["pred_class"]), str(int(row["pred_class"]))),
                "count": int(row["count"]),
                "row_ratio": safe_float(row.get("row_ratio")),
                "kind": "top_confusion",
            }
        )
    return rows


def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg = read_json(run_dir / "resolved_config.json")
    if cfg:
        return cfg
    return read_yaml(run_dir / "resolved_config.yaml")


def checkpoint_contract_status(test: pd.DataFrame) -> str:
    if test.empty:
        return "TEST_METRICS_MISSING"
    missing = [col for col in ["checkpoint_name", "checkpoint_epoch"] if col not in test.columns]
    if missing:
        return "TEST_CONTRACT_INCOMPLETE"
    if str(test["checkpoint_name"].iloc[-1]) != "best.pt":
        return "TEST_NOT_BEST_CHECKPOINT"
    return "TEST_BEST_CONTRACT_OK"


def head_routing_status(predictions: pd.DataFrame, dual_head: bool) -> Dict[str, Any]:
    if predictions.empty:
        return {
            "head_contract_status": "PREDICTIONS_MISSING",
            "routed_head_missing": True,
            "head_unused_warning": False,
            "wrong_route_count": "",
            "detected_head_count": 0,
            "fallback_head_count": 0,
            "single_head_count": 0,
            "detected_count": 0,
            "fallback_count": 0,
        }
    if "routed_head" not in predictions.columns:
        return {
            "head_contract_status": "DUAL_HEAD_CONTRACT_FAIL" if dual_head else "SINGLE_HEAD_NO_ROUTED_COLUMN",
            "routed_head_missing": True,
            "head_unused_warning": False,
            "wrong_route_count": "",
            "detected_head_count": 0,
            "fallback_head_count": 0,
            "single_head_count": 0,
            "detected_count": int(len(predictions)),
            "fallback_count": 0,
        }
    heads = predictions["routed_head"].astype(str)
    detected_head_count = int((heads == "detected_head").sum())
    fallback_head_count = int((heads == "fallback_head").sum())
    single_head_count = int((heads == "single_head").sum())
    detected_count = int((pd.to_numeric(predictions.get("landmark_missing_flag"), errors="coerce") == 0).sum()) if "landmark_missing_flag" in predictions.columns else int((pd.to_numeric(predictions.get("detected"), errors="coerce") == 1).sum())
    fallback_count = int((pd.to_numeric(predictions.get("landmark_missing_flag"), errors="coerce") == 1).sum()) if "landmark_missing_flag" in predictions.columns else int((pd.to_numeric(predictions.get("detected"), errors="coerce") == 0).sum())
    wrong_route_count: Any = ""
    if "landmark_missing_flag" in predictions.columns:
        missing = pd.to_numeric(predictions["landmark_missing_flag"], errors="coerce")
        wrong_route_count = int(((missing == 0) & (heads != "detected_head")).sum() + ((missing == 1) & (heads != "fallback_head")).sum())
    head_unused = bool(dual_head and detected_count > 0 and fallback_count > 0 and (detected_head_count == 0 or fallback_head_count == 0))
    if dual_head and wrong_route_count not in ("", 0):
        status = "HEAD_ROUTING_FAIL_WRONG_ROUTE"
    elif dual_head and head_unused:
        status = "HEAD_ROUTING_WARNING_ONE_HEAD_UNUSED"
    elif dual_head:
        status = "HEAD_ROUTING_VALID"
    else:
        status = "SINGLE_HEAD_CONTROL"
    return {
        "head_contract_status": status,
        "routed_head_missing": False,
        "head_unused_warning": head_unused,
        "wrong_route_count": wrong_route_count,
        "detected_head_count": detected_head_count,
        "fallback_head_count": fallback_head_count,
        "single_head_count": single_head_count,
        "detected_count": detected_count,
        "fallback_count": fallback_count,
    }


def group_metrics(fallback: pd.DataFrame) -> Dict[str, float]:
    detected = fallback[fallback["group"].astype(str) == "detected"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    fb = fallback[fallback["group"].astype(str) == "fallback"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    detected_acc = metric(detected, "accuracy")
    fallback_acc = metric(fb, "accuracy")
    detected_macro = metric(detected, "macro_f1")
    fallback_macro = metric(fb, "macro_f1")
    return {
        "detected_accuracy": detected_acc,
        "detected_macro_f1": detected_macro,
        "fallback_accuracy": fallback_acc,
        "fallback_macro_f1": fallback_macro,
        "detected_minus_fallback_acc_gap": detected_acc - fallback_acc if finite(detected_acc) and finite(fallback_acc) else float("nan"),
        "detected_minus_fallback_macro_gap": detected_macro - fallback_macro if finite(detected_macro) and finite(fallback_macro) else float("nan"),
    }


def run_type(name: str, cfg: Dict[str, Any]) -> str:
    if "repeat_v1" in name:
        return "v1_seed_repeat"
    if cfg.get("model", {}).get("dual_head"):
        return "dual_head"
    return "single_head_control"


def run_decision(row: Dict[str, Any]) -> str:
    if row.get("missing_artifacts") or row.get("head_contract_status") in {
        "DUAL_HEAD_CONTRACT_FAIL",
        "HEAD_ROUTING_FAIL_WRONG_ROUTE",
        "PREDICTIONS_MISSING",
        "TEST_METRICS_MISSING",
    }:
        return "D16_V3_INVALID_OR_INCOMPLETE"
    macro = safe_float(row.get("test_macro_f1"))
    acc = safe_float(row.get("test_accuracy"))
    collapse = str(row.get("collapse_risk")) == "NO_COLLAPSE"
    if macro > V1_BEST["macro_f1"] and acc >= 0.634 and collapse:
        if macro >= 0.645 and acc >= 0.650 and safe_float(row.get("fallback_macro_f1")) >= 0.43 and safe_float(row.get("hard_F1")) >= 0.52:
            return "STRONG_NEW_BEST"
        return "D16_V3_BEATS_V1_BEST"
    if macro > D15["macro_f1"] and collapse:
        return "D16_V3_BEATS_D15_ONLY"
    if abs(macro - V1_BEST["macro_f1"]) <= 0.005 and collapse:
        return "D16_V3_NEAR_V1_BEST"
    return "D16_V3_BELOW_V1_BEST"


def classify_train_dynamics(train: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any]:
    if train.empty:
        return {
            "best_epoch": summary.get("best_epoch", ""),
            "early_stop_epoch": "",
            "val_macro_f1_best": float("nan"),
            "val_macro_f1_final": float("nan"),
            "train_loss_first": float("nan"),
            "train_loss_final": float("nan"),
            "val_slope_last10": float("nan"),
            "train_loss_slope_last10": float("nan"),
            "learning_decision": "DO_NOT_RESUME_SAME_SETTING",
        }
    epochs = pd.to_numeric(train.get("epoch"), errors="coerce").dropna()
    val = pd.to_numeric(train.get("val_macro_f1"), errors="coerce").dropna() if "val_macro_f1" in train else pd.Series(dtype=float)
    loss = pd.to_numeric(train.get("train_loss"), errors="coerce").dropna() if "train_loss" in train else pd.Series(dtype=float)
    max_epoch = int(epochs.max()) if not epochs.empty else ""
    max_epochs_cfg = safe_float(summary.get("max_epochs"), 150.0)
    val_best = float(val.max()) if not val.empty else float("nan")
    val_final = float(val.iloc[-1]) if not val.empty else float("nan")
    loss_first = float(loss.iloc[0]) if not loss.empty else float("nan")
    loss_final = float(loss.iloc[-1]) if not loss.empty else float("nan")
    val_slope = slope_last(val.tolist())
    loss_slope = slope_last(loss.tolist())
    if max_epoch != "" and max_epoch < max_epochs_cfg:
        decision = "PLATEAU_EARLY_STOP_OK"
    elif math.isfinite(val_slope) and val_slope > 0.0005:
        decision = "UNDERTRAINED_EXTEND_CANDIDATE"
    elif math.isfinite(val_slope) and val_slope < -0.001 and math.isfinite(loss_slope) and loss_slope < 0:
        decision = "DO_NOT_RESUME_SAME_SETTING"
    else:
        decision = "SCHEDULER_TUNING_CANDIDATE"
    return {
        "best_epoch": summary.get("best_epoch", ""),
        "early_stop_epoch": max_epoch,
        "val_macro_f1_best": val_best,
        "val_macro_f1_final": val_final,
        "train_loss_first": loss_first,
        "train_loss_final": loss_final,
        "detected_loss_final": metric(train, "detected_loss_mean"),
        "fallback_loss_final": metric(train, "fallback_loss_mean"),
        "lr_at_best": metric(train, "lr"),
        "val_slope_last10": val_slope,
        "train_loss_slope_last10": loss_slope,
        "learning_decision": decision,
    }


def runtime_stats(train: pd.DataFrame) -> Dict[str, Any]:
    if train.empty:
        return {
            "epoch_time_mean": float("nan"),
            "epoch_time_total": float("nan"),
            "train_epoch_time_mean": float("nan"),
            "memory_reserved_mb_max": float("nan"),
        }
    epoch_times = pd.to_numeric(train.get("epoch_time_sec"), errors="coerce").dropna() if "epoch_time_sec" in train else pd.Series(dtype=float)
    train_times = pd.to_numeric(train.get("train_epoch_time_sec"), errors="coerce").dropna() if "train_epoch_time_sec" in train else pd.Series(dtype=float)
    mem = pd.to_numeric(train.get("memory_reserved_mb"), errors="coerce").dropna() if "memory_reserved_mb" in train else pd.Series(dtype=float)
    return {
        "epoch_time_mean": float(epoch_times.mean()) if not epoch_times.empty else float("nan"),
        "epoch_time_total": float(epoch_times.sum()) if not epoch_times.empty else float("nan"),
        "train_epoch_time_mean": float(train_times.mean()) if not train_times.empty else float("nan"),
        "memory_reserved_mb_max": float(mem.max()) if not mem.empty else float("nan"),
    }


def analyze_run(run_dir: Path, name: str) -> Dict[str, Any]:
    cfg = load_config(run_dir)
    summary = read_json(run_dir / "d16_train_summary.json")
    train = read_csv(run_dir / "train_log.csv")
    val_metrics = read_csv(run_dir / "val_metrics.csv")
    test = read_csv(run_dir / "test_metrics.csv")
    last_test = read_csv(run_dir / "last_test_metrics.csv")
    per_class = read_csv(run_dir / "per_class_metrics.csv")
    pred_count = read_csv(run_dir / "pred_count.csv")
    fallback = read_csv(run_dir / "detected_vs_fallback_metrics.csv")
    group_per_class = read_csv(run_dir / "detected_fallback_per_class_metrics.csv")
    predictions = read_csv(run_dir / "predictions.csv")
    confusion = read_csv(run_dir / "confusion_matrix.csv")
    checker = read_json(run_dir / "d16_v3_check_summary.json") or read_json(run_dir / "d16_v1_check_summary.json")
    expected = [
        "d16_train_summary.json",
        "train_log.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "per_class_metrics.csv",
        "pred_count.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "predictions.csv",
        "confusion_matrix.csv",
        "d16_report.md",
    ]
    missing = [item for item in expected if not (run_dir / item).exists()]
    dual = bool(cfg.get("model", {}).get("dual_head", False))
    loss_cfg = cfg.get("loss", {}) or {}
    groups = group_metrics(fallback)
    pred_stats = pred_distribution(pred_count)
    head = head_routing_status(predictions, dual)
    learning = classify_train_dynamics(train, summary)
    runtime = runtime_stats(train)
    row: Dict[str, Any] = {
        "run_name": name,
        "run_dir": str(run_dir),
        "run_type": run_type(name, cfg),
        "seed": cfg.get("seed", cfg.get("training", {}).get("seed", "")),
        "dual_head": dual,
        "fallback_weight": loss_cfg.get("fallback_weight", ""),
        "use_supcon": bool(safe_float(loss_cfg.get("lambda_part_supcon", 0.0), 0.0) > 0),
        "use_class_weight": bool(loss_cfg.get("mode") == "class_weighted_ce" or loss_cfg.get("class_weights") or loss_cfg.get("class_weights_auto")),
        "best_epoch": summary.get("best_epoch", learning.get("best_epoch", "")),
        "best_val_macro_f1": summary.get("best_val_macro_f1", learning.get("val_macro_f1_best", float("nan"))),
        "test_accuracy": metric(test, "accuracy"),
        "test_macro_f1": metric(test, "macro_f1"),
        "test_weighted_f1": weighted_f1(per_class),
        "last_test_accuracy": metric(last_test, "accuracy"),
        "last_test_macro_f1": metric(last_test, "macro_f1"),
        "hard_F1": hard_f1(per_class),
        "checker_decision": checker.get("decision", ""),
        "test_contract_status": checkpoint_contract_status(test),
        "missing_artifacts": ";".join(missing),
    }
    row.update(groups)
    row.update(pred_stats)
    row.update(head)
    row.update(learning)
    row.update(runtime)
    row["delta_vs_D15_macro_f1"] = row["test_macro_f1"] - D15["macro_f1"] if finite(row["test_macro_f1"]) else float("nan")
    row["delta_vs_v1_best_macro_f1"] = row["test_macro_f1"] - V1_BEST["macro_f1"] if finite(row["test_macro_f1"]) else float("nan")
    row["delta_vs_v2_best_macro_f1"] = row["test_macro_f1"] - V2_BEST["macro_f1"] if finite(row["test_macro_f1"]) else float("nan")
    row["delta_vs_v1_fallback_macro_f1"] = row["fallback_macro_f1"] - V1_BEST["fallback_macro_f1"] if finite(row["fallback_macro_f1"]) else float("nan")
    row["delta_vs_v2_fallback_macro_f1"] = row["fallback_macro_f1"] - V2_BEST["fallback_macro_f1"] if finite(row["fallback_macro_f1"]) else float("nan")
    row["run_decision"] = run_decision(row)
    row["_dfs"] = {
        "train": train,
        "val_metrics": val_metrics,
        "test": test,
        "last_test": last_test,
        "per_class": per_class,
        "pred_count": pred_count,
        "fallback": fallback,
        "group_per_class": group_per_class,
        "predictions": predictions,
        "confusion": confusion,
        "checker": checker,
        "summary": summary,
        "cfg": cfg,
    }
    return row


def load_control_per_class(path: Path) -> pd.DataFrame:
    return read_csv(path / "per_class_metrics.csv")


def per_class_compare(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    v1_per = load_control_per_class(V1_BEST["path"])
    v2_per = load_control_per_class(V2_BEST["path"])
    out: List[Dict[str, Any]] = []
    for class_id, class_name in CLASS_NAMES.items():
        row: Dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "D15_F1": D15["per_class_f1"].get(class_id, float("nan")),
            "v1_best_F1": class_f1(v1_per, class_id),
            "v2_best_F1": class_f1(v2_per, class_id),
        }
        best_name = ""
        best_value = -1.0
        for run in rows:
            value = class_f1(run["_dfs"]["per_class"], class_id)
            row[f"{run['run_name']}_F1"] = value
            if math.isfinite(value) and value > best_value:
                best_value = value
                best_name = str(run["run_name"])
        row["best_v3_for_class"] = best_name
        row["best_v3_F1"] = best_value if best_value >= 0 else float("nan")
        row["delta_best_v3_vs_v1"] = row["best_v3_F1"] - row["v1_best_F1"] if finite(row["best_v3_F1"]) and finite(row["v1_best_F1"]) else float("nan")
        row["delta_best_v3_vs_D15"] = row["best_v3_F1"] - row["D15_F1"] if finite(row["best_v3_F1"]) and finite(row["D15_F1"]) else float("nan")
        notes = []
        if class_id in HARD_CLASS_IDS:
            notes.append("hard_class")
        if class_id == 1:
            notes.append("low_support_fallback_sensitive")
        if row["delta_best_v3_vs_v1"] > 0:
            notes.append("v3_best_above_v1")
        elif row["delta_best_v3_vs_v1"] < -0.02:
            notes.append("v3_best_below_v1")
        row["notes"] = ";".join(notes)
        out.append(row)
    return out


def group_per_class_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in rows:
        df = run["_dfs"]["group_per_class"]
        if df.empty:
            out.append({"run_name": run["run_name"], "missing": "detected_fallback_per_class_metrics.csv"})
            continue
        for item in df.to_dict("records"):
            item["run_name"] = run["run_name"]
            out.append(item)
    return out


def head_routing_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "run_name": row["run_name"],
                "dual_head": row["dual_head"],
                "detected_count": row.get("detected_count"),
                "fallback_count": row.get("fallback_count"),
                "detected_head_count": row.get("detected_head_count"),
                "fallback_head_count": row.get("fallback_head_count"),
                "single_head_count": row.get("single_head_count"),
                "detected_head_accuracy": row.get("detected_accuracy"),
                "fallback_head_accuracy": row.get("fallback_accuracy"),
                "detected_head_macro_f1": row.get("detected_macro_f1"),
                "fallback_head_macro_f1": row.get("fallback_macro_f1"),
                "routed_head_missing": row.get("routed_head_missing"),
                "head_unused_warning": row.get("head_unused_warning"),
                "wrong_route_count": row.get("wrong_route_count"),
                "head_contract_status": row.get("head_contract_status"),
            }
        )
    return out


def prediction_distribution_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in rows:
        base = {
            "run_name": run["run_name"],
            "scope": "overall",
            "predicted_classes": run.get("predicted_classes"),
            "pred_total": run.get("pred_total"),
            "pred_max_class": run.get("pred_max_class"),
            "pred_max_ratio": run.get("pred_max_ratio"),
            "class1_pred_ratio": run.get("class1_pred_ratio"),
            "class2_pred_ratio": run.get("class2_pred_ratio"),
            "class4_pred_ratio": run.get("class4_pred_ratio"),
            "class6_pred_ratio": run.get("class6_pred_ratio"),
            "collapse_risk": run.get("collapse_risk"),
        }
        out.append(base)
        pred = run["_dfs"]["predictions"]
        if not pred.empty and "routed_head" in pred.columns and "y_pred" in pred.columns:
            for head in ["detected_head", "fallback_head", "single_head"]:
                subset = pred[pred["routed_head"].astype(str) == head]
                if subset.empty:
                    continue
                counts = subset["y_pred"].value_counts().to_dict()
                total = int(sum(int(v) for v in counts.values()))
                max_cls = max(counts, key=lambda key: counts[key]) if counts else ""
                out.append(
                    {
                        "run_name": run["run_name"],
                        "scope": head,
                        "predicted_classes": int(len(counts)),
                        "pred_total": total,
                        "pred_max_class": f"{max_cls}:{CLASS_NAMES.get(int(max_cls), max_cls)}" if str(max_cls) != "" else "",
                        "pred_max_ratio": counts[max_cls] / total if total else float("nan"),
                        "class1_pred_ratio": counts.get(1, counts.get("1", 0)) / total if total else float("nan"),
                        "class2_pred_ratio": counts.get(2, counts.get("2", 0)) / total if total else float("nan"),
                        "class4_pred_ratio": counts.get(4, counts.get("4", 0)) / total if total else float("nan"),
                        "class6_pred_ratio": counts.get(6, counts.get("6", 0)) / total if total else float("nan"),
                        "collapse_risk": "FALLBACK_HEAD_COLLAPSE_RISK" if head == "fallback_head" and len(counts) <= 2 else "",
                    }
                )
    return out


def learning_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = [
        "run_name",
        "best_epoch",
        "early_stop_epoch",
        "best_val_macro_f1",
        "val_macro_f1_best",
        "val_macro_f1_final",
        "train_loss_first",
        "train_loss_final",
        "detected_loss_final",
        "fallback_loss_final",
        "lr_at_best",
        "val_slope_last10",
        "train_loss_slope_last10",
        "learning_decision",
    ]
    return [{field: run.get(field, "") for field in fields} for run in rows]


def runtime_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_ce = next((r for r in rows if r["run_name"] == "dual_head_w15_ce"), None)
    base_time = safe_float(base_ce.get("epoch_time_mean")) if base_ce else float("nan")
    out = []
    for run in rows:
        epoch_time = safe_float(run.get("epoch_time_mean"))
        out.append(
            {
                "run_name": run["run_name"],
                "run_type": run["run_type"],
                "use_supcon": run["use_supcon"],
                "use_class_weight": run["use_class_weight"],
                "epoch_time_mean": epoch_time,
                "epoch_time_total": run.get("epoch_time_total"),
                "train_epoch_time_mean": run.get("train_epoch_time_mean"),
                "memory_reserved_mb_max": run.get("memory_reserved_mb_max"),
                "overhead_vs_dual_head_w15_ce": epoch_time - base_time if finite(epoch_time) and finite(base_time) else float("nan"),
                "overhead_ratio_vs_dual_head_w15_ce": epoch_time / base_time if finite(epoch_time) and finite(base_time) and base_time > 0 else float("nan"),
            }
        )
    return out


def seed_stability(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    repeats = [r for r in rows if r["run_name"] in {"repeat_v1_seed2", "repeat_v1_seed3"}]
    out = [
        {
            "run": "v1_best_original",
            "seed": "",
            "test_macro_f1": V1_BEST["macro_f1"],
            "test_accuracy": V1_BEST["accuracy"],
            "fallback_macro_f1": V1_BEST["fallback_macro_f1"],
            "detected_macro_f1": "",
            "hard_F1": V1_BEST["hard_f1"],
            "best_epoch": "",
            "early_stop_epoch": "",
            "pred_max_ratio": "",
        }
    ]
    macros = []
    for run in repeats:
        macros.append(safe_float(run["test_macro_f1"]))
        out.append(
            {
                "run": run["run_name"],
                "seed": run.get("seed", ""),
                "test_macro_f1": run.get("test_macro_f1"),
                "test_accuracy": run.get("test_accuracy"),
                "fallback_macro_f1": run.get("fallback_macro_f1"),
                "detected_macro_f1": run.get("detected_macro_f1"),
                "hard_F1": run.get("hard_F1"),
                "best_epoch": run.get("best_epoch"),
                "early_stop_epoch": run.get("early_stop_epoch"),
                "pred_max_ratio": run.get("pred_max_ratio"),
            }
        )
    valid = [m for m in macros if math.isfinite(m)]
    if len(valid) < 2:
        decision = "V1_REPEAT_FAILED_OR_INVALID"
    else:
        mean = sum(valid) / len(valid)
        std = float(pd.Series(valid).std(ddof=0))
        delta = V1_BEST["macro_f1"] - mean
        if all(0.628 <= v <= 0.636 for v in valid):
            decision = "V1_STABLE_ANCHOR"
        elif delta > 0.012:
            decision = "V1_POSSIBLE_SEED_OUTLIER"
        elif std > 0.006:
            decision = "V1_HIGH_VARIANCE_BUT_VALID"
        else:
            decision = "V1_HIGH_VARIANCE_BUT_VALID"
    stats = {
        "seed_repeat_count": len(valid),
        "repeat_macro_mean": sum(valid) / len(valid) if valid else float("nan"),
        "repeat_macro_std": float(pd.Series(valid).std(ddof=0)) if valid else float("nan"),
        "repeat_macro_min": min(valid) if valid else float("nan"),
        "repeat_macro_max": max(valid) if valid else float("nan"),
        "delta_original_v1_best_vs_repeat_mean": V1_BEST["macro_f1"] - (sum(valid) / len(valid)) if valid else float("nan"),
        "stability_band": f"{min(valid):.6f}-{max(valid):.6f}" if valid else "",
        "seed_stability_decision": decision,
    }
    out.append({"run": "repeat_summary", **stats})
    return out, stats


def dual_head_ablation(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
    by_name = {r["run_name"]: r for r in rows}
    comparisons = []

    def add(name: str, a: str, b_label: str, b_metrics: Dict[str, float]) -> None:
        run = by_name.get(a)
        if not run:
            comparisons.append({"comparison": name, "status": "missing_run"})
            return
        comparisons.append(
            {
                "comparison": name,
                "candidate": a,
                "baseline": b_label,
                "delta_macro_f1": safe_float(run.get("test_macro_f1")) - b_metrics.get("macro_f1", float("nan")),
                "delta_accuracy": safe_float(run.get("test_accuracy")) - b_metrics.get("accuracy", float("nan")),
                "delta_fallback_macro_f1": safe_float(run.get("fallback_macro_f1")) - b_metrics.get("fallback_macro_f1", float("nan")),
                "delta_hard_F1": safe_float(run.get("hard_F1")) - b_metrics.get("hard_f1", float("nan")),
                "candidate_macro_f1": run.get("test_macro_f1"),
                "candidate_fallback_macro_f1": run.get("fallback_macro_f1"),
            }
        )

    add("dual_head_w15_ce_vs_v1_best", "dual_head_w15_ce", "v1_best", V1_BEST)
    if "dual_head_w15_supcon_l002" in by_name and "dual_head_w15_ce" in by_name:
        ce = by_name["dual_head_w15_ce"]
        add(
            "dual_head_supcon_vs_dual_head_ce",
            "dual_head_w15_supcon_l002",
            "dual_head_w15_ce",
            {
                "macro_f1": safe_float(ce["test_macro_f1"]),
                "accuracy": safe_float(ce["test_accuracy"]),
                "fallback_macro_f1": safe_float(ce["fallback_macro_f1"]),
                "hard_f1": safe_float(ce["hard_F1"]),
            },
        )
    if "dual_head_w15_class_weighted_ce" in by_name and "dual_head_w15_ce" in by_name:
        ce = by_name["dual_head_w15_ce"]
        add(
            "dual_head_class_weight_vs_dual_head_ce",
            "dual_head_w15_class_weighted_ce",
            "dual_head_w15_ce",
            {
                "macro_f1": safe_float(ce["test_macro_f1"]),
                "accuracy": safe_float(ce["test_accuracy"]),
                "fallback_macro_f1": safe_float(ce["fallback_macro_f1"]),
                "hard_f1": safe_float(ce["hard_F1"]),
            },
        )
    if "dual_head_ce" in by_name:
        add(
            "dual_head_ce_vs_D15",
            "dual_head_ce",
            "D15",
            {"macro_f1": D15["macro_f1"], "accuracy": D15["accuracy"], "fallback_macro_f1": float("nan"), "hard_f1": float("nan")},
        )
    dual_best = max(
        [r for r in rows if r.get("dual_head")],
        key=lambda r: safe_float(r.get("test_macro_f1"), -999.0),
        default=None,
    )
    if not dual_best:
        decision = "DUAL_HEAD_INCONCLUSIVE"
    elif safe_float(dual_best["test_macro_f1"]) > V1_BEST["macro_f1"]:
        decision = "DUAL_HEAD_ADDS_GAIN"
    elif safe_float(dual_best["fallback_macro_f1"]) > V1_BEST["fallback_macro_f1"] and safe_float(dual_best["test_macro_f1"]) <= V1_BEST["macro_f1"]:
        decision = "DUAL_HEAD_IMPROVES_FALLBACK_ONLY"
    elif safe_float(dual_best["detected_macro_f1"]) < 0.62:
        decision = "DUAL_HEAD_HURTS_DETECTED"
    else:
        decision = "DUAL_HEAD_NO_GAIN"
    for row in comparisons:
        row["dual_head_decision"] = decision
    return comparisons, decision


def fallback_branch_decision(rows: List[Dict[str, Any]]) -> str:
    dual_rows = [r for r in rows if r.get("dual_head")]
    if not dual_rows:
        return "FALLBACK_BRANCH_NO_GAIN"
    best_fallback = max(dual_rows, key=lambda r: safe_float(r.get("fallback_macro_f1"), -999))
    best_macro = max(dual_rows, key=lambda r: safe_float(r.get("test_macro_f1"), -999))
    if safe_float(best_fallback["fallback_macro_f1"]) > V2_BEST["fallback_macro_f1"] and safe_float(best_macro["test_macro_f1"]) >= V1_BEST["macro_f1"] - 0.003:
        return "FALLBACK_BRANCH_CONFIRMED"
    if safe_float(best_fallback["fallback_macro_f1"]) > V1_BEST["fallback_macro_f1"] and safe_float(best_macro["test_macro_f1"]) < V1_BEST["macro_f1"]:
        return "FALLBACK_BRANCH_IMPROVES_FALLBACK_ONLY"
    if safe_float(best_fallback["fallback_macro_f1"]) > V1_BEST["fallback_macro_f1"] and safe_float(best_macro["detected_macro_f1"]) < 0.62:
        return "FALLBACK_BRANCH_TRADEOFF_TOO_HIGH"
    return "FALLBACK_BRANCH_NO_GAIN"


def risk_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        risks = []
        for key in ["missing_artifacts", "test_contract_status", "head_contract_status", "collapse_risk", "checker_decision", "learning_decision"]:
            value = str(row.get(key, ""))
            if value and value not in {"TEST_BEST_CONTRACT_OK", "HEAD_ROUTING_VALID", "SINGLE_HEAD_CONTROL", "NO_COLLAPSE", "D16_V3_RUN_PASS", "PLATEAU_EARLY_STOP_OK"}:
                risks.append(f"{key}={value}")
        if risks:
            out.append({"run_name": row["run_name"], "risk_count": len(risks), "risks": "; ".join(risks)})
    return out


def final_decision(rows: List[Dict[str, Any]], seed_stats: Dict[str, Any], dual_decision: str) -> str:
    if any(row["run_decision"] == "D16_V3_BEATS_V1_BEST" or row["run_decision"] == "STRONG_NEW_BEST" for row in rows):
        return "D16_V3_DUAL_HEAD_NEW_BEST"
    if seed_stats.get("seed_stability_decision") == "V1_POSSIBLE_SEED_OUTLIER":
        return "D16_V3_SEED_VARIANCE_HIGH_REPEAT_MORE"
    if dual_decision == "DUAL_HEAD_IMPROVES_FALLBACK_ONLY":
        return "D16_V3_DUAL_HEAD_IMPROVES_FALLBACK_ONLY"
    if seed_stats.get("seed_stability_decision") in {"V1_STABLE_ANCHOR", "V1_HIGH_VARIANCE_BUT_VALID"}:
        return "D16_V3_NO_GAIN_OVER_V1_USE_V1_ANCHOR"
    if any(row["run_decision"] == "D16_V3_INVALID_OR_INCOMPLETE" for row in rows):
        return "D16_V3_INVALID_NEEDS_RERUN"
    return "D16_V3_NEEDS_DEEPER_FALLBACK_ARCHITECTURE"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def md_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], max_rows: int = 20) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(
    path: Path,
    rows: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    per_class_rows_out: List[Dict[str, Any]],
    seed_stats: Dict[str, Any],
    dual_rows: List[Dict[str, Any]],
    dual_decision: str,
    fallback_decision: str,
    final: str,
) -> None:
    best = ranked[0] if ranked else {}
    top3 = ranked[:3]
    report: List[str] = [
        "# D16 v3 Deep Analysis Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## 1. Context",
        "",
        "D16 v3 evaluates whether a shared D16 pixel-graph encoder with separate detected/fallback classifier heads improves the D16 fallback bottleneck after v2 failed to improve over v1.",
        "",
        "## 2. Baselines",
        "",
        f"- D15 baseline: accuracy `{D15['accuracy']:.6f}`, macro-F1 `{D15['macro_f1']:.6f}`, weighted-F1 `{D15['weighted_f1']:.6f}`.",
        f"- D16 v1 best `{V1_BEST['name']}`: accuracy `{V1_BEST['accuracy']:.6f}`, macro-F1 `{V1_BEST['macro_f1']:.6f}`, fallback macro-F1 `{V1_BEST['fallback_macro_f1']:.6f}`, hard_F1 `{V1_BEST['hard_f1']:.6f}`.",
        f"- D16 v2 best `{V2_BEST['name']}`: accuracy `{V2_BEST['accuracy']:.6f}`, macro-F1 `{V2_BEST['macro_f1']:.6f}`, fallback macro-F1 `{V2_BEST['fallback_macro_f1']:.6f}`, hard_F1 `{V2_BEST['hard_f1']:.6f}`.",
        "- Prior v2 decision: `D16_V2_NO_GAIN_NEEDS_ARCHITECTURE_CHANGE`.",
        "",
        "## 3. Run Validity",
        "",
    ]
    valid_count = sum(1 for row in rows if row.get("test_contract_status") == "TEST_BEST_CONTRACT_OK")
    report.append(f"- Runs included: `{len(rows)}`.")
    report.append(f"- Runs with best-checkpoint test contract: `{valid_count}`.")
    incomplete = [row for row in rows if row.get("missing_artifacts") or row.get("test_contract_status") != "TEST_BEST_CONTRACT_OK"]
    report.append(f"- Runs with missing/contract warnings: `{len(incomplete)}`.")
    report.extend(["", "## 4. Main Ranking", ""])
    report.extend(md_table(ranked, ["run_name", "run_type", "test_macro_f1", "test_accuracy", "fallback_macro_f1", "detected_macro_f1", "hard_F1", "run_decision", "collapse_risk"]))
    report.extend(["", "## 5. Seed Stability", ""])
    report.append(f"- Decision: `{seed_stats.get('seed_stability_decision')}`.")
    report.append(f"- Repeat macro-F1 mean/std: `{fmt(seed_stats.get('repeat_macro_mean'))}` / `{fmt(seed_stats.get('repeat_macro_std'))}`.")
    report.append(f"- Stability band: `{seed_stats.get('stability_band')}`.")
    report.append(f"- Original v1 best minus repeat mean: `{fmt(seed_stats.get('delta_original_v1_best_vs_repeat_mean'))}`.")
    if seed_stats.get("seed_stability_decision") == "V1_POSSIBLE_SEED_OUTLIER":
        report.append("- Interpretation: the original v1 best should be treated as a possible upper outlier until more repeats are available.")
    else:
        report.append("- Interpretation: the repeats support using v1 as a practical anchor, with the observed variance noted.")
    report.extend(["", "## 6. Dual-Head Ablation", ""])
    report.append(f"- Decision: `{dual_decision}`.")
    report.extend(md_table(dual_rows, ["comparison", "delta_macro_f1", "delta_accuracy", "delta_fallback_macro_f1", "delta_hard_F1"], max_rows=10))
    report.extend(["", "## 7. Head Routing Analysis", ""])
    head_rows = head_routing_rows(rows)
    report.extend(md_table(head_rows, ["run_name", "detected_head_count", "fallback_head_count", "wrong_route_count", "head_contract_status"], max_rows=10))
    wrong = [row for row in head_rows if str(row.get("head_contract_status", "")).startswith("HEAD_ROUTING_FAIL")]
    report.append(f"- Head routing failures: `{len(wrong)}`.")
    report.extend(["", "## 8. Detected vs Fallback Analysis", ""])
    report.append(f"- Decision: `{fallback_decision}`.")
    report.extend(md_table(rows, ["run_name", "detected_accuracy", "detected_macro_f1", "fallback_accuracy", "fallback_macro_f1", "detected_minus_fallback_macro_gap"], max_rows=10))
    report.extend(["", "## 9. Per-Class Analysis", ""])
    report.extend(md_table(per_class_rows_out, ["class_name", "D15_F1", "v1_best_F1", "v2_best_F1", "best_v3_for_class", "best_v3_F1", "delta_best_v3_vs_v1"], max_rows=10))
    hard_best = max(rows, key=lambda row: safe_float(row.get("hard_F1"), -999), default={})
    report.append(f"- Best hard_F1 v3 run: `{hard_best.get('run_name', '')}` with `{fmt(hard_best.get('hard_F1'))}`.")
    report.extend(["", "## 10. Confusion Analysis", ""])
    report.append("Focus confusion pairs are recorded in `d16_v3_confusion_compare.csv`.")
    report.append("- The report does not infer unavailable detected/fallback-specific confusion matrices when those files are absent.")
    report.extend(["", "## 11. Prediction Distribution", ""])
    dist = prediction_distribution_rows(rows)
    report.extend(md_table(dist, ["run_name", "scope", "predicted_classes", "pred_max_class", "pred_max_ratio", "collapse_risk"], max_rows=18))
    report.extend(["", "## 12. Learning Dynamics", ""])
    report.extend(md_table(learning_rows(rows), ["run_name", "best_epoch", "early_stop_epoch", "val_macro_f1_best", "val_macro_f1_final", "val_slope_last10", "learning_decision"], max_rows=10))
    report.extend(["", "## 13. Runtime Practicality", ""])
    report.extend(md_table(runtime_rows(rows), ["run_name", "epoch_time_mean", "epoch_time_total", "memory_reserved_mb_max", "overhead_ratio_vs_dual_head_w15_ce"], max_rows=10))
    report.append("- Graph cache is not recommended from these artifacts alone; the current runs completed with online graph construction and disabled graph cache.")
    report.extend(["", "## 14. Final Recommendation", ""])
    report.append(f"- Best v3 run by best-checkpoint macro-F1: `{best.get('run_name', '')}` with macro-F1 `{fmt(best.get('test_macro_f1'))}` and accuracy `{fmt(best.get('test_accuracy'))}`.")
    report.append(f"- Top 3 runs: `{', '.join(str(row.get('run_name')) for row in top3)}`.")
    beats_v1 = safe_float(best.get("test_macro_f1")) > V1_BEST["macro_f1"] and safe_float(best.get("test_accuracy")) >= 0.634
    beats_d15 = safe_float(best.get("test_macro_f1")) > D15["macro_f1"]
    report.append(f"- v3 beats v1 best: `{beats_v1}`.")
    report.append(f"- v3 beats D15 macro-F1: `{beats_d15}`.")
    if final == "D16_V3_NO_GAIN_OVER_V1_USE_V1_ANCHOR":
        report.append("- Recommended next experiment: use v1 `fallback_weighted_ce` as the D16 candidate anchor and run a small seed ensemble or validation repeat around that setting.")
        report.append("- Secondary option: test a deeper fallback branch/readout only if a new architecture pass is opened.")
    elif final == "D16_V3_DUAL_HEAD_IMPROVES_FALLBACK_ONLY":
        report.append("- Recommended next experiment: deeper fallback branch/readout, because dual-head routing alone improves fallback more than overall quality.")
    elif final == "D16_V3_SEED_VARIANCE_HIGH_REPEAT_MORE":
        report.append("- Recommended next experiment: repeat the v1 and best v3 settings with additional seeds before selecting an anchor.")
    elif final == "D16_V3_DUAL_HEAD_NEW_BEST":
        report.append("- Recommended next experiment: graph-cache/runtime benchmark plus seed repeats of the best v3 run.")
    else:
        report.append("- Recommended next experiment: architecture change beyond the current dual-head classifier, or move to ensemble if no new architecture work is planned.")
    report.extend(["", "## 15. Final Decision", "", f"`{final}`", ""])
    text = "\n".join(report)
    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise RuntimeError(f"Forbidden token found in report: {token}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v3_deep_analysis")
    args = parser.parse_args()
    if len(args.runs) != len(args.names):
        raise ValueError("--runs and --names must have same length")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [analyze_run(Path(path), name) for path, name in zip(args.runs, args.names)]
    public_rows = [{k: v for k, v in row.items() if k != "_dfs"} for row in rows]
    ranked = sorted(
        public_rows,
        key=lambda row: (
            safe_float(row.get("test_macro_f1"), -999),
            safe_float(row.get("test_accuracy"), -999),
            safe_float(row.get("fallback_macro_f1"), -999),
            safe_float(row.get("detected_macro_f1"), -999),
            safe_float(row.get("hard_F1"), -999),
            -safe_float(row.get("detected_minus_fallback_macro_gap"), 999),
            -int(row.get("collapse_risk_order", 9)),
        ),
        reverse=True,
    )
    summary_fields = [
        "run_name",
        "run_type",
        "seed",
        "dual_head",
        "fallback_weight",
        "use_supcon",
        "use_class_weight",
        "best_epoch",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "fallback_macro_f1",
        "detected_macro_f1",
        "hard_F1",
        "delta_vs_D15_macro_f1",
        "delta_vs_v1_best_macro_f1",
        "delta_vs_v2_best_macro_f1",
        "checker_decision",
        "collapse_risk",
        "head_contract_status",
        "run_decision",
        "missing_artifacts",
        "test_contract_status",
    ]
    write_csv(output_dir / "d16_v3_deep_summary.csv", public_rows, summary_fields)
    write_csv(output_dir / "d16_v3_ranked_summary.csv", ranked, summary_fields)
    per_rows = per_class_compare(rows)
    write_csv(output_dir / "d16_v3_per_class_compare.csv", per_rows)
    write_csv(output_dir / "d16_v3_detected_fallback_compare.csv", [
        {
            "run_name": row["run_name"],
            "detected_accuracy": row["detected_accuracy"],
            "detected_macro_f1": row["detected_macro_f1"],
            "fallback_accuracy": row["fallback_accuracy"],
            "fallback_macro_f1": row["fallback_macro_f1"],
            "detected_minus_fallback_acc_gap": row["detected_minus_fallback_acc_gap"],
            "detected_minus_fallback_macro_gap": row["detected_minus_fallback_macro_gap"],
        }
        for row in public_rows
    ])
    write_csv(output_dir / "d16_v3_head_routing_compare.csv", head_routing_rows(public_rows))
    conf_rows: List[Dict[str, Any]] = []
    for row in rows:
        conf_rows.extend(confusion_focus(row["_dfs"]["confusion"], row["run_name"]))
    write_csv(output_dir / "d16_v3_confusion_compare.csv", conf_rows)
    write_csv(output_dir / "d16_v3_prediction_distribution.csv", prediction_distribution_rows(rows))
    write_csv(output_dir / "d16_v3_learning_dynamics.csv", learning_rows(public_rows))
    write_csv(output_dir / "d16_v3_runtime_compare.csv", runtime_rows(public_rows))
    seed_rows, seed_stats = seed_stability(public_rows)
    write_csv(output_dir / "d16_v3_seed_stability.csv", seed_rows)
    dual_rows, dual_decision = dual_head_ablation(public_rows)
    write_csv(output_dir / "d16_v3_dual_head_ablation.csv", dual_rows)
    write_csv(output_dir / "d16_v3_risk_cases.csv", risk_rows(public_rows))
    fallback_decision = fallback_branch_decision(public_rows)
    final = final_decision(public_rows, seed_stats, dual_decision)
    write_report(
        output_dir / "D16_V3_DEEP_ANALYSIS_REPORT.md",
        rows,
        ranked,
        per_rows,
        seed_stats,
        dual_rows,
        dual_decision,
        fallback_decision,
        final,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runs": len(rows),
                "best_run": ranked[0]["run_name"] if ranked else "",
                "best_macro_f1": ranked[0]["test_macro_f1"] if ranked else "",
                "seed_stability_decision": seed_stats.get("seed_stability_decision"),
                "dual_head_decision": dual_decision,
                "fallback_branch_decision": fallback_decision,
                "final_decision": final,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
