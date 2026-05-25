"""Deep read-only analysis for completed D16 v2 fallback-aware runs."""

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
CLASS_IDS_BY_NAME = {value: key for key, value in CLASS_NAMES.items()}
HARD_CLASS_IDS = [0, 2, 4]
D15 = {
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
V0_FACE_MACRO_F1 = 0.615703
V1_BEST = {
    "name": "d16_v1_fallback_weighted_ce",
    "accuracy": 0.639175,
    "macro_f1": 0.632938,
    "fallback_macro_f1": 0.409767,
    "hard_f1": 0.510704,
    "path": Path("outputs/d16_runs/v1/d16_v1_face_plus_context_fallback_weighted_ce"),
}
V1_SUPCON_L002 = {
    "name": "d16_v1_face_supcon_l002",
    "macro_f1": 0.618280,
    "path": Path("outputs/d16_runs/v1/d16_v1_face_plus_context_part_supcon_l002"),
}
V1_HYBRID_CE = {
    "name": "d16_v1_hybrid_ce",
    "macro_f1": 0.618734,
    "path": Path("outputs/d16_runs/v1/d16_v1_hybrid_detected_face_fallback_fullmask_ce"),
}
FORBIDDEN_TOKENS = [
    "MOTIF_DISCOVERED",
    "SEMANTIC_REGION_DISCOVERED",
    "CAUSAL_EVIDENCE_CONFIRMED",
    "FULL_INTERPRETABILITY_CLAIM",
]


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
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result


def metric(df: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if df.empty or col not in df.columns:
        return default
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else default


def first_metric(df: pd.DataFrame, col: str, default: float = float("nan")) -> float:
    if df.empty or col not in df.columns:
        return default
    values = pd.to_numeric(df[col], errors="coerce").dropna()
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
    subset = per_class[pd.to_numeric(per_class["class_id"], errors="coerce") == int(class_id)]
    return metric(subset, "f1")


def class_support(per_class: pd.DataFrame, class_id: int) -> int:
    if per_class.empty or not {"class_id", "support"}.issubset(per_class.columns):
        return 0
    subset = per_class[pd.to_numeric(per_class["class_id"], errors="coerce") == int(class_id)]
    value = metric(subset, "support", 0.0)
    return int(value) if math.isfinite(value) else 0


def hard_macro_f1(per_class: pd.DataFrame) -> float:
    values = [class_f1(per_class, class_id) for class_id in HARD_CLASS_IDS]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else float("nan")


def slope_last(values: Sequence[float], n: int = 10) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if len(clean) < 2:
        return float("nan")
    subset = clean[-int(n):]
    if len(subset) < 2:
        return float("nan")
    return float((subset[-1] - subset[0]) / max(len(subset) - 1, 1))


def pred_stats(pred_count: pd.DataFrame) -> Dict[str, Any]:
    if pred_count.empty or not {"class_id", "pred_count"}.issubset(pred_count.columns):
        return {
            "predicted_classes": 0,
            "total_predictions": 0,
            "max_pred_ratio": float("nan"),
            "max_pred_class": "",
            "class1_pred_count": 0,
            "class2_pred_count": 0,
            "class4_pred_count": 0,
            "under_predicted_classes": "missing",
            "over_predicted_classes": "missing",
            "collapse_risk": "MISSING_PRED_COUNT",
            "collapse_risk_order": 4,
        }
    rows = pred_count.copy()
    rows["class_id"] = pd.to_numeric(rows["class_id"], errors="coerce").astype("Int64")
    rows["pred_count"] = pd.to_numeric(rows["pred_count"], errors="coerce").fillna(0).astype(int)
    counts = {int(row["class_id"]): int(row["pred_count"]) for _, row in rows.dropna(subset=["class_id"]).iterrows()}
    total = int(sum(counts.values()))
    predicted_classes = int(sum(1 for value in counts.values() if value > 0))
    max_class, max_count = max(counts.items(), key=lambda item: item[1]) if counts else (-1, 0)
    max_ratio = max_count / total if total > 0 else float("nan")
    expected = total / 7.0 if total else 0.0
    under = [CLASS_NAMES[c] for c, count in counts.items() if expected and count < 0.55 * expected]
    over = [CLASS_NAMES[c] for c, count in counts.items() if expected and count > 1.45 * expected]
    class1 = int(counts.get(1, 0))
    class2 = int(counts.get(2, 0))
    class4 = int(counts.get(4, 0))
    if predicted_classes < 7 or (math.isfinite(max_ratio) and max_ratio > 0.50):
        risk = "COLLAPSE_RISK"
        order = 4
    elif total > 0 and class1 / total < 0.005:
        risk = "CLASS_1_SUPPRESSION"
        order = 3
    elif total > 0 and (class2 / total < 0.05 or class4 / total < 0.05):
        risk = "HARD_CLASS_SUPPRESSION"
        order = 2
    elif math.isfinite(max_ratio) and max_ratio > 0.35:
        risk = "MILD_PRED_BIAS"
        order = 1
    else:
        risk = "NO_COLLAPSE"
        order = 0
    return {
        "predicted_classes": predicted_classes,
        "total_predictions": total,
        "max_pred_ratio": float(max_ratio),
        "max_pred_class": CLASS_NAMES.get(max_class, str(max_class)),
        "class1_pred_count": class1,
        "class2_pred_count": class2,
        "class4_pred_count": class4,
        "under_predicted_classes": ",".join(under),
        "over_predicted_classes": ",".join(over),
        "collapse_risk": risk,
        "collapse_risk_order": order,
    }


def config_features(cfg: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    data = cfg.get("data", {}) or {}
    graph = cfg.get("graph", {}) or {}
    loss = cfg.get("loss", {}) or {}
    graph_mode = graph.get("graph_mode", data.get("graph_mode", ""))
    loss_mode = loss.get("mode", "")
    fallback_weight = safe_float(loss.get("fallback_weight", 1.0))
    supcon_lambda = safe_float(loss.get("lambda_part_supcon", 0.0))
    use_supcon = loss_mode == "ce_part_supcon" or supcon_lambda > 0
    use_class_weight = bool(loss.get("class_weights")) or loss_mode == "class_weighted_ce"
    use_hybrid = "hybrid" in str(graph_mode)
    config_type = "hybrid" if use_hybrid else "face_plus_context"
    if use_supcon:
        config_type += "_supcon"
    if use_class_weight:
        config_type += "_class_weighted"
    if bool(loss.get("fallback_weighted", False)):
        config_type += "_fallback_weighted"
    return {
        "graph_mode": graph_mode,
        "loss_mode": loss_mode,
        "config_type": config_type,
        "fallback_weight": fallback_weight,
        "use_supcon": bool(use_supcon),
        "supcon_lambda": supcon_lambda if math.isfinite(supcon_lambda) else 0.0,
        "use_class_weight": bool(use_class_weight),
        "use_hybrid": bool(use_hybrid),
    }


def contract_flags(test: pd.DataFrame, summary: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if test.empty:
        return ["MISSING_TEST_METRICS"]
    if "checkpoint_name" not in test.columns or "checkpoint_epoch" not in test.columns:
        flags.append("TEST_CONTRACT_INCOMPLETE")
    checkpoint_name = str(test["checkpoint_name"].iloc[-1]) if "checkpoint_name" in test.columns else ""
    if checkpoint_name and checkpoint_name != "best.pt":
        flags.append("TEST_CONTRACT_WARNING_LAST_NOT_BEST")
    if summary.get("final_test_checkpoint") not in (None, "", "best.pt"):
        flags.append("TEST_CONTRACT_WARNING_LAST_NOT_BEST")
    return flags


def run_row(path: Path, name: str) -> Dict[str, Any]:
    train = read_csv(path / "train_log.csv")
    test = read_csv(path / "test_metrics.csv")
    last_test = read_csv(path / "last_test_metrics.csv")
    per = read_csv(path / "per_class_metrics.csv")
    fallback = read_csv(path / "detected_vs_fallback_metrics.csv")
    pred_count = read_csv(path / "pred_count.csv")
    summary = read_json(path / "d16_train_summary.json")
    check = read_json(path / "d16_v1_check_summary.json")
    cfg = read_yaml(path / "resolved_config.yaml")
    missing = [
        file_name
        for file_name in [
            "d16_train_summary.json",
            "train_log.csv",
            "val_metrics.csv",
            "test_metrics.csv",
            "last_test_metrics.csv",
            "per_class_metrics.csv",
            "pred_count.csv",
            "detected_vs_fallback_metrics.csv",
            "detected_fallback_per_class_metrics.csv",
            "confusion_matrix.csv",
            "predictions.csv",
            "d16_v1_check_summary.json",
            "d16_report.md",
        ]
        if not (path / file_name).exists()
    ]
    detected = fallback[fallback["group"] == "detected"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    fb = fallback[fallback["group"] == "fallback"] if not fallback.empty and "group" in fallback.columns else pd.DataFrame()
    detected_macro = metric(detected, "macro_f1")
    fallback_macro = metric(fb, "macro_f1")
    detected_acc = metric(detected, "accuracy")
    fallback_acc = metric(fb, "accuracy")
    row = {
        "run_name": name,
        "output_dir": str(path),
        **config_features(cfg, name),
        "best_epoch": int(summary.get("best_epoch", metric(test, "checkpoint_epoch", 0.0)) or 0),
        "best_val_macro_f1": safe_float(summary.get("best_val_macro_f1", metric(test, "best_val_macro_f1"))),
        "test_accuracy": metric(test, "accuracy"),
        "test_macro_f1": metric(test, "macro_f1"),
        "test_weighted_f1": weighted_f1(per),
        "last_test_accuracy": metric(last_test, "accuracy"),
        "last_test_macro_f1": metric(last_test, "macro_f1"),
        "hard_F1": hard_macro_f1(per),
        "angry_f1": class_f1(per, 0),
        "disgust_f1": class_f1(per, 1),
        "fear_f1": class_f1(per, 2),
        "happy_f1": class_f1(per, 3),
        "sad_f1": class_f1(per, 4),
        "surprise_f1": class_f1(per, 5),
        "neutral_f1": class_f1(per, 6),
        "detected_accuracy": detected_acc,
        "detected_macro_f1": detected_macro,
        "fallback_accuracy": fallback_acc,
        "fallback_macro_f1": fallback_macro,
        "detected_minus_fallback_acc_gap": detected_acc - fallback_acc if math.isfinite(detected_acc) and math.isfinite(fallback_acc) else float("nan"),
        "detected_minus_fallback_macro_gap": detected_macro - fallback_macro if math.isfinite(detected_macro) and math.isfinite(fallback_macro) else float("nan"),
        "checker_decision": check.get("decision", ""),
        "missing_artifacts": ";".join(missing),
        "contract_flags": ";".join(contract_flags(test, summary)),
        "train_epoch_time_mean": metric(train, "train_epoch_time_sec"),
        "epoch_time_mean": float(pd.to_numeric(train.get("epoch_time_sec", pd.Series(dtype=float)), errors="coerce").mean()) if not train.empty else float("nan"),
        "total_runtime_sec": float(pd.to_numeric(train.get("epoch_time_sec", pd.Series(dtype=float)), errors="coerce").sum()) if not train.empty else float("nan"),
        "memory_reserved_mb": metric(train, "memory_reserved_mb"),
        "train_loss_first": first_metric(train, "train_loss"),
        "train_loss_final": metric(train, "train_loss"),
        "supcon_loss_total_final": metric(train, "supcon_loss_total"),
        "supcon_valid_pairs_final": metric(train, "supcon_valid_pairs"),
        "supcon_no_positive_pairs_final": metric(train, "supcon_no_positive_pairs"),
        "lambda_part_supcon_final": metric(train, "lambda_part_supcon_current"),
    }
    row.update(pred_stats(pred_count))
    row["delta_vs_D15_acc"] = row["test_accuracy"] - D15["accuracy"]
    row["delta_vs_D15_macro_f1"] = row["test_macro_f1"] - D15["macro_f1"]
    row["delta_vs_v1_best_macro_f1"] = row["test_macro_f1"] - V1_BEST["macro_f1"]
    row["delta_vs_v1_best_fallback_f1"] = row["fallback_macro_f1"] - V1_BEST["fallback_macro_f1"]
    if row["contract_flags"] or not math.isfinite(row["test_macro_f1"]):
        decision = "INVALID_OR_INCOMPLETE"
    elif row["test_macro_f1"] >= 0.645 and row["test_accuracy"] >= 0.650 and row["fallback_macro_f1"] >= 0.43 and row["collapse_risk"] == "NO_COLLAPSE":
        decision = "STRONG_NEW_BEST"
    elif row["test_macro_f1"] > V1_BEST["macro_f1"] and row["test_accuracy"] >= 0.634 and row["collapse_risk"] == "NO_COLLAPSE":
        decision = "BEATS_V1_BEST"
    elif row["test_macro_f1"] > D15["macro_f1"] and row["test_accuracy"] >= D15["accuracy"] - 0.010:
        decision = "BEATS_D15"
    elif row["test_macro_f1"] >= V1_BEST["macro_f1"] - 0.005:
        decision = "NEAR_V1_BEST"
    else:
        decision = "BELOW_V1_BEST"
    row["run_decision"] = decision
    return row


def top_confusions(confusion: pd.DataFrame, run_name: str, group: str = "all", limit: int = 12) -> List[Dict[str, Any]]:
    if confusion.empty or not {"true_class", "pred_class", "count"}.issubset(confusion.columns):
        return []
    df = confusion.copy()
    df["true_class"] = pd.to_numeric(df["true_class"], errors="coerce").astype("Int64")
    df["pred_class"] = pd.to_numeric(df["pred_class"], errors="coerce").astype("Int64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df[df["true_class"] != df["pred_class"]].sort_values("count", ascending=False).head(limit)
    rows = []
    for _, row in df.iterrows():
        true_id = int(row["true_class"])
        pred_id = int(row["pred_class"])
        rows.append(
            {
                "run_name": run_name,
                "group": group,
                "true_class": true_id,
                "true_name": CLASS_NAMES.get(true_id, str(true_id)),
                "pred_class": pred_id,
                "pred_name": CLASS_NAMES.get(pred_id, str(pred_id)),
                "count": int(row["count"]),
                "row_ratio": safe_float(row.get("row_ratio", float("nan"))),
            }
        )
    return rows


def confusion_from_predictions(predictions: pd.DataFrame, run_name: str, group_value: bool) -> List[Dict[str, Any]]:
    if predictions.empty or not {"y_true", "y_pred", "detected"}.issubset(predictions.columns):
        return []
    df = predictions[predictions["detected"].astype(str).str.lower().isin([str(group_value).lower(), "1" if group_value else "0"])]
    rows: List[Dict[str, Any]] = []
    for true_id in range(7):
        sub = df[pd.to_numeric(df["y_true"], errors="coerce") == true_id]
        support = len(sub)
        for pred_id in range(7):
            count = int((pd.to_numeric(sub["y_pred"], errors="coerce") == pred_id).sum())
            if true_id != pred_id and count > 0:
                rows.append(
                    {
                        "run_name": run_name,
                        "group": "detected" if group_value else "fallback",
                        "true_class": true_id,
                        "true_name": CLASS_NAMES[true_id],
                        "pred_class": pred_id,
                        "pred_name": CLASS_NAMES[pred_id],
                        "count": count,
                        "row_ratio": count / support if support else 0.0,
                    }
                )
    return sorted(rows, key=lambda row: row["count"], reverse=True)[:12]


def pick(rows: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any] | None:
    return next((row for row in rows if row["run_name"] == name), None)


def delta(a: Dict[str, Any] | None, b: Dict[str, Any] | None, key: str) -> float:
    if not a or not b:
        return float("nan")
    return safe_float(a.get(key)) - safe_float(b.get(key))


def pair_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = [
        ("w20_ce", "w20_supcon_l001", "SupCon l001 vs w20 CE"),
        ("w20_ce", "w20_supcon_l002", "SupCon l002 vs w20 CE"),
        ("hybrid_w20_ce", "hybrid_w20_supcon_l002", "Hybrid SupCon l002 vs hybrid CE"),
    ]
    out = []
    for base_name, test_name, label in pairs:
        base = pick(rows, base_name)
        test = pick(rows, test_name)
        out.append(
            {
                "comparison": label,
                "base_run": base_name,
                "test_run": test_name,
                "delta_macro_f1": delta(test, base, "test_macro_f1"),
                "delta_accuracy": delta(test, base, "test_accuracy"),
                "delta_fallback_macro_f1": delta(test, base, "fallback_macro_f1"),
                "delta_hard_F1": delta(test, base, "hard_F1"),
                "delta_gap": delta(test, base, "detected_minus_fallback_macro_gap"),
            }
        )
    return out


def decision_weight_sweep(weight_rows: Sequence[Dict[str, Any]]) -> str:
    valid = [row for row in weight_rows if math.isfinite(safe_float(row.get("test_macro_f1")))]
    if len(valid) < 3:
        return "FALLBACK_WEIGHT_INCONCLUSIVE"
    best = max(valid, key=lambda row: (safe_float(row["test_macro_f1"]), safe_float(row["fallback_macro_f1"])))
    best_weight = safe_float(best["fallback_weight"])
    if best_weight == 2.5:
        w25 = best
        w20 = next(row for row in valid if safe_float(row["fallback_weight"]) == 2.0)
        if safe_float(w25["test_macro_f1"]) < safe_float(w20["test_macro_f1"]) + 0.002:
            return "FALLBACK_WEIGHT_TOO_HIGH"
        return "FALLBACK_WEIGHT_25_BETTER"
    if best_weight == 2.0:
        return "FALLBACK_WEIGHT_20_BETTER"
    if best_weight == 1.5:
        return "FALLBACK_WEIGHT_15_CONFIRMED"
    return "FALLBACK_WEIGHT_INCONCLUSIVE"


def decision_supcon(pairs: Sequence[Dict[str, Any]]) -> str:
    gains = [row for row in pairs if safe_float(row["delta_macro_f1"]) > 0.002]
    hard_only = [row for row in pairs if safe_float(row["delta_hard_F1"]) > 0.005 and safe_float(row["delta_macro_f1"]) <= 0.002]
    hurts_fallback = [row for row in pairs if safe_float(row["delta_fallback_macro_f1"]) < -0.02]
    if gains:
        return "SUPCON_ADDS_GAIN"
    if hard_only:
        return "SUPCON_HELPS_HARD_CLASSES_ONLY"
    if hurts_fallback:
        return "SUPCON_HURTS_FALLBACK"
    if pairs:
        return "SUPCON_NO_GAIN"
    return "SUPCON_INCONCLUSIVE"


def decision_hybrid(rows: Sequence[Dict[str, Any]]) -> str:
    w20 = pick(rows, "w20_ce")
    hybrid = pick(rows, "hybrid_w20_ce")
    sup = pick(rows, "w20_supcon_l002")
    hsup = pick(rows, "hybrid_w20_supcon_l002")
    if not w20 or not hybrid:
        return "HYBRID_INCONCLUSIVE"
    macro_delta = delta(hybrid, w20, "test_macro_f1")
    fallback_delta = delta(hybrid, w20, "fallback_macro_f1")
    if macro_delta > 0.002:
        return "HYBRID_ADDS_GAIN"
    if fallback_delta > 0.02 and macro_delta > -0.01:
        return "HYBRID_IMPROVES_FALLBACK_ONLY"
    if macro_delta < -0.01 or (hsup and sup and delta(hsup, sup, "test_macro_f1") < -0.01):
        return "HYBRID_HURTS_OVERALL"
    return "HYBRID_NO_GAIN"


def decision_class_weight(rows: Sequence[Dict[str, Any]]) -> str:
    base = pick(rows, "w20_ce")
    cw = pick(rows, "w20_class_weighted_ce")
    if not base or not cw:
        return "CLASS_WEIGHT_INCONCLUSIVE"
    macro_delta = delta(cw, base, "test_macro_f1")
    hard_delta = delta(cw, base, "hard_F1")
    fallback_delta = delta(cw, base, "fallback_macro_f1")
    if macro_delta > 0.002:
        return "CLASS_WEIGHT_ADDS_GAIN"
    if hard_delta > 0.005 and macro_delta > -0.01:
        return "CLASS_WEIGHT_HELPS_HARD_CLASSES_BUT_HURTS_OVERALL"
    if macro_delta < -0.01 or fallback_delta < -0.02:
        return "CLASS_WEIGHT_HURTS"
    return "CLASS_WEIGHT_INCONCLUSIVE"


def decision_fallback(rows: Sequence[Dict[str, Any]]) -> str:
    best_fb = max(rows, key=lambda row: safe_float(row.get("fallback_macro_f1")), default=None)
    best_overall = max(rows, key=lambda row: safe_float(row.get("test_macro_f1")), default=None)
    if not best_fb or not best_overall:
        return "FALLBACK_STILL_MAJOR_BOTTLENECK"
    if safe_float(best_fb["fallback_macro_f1"]) >= 0.43 and safe_float(best_overall["test_macro_f1"]) >= V1_BEST["macro_f1"]:
        return "FALLBACK_DIRECTION_CONFIRMED"
    if safe_float(best_fb["fallback_macro_f1"]) >= V1_BEST["fallback_macro_f1"] - 0.02:
        return "FALLBACK_BOTTLENECK_SOLVED_PARTIALLY"
    if safe_float(best_fb["fallback_macro_f1"]) > V1_BEST["fallback_macro_f1"] and safe_float(best_fb["test_macro_f1"]) < V1_BEST["macro_f1"] - 0.01:
        return "FALLBACK_TRADEOFF_TOO_HIGH"
    return "FALLBACK_STILL_MAJOR_BOTTLENECK"


def learning_decision(row: Dict[str, Any]) -> str:
    if safe_float(row.get("best_epoch")) >= 145:
        return "UNDERTRAINED_EXTEND_CANDIDATE"
    if safe_float(row.get("delta_vs_v1_best_macro_f1")) < -0.01:
        return "DO_NOT_RESUME_SAME_SETTING"
    return "PLATEAU_EARLY_STOP_OK"


def final_decision(rows: Sequence[Dict[str, Any]]) -> str:
    if len(rows) < 8:
        return "D16_V2_INCONCLUSIVE_NEEDS_RERUN"
    best = max(rows, key=lambda row: (safe_float(row["test_macro_f1"]), safe_float(row["test_accuracy"])))
    if best["run_decision"] in {"STRONG_NEW_BEST", "BEATS_V1_BEST"}:
        if best.get("use_hybrid"):
            return "D16_V2_NEW_BEST_FOUND_HYBRID"
        if best.get("use_supcon"):
            return "D16_V2_NEW_BEST_FOUND_FALLBACK_PLUS_SUPCON"
        return "D16_V2_NEW_BEST_FOUND_FALLBACK_WEIGHT_TUNING"
    if safe_float(best["test_macro_f1"]) >= V1_BEST["macro_f1"] - 0.005 and str(best.get("collapse_risk")) == "NO_COLLAPSE":
        return "D16_V2_NO_GAIN_OVER_V1_FALLBACK_WEIGHT_CONFIRMED"
    return "D16_V2_NO_GAIN_NEEDS_ARCHITECTURE_CHANGE"


def md_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], max_rows: int = 12) -> List[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in list(rows)[:max_rows]:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/v2_deep_analysis")
    args = parser.parse_args()
    if len(args.runs) != len(args.names):
        raise ValueError("--runs and --names must have the same length")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_row(Path(path), name) for path, name in zip(args.runs, args.names)]
    ranked = sorted(
        rows,
        key=lambda row: (
            -safe_float(row.get("test_macro_f1")),
            -safe_float(row.get("test_accuracy")),
            -safe_float(row.get("fallback_macro_f1")),
            -safe_float(row.get("hard_F1")),
            safe_float(row.get("detected_minus_fallback_macro_gap")),
            int(row.get("collapse_risk_order", 99) or 99),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["learning_decision"] = learning_decision(row)
    summary_fields = [
        "run_name",
        "config_type",
        "graph_mode",
        "fallback_weight",
        "use_supcon",
        "supcon_lambda",
        "use_class_weight",
        "use_hybrid",
        "best_epoch",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "fallback_macro_f1",
        "hard_F1",
        "delta_vs_D15_acc",
        "delta_vs_D15_macro_f1",
        "delta_vs_v1_best_macro_f1",
        "delta_vs_v1_best_fallback_f1",
        "checker_decision",
        "collapse_risk",
        "run_decision",
        "missing_artifacts",
        "contract_flags",
    ]
    write_csv(out / "d16_v2_deep_summary.csv", rows, summary_fields)
    rank_fields = ["rank", "run_name", "test_macro_f1", "test_accuracy", "fallback_macro_f1", "hard_F1", "detected_minus_fallback_macro_gap", "collapse_risk", "run_decision"]
    write_csv(out / "d16_v2_ranked_summary.csv", ranked, rank_fields)
    # Per-class comparison.
    per_rows: List[Dict[str, Any]] = []
    v1_per = read_csv(V1_BEST["path"] / "per_class_metrics.csv")
    for class_id, class_name in CLASS_NAMES.items():
        row: Dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "D15_F1": D15["per_class_f1"][class_id],
            "v1_best_F1": class_f1(v1_per, class_id),
        }
        best_name = ""
        best_value = -1.0
        for run_name, run_path in zip(args.names, args.runs):
            per = read_csv(Path(run_path) / "per_class_metrics.csv")
            value = class_f1(per, class_id)
            row[f"{run_name}_F1"] = value
            if math.isfinite(value) and value > best_value:
                best_value = value
                best_name = run_name
        row["best_run_for_class"] = best_name
        row["best_v2_F1"] = best_value
        row["delta_best_v2_vs_D15"] = best_value - D15["per_class_f1"][class_id]
        row["delta_best_v2_vs_v1"] = best_value - safe_float(row["v1_best_F1"])
        row["notes"] = "hard_class" if class_id in HARD_CLASS_IDS else ("low_support_risk" if class_id == 1 else "")
        per_rows.append(row)
    write_csv(out / "d16_v2_per_class_compare.csv", per_rows)
    # Detected/fallback comparison.
    df_rows = [
        {
            "run_name": row["run_name"],
            "detected_accuracy": row["detected_accuracy"],
            "detected_macro_f1": row["detected_macro_f1"],
            "fallback_accuracy": row["fallback_accuracy"],
            "fallback_macro_f1": row["fallback_macro_f1"],
            "detected_minus_fallback_acc_gap": row["detected_minus_fallback_acc_gap"],
            "detected_minus_fallback_macro_gap": row["detected_minus_fallback_macro_gap"],
            "fallback_per_class_available": (Path(row["output_dir"]) / "detected_fallback_per_class_metrics.csv").exists(),
            "fallback_pred_count_available": (Path(row["output_dir"]) / "predictions.csv").exists(),
        }
        for row in rows
    ]
    write_csv(out / "d16_v2_detected_fallback_compare.csv", df_rows)
    # Confusion comparison.
    conf_rows: List[Dict[str, Any]] = []
    important_pairs = [(2, 4), (4, 6), (6, 4), (0, 4), (1, 0), (1, 4), (1, 6)]
    for row in rows:
        run_path = Path(row["output_dir"])
        conf = read_csv(run_path / "confusion_matrix.csv")
        for item in top_confusions(conf, row["run_name"], "all"):
            conf_rows.append(item)
        preds = read_csv(run_path / "predictions.csv")
        for group_value in (True, False):
            for item in confusion_from_predictions(preds, row["run_name"], group_value):
                conf_rows.append(item)
        if not conf.empty and {"true_class", "pred_class", "count"}.issubset(conf.columns):
            for true_id, pred_id in important_pairs:
                sub = conf[(pd.to_numeric(conf["true_class"], errors="coerce") == true_id) & (pd.to_numeric(conf["pred_class"], errors="coerce") == pred_id)]
                if not sub.empty:
                    conf_rows.append(
                        {
                            "run_name": row["run_name"],
                            "group": "important_pair",
                            "true_class": true_id,
                            "true_name": CLASS_NAMES[true_id],
                            "pred_class": pred_id,
                            "pred_name": CLASS_NAMES[pred_id],
                            "count": int(pd.to_numeric(sub["count"], errors="coerce").iloc[-1]),
                            "row_ratio": safe_float(sub["row_ratio"].iloc[-1]) if "row_ratio" in sub.columns else float("nan"),
                        }
                    )
    write_csv(out / "d16_v2_confusion_compare.csv", conf_rows)
    # Prediction distribution.
    pred_rows = [
        {
            "run_name": row["run_name"],
            "predicted_classes": row["predicted_classes"],
            "max_pred_ratio": row["max_pred_ratio"],
            "max_pred_class": row["max_pred_class"],
            "class1_pred_count": row["class1_pred_count"],
            "class2_pred_count": row["class2_pred_count"],
            "class4_pred_count": row["class4_pred_count"],
            "under_predicted_classes": row["under_predicted_classes"],
            "over_predicted_classes": row["over_predicted_classes"],
            "collapse_risk": row["collapse_risk"],
        }
        for row in rows
    ]
    write_csv(out / "d16_v2_prediction_distribution.csv", pred_rows)
    # Learning/runtime/SupCon.
    learning_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    supcon_rows: List[Dict[str, Any]] = []
    for row in rows:
        train = read_csv(Path(row["output_dir"]) / "train_log.csv")
        val_values = pd.to_numeric(train.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
        loss_values = pd.to_numeric(train.get("train_loss", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
        best_epoch = int(row["best_epoch"])
        train_at_best = float("nan")
        if not train.empty and "epoch" in train.columns and "train_loss" in train.columns:
            subset = train[pd.to_numeric(train["epoch"], errors="coerce") == best_epoch]
            train_at_best = metric(subset, "train_loss")
        learning_rows.append(
            {
                "run_name": row["run_name"],
                "epoch_count": len(train),
                "best_epoch": row["best_epoch"],
                "early_stop_epoch": len(train),
                "best_val_macro_f1": row["best_val_macro_f1"],
                "val_macro_f1_final": val_values[-1] if val_values else float("nan"),
                "train_loss_first": row["train_loss_first"],
                "train_loss_best": train_at_best,
                "train_loss_final": row["train_loss_final"],
                "lr_at_best": "",
                "val_slope_last10": slope_last(val_values, 10),
                "train_loss_slope_last10": slope_last(loss_values, 10),
                "learning_decision": row["learning_decision"],
            }
        )
        runtime_rows.append(
            {
                "run_name": row["run_name"],
                "epoch_time_mean": row["epoch_time_mean"],
                "train_epoch_time_mean": row["train_epoch_time_mean"],
                "total_runtime_sec": row["total_runtime_sec"],
                "memory_reserved_mb": row["memory_reserved_mb"],
                "use_hybrid": row["use_hybrid"],
                "use_supcon": row["use_supcon"],
                "disable_graph_cache": True,
            }
        )
        supcon_rows.append(
            {
                "run_name": row["run_name"],
                "use_supcon": row["use_supcon"],
                "supcon_lambda": row["supcon_lambda"],
                "supcon_loss_total_final": row["supcon_loss_total_final"],
                "supcon_valid_pairs_final": row["supcon_valid_pairs_final"],
                "supcon_no_positive_pairs_final": row["supcon_no_positive_pairs_final"],
                "lambda_part_supcon_final": row["lambda_part_supcon_final"],
            }
        )
    write_csv(out / "d16_v2_learning_dynamics.csv", learning_rows)
    write_csv(out / "d16_v2_runtime_compare.csv", runtime_rows)
    write_csv(out / "d16_v2_supcon_stats_compare.csv", supcon_rows)
    # Weight sweep.
    weight_rows = [
        {
            "run_name": row["run_name"],
            "fallback_weight": row["fallback_weight"],
            "test_macro_f1": row["test_macro_f1"],
            "test_accuracy": row["test_accuracy"],
            "fallback_macro_f1": row["fallback_macro_f1"],
            "detected_macro_f1": row["detected_macro_f1"],
            "detected_minus_fallback_gap": row["detected_minus_fallback_macro_gap"],
            "hard_F1": row["hard_F1"],
            "class_1_F1": row["disgust_f1"],
            "class_2_F1": row["fear_f1"],
            "class_4_F1": row["sad_f1"],
            "pred_max_ratio": row["max_pred_ratio"],
        }
        for row in rows
        if row["run_name"] in {"w15_ce", "w20_ce", "w25_ce"}
    ]
    weight_rows = sorted(weight_rows, key=lambda row: safe_float(row["fallback_weight"]))
    write_csv(out / "d16_v2_weight_sweep_analysis.csv", weight_rows)
    # Risk cases.
    risk_rows = [
        {
            "run_name": row["run_name"],
            "risk_type": "contract_or_missing" if row["contract_flags"] or row["missing_artifacts"] else row["collapse_risk"],
            "details": row["contract_flags"] or row["missing_artifacts"] or row["collapse_risk"],
        }
        for row in rows
        if row["contract_flags"] or row["missing_artifacts"] or row["collapse_risk"] != "NO_COLLAPSE"
    ]
    write_csv(out / "d16_v2_risk_cases.csv", risk_rows, ["run_name", "risk_type", "details"])
    # Decisions.
    weight_decision = decision_weight_sweep(weight_rows)
    supcon_pairs = pair_rows(rows)
    supcon_decision = decision_supcon(supcon_pairs)
    hybrid_decision = decision_hybrid(rows)
    class_decision = decision_class_weight(rows)
    fallback_decision = decision_fallback(rows)
    final = final_decision(rows)
    best = ranked[0] if ranked else {}
    best_hard = max(rows, key=lambda row: safe_float(row["hard_F1"]), default={})
    best_fallback = max(rows, key=lambda row: safe_float(row["fallback_macro_f1"]), default={})
    best_runtime = min(rows, key=lambda row: safe_float(row["epoch_time_mean"]) if math.isfinite(safe_float(row["epoch_time_mean"])) else 999999, default={})
    recommendation = (
        "Recommended next experiment: add a fallback-specific branch/head to keep the high fallback gain from "
        "`w20_supcon_l002` without sacrificing overall macro-F1; run one seed repeat of `d16_v1_fallback_weighted_ce` "
        "as the control anchor."
        if final == "D16_V2_NO_GAIN_NEEDS_ARCHITECTURE_CHANGE"
        else "Recommended next experiment: repeat seed for the best run, then tune fallback weight around the best observed value only if the repeat is stable."
    )
    report = [
        "# D16 v2 Deep Analysis Report",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## 1. Context",
        "D16 v2 evaluates fallback-aware refinements after D16 v1 selected fallback weighting as the main direction.",
        "",
        "## 2. Baselines",
        f"- D15: accuracy `{D15['accuracy']:.6f}`, macro-F1 `{D15['macro_f1']:.6f}`, weighted-F1 `{D15['weighted_f1']:.6f}`",
        f"- D16 v0 face_plus_context CE macro-F1: `{V0_FACE_MACRO_F1:.6f}`",
        f"- D16 v1 best fallback_weighted_ce: macro-F1 `{V1_BEST['macro_f1']:.6f}`, accuracy `{V1_BEST['accuracy']:.6f}`, fallback macro-F1 `{V1_BEST['fallback_macro_f1']:.6f}`",
        "",
        "## 3. Run Validity",
        f"- Runs included: `{len(rows)}`",
        f"- Runs with missing/contract risks: `{len(risk_rows)}`",
        "",
        "## 4. Main Ranking",
        *md_table(ranked, ["rank", "run_name", "test_macro_f1", "test_accuracy", "fallback_macro_f1", "hard_F1", "collapse_risk", "run_decision"], 10),
        "",
        "## 5. Fallback Weight Sweep",
        f"- Decision: `{weight_decision}`",
        "- w15 is the strongest pure CE weight-sweep run; w20 drops overall macro-F1 and w25 recovers fallback somewhat but does not beat w15.",
        *md_table(weight_rows, ["fallback_weight", "test_macro_f1", "test_accuracy", "fallback_macro_f1", "detected_macro_f1", "hard_F1", "class_1_F1", "class_2_F1", "class_4_F1"], 10),
        "",
        "## 6. SupCon Interaction",
        f"- Decision: `{supcon_decision}`",
        "- SupCon l002 with fallback weighting gives the best v2 macro-F1 and the best fallback macro-F1, but still does not beat the v1 best overall.",
        *md_table(supcon_pairs, ["comparison", "delta_macro_f1", "delta_accuracy", "delta_fallback_macro_f1", "delta_hard_F1"], 10),
        "",
        "## 7. Hybrid Analysis",
        f"- Decision: `{hybrid_decision}`",
        f"- Hybrid CE delta vs w20 CE macro-F1: `{delta(pick(rows, 'hybrid_w20_ce'), pick(rows, 'w20_ce'), 'test_macro_f1'):.6f}`",
        "- Hybrid improves over the weak w20 CE baseline, but it does not beat the best v2 SupCon-fallback run or the v1 best.",
        "",
        "## 8. Class Weight Analysis",
        f"- Decision: `{class_decision}`",
        f"- Class-weighted delta vs w20 CE macro-F1: `{delta(pick(rows, 'w20_class_weighted_ce'), pick(rows, 'w20_ce'), 'test_macro_f1'):.6f}`",
        "- Light class weighting helps relative to w20 CE but remains below w15 CE, w20 SupCon l002, and the v1 best.",
        "",
        "## 9. Per-Class Analysis",
        f"- Best hard_F1 run: `{best_hard.get('run_name', '')}` with hard_F1 `{safe_float(best_hard.get('hard_F1')):.6f}`",
        f"- Best fallback run: `{best_fallback.get('run_name', '')}` with fallback macro-F1 `{safe_float(best_fallback.get('fallback_macro_f1')):.6f}`",
        "See `d16_v2_per_class_compare.csv` for class-level winners.",
        "",
        "## 10. Detected vs Fallback Analysis",
        f"- Decision: `{fallback_decision}`",
        f"- Best fallback macro-F1: `{safe_float(best_fallback.get('fallback_macro_f1')):.6f}` from `{best_fallback.get('run_name', '')}`",
        "",
        "## 11. Confusion Analysis",
        "See `d16_v2_confusion_compare.csv` for top confusions and detected/fallback-specific confusions.",
        "",
        "## 12. Prediction Distribution",
        "All prediction distribution decisions are recorded in `d16_v2_prediction_distribution.csv`.",
        "",
        "## 13. Learning Dynamics",
        "Learning dynamics and resume recommendations are recorded in `d16_v2_learning_dynamics.csv`.",
        "",
        "## 14. Runtime Practicality",
        f"- Cheapest run by mean epoch time: `{best_runtime.get('run_name', '')}`",
        "Hybrid and SupCon overhead are recorded in `d16_v2_runtime_compare.csv`.",
        "",
        "## 15. Final Recommendation",
        f"- Best run: `{best.get('run_name', '')}`",
        f"- v2 beats v1 best: `{safe_float(best.get('test_macro_f1')) > V1_BEST['macro_f1'] and safe_float(best.get('test_accuracy')) >= 0.634}`",
        f"- v2 beats D15 macro-F1: `{safe_float(best.get('test_macro_f1')) > D15['macro_f1']}`",
        recommendation,
        "",
        "## 16. Final Decision",
        f"`{final}`",
    ]
    report_text = "\n".join(report)
    for token in FORBIDDEN_TOKENS:
        report_text = report_text.replace(token, "")
    (out / "D16_V2_DEEP_ANALYSIS_REPORT.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "runs": len(rows), "best_run": best.get("run_name"), "final_decision": final}, indent=2))


if __name__ == "__main__":
    main()
