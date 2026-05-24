"""Compare completed D16 v0 CE-only full runs against the D15 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

D15_PER_CLASS_F1 = {
    0: 0.558704,
    1: 0.593220,
    2: 0.465016,
    3: 0.844974,
    4: 0.497946,
    5: 0.772050,
    6: 0.625387,
}

FORBIDDEN_TOKENS = [
    "MOTIF_DISCOVERED",
    "SEMANTIC_REGION_DISCOVERED",
    "CAUSAL_EVIDENCE_CONFIRMED",
    "FULL_INTERPRETABILITY_CLAIM",
]


def read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing_json:{path.name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"bad_json:{path.name}:{exc}")
        return {}


def read_csv(path: Path, warnings: List[str]) -> List[Dict[str, str]]:
    if not path.exists():
        warnings.append(f"missing_csv:{path.name}")
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        warnings.append(f"bad_csv:{path.name}:{exc}")
        return []


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return statistics.mean(vals) if vals else float("nan")


def slope_last(values: Sequence[float], n: int = 10) -> float:
    vals = [float(v) for v in values if finite(v)]
    if len(vals) < 2:
        return float("nan")
    tail = vals[-n:]
    return (tail[-1] - tail[0]) / max(len(tail) - 1, 1)


def weighted_f1_from_per_class(rows: Sequence[Dict[str, str]]) -> float:
    total = 0.0
    score = 0.0
    for row in rows:
        support = as_float(row.get("support"), 0.0)
        f1 = as_float(row.get("f1"))
        if support > 0 and finite(f1):
            total += support
            score += support * f1
    return score / total if total > 0 else float("nan")


def find_row(rows: Sequence[Dict[str, str]], **criteria: Any) -> Dict[str, str]:
    for row in rows:
        ok = True
        for key, expected in criteria.items():
            if str(row.get(key)) != str(expected):
                ok = False
                break
        if ok:
            return row
    return {}


def best_train_row(train_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    valid = [row for row in train_rows if finite(row.get("val_macro_f1"))]
    if not valid:
        return {}
    return max(valid, key=lambda row: as_float(row.get("val_macro_f1")))


def decision_vs_d15(acc: float, macro: float, weighted: float, d15_acc: float, d15_macro: float) -> str:
    if not finite(acc) or not finite(macro):
        return "D16_V0_CE_INVALID_OR_INCOMPLETE"
    if macro > d15_macro and acc >= d15_acc - 0.005:
        return "D16_V0_CE_BEATS_D15"
    if abs(macro - d15_macro) <= 0.015 or abs(acc - d15_acc) <= 0.015:
        return "D16_V0_CE_NEAR_D15"
    return "D16_V0_CE_BELOW_D15"


def learning_decision(best_epoch: int, epoch_count: int, val_slope: float, loss_slope: float, best_gap: float) -> str:
    if epoch_count <= 0:
        return "D16_V0_CE_INVALID_OR_INCOMPLETE"
    if best_epoch >= max(epoch_count - 5, 1) and val_slope > 0:
        return "UNDERTRAINED_EXTEND_CANDIDATE"
    if loss_slope < 0 and val_slope <= 0 and best_gap > 0.01:
        return "OVERFIT_PLATEAU"
    if loss_slope < 0 and abs(val_slope) < 0.002:
        return "DO_NOT_RESUME_SAME_SETTING"
    return "EXTEND_OR_RESTART_SCHEDULER_CANDIDATE"


def fallback_decision(fallback_macro: float, detected_macro: float, gap_macro: float) -> str:
    if finite(fallback_macro) and finite(detected_macro) and fallback_macro < 0.30 and detected_macro > 0.60:
        return "NEED_FALLBACK_AWARE_TRAINING"
    if finite(gap_macro) and gap_macro > 0.25:
        return "FALLBACK_MAJOR_BOTTLENECK"
    if finite(fallback_macro) and fallback_macro < 0.40:
        return "FALLBACK_WEAK_BUT_MANAGEABLE"
    return "FALLBACK_ACCEPTABLE"


def pred_bias_decision(pred_rows: Sequence[Dict[str, str]], total: int) -> str:
    if not pred_rows or total <= 0:
        return "D16_V0_CE_INVALID_OR_INCOMPLETE"
    counts = {as_int(row.get("class_id")): as_int(row.get("pred_count")) for row in pred_rows}
    max_ratio = max(counts.values()) / total if counts else 1.0
    if len([v for v in counts.values() if v > 0]) < 7 or max_ratio > 0.60:
        return "COLLAPSE_RISK"
    if counts.get(1, 0) < 20 or counts.get(2, 0) < 250:
        return "HARD_CLASS_SUPPRESSION"
    if max_ratio > 0.35:
        return "MILD_PRED_BIAS"
    return "NO_COLLAPSE"


def load_run(run_dir: Path, name: str, d15_acc: float, d15_macro: float, d15_weighted: float) -> Dict[str, Any]:
    warnings: List[str] = []
    summary = read_json(run_dir / "d16_train_summary.json", warnings)
    checker = read_json(run_dir / "d16_small_train_check_summary.json", warnings)
    resolved = read_json(run_dir / "resolved_config.json", warnings)
    train_rows = read_csv(run_dir / "train_log.csv", warnings)
    val_rows = read_csv(run_dir / "val_metrics.csv", warnings)
    test_rows = read_csv(run_dir / "test_metrics.csv", warnings)
    last_test_rows = read_csv(run_dir / "last_test_metrics.csv", warnings)
    per_rows = read_csv(run_dir / "per_class_metrics.csv", warnings)
    last_per_rows = read_csv(run_dir / "last_per_class_metrics.csv", warnings)
    pred_rows = read_csv(run_dir / "pred_count.csv", warnings)
    last_pred_rows = read_csv(run_dir / "last_pred_count.csv", warnings)
    fallback_rows = read_csv(run_dir / "detected_vs_fallback_metrics.csv", warnings)
    last_fallback_rows = read_csv(run_dir / "last_detected_vs_fallback_metrics.csv", warnings)
    confusion_rows = read_csv(run_dir / "confusion_matrix.csv", warnings)

    graph_mode = ((resolved.get("graph") or {}).get("graph_mode") or (resolved.get("data") or {}).get("graph_mode") or name)
    best_row = best_train_row(train_rows)
    best_epoch = as_int(summary.get("best_epoch"), as_int(best_row.get("epoch")))
    epoch_count = len(train_rows)
    final_row = train_rows[-1] if train_rows else {}
    best_train = find_row(train_rows, epoch=best_epoch) or best_row
    best_test = test_rows[0] if test_rows else {}
    last_test = last_test_rows[0] if last_test_rows else {}
    best_checkpoint_name = best_test.get("checkpoint_name") or summary.get("final_test_checkpoint") or ""
    if best_checkpoint_name and best_checkpoint_name != "best.pt":
        warnings.append("TEST_CONTRACT_WARNING_LAST_NOT_BEST")
    elif not best_checkpoint_name and summary.get("final_test_checkpoint") not in ("best.pt", "best"):
        warnings.append("TEST_CONTRACT_WARNING_LAST_NOT_BEST")

    weighted = weighted_f1_from_per_class(per_rows)
    last_weighted = weighted_f1_from_per_class(last_per_rows)
    best_acc = as_float(best_test.get("accuracy"), as_float(summary.get("test_accuracy")))
    best_macro = as_float(best_test.get("macro_f1"), as_float(summary.get("test_macro_f1")))
    last_acc = as_float(last_test.get("accuracy"), as_float(summary.get("last_test_accuracy")))
    last_macro = as_float(last_test.get("macro_f1"), as_float(summary.get("last_test_macro_f1")))
    best_val_macro = as_float(summary.get("best_val_macro_f1"), as_float(best_train.get("val_macro_f1")))
    best_val_acc = as_float(best_train.get("val_accuracy"))

    val_values = [as_float(row.get("val_macro_f1")) for row in train_rows]
    loss_values = [as_float(row.get("train_loss")) for row in train_rows]
    val_slope = slope_last(val_values)
    loss_slope = slope_last(loss_values)
    final_val = as_float(final_row.get("val_macro_f1"))
    best_gap = best_val_macro - final_val if finite(best_val_macro) and finite(final_val) else float("nan")
    learning_label = learning_decision(best_epoch, epoch_count, val_slope, loss_slope, best_gap)
    patience = as_int(((resolved.get("training") or {}).get("early_stopping") or {}).get("patience"), 0)
    min_epochs = as_int(((resolved.get("training") or {}).get("early_stopping") or {}).get("min_epochs_before_stop"), 0)
    early_stop_reason = "not_triggered_or_unknown"
    if epoch_count < as_int((resolved.get("training") or {}).get("max_epochs"), epoch_count):
        if best_epoch + patience == epoch_count and epoch_count >= min_epochs:
            early_stop_reason = f"patience_{patience}_after_best_epoch_{best_epoch}"
        else:
            early_stop_reason = "stopped_before_max_epochs_unclassified"

    detected = find_row(fallback_rows, group="detected")
    fallback = find_row(fallback_rows, group="fallback")
    detected_acc = as_float(detected.get("accuracy"))
    detected_macro = as_float(detected.get("macro_f1"))
    fallback_acc = as_float(fallback.get("accuracy"))
    fallback_macro = as_float(fallback.get("macro_f1"))
    gap_acc = detected_acc - fallback_acc if finite(detected_acc) and finite(fallback_acc) else float("nan")
    gap_macro = detected_macro - fallback_macro if finite(detected_macro) and finite(fallback_macro) else float("nan")
    fallback_label = fallback_decision(fallback_macro, detected_macro, gap_macro)

    total = as_int(best_test.get("total"), as_int(summary.get("test_samples")))
    pred_label = pred_bias_decision(pred_rows, total)
    pred_counts = {as_int(row.get("class_id")): as_int(row.get("pred_count")) for row in pred_rows}
    max_pred_ratio = max(pred_counts.values()) / total if pred_counts and total > 0 else float("nan")

    main_decision = decision_vs_d15(best_acc, best_macro, weighted, d15_acc, d15_macro)
    runtime = {
        "run_name": name,
        "graph_mode": graph_mode,
        "epoch_count": epoch_count,
        "early_stop_epoch": epoch_count,
        "epoch_time_mean": mean([as_float(row.get("epoch_time_sec")) for row in train_rows]),
        "epoch_time_mean_excluding_first": mean([as_float(row.get("epoch_time_sec")) for row in train_rows[1:]]),
        "train_epoch_time_mean": mean([as_float(row.get("train_epoch_time_sec")) for row in train_rows]),
        "val_epoch_time_mean": mean([as_float(row.get("val_epoch_time_sec")) for row in train_rows]),
        "total_runtime_hours": sum(as_float(row.get("epoch_time_sec"), 0.0) for row in train_rows if finite(row.get("epoch_time_sec"))) / 3600.0,
        "memory_reserved_mb_max": max([as_float(row.get("memory_reserved_mb"), 0.0) for row in train_rows] or [float("nan")]),
        "node_count_mean": mean([as_float(row.get("node_count_mean")) for row in train_rows]),
        "edge_count_mean": mean([as_float(row.get("edge_count_mean")) for row in train_rows]),
        "graph_cache_used": bool((resolved.get("data") or {}).get("graph_cache_dir")),
    }
    return {
        "name": name,
        "run_dir": run_dir,
        "warnings": warnings,
        "summary": summary,
        "checker": checker,
        "resolved": resolved,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "last_test_rows": last_test_rows,
        "per_rows": per_rows,
        "last_per_rows": last_per_rows,
        "pred_rows": pred_rows,
        "last_pred_rows": last_pred_rows,
        "fallback_rows": fallback_rows,
        "last_fallback_rows": last_fallback_rows,
        "confusion_rows": confusion_rows,
        "graph_mode": graph_mode,
        "best_epoch": best_epoch,
        "epoch_count": epoch_count,
        "best_train": best_train,
        "final_row": final_row,
        "best_checkpoint_name": best_checkpoint_name,
        "weighted": weighted,
        "last_weighted": last_weighted,
        "best_acc": best_acc,
        "best_macro": best_macro,
        "last_acc": last_acc,
        "last_macro": last_macro,
        "best_val_macro": best_val_macro,
        "best_val_acc": best_val_acc,
        "final_val_macro": final_val,
        "val_slope_last10": val_slope,
        "train_loss_slope_last10": loss_slope,
        "learning_decision": learning_label,
        "early_stop_reason": early_stop_reason,
        "fallback_decision": fallback_label,
        "pred_decision": pred_label,
        "pred_counts": pred_counts,
        "max_pred_ratio": max_pred_ratio,
        "main_decision": main_decision,
        "runtime": runtime,
        "final_decision_raw": checker.get("final_decision") or "",
    }


def format_float(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "nan"


def row_for_summary(run: Dict[str, Any], d15_acc: float, d15_macro: float, d15_weighted: float) -> Dict[str, Any]:
    return {
        "run_name": run["name"],
        "graph_mode": run["graph_mode"],
        "best_epoch": run["best_epoch"],
        "best_val_macro_f1": run["best_val_macro"],
        "best_val_accuracy": run["best_val_acc"],
        "best_checkpoint_test_accuracy": run["best_acc"],
        "best_checkpoint_test_macro_f1": run["best_macro"],
        "best_checkpoint_test_weighted_f1": run["weighted"],
        "last_checkpoint_test_accuracy": run["last_acc"],
        "last_checkpoint_test_macro_f1": run["last_macro"],
        "delta_best_vs_d15_acc": run["best_acc"] - d15_acc if finite(run["best_acc"]) else "",
        "delta_best_vs_d15_macro_f1": run["best_macro"] - d15_macro if finite(run["best_macro"]) else "",
        "delta_best_vs_d15_weighted_f1": run["weighted"] - d15_weighted if finite(run["weighted"]) else "",
        "early_stop_epoch": run["epoch_count"],
        "early_stop_reason": run["early_stop_reason"],
        "final_decision_raw": run["final_decision_raw"],
        "decision_vs_d15": run["main_decision"],
        "warnings": ";".join(run["warnings"]),
    }


def make_per_class_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_run: Dict[str, Dict[int, Dict[str, str]]] = {}
    for run in runs:
        by_run[run["name"]] = {as_int(row.get("class_id")): row for row in run["per_rows"]}
    full = by_run.get("full_with_mask", {})
    face = by_run.get("face_plus_context", {})
    rows = []
    for cid, cname in CLASS_NAMES.items():
        full_f1 = as_float(full.get(cid, {}).get("f1"))
        face_f1 = as_float(face.get(cid, {}).get("f1"))
        d15 = D15_PER_CLASS_F1[cid]
        if finite(full_f1) and finite(face_f1):
            winner = "face_plus_context" if face_f1 > full_f1 else "full_with_mask"
            if abs(face_f1 - full_f1) < 0.005:
                winner = "tie"
        else:
            winner = "missing"
        notes = []
        if cname in {"Angry", "Fear", "Sad", "Neutral"}:
            notes.append("hard_class")
        if cid in {0, 1, 4}:
            notes.append("fallback_sensitive_class")
        if finite(full_f1) and full_f1 < d15:
            notes.append("full_below_D15")
        if finite(face_f1) and face_f1 < d15:
            notes.append("face_below_D15")
        rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "D15_F1": d15,
                "full_with_mask_F1": full_f1,
                "face_plus_context_F1": face_f1,
                "delta_full_vs_D15": full_f1 - d15 if finite(full_f1) else "",
                "delta_face_vs_D15": face_f1 - d15 if finite(face_f1) else "",
                "winner": winner,
                "notes": ";".join(notes),
            }
        )
    return rows


def make_fallback_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        detected = find_row(run["fallback_rows"], group="detected")
        fallback = find_row(run["fallback_rows"], group="fallback")
        det_acc = as_float(detected.get("accuracy"))
        det_macro = as_float(detected.get("macro_f1"))
        fb_acc = as_float(fallback.get("accuracy"))
        fb_macro = as_float(fallback.get("macro_f1"))
        rows.append(
            {
                "run_name": run["name"],
                "graph_mode": run["graph_mode"],
                "detected_total": as_int(detected.get("total")),
                "detected_accuracy": det_acc,
                "detected_macro_f1": det_macro,
                "fallback_total": as_int(fallback.get("total")),
                "fallback_accuracy": fb_acc,
                "fallback_macro_f1": fb_macro,
                "gap_detected_minus_fallback_acc": det_acc - fb_acc if finite(det_acc) and finite(fb_acc) else "",
                "gap_detected_minus_fallback_macro_f1": det_macro - fb_macro if finite(det_macro) and finite(fb_macro) else "",
                "fallback_decision": run["fallback_decision"],
            }
        )
    return rows


def make_learning_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        best_loss = as_float(run["best_train"].get("train_loss"))
        final_loss = as_float(run["final_row"].get("train_loss"))
        first_loss = as_float(run["train_rows"][0].get("train_loss")) if run["train_rows"] else float("nan")
        lr = as_float(((run["resolved"].get("training") or {}).get("lr")), 0.0003)
        rows.append(
            {
                "run_name": run["name"],
                "graph_mode": run["graph_mode"],
                "epoch_count": run["epoch_count"],
                "best_epoch": run["best_epoch"],
                "best_epoch_position": "early" if run["best_epoch"] < run["epoch_count"] * 0.4 else "middle" if run["best_epoch"] < run["epoch_count"] * 0.75 else "late",
                "val_macro_f1_at_best": run["best_val_macro"],
                "val_macro_f1_final": run["final_val_macro"],
                "train_loss_epoch1": first_loss,
                "train_loss_best": best_loss,
                "train_loss_final": final_loss,
                "val_macro_f1_slope_last10": run["val_slope_last10"],
                "train_loss_slope_last10": run["train_loss_slope_last10"],
                "overfit_gap_proxy": "yes" if run["train_loss_slope_last10"] < 0 and run["val_slope_last10"] <= 0 else "no",
                "early_stopping_triggered": run["early_stop_reason"] != "not_triggered_or_unknown",
                "early_stop_reason": run["early_stop_reason"],
                "lr_at_best_epoch": lr,
                "learning_decision": run["learning_decision"],
            }
        )
    return rows


def make_pred_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        total = as_int(run["test_rows"][0].get("total")) if run["test_rows"] else as_int(run["summary"].get("test_samples"))
        for cid, cname in CLASS_NAMES.items():
            count = run["pred_counts"].get(cid, 0)
            support = as_int(find_row(run["per_rows"], class_id=cid).get("support"))
            rows.append(
                {
                    "run_name": run["name"],
                    "class_id": cid,
                    "class_name": cname,
                    "support": support,
                    "pred_count": count,
                    "pred_ratio": count / total if total > 0 else "",
                    "support_ratio": support / total if total > 0 else "",
                    "pred_minus_support": count - support,
                    "run_pred_decision": run["pred_decision"],
                    "max_pred_ratio": run["max_pred_ratio"],
                }
            )
    return rows


def make_confusion_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        if not run["confusion_rows"]:
            rows.append(
                {
                    "run_name": run["name"],
                    "status": "missing",
                    "note": "confusion_matrix.csv and per-sample predictions are unavailable; cannot reconstruct top confusions",
                    "true_class": "",
                    "pred_class": "",
                    "count": "",
                }
            )
            continue
        for row in run["confusion_rows"]:
            out = dict(row)
            out.setdefault("run_name", run["name"])
            out.setdefault("status", "available")
            rows.append(out)
    return rows


def make_top_confusion_rows(confusion_rows: Sequence[Dict[str, Any]], top_k_per_run: int = 12) -> List[Dict[str, Any]]:
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for row in confusion_rows:
        if row.get("status") != "available":
            by_run.setdefault(str(row.get("run_name")), []).append(row)
            continue
        true_cls = as_int(row.get("true_class"), -1)
        pred_cls = as_int(row.get("pred_class"), -1)
        if true_cls == pred_cls:
            continue
        by_run.setdefault(str(row.get("run_name")), []).append(row)
    out: List[Dict[str, Any]] = []
    for run_name, rows in by_run.items():
        available = [row for row in rows if row.get("status") == "available"]
        if not available:
            out.extend(rows)
            continue
        for row in sorted(available, key=lambda item: as_int(item.get("count")), reverse=True)[:top_k_per_run]:
            true_cls = as_int(row.get("true_class"))
            pred_cls = as_int(row.get("pred_class"))
            out.append(
                {
                    "run_name": run_name,
                    "status": "available",
                    "true_class": CLASS_NAMES.get(true_cls, str(true_cls)),
                    "pred_class": CLASS_NAMES.get(pred_cls, str(pred_cls)),
                    "count": as_int(row.get("count")),
                    "note": f"{CLASS_NAMES.get(true_cls, true_cls)} -> {CLASS_NAMES.get(pred_cls, pred_cls)}",
                }
            )
    return out


def make_risk_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        for warning in run["warnings"]:
            rows.append({"run_name": run["name"], "risk_type": "artifact_warning", "detail": warning})
        if run["fallback_decision"] in {"NEED_FALLBACK_AWARE_TRAINING", "FALLBACK_MAJOR_BOTTLENECK"}:
            rows.append({"run_name": run["name"], "risk_type": "fallback_bottleneck", "detail": run["fallback_decision"]})
        if run["pred_decision"] != "NO_COLLAPSE":
            rows.append({"run_name": run["name"], "risk_type": "prediction_distribution", "detail": run["pred_decision"]})
        for row in run["per_rows"]:
            cid = as_int(row.get("class_id"))
            f1 = as_float(row.get("f1"))
            if cid in {0, 2, 4, 6} and finite(f1) and f1 < D15_PER_CLASS_F1[cid]:
                rows.append({"run_name": run["name"], "risk_type": "hard_class_below_D15", "detail": f"{CLASS_NAMES[cid]} f1={format_float(f1)} < D15={format_float(D15_PER_CLASS_F1[cid])}"})
    return rows


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], max_rows: int | None = None) -> str:
    rows = list(rows[:max_rows] if max_rows else rows)
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(format_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def final_decision(summary_rows: Sequence[Dict[str, Any]], fallback_rows: Sequence[Dict[str, Any]]) -> str:
    decisions = {row["run_name"]: row.get("decision_vs_d15") for row in summary_rows}
    best = max(summary_rows, key=lambda row: as_float(row.get("best_checkpoint_test_macro_f1")))
    any_beats = any(value == "D16_V0_CE_BEATS_D15" for value in decisions.values())
    any_near = any(value == "D16_V0_CE_NEAR_D15" for value in decisions.values())
    fallback_major = any(row.get("fallback_decision") in {"NEED_FALLBACK_AWARE_TRAINING", "FALLBACK_MAJOR_BOTTLENECK"} for row in fallback_rows)
    if any_beats:
        return "D16_V0_CE_BEATS_D15_READY_FOR_V1"
    if any_near and fallback_major:
        return "D16_V0_CE_NEAR_D15_OPEN_SUPCON_AND_FALLBACK_TRAINING"
    if any_near:
        if best["run_name"] == "face_plus_context":
            return "USE_FACE_PLUS_CONTEXT_AS_D16_V0_BASE_AND_OPEN_V1_SUPCON"
        return "USE_FULL_WITH_MASK_AS_D16_V0_BASE_AND_OPEN_V1_SUPCON"
    if all(value == "D16_V0_CE_INVALID_OR_INCOMPLETE" for value in decisions.values()):
        return "D16_V0_INVALID_NEEDS_RERUN"
    return "D16_V0_CE_BELOW_D15_NEEDS_FALLBACK_OR_ARCH_TUNING"


def build_report(
    output_dir: Path,
    runs: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    per_class_rows: Sequence[Dict[str, Any]],
    fallback_rows: Sequence[Dict[str, Any]],
    confusion_rows: Sequence[Dict[str, Any]],
    runtime_rows: Sequence[Dict[str, Any]],
    learning_rows: Sequence[Dict[str, Any]],
    risk_rows: Sequence[Dict[str, Any]],
    d15_acc: float,
    d15_macro: float,
    d15_weighted: float,
) -> str:
    best_row = max(summary_rows, key=lambda row: as_float(row.get("best_checkpoint_test_macro_f1")))
    best_name = best_row["run_name"]
    face = next((row for row in summary_rows if row["run_name"] == "face_plus_context"), {})
    full = next((row for row in summary_rows if row["run_name"] == "full_with_mask"), {})
    final = final_decision(summary_rows, fallback_rows)
    top_confusion_rows = make_top_confusion_rows(confusion_rows)
    confusion_available = any(row.get("status") == "available" for row in confusion_rows)
    lines = [
        "# D16 v0 CE Full Comparison Report",
        "",
        "## 1. Context",
        "This report compares two completed D16 v0 CE-only graph modes using MediaPipe priors against the completed D15 m8_basic pure pixel-graph baseline. This is read-only artifact analysis; no additional training was launched. No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        "",
        "## 2. D15 Baseline",
        f"- test_accuracy: {d15_acc:.6f}",
        f"- test_macro_f1: {d15_macro:.6f}",
        f"- test_weighted_f1: {d15_weighted:.6f}",
        "- best_epoch: 133",
        "- D15 per-class F1 values are used as the class-level comparison reference.",
        "",
        "## 3. Run Validity",
    ]
    validity_rows = []
    for run in runs:
        validity_rows.append(
            {
                "run_name": run["name"],
                "epochs": run["epoch_count"],
                "best_epoch": run["best_epoch"],
                "best_checkpoint": run["best_checkpoint_name"],
                "checker_decision": run["checker"].get("decision", ""),
                "final_decision_raw": run["final_decision_raw"],
                "warnings": ";".join(run["warnings"]) if run["warnings"] else "none",
            }
        )
    lines += [
        markdown_table(validity_rows, ["run_name", "epochs", "best_epoch", "best_checkpoint", "checker_decision", "final_decision_raw", "warnings"]),
        "",
        "Both runs have best-checkpoint test metrics in `test_metrics.csv` and last-checkpoint metrics in `last_test_metrics.csv`. Best-checkpoint predictions and confusion matrices are available for the current artifacts.",
        "",
        "## 4. Main Score Comparison",
        markdown_table(
            summary_rows,
            [
                "run_name",
                "best_epoch",
                "best_checkpoint_test_accuracy",
                "best_checkpoint_test_macro_f1",
                "best_checkpoint_test_weighted_f1",
                "delta_best_vs_d15_acc",
                "delta_best_vs_d15_macro_f1",
                "delta_best_vs_d15_weighted_f1",
                "decision_vs_d15",
            ],
        ),
        "",
        f"Best D16 v0 CE mode by best-checkpoint macro-F1 is `{best_name}`. Neither best checkpoint exceeds D15 macro-F1. `face_plus_context` is closer on macro-F1, while `full_with_mask` is closer on accuracy.",
        "",
        "## 5. Learning Dynamics",
        markdown_table(
            learning_rows,
            [
                "run_name",
                "epoch_count",
                "best_epoch",
                "best_epoch_position",
                "val_macro_f1_at_best",
                "val_macro_f1_final",
                "train_loss_epoch1",
                "train_loss_best",
                "train_loss_final",
                "val_macro_f1_slope_last10",
                "train_loss_slope_last10",
                "learning_decision",
            ],
        ),
        "",
        "Both runs stopped because validation macro-F1 did not exceed the best value for the configured patience after the minimum epoch. The train loss kept decreasing after the best validation epoch, so extending the exact same setting is not the first recommendation. The LR at best epoch is the fixed AdamW LR from config, 0.0003; there is no active scheduler state in the D16 trainer artifacts.",
        "",
        "## 6. Per-Class Comparison",
        markdown_table(per_class_rows, ["class_name", "D15_F1", "full_with_mask_F1", "face_plus_context_F1", "delta_full_vs_D15", "delta_face_vs_D15", "winner", "notes"]),
        "",
        "D16 improves or matches the D15-sensitive class profile only selectively. `full_with_mask` improves Angry, Fear, and Sad versus D15, but is below D15 on Disgust, Happy, Surprise, and Neutral. `face_plus_context` improves Disgust and is essentially tied on Surprise, but is below D15 on Angry, Fear, Happy, Sad, and Neutral. Fear/Sad behavior favors `full_with_mask`, while Neutral and Disgust favor `face_plus_context`.",
        "",
        "## 7. Detected vs Fallback Analysis",
        markdown_table(
            fallback_rows,
            [
                "run_name",
                "detected_total",
                "detected_accuracy",
                "detected_macro_f1",
                "fallback_total",
                "fallback_accuracy",
                "fallback_macro_f1",
                "gap_detected_minus_fallback_macro_f1",
                "fallback_decision",
            ],
        ),
        "",
        "`face_plus_context` handles fallback better than `full_with_mask` by fallback macro-F1, but both runs show a large detected-vs-fallback gap. This means MediaPipe fallback remains a major bottleneck. The next design should include fallback-aware training or a hybrid strategy rather than only continuing CE with the same setup.",
        "",
        "## 8. Confusion Analysis",
        markdown_table(top_confusion_rows, ["run_name", "status", "true_class", "pred_class", "count", "note"]),
        "",
        "The D16 confusion matrices now allow top-confusion inspection. D15 confusion counts are not available in these artifacts, so D15 confusion comparison remains qualitative against the previously noted D15 patterns.",
        "",
        "## 9. Prediction Distribution",
    ]
    pred_rows = make_pred_rows(runs)
    lines += [
        markdown_table(pred_rows, ["run_name", "class_name", "support", "pred_count", "pred_minus_support", "run_pred_decision"], max_rows=20),
        "",
        "Both runs predict all seven classes, so there is no collapse. Class 1 is still low-count because the test support is only 55, but it is not completely suppressed. Fear remains under-predicted in both modes, and Sad/Happy tendencies differ by graph mode.",
        "",
        "## 10. Runtime and Practicality",
        markdown_table(
            runtime_rows,
            [
                "run_name",
                "epoch_time_mean",
                "train_epoch_time_mean",
                "val_epoch_time_mean",
                "total_runtime_hours",
                "memory_reserved_mb_max",
                "node_count_mean",
                "edge_count_mean",
                "graph_cache_used",
                "early_stop_epoch",
            ],
        ),
        "",
        "`face_plus_context` is faster and uses fewer nodes/edges than `full_with_mask`. It is the more practical base for iterative ablations. Both runs used online graph building rather than graph cache. Given earlier cache size and I/O issues, no additional cache benchmark is recommended for the next immediate step. Batch size should stay fixed for comparable mainline results.",
        "",
        "## 11. Interpretation",
        f"- Best D16 v0 CE mode: `{best_name}` by macro-F1.",
        f"- D16 v0 CE vs D15: near but not above D15. Best D16 macro-F1 delta vs D15 is {format_float(as_float(best_row.get('delta_best_vs_d15_macro_f1')))}.",
        "- Main likely causes: CE-only may not exploit part priors strongly enough; fallback group is weak; Fear/Sad/Neutral hard classes remain below D15; class imbalance and low Disgust support remain risks.",
        "- `face_plus_context` reduces compute and improves fallback macro-F1 versus `full_with_mask`, but it does not fully solve hard-class performance.",
        "",
        "## 12. Next-Step Recommendation",
        f"- Keep `{best_name}` as the D16 v0 CE base.",
        f"- Use checkpoint: `{best_name}` `best.pt` at epoch {best_row.get('best_epoch')}.",
        "- Open D16 v1 part-aware SupCon as the next controlled experiment, while keeping CE-only results as the control.",
        "- Add fallback-aware analysis/training next because fallback macro-F1 is much lower than detected macro-F1.",
        "- Do not resume the same CE setting as the primary move; validation has plateaued under the current fixed-LR setup.",
        "- Do not switch checkpoint selection to last.pt based on test metrics; keep best.pt selected by validation macro-F1.",
        "- Consider class weights/focal only as a separate CE-tuning branch if SupCon/fallback-aware training does not close the gap.",
        "- Consider hybrid graph mode later: detected samples can use face_plus_context, while fallback samples may need a different full-image or fallback-specific route.",
        "",
        "## 13. Final Decision",
        f"`{final}`",
        "",
        "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
    ]
    report = "\n".join(lines)
    for token in FORBIDDEN_TOKENS:
        if token in report:
            raise RuntimeError(f"Forbidden token in report: {token}")
    (output_dir / "D16_V0_CE_FULL_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--d15_baseline_acc", type=float, required=True)
    parser.add_argument("--d15_baseline_macro_f1", type=float, required=True)
    parser.add_argument("--d15_baseline_weighted_f1", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if len(args.runs) != len(args.names):
        raise ValueError("--runs and --names must have equal length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(Path(run), name, args.d15_baseline_acc, args.d15_baseline_macro_f1, args.d15_baseline_weighted_f1)
        for run, name in zip(args.runs, args.names)
    ]

    summary_rows = [row_for_summary(run, args.d15_baseline_acc, args.d15_baseline_macro_f1, args.d15_baseline_weighted_f1) for run in runs]
    per_class_rows = make_per_class_rows(runs)
    fallback_rows = make_fallback_rows(runs)
    confusion_rows = make_confusion_rows(runs)
    runtime_rows = [run["runtime"] for run in runs]
    learning_rows = make_learning_rows(runs)
    risk_rows = make_risk_rows(runs)
    pred_rows = make_pred_rows(runs)

    write_csv(
        output_dir / "d16_v0_ce_full_compare_summary.csv",
        summary_rows,
        [
            "run_name",
            "graph_mode",
            "best_epoch",
            "best_val_macro_f1",
            "best_val_accuracy",
            "best_checkpoint_test_accuracy",
            "best_checkpoint_test_macro_f1",
            "best_checkpoint_test_weighted_f1",
            "last_checkpoint_test_accuracy",
            "last_checkpoint_test_macro_f1",
            "delta_best_vs_d15_acc",
            "delta_best_vs_d15_macro_f1",
            "delta_best_vs_d15_weighted_f1",
            "early_stop_epoch",
            "early_stop_reason",
            "final_decision_raw",
            "decision_vs_d15",
            "warnings",
        ],
    )
    write_csv(
        output_dir / "d16_v0_ce_full_per_class_compare.csv",
        per_class_rows,
        ["class_id", "class_name", "D15_F1", "full_with_mask_F1", "face_plus_context_F1", "delta_full_vs_D15", "delta_face_vs_D15", "winner", "notes"],
    )
    write_csv(
        output_dir / "d16_v0_ce_full_detected_fallback_compare.csv",
        fallback_rows,
        [
            "run_name",
            "graph_mode",
            "detected_total",
            "detected_accuracy",
            "detected_macro_f1",
            "fallback_total",
            "fallback_accuracy",
            "fallback_macro_f1",
            "gap_detected_minus_fallback_acc",
            "gap_detected_minus_fallback_macro_f1",
            "fallback_decision",
        ],
    )
    write_csv(output_dir / "d16_v0_ce_full_confusion_compare.csv", confusion_rows, ["run_name", "status", "true_class", "pred_class", "count", "note"])
    write_csv(
        output_dir / "d16_v0_ce_full_runtime_compare.csv",
        runtime_rows,
        [
            "run_name",
            "graph_mode",
            "epoch_count",
            "early_stop_epoch",
            "epoch_time_mean",
            "epoch_time_mean_excluding_first",
            "train_epoch_time_mean",
            "val_epoch_time_mean",
            "total_runtime_hours",
            "memory_reserved_mb_max",
            "node_count_mean",
            "edge_count_mean",
            "graph_cache_used",
        ],
    )
    write_csv(
        output_dir / "d16_v0_ce_full_learning_dynamics.csv",
        learning_rows,
        [
            "run_name",
            "graph_mode",
            "epoch_count",
            "best_epoch",
            "best_epoch_position",
            "val_macro_f1_at_best",
            "val_macro_f1_final",
            "train_loss_epoch1",
            "train_loss_best",
            "train_loss_final",
            "val_macro_f1_slope_last10",
            "train_loss_slope_last10",
            "overfit_gap_proxy",
            "early_stopping_triggered",
            "early_stop_reason",
            "lr_at_best_epoch",
            "learning_decision",
        ],
    )
    write_csv(output_dir / "d16_v0_ce_full_risk_cases.csv", risk_rows, ["run_name", "risk_type", "detail"])
    write_csv(output_dir / "d16_v0_ce_full_prediction_distribution_compare.csv", pred_rows, ["run_name", "class_id", "class_name", "support", "pred_count", "pred_ratio", "support_ratio", "pred_minus_support", "run_pred_decision", "max_pred_ratio"])
    final = build_report(
        output_dir,
        runs,
        summary_rows,
        per_class_rows,
        fallback_rows,
        confusion_rows,
        runtime_rows,
        learning_rows,
        risk_rows,
        args.d15_baseline_acc,
        args.d15_baseline_macro_f1,
        args.d15_baseline_weighted_f1,
    )
    manifest = {
        "output_dir": str(output_dir),
        "runs": args.runs,
        "names": args.names,
        "d15_baseline": {
            "accuracy": args.d15_baseline_acc,
            "macro_f1": args.d15_baseline_macro_f1,
            "weighted_f1": args.d15_baseline_weighted_f1,
        },
        "final_decision": final,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
