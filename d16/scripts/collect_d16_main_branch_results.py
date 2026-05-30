"""Collect D16R main-branch results against fixed accuracy-first anchors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
HARD_CLASS_IDS = {0, 2, 4, 6}
D15_ACC = 0.645026
D15_MACRO = 0.622471
A2_ACC = 0.618835
A2_MACRO = 0.600635
A3_ACC = 0.631652
A3_MACRO = 0.621532
A3_DETECTED_ACC = 0.645584
A3_DETECTED_MACRO = 0.633986
A3_HARD_MEAN = 0.524581
A4_ACC = 0.634717
A4_MACRO = 0.622718
A4_DETECTED_ACC = 0.647625
A4_DETECTED_MACRO = 0.635603
A4_HARD_MEAN = 0.535498

ANCHORS = [
    {
        "run_name": "D15 baseline",
        "test_accuracy": D15_ACC,
        "test_macro_f1": D15_MACRO,
        "detected_accuracy": "",
        "detected_macro_f1": "",
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue",
        "test_accuracy": 0.633881,
        "test_macro_f1": 0.623164,
        "detected_accuracy": 0.647042,
        "detected_macro_f1": 0.635443,
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "A1: d16r_part_attention_readout_ce_seed42",
        "test_accuracy": 0.614656,
        "test_macro_f1": 0.590668,
        "detected_accuracy": 0.628097,
        "detected_macro_f1": 0.601891,
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "D16 v1 original best observed",
        "test_accuracy": 0.639175,
        "test_macro_f1": 0.632938,
        "detected_accuracy": "",
        "detected_macro_f1": "",
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "A2: d16r_part_token_transformer_ce_seed42",
        "test_accuracy": A2_ACC,
        "test_macro_f1": A2_MACRO,
        "detected_accuracy": 0.632760,
        "detected_macro_f1": 0.612370,
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "A3: d16r_part_motif_query_ce_seed42",
        "test_accuracy": A3_ACC,
        "test_macro_f1": A3_MACRO,
        "detected_accuracy": A3_DETECTED_ACC,
        "detected_macro_f1": A3_DETECTED_MACRO,
        "predicted_classes": 7,
        "source": "anchor",
    },
    {
        "run_name": "A4: d16r_micro_motif_support_ce_seed42",
        "test_accuracy": A4_ACC,
        "test_macro_f1": A4_MACRO,
        "detected_accuracy": A4_DETECTED_ACC,
        "detected_macro_f1": A4_DETECTED_MACRO,
        "predicted_classes": 7,
        "source": "anchor",
    },
]
BEST_RESCUE_ACC = 0.633881
BEST_RESCUE_HARD_F1 = {
    0: 0.534737,
    2: 0.465553,
    4: 0.499613,
    6: 0.615020,
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def latest(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return rows[-1] if rows else {}


def group_row(rows: List[Dict[str, str]], group: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("group")) == group:
            return row
    return {}


def finite(value: Any) -> bool:
    return math.isfinite(as_float(value))


def hard_mean_from_rows(rows: List[Dict[str, Any]]) -> float:
    vals = [as_float(row.get("f1")) for row in rows if as_int(row.get("class_id")) in HARD_CLASS_IDS]
    vals = [value for value in vals if math.isfinite(value)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def is_a3_run(row: Dict[str, Any]) -> bool:
    name = str(row.get("run_name", ""))
    return "part_motif_query" in name


def is_a2_run(row: Dict[str, Any]) -> bool:
    name = str(row.get("run_name", ""))
    return "part_token_transformer" in name


def is_a4_run(row: Dict[str, Any]) -> bool:
    name = str(row.get("run_name", ""))
    return "micro_motif_support" in name


def is_a4b_run(row: Dict[str, Any]) -> bool:
    name = str(row.get("run_name", ""))
    return "micro_motif_support_no_global_micro" in name


def collect_run(run_dir: Path) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not run_dir.exists():
        return None, [], [], [f"missing run_dir: {run_dir}"]
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    pred_count = read_rows(run_dir / "pred_count.csv")
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    run_name = str(summary.get("run_name") or run_dir.name)
    required = [
        "checkpoints/best.pt",
        "test_metrics.csv",
        "per_class_metrics.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "d16_train_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        warnings.append(f"{run_name}: missing files: {', '.join(missing)}")
    detected = group_row(groups, "detected")
    fallback = group_row(groups, "fallback")
    predicted_classes = as_int(test.get("predicted_classes"))
    if pred_count:
        predicted_classes = sum(1 for row in pred_count if as_int(row.get("pred_count")) > 0)
    row = {
        "run_name": run_name,
        "test_accuracy": as_float(summary.get("test_accuracy", test.get("accuracy"))),
        "test_macro_f1": as_float(summary.get("test_macro_f1", test.get("macro_f1"))),
        "best_val_macro_f1": as_float(summary.get("best_val_macro_f1")),
        "best_epoch": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch") or test.get("epoch")),
        "detected_accuracy": as_float(detected.get("accuracy")),
        "detected_macro_f1": as_float(detected.get("macro_f1")),
        "fallback_accuracy": as_float(fallback.get("accuracy")),
        "fallback_macro_f1": as_float(fallback.get("macro_f1")),
        "predicted_classes": predicted_classes,
        "total": as_int(test.get("total") or summary.get("test_samples")),
        "output_dir": str(run_dir),
        "missing_files": ";".join(missing),
        "source": "run",
    }
    group_rows = [
        {
            "run_name": run_name,
            "group": item.get("group", ""),
            "total": as_int(item.get("total")),
            "accuracy": as_float(item.get("accuracy")),
            "macro_f1": as_float(item.get("macro_f1")),
        }
        for item in groups
    ]
    pred_by_class = {as_int(item.get("class_id")): as_int(item.get("pred_count")) for item in pred_count}
    hard_rows: List[Dict[str, Any]] = []
    for item in per_class:
        cid = as_int(item.get("class_id"))
        if cid not in HARD_CLASS_IDS:
            continue
        hard_rows.append(
            {
                "run_name": run_name,
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "support": as_int(item.get("support")),
                "pred_count": pred_by_class.get(cid, as_int(item.get("pred_count"))),
                "precision": as_float(item.get("precision")),
                "recall": as_float(item.get("recall")),
                "f1": as_float(item.get("f1")),
            }
        )
    return row, group_rows, hard_rows, warnings


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    value = as_float(value)
    return "" if not math.isfinite(value) else f"{value:.6f}"


def md_table(rows: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def decision(run_rows: List[Dict[str, Any]], warnings: List[str]) -> str:
    valid_rows = [row for row in run_rows if math.isfinite(as_float(row.get("test_accuracy")))]
    if not valid_rows:
        return "RUN_NOT_AVAILABLE"
    best = max(valid_rows, key=lambda row: as_float(row.get("test_accuracy")))
    acc = as_float(best.get("test_accuracy"))
    predicted_classes = as_int(best.get("predicted_classes"))
    if predicted_classes < 7:
        return "REJECT_RUN_COLLAPSE"
    if warnings and best.get("missing_files"):
        return "RUN_FAILED_NEEDS_DEBUG"
    run_dir = Path(str(best.get("output_dir", "")))
    if is_a4b_run(best):
        micro_warnings = _micro_motif_warnings(run_dir)
        acc = as_float(best.get("test_accuracy"))
        macro = as_float(best.get("test_macro_f1"))
        per_class = read_rows(run_dir / "per_class_metrics.csv")
        hard_mean = hard_mean_from_rows(per_class)
        if predicted_classes < 7:
            return "REJECT_COLLAPSE"
        if any("micro_noise" in item or "collapse" in item for item in micro_warnings):
            return "A4B_MICRO_COLLAPSE_OR_NOISE_RETHINK"
        if acc >= 0.650:
            return "STRONG_A4B_SIGNAL"
        if acc > D15_ACC:
            return "BEATS_D15_KEEP_A4B_AND_REPEAT"
        if acc > A4_ACC and macro >= A4_MACRO:
            return "A4B_IMPROVES_A4_DISABLE_GLOBAL_MICRO"
        if math.isfinite(hard_mean) and hard_mean >= A4_HARD_MEAN and acc <= A4_ACC:
            return "HARD_GAIN_NOT_ACCURACY_ROUTE"
        if acc > BEST_RESCUE_ACC:
            return "A4B_USEFUL_BUT_A4_STILL_BETTER"
        if acc <= A3_ACC:
            return "A4B_REMOVAL_HURTS_MOVE_TO_NODE_GNN_UPGRADE"
        return "A4B_WEAK_OR_INCONCLUSIVE"
    if is_a4_run(best):
        micro_warnings = _micro_motif_warnings(run_dir)
        acc = as_float(best.get("test_accuracy"))
        if any("micro_noise" in item or "collapse" in item for item in micro_warnings):
            return "A4_MICRO_NOISE_RISK_DISABLE_DETAIL_BIAS_OR_RETHINK"
        if micro_warnings:
            return "A4_MOTIF_COLLAPSE_NEEDS_SIMPLER_DESIGN"
        if acc >= 0.650:
            return "STRONG_A4_SIGNAL"
        if acc > D15_ACC:
            return "BEATS_D15_KEEP_A4_AND_REPEAT"
        if acc > BEST_RESCUE_ACC:
            return "A4_USEFUL_BUT_NOT_ENOUGH"
        if acc > A3_ACC:
            return "A4_WEAK_GAIN_NEEDS_ANALYSIS"
        per_class = read_rows(run_dir / "per_class_metrics.csv")
        hard_mean = hard_mean_from_rows(per_class)
        if math.isfinite(hard_mean) and hard_mean > A3_HARD_MEAN and acc <= A3_ACC:
            return "BALANCE_GAIN_NOT_ACCURACY_ROUTE"
        return "A4_NOT_ENOUGH_MOVE_TO_NODE_FEATURE_OR_GNN_UPGRADE"
    if is_a3_run(best):
        motif_warnings = _motif_collapse_warnings(run_dir)
        if motif_warnings:
            return "A3_MOTIF_COLLAPSE_NEEDS_K_OR_REG_FIX"
        if acc >= 0.650:
            return "STRONG_A3_SIGNAL"
        if acc > D15_ACC:
            return "BEATS_D15_KEEP_A3_AND_REPEAT"
        if acc > BEST_RESCUE_ACC:
            return "A3_USEFUL_BUT_NOT_ENOUGH"
        if acc > A2_ACC:
            return "A3_WEAK_GAIN_NEEDS_ANALYSIS"
        per_class = read_rows(run_dir / "per_class_metrics.csv")
        hard_mean = hard_mean_from_rows(per_class)
        if math.isfinite(hard_mean) and hard_mean > sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1):
            return "BALANCE_GAIN_NOT_ACCURACY_ROUTE"
        return "A3_NOT_ENOUGH_RETHINK_NODE_FEATURE_OR_MICRO_DETAIL"
    if acc >= 0.650:
        return "STRONG_MAIN_BRANCH_SIGNAL"
    if acc > D15_ACC:
        return "BEATS_D15_ACCURACY_KEEP_AND_REPEAT"
    if acc > BEST_RESCUE_ACC:
        return "MAIN_BRANCH_USEFUL_BUT_NOT_ENOUGH"

    per_class = read_rows(run_dir / "per_class_metrics.csv")
    hard_mean = hard_mean_from_rows(per_class)
    if math.isfinite(hard_mean) and hard_mean > sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1):
        return "BALANCE_GAIN_NOT_ACCURACY_ROUTE"
    return "A2_NOT_ENOUGH_MOVE_TO_A3_MOTIF_QUERY"


def _top_confusions(run_dir: Path, limit: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for row in read_rows(run_dir / "confusion_matrix.csv"):
        true_cls = as_int(row.get("true_class"))
        pred_cls = as_int(row.get("pred_class"))
        count = as_int(row.get("count"))
        if true_cls == pred_cls or count <= 0:
            continue
        rows.append(
            {
                "true": true_cls,
                "predicted": pred_cls,
                "count": count,
                "support": as_int(row.get("support")),
                "row_ratio": as_float(row.get("row_ratio")),
            }
        )
    return sorted(rows, key=lambda row: as_int(row.get("count")), reverse=True)[:limit]


def _prediction_distribution(run_dir: Path) -> List[Dict[str, Any]]:
    rows = read_rows(run_dir / "pred_count.csv")
    total = sum(as_int(row.get("pred_count")) for row in rows)
    return [
        {
            "class": as_int(row.get("class_id")),
            "pred_count": as_int(row.get("pred_count")),
            "pred_ratio": as_int(row.get("pred_count")) / total if total > 0 else float("nan"),
        }
        for row in rows
    ]


def _part_token_rows(run_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "part": row.get("part_name"),
            "token_norm_mean": as_float(row.get("token_norm_mean")),
            "transformed_token_norm_mean": as_float(row.get("transformed_token_norm_mean")),
            "valid_samples": as_int(row.get("valid_samples")),
        }
        for row in read_rows(run_dir / "part_token_transformer_summary.csv")
    ]


def _part_motif_rows(run_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "motif": row.get("motif_name"),
            "part": row.get("part_name"),
            "usage": as_float(row.get("motif_usage_mean")),
            "entropy": as_float(row.get("motif_attention_entropy_mean")),
            "peak": as_float(row.get("motif_attention_peak_mean")),
            "part_mass": as_float(row.get("motif_part_mass_mean")),
            "token_norm": as_float(row.get("motif_token_norm_mean")),
            "transformed_norm": as_float(row.get("motif_transformed_token_norm_mean")),
            "samples": as_int(row.get("samples")),
            "effective_motif_count": as_float(row.get("effective_motif_count_mean")),
            "avg_offdiag_similarity": as_float(row.get("avg_offdiag_similarity_mean")),
        }
        for row in read_rows(run_dir / "part_motif_summary.csv")
    ]


def _motif_collapse_warnings(run_dir: Path) -> List[str]:
    rows = _part_motif_rows(run_dir)
    if not rows:
        return []
    warnings: List[str] = []
    offdiag = rows[0].get("avg_offdiag_similarity")
    effective = rows[0].get("effective_motif_count")
    if math.isfinite(as_float(offdiag)) and as_float(offdiag) > 0.90:
        warnings.append(f"avg_offdiag_similarity={as_float(offdiag):.6f} > 0.90")
    if math.isfinite(as_float(effective)) and as_float(effective) < 2.0:
        warnings.append(f"effective_motif_count={as_float(effective):.6f} < 2")
    usage_by_part: Dict[str, List[float]] = {}
    for row in rows:
        part = str(row.get("part", ""))
        usage_by_part.setdefault(part, []).append(as_float(row.get("usage")))
        mass = as_float(row.get("part_mass"))
        if part != "global" and math.isfinite(mass) and mass < 0.20:
            warnings.append(f"{row.get('motif')} part_mass={mass:.6f} < 0.20")
        peak = as_float(row.get("peak"))
        if math.isfinite(peak) and peak > 0.90:
            warnings.append(f"{row.get('motif')} peak={peak:.6f} > 0.90")
    for part, values in usage_by_part.items():
        finite_values = [value for value in values if math.isfinite(value)]
        total = sum(finite_values)
        if total > 0.0 and len(finite_values) > 1 and max(finite_values) / total > 0.80:
            warnings.append(f"one motif dominates {part} usage")
    return warnings


def _micro_motif_rows(run_dir: Path) -> List[Dict[str, Any]]:
    return [
        {
            "branch": row.get("branch"),
            "motif": row.get("motif_name"),
            "part": row.get("part_name"),
            "usage": as_float(row.get("motif_usage_mean")),
            "entropy": as_float(row.get("motif_attention_entropy_mean")),
            "peak": as_float(row.get("motif_attention_peak_mean")),
            "part_mass": as_float(row.get("motif_part_mass_mean")),
            "detail_score": as_float(row.get("micro_detail_score_mean")),
            "token_norm": as_float(row.get("motif_token_norm_mean")),
            "transformed_norm": as_float(row.get("motif_transformed_token_norm_mean")),
            "samples": as_int(row.get("samples")),
            "effective_motif_count": as_float(row.get("effective_motif_count_mean")),
            "avg_offdiag_similarity": as_float(row.get("avg_offdiag_similarity_mean")),
            "micro_gate_mean": as_float(row.get("micro_gate_mean")),
            "detail_available_ratio": as_float(row.get("detail_available_ratio")),
        }
        for row in read_rows(run_dir / "micro_motif_summary.csv")
    ]


def _micro_motif_warnings(run_dir: Path) -> List[str]:
    rows = _micro_motif_rows(run_dir)
    micro_rows = [row for row in rows if row.get("branch") == "micro"]
    if not micro_rows:
        return []
    warnings: List[str] = []
    offdiag = as_float(micro_rows[0].get("avg_offdiag_similarity"))
    effective = as_float(micro_rows[0].get("effective_motif_count"))
    gate = as_float(micro_rows[0].get("micro_gate_mean"))
    if math.isfinite(offdiag) and offdiag > 0.90:
        warnings.append(f"collapse: avg_micro_offdiag_similarity={offdiag:.6f} > 0.90")
    if math.isfinite(effective) and effective < 2.0:
        warnings.append(f"collapse: effective_micro_motif_count={effective:.6f} < 2")
    if math.isfinite(gate) and gate > 0.90:
        warnings.append(f"micro_noise: micro_gate_mean={gate:.6f} near 1")
    if math.isfinite(gate) and gate < 0.05:
        warnings.append(f"micro_noise: micro_gate_mean={gate:.6f} near 0")
    usage_by_part: Dict[str, List[float]] = {}
    for row in micro_rows:
        part = str(row.get("part", ""))
        usage_by_part.setdefault(part, []).append(as_float(row.get("usage")))
        mass = as_float(row.get("part_mass"))
        if part != "global" and math.isfinite(mass) and mass < 0.20:
            warnings.append(f"micro_noise: {row.get('motif')} part_mass={mass:.6f} < 0.20")
        peak = as_float(row.get("peak"))
        entropy = as_float(row.get("entropy"))
        detail = as_float(row.get("detail_score"))
        if math.isfinite(peak) and peak > 0.90:
            warnings.append(f"micro_noise: {row.get('motif')} peak={peak:.6f} > 0.90")
        if math.isfinite(entropy) and entropy > 7.5:
            warnings.append(f"micro_noise: {row.get('motif')} entropy={entropy:.6f} very high")
        if math.isfinite(detail) and abs(detail) > 5.0:
            warnings.append(f"micro_noise: {row.get('motif')} detail_score={detail:.6f} abnormal")
    for part, values in usage_by_part.items():
        finite_values = [value for value in values if math.isfinite(value)]
        total = sum(finite_values)
        if total > 0.0 and len(finite_values) > 1 and max(finite_values) / total > 0.80:
            warnings.append(f"collapse: one micro motif dominates {part} usage")
    return warnings


def _a3_detailed_report(run_rows: List[Dict[str, Any]], hard_rows: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    a3_rows = [row for row in run_rows if is_a3_run(row)]
    if not a3_rows:
        return []
    run = max(a3_rows, key=lambda row: as_float(row.get("test_accuracy")))
    run_dir = Path(str(run.get("output_dir")))
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    last = latest(read_rows(run_dir / "last_test_metrics.csv"))
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    hard_for_run = [row for row in hard_rows if row.get("run_name") == run.get("run_name")]
    hard_mean = hard_mean_from_rows(hard_for_run)
    best_rescue_hard_mean = sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1)
    motif_rows = _part_motif_rows(run_dir)
    motif_warnings = _motif_collapse_warnings(run_dir)
    dec = decision([run], warnings)
    accuracy_rows = [
        {"run": "D15 baseline", "accuracy": D15_ACC, "macro_f1": D15_MACRO, "A3_minus_anchor_acc": as_float(run.get("test_accuracy")) - D15_ACC, "A3_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - D15_MACRO},
        {"run": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue", "accuracy": BEST_RESCUE_ACC, "macro_f1": 0.623164, "A3_minus_anchor_acc": as_float(run.get("test_accuracy")) - BEST_RESCUE_ACC, "A3_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.623164},
        {"run": "A1: d16r_part_attention_readout_ce_seed42", "accuracy": 0.614656, "macro_f1": 0.590668, "A3_minus_anchor_acc": as_float(run.get("test_accuracy")) - 0.614656, "A3_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.590668},
        {"run": "A2: d16r_part_token_transformer_ce_seed42", "accuracy": A2_ACC, "macro_f1": A2_MACRO, "A3_minus_anchor_acc": as_float(run.get("test_accuracy")) - A2_ACC, "A3_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - A2_MACRO},
        {"run": str(run.get("run_name")), "accuracy": as_float(run.get("test_accuracy")), "macro_f1": as_float(run.get("test_macro_f1")), "A3_minus_anchor_acc": 0.0, "A3_minus_anchor_macro_f1": 0.0},
    ]
    best_last_rows = [
        {"checkpoint": "best.pt", "epoch": as_int(test.get("checkpoint_epoch") or test.get("epoch")), "accuracy": as_float(test.get("accuracy")), "macro_f1": as_float(test.get("macro_f1")), "loss": as_float(test.get("loss")), "detected_loss": as_float(test.get("detected_loss_mean")), "fallback_loss": as_float(test.get("fallback_loss_mean"))},
        {"checkpoint": "last.pt", "epoch": as_int(last.get("checkpoint_epoch") or last.get("epoch")), "accuracy": as_float(last.get("accuracy")), "macro_f1": as_float(last.get("macro_f1")), "loss": as_float(last.get("loss")), "detected_loss": as_float(last.get("detected_loss_mean")), "fallback_loss": as_float(last.get("fallback_loss_mean"))},
    ]
    group_rows = [
        {"group": row.get("group"), "total": as_int(row.get("total")), "accuracy": as_float(row.get("accuracy")), "macro_f1": as_float(row.get("macro_f1"))}
        for row in groups
    ]
    class_rows = [
        {"class": CLASS_NAMES.get(as_int(row.get("class_id")), str(row.get("class_id"))), "support": as_int(row.get("support")), "pred_count": as_int(row.get("pred_count")), "precision": as_float(row.get("precision")), "recall": as_float(row.get("recall")), "f1": as_float(row.get("f1"))}
        for row in per_class
    ]
    hard_compare = [
        {"class": row.get("class_name"), "A3_f1": as_float(row.get("f1")), "best_rescue_f1": BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")), "delta": as_float(row.get("f1")) - BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan"))}
        for row in hard_for_run
    ]
    lines = [
        "# D16R-A3 Part-Motif Query Analysis",
        "",
        "## Verdict",
        f"`{dec}`",
        "",
        "D16R-A3 uses MediaPipe-guided part-conditioned latent motif queries over pixel-GNN node embeddings. These are learned readout patterns, not semantic motifs, evidence, or causal explanations.",
        "",
        "## Run Integrity",
        *md_table(
            [
                {"item": "motif diagnostics", "value": "PASS" if motif_rows else "NOT_AVAILABLE"},
                {"item": "motif collapse warnings", "value": len(motif_warnings)},
                {"item": "predicted classes", "value": as_int(run.get("predicted_classes"))},
                {"item": "best epoch", "value": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch"))},
                {"item": "final trained epoch", "value": as_int(last.get("checkpoint_epoch") or last.get("epoch"))},
                {"item": "train samples", "value": as_int(summary.get("train_samples"))},
                {"item": "val samples", "value": as_int(summary.get("val_samples"))},
                {"item": "test samples", "value": as_int(summary.get("test_samples") or test.get("total"))},
                {"item": "device", "value": summary.get("device", "")},
            ],
            ["item", "value"],
        ),
        "",
        "## Accuracy-First Anchor Comparison",
        *md_table(accuracy_rows, ["run", "accuracy", "macro_f1", "A3_minus_anchor_acc", "A3_minus_anchor_macro_f1"]),
        "",
        "## Best vs Last Checkpoint",
        *md_table(best_last_rows, ["checkpoint", "epoch", "accuracy", "macro_f1", "loss", "detected_loss", "fallback_loss"]),
        "",
        "## Detected vs Fallback",
        *md_table(group_rows, ["group", "total", "accuracy", "macro_f1"]),
        "",
        "## Per-Class Metrics",
        *md_table(class_rows, ["class", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Hard-Class Comparison",
        f"Hard-class mean A3: `{fmt(hard_mean)}`; best rescue hard-class mean: `{fmt(best_rescue_hard_mean)}`.",
        *md_table(hard_compare, ["class", "A3_f1", "best_rescue_f1", "delta"]),
        "",
        "## Top Confusions",
        *md_table(_top_confusions(run_dir), ["true", "predicted", "count", "support", "row_ratio"]),
        "",
        "## Prediction Distribution",
        *md_table(_prediction_distribution(run_dir), ["class", "pred_count", "pred_ratio"]),
        "",
        "## Motif Diagnostics",
        *md_table(motif_rows, ["motif", "part", "usage", "entropy", "peak", "part_mass", "effective_motif_count", "avg_offdiag_similarity"]),
        "",
        "## Motif Collapse Check",
    ]
    lines.extend([f"- {item}" for item in motif_warnings] if motif_warnings else ["- no collapse warning from aggregate heuristics"])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    return lines


def _a4_detailed_report(run_rows: List[Dict[str, Any]], hard_rows: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    a4_rows = [row for row in run_rows if is_a4_run(row) and not is_a4b_run(row)]
    if not a4_rows:
        return []
    run = max(a4_rows, key=lambda row: as_float(row.get("test_accuracy")))
    run_dir = Path(str(run.get("output_dir")))
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    last = latest(read_rows(run_dir / "last_test_metrics.csv"))
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    hard_for_run = [row for row in hard_rows if row.get("run_name") == run.get("run_name")]
    hard_mean = hard_mean_from_rows(hard_for_run)
    best_rescue_hard_mean = sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1)
    micro_rows = _micro_motif_rows(run_dir)
    major_rows = [row for row in micro_rows if row.get("branch") == "major"]
    support_rows = [row for row in micro_rows if row.get("branch") == "micro"]
    micro_warnings = _micro_motif_warnings(run_dir)
    dec = decision([run], warnings)
    accuracy_rows = [
        {"run": "D15 baseline", "accuracy": D15_ACC, "macro_f1": D15_MACRO, "A4_minus_anchor_acc": as_float(run.get("test_accuracy")) - D15_ACC, "A4_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - D15_MACRO},
        {"run": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue", "accuracy": BEST_RESCUE_ACC, "macro_f1": 0.623164, "A4_minus_anchor_acc": as_float(run.get("test_accuracy")) - BEST_RESCUE_ACC, "A4_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.623164},
        {"run": "A1: d16r_part_attention_readout_ce_seed42", "accuracy": 0.614656, "macro_f1": 0.590668, "A4_minus_anchor_acc": as_float(run.get("test_accuracy")) - 0.614656, "A4_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.590668},
        {"run": "A2: d16r_part_token_transformer_ce_seed42", "accuracy": A2_ACC, "macro_f1": A2_MACRO, "A4_minus_anchor_acc": as_float(run.get("test_accuracy")) - A2_ACC, "A4_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - A2_MACRO},
        {"run": "A3: d16r_part_motif_query_ce_seed42", "accuracy": A3_ACC, "macro_f1": A3_MACRO, "A4_minus_anchor_acc": as_float(run.get("test_accuracy")) - A3_ACC, "A4_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - A3_MACRO},
        {"run": str(run.get("run_name")), "accuracy": as_float(run.get("test_accuracy")), "macro_f1": as_float(run.get("test_macro_f1")), "A4_minus_anchor_acc": 0.0, "A4_minus_anchor_macro_f1": 0.0},
    ]
    best_last_rows = [
        {"checkpoint": "best.pt", "epoch": as_int(test.get("checkpoint_epoch") or test.get("epoch")), "accuracy": as_float(test.get("accuracy")), "macro_f1": as_float(test.get("macro_f1")), "loss": as_float(test.get("loss")), "detected_loss": as_float(test.get("detected_loss_mean")), "fallback_loss": as_float(test.get("fallback_loss_mean"))},
        {"checkpoint": "last.pt", "epoch": as_int(last.get("checkpoint_epoch") or last.get("epoch")), "accuracy": as_float(last.get("accuracy")), "macro_f1": as_float(last.get("macro_f1")), "loss": as_float(last.get("loss")), "detected_loss": as_float(last.get("detected_loss_mean")), "fallback_loss": as_float(last.get("fallback_loss_mean"))},
    ]
    group_rows = [
        {"group": row.get("group"), "total": as_int(row.get("total")), "accuracy": as_float(row.get("accuracy")), "macro_f1": as_float(row.get("macro_f1"))}
        for row in groups
    ]
    class_rows = [
        {"class": CLASS_NAMES.get(as_int(row.get("class_id")), str(row.get("class_id"))), "support": as_int(row.get("support")), "pred_count": as_int(row.get("pred_count")), "precision": as_float(row.get("precision")), "recall": as_float(row.get("recall")), "f1": as_float(row.get("f1"))}
        for row in per_class
    ]
    hard_compare = [
        {
            "class": row.get("class_name"),
            "A4_f1": as_float(row.get("f1")),
            "A3_f1_or_mean_ref": A3_HARD_MEAN if as_int(row.get("class_id")) not in BEST_RESCUE_HARD_F1 else "",
            "best_rescue_f1": BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
            "delta_vs_best_rescue": as_float(row.get("f1")) - BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
        }
        for row in hard_for_run
    ]
    gate_rows = []
    if support_rows:
        gate_rows.append(
            {
                "micro_gate_mean": support_rows[0].get("micro_gate_mean"),
                "detail_available_ratio": support_rows[0].get("detail_available_ratio"),
                "effective_micro_motif_count": support_rows[0].get("effective_motif_count"),
                "avg_micro_offdiag_similarity": support_rows[0].get("avg_offdiag_similarity"),
            }
        )
    lines = [
        "# D16R-A4 Micro-Motif Support Analysis",
        "",
        "## Verdict",
        f"`{dec}`",
        "",
        "D16R-A4 adds weak part-gated micro-detail support tokens to A3-style major part-conditioned motif queries. These are learned readout/support tokens, not semantic motifs, evidence, or causal explanations.",
        "",
        "## Run Integrity",
        *md_table(
            [
                {"item": "micro diagnostics", "value": "PASS" if micro_rows else "NOT_AVAILABLE"},
                {"item": "micro/collapse warnings", "value": len(micro_warnings)},
                {"item": "predicted classes", "value": as_int(run.get("predicted_classes"))},
                {"item": "best epoch", "value": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch"))},
                {"item": "final trained epoch", "value": as_int(last.get("checkpoint_epoch") or last.get("epoch"))},
                {"item": "train samples", "value": as_int(summary.get("train_samples"))},
                {"item": "val samples", "value": as_int(summary.get("val_samples"))},
                {"item": "test samples", "value": as_int(summary.get("test_samples") or test.get("total"))},
                {"item": "device", "value": summary.get("device", "")},
            ],
            ["item", "value"],
        ),
        "",
        "## Accuracy-First Anchor Comparison",
        *md_table(accuracy_rows, ["run", "accuracy", "macro_f1", "A4_minus_anchor_acc", "A4_minus_anchor_macro_f1"]),
        "",
        "## Best vs Last Checkpoint",
        *md_table(best_last_rows, ["checkpoint", "epoch", "accuracy", "macro_f1", "loss", "detected_loss", "fallback_loss"]),
        "",
        "## Detected vs Fallback",
        *md_table(group_rows, ["group", "total", "accuracy", "macro_f1"]),
        "",
        "## Per-Class Metrics",
        *md_table(class_rows, ["class", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Hard-Class Comparison",
        f"Hard-class mean A4: `{fmt(hard_mean)}`; A3 hard-class mean: `{fmt(A3_HARD_MEAN)}`; best rescue hard-class mean: `{fmt(best_rescue_hard_mean)}`.",
        *md_table(hard_compare, ["class", "A4_f1", "best_rescue_f1", "delta_vs_best_rescue"]),
        "",
        "## Top Confusions",
        *md_table(_top_confusions(run_dir), ["true", "predicted", "count", "support", "row_ratio"]),
        "",
        "## Prediction Distribution",
        *md_table(_prediction_distribution(run_dir), ["class", "pred_count", "pred_ratio"]),
        "",
        "## Major Motif Diagnostics",
        *md_table(major_rows, ["motif", "part", "usage", "entropy", "peak", "part_mass", "effective_motif_count", "avg_offdiag_similarity"]),
        "",
        "## Micro Motif Diagnostics",
        *md_table(support_rows, ["motif", "part", "usage", "entropy", "peak", "part_mass", "detail_score", "effective_motif_count", "avg_offdiag_similarity"]),
        "",
        "## Micro Support Gate Diagnostics",
        *md_table(gate_rows, ["micro_gate_mean", "detail_available_ratio", "effective_micro_motif_count", "avg_micro_offdiag_similarity"]),
        "",
        "## Collapse / Noise Check",
    ]
    lines.extend([f"- {item}" for item in micro_warnings] if micro_warnings else ["- no collapse/noise warning from aggregate heuristics"])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    return lines


def _a4b_detailed_report(run_rows: List[Dict[str, Any]], hard_rows: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    a4b_rows = [row for row in run_rows if is_a4b_run(row)]
    if not a4b_rows:
        return []
    run = max(a4b_rows, key=lambda row: as_float(row.get("test_accuracy")))
    run_dir = Path(str(run.get("output_dir")))
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    last = latest(read_rows(run_dir / "last_test_metrics.csv"))
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    hard_for_run = [row for row in hard_rows if row.get("run_name") == run.get("run_name")]
    hard_mean = hard_mean_from_rows(hard_for_run)
    best_rescue_hard_mean = sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1)
    micro_rows = _micro_motif_rows(run_dir)
    major_rows = [row for row in micro_rows if row.get("branch") == "major"]
    support_rows = [row for row in micro_rows if row.get("branch") == "micro"]
    micro_warnings = _micro_motif_warnings(run_dir)
    dec = decision([run], warnings)
    global_micro_present = any(str(row.get("motif", "")).startswith("global_micro") for row in support_rows)
    accuracy_rows = [
        {"run": "D15 baseline", "accuracy": D15_ACC, "macro_f1": D15_MACRO, "A4b_minus_anchor_acc": as_float(run.get("test_accuracy")) - D15_ACC, "A4b_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - D15_MACRO},
        {"run": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue", "accuracy": BEST_RESCUE_ACC, "macro_f1": 0.623164, "A4b_minus_anchor_acc": as_float(run.get("test_accuracy")) - BEST_RESCUE_ACC, "A4b_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.623164},
        {"run": "A3: d16r_part_motif_query_ce_seed42", "accuracy": A3_ACC, "macro_f1": A3_MACRO, "A4b_minus_anchor_acc": as_float(run.get("test_accuracy")) - A3_ACC, "A4b_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - A3_MACRO},
        {"run": "A4: d16r_micro_motif_support_ce_seed42", "accuracy": A4_ACC, "macro_f1": A4_MACRO, "A4b_minus_anchor_acc": as_float(run.get("test_accuracy")) - A4_ACC, "A4b_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - A4_MACRO},
        {"run": str(run.get("run_name")), "accuracy": as_float(run.get("test_accuracy")), "macro_f1": as_float(run.get("test_macro_f1")), "A4b_minus_anchor_acc": 0.0, "A4b_minus_anchor_macro_f1": 0.0},
    ]
    best_last_rows = [
        {"checkpoint": "best.pt", "epoch": as_int(test.get("checkpoint_epoch") or test.get("epoch")), "accuracy": as_float(test.get("accuracy")), "macro_f1": as_float(test.get("macro_f1")), "loss": as_float(test.get("loss")), "detected_loss": as_float(test.get("detected_loss_mean")), "fallback_loss": as_float(test.get("fallback_loss_mean"))},
        {"checkpoint": "last.pt", "epoch": as_int(last.get("checkpoint_epoch") or last.get("epoch")), "accuracy": as_float(last.get("accuracy")), "macro_f1": as_float(last.get("macro_f1")), "loss": as_float(last.get("loss")), "detected_loss": as_float(last.get("detected_loss_mean")), "fallback_loss": as_float(last.get("fallback_loss_mean"))},
    ]
    group_rows = [
        {"group": row.get("group"), "total": as_int(row.get("total")), "accuracy": as_float(row.get("accuracy")), "macro_f1": as_float(row.get("macro_f1"))}
        for row in groups
    ]
    class_rows = [
        {"class": CLASS_NAMES.get(as_int(row.get("class_id")), str(row.get("class_id"))), "support": as_int(row.get("support")), "pred_count": as_int(row.get("pred_count")), "precision": as_float(row.get("precision")), "recall": as_float(row.get("recall")), "f1": as_float(row.get("f1"))}
        for row in per_class
    ]
    hard_compare = [
        {
            "class": row.get("class_name"),
            "A4b_f1": as_float(row.get("f1")),
            "best_rescue_f1": BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
            "A4b_minus_best_rescue": as_float(row.get("f1")) - BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
        }
        for row in hard_for_run
    ]
    target_rows = [
        {
            "check": "micro token count",
            "expected": 7,
            "observed": len(support_rows),
            "status": "PASS" if len(support_rows) == 7 else "FAIL",
        },
        {
            "check": "global_micro_0 removed",
            "expected": "absent",
            "observed": "present" if global_micro_present else "absent",
            "status": "PASS" if not global_micro_present else "FAIL",
        },
        {
            "check": "A4b minus A4 accuracy",
            "expected": "> 0 for improvement",
            "observed": as_float(run.get("test_accuracy")) - A4_ACC,
            "status": "",
        },
        {
            "check": "A4b minus A4 macro_f1",
            "expected": ">= 0 for clean win",
            "observed": as_float(run.get("test_macro_f1")) - A4_MACRO,
            "status": "",
        },
        {
            "check": "A4b minus A4 hard mean",
            "expected": ">= 0 for hard-class gain",
            "observed": hard_mean - A4_HARD_MEAN,
            "status": "",
        },
    ]
    gate_rows = []
    if support_rows:
        gate_rows.append(
            {
                "micro_gate_mean": support_rows[0].get("micro_gate_mean"),
                "detail_available_ratio": support_rows[0].get("detail_available_ratio"),
                "effective_micro_motif_count": support_rows[0].get("effective_motif_count"),
                "avg_micro_offdiag_similarity": support_rows[0].get("avg_offdiag_similarity"),
            }
        )
    lines = [
        "# D16R-A4b No-Global-Micro Analysis",
        "",
        "## Verdict",
        f"`{dec}`",
        "",
        "D16R-A4b is a targeted ablation of A4: the only intended recipe change is removing the near-uniform global micro support token. These tokens are learned readout/support components, not semantic motifs, evidence, or causal explanations.",
        "",
        "## Run Integrity",
        *md_table(
            [
                {"item": "micro diagnostics", "value": "PASS" if micro_rows else "NOT_AVAILABLE"},
                {"item": "micro/collapse warnings", "value": len(micro_warnings)},
                {"item": "micro support token count", "value": len(support_rows)},
                {"item": "global micro present", "value": str(global_micro_present)},
                {"item": "predicted classes", "value": as_int(run.get("predicted_classes"))},
                {"item": "best epoch", "value": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch"))},
                {"item": "final trained epoch", "value": as_int(last.get("checkpoint_epoch") or last.get("epoch"))},
                {"item": "train samples", "value": as_int(summary.get("train_samples"))},
                {"item": "val samples", "value": as_int(summary.get("val_samples"))},
                {"item": "test samples", "value": as_int(summary.get("test_samples") or test.get("total"))},
                {"item": "device", "value": summary.get("device", "")},
            ],
            ["item", "value"],
        ),
        "",
        "## Accuracy-First Anchor Comparison",
        *md_table(accuracy_rows, ["run", "accuracy", "macro_f1", "A4b_minus_anchor_acc", "A4b_minus_anchor_macro_f1"]),
        "",
        "## Best vs Last Checkpoint",
        *md_table(best_last_rows, ["checkpoint", "epoch", "accuracy", "macro_f1", "loss", "detected_loss", "fallback_loss"]),
        "",
        "## Detected vs Fallback",
        *md_table(group_rows, ["group", "total", "accuracy", "macro_f1"]),
        "",
        "## Per-Class Metrics",
        *md_table(class_rows, ["class", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Hard-Class Comparison",
        f"Hard-class mean A4b: `{fmt(hard_mean)}`; A4 hard-class mean: `{fmt(A4_HARD_MEAN)}`; best rescue hard-class mean: `{fmt(best_rescue_hard_mean)}`.",
        *md_table(hard_compare, ["class", "A4b_f1", "best_rescue_f1", "A4b_minus_best_rescue"]),
        "",
        "## Top Confusions",
        *md_table(_top_confusions(run_dir), ["true", "predicted", "count", "support", "row_ratio"]),
        "",
        "## Prediction Distribution",
        *md_table(_prediction_distribution(run_dir), ["class", "pred_count", "pred_ratio"]),
        "",
        "## Major Motif Diagnostics",
        *md_table(major_rows, ["motif", "part", "usage", "entropy", "peak", "part_mass", "effective_motif_count", "avg_offdiag_similarity"]),
        "",
        "## Micro Motif Diagnostics",
        *md_table(support_rows, ["motif", "part", "usage", "entropy", "peak", "part_mass", "detail_score", "effective_motif_count", "avg_offdiag_similarity"]),
        "",
        "## Micro Support Gate Diagnostics",
        *md_table(gate_rows, ["micro_gate_mean", "detail_available_ratio", "effective_micro_motif_count", "avg_micro_offdiag_similarity"]),
        "",
        "## A4 vs A4b Targeted Ablation",
        *md_table(target_rows, ["check", "expected", "observed", "status"]),
        "",
        "## Whether Global Micro Removal Helped",
        "Use the table above as the primary targeted-ablation answer: a clean A4b win requires accuracy above A4 and macro-F1 at least tied with A4, with 7 micro tokens and no `global_micro_0` diagnostics.",
        "",
        "## Collapse / Noise Check",
    ]
    lines.extend([f"- {item}" for item in micro_warnings] if micro_warnings else ["- no collapse/noise warning from aggregate heuristics"])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    return lines


def _a2_detailed_report(run_rows: List[Dict[str, Any]], hard_rows: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    a2_rows = [row for row in run_rows if is_a2_run(row)]
    if not a2_rows:
        return []
    run = max(a2_rows, key=lambda row: as_float(row.get("test_accuracy")))
    run_dir = Path(str(run.get("output_dir")))
    summary = read_json(run_dir / "d16_train_summary.json")
    test = latest(read_rows(run_dir / "test_metrics.csv"))
    last = latest(read_rows(run_dir / "last_test_metrics.csv"))
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    hard_for_run = [row for row in hard_rows if row.get("run_name") == run.get("run_name")]
    hard_mean = hard_mean_from_rows(hard_for_run)
    best_rescue_hard_mean = sum(BEST_RESCUE_HARD_F1.values()) / len(BEST_RESCUE_HARD_F1)
    dec = decision(run_rows, warnings)
    required = [
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "per_class_metrics.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "pred_count.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "d16_train_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    checker_decision = "D16_MAIN_BRANCH_CHECK_PASS" if not missing and as_int(run.get("predicted_classes")) == 7 else "D16_MAIN_BRANCH_CHECK_NOT_PASS"
    diag_status = "PASS" if read_rows(run_dir / "part_token_transformer_summary.csv") else "NOT_AVAILABLE"
    accuracy_rows = [
        {
            "run": "D15 baseline",
            "accuracy": D15_ACC,
            "macro_f1": D15_MACRO,
            "A2_minus_anchor_acc": as_float(run.get("test_accuracy")) - D15_ACC,
            "A2_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - D15_MACRO,
        },
        {
            "run": "best rescue: d16_v4_grid8_ce_seed42_pixel_rescue",
            "accuracy": 0.633881,
            "macro_f1": 0.623164,
            "A2_minus_anchor_acc": as_float(run.get("test_accuracy")) - 0.633881,
            "A2_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.623164,
        },
        {
            "run": "A1: d16r_part_attention_readout_ce_seed42",
            "accuracy": 0.614656,
            "macro_f1": 0.590668,
            "A2_minus_anchor_acc": as_float(run.get("test_accuracy")) - 0.614656,
            "A2_minus_anchor_macro_f1": as_float(run.get("test_macro_f1")) - 0.590668,
        },
        {
            "run": str(run.get("run_name")),
            "accuracy": as_float(run.get("test_accuracy")),
            "macro_f1": as_float(run.get("test_macro_f1")),
            "A2_minus_anchor_acc": 0.0,
            "A2_minus_anchor_macro_f1": 0.0,
        },
    ]
    best_last_rows = [
        {
            "checkpoint": "best.pt",
            "epoch": as_int(test.get("checkpoint_epoch") or test.get("epoch")),
            "accuracy": as_float(test.get("accuracy")),
            "macro_f1": as_float(test.get("macro_f1")),
            "loss": as_float(test.get("loss")),
            "detected_loss": as_float(test.get("detected_loss_mean")),
            "fallback_loss": as_float(test.get("fallback_loss_mean")),
        },
        {
            "checkpoint": "last.pt",
            "epoch": as_int(last.get("checkpoint_epoch") or last.get("epoch")),
            "accuracy": as_float(last.get("accuracy")),
            "macro_f1": as_float(last.get("macro_f1")),
            "loss": as_float(last.get("loss")),
            "detected_loss": as_float(last.get("detected_loss_mean")),
            "fallback_loss": as_float(last.get("fallback_loss_mean")),
        },
    ]
    group_rows = [
        {
            "group": row.get("group"),
            "total": as_int(row.get("total")),
            "accuracy": as_float(row.get("accuracy")),
            "macro_f1": as_float(row.get("macro_f1")),
            "delta_acc_vs_best_rescue": as_float(row.get("accuracy")) - (0.647042 if row.get("group") == "detected" else float("nan")),
            "delta_macro_f1_vs_best_rescue": as_float(row.get("macro_f1")) - (0.635443 if row.get("group") == "detected" else float("nan")),
        }
        for row in groups
    ]
    class_rows = [
        {
            "class": CLASS_NAMES.get(as_int(row.get("class_id")), str(row.get("class_id"))),
            "support": as_int(row.get("support")),
            "pred_count": as_int(row.get("pred_count")),
            "precision": as_float(row.get("precision")),
            "recall": as_float(row.get("recall")),
            "f1": as_float(row.get("f1")),
        }
        for row in per_class
    ]
    hard_compare = [
        {
            "class": row.get("class_name"),
            "A2_f1": as_float(row.get("f1")),
            "best_rescue_f1": BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
            "delta": as_float(row.get("f1")) - BEST_RESCUE_HARD_F1.get(as_int(row.get("class_id")), float("nan")),
        }
        for row in hard_for_run
    ]
    lines = [
        "# D16R-A2 Part-token Transformer Analysis",
        "",
        "## Verdict",
        f"`{dec}`",
        "",
        "D16R-A2 keeps all five part tokens and uses a compact Transformer readout plus residual concat. This report does not make motif, causal-evidence, semantic-region, or interpretability claims.",
        "",
        "## Run Integrity",
        *md_table(
            [
                {"item": "checker decision", "value": checker_decision},
                {"item": "part-token diagnostics", "value": diag_status},
                {"item": "missing artifacts", "value": len(missing)},
                {"item": "predicted classes", "value": as_int(run.get("predicted_classes"))},
                {"item": "best epoch", "value": as_int(summary.get("best_epoch") or test.get("checkpoint_epoch"))},
                {"item": "final trained epoch", "value": as_int(last.get("checkpoint_epoch") or last.get("epoch"))},
                {"item": "train samples", "value": as_int(summary.get("train_samples"))},
                {"item": "val samples", "value": as_int(summary.get("val_samples"))},
                {"item": "test samples", "value": as_int(summary.get("test_samples") or test.get("total"))},
                {"item": "device", "value": summary.get("device", "")},
            ],
            ["item", "value"],
        ),
        "",
        "## Accuracy-First Anchor Comparison",
        *md_table(accuracy_rows, ["run", "accuracy", "macro_f1", "A2_minus_anchor_acc", "A2_minus_anchor_macro_f1"]),
        "",
        "## Best vs Last Checkpoint",
        *md_table(best_last_rows, ["checkpoint", "epoch", "accuracy", "macro_f1", "loss", "detected_loss", "fallback_loss"]),
        "",
        "## Detected vs Fallback",
        *md_table(group_rows, ["group", "total", "accuracy", "macro_f1", "delta_acc_vs_best_rescue", "delta_macro_f1_vs_best_rescue"]),
        "",
        "## Per-Class Metrics",
        *md_table(class_rows, ["class", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Hard-Class Comparison",
        f"Hard-class mean A2: `{fmt(hard_mean)}`; best rescue hard-class mean: `{fmt(best_rescue_hard_mean)}`.",
        *md_table(hard_compare, ["class", "A2_f1", "best_rescue_f1", "delta"]),
        "",
        "## Top Confusions",
        *md_table(_top_confusions(run_dir), ["true", "predicted", "count", "support", "row_ratio"]),
        "",
        "## Prediction Distribution",
        *md_table(_prediction_distribution(run_dir), ["class", "pred_count", "pred_ratio"]),
    ]
    token_rows = _part_token_rows(run_dir)
    if token_rows:
        lines.extend(["", "## Part-Token Diagnostics", *md_table(token_rows, ["part", "token_norm_mean", "transformed_token_norm_mean", "valid_samples"])])
    lines.extend(
        [
            "",
            "## Decision",
            f"`{dec}`",
            "",
            "If A2 does not beat D15, the next architecture direction remains A3 MediaPipe-guided Part-conditioned Multi-Motif Query rather than another A1 seed or fallback rescue sweep.",
            "",
        ]
    )
    return lines


def write_report(
    output_dir: Path,
    run_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    hard_rows: List[Dict[str, Any]],
    warnings: List[str],
) -> str:
    dec = decision(run_rows, warnings)
    accuracy_rows = sorted(ANCHORS + run_rows, key=lambda row: as_float(row.get("test_accuracy")), reverse=True)
    macro_rows = sorted(ANCHORS + run_rows, key=lambda row: as_float(row.get("test_macro_f1")), reverse=True)
    pred_rows = [
        {
            "run_name": row.get("run_name"),
            "predicted_classes": row.get("predicted_classes"),
            "total": row.get("total", ""),
            "source": row.get("source", ""),
        }
        for row in run_rows
    ]
    lines = [
        "# D16R Main Branch Compare",
        "",
        "D16R main-branch readout runs use the MediaPipe pixel-prior rescue path. This comparison does not add region masks, fallback rescue, SupCon, multi-seed runs, or ensemble logic.",
        "",
        "## Accuracy-First Table",
        *md_table(
            accuracy_rows,
            [
                "run_name",
                "test_accuracy",
                "test_macro_f1",
                "detected_accuracy",
                "detected_macro_f1",
                "predicted_classes",
                "source",
            ],
        ),
        "",
        "## Macro-F1 Secondary Table",
        *md_table(macro_rows, ["run_name", "test_macro_f1", "test_accuracy", "source"]),
        "",
        "## Detected vs Fallback Group Table",
        *md_table(group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"]),
        "",
        "## Hard Classes",
        *md_table(hard_rows, ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"]),
        "",
        "## Predicted Class Count / No Collapse",
        *md_table(pred_rows, ["run_name", "predicted_classes", "total", "source"]),
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in warnings]])
    lines.extend(["", "## Decision", f"`{dec}`", ""])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("D16R_MAIN_BRANCH_COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    detailed = _a2_detailed_report(run_rows, hard_rows, warnings)
    if detailed:
        output_dir.joinpath("D16R_A2_PART_TOKEN_TRANSFORMER_ANALYSIS.md").write_text(
            "\n".join(detailed),
            encoding="utf-8",
        )
    a3_detailed = _a3_detailed_report(run_rows, hard_rows, warnings)
    if a3_detailed:
        output_dir.joinpath("D16R_A3_PART_MOTIF_QUERY_ANALYSIS.md").write_text(
            "\n".join(a3_detailed),
            encoding="utf-8",
        )
    a4_detailed = _a4_detailed_report(run_rows, hard_rows, warnings)
    if a4_detailed:
        output_dir.joinpath("D16R_A4_MICRO_MOTIF_SUPPORT_ANALYSIS.md").write_text(
            "\n".join(a4_detailed),
            encoding="utf-8",
        )
    a4b_detailed = _a4b_detailed_report(run_rows, hard_rows, warnings)
    if a4b_detailed:
        output_dir.joinpath("D16R_A4B_NO_GLOBAL_MICRO_ANALYSIS.md").write_text(
            "\n".join(a4b_detailed),
            encoding="utf-8",
        )
    return dec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for text in args.run_dirs:
        row, groups, hard, run_warnings = collect_run(Path(text))
        if row is not None:
            run_rows.append(row)
        group_rows.extend(groups)
        hard_rows.extend(hard)
        warnings.extend(run_warnings)

    write_csv(
        output_dir / "d16r_main_branch_summary.csv",
        run_rows,
        [
            "run_name",
            "test_accuracy",
            "test_macro_f1",
            "best_val_macro_f1",
            "best_epoch",
            "detected_accuracy",
            "detected_macro_f1",
            "fallback_accuracy",
            "fallback_macro_f1",
            "predicted_classes",
            "total",
            "output_dir",
            "missing_files",
            "source",
        ],
    )
    write_csv(
        output_dir / "d16r_main_branch_hard_class.csv",
        hard_rows,
        ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"],
    )
    write_csv(output_dir / "d16r_main_branch_group_metrics.csv", group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"])
    dec = write_report(output_dir, run_rows, group_rows, hard_rows, warnings)
    print(json.dumps({"output_dir": str(output_dir), "decision": dec, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
