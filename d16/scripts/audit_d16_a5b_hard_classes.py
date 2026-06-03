"""Read-only hard-class prediction audit for A5b seeds and A5c."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
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
HARD_IDS = [0, 2, 4, 6]
HARD_NAMES = {idx: CLASS_NAMES[idx] for idx in HARD_IDS}
WATCH_PATTERNS = [
    (2, 4),  # Fear -> Sad
    (2, 0),  # Fear -> Angry
    (2, 6),  # Fear -> Neutral
    (4, 6),  # Sad -> Neutral
    (4, 2),  # Sad -> Fear
    (4, 0),  # Sad -> Angry
    (6, 4),  # Neutral -> Sad
    (0, 4),  # Angry -> Sad
]
REQUIRED = [
    "predictions.csv",
    "confusion_matrix.csv",
    "per_class_metrics.csv",
    "detected_vs_fallback_metrics.csv",
    "detected_fallback_per_class_metrics.csv",
]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _fmt(value: Any, digits: int = 6) -> str:
    val = _float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _safe_ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else float("nan")


def _safe_mean(values: Iterable[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return mean(vals) if vals else float("nan")


def _pattern(true_id: int, pred_id: int) -> str:
    return f"{CLASS_NAMES.get(true_id, true_id)}->{CLASS_NAMES.get(pred_id, pred_id)}"


def _resolve_run_dir(root: Path, run_name: str, preferred: Path | None = None) -> Path:
    if preferred is not None and preferred.exists():
        return preferred
    matches = [path for path in root.rglob(run_name) if path.is_dir() and (path / "predictions.csv").exists()]
    if not matches:
        raise FileNotFoundError(f"Could not find run dir for {run_name} under {root}")
    # Prefer the shallowest path to avoid nested duplicate unzips.
    return sorted(matches, key=lambda item: (len(item.parts), str(item)))[0]


def _load_predictions(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for row in _read_rows(run_dir / "predictions.csv"):
        sample_index = _int(row.get("sample_index"), -1)
        if sample_index < 0:
            continue
        rows[sample_index] = {
            "sample_index": sample_index,
            "true": _int(row.get("y_true")),
            "pred": _int(row.get("y_pred")),
            "correct": _int(row.get("correct")) == 1,
            "detected": _int(row.get("detected")) == 1,
            "landmark_missing": _int(row.get("landmark_missing_flag")) == 1,
        }
    return rows


def _load_run(run_name: str, run_dir: Path, label: str) -> Dict[str, Any]:
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    predictions = _load_predictions(run_dir)
    per_class = {
        _int(row.get("class_id")): {
            "precision": _float(row.get("precision")),
            "recall": _float(row.get("recall")),
            "f1": _float(row.get("f1")),
            "support": _int(row.get("support")),
            "pred_count": _int(row.get("pred_count")),
        }
        for row in _read_rows(run_dir / "per_class_metrics.csv")
    }
    return {
        "run_name": run_name,
        "label": label,
        "run_dir": str(run_dir),
        "missing": missing,
        "predictions": predictions,
        "per_class": per_class,
    }


def _hard_samples(a5b_runs: Dict[str, Dict[str, Any]], a5c: Dict[str, Any]) -> List[Dict[str, Any]]:
    seed42 = a5b_runs["seed42"]["predictions"]
    seed43 = a5b_runs["seed43"]["predictions"]
    seed44 = a5b_runs["seed44"]["predictions"]
    a5c_preds = a5c["predictions"]
    common = sorted(set(seed42) & set(seed43) & set(seed44) & set(a5c_preds))
    rows = []
    for sample_index in common:
        true_id = seed42[sample_index]["true"]
        if true_id not in HARD_IDS:
            continue
        if any(preds[sample_index]["true"] != true_id for preds in (seed43, seed44, a5c_preds)):
            continue
        preds = {
            "seed42": seed42[sample_index]["pred"],
            "seed43": seed43[sample_index]["pred"],
            "seed44": seed44[sample_index]["pred"],
        }
        correct_count = sum(pred == true_id for pred in preds.values())
        majority_a5b_correct = correct_count >= 2
        a5c_pred = a5c_preds[sample_index]["pred"]
        a5c_correct = a5c_pred == true_id
        detected_votes = sum(
            int(preds_map[sample_index]["detected"])
            for preds_map in (seed42, seed43, seed44, a5c_preds)
        )
        detected = seed42[sample_index]["detected"]
        rows.append(
            {
                "sample_index": sample_index,
                "true": true_id,
                "true_name": CLASS_NAMES[true_id],
                "seed42_pred": preds["seed42"],
                "seed42_pred_name": CLASS_NAMES[preds["seed42"]],
                "seed43_pred": preds["seed43"],
                "seed43_pred_name": CLASS_NAMES[preds["seed43"]],
                "seed44_pred": preds["seed44"],
                "seed44_pred_name": CLASS_NAMES[preds["seed44"]],
                "a5c_pred": a5c_pred,
                "a5c_pred_name": CLASS_NAMES[a5c_pred],
                "seed_correct_count": correct_count,
                "agreement_bucket": f"{correct_count}/3_correct",
                "majority_a5b_correct": majority_a5b_correct,
                "majority_a5b_wrong": not majority_a5b_correct,
                "a5c_correct": a5c_correct,
                "a5c_fixes_majority_wrong": (not majority_a5b_correct) and a5c_correct,
                "a5c_hurts_majority_correct": majority_a5b_correct and (not a5c_correct),
                "all_a5b_wrong_same_pred": correct_count == 0 and len(set(preds.values())) == 1,
                "consistent_wrong_pred": preds["seed42"] if correct_count == 0 and len(set(preds.values())) == 1 else "",
                "consistent_wrong_pred_name": CLASS_NAMES[preds["seed42"]] if correct_count == 0 and len(set(preds.values())) == 1 else "",
                "detected": detected,
                "detected_label": "detected" if detected else "fallback",
                "detected_votes_4runs": detected_votes,
            }
        )
    return rows


def _agreement_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    classes = ["ALL"] + [CLASS_NAMES[idx] for idx in HARD_IDS]
    for class_name in classes:
        subset = rows if class_name == "ALL" else [row for row in rows if row["true_name"] == class_name]
        total = len(subset)
        counts = {i: sum(1 for row in subset if row["seed_correct_count"] == i) for i in range(4)}
        out.append(
            {
                "class": class_name,
                "total": total,
                "all_3_correct": counts[3],
                "all_3_correct_ratio": _safe_ratio(counts[3], total),
                "two_of_3_correct": counts[2],
                "two_of_3_correct_ratio": _safe_ratio(counts[2], total),
                "one_of_3_correct": counts[1],
                "one_of_3_correct_ratio": _safe_ratio(counts[1], total),
                "zero_of_3_correct": counts[0],
                "zero_of_3_correct_ratio": _safe_ratio(counts[0], total),
                "majority_correct": counts[2] + counts[3],
                "majority_correct_ratio": _safe_ratio(counts[2] + counts[3], total),
                "majority_wrong": counts[0] + counts[1],
                "majority_wrong_ratio": _safe_ratio(counts[0] + counts[1], total),
            }
        )
    return out


def _consistent_errors(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if not row["all_a5b_wrong_same_pred"]:
            continue
        true_id = row["true"]
        pred_id = _int(row["consistent_wrong_pred"])
        out.append(
            {
                "sample_index": row["sample_index"],
                "true": true_id,
                "true_name": row["true_name"],
                "consistent_pred": pred_id,
                "consistent_pred_name": row["consistent_wrong_pred_name"],
                "pattern": _pattern(true_id, pred_id),
                "a5c_pred": row["a5c_pred"],
                "a5c_pred_name": row["a5c_pred_name"],
                "a5c_correct": row["a5c_correct"],
                "detected": row["detected"],
                "detected_label": row["detected_label"],
            }
        )
    return out


def _consistent_pattern_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row["pattern"]
        item = grouped.setdefault(
            key,
            {
                "pattern": key,
                "true_name": row["true_name"],
                "pred_name": row["consistent_pred_name"],
                "count": 0,
                "detected_count": 0,
                "fallback_count": 0,
                "a5c_fixes": 0,
            },
        )
        item["count"] += 1
        item["detected_count"] += int(row["detected"])
        item["fallback_count"] += int(not row["detected"])
        item["a5c_fixes"] += int(row["a5c_correct"])
    return sorted(grouped.values(), key=lambda row: row["count"], reverse=True)


def _detected_fallback_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for class_name in ["ALL"] + [CLASS_NAMES[idx] for idx in HARD_IDS]:
        subset = rows if class_name == "ALL" else [row for row in rows if row["true_name"] == class_name]
        for label, label_rows in (
            ("detected", [row for row in subset if row["detected"]]),
            ("fallback", [row for row in subset if not row["detected"]]),
        ):
            total = len(label_rows)
            majority_wrong = sum(1 for row in label_rows if row["majority_a5b_wrong"])
            zero_correct = sum(1 for row in label_rows if row["seed_correct_count"] == 0)
            consistent = sum(1 for row in label_rows if row["all_a5b_wrong_same_pred"])
            out.append(
                {
                    "class": class_name,
                    "group": label,
                    "total_hard_samples": total,
                    "majority_a5b_wrong": majority_wrong,
                    "majority_a5b_wrong_ratio": _safe_ratio(majority_wrong, total),
                    "zero_of_3_correct": zero_correct,
                    "zero_of_3_correct_ratio": _safe_ratio(zero_correct, total),
                    "consistent_same_wrong_pred": consistent,
                    "consistent_same_wrong_pred_ratio": _safe_ratio(consistent, total),
                }
            )
    return out


def _a5c_tradeoff(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row["a5c_fixes_majority_wrong"] or row["a5c_hurts_majority_correct"] or row["majority_a5b_wrong"]:
            out.append(
                {
                    "sample_index": row["sample_index"],
                    "true": row["true"],
                    "true_name": row["true_name"],
                    "seed42_pred": row["seed42_pred"],
                    "seed43_pred": row["seed43_pred"],
                    "seed44_pred": row["seed44_pred"],
                    "a5c_pred": row["a5c_pred"],
                    "seed42_pred_name": row["seed42_pred_name"],
                    "seed43_pred_name": row["seed43_pred_name"],
                    "seed44_pred_name": row["seed44_pred_name"],
                    "a5c_pred_name": row["a5c_pred_name"],
                    "seed_correct_count": row["seed_correct_count"],
                    "majority_a5b_correct": row["majority_a5b_correct"],
                    "majority_a5b_wrong": row["majority_a5b_wrong"],
                    "a5c_correct": row["a5c_correct"],
                    "a5c_fixes_majority_wrong": row["a5c_fixes_majority_wrong"],
                    "a5c_hurts_majority_correct": row["a5c_hurts_majority_correct"],
                    "detected_label": row["detected_label"],
                }
            )
    return out


def _tradeoff_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for class_name in ["ALL"] + [CLASS_NAMES[idx] for idx in HARD_IDS]:
        subset = rows if class_name == "ALL" else [row for row in rows if row["true_name"] == class_name]
        majority_wrong = [row for row in subset if row["majority_a5b_wrong"]]
        majority_correct = [row for row in subset if row["majority_a5b_correct"]]
        fixes = sum(1 for row in subset if row["a5c_fixes_majority_wrong"])
        hurts = sum(1 for row in subset if row["a5c_hurts_majority_correct"])
        out.append(
            {
                "class": class_name,
                "hard_samples": len(subset),
                "majority_a5b_wrong": len(majority_wrong),
                "a5c_fixes_majority_wrong": fixes,
                "fix_ratio_among_majority_wrong": _safe_ratio(fixes, len(majority_wrong)),
                "majority_a5b_correct": len(majority_correct),
                "a5c_hurts_majority_correct": hurts,
                "hurt_ratio_among_majority_correct": _safe_ratio(hurts, len(majority_correct)),
                "net_fix_minus_hurt": fixes - hurts,
            }
        )
    return out


def _watch_pattern_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for true_id, pred_id in WATCH_PATTERNS:
        pattern = _pattern(true_id, pred_id)
        per_seed_counts = {}
        for seed_key in ("seed42", "seed43", "seed44"):
            per_seed_counts[seed_key] = sum(
                1 for row in rows if row["true"] == true_id and row[f"{seed_key}_pred"] == pred_id
            )
        consistent = sum(
            1
            for row in rows
            if row["true"] == true_id
            and row["seed42_pred"] == pred_id
            and row["seed43_pred"] == pred_id
            and row["seed44_pred"] == pred_id
        )
        out.append(
            {
                "pattern": pattern,
                "seed42_count": per_seed_counts["seed42"],
                "seed43_count": per_seed_counts["seed43"],
                "seed44_count": per_seed_counts["seed44"],
                "mean_seed_count": _safe_mean(float(x) for x in per_seed_counts.values()),
                "consistent_all3_count": consistent,
            }
        )
    return out


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _report(
    agreement: List[Dict[str, Any]],
    consistent_errors: List[Dict[str, Any]],
    consistent_patterns: List[Dict[str, Any]],
    detected_fallback: List[Dict[str, Any]],
    tradeoff_summary: List[Dict[str, Any]],
    watch_patterns: List[Dict[str, Any]],
    decisions: Dict[str, Any],
) -> str:
    top_consistent = consistent_patterns[:10]
    df_all = [row for row in detected_fallback if row["class"] == "ALL"]
    hard_structural = decisions["hard_error_decision"]
    a5c_decision = decisions["a5c_decision"]
    fallback_decision = decisions["fallback_decision"]
    a6_targets = ", ".join(decisions["a6_targets"])
    lines = [
        "# D16R A5b Hard-Class Prediction Audit",
        "",
        "## Executive Summary",
        f"- Hard-class decision: `{hard_structural}`.",
        f"- A5c decision: `{a5c_decision}`.",
        f"- Fallback decision: `{fallback_decision}`.",
        f"- Recommended A6 target(s): {a6_targets}.",
        "- This audit is read-only over predictions/confusion/per-class/group metrics; no model, checkpoint, or training code was changed.",
        "",
        "## Hard-Class Sample Agreement",
        _markdown_table(
            agreement,
            [
                "class",
                "total",
                "all_3_correct",
                "two_of_3_correct",
                "one_of_3_correct",
                "zero_of_3_correct",
                "majority_correct",
                "majority_wrong",
            ],
        ),
        "",
        "## Watched Error Patterns Across Seeds",
        _markdown_table(
            watch_patterns,
            ["pattern", "seed42_count", "seed43_count", "seed44_count", "mean_seed_count", "consistent_all3_count"],
        ),
        "",
        "## Top Consistent Same-Class Errors",
        _markdown_table(
            top_consistent,
            ["pattern", "count", "detected_count", "fallback_count", "a5c_fixes"],
        ),
        "",
        "These are samples where all three A5b seeds are wrong and choose the same incorrect class. This supports a structural hard-class issue rather than only seed variance.",
        "",
        "## Detected vs Fallback",
        _markdown_table(
            df_all,
            [
                "class",
                "group",
                "total_hard_samples",
                "majority_a5b_wrong",
                "majority_a5b_wrong_ratio",
                "zero_of_3_correct",
                "consistent_same_wrong_pred",
            ],
        ),
        "",
        "Hard-class errors are dominated by detected samples in count. Fallback remains weak but is too small to justify returning to a fallback branch as the next development step.",
        "",
        "## A5c Tradeoff",
        _markdown_table(
            tradeoff_summary,
            [
                "class",
                "hard_samples",
                "majority_a5b_wrong",
                "a5c_fixes_majority_wrong",
                "fix_ratio_among_majority_wrong",
                "majority_a5b_correct",
                "a5c_hurts_majority_correct",
                "hurt_ratio_among_majority_correct",
                "net_fix_minus_hurt",
            ],
        ),
        "",
        "A5c fixes some majority-wrong hard samples, but it also breaks many majority-correct samples. This is a trade-off signal, not a replacement signal.",
        "",
        "## Paper-Safe Discussion Notes",
        "- A5b is stable at the metric level, but hard-class prediction errors concentrate in visually similar negative/neutral expressions.",
        "- Fear, Sad, Neutral, and Angry form a recurring confusion cluster across seeds.",
        "- The audit does not establish causal evidence or semantic motif evidence; it only characterizes prediction-level behavior.",
        "",
        "## Recommended Next Decision",
        "- Do not repeat A5c now.",
        "- Do not return to fallback rescue.",
        "- A6 should begin with Fear-vs-Sad and Sad-vs-Neutral relation analysis, plus calibration/audit-only work before adding an auxiliary relation head.",
    ]
    return "\n".join(lines) + "\n"


def audit(run_root: Path, output_dir: Path, paths: Dict[str, Path | None]) -> Dict[str, Any]:
    run_names = {
        "seed42": "d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150",
        "seed43": "d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150",
        "seed44": "d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150",
        "a5c": "d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150",
    }
    a5b_runs = {
        key: _load_run(name, _resolve_run_dir(run_root, name, paths.get(key)), key)
        for key, name in run_names.items()
        if key != "a5c"
    }
    a5c = _load_run(run_names["a5c"], _resolve_run_dir(run_root, run_names["a5c"], paths.get("a5c")), "a5c")
    missing = {key: run["missing"] for key, run in {**a5b_runs, "a5c": a5c}.items() if run["missing"]}
    if missing:
        raise FileNotFoundError(f"Missing required audit files: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hard_rows = _hard_samples(a5b_runs, a5c)
    agreement = _agreement_summary(hard_rows)
    consistent = _consistent_errors(hard_rows)
    consistent_patterns = _consistent_pattern_summary(consistent)
    detected_fallback = _detected_fallback_summary(hard_rows)
    tradeoff = _a5c_tradeoff(hard_rows)
    tradeoff_summary = _tradeoff_summary(hard_rows)
    watch_patterns = _watch_pattern_summary(hard_rows)

    total_hard = len(hard_rows)
    consistent_count = len(consistent)
    zero_wrong = sum(1 for row in hard_rows if row["seed_correct_count"] == 0)
    a5c_fixes = sum(1 for row in hard_rows if row["a5c_fixes_majority_wrong"])
    a5c_hurts = sum(1 for row in hard_rows if row["a5c_hurts_majority_correct"])
    fallback_majority_wrong = sum(1 for row in hard_rows if row["majority_a5b_wrong"] and not row["detected"])
    total_majority_wrong = sum(1 for row in hard_rows if row["majority_a5b_wrong"])
    structural_ratio = _safe_ratio(consistent_count, max(zero_wrong, 1))
    hard_error_decision = "HARD_ERRORS_ARE_STRUCTURAL" if structural_ratio >= 0.30 else "HARD_ERRORS_ARE_UNSTABLE_VARIANCE"
    a5c_decision = "A5C_HAS_HARD_CLASS_SIGNAL_BUT_TRADEOFF" if a5c_fixes > 0 and a5c_hurts >= a5c_fixes else "A5C_HARD_CLASS_SIGNAL_WEAK"
    fallback_decision = "DO_NOT_RETURN_TO_FALLBACK" if _safe_ratio(fallback_majority_wrong, max(total_majority_wrong, 1)) < 0.15 else "FALLBACK_CONTRIBUTES_TO_HARD_ERRORS"
    a6_targets = ["Fear-vs-Sad relation", "Sad-vs-Neutral relation", "calibration/audit only"]
    if consistent_count >= 100:
        a6_targets.append("hard-class auxiliary relation head after risk analysis")
    decisions = {
        "hard_error_decision": hard_error_decision,
        "a5c_decision": a5c_decision,
        "fallback_decision": fallback_decision,
        "a6_targets": a6_targets,
        "total_hard_samples": total_hard,
        "zero_of_3_correct": zero_wrong,
        "consistent_same_wrong_pred": consistent_count,
        "consistent_among_zero_correct_ratio": structural_ratio,
        "majority_a5b_wrong": total_majority_wrong,
        "fallback_majority_wrong": fallback_majority_wrong,
        "fallback_majority_wrong_share": _safe_ratio(fallback_majority_wrong, max(total_majority_wrong, 1)),
        "a5c_fixes_majority_wrong": a5c_fixes,
        "a5c_hurts_majority_correct": a5c_hurts,
        "run_dirs": {key: run["run_dir"] for key, run in {**a5b_runs, "a5c": a5c}.items()},
    }

    _write_csv(
        output_dir / "d16r_a5b_hard_class_sample_agreement.csv",
        agreement,
        [
            "class",
            "total",
            "all_3_correct",
            "all_3_correct_ratio",
            "two_of_3_correct",
            "two_of_3_correct_ratio",
            "one_of_3_correct",
            "one_of_3_correct_ratio",
            "zero_of_3_correct",
            "zero_of_3_correct_ratio",
            "majority_correct",
            "majority_correct_ratio",
            "majority_wrong",
            "majority_wrong_ratio",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_consistent_errors.csv",
        consistent,
        [
            "sample_index",
            "true",
            "true_name",
            "consistent_pred",
            "consistent_pred_name",
            "pattern",
            "a5c_pred",
            "a5c_pred_name",
            "a5c_correct",
            "detected",
            "detected_label",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_a5c_hard_sample_tradeoff.csv",
        tradeoff,
        [
            "sample_index",
            "true",
            "true_name",
            "seed42_pred",
            "seed43_pred",
            "seed44_pred",
            "a5c_pred",
            "seed42_pred_name",
            "seed43_pred_name",
            "seed44_pred_name",
            "a5c_pred_name",
            "seed_correct_count",
            "majority_a5b_correct",
            "majority_a5b_wrong",
            "a5c_correct",
            "a5c_fixes_majority_wrong",
            "a5c_hurts_majority_correct",
            "detected_label",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_hard_class_detected_fallback.csv",
        detected_fallback,
        [
            "class",
            "group",
            "total_hard_samples",
            "majority_a5b_wrong",
            "majority_a5b_wrong_ratio",
            "zero_of_3_correct",
            "zero_of_3_correct_ratio",
            "consistent_same_wrong_pred",
            "consistent_same_wrong_pred_ratio",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_hard_class_sample_stability.csv",
        hard_rows,
        [
            "sample_index",
            "true",
            "true_name",
            "seed42_pred",
            "seed42_pred_name",
            "seed43_pred",
            "seed43_pred_name",
            "seed44_pred",
            "seed44_pred_name",
            "a5c_pred",
            "a5c_pred_name",
            "seed_correct_count",
            "agreement_bucket",
            "majority_a5b_correct",
            "majority_a5b_wrong",
            "a5c_correct",
            "a5c_fixes_majority_wrong",
            "a5c_hurts_majority_correct",
            "all_a5b_wrong_same_pred",
            "consistent_wrong_pred",
            "consistent_wrong_pred_name",
            "detected",
            "detected_label",
            "detected_votes_4runs",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_hard_class_pattern_summary.csv",
        consistent_patterns,
        ["pattern", "true_name", "pred_name", "count", "detected_count", "fallback_count", "a5c_fixes"],
    )
    _write_csv(
        output_dir / "d16r_a5b_hard_class_watch_patterns.csv",
        watch_patterns,
        ["pattern", "seed42_count", "seed43_count", "seed44_count", "mean_seed_count", "consistent_all3_count"],
    )
    (output_dir / "d16r_a5b_hard_class_next_decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    _write_text(
        output_dir / "D16R_A5B_HARD_CLASS_AUDIT.md",
        _report(agreement, consistent, consistent_patterns, detected_fallback, tradeoff_summary, watch_patterns, decisions),
    )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="outputs/d16_runs/r")
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    parser.add_argument("--seed42_dir", default=None)
    parser.add_argument("--seed43_dir", default=None)
    parser.add_argument("--seed44_dir", default=None)
    parser.add_argument("--a5c_dir", default=None)
    args = parser.parse_args()
    paths = {
        "seed42": Path(args.seed42_dir) if args.seed42_dir else None,
        "seed43": Path(args.seed43_dir) if args.seed43_dir else None,
        "seed44": Path(args.seed44_dir) if args.seed44_dir else None,
        "a5c": Path(args.a5c_dir) if args.a5c_dir else None,
    }
    result = audit(Path(args.run_root), Path(args.output_dir), paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
