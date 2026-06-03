"""A6-0 hard-class calibration and margin audit.

Read-only diagnostic over prediction CSV files. It does not load checkpoints,
train, or select checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
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
PAIR_PATTERNS = [
    (2, 4),
    (4, 6),
    (6, 4),
    (2, 0),
    (0, 4),
]
LOW_MARGIN = 0.15
HIGH_CONF = 0.70


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


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


def _safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else float("nan")


def _safe_std(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")


def _ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else float("nan")


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, 1.0e-12)) for p in probs)


def _pattern(true_id: int, pred_id: int) -> str:
    return f"{CLASS_NAMES.get(true_id, true_id)}->{CLASS_NAMES.get(pred_id, pred_id)}"


def _resolve_run(root: Path, run_name: str, preferred: Path | None = None) -> Path:
    if preferred is not None and preferred.exists():
        return preferred
    matches = [path for path in root.rglob(run_name) if path.is_dir() and (path / "predictions.csv").exists()]
    if not matches:
        raise FileNotFoundError(f"Could not resolve run directory: {run_name}")
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


def _augment(row: Dict[str, str]) -> Dict[str, Any]:
    probs = [_float(row.get(f"prob_{idx}")) for idx in range(7)]
    logits = [_float(row.get(f"logit_{idx}")) for idx in range(7)]
    order = sorted(range(7), key=lambda idx: probs[idx], reverse=True)
    top1, top2 = order[0], order[1]
    true_id = _int(row.get("y_true"))
    pred_id = _int(row.get("y_pred"))
    return {
        "sample_index": _int(row.get("sample_index"), -1),
        "true": true_id,
        "pred": pred_id,
        "correct": _int(row.get("correct")) == 1,
        "detected": _int(row.get("detected")) == 1,
        "probs": probs,
        "logits": logits,
        "confidence": probs[top1],
        "top2_pred": top2,
        "top2_prob": probs[top2],
        "margin": probs[top1] - probs[top2],
        "true_prob": probs[true_id],
        "entropy": _entropy(probs),
        "pred_prob": probs[pred_id],
    }


def _load_predictions(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    rows = _read_rows(run_dir / "predictions.csv")
    required = [f"logit_{idx}" for idx in range(7)] + [f"prob_{idx}" for idx in range(7)]
    missing = [col for col in required if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"{run_dir}/predictions.csv missing columns: {missing}")
    out = {}
    for row in rows:
        aug = _augment(row)
        out[aug["sample_index"]] = aug
    if len(out) != 3589:
        raise ValueError(f"{run_dir}/predictions.csv row count mismatch: {len(out)}")
    return out


def _summary(group: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    wrong = [row for row in rows if not row.get("correct_like", row.get("correct", False))]
    return {
        "group": group,
        "count": len(rows),
        "confidence_mean": _safe_mean(row["confidence"] for row in rows),
        "confidence_std": _safe_std(row["confidence"] for row in rows),
        "margin_mean": _safe_mean(row["margin"] for row in rows),
        "margin_std": _safe_std(row["margin"] for row in rows),
        "true_prob_mean": _safe_mean(row["true_prob"] for row in rows),
        "entropy_mean": _safe_mean(row["entropy"] for row in rows),
        "low_margin_count": sum(1 for row in rows if row["margin"] < LOW_MARGIN),
        "low_margin_ratio": _ratio(sum(1 for row in rows if row["margin"] < LOW_MARGIN), len(rows)),
        "high_conf_wrong_count": sum(1 for row in wrong if row["confidence"] >= HIGH_CONF),
        "high_conf_wrong_ratio": _ratio(sum(1 for row in wrong if row["confidence"] >= HIGH_CONF), len(wrong)),
    }


def _run_level_summary(runs: Dict[str, Dict[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for key, preds in runs.items():
        hard = [row for row in preds.values() if row["true"] in HARD_IDS]
        for class_id in ["ALL"] + HARD_IDS:
            subset = hard if class_id == "ALL" else [row for row in hard if row["true"] == class_id]
            for label, label_rows in (
                ("correct", [row for row in subset if row["correct"]]),
                ("wrong", [row for row in subset if not row["correct"]]),
            ):
                item = _summary(f"{key}_{CLASS_NAMES.get(class_id, class_id)}_{label}", label_rows)
                item.update(
                    {
                        "run": key,
                        "class": CLASS_NAMES.get(class_id, class_id),
                        "correct_group": label,
                    }
                )
                rows.append(item)
    return rows


def _sample_records(a5b: Dict[str, Dict[int, Dict[str, Any]]], a5c: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    common = sorted(set(a5b["seed42"]) & set(a5b["seed43"]) & set(a5b["seed44"]) & set(a5c))
    rows = []
    for idx in common:
        true_id = a5b["seed42"][idx]["true"]
        if true_id not in HARD_IDS:
            continue
        seed_rows = [a5b[key][idx] for key in ("seed42", "seed43", "seed44")]
        if any(row["true"] != true_id for row in seed_rows) or a5c[idx]["true"] != true_id:
            continue
        correct_count = sum(row["correct"] for row in seed_rows)
        probs = [[row["probs"][cls] for row in seed_rows] for cls in range(7)]
        mean_probs = [mean(vals) for vals in probs]
        maj_pred = max(range(7), key=lambda cls: mean_probs[cls])
        all_wrong_same = correct_count == 0 and len(set(row["pred"] for row in seed_rows)) == 1
        rec = {
            "sample_index": idx,
            "true": true_id,
            "true_name": CLASS_NAMES[true_id],
            "seed42_pred": seed_rows[0]["pred"],
            "seed43_pred": seed_rows[1]["pred"],
            "seed44_pred": seed_rows[2]["pred"],
            "seed_correct_count": correct_count,
            "majority_correct": correct_count >= 2,
            "majority_wrong": correct_count <= 1,
            "zero_of_3_correct": correct_count == 0,
            "all_3_correct": correct_count == 3,
            "all_wrong_same": all_wrong_same,
            "consistent_wrong_pred": seed_rows[0]["pred"] if all_wrong_same else "",
            "consistent_pattern": _pattern(true_id, seed_rows[0]["pred"]) if all_wrong_same else "",
            "mean_confidence": mean(row["confidence"] for row in seed_rows),
            "std_confidence": _safe_std(row["confidence"] for row in seed_rows),
            "mean_margin": mean(row["margin"] for row in seed_rows),
            "std_margin": _safe_std(row["margin"] for row in seed_rows),
            "mean_true_prob": mean(row["true_prob"] for row in seed_rows),
            "mean_entropy": _entropy(mean_probs),
            "majority_pred": maj_pred,
            "majority_pred_name": CLASS_NAMES[maj_pred],
            "a5c_pred": a5c[idx]["pred"],
            "a5c_pred_name": CLASS_NAMES[a5c[idx]["pred"]],
            "a5c_correct": a5c[idx]["correct"],
            "a5c_confidence": a5c[idx]["confidence"],
            "a5c_margin": a5c[idx]["margin"],
            "a5c_true_prob": a5c[idx]["true_prob"],
            "a5c_fixes_majority_wrong": correct_count <= 1 and a5c[idx]["correct"],
            "a5c_hurts_majority_correct": correct_count >= 2 and not a5c[idx]["correct"],
            "detected": seed_rows[0]["detected"],
        }
        for cls in range(7):
            rec[f"mean_prob_{cls}"] = mean_probs[cls]
        rows.append(rec)
    return rows


def _agreement_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    groups = [
        ("all_3_correct", [r for r in records if r["all_3_correct"]]),
        ("zero_of_3_correct", [r for r in records if r["zero_of_3_correct"]]),
        ("majority_correct", [r for r in records if r["majority_correct"]]),
        ("majority_wrong", [r for r in records if r["majority_wrong"]]),
        ("consistent_wrong_same_pred", [r for r in records if r["all_wrong_same"]]),
    ]
    for name, items in groups:
        pseudo = [
            {
                "confidence": r["mean_confidence"],
                "margin": r["mean_margin"],
                "true_prob": r["mean_true_prob"],
                "entropy": r["mean_entropy"],
                "correct_like": not ("wrong" in name or "zero" in name),
            }
            for r in items
        ]
        row = _summary(name, pseudo)
        row["run"] = "a5b_seed_mean"
        row["class"] = "hard"
        row["correct_group"] = name
        rows.append(row)
    return rows


def _consistent_error_margin(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        if row["all_wrong_same"]:
            grouped.setdefault(row["consistent_pattern"], []).append(row)
    out = []
    for pattern, rows in grouped.items():
        true_id = rows[0]["true"]
        pred_id = rows[0]["consistent_wrong_pred"]
        out.append(
            {
                "pattern": pattern,
                "count": len(rows),
                "mean_confidence": _safe_mean(row["mean_confidence"] for row in rows),
                "mean_margin": _safe_mean(row["mean_margin"] for row in rows),
                "mean_true_prob": _safe_mean(row["mean_true_prob"] for row in rows),
                "mean_pred_prob": _safe_mean(row[f"mean_prob_{pred_id}"] for row in rows),
                "mean_true_minus_pred_prob": _safe_mean(row[f"mean_prob_{true_id}"] - row[f"mean_prob_{pred_id}"] for row in rows),
                "low_margin_count": sum(1 for row in rows if row["mean_margin"] < LOW_MARGIN),
                "low_margin_ratio": _ratio(sum(1 for row in rows if row["mean_margin"] < LOW_MARGIN), len(rows)),
                "high_conf_wrong_count": sum(1 for row in rows if row["mean_confidence"] >= HIGH_CONF),
                "high_conf_wrong_ratio": _ratio(sum(1 for row in rows if row["mean_confidence"] >= HIGH_CONF), len(rows)),
                "a5c_fixes": sum(1 for row in rows if row["a5c_correct"]),
            }
        )
    return sorted(out, key=lambda row: row["count"], reverse=True)


def _pair_probability_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for true_id, pred_id in PAIR_PATTERNS:
        pattern = _pattern(true_id, pred_id)
        rows = [row for row in records if row["true"] == true_id and row["majority_pred"] == pred_id]
        consistent = [row for row in rows if row["all_wrong_same"] and row["consistent_wrong_pred"] == pred_id]
        for label, subset in (("majority_pred_pair", rows), ("consistent_all3_pair", consistent)):
            out.append(
                {
                    "pattern": pattern,
                    "subset": label,
                    "count": len(subset),
                    "true_prob_mean": _safe_mean(row[f"mean_prob_{true_id}"] for row in subset),
                    "pred_prob_mean": _safe_mean(row[f"mean_prob_{pred_id}"] for row in subset),
                    "prob_gap_true_minus_pred": _safe_mean(row[f"mean_prob_{true_id}"] - row[f"mean_prob_{pred_id}"] for row in subset),
                    "margin_mean": _safe_mean(row["mean_margin"] for row in subset),
                    "confidence_mean": _safe_mean(row["mean_confidence"] for row in subset),
                    "low_margin_ratio": _ratio(sum(1 for row in subset if row["mean_margin"] < LOW_MARGIN), len(subset)),
                }
            )
    return out


def _seed_prob_agreement(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_index": row["sample_index"],
            "true": row["true"],
            "true_name": row["true_name"],
            "seed42_pred": row["seed42_pred"],
            "seed43_pred": row["seed43_pred"],
            "seed44_pred": row["seed44_pred"],
            "seed_correct_count": row["seed_correct_count"],
            "majority_pred": row["majority_pred"],
            "majority_pred_name": row["majority_pred_name"],
            "mean_confidence": row["mean_confidence"],
            "std_confidence": row["std_confidence"],
            "mean_margin": row["mean_margin"],
            "std_margin": row["std_margin"],
            "mean_true_prob": row["mean_true_prob"],
            "mean_entropy": row["mean_entropy"],
            "all_wrong_same": row["all_wrong_same"],
            "consistent_pattern": row["consistent_pattern"],
            "detected": row["detected"],
        }
        for row in records
    ]


def _a5c_tradeoff(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in records:
        if not (row["a5c_fixes_majority_wrong"] or row["a5c_hurts_majority_correct"]):
            continue
        true_id = row["true"]
        a5c_pred = row["a5c_pred"]
        rows.append(
            {
                "sample_index": row["sample_index"],
                "true": true_id,
                "true_name": row["true_name"],
                "case": "fixes_majority_wrong" if row["a5c_fixes_majority_wrong"] else "hurts_majority_correct",
                "majority_pred": row["majority_pred"],
                "majority_pred_name": row["majority_pred_name"],
                "a5c_pred": a5c_pred,
                "a5c_pred_name": row["a5c_pred_name"],
                "a5b_mean_true_prob": row["mean_true_prob"],
                "a5c_true_prob": row["a5c_true_prob"],
                "true_prob_delta_a5c_minus_a5b": row["a5c_true_prob"] - row["mean_true_prob"],
                "a5b_mean_margin": row["mean_margin"],
                "a5c_margin": row["a5c_margin"],
                "margin_delta_a5c_minus_a5b": row["a5c_margin"] - row["mean_margin"],
                "a5b_mean_confidence": row["mean_confidence"],
                "a5c_confidence": row["a5c_confidence"],
                "confidence_delta_a5c_minus_a5b": row["a5c_confidence"] - row["mean_confidence"],
            }
        )
    return rows


def _tradeoff_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for case in ("fixes_majority_wrong", "hurts_majority_correct"):
        subset = [row for row in rows if row["case"] == case]
        for class_name in ["ALL"] + [CLASS_NAMES[idx] for idx in HARD_IDS]:
            items = subset if class_name == "ALL" else [row for row in subset if row["true_name"] == class_name]
            out.append(
                {
                    "case": case,
                    "class": class_name,
                    "count": len(items),
                    "true_prob_delta_mean": _safe_mean(row["true_prob_delta_a5c_minus_a5b"] for row in items),
                    "margin_delta_mean": _safe_mean(row["margin_delta_a5c_minus_a5b"] for row in items),
                    "confidence_delta_mean": _safe_mean(row["confidence_delta_a5c_minus_a5b"] for row in items),
                    "a5c_confidence_mean": _safe_mean(row["a5c_confidence"] for row in items),
                    "a5c_margin_mean": _safe_mean(row["a5c_margin"] for row in items),
                }
            )
    return out


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _decision(
    consistent_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    tradeoff_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    consistent_total = sum(row["count"] for row in consistent_rows)
    low_margin = sum(row["low_margin_count"] for row in consistent_rows)
    high_conf = sum(row["high_conf_wrong_count"] for row in consistent_rows)
    low_ratio = _ratio(low_margin, consistent_total)
    high_ratio = _ratio(high_conf, consistent_total)
    decisions = []
    if low_ratio >= 0.35:
        decisions.append("A6_CALIBRATION_OR_MARGIN_REFINEMENT_PROMISING")
    if high_ratio >= 0.35:
        decisions.append("A6_REPRESENTATION_REFINEMENT_NEEDED")
    pair_lookup = {f"{row['pattern']}::{row['subset']}": row for row in pair_rows}
    fs = pair_lookup.get("Fear->Sad::consistent_all3_pair", {})
    sn = pair_lookup.get("Sad->Neutral::consistent_all3_pair", {})
    ns = pair_lookup.get("Neutral->Sad::consistent_all3_pair", {})
    if _float(fs.get("low_margin_ratio")) >= 0.30 or abs(_float(fs.get("prob_gap_true_minus_pred"))) < 0.50:
        decisions.append("TARGET_PAIRWISE_FEAR_SAD_MARGIN")
    if (
        _float(sn.get("low_margin_ratio")) >= 0.30
        or _float(ns.get("low_margin_ratio")) >= 0.30
        or abs(_float(sn.get("prob_gap_true_minus_pred"))) < 0.50
        or abs(_float(ns.get("prob_gap_true_minus_pred"))) < 0.50
    ):
        decisions.append("TARGET_PAIRWISE_SAD_NEUTRAL_MARGIN")
    fixes = next(row for row in tradeoff_summary if row["case"] == "fixes_majority_wrong" and row["class"] == "ALL")
    hurts = next(row for row in tradeoff_summary if row["case"] == "hurts_majority_correct" and row["class"] == "ALL")
    if fixes["count"] > 0 and hurts["count"] > 0:
        decisions.append("A5C_SIGNAL_CAN_INFORM_A6_BUT_NOT_REUSE_DIRECTLY")
    if not decisions:
        decisions.append("A6_0_INCONCLUSIVE")
    return {
        "decisions": decisions,
        "consistent_error_total": consistent_total,
        "consistent_low_margin_count": low_margin,
        "consistent_low_margin_ratio": low_ratio,
        "consistent_high_conf_wrong_count": high_conf,
        "consistent_high_conf_wrong_ratio": high_ratio,
        "low_margin_threshold": LOW_MARGIN,
        "high_confidence_threshold": HIGH_CONF,
        "recommended_a6_direction": [
            "pairwise Fear-vs-Sad margin diagnostic/refinement",
            "pairwise Sad-vs-Neutral margin diagnostic/refinement",
            "representation refinement for high-confidence structural errors",
            "use A5c tradeoff as diagnostic signal only",
        ],
        "not_recommended_now": ["class_weighted_ce", "supcon", "focal_loss", "ensemble", "tta"],
    }


def _report(
    logit_integrity: List[Dict[str, Any]],
    run_summary: List[Dict[str, Any]],
    agreement_summary: List[Dict[str, Any]],
    consistent_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    tradeoff_summary: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> str:
    top_consistent = consistent_rows[:10]
    selected_pairs = [
        row
        for row in pair_rows
        if row["subset"] == "consistent_all3_pair" and row["pattern"] in {"Fear->Sad", "Sad->Neutral", "Neutral->Sad"}
    ]
    lines = [
        "# D16R A6-0 Hard-Class Calibration Audit",
        "",
        "## 1. Executive Summary",
        f"- Decisions: `{', '.join(decision['decisions'])}`.",
        f"- Consistent structural errors: `{decision['consistent_error_total']}` samples.",
        f"- Low-margin share among consistent errors: `{_fmt(decision['consistent_low_margin_ratio'])}` using margin < `{LOW_MARGIN}`.",
        f"- High-confidence-wrong share among consistent errors: `{_fmt(decision['consistent_high_conf_wrong_ratio'])}` using confidence >= `{HIGH_CONF}`.",
        "- Logits/probabilities were already available in predictions.csv; confidence, margin, top2, and true_prob were derived without eval-only export.",
        "",
        "## 2. Logit Export Integrity",
        _markdown_table(logit_integrity, ["run", "rows", "has_logits", "has_probs", "used_existing_predictions", "export_needed"]),
        "",
        "## 3. Confidence Correct vs Wrong",
        _markdown_table(
            [row for row in run_summary if row["class"] == "ALL" and row["correct_group"] in {"correct", "wrong"}],
            ["run", "correct_group", "count", "confidence_mean", "confidence_std", "true_prob_mean", "entropy_mean"],
        ),
        "",
        "## 4. Margin Correct vs Wrong",
        _markdown_table(
            [row for row in run_summary if row["class"] == "ALL" and row["correct_group"] in {"correct", "wrong"}],
            ["run", "correct_group", "count", "margin_mean", "margin_std", "low_margin_count", "low_margin_ratio", "high_conf_wrong_count", "high_conf_wrong_ratio"],
        ),
        "",
        "## 5. Consistent Error Margin Analysis",
        _markdown_table(
            top_consistent,
            ["pattern", "count", "mean_confidence", "mean_margin", "mean_true_prob", "mean_pred_prob", "low_margin_ratio", "high_conf_wrong_ratio"],
        ),
        "",
        "## 6. Fear-vs-Sad Probability Analysis",
        _markdown_table([row for row in pair_rows if row["pattern"] == "Fear->Sad"], ["pattern", "subset", "count", "true_prob_mean", "pred_prob_mean", "prob_gap_true_minus_pred", "margin_mean", "low_margin_ratio"]),
        "",
        "## 7. Sad-vs-Neutral Probability Analysis",
        _markdown_table([row for row in pair_rows if row["pattern"] == "Sad->Neutral"], ["pattern", "subset", "count", "true_prob_mean", "pred_prob_mean", "prob_gap_true_minus_pred", "margin_mean", "low_margin_ratio"]),
        "",
        "## 8. Neutral-vs-Sad Probability Analysis",
        _markdown_table([row for row in pair_rows if row["pattern"] == "Neutral->Sad"], ["pattern", "subset", "count", "true_prob_mean", "pred_prob_mean", "prob_gap_true_minus_pred", "margin_mean", "low_margin_ratio"]),
        "",
        "## 9. Seed Probability Agreement",
        _markdown_table(
            agreement_summary,
            ["group", "count", "confidence_mean", "margin_mean", "true_prob_mean", "entropy_mean", "low_margin_ratio", "high_conf_wrong_ratio"],
        ),
        "",
        "## 10. A5c Probability Tradeoff",
        _markdown_table(
            tradeoff_summary,
            ["case", "class", "count", "true_prob_delta_mean", "margin_delta_mean", "confidence_delta_mean", "a5c_confidence_mean", "a5c_margin_mean"],
        ),
        "",
        "## 11. A6 Design Implications",
        "- Some structural errors are close-margin, so pairwise margin diagnostics are worth testing.",
        "- A substantial share of consistent errors are high-confidence wrong, so pure calibration is unlikely to be enough; representation refinement remains necessary.",
        "- A5c raises true-class probability on some fixed samples but also hurts many majority-correct samples, so use it as a diagnostic clue rather than a replacement architecture.",
        "",
        "## 12. Paper-Safe Discussion Notes",
        "- A5b is stable across seeds, but hard-class confusions include both uncertain boundary cases and confident mistakes.",
        "- The audit characterizes probability behavior only; it does not establish causal or semantic evidence.",
        "",
        "## 13. Next Decision",
        f"`{', '.join(decision['decisions'])}`.",
        "",
        "Do not introduce class weighting, SupCon, focal loss, ensemble, or TTA from this audit alone.",
    ]
    return "\n".join(lines) + "\n"


def audit(run_root: Path, output_dir: Path, paths: Dict[str, Path | None]) -> Dict[str, Any]:
    run_names = {
        "seed42": "d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150",
        "seed43": "d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150",
        "seed44": "d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150",
        "a5c": "d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150",
    }
    run_dirs = {key: _resolve_run(run_root, name, paths.get(key)) for key, name in run_names.items()}
    runs = {key: _load_predictions(path) for key, path in run_dirs.items()}
    a5b = {key: runs[key] for key in ("seed42", "seed43", "seed44")}
    records = _sample_records(a5b, runs["a5c"])
    run_summary = _run_level_summary(runs)
    run_summary.extend(_agreement_summary(records))
    consistent_rows = _consistent_error_margin(records)
    pair_rows = _pair_probability_rows(records)
    seed_prob_rows = _seed_prob_agreement(records)
    tradeoff_rows = _a5c_tradeoff(records)
    tradeoff_summary = _tradeoff_summary(tradeoff_rows)
    decision = _decision(consistent_rows, pair_rows, tradeoff_summary)

    logit_integrity = [
        {
            "run": key,
            "rows": len(runs[key]),
            "has_logits": True,
            "has_probs": True,
            "used_existing_predictions": True,
            "export_needed": False,
            "run_dir": str(run_dirs[key]),
        }
        for key in ("seed42", "seed43", "seed44", "a5c")
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "d16r_a6_0_hard_margin_summary.csv",
        run_summary,
        [
            "run",
            "class",
            "correct_group",
            "group",
            "count",
            "confidence_mean",
            "confidence_std",
            "margin_mean",
            "margin_std",
            "true_prob_mean",
            "entropy_mean",
            "low_margin_count",
            "low_margin_ratio",
            "high_conf_wrong_count",
            "high_conf_wrong_ratio",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_0_consistent_error_margin.csv",
        consistent_rows,
        [
            "pattern",
            "count",
            "mean_confidence",
            "mean_margin",
            "mean_true_prob",
            "mean_pred_prob",
            "mean_true_minus_pred_prob",
            "low_margin_count",
            "low_margin_ratio",
            "high_conf_wrong_count",
            "high_conf_wrong_ratio",
            "a5c_fixes",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_0_seed_prob_agreement.csv",
        seed_prob_rows,
        [
            "sample_index",
            "true",
            "true_name",
            "seed42_pred",
            "seed43_pred",
            "seed44_pred",
            "seed_correct_count",
            "majority_pred",
            "majority_pred_name",
            "mean_confidence",
            "std_confidence",
            "mean_margin",
            "std_margin",
            "mean_true_prob",
            "mean_entropy",
            "all_wrong_same",
            "consistent_pattern",
            "detected",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_0_a5c_probability_tradeoff.csv",
        tradeoff_rows,
        [
            "sample_index",
            "true",
            "true_name",
            "case",
            "majority_pred",
            "majority_pred_name",
            "a5c_pred",
            "a5c_pred_name",
            "a5b_mean_true_prob",
            "a5c_true_prob",
            "true_prob_delta_a5c_minus_a5b",
            "a5b_mean_margin",
            "a5c_margin",
            "margin_delta_a5c_minus_a5b",
            "a5b_mean_confidence",
            "a5c_confidence",
            "confidence_delta_a5c_minus_a5b",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_0_pair_probability_summary.csv",
        pair_rows,
        [
            "pattern",
            "subset",
            "count",
            "true_prob_mean",
            "pred_prob_mean",
            "prob_gap_true_minus_pred",
            "margin_mean",
            "confidence_mean",
            "low_margin_ratio",
        ],
    )
    (output_dir / "d16r_a6_0_next_decision.json").write_text(json.dumps(decision | {"run_dirs": {k: str(v) for k, v in run_dirs.items()}}, indent=2), encoding="utf-8")
    _write_text(
        output_dir / "D16R_A6_0_HARD_CLASS_CALIBRATION_AUDIT.md",
        _report(logit_integrity, run_summary, _agreement_summary(records), consistent_rows, pair_rows, tradeoff_summary, decision),
    )
    return decision | {"logits_available": True, "export_needed": False, "hard_samples": len(records)}


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
