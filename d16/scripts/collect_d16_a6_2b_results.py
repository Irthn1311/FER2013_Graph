"""Collect A6-2b pairwise hard-relation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
PAIR_PATTERNS = [(2, 4), (4, 2), (4, 6), (6, 4)]
ANCHORS = {
    "D15": {"accuracy": 0.645026, "macro_f1": 0.622471, "hard_mean": ""},
    "A5b_seed42": {
        "accuracy": 0.651435,
        "macro_f1": 0.637964,
        "hard_mean": 0.548988,
        "Angry": 0.537678,
        "Fear": 0.481518,
        "Sad": 0.521739,
        "Neutral": 0.655015,
    },
    "A5b_mean": {"accuracy": 0.650413, "macro_f1": 0.633385, "hard_mean": 0.550115},
    "A6_2a_best_official": {"accuracy": 0.634717, "macro_f1": 0.621808, "hard_mean": 0.530272},
    "A6_2a_last_diagnostic_only": {"accuracy": 0.652549, "macro_f1": 0.634741, "hard_mean": 0.552735},
}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
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


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _fmt(value: Any, digits: int = 6) -> str:
    val = _float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _first(path: Path) -> Dict[str, str]:
    rows = _read_rows(path)
    return rows[0] if rows else {}


def _hard_mean(per_class: Dict[int, Dict[str, Any]]) -> float:
    return _mean(_float(per_class[class_id].get("f1")) for class_id in HARD_IDS if class_id in per_class)


def _top_confusions(path: Path, topn: int = 15) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_rows(path):
        true_id = _int(row.get("true_class"), -1)
        pred_id = _int(row.get("pred_class"), -1)
        if true_id == pred_id or true_id < 0 or pred_id < 0:
            continue
        rows.append(
            {
                "true_class": true_id,
                "true_name": CLASS_NAMES.get(true_id, str(true_id)),
                "pred_class": pred_id,
                "pred_name": CLASS_NAMES.get(pred_id, str(pred_id)),
                "count": _int(row.get("count")),
                "support": _int(row.get("support")),
                "row_ratio": _float(row.get("row_ratio")),
            }
        )
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:topn]


def _confusion_map(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out = {}
    for row in _read_rows(path):
        true_id = _int(row.get("true_class"), -1)
        pred_id = _int(row.get("pred_class"), -1)
        if true_id < 0 or pred_id < 0 or true_id == pred_id:
            continue
        out[(true_id, pred_id)] = {
            "count": _int(row.get("count")),
            "row_ratio": _float(row.get("row_ratio")),
            "support": _int(row.get("support")),
        }
    return out


def _find_run_dir(run_name: str) -> Path | None:
    roots = [Path("outputs/d16_runs/main_branch"), Path("outputs/d16_runs/r")]
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(run_name))
        for match in matches:
            if match.is_dir() and (match / "confusion_matrix.csv").exists():
                return match
    return None


def _micro_gate(run_dir: Path, prefix: str = "") -> Dict[str, Any]:
    rows = _read_rows(run_dir / f"{prefix}micro_motif_summary.csv")
    micro = [row for row in rows if str(row.get("branch")) == "micro"]
    source = micro or rows
    if not source:
        return {"micro_gate_mean": float("nan"), "effective_motif_count_mean": float("nan"), "avg_offdiag_similarity_mean": float("nan")}
    return {
        "micro_gate_mean": _float(source[0].get("micro_gate_mean")),
        "effective_motif_count_mean": _float(source[0].get("effective_motif_count_mean")),
        "avg_offdiag_similarity_mean": _float(source[0].get("avg_offdiag_similarity_mean")),
    }


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col, "")
            values.append(_fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect(run_dir: Path, output_dir: Path) -> Dict[str, Any]:
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
    test = _first(run_dir / "test_metrics.csv")
    last = _first(run_dir / "last_test_metrics.csv")
    train_summary = {}
    if (run_dir / "d16_train_summary.json").exists():
        train_summary = json.loads((run_dir / "d16_train_summary.json").read_text(encoding="utf-8"))
    per_class = {
        _int(row.get("class_id")): {
            "class_id": _int(row.get("class_id")),
            "class_name": CLASS_NAMES.get(_int(row.get("class_id")), row.get("class_id")),
            "support": _int(row.get("support")),
            "pred_count": _int(row.get("pred_count")),
            "precision": _float(row.get("precision")),
            "recall": _float(row.get("recall")),
            "f1": _float(row.get("f1")),
        }
        for row in _read_rows(run_dir / "per_class_metrics.csv")
    }
    accuracy = _float(test.get("accuracy"))
    macro_f1 = _float(test.get("macro_f1"))
    hard_mean = _hard_mean(per_class)
    predicted_classes = _int(test.get("predicted_classes"), -1)
    best_epoch = _int(train_summary.get("best_epoch", test.get("checkpoint_epoch")))
    group_rows = _read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    groups = {row.get("group"): row for row in group_rows}
    train_log = _read_rows(run_dir / "train_log.csv")
    final_train = train_log[-1] if train_log else {}
    micro = _micro_gate(run_dir)
    last_micro = _micro_gate(run_dir, prefix="last_")

    comparison_rows = []
    for name, anchor in ANCHORS.items():
        comparison_rows.append(
            {
                "run": name,
                "accuracy": anchor["accuracy"],
                "macro_f1": anchor["macro_f1"],
                "hard_mean": anchor.get("hard_mean", ""),
                "a6_2b_minus_acc": accuracy - _float(anchor["accuracy"]),
                "a6_2b_minus_macro": macro_f1 - _float(anchor["macro_f1"]),
            }
        )
    comparison_rows.append(
        {
            "run": "A6_2b",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "hard_mean": hard_mean,
            "a6_2b_minus_acc": 0.0,
            "a6_2b_minus_macro": 0.0,
        }
    )

    per_class_out = []
    for class_id in range(7):
        row = dict(per_class.get(class_id, {"class_id": class_id, "class_name": CLASS_NAMES[class_id]}))
        anchor_f1 = ANCHORS["A5b_seed42"].get(CLASS_NAMES[class_id], "")
        row["a5b_seed42_f1"] = anchor_f1
        row["delta_f1_vs_a5b_seed42"] = "" if not isinstance(anchor_f1, float) else _float(row.get("f1")) - anchor_f1
        per_class_out.append(row)

    a5b_run = _find_run_dir("d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150")
    a6_2a_run = _find_run_dir("d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150")
    confusion_maps = {"A6_2b": _confusion_map(run_dir / "confusion_matrix.csv")}
    if a5b_run is not None:
        confusion_maps["A5b_seed42"] = _confusion_map(a5b_run / "confusion_matrix.csv")
    if a6_2a_run is not None:
        confusion_maps["A6_2a_best"] = _confusion_map(a6_2a_run / "confusion_matrix.csv")
        confusion_maps["A6_2a_last_diag"] = _confusion_map(a6_2a_run / "last_confusion_matrix.csv")
    pair_confusion_rows = []
    for true_id, pred_id in PAIR_PATTERNS:
        row = {
            "pattern": f"{CLASS_NAMES[true_id]}->{CLASS_NAMES[pred_id]}",
            "true_class": true_id,
            "pred_class": pred_id,
        }
        for name, cmap in confusion_maps.items():
            item = cmap.get((true_id, pred_id), {})
            row[f"{name}_count"] = item.get("count", "")
            row[f"{name}_row_ratio"] = item.get("row_ratio", "")
        pair_confusion_rows.append(row)
    top_confusion_rows = _top_confusions(run_dir / "confusion_matrix.csv", topn=15)

    loss_rows = []
    for row in train_log:
        loss_rows.append(
            {
                "epoch": _int(row.get("epoch")),
                "train_loss": _float(row.get("train_loss")),
                "ce_loss": _float(row.get("ce_loss")),
                "pairwise_loss_total": _float(row.get("pairwise_loss_total")),
                "pairwise_loss_fear_sad": _float(row.get("pairwise_loss_fear_sad")),
                "pairwise_loss_sad_neutral": _float(row.get("pairwise_loss_sad_neutral")),
                "lambda_pair_current": _float(row.get("lambda_pair_current")),
                "pair_count_fear_sad": _float(row.get("pair_count_fear_sad")),
                "pair_count_sad_neutral": _float(row.get("pair_count_sad_neutral")),
                "pair_acc_fear_sad_train": _float(row.get("pair_acc_fear_sad_train")),
                "pair_acc_sad_neutral_train": _float(row.get("pair_acc_sad_neutral_train")),
                "val_accuracy": _float(row.get("val_accuracy")),
                "val_macro_f1": _float(row.get("val_macro_f1")),
            }
        )

    decisions = []
    if predicted_classes < 7:
        decisions.append("REJECT_COLLAPSE")
    if missing:
        decisions.append("MISSING_ARTIFACTS")
    if accuracy < ANCHORS["D15"]["accuracy"]:
        decisions.append("REJECT_A6_2B")
    if accuracy > ANCHORS["A5b_seed42"]["accuracy"] and macro_f1 >= ANCHORS["A5b_mean"]["macro_f1"]:
        decisions.append("A6_2B_BEATS_A5B_SEED42")
    if accuracy >= ANCHORS["A5b_mean"]["accuracy"] and hard_mean > ANCHORS["A5b_mean"]["hard_mean"]:
        decisions.append("A6_2B_USEFUL_PAIRWISE_REFINEMENT")
    fear_f1 = _float(per_class.get(2, {}).get("f1"))
    sad_f1 = _float(per_class.get(4, {}).get("f1"))
    neutral_f1 = _float(per_class.get(6, {}).get("f1"))
    if (
        fear_f1 > ANCHORS["A5b_seed42"]["Fear"]
        and sad_f1 >= ANCHORS["A5b_seed42"]["Sad"] - 0.02
        and neutral_f1 >= ANCHORS["A5b_seed42"]["Neutral"] - 0.02
        and accuracy >= ANCHORS["A5b_mean"]["accuracy"] - 0.002
    ):
        decisions.append("A6_2B_HARD_GAIN_ACCEPTABLE")
    fs = confusion_maps["A6_2b"].get((2, 4), {}).get("count")
    sn = confusion_maps["A6_2b"].get((4, 6), {}).get("count")
    a5b_fs = confusion_maps.get("A5b_seed42", {}).get((2, 4), {}).get("count")
    a5b_sn = confusion_maps.get("A5b_seed42", {}).get((4, 6), {}).get("count")
    if isinstance(fs, int) and isinstance(sn, int) and isinstance(a5b_fs, int) and isinstance(a5b_sn, int):
        if fs < a5b_fs and sn > a5b_sn + 20:
            decisions.append("PAIRWISE_TRADEOFF_NOT_SOLVED")
    if _float(micro.get("micro_gate_mean")) < 0.05:
        decisions.append("AUX_LOSS_SUPPRESSES_MICRO_SUPPORT")
    if not decisions:
        decisions.append("A6_2B_COMPLETED_NEEDS_MANUAL_REVIEW")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "d16r_a6_2b_comparison.csv",
        comparison_rows,
        ["run", "accuracy", "macro_f1", "hard_mean", "a6_2b_minus_acc", "a6_2b_minus_macro"],
    )
    _write_csv(
        output_dir / "d16r_a6_2b_per_class.csv",
        per_class_out,
        ["class_id", "class_name", "support", "pred_count", "precision", "recall", "f1", "a5b_seed42_f1", "delta_f1_vs_a5b_seed42"],
    )
    _write_csv(
        output_dir / "d16r_a6_2b_confusion_summary.csv",
        pair_confusion_rows + top_confusion_rows,
        [
            "pattern",
            "true_class",
            "true_name",
            "pred_class",
            "pred_name",
            "count",
            "support",
            "row_ratio",
            "A6_2b_count",
            "A6_2b_row_ratio",
            "A5b_seed42_count",
            "A5b_seed42_row_ratio",
            "A6_2a_best_count",
            "A6_2a_best_row_ratio",
            "A6_2a_last_diag_count",
            "A6_2a_last_diag_row_ratio",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_2b_loss_diagnostics.csv",
        loss_rows,
        [
            "epoch",
            "train_loss",
            "ce_loss",
            "pairwise_loss_total",
            "pairwise_loss_fear_sad",
            "pairwise_loss_sad_neutral",
            "lambda_pair_current",
            "pair_count_fear_sad",
            "pair_count_sad_neutral",
            "pair_acc_fear_sad_train",
            "pair_acc_sad_neutral_train",
            "val_accuracy",
            "val_macro_f1",
        ],
    )

    next_decision = {
        "run_dir": str(run_dir),
        "integrity": "PASS" if not missing else "PARTIAL",
        "missing": missing,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "hard_mean": hard_mean,
        "predicted_classes": predicted_classes,
        "best_epoch": best_epoch,
        "last_accuracy": _float(last.get("accuracy")),
        "last_macro_f1": _float(last.get("macro_f1")),
        "detected": groups.get("detected", {}),
        "fallback": groups.get("fallback", {}),
        "final_loss_diagnostics": loss_rows[-1] if loss_rows else {},
        "micro_gate": micro,
        "last_micro_gate": last_micro,
        "a5b_confusion_source": "" if a5b_run is None else str(a5b_run),
        "a6_2a_confusion_source": "" if a6_2a_run is None else str(a6_2a_run),
        "decisions": decisions,
    }
    (output_dir / "d16r_a6_2b_next_decision.json").write_text(json.dumps(next_decision, indent=2), encoding="utf-8")

    hard_rows = [
        {
            "class": CLASS_NAMES[class_id],
            "A6_2b_f1": _float(per_class.get(class_id, {}).get("f1")),
            "A5b_seed42_f1": ANCHORS["A5b_seed42"][CLASS_NAMES[class_id]],
            "delta": _float(per_class.get(class_id, {}).get("f1")) - ANCHORS["A5b_seed42"][CLASS_NAMES[class_id]],
        }
        for class_id in HARD_IDS
    ]
    pred_rows = []
    for row in _read_rows(run_dir / "pred_count.csv"):
        class_id = _int(row.get("class_id"))
        pred_rows.append({"class": CLASS_NAMES.get(class_id, class_id), "pred_count": _int(row.get("pred_count"))})
    report = f"""# D16R A6-2b Pairwise Hard Relation Analysis

## Executive Summary
Decision: `{", ".join(decisions)}`.

A6-2b keeps the A5b backbone, EdgeContextGNN, A4 micro-motif readout, node/edge features, and inference classifier unchanged. The only intended change is a training-only pairwise auxiliary relation loss for Fear/Sad and Sad/Neutral.

## Run Integrity
- integrity: `{"PASS" if not missing else "PARTIAL"}`
- missing: `{missing}`
- run_dir: `{run_dir}`
- predicted_classes: `{predicted_classes}`
- best_epoch: `{best_epoch}`
- monitor: `{train_summary.get("best_monitor_metric", "")}`

## Loss Diagnostics
{_markdown_table(loss_rows[-8:], ["epoch", "ce_loss", "pairwise_loss_total", "pairwise_loss_fear_sad", "pairwise_loss_sad_neutral", "lambda_pair_current", "pair_acc_fear_sad_train", "pair_acc_sad_neutral_train"])}

## Best vs Last
| checkpoint | accuracy | macro_f1 |
|---|---:|---:|
| best.pt | {_fmt(accuracy)} | {_fmt(macro_f1)} |
| last.pt | {_fmt(last.get("accuracy"))} | {_fmt(last.get("macro_f1"))} |

## Accuracy/Macro Comparison
{_markdown_table(comparison_rows, ["run", "accuracy", "macro_f1", "hard_mean", "a6_2b_minus_acc", "a6_2b_minus_macro"])}

## Hard-Class F1 Comparison
{_markdown_table(hard_rows, ["class", "A6_2b_f1", "A5b_seed42_f1", "delta"])}

## Pairwise Confusion Comparison
{_markdown_table(pair_confusion_rows, ["pattern", "A6_2b_count", "A5b_seed42_count", "A6_2a_best_count", "A6_2a_last_diag_count"])}

## Prediction Distribution
{_markdown_table(pred_rows, ["class", "pred_count"])}

## Top Confusions
{_markdown_table(top_confusion_rows[:10], ["true_name", "pred_name", "count", "support", "row_ratio"])}

## Micro-Gate Diagnostics
| checkpoint | micro_gate_mean | effective_motif_count | avg_offdiag_similarity |
|---|---:|---:|---:|
| best.pt | {_fmt(micro.get("micro_gate_mean"))} | {_fmt(micro.get("effective_motif_count_mean"))} | {_fmt(micro.get("avg_offdiag_similarity_mean"))} |
| last.pt | {_fmt(last_micro.get("micro_gate_mean"))} | {_fmt(last_micro.get("effective_motif_count_mean"))} | {_fmt(last_micro.get("avg_offdiag_similarity_mean"))} |

## Detected/Fallback
| group | total | accuracy | macro_f1 |
|---|---:|---:|---:|
| detected | {groups.get("detected", {}).get("total", "")} | {_fmt(groups.get("detected", {}).get("accuracy"))} | {_fmt(groups.get("detected", {}).get("macro_f1"))} |
| fallback | {groups.get("fallback", {}).get("total", "")} | {_fmt(groups.get("fallback", {}).get("accuracy"))} | {_fmt(groups.get("fallback", {}).get("macro_f1"))} |

## Decision
`{", ".join(decisions)}`
"""
    _write_text(output_dir / "D16R_A6_2B_PAIRWISE_RELATION_ANALYSIS.md", report)
    return next_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()
    decision = collect(Path(args.run_dir), Path(args.output_dir))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
