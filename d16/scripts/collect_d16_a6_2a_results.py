"""Collect A6-2a hard prototype separation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
HARD_IDS = [0, 2, 4, 6]
ANCHORS = {
    "D15": {"accuracy": 0.645026, "macro_f1": 0.622471},
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
    "A5c": {"accuracy": 0.646698, "macro_f1": 0.630719, "hard_mean": 0.546327},
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


def _top_confusions(path: Path, topn: int = 10) -> List[Dict[str, Any]]:
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


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            vals.append(_fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
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
    summary_path = run_dir / "d16_train_summary.json"
    if summary_path.exists():
        train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_class_rows = _read_rows(run_dir / "per_class_metrics.csv")
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
        for row in per_class_rows
    }
    hard_mean = _hard_mean(per_class)
    accuracy = _float(test.get("accuracy"))
    macro_f1 = _float(test.get("macro_f1"))
    predicted_classes = _int(test.get("predicted_classes"), -1)
    group_rows = _read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    groups = {row.get("group"): row for row in group_rows}
    train_log = _read_rows(run_dir / "train_log.csv")
    last_train = train_log[-1] if train_log else {}
    best_epoch = _int(train_summary.get("best_epoch", test.get("checkpoint_epoch", 0)))

    comparison_rows = [
        {
            "run": "D15",
            "accuracy": ANCHORS["D15"]["accuracy"],
            "macro_f1": ANCHORS["D15"]["macro_f1"],
            "hard_mean": "",
            "delta_acc_vs_a6_2a": accuracy - ANCHORS["D15"]["accuracy"],
            "delta_macro_vs_a6_2a": macro_f1 - ANCHORS["D15"]["macro_f1"],
        },
        {
            "run": "A5b_seed42",
            "accuracy": ANCHORS["A5b_seed42"]["accuracy"],
            "macro_f1": ANCHORS["A5b_seed42"]["macro_f1"],
            "hard_mean": ANCHORS["A5b_seed42"]["hard_mean"],
            "delta_acc_vs_a6_2a": accuracy - ANCHORS["A5b_seed42"]["accuracy"],
            "delta_macro_vs_a6_2a": macro_f1 - ANCHORS["A5b_seed42"]["macro_f1"],
        },
        {
            "run": "A5b_mean",
            "accuracy": ANCHORS["A5b_mean"]["accuracy"],
            "macro_f1": ANCHORS["A5b_mean"]["macro_f1"],
            "hard_mean": ANCHORS["A5b_mean"]["hard_mean"],
            "delta_acc_vs_a6_2a": accuracy - ANCHORS["A5b_mean"]["accuracy"],
            "delta_macro_vs_a6_2a": macro_f1 - ANCHORS["A5b_mean"]["macro_f1"],
        },
        {
            "run": "A5c",
            "accuracy": ANCHORS["A5c"]["accuracy"],
            "macro_f1": ANCHORS["A5c"]["macro_f1"],
            "hard_mean": ANCHORS["A5c"]["hard_mean"],
            "delta_acc_vs_a6_2a": accuracy - ANCHORS["A5c"]["accuracy"],
            "delta_macro_vs_a6_2a": macro_f1 - ANCHORS["A5c"]["macro_f1"],
        },
        {
            "run": "A6_2a",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "hard_mean": hard_mean,
            "delta_acc_vs_a6_2a": 0.0,
            "delta_macro_vs_a6_2a": 0.0,
        },
    ]
    per_class_out = []
    for class_id in range(7):
        row = per_class.get(class_id, {"class_id": class_id, "class_name": CLASS_NAMES[class_id]})
        anchor_key = CLASS_NAMES[class_id]
        row = dict(row)
        row["a5b_seed42_f1"] = ANCHORS["A5b_seed42"].get(anchor_key, "")
        if isinstance(row["a5b_seed42_f1"], float):
            row["delta_f1_vs_a5b_seed42"] = _float(row.get("f1")) - row["a5b_seed42_f1"]
        per_class_out.append(row)
    confusion_rows = _top_confusions(run_dir / "confusion_matrix.csv", topn=15)

    decisions = []
    if predicted_classes < 7:
        decisions.append("REJECT_COLLAPSE")
    if missing:
        decisions.append("MISSING_ARTIFACTS")
    if accuracy < ANCHORS["D15"]["accuracy"]:
        decisions.append("REJECT_A6_2A")
    if accuracy > ANCHORS["A5b_seed42"]["accuracy"] and macro_f1 >= ANCHORS["A5b_seed42"]["macro_f1"]:
        decisions.append("A6_2A_BEATS_A5B_SEED42")
    if accuracy >= ANCHORS["A5b_mean"]["accuracy"] and hard_mean > ANCHORS["A5b_mean"]["hard_mean"]:
        decisions.append("A6_2A_USEFUL_HARD_SEPARATION")
    if hard_mean > ANCHORS["A5b_mean"]["hard_mean"] and accuracy < ANCHORS["A5b_mean"]["accuracy"]:
        decisions.append("HARD_GAIN_ACCURACY_TRADEOFF")
    fear_gain = _float(per_class.get(2, {}).get("f1")) > ANCHORS["A5b_seed42"]["Fear"]
    hard_drops = any(
        _float(per_class.get(class_id, {}).get("f1")) < ANCHORS["A5b_seed42"][CLASS_NAMES[class_id]]
        for class_id in (0, 4, 6)
    )
    if fear_gain and hard_drops and accuracy < ANCHORS["A5b_seed42"]["accuracy"]:
        decisions.append("PAIRWISE_GAIN_WITH_CLASS_TRADEOFF")
    if not decisions:
        decisions.append("A6_2A_COMPLETED_NEEDS_MANUAL_REVIEW")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "d16r_a6_2a_comparison.csv",
        comparison_rows,
        ["run", "accuracy", "macro_f1", "hard_mean", "delta_acc_vs_a6_2a", "delta_macro_vs_a6_2a"],
    )
    _write_csv(
        output_dir / "d16r_a6_2a_per_class.csv",
        per_class_out,
        ["class_id", "class_name", "support", "pred_count", "precision", "recall", "f1", "a5b_seed42_f1", "delta_f1_vs_a5b_seed42"],
    )
    _write_csv(
        output_dir / "d16r_a6_2a_confusion_summary.csv",
        confusion_rows,
        ["true_class", "true_name", "pred_class", "pred_name", "count", "support", "row_ratio"],
    )

    proto_diag = {
        "final_train_hard_proto_loss_total": _float(last_train.get("hard_proto_loss_total")),
        "final_train_hard_proto_loss_ce": _float(last_train.get("hard_proto_loss_ce")),
        "final_train_hard_proto_loss_margin": _float(last_train.get("hard_proto_loss_margin")),
        "final_lambda_hard_proto_current": _float(last_train.get("lambda_hard_proto_current")),
        "final_hard_proto_sample_count_mean": _float(last_train.get("hard_proto_sample_count_mean")),
        "final_hard_proto_positive_sim_mean": _float(last_train.get("hard_proto_positive_sim_mean")),
        "final_hard_proto_max_negative_sim_mean": _float(last_train.get("hard_proto_max_negative_sim_mean")),
    }
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
        "prototype_diagnostics": proto_diag,
        "decisions": decisions,
    }
    (output_dir / "d16r_a6_2a_next_decision.json").write_text(json.dumps(next_decision, indent=2), encoding="utf-8")

    hard_rows = [
        {
            "class": CLASS_NAMES[class_id],
            "A6_2a_f1": _float(per_class.get(class_id, {}).get("f1")),
            "A5b_seed42_f1": ANCHORS["A5b_seed42"][CLASS_NAMES[class_id]],
            "delta": _float(per_class.get(class_id, {}).get("f1")) - ANCHORS["A5b_seed42"][CLASS_NAMES[class_id]],
        }
        for class_id in HARD_IDS
    ]
    report = f"""# D16R A6-2a Hard Prototype Separation Analysis

## Executive Summary
Decision: `{", ".join(decisions)}`.

A6-2a is the first trainable hard-class prototype separation test on top of A5b. It keeps A5b architecture and inference unchanged, adding only a small training-time auxiliary loss over Angry/Fear/Sad/Neutral embeddings.

## Run Integrity
- integrity: `{"PASS" if not missing else "PARTIAL"}`
- missing: `{missing}`
- run_dir: `{run_dir}`
- predicted_classes: `{predicted_classes}`
- best_epoch: `{best_epoch}`

## Loss/Training Diagnostics
- final hard_proto_loss_total: `{_fmt(proto_diag["final_train_hard_proto_loss_total"])}`
- final hard_proto_loss_ce: `{_fmt(proto_diag["final_train_hard_proto_loss_ce"])}`
- final hard_proto_loss_margin: `{_fmt(proto_diag["final_train_hard_proto_loss_margin"])}`
- final lambda_hard_proto_current: `{_fmt(proto_diag["final_lambda_hard_proto_current"])}`
- final hard_proto_sample_count_mean: `{_fmt(proto_diag["final_hard_proto_sample_count_mean"])}`

## Best vs Last
| checkpoint | accuracy | macro_f1 |
|---|---:|---:|
| best.pt | {_fmt(accuracy)} | {_fmt(macro_f1)} |
| last.pt | {_fmt(last.get("accuracy"))} | {_fmt(last.get("macro_f1"))} |

## Accuracy/Macro Comparison
{_markdown_table(comparison_rows, ["run", "accuracy", "macro_f1", "hard_mean", "delta_acc_vs_a6_2a", "delta_macro_vs_a6_2a"])}

## Hard-Class F1 Comparison
{_markdown_table(hard_rows, ["class", "A6_2a_f1", "A5b_seed42_f1", "delta"])}

## Fear/Sad/Neutral Confusion Comparison
{_markdown_table(confusion_rows[:10], ["true_name", "pred_name", "count", "support", "row_ratio"])}

## Detected/Fallback
| group | total | accuracy | macro_f1 |
|---|---:|---:|---:|
| detected | {groups.get("detected", {}).get("total", "")} | {_fmt(groups.get("detected", {}).get("accuracy"))} | {_fmt(groups.get("detected", {}).get("macro_f1"))} |
| fallback | {groups.get("fallback", {}).get("total", "")} | {_fmt(groups.get("fallback", {}).get("accuracy"))} | {_fmt(groups.get("fallback", {}).get("macro_f1"))} |

## Prototype Diagnostics
The auxiliary loss is training-only. Prototypes are saved in the checkpoint as model parameters under `hard_proto_sep_loss.prototypes` and are not used by the inference classifier.

## Decision
`{", ".join(decisions)}`
"""
    _write_text(output_dir / "D16R_A6_2A_HARD_PROTO_SEP_ANALYSIS.md", report)
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
