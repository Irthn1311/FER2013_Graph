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
    if acc >= 0.650:
        return "STRONG_A2_SIGNAL"
    if acc > D15_ACC:
        return "BEATS_D15_ACCURACY_KEEP_AND_REPEAT"
    if acc > BEST_RESCUE_ACC:
        return "A2_USEFUL_BUT_NOT_ENOUGH"

    run_dir = Path(str(best.get("output_dir", "")))
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


def _a2_detailed_report(run_rows: List[Dict[str, Any]], hard_rows: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    if not run_rows:
        return []
    run = max(run_rows, key=lambda row: as_float(row.get("test_accuracy")))
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
        "D16R-A1 evaluates learned part weighting/readout on the MediaPipe pixel prior rescue path. It does not add region masks, fallback rescue, SupCon, multi-seed runs, or ensemble logic.",
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
