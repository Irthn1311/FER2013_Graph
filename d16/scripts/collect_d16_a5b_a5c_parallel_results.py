"""Collect A5b seed-repeat and A5c multiscale comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


D15_ACC = 0.645026
D15_MACRO = 0.622471
A4_ACC = 0.634717
A4_MACRO = 0.622718
A5A_ACCMON_ACC = 0.638061
A5A_ACCMON_MACRO = 0.619980
A5B_SEED42_ACC = 0.651435
A5B_SEED42_MACRO = 0.637964
CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
HARD_CLASSES = {0: "Angry", 2: "Fear", 4: "Sad", 6: "Neutral"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _run_kind(run_name: str) -> str:
    if "a5c_multiscale" in run_name:
        return "a5c_multiscale"
    if "seed43" in run_name:
        return "a5b_seed43"
    if "seed44" in run_name:
        return "a5b_seed44"
    if "seed42" in run_name and "a5b_heavy_opt" in run_name:
        return "a5b_seed42"
    return "unknown"


def _seed_from_name(run_name: str) -> int | None:
    for seed in (42, 43, 44):
        if f"seed{seed}" in run_name:
            return seed
    return None


def _metrics_row(run_dir: Path) -> Dict[str, Any]:
    summary = _read_json(run_dir / "d16_train_summary.json")
    test = (_read_rows(run_dir / "test_metrics.csv") or [{}])[0]
    last_test = (_read_rows(run_dir / "last_test_metrics.csv") or [{}])[0]
    pred_rows = _read_rows(run_dir / "pred_count.csv")
    run_name = str(summary.get("run_name") or run_dir.name)
    required = [
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "per_class_metrics.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "d16_train_summary.json",
    ]
    missing = [item for item in required if not (run_dir / item).exists()]
    group = {row.get("group", ""): row for row in _read_rows(run_dir / "detected_vs_fallback_metrics.csv")}
    train_rows = _read_rows(run_dir / "train_log.csv")
    epoch_times = [_float(row.get("epoch_time_sec")) for row in train_rows]
    epoch_times = [x for x in epoch_times if math.isfinite(x)]
    return {
        "run_name": run_name,
        "kind": _run_kind(run_name),
        "seed": _seed_from_name(run_name),
        "run_dir": str(run_dir),
        "test_accuracy": _float(summary.get("test_accuracy", test.get("accuracy"))),
        "test_macro_f1": _float(summary.get("test_macro_f1", test.get("macro_f1"))),
        "last_test_accuracy": _float(summary.get("last_test_accuracy", last_test.get("accuracy"))),
        "last_test_macro_f1": _float(summary.get("last_test_macro_f1", last_test.get("macro_f1"))),
        "best_epoch": _int(summary.get("best_epoch", test.get("checkpoint_epoch"))),
        "best_monitor_metric": summary.get("best_monitor_metric", ""),
        "best_monitor_score": _float(summary.get("best_monitor_score")),
        "predicted_classes": _int(test.get("predicted_classes")),
        "total": _int(test.get("total", summary.get("test_samples"))),
        "detected_accuracy": _float(group.get("detected", {}).get("accuracy")),
        "detected_macro_f1": _float(group.get("detected", {}).get("macro_f1")),
        "fallback_accuracy": _float(group.get("fallback", {}).get("accuracy")),
        "fallback_macro_f1": _float(group.get("fallback", {}).get("macro_f1")),
        "mean_epoch_time_sec": mean(epoch_times) if epoch_times else float("nan"),
        "last10_epoch_time_sec": mean(epoch_times[-10:]) if len(epoch_times) >= 10 else (mean(epoch_times) if epoch_times else float("nan")),
        "missing_files": ";".join(missing),
        "artifact_complete": not missing,
        "pred_count_total": sum(_int(row.get("pred_count")) for row in pred_rows),
    }


def _per_class_rows(run_dir: Path, run_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _read_rows(run_dir / "per_class_metrics.csv"):
        class_id = _int(row.get("class_id"))
        rows.append(
            {
                "run_name": run_name,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, row.get("class_name", str(class_id))),
                "support": _int(row.get("support")),
                "pred_count": _int(row.get("pred_count")),
                "precision": _float(row.get("precision")),
                "recall": _float(row.get("recall")),
                "f1": _float(row.get("f1")),
            }
        )
    return rows


def _group_rows(run_dir: Path, run_name: str) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": run_name,
            "group": row.get("group", ""),
            "total": _int(row.get("total")),
            "accuracy": _float(row.get("accuracy")),
            "macro_f1": _float(row.get("macro_f1")),
        }
        for row in _read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    ]


def _hard_mean(per_class: Iterable[Dict[str, Any]], run_name: str) -> float:
    values = [_float(row.get("f1")) for row in per_class if row.get("run_name") == run_name and _int(row.get("class_id")) in HARD_CLASSES]
    values = [x for x in values if math.isfinite(x)]
    return mean(values) if values else float("nan")


def _top_confusions(run_dir: Path, run_name: str, limit: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_rows(run_dir / "confusion_matrix.csv"):
        true_id = _int(row.get("true", row.get("true_class", row.get("class_id"))), -1)
        pred_id = _int(row.get("predicted", row.get("pred", row.get("pred_class"))), -1)
        count = _int(row.get("count"))
        support = _int(row.get("support"))
        if true_id == pred_id or count <= 0:
            continue
        rows.append(
            {
                "run_name": run_name,
                "true": true_id,
                "predicted": pred_id,
                "count": count,
                "support": support,
                "row_ratio": _float(row.get("row_ratio"), count / support if support else float("nan")),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)[:limit]


def _a5b_decision(a5b_rows: List[Dict[str, Any]]) -> str:
    complete = [row for row in a5b_rows if row.get("artifact_complete")]
    if any(_int(row.get("predicted_classes")) < 7 for row in complete):
        return "FLAG_COLLAPSE_SEED"
    if len(complete) < 3:
        return "A5B_REPEAT_PENDING"
    accs = [_float(row.get("test_accuracy")) for row in complete]
    mean_acc = mean(accs)
    seeds_over_d15 = sum(acc >= D15_ACC for acc in accs)
    macros = [_float(row.get("test_macro_f1")) for row in complete]
    if mean_acc >= 0.650:
        return "A5B_STRONG_STABLE_ROUTE"
    if mean_acc >= 0.648 and seeds_over_d15 >= 2:
        return "A5B_CONFIRMED_MAIN_ROUTE"
    if _float(next((row for row in complete if row.get("seed") == 42), {}).get("test_accuracy")) > D15_ACC and seeds_over_d15 < 2:
        return "A5B_SEED42_POSSIBLE_OUTLIER"
    if mean(macros) >= 0.63 and mean_acc < D15_ACC:
        return "A5B_GOOD_BALANCE_BUT_ACCURACY_UNSTABLE"
    return "A5B_REPEAT_INCONCLUSIVE"


def _a5c_decision(a5c: Dict[str, Any] | None, a5b_mean_acc: float | None, a5b_mean_macro: float | None, per_class: List[Dict[str, Any]]) -> str:
    if not a5c or not a5c.get("artifact_complete"):
        return "A5C_PENDING"
    if _int(a5c.get("predicted_classes")) < 7:
        return "REJECT_A5C_COLLAPSE"
    acc = _float(a5c.get("test_accuracy"))
    macro = _float(a5c.get("test_macro_f1"))
    if acc > A5B_SEED42_ACC:
        return "A5C_BEATS_A5B_SEED42"
    if acc >= 0.650 and macro >= 0.638:
        return "A5C_STRONG_SIGNAL"
    if a5b_mean_acc is not None and math.isfinite(a5b_mean_acc) and acc > a5b_mean_acc:
        return "A5C_BEATS_A5B_MEAN"
    if acc <= A5A_ACCMON_ACC:
        return "A5C_NOT_ENOUGH"
    a5c_hard = _hard_mean(per_class, str(a5c.get("run_name")))
    if math.isfinite(a5c_hard) and a5c_hard >= 0.548988 and acc <= A5B_SEED42_ACC:
        return "A5C_HARD_GAIN_NOT_ACCURACY_ROUTE"
    return "A5C_INCONCLUSIVE"


def collect(run_dirs: List[Path], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [_metrics_row(path) for path in run_dirs if path.exists()]
    per_class: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    for run_dir, run in zip([p for p in run_dirs if p.exists()], run_rows):
        run_name = str(run["run_name"])
        per_class.extend(_per_class_rows(run_dir, run_name))
        group_rows.extend(_group_rows(run_dir, run_name))
        confusion_rows.extend(_top_confusions(run_dir, run_name))
        run["hard_class_mean"] = _hard_mean(per_class, run_name)

    a5b_rows = [row for row in run_rows if str(row.get("kind", "")).startswith("a5b_seed")]
    a5b_complete = [row for row in a5b_rows if row.get("artifact_complete")]
    a5b_accs = [_float(row.get("test_accuracy")) for row in a5b_complete]
    a5b_macros = [_float(row.get("test_macro_f1")) for row in a5b_complete]
    a5b_mean_acc = mean(a5b_accs) if a5b_accs else float("nan")
    a5b_mean_macro = mean(a5b_macros) if a5b_macros else float("nan")
    a5c_row = next((row for row in run_rows if row.get("kind") == "a5c_multiscale"), None)
    a5b_decision = _a5b_decision(a5b_rows)
    a5c_decision = _a5c_decision(a5c_row, a5b_mean_acc, a5b_mean_macro, per_class)

    seed_summary = [
        {
            "metric": "test_accuracy",
            "count": len(a5b_accs),
            "mean": a5b_mean_acc,
            "std": pstdev(a5b_accs) if len(a5b_accs) > 1 else 0.0,
            "min": min(a5b_accs) if a5b_accs else float("nan"),
            "max": max(a5b_accs) if a5b_accs else float("nan"),
        },
        {
            "metric": "test_macro_f1",
            "count": len(a5b_macros),
            "mean": a5b_mean_macro,
            "std": pstdev(a5b_macros) if len(a5b_macros) > 1 else 0.0,
            "min": min(a5b_macros) if a5b_macros else float("nan"),
            "max": max(a5b_macros) if a5b_macros else float("nan"),
        },
    ]
    _write_csv(
        output_dir / "d16r_a5b_seed_repeat_summary.csv",
        seed_summary,
        ["metric", "count", "mean", "std", "min", "max"],
    )
    _write_csv(
        output_dir / "d16r_a5b_a5c_comparison.csv",
        run_rows,
        [
            "run_name",
            "kind",
            "seed",
            "test_accuracy",
            "test_macro_f1",
            "last_test_accuracy",
            "last_test_macro_f1",
            "best_epoch",
            "best_monitor_metric",
            "best_monitor_score",
            "detected_accuracy",
            "detected_macro_f1",
            "fallback_accuracy",
            "fallback_macro_f1",
            "hard_class_mean",
            "predicted_classes",
            "mean_epoch_time_sec",
            "last10_epoch_time_sec",
            "artifact_complete",
            "missing_files",
            "run_dir",
        ],
    )
    _write_csv(output_dir / "d16r_a5b_a5c_per_class.csv", per_class, ["run_name", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"])
    _write_csv(output_dir / "d16r_a5b_a5c_group_metrics.csv", group_rows, ["run_name", "group", "total", "accuracy", "macro_f1"])

    def fmt(x: Any) -> str:
        value = _float(x)
        return f"{value:.6f}" if math.isfinite(value) else ""

    lines = [
        "# D16R-A5b Seed Repeat + A5c Parallel Analysis",
        "",
        "## Run Integrity Table",
        "| run | kind | complete | predicted_classes | best_epoch | acc | macro_f1 | missing |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['run_name']} | {row['kind']} | {row['artifact_complete']} | {row['predicted_classes']} | "
            f"{row['best_epoch']} | {fmt(row['test_accuracy'])} | {fmt(row['test_macro_f1'])} | {row['missing_files']} |"
        )
    lines.extend(
        [
            "",
            "## A5b Seed Repeat Summary",
            "| metric | count | mean | std | min | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed_summary:
        lines.append(f"| {row['metric']} | {row['count']} | {fmt(row['mean'])} | {fmt(row['std'])} | {fmt(row['min'])} | {fmt(row['max'])} |")
    lines.extend(
        [
            "",
            "## A5c vs A5b Seed42",
            f"- A5b seed42 acc/macro: `{A5B_SEED42_ACC:.6f}` / `{A5B_SEED42_MACRO:.6f}`",
            f"- A5c acc/macro: `{fmt(a5c_row.get('test_accuracy') if a5c_row else None)}` / `{fmt(a5c_row.get('test_macro_f1') if a5c_row else None)}`",
            "",
            "## A5c vs A5b Seed Mean",
            f"- A5b mean acc/macro: `{fmt(a5b_mean_acc)}` / `{fmt(a5b_mean_macro)}`",
            f"- A5c minus mean acc: `{fmt((_float(a5c_row.get('test_accuracy')) - a5b_mean_acc) if a5c_row and math.isfinite(a5b_mean_acc) else None)}`",
            "",
            "## Detected/Fallback Metrics",
            "| run | group | total | accuracy | macro_f1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in group_rows:
        lines.append(f"| {row['run_name']} | {row['group']} | {row['total']} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} |")
    lines.extend(["", "## Hard-Class Mean", "| run | hard_mean |", "|---|---:|"])
    for row in run_rows:
        lines.append(f"| {row['run_name']} | {fmt(row.get('hard_class_mean'))} |")
    lines.extend(["", "## Per-Class Metrics", "| run | class | support | pred_count | f1 |", "|---|---|---:|---:|---:|"])
    for row in per_class:
        lines.append(f"| {row['run_name']} | {row['class_name']} | {row['support']} | {row['pred_count']} | {fmt(row['f1'])} |")
    lines.extend(["", "## Top Confusions", "| run | true | predicted | count | row_ratio |", "|---|---:|---:|---:|---:|"])
    for row in confusion_rows:
        lines.append(f"| {row['run_name']} | {row['true']} | {row['predicted']} | {row['count']} | {fmt(row['row_ratio'])} |")
    lines.extend(["", "## Runtime Comparison", "| run | mean_epoch_sec | last10_epoch_sec |", "|---|---:|---:|"])
    for row in run_rows:
        lines.append(f"| {row['run_name']} | {fmt(row['mean_epoch_time_sec'])} | {fmt(row['last10_epoch_time_sec'])} |")
    lines.extend(["", "## Decision", f"- A5b repeat decision: `{a5b_decision}`", f"- A5c decision: `{a5c_decision}`"])
    (output_dir / "D16R_A5B_A5C_PARALLEL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "run_count": len(run_rows),
        "a5b_decision": a5b_decision,
        "a5c_decision": a5c_decision,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()
    default_run_dirs = [
        Path("outputs/d16_runs/main_branch/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150"),
        Path("outputs/d16_runs/main_branch/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150"),
        Path("outputs/d16_runs/main_branch/d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150"),
        Path("outputs/d16_runs/main_branch/d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150"),
    ]
    run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else default_run_dirs
    summary = collect(run_dirs, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
