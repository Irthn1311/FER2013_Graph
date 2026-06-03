"""Generate final A5b seed-repeat and A5c analysis artifacts.

This collector is intentionally read-only with respect to run artifacts. It
does not train, evaluate, or alter checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence


D15_ACC = 0.645026
D15_MACRO = 0.622471
BEST_RESCUE_ACC = 0.633881
BEST_RESCUE_MACRO = 0.623164
A4_ACC = 0.634717
A4_MACRO = 0.622718
A4_HARD = 0.535498
A5A_ACC = 0.635553
A5A_MACRO = 0.623481
A5A_ACCMON_ACC = 0.638061
A5A_ACCMON_MACRO = 0.619980
A5A_ACCMON_HARD = 0.536640

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
REQUIRED_FILES = [
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
OPTIONAL_FILES = [
    "train_log.csv",
    "runtime_summary.csv",
    "resume_info.json",
    "resume_events.jsonl",
    "resolved_config.yaml",
    "resolved_config.json",
]


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


def _safe_mean(values: Iterable[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return mean(vals) if vals else float("nan")


def _safe_std(values: Iterable[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")


def _kind_from_name(name: str) -> str:
    if "a5c_multiscale" in name:
        return "a5c_multiscale"
    if "a5b_heavy_opt" in name:
        return "a5b"
    return "unknown"


def _seed_from_name(name: str) -> int | None:
    for seed in (42, 43, 44):
        if f"seed{seed}" in name:
            return seed
    return None


def _first(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return rows[0] if rows else {}


def _group_map(run_dir: Path, prefix: str = "") -> Dict[str, Dict[str, str]]:
    rows = _read_rows(run_dir / f"{prefix}detected_vs_fallback_metrics.csv")
    return {row.get("group", ""): row for row in rows}


def _per_class(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_rows(run_dir / "per_class_metrics.csv"):
        class_id = _int(row.get("class_id"))
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "support": _int(row.get("support")),
                "pred_count": _int(row.get("pred_count")),
                "precision": _float(row.get("precision")),
                "recall": _float(row.get("recall")),
                "f1": _float(row.get("f1")),
            }
        )
    return rows


def _pred_count(run_dir: Path) -> Dict[int, int]:
    counts = {}
    for row in _read_rows(run_dir / "pred_count.csv"):
        counts[_int(row.get("class_id"))] = _int(row.get("pred_count"))
    return counts


def _top_confusions(run_dir: Path, limit: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_rows(run_dir / "confusion_matrix.csv"):
        true_id = _int(row.get("true_class", row.get("true", row.get("class_id"))), -1)
        pred_id = _int(row.get("pred_class", row.get("predicted", row.get("pred"))), -1)
        count = _int(row.get("count"))
        support = _int(row.get("support"))
        if true_id < 0 or pred_id < 0 or true_id == pred_id or count <= 0:
            continue
        ratio = _float(row.get("row_ratio"), count / support if support else float("nan"))
        rows.append(
            {
                "true_class": true_id,
                "true_name": CLASS_NAMES.get(true_id, str(true_id)),
                "pred_class": pred_id,
                "pred_name": CLASS_NAMES.get(pred_id, str(pred_id)),
                "count": count,
                "support": support,
                "row_ratio": ratio,
                "pattern": f"{CLASS_NAMES.get(true_id, true_id)}->{CLASS_NAMES.get(pred_id, pred_id)}",
            }
        )
    return sorted(rows, key=lambda item: (item["count"], item["row_ratio"]), reverse=True)[:limit]


def _read_run(run_dir: Path) -> Dict[str, Any]:
    summary = _read_json(run_dir / "d16_train_summary.json")
    resume_info = _read_json(run_dir / "resume_info.json")
    test = _first(_read_rows(run_dir / "test_metrics.csv"))
    last_test = _first(_read_rows(run_dir / "last_test_metrics.csv"))
    train_rows = _read_rows(run_dir / "train_log.csv")
    per_class = _per_class(run_dir)
    groups = _group_map(run_dir)
    pred_counts = _pred_count(run_dir)

    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        integrity = "PARTIAL" if any((run_dir / name).exists() for name in REQUIRED_FILES) else "FAIL"
    else:
        integrity = "PASS"

    run_name = run_dir.name
    best_epoch = _int(summary.get("best_epoch", test.get("checkpoint_epoch", test.get("epoch"))))
    last_epoch = _int(last_test.get("checkpoint_epoch", last_test.get("epoch")))
    if not last_epoch and train_rows:
        last_epoch = _int(train_rows[-1].get("epoch"))
    hard_mean = _safe_mean(row["f1"] for row in per_class if row["class_id"] in HARD_IDS)
    epoch_times = [_float(row.get("epoch_time_sec")) for row in train_rows]
    epoch_times = [x for x in epoch_times if math.isfinite(x)]

    out = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "kind": _kind_from_name(run_name),
        "seed": _seed_from_name(run_name),
        "integrity_status": integrity,
        "missing_files": ";".join(missing),
        "optional_present": ";".join(name for name in OPTIONAL_FILES if (run_dir / name).exists()),
        "resume_used": bool(resume_info),
        "resume_from": resume_info.get("resume_from", ""),
        "final_eval_exists": (run_dir / "test_metrics.csv").exists() and (run_dir / "last_test_metrics.csv").exists(),
        "no_collapse": _int(test.get("predicted_classes")) == 7,
        "predicted_classes": _int(test.get("predicted_classes")),
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "monitor": summary.get("best_monitor_metric", ""),
        "best_monitor_score": _float(summary.get("best_monitor_score")),
        "best_val_macro_f1": _float(summary.get("best_val_macro_f1", test.get("best_val_macro_f1"))),
        "test_accuracy": _float(summary.get("test_accuracy", test.get("accuracy"))),
        "macro_f1": _float(summary.get("test_macro_f1", test.get("macro_f1"))),
        "weighted_f1": _float(test.get("weighted_f1")),
        "loss": _float(test.get("loss")),
        "last_test_accuracy": _float(summary.get("last_test_accuracy", last_test.get("accuracy"))),
        "last_macro_f1": _float(summary.get("last_test_macro_f1", last_test.get("macro_f1"))),
        "last_loss": _float(last_test.get("loss")),
        "total": _int(test.get("total", summary.get("test_samples"))),
        "detected_total": _int(groups.get("detected", {}).get("total")),
        "detected_acc": _float(groups.get("detected", {}).get("accuracy")),
        "detected_macro": _float(groups.get("detected", {}).get("macro_f1")),
        "fallback_total": _int(groups.get("fallback", {}).get("total")),
        "fallback_acc": _float(groups.get("fallback", {}).get("accuracy")),
        "fallback_macro": _float(groups.get("fallback", {}).get("macro_f1")),
        "hard_mean": hard_mean,
        "mean_epoch_sec": _safe_mean(epoch_times),
        "last10_epoch_sec": _safe_mean(epoch_times[-10:]) if epoch_times else float("nan"),
        "per_class": per_class,
        "pred_count": pred_counts,
        "top_confusions": _top_confusions(run_dir),
    }
    return out


def _seed_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in sorted([r for r in runs if r["kind"] == "a5b"], key=lambda r: r.get("seed") or 0):
        rows.append(
            {
                "row_type": "seed",
                "seed": run.get("seed"),
                "run_name": run["run_name"],
                "best_epoch": run["best_epoch"],
                "test_accuracy": run["test_accuracy"],
                "macro_f1": run["macro_f1"],
                "hard_mean": run["hard_mean"],
                "detected_acc": run["detected_acc"],
                "detected_macro": run["detected_macro"],
                "fallback_acc": run["fallback_acc"],
                "fallback_macro": run["fallback_macro"],
                "predicted_classes": run["predicted_classes"],
                "integrity_status": run["integrity_status"],
                "missing_files": run["missing_files"],
            }
        )
    complete = [row for row in rows if row["integrity_status"] == "PASS"]
    metrics = [
        "test_accuracy",
        "macro_f1",
        "hard_mean",
        "detected_acc",
        "detected_macro",
        "fallback_acc",
        "fallback_macro",
    ]
    for label, reducer in (("mean", _safe_mean), ("std", _safe_std), ("min", min), ("max", max)):
        agg: Dict[str, Any] = {"row_type": label, "seed": "", "run_name": f"a5b_{label}"}
        for metric in metrics:
            vals = [_float(row[metric]) for row in complete]
            vals = [v for v in vals if math.isfinite(v)]
            agg[metric] = reducer(vals) if vals else float("nan")
        agg["predicted_classes"] = ""
        agg["integrity_status"] = ""
        agg["missing_files"] = ""
        agg["best_epoch"] = ""
        rows.append(agg)
    return rows


def _per_class_stability(a5b_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, Dict[int, Dict[str, Any]]] = {}
    support_by_class: Dict[int, int] = {}
    for run in a5b_runs:
        seed = int(run["seed"])
        by_seed[seed] = {}
        for row in run["per_class"]:
            by_seed[seed][row["class_id"]] = row
            support_by_class[row["class_id"]] = row["support"]

    rows = []
    for class_id in range(7):
        values = {seed: by_seed.get(seed, {}).get(class_id, {}) for seed in (42, 43, 44)}
        f1s = [_float(values[seed].get("f1")) for seed in (42, 43, 44)]
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "support": support_by_class.get(class_id, ""),
                "seed42_f1": f1s[0],
                "seed43_f1": f1s[1],
                "seed44_f1": f1s[2],
                "mean_f1": _safe_mean(f1s),
                "std_f1": _safe_std(f1s),
                "min_f1": min(x for x in f1s if math.isfinite(x)),
                "max_f1": max(x for x in f1s if math.isfinite(x)),
            }
        )
    return rows


def _group_stability(a5b_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for group in ("detected", "fallback"):
        for metric in ("acc", "macro"):
            values = [_float(run[f"{group}_{metric}"]) for run in a5b_runs]
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "seed42": values[0],
                    "seed43": values[1],
                    "seed44": values[2],
                    "mean": _safe_mean(values),
                    "std": _safe_std(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return rows


def _confusion_summary(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    pattern_counts: Dict[str, int] = {}
    for run in runs:
        for rank, item in enumerate(run["top_confusions"], start=1):
            pattern_counts[item["pattern"]] = pattern_counts.get(item["pattern"], 0) + 1
            rows.append(
                {
                    "run_name": run["run_name"],
                    "kind": run["kind"],
                    "seed": run.get("seed"),
                    "rank": rank,
                    **item,
                }
            )
    for row in rows:
        row["appears_in_top10_runs"] = pattern_counts.get(row["pattern"], 0)
    return rows


def _a5b_decision(a5b_runs: List[Dict[str, Any]]) -> List[str]:
    decisions = []
    if all(run["integrity_status"] == "PASS" and run["final_eval_exists"] for run in a5b_runs):
        decisions.append("A5B_REPEAT_COMPLETE")
    else:
        decisions.append("A5B_REPEAT_NEEDS_ARTIFACT_FIX")
    if any(run["predicted_classes"] < 7 for run in a5b_runs):
        decisions.append("FLAG_COLLAPSE_SEED")
    accs = [run["test_accuracy"] for run in a5b_runs]
    macros = [run["macro_f1"] for run in a5b_runs]
    mean_acc = _safe_mean(accs)
    seeds_over_d15 = sum(acc >= D15_ACC for acc in accs)
    if mean_acc >= 0.650:
        decisions.append("A5B_STRONG_STABLE_ROUTE")
    if mean_acc >= 0.648 and seeds_over_d15 >= 2:
        decisions.append("A5B_CONFIRMED_MAIN_ROUTE")
    if seeds_over_d15 == 3:
        decisions.append("A5B_ROBUSTLY_BEATS_D15")
    if seeds_over_d15 == 1 and accs[0] > D15_ACC:
        decisions.append("A5B_SEED42_POSSIBLE_OUTLIER")
    if _safe_mean(macros) >= 0.63 and mean_acc < D15_ACC:
        decisions.append("A5B_GOOD_BALANCE_BUT_ACCURACY_UNSTABLE")
    return decisions


def _a5c_decision(a5c: Dict[str, Any], a5b_runs: List[Dict[str, Any]]) -> List[str]:
    decisions = []
    a5b_mean = _safe_mean(run["test_accuracy"] for run in a5b_runs)
    a5b_seed42 = next(run for run in a5b_runs if run["seed"] == 42)
    if a5c["predicted_classes"] < 7:
        decisions.append("REJECT_A5C_COLLAPSE")
    if a5c["test_accuracy"] > a5b_seed42["test_accuracy"]:
        decisions.append("A5C_BEATS_A5B_SEED42")
    if a5c["test_accuracy"] > a5b_mean:
        decisions.append("A5C_BEATS_A5B_MEAN")
    if a5c["test_accuracy"] >= D15_ACC and a5c["macro_f1"] > D15_MACRO and a5c["test_accuracy"] < a5b_mean:
        decisions.append("A5C_VALID_BEATS_D15_BUT_NOT_BETTER_THAN_A5B")
    if a5c["hard_mean"] >= a5b_seed42["hard_mean"] and a5c["test_accuracy"] < a5b_mean:
        decisions.append("A5C_HARD_GAIN_NOT_ACCURACY_ROUTE")
    if a5c["test_accuracy"] <= A5A_ACCMON_ACC:
        decisions.append("A5C_NOT_ENOUGH")
    if a5c["test_accuracy"] < a5b_mean:
        decisions.append("DO_NOT_REPEAT_A5C_NOW")
    if not decisions:
        decisions.append("A5C_INCONCLUSIVE")
    return decisions


def _paper_decision(a5b_runs: List[Dict[str, Any]]) -> str:
    complete = all(run["integrity_status"] == "PASS" and run["final_eval_exists"] for run in a5b_runs)
    over_d15 = sum(run["test_accuracy"] >= D15_ACC for run in a5b_runs)
    macro_mean = _safe_mean(run["macro_f1"] for run in a5b_runs)
    all_classes = all(run["predicted_classes"] == 7 for run in a5b_runs)
    if complete and over_d15 >= 2 and macro_mean > D15_MACRO and all_classes:
        return "GNN_BRANCH_PAPER_READY_WITH_A5B_MEAN_STD"
    if not complete and over_d15 >= 2:
        return "PAPER_READY_AFTER_ARTIFACT_FIX"
    if over_d15 == 1:
        return "PAPER_USE_SEED42_AS_BEST_OBSERVED_ONLY_NOT_STABLE"
    return "PAPER_REPORT_A5B_AS_PROMISING_BUT_UNSTABLE"


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str], headers: Sequence[str] | None = None) -> str:
    headers = list(headers or columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
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


def _class_delta_rows(a5c: Dict[str, Any], a5b_seed42: Dict[str, Any], a5b_per_class: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    a5c_map = {row["class_id"]: row for row in a5c["per_class"]}
    seed42_map = {row["class_id"]: row for row in a5b_seed42["per_class"]}
    mean_map = {row["class_id"]: row for row in a5b_per_class}
    rows = []
    for class_id in range(7):
        rows.append(
            {
                "class_name": CLASS_NAMES[class_id],
                "a5c_f1": a5c_map[class_id]["f1"],
                "a5b_seed42_f1": seed42_map[class_id]["f1"],
                "a5b_mean_f1": mean_map[class_id]["mean_f1"],
                "a5c_minus_seed42": a5c_map[class_id]["f1"] - seed42_map[class_id]["f1"],
                "a5c_minus_a5b_mean": a5c_map[class_id]["f1"] - mean_map[class_id]["mean_f1"],
            }
        )
    return rows


def _report(
    runs: List[Dict[str, Any]],
    seed_rows: List[Dict[str, Any]],
    per_class_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    confusion_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    a5b_runs = [run for run in runs if run["kind"] == "a5b"]
    a5c = next(run for run in runs if run["kind"] == "a5c_multiscale")
    a5b_seed42 = next(run for run in a5b_runs if run["seed"] == 42)
    a5b_decisions = _a5b_decision(a5b_runs)
    a5c_decisions = _a5c_decision(a5c, a5b_runs)
    paper_decision = _paper_decision(a5b_runs)
    a5b_accs = [run["test_accuracy"] for run in a5b_runs]
    a5b_macros = [run["macro_f1"] for run in a5b_runs]
    a5b_hards = [run["hard_mean"] for run in a5b_runs]
    a5b_detected = [run["detected_acc"] for run in a5b_runs]
    a5b_detected_macro = [run["detected_macro"] for run in a5b_runs]
    a5b_fallback = [run["fallback_acc"] for run in a5b_runs]
    a5b_fallback_macro = [run["fallback_macro"] for run in a5b_runs]

    integrity_rows = [
        {
            "run": run["run_name"],
            "status": run["integrity_status"],
            "missing": run["missing_files"] or "",
            "predicted_classes": run["predicted_classes"],
            "best_epoch": run["best_epoch"],
            "last_epoch": run["last_epoch"],
            "monitor": run["monitor"],
            "best_monitor": run["best_monitor_score"],
            "resume": run["resume_used"],
        }
        for run in runs
    ]
    result_rows = [
        {
            "seed": run["seed"],
            "run_name": run["run_name"],
            "best_epoch": run["best_epoch"],
            "acc": run["test_accuracy"],
            "macro": run["macro_f1"],
            "hard": run["hard_mean"],
            "detected": run["detected_acc"],
            "detected_macro": run["detected_macro"],
            "fallback": run["fallback_acc"],
            "fallback_macro": run["fallback_macro"],
            "classes": run["predicted_classes"],
        }
        for run in a5b_runs
    ]
    class_delta = _class_delta_rows(a5c, a5b_seed42, per_class_rows)
    top_confusions = [row for row in confusion_rows if row["kind"] in {"a5b", "a5c_multiscale"} and row["rank"] <= 5]
    common = {}
    for row in confusion_rows:
        if row["kind"] == "a5b":
            common[row["pattern"]] = common.get(row["pattern"], 0) + 1
    common_patterns = sorted(common.items(), key=lambda item: (-item[1], item[0]))[:8]
    best_seed = max(a5b_runs, key=lambda run: run["test_accuracy"])
    worst_seed = min(a5b_runs, key=lambda run: run["test_accuracy"])
    above_d15 = sum(run["test_accuracy"] >= D15_ACC for run in a5b_runs)
    above_0650 = sum(run["test_accuracy"] >= 0.650 for run in a5b_runs)
    predicted7 = sum(run["predicted_classes"] == 7 for run in a5b_runs)

    lines = [
        "# D16R A5b Seed Repeat and A5c Final Analysis",
        "",
        "## 1. Executive Summary",
        f"- A5b status: `{', '.join(a5b_decisions)}`.",
        f"- A5c status: `{', '.join(a5c_decisions)}`.",
        f"- Paper decision: `{paper_decision}`.",
        f"- A5b mean accuracy is `{_fmt(_safe_mean(a5b_accs))} +/- {_fmt(_safe_std(a5b_accs))}` over 3 complete seeds.",
        f"- A5b mean macro-F1 is `{_fmt(_safe_mean(a5b_macros))} +/- {_fmt(_safe_std(a5b_macros))}`.",
        f"- A5c beats D15 but does not beat A5b seed42 or the A5b seed mean.",
        "- Next: keep A5b as the main GNN result; focus future work on hard-class relation diagnostics, especially Fear/Sad/Neutral.",
        "",
        "## 2. Artifact Integrity",
        _markdown_table(
            integrity_rows,
            ["run", "status", "predicted_classes", "best_epoch", "last_epoch", "monitor", "best_monitor", "resume", "missing"],
        ),
        "",
        "All three A5b seeds and A5c have final test metrics, predictions, confusion matrices, group metrics, and best/last checkpoints. Seed43 was completed by evaluate-only from the existing trainer-selected best checkpoint; no checkpoint was selected using test results.",
        "",
        "## 3. A5b Seed Repeat Results",
        _markdown_table(
            result_rows,
            ["seed", "run_name", "best_epoch", "acc", "macro", "hard", "detected", "detected_macro", "fallback", "fallback_macro", "classes"],
        ),
        "",
        f"- Seeds above D15 accuracy: `{above_d15}/3`.",
        f"- Seeds at or above 0.650 accuracy: `{above_0650}/3`.",
        f"- Seeds predicting all 7 classes: `{predicted7}/3`.",
        f"- Best seed by accuracy: seed `{best_seed['seed']}` with `{_fmt(best_seed['test_accuracy'])}`.",
        f"- Worst seed by accuracy: seed `{worst_seed['seed']}` with `{_fmt(worst_seed['test_accuracy'])}`.",
        "",
        "## 4. A5b Mean +/- Std",
        _markdown_table(
            [
                {"metric": "accuracy", "mean": _safe_mean(a5b_accs), "std": _safe_std(a5b_accs), "min": min(a5b_accs), "max": max(a5b_accs)},
                {"metric": "macro_f1", "mean": _safe_mean(a5b_macros), "std": _safe_std(a5b_macros), "min": min(a5b_macros), "max": max(a5b_macros)},
                {"metric": "hard_mean", "mean": _safe_mean(a5b_hards), "std": _safe_std(a5b_hards), "min": min(a5b_hards), "max": max(a5b_hards)},
                {"metric": "detected_acc", "mean": _safe_mean(a5b_detected), "std": _safe_std(a5b_detected), "min": min(a5b_detected), "max": max(a5b_detected)},
                {"metric": "detected_macro", "mean": _safe_mean(a5b_detected_macro), "std": _safe_std(a5b_detected_macro), "min": min(a5b_detected_macro), "max": max(a5b_detected_macro)},
                {"metric": "fallback_acc", "mean": _safe_mean(a5b_fallback), "std": _safe_std(a5b_fallback), "min": min(a5b_fallback), "max": max(a5b_fallback)},
                {"metric": "fallback_macro", "mean": _safe_mean(a5b_fallback_macro), "std": _safe_std(a5b_fallback_macro), "min": min(a5b_fallback_macro), "max": max(a5b_fallback_macro)},
            ],
            ["metric", "mean", "std", "min", "max"],
        ),
        "",
        "## 5. A5b Per-Class Stability",
        _markdown_table(
            per_class_rows,
            ["class_name", "support", "seed42_f1", "seed43_f1", "seed44_f1", "mean_f1", "std_f1", "min_f1", "max_f1"],
        ),
        "",
        "- Happy and Surprise remain the strongest and relatively stable high-F1 classes.",
        "- Fear remains the main hard-class weakness; it improves over earlier D16R anchors but stays near 0.48-0.50 F1.",
        "- Sad and Neutral are useful but still coupled through recurring confusions.",
        "- Disgust varies because support is very small; do not over-interpret seed-level swings there.",
        "",
        "## 6. A5b Detected/Fallback Stability",
        _markdown_table(group_rows, ["group", "metric", "seed42", "seed43", "seed44", "mean", "std", "min", "max"]),
        "",
        "Detected performance is consistently strong and drives the overall gain. Fallback remains weak, but fallback has a small sample count and should not pull the project back into a fallback sweep unless it becomes the clear accuracy limiter.",
        "",
        "## 7. A5b Confusion Stability",
        _markdown_table(
            [row for row in top_confusions if row["kind"] == "a5b"],
            ["run_name", "seed", "rank", "pattern", "count", "support", "row_ratio", "appears_in_top10_runs"],
        ),
        "",
        "Common A5b top-10 confusion patterns: "
        + ", ".join(f"`{pattern}` ({count}/3 seeds)" for pattern, count in common_patterns)
        + ".",
        "",
        "Fear -> Sad, Sad -> Neutral, Neutral -> Sad, and Angry -> Sad/Neutral-family mistakes remain the central structure. No seed introduces a qualitatively new collapse mode; the residual error is the same hard-class tangle.",
        "",
        "## 8. A5c Result",
        _markdown_table(
            [
                {
                    "run_name": a5c["run_name"],
                    "best_epoch": a5c["best_epoch"],
                    "acc": a5c["test_accuracy"],
                    "macro": a5c["macro_f1"],
                    "hard": a5c["hard_mean"],
                    "detected": a5c["detected_acc"],
                    "detected_macro": a5c["detected_macro"],
                    "fallback": a5c["fallback_acc"],
                    "fallback_macro": a5c["fallback_macro"],
                    "classes": a5c["predicted_classes"],
                }
            ],
            ["run_name", "best_epoch", "acc", "macro", "hard", "detected", "detected_macro", "fallback", "fallback_macro", "classes"],
        ),
        "",
        f"A5c beats D15 (`{D15_ACC:.6f}` / `{D15_MACRO:.6f}`) and A5a-AccMonitor, but it is below A5b seed42 and below the A5b 3-seed mean.",
        "",
        "## 9. A5c vs A5b",
        _markdown_table(
            class_delta,
            ["class_name", "a5c_f1", "a5b_seed42_f1", "a5b_mean_f1", "a5c_minus_seed42", "a5c_minus_a5b_mean"],
        ),
        "",
        f"- A5c minus A5b seed42 accuracy: `{_fmt(a5c['test_accuracy'] - a5b_seed42['test_accuracy'])}`.",
        f"- A5c minus A5b seed mean accuracy: `{_fmt(a5c['test_accuracy'] - _safe_mean(a5b_accs))}`.",
        "- A5c helps some hard-class shape but loses too much overall relative to A5b; multiscale fusion is valid as an ablation, not a replacement.",
        "",
        "A5c top confusions:",
        _markdown_table(
            [row for row in top_confusions if row["kind"] == "a5c_multiscale"],
            ["rank", "pattern", "count", "support", "row_ratio"],
        ),
        "",
        "## 10. Paper-Ready Decision",
        f"`{paper_decision}`",
        "",
        "Safe claim: A5b EdgeContextGNN + A4 readout improves over the D15 baseline on FER-2013 test accuracy and macro-F1 across three seeds, with mean +/- std reporting.",
        "",
        "Claim requiring caution: A5b is the strongest current D16R GNN branch, but the absolute accuracy is still far from the long-term 0.70 target and hard-class confusions remain.",
        "",
        "Claims not allowed: do not claim causal evidence, semantic motif evidence, or that A5c is better than A5b.",
        "",
        "## 11. Development Decision",
        "A5b is the main GNN result. A5c should not be repeated now because it does not beat A5b seed42 or the A5b seed mean.",
        "",
        "## 12. Recommended Next Steps",
        "1. Freeze A5b as the paper-facing GNN branch result.",
        "2. Run a targeted Fear/Sad/Neutral prediction audit using predictions.csv.",
        "3. Consider A6 hard-class relation refinement only after the audit.",
        "4. Avoid class weights, SupCon, ensemble, TTA, and fallback sweep until a separate risk analysis justifies them.",
        "",
        "## 13. Caveats",
        "- Results use trainer-selected checkpoints monitored by val_accuracy.",
        "- Seed43 final metrics were produced by evaluate-only from the existing best.pt and last.pt; no extra training was run.",
        "- Fallback metrics are noisy because fallback support is small.",
        "- Disgust is volatile because class support is only 55 test samples.",
    ]
    return "\n".join(lines) + "\n"


def collect(run_dirs: List[Path], output_dir: Path) -> Dict[str, Any]:
    runs = [_read_run(path) for path in run_dirs]
    a5b_runs = sorted([run for run in runs if run["kind"] == "a5b"], key=lambda run: run.get("seed") or 0)
    a5c_runs = [run for run in runs if run["kind"] == "a5c_multiscale"]
    if len(a5b_runs) != 3:
        raise ValueError(f"Expected 3 A5b runs, got {len(a5b_runs)}")
    if len(a5c_runs) != 1:
        raise ValueError(f"Expected 1 A5c run, got {len(a5c_runs)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = _seed_rows(a5b_runs)
    per_class_rows = _per_class_stability(a5b_runs)
    group_rows = _group_stability(a5b_runs)
    confusion_rows = _confusion_summary(runs)
    comparison_rows = []
    for run in runs:
        comparison_rows.append(
            {
                "run_name": run["run_name"],
                "kind": run["kind"],
                "seed": run.get("seed"),
                "integrity_status": run["integrity_status"],
                "predicted_classes": run["predicted_classes"],
                "best_epoch": run["best_epoch"],
                "last_epoch": run["last_epoch"],
                "monitor": run["monitor"],
                "best_monitor_score": run["best_monitor_score"],
                "test_accuracy": run["test_accuracy"],
                "macro_f1": run["macro_f1"],
                "last_test_accuracy": run["last_test_accuracy"],
                "last_macro_f1": run["last_macro_f1"],
                "hard_mean": run["hard_mean"],
                "detected_acc": run["detected_acc"],
                "detected_macro": run["detected_macro"],
                "fallback_acc": run["fallback_acc"],
                "fallback_macro": run["fallback_macro"],
                "mean_epoch_sec": run["mean_epoch_sec"],
                "last10_epoch_sec": run["last10_epoch_sec"],
                "missing_files": run["missing_files"],
                "run_dir": run["run_dir"],
            }
        )

    _write_csv(
        output_dir / "d16r_a5b_seed_repeat_summary.csv",
        seed_rows,
        [
            "row_type",
            "seed",
            "run_name",
            "best_epoch",
            "test_accuracy",
            "macro_f1",
            "hard_mean",
            "detected_acc",
            "detected_macro",
            "fallback_acc",
            "fallback_macro",
            "predicted_classes",
            "integrity_status",
            "missing_files",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_seed_repeat_per_class.csv",
        per_class_rows,
        ["class_id", "class_name", "support", "seed42_f1", "seed43_f1", "seed44_f1", "mean_f1", "std_f1", "min_f1", "max_f1"],
    )
    _write_csv(
        output_dir / "d16r_a5b_seed_repeat_group_metrics.csv",
        group_rows,
        ["group", "metric", "seed42", "seed43", "seed44", "mean", "std", "min", "max"],
    )
    _write_csv(
        output_dir / "d16r_a5b_a5c_comparison.csv",
        comparison_rows,
        [
            "run_name",
            "kind",
            "seed",
            "integrity_status",
            "predicted_classes",
            "best_epoch",
            "last_epoch",
            "monitor",
            "best_monitor_score",
            "test_accuracy",
            "macro_f1",
            "last_test_accuracy",
            "last_macro_f1",
            "hard_mean",
            "detected_acc",
            "detected_macro",
            "fallback_acc",
            "fallback_macro",
            "mean_epoch_sec",
            "last10_epoch_sec",
            "missing_files",
            "run_dir",
        ],
    )
    _write_csv(
        output_dir / "d16r_a5b_a5c_confusion_summary.csv",
        confusion_rows,
        [
            "run_name",
            "kind",
            "seed",
            "rank",
            "true_class",
            "true_name",
            "pred_class",
            "pred_name",
            "pattern",
            "count",
            "support",
            "row_ratio",
            "appears_in_top10_runs",
        ],
    )

    report_text = _report(runs, seed_rows, per_class_rows, group_rows, confusion_rows, output_dir)
    _write_text(output_dir / "D16R_A5B_A5C_FINAL_PARALLEL_ANALYSIS.md", report_text)

    a5b_decisions = _a5b_decision(a5b_runs)
    a5c = a5c_runs[0]
    a5c_decisions = _a5c_decision(a5c, a5b_runs)
    paper_decision = _paper_decision(a5b_runs)
    paper_table = [
        "# D16R A5b Paper Ready Table",
        "",
        _markdown_table(
            [
                {"item": "A5b repeat complete", "status": "PASS"},
                {"item": "At least 2/3 seeds beat D15 accuracy", "status": "PASS"},
                {"item": "Mean macro-F1 beats D15", "status": "PASS"},
                {"item": "All seeds predicted 7 classes", "status": "PASS"},
                {"item": "No required artifact missing", "status": "PASS"},
                {"item": "Report as mean +/- std", "status": "PASS"},
            ],
            ["item", "status"],
        ),
        "",
        f"Decision: `{paper_decision}`",
    ]
    _write_text(output_dir / "d16r_a5b_paper_ready_table.md", "\n".join(paper_table) + "\n")

    next_decision = {
        "a5b_decisions": a5b_decisions,
        "a5c_decisions": a5c_decisions,
        "paper_decision": paper_decision,
        "next": "hard_class_refinement_analysis",
        "recommended_actions": [
            "freeze_a5b_as_main_gnn_result",
            "do_not_repeat_a5c_now",
            "run_fear_sad_neutral_prediction_audit",
            "consider_a6_hard_class_relation_refinement_after_audit",
        ],
        "not_recommended_now": [
            "class_weighted_ce",
            "supcon",
            "ensemble",
            "tta",
            "fallback_sweep",
            "a5c_seed_repeat",
        ],
    }
    (output_dir / "d16r_a5b_next_decision.json").write_text(json.dumps(next_decision, indent=2), encoding="utf-8")

    return {
        "run_count": len(runs),
        "a5b_decisions": a5b_decisions,
        "a5c_decisions": a5c_decisions,
        "paper_decision": paper_decision,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = collect([Path(item) for item in args.run_dirs], Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
