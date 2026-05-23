"""Check D16 v0 small-train artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite_series(df: pd.DataFrame, column: str) -> bool:
    if df.empty or column not in df:
        return False
    vals = pd.to_numeric(df[column], errors="coerce")
    return bool(vals.notna().all() and vals.map(math.isfinite).all())


def _latest_pred_classes(pred: pd.DataFrame, split: str = "val") -> int:
    if pred.empty:
        return 0
    work = pred[pred["split"] == split].copy() if "split" in pred else pred.copy()
    if work.empty:
        return 0
    epoch = pd.to_numeric(work["epoch"], errors="coerce").max() if "epoch" in work else None
    if epoch is not None and not pd.isna(epoch):
        work = work[pd.to_numeric(work["epoch"], errors="coerce") == epoch]
    counts = pd.to_numeric(work.get("pred_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return int((counts > 0).sum())


def _single_metric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    vals = pd.to_numeric(df[column], errors="coerce")
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _has_columns(df: pd.DataFrame, columns: List[str]) -> bool:
    return all(column in df.columns for column in columns)


def check_run(output_dir: Path, strict: bool = False) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    ckpt_dir = output_dir / "checkpoints"
    required_files = {
        "best.pt": ckpt_dir / "best.pt",
        "last.pt": ckpt_dir / "last.pt",
        "train_log.csv": output_dir / "train_log.csv",
        "val_metrics.csv": output_dir / "val_metrics.csv",
        "test_metrics.csv": output_dir / "test_metrics.csv",
        "last_test_metrics.csv": output_dir / "last_test_metrics.csv",
        "per_class_metrics.csv": output_dir / "per_class_metrics.csv",
        "pred_count.csv": output_dir / "pred_count.csv",
        "detected_vs_fallback_metrics.csv": output_dir / "detected_vs_fallback_metrics.csv",
    }
    for name, path in required_files.items():
        if not path.exists():
            failures.append(f"missing {name}")

    train = _read_csv(output_dir / "train_log.csv")
    val = _read_csv(output_dir / "val_metrics.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    last_test = _read_csv(output_dir / "last_test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    fallback = _read_csv(output_dir / "detected_vs_fallback_metrics.csv")

    if not (ckpt_dir / "best.pt").exists():
        failures.append("missing best.pt")
    if not _finite_series(train, "train_loss"):
        failures.append("train_loss missing or non-finite")
    if not _finite_series(train, "val_macro_f1"):
        failures.append("val_macro_f1 missing or non-finite")
    if not _finite_series(train, "node_count_mean") or not _finite_series(train, "edge_count_mean"):
        failures.append("node/edge stats missing or non-finite")
    if not _finite_series(val, "macro_f1"):
        failures.append("val metrics macro_f1 missing or non-finite")
    if test.empty or not _finite_series(test, "macro_f1"):
        failures.append("test metrics missing or non-finite")
    required_test_columns = ["checkpoint_name", "checkpoint_epoch", "best_val_macro_f1"]
    if not _has_columns(test, required_test_columns):
        failures.append("test_metrics.csv missing checkpoint contract columns")
    else:
        checkpoint_name = str(test["checkpoint_name"].iloc[-1])
        if checkpoint_name != "best.pt":
            if (ckpt_dir / "best.pt").exists() and not failures and not strict:
                warnings.append("test_metrics.csv appears to use last checkpoint while best.pt exists")
            else:
                failures.append(f"test_metrics.csv checkpoint_name={checkpoint_name!r}, expected 'best.pt'")
        if not math.isfinite(_single_metric(test, "checkpoint_epoch")):
            failures.append("test_metrics.csv checkpoint_epoch missing or non-finite")
        if not math.isfinite(_single_metric(test, "best_val_macro_f1")):
            failures.append("test_metrics.csv best_val_macro_f1 missing or non-finite")
    if not last_test.empty and _has_columns(last_test, ["checkpoint_name"]):
        last_name = str(last_test["checkpoint_name"].iloc[-1])
        if last_name != "last.pt":
            warnings.append(f"last_test_metrics.csv checkpoint_name={last_name!r}, expected 'last.pt'")
    if fallback.empty or set(fallback.get("group", [])) < {"detected", "fallback"}:
        failures.append("detected_vs_fallback metrics missing detected/fallback groups")

    predicted_classes = _latest_pred_classes(pred, split="test")
    if predicted_classes == 0 and "predicted_classes" in val and not val.empty:
        predicted_classes = int(pd.to_numeric(val["predicted_classes"], errors="coerce").max())
    epoch_count = int(train["epoch"].nunique()) if not train.empty and "epoch" in train else 0
    if predicted_classes <= 2 and epoch_count >= 5:
        failures.append(f"prediction collapse: val predicted_classes={predicted_classes}")
    elif predicted_classes < 4 and epoch_count >= 5:
        warnings.append(f"prediction bias: val predicted_classes={predicted_classes}")

    best_val = float(pd.to_numeric(train["val_macro_f1"], errors="coerce").max()) if "val_macro_f1" in train and not train.empty else float("nan")
    final_train_loss = float(pd.to_numeric(train["train_loss"], errors="coerce").iloc[-1]) if "train_loss" in train and not train.empty else float("nan")
    first_train_loss = float(pd.to_numeric(train["train_loss"], errors="coerce").iloc[0]) if "train_loss" in train and not train.empty else float("nan")
    test_macro = _single_metric(test, "macro_f1")
    test_acc = _single_metric(test, "accuracy")
    last_test_macro = _single_metric(last_test, "macro_f1")
    last_test_acc = _single_metric(last_test, "accuracy")
    checkpoint_name = str(test["checkpoint_name"].iloc[-1]) if "checkpoint_name" in test and not test.empty else None
    checkpoint_epoch = int(_single_metric(test, "checkpoint_epoch")) if math.isfinite(_single_metric(test, "checkpoint_epoch")) else None

    if failures and any("collapse" in item for item in failures):
        decision = "D16_SMALL_TRAIN_FAIL_COLLAPSE"
    elif failures and any("non-finite" in item or "NaN" in item for item in failures):
        decision = "D16_SMALL_TRAIN_FAIL_NAN"
    elif failures:
        decision = "D16_SMALL_TRAIN_FAIL_DATA"
    elif any("last checkpoint" in item for item in warnings):
        decision = "D16_SMALL_TRAIN_WARN_TESTED_LAST_NOT_BEST"
    elif warnings:
        decision = "D16_SMALL_TRAIN_WARN_PRED_BIAS"
    else:
        decision = "D16_SMALL_TRAIN_PASS"

    if decision == "D16_SMALL_TRAIN_FAIL_COLLAPSE":
        final_decision = "D16_V0_SMALL_TRAIN_COLLAPSE_FIX_REQUIRED"
    elif decision.startswith("D16_SMALL_TRAIN_FAIL"):
        final_decision = "D16_V0_SMALL_TRAIN_DATA_OR_PRIOR_ISSUE"
    elif best_val >= 0.20 and predicted_classes >= 5:
        final_decision = "D16_V0_SMALL_TRAIN_PASS_READY_FOR_FULL"
    else:
        final_decision = "D16_V0_SMALL_TRAIN_PROMISING_NEEDS_TUNING"

    summary = {
        "output_dir": str(output_dir),
        "decision": decision,
        "final_decision": final_decision,
        "epoch_count": epoch_count,
        "best_val_macro_f1": best_val,
        "final_test_checkpoint": checkpoint_name,
        "checkpoint_epoch": checkpoint_epoch,
        "test_macro_f1": test_macro,
        "test_accuracy": test_acc,
        "last_test_macro_f1": last_test_macro,
        "last_test_accuracy": last_test_acc,
        "first_train_loss": first_train_loss,
        "final_train_loss": final_train_loss,
        "val_predicted_classes": predicted_classes,
        "failures": failures,
        "warnings": warnings,
    }
    return summary


def write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    pred = _read_csv(output_dir / "pred_count.csv")
    per_class = _read_csv(output_dir / "per_class_metrics.csv")
    fallback = _read_csv(output_dir / "detected_vs_fallback_metrics.csv")
    last_fallback = _read_csv(output_dir / "last_detected_vs_fallback_metrics.csv")
    train = _read_csv(output_dir / "train_log.csv")
    latest_epoch = int(train["epoch"].max()) if not train.empty and "epoch" in train else -1

    lines = [
        "# D16 v0 Small Train Report",
        "",
        "## 1. Training Health",
        f"- checker_decision: `{summary['decision']}`",
        f"- final_decision: `{summary['final_decision']}`",
        f"- epochs: {summary['epoch_count']}",
        f"- train_loss: {summary['first_train_loss']:.6f} -> {summary['final_train_loss']:.6f}",
        "",
        "## 2. Best Validation",
        f"- best_val_macro_f1: {summary['best_val_macro_f1']:.6f}",
        "",
        "## 3. Test Metrics",
        f"- final_test_checkpoint: `{summary.get('final_test_checkpoint')}`",
        f"- checkpoint_epoch: {summary.get('checkpoint_epoch')}",
        f"- test_macro_f1: {summary['test_macro_f1']:.6f}",
        f"- test_accuracy: {summary['test_accuracy']:.6f}",
        f"- last_test_macro_f1: {summary['last_test_macro_f1']:.6f}",
        f"- last_test_accuracy: {summary['last_test_accuracy']:.6f}",
        f"- best_minus_last_macro_f1: {summary['test_macro_f1'] - summary['last_test_macro_f1']:.6f}",
        f"- best_minus_last_accuracy: {summary['test_accuracy'] - summary['last_test_accuracy']:.6f}",
        "",
        "## 4. Prediction Distribution",
        f"- test_predicted_classes: {summary['val_predicted_classes']}",
    ]
    if not pred.empty:
        latest_pred = pred[pred["epoch"] == pred["epoch"].max()]
        lines.extend(["", "| split | class_id | pred_count |", "|---|---:|---:|"])
        for row in latest_pred.itertuples(index=False):
            lines.append(f"| {row.split} | {int(row.class_id)} | {int(row.pred_count)} |")

    lines.extend(["", "## 5. Per-Class F1"])
    if not per_class.empty:
        work_pc = per_class[per_class["split"] == "test"] if "split" in per_class else per_class
        latest_pc_epoch = work_pc["epoch"].max() if "epoch" in work_pc and not work_pc.empty else latest_epoch
        latest_pc = work_pc[work_pc["epoch"] == latest_pc_epoch]
        lines.extend(["| split | class_id | support | pred_count | f1 |", "|---|---:|---:|---:|---:|"])
        for row in latest_pc.itertuples(index=False):
            lines.append(f"| {row.split} | {int(row.class_id)} | {int(row.support)} | {int(row.pred_count)} | {float(row.f1):.4f} |")
    else:
        lines.append("- missing")

    lines.extend(["", "## 6. Best Checkpoint Detected Vs Fallback"])
    if not fallback.empty:
        latest_fb = fallback[fallback["epoch"] == fallback["epoch"].max()]
        lines.extend(["| split | group | total | accuracy | macro_f1 |", "|---|---|---:|---:|---:|"])
        for row in latest_fb.itertuples(index=False):
            lines.append(f"| {row.split} | {row.group} | {int(row.total)} | {float(row.accuracy):.4f} | {float(row.macro_f1):.4f} |")
    else:
        lines.append("- missing")

    lines.extend(["", "## 6b. Last Checkpoint Detected Vs Fallback"])
    if not last_fallback.empty:
        latest_last_fb = last_fallback[last_fallback["epoch"] == last_fallback["epoch"].max()]
        lines.extend(["| split | group | total | accuracy | macro_f1 |", "|---|---|---:|---:|---:|"])
        for row in latest_last_fb.itertuples(index=False):
            lines.append(f"| {row.split} | {row.group} | {int(row.total)} | {float(row.accuracy):.4f} | {float(row.macro_f1):.4f} |")
    else:
        lines.append("- missing")

    lines.extend(
        [
            "",
            "## 7. Runtime/Memory",
        ]
    )
    if not train.empty:
        epoch_time = pd.to_numeric(train.get("epoch_time_sec"), errors="coerce").mean()
        memory = pd.to_numeric(train.get("memory_reserved_mb"), errors="coerce").max()
        node_mean = pd.to_numeric(train.get("node_count_mean"), errors="coerce").mean()
        edge_mean = pd.to_numeric(train.get("edge_count_mean"), errors="coerce").mean()
        lines.extend(
            [
                f"- epoch_time_sec_mean: {float(epoch_time):.3f}",
                f"- memory_reserved_mb_max: {float(memory):.3f}",
                f"- node_count_mean: {float(node_mean):.3f}",
                f"- edge_count_mean: {float(edge_mean):.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "## 8. Failure Cases",
            *([f"- {item}" for item in summary.get("failures", [])] or ["- none"]),
            "",
            "## 9. Decision",
            f"- `{summary['final_decision']}`",
            "",
            "No full D16 training was launched. No part-aware SupCon was enabled. No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        ]
    )
    (output_dir / "D16_V0_SMALL_TRAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "d16_small_train_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = check_run(output_dir, strict=bool(args.strict))
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2))
    if str(summary["decision"]).startswith("D16_SMALL_TRAIN_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
