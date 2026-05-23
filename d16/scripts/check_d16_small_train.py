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


def check_run(output_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    ckpt_dir = output_dir / "checkpoints"
    required_files = {
        "best.pt": ckpt_dir / "best.pt",
        "last.pt": ckpt_dir / "last.pt",
        "train_log.csv": output_dir / "train_log.csv",
        "val_metrics.csv": output_dir / "val_metrics.csv",
        "test_metrics.csv": output_dir / "test_metrics.csv",
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
    pred = _read_csv(output_dir / "pred_count.csv")
    fallback = _read_csv(output_dir / "detected_vs_fallback_metrics.csv")

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
    if fallback.empty or set(fallback.get("group", [])) < {"detected", "fallback"}:
        failures.append("detected_vs_fallback metrics missing detected/fallback groups")

    predicted_classes = _latest_pred_classes(pred, split="val")
    epoch_count = int(train["epoch"].nunique()) if not train.empty and "epoch" in train else 0
    if predicted_classes <= 2 and epoch_count >= 5:
        failures.append(f"prediction collapse: val predicted_classes={predicted_classes}")
    elif predicted_classes < 4 and epoch_count >= 5:
        warnings.append(f"prediction bias: val predicted_classes={predicted_classes}")

    best_val = float(pd.to_numeric(train["val_macro_f1"], errors="coerce").max()) if "val_macro_f1" in train and not train.empty else float("nan")
    final_train_loss = float(pd.to_numeric(train["train_loss"], errors="coerce").iloc[-1]) if "train_loss" in train and not train.empty else float("nan")
    first_train_loss = float(pd.to_numeric(train["train_loss"], errors="coerce").iloc[0]) if "train_loss" in train and not train.empty else float("nan")
    test_macro = float(pd.to_numeric(test["macro_f1"], errors="coerce").iloc[-1]) if "macro_f1" in test and not test.empty else float("nan")
    test_acc = float(pd.to_numeric(test["accuracy"], errors="coerce").iloc[-1]) if "accuracy" in test and not test.empty else float("nan")

    if failures and any("collapse" in item for item in failures):
        decision = "D16_SMALL_TRAIN_FAIL_COLLAPSE"
    elif failures and any("non-finite" in item or "NaN" in item for item in failures):
        decision = "D16_SMALL_TRAIN_FAIL_NAN"
    elif failures:
        decision = "D16_SMALL_TRAIN_FAIL_DATA"
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
        "test_macro_f1": test_macro,
        "test_accuracy": test_acc,
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
        f"- test_macro_f1: {summary['test_macro_f1']:.6f}",
        f"- test_accuracy: {summary['test_accuracy']:.6f}",
        "",
        "## 4. Prediction Distribution",
        f"- val_predicted_classes: {summary['val_predicted_classes']}",
    ]
    if not pred.empty:
        latest_pred = pred[pred["epoch"] == pred["epoch"].max()]
        lines.extend(["", "| split | class_id | pred_count |", "|---|---:|---:|"])
        for row in latest_pred.itertuples(index=False):
            lines.append(f"| {row.split} | {int(row.class_id)} | {int(row.pred_count)} |")

    lines.extend(["", "## 5. Per-Class F1"])
    if not per_class.empty:
        latest_pc = per_class[per_class["epoch"] == latest_epoch]
        lines.extend(["| split | class_id | support | pred_count | f1 |", "|---|---:|---:|---:|---:|"])
        for row in latest_pc.itertuples(index=False):
            lines.append(f"| {row.split} | {int(row.class_id)} | {int(row.support)} | {int(row.pred_count)} | {float(row.f1):.4f} |")
    else:
        lines.append("- missing")

    lines.extend(["", "## 6. Detected Vs Fallback"])
    if not fallback.empty:
        latest_fb = fallback[fallback["epoch"] == fallback["epoch"].max()]
        lines.extend(["| split | group | total | accuracy | macro_f1 |", "|---|---|---:|---:|---:|"])
        for row in latest_fb.itertuples(index=False):
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
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = check_run(output_dir)
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2))
    if str(summary["decision"]).startswith("D16_SMALL_TRAIN_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
