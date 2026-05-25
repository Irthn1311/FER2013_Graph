"""Check D16 v3 dual-head run artifacts.

This checker validates performance-run contracts only. It does not make motif,
semantic-region, causal-evidence, or interpretability claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _finite_series(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return bool(values.notna().all() and values.map(math.isfinite).all())


def _finite(df: pd.DataFrame, column: str) -> bool:
    return bool(not df.empty and column in df.columns and _finite_series(df[column]))


def _has_nan(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.isnan(values).any() or (~np.isfinite(values)).any())


def _predicted_class_count(pred_count: pd.DataFrame) -> int:
    if pred_count.empty or "pred_count" not in pred_count.columns:
        return 0
    counts = pd.to_numeric(pred_count["pred_count"], errors="coerce").fillna(0)
    return int((counts > 0).sum())


def _head_counts(predictions: pd.DataFrame) -> Dict[str, int]:
    if predictions.empty or "routed_head" not in predictions.columns:
        return {}
    values = predictions["routed_head"].astype(str)
    return {
        "detected_head": int((values == "detected_head").sum()),
        "fallback_head": int((values == "fallback_head").sum()),
        "single_head": int((values == "single_head").sum()),
    }


def check_run(output_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

    train = _read_csv(output_dir / "train_log.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred_count = _read_csv(output_dir / "pred_count.csv")
    fallback = _read_csv(output_dir / "detected_vs_fallback_metrics.csv")
    predictions = _read_csv(output_dir / "predictions.csv")
    confusion = _read_csv(output_dir / "confusion_matrix.csv")
    per_class = _read_csv(output_dir / "per_class_metrics.csv")

    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"
    if not best_path.exists():
        failures.append("missing checkpoints/best.pt")
    if not last_path.exists():
        failures.append("missing checkpoints/last.pt")
    if train.empty:
        failures.append("missing train_log.csv")
    if test.empty:
        failures.append("missing test_metrics.csv")
    if predictions.empty:
        failures.append("missing predictions.csv")
    if confusion.empty:
        failures.append("missing confusion_matrix.csv")
    if per_class.empty:
        failures.append("missing per_class_metrics.csv")
    if fallback.empty:
        failures.append("missing detected_vs_fallback_metrics.csv")

    if not test.empty:
        if "checkpoint_name" not in test.columns or str(test["checkpoint_name"].iloc[-1]) != "best.pt":
            failures.append("test_metrics.csv is not from best.pt")
        for col in ["accuracy", "macro_f1", "loss"]:
            if col in test.columns and not _finite(test, col):
                failures.append(f"test {col} missing or non-finite")
    for col in ["train_loss", "val_macro_f1", "ce_loss"]:
        if col in train.columns and not _finite(train, col):
            failures.append(f"{col} missing or non-finite")
    if _has_nan(train) or _has_nan(test) or _has_nan(fallback):
        failures.append("NaN or non-finite value detected")

    if predictions.empty or "routed_head" not in predictions.columns:
        failures.append("missing routed_head column in predictions.csv")
        head_counts = {}
    else:
        head_counts = _head_counts(predictions)
        total_predictions = int(len(predictions))
        used_heads = [name for name, count in head_counts.items() if count > 0]
        if "single_head" in used_heads:
            warnings.append("predictions.csv contains single_head rows; this is expected only for v1 repeat controls")
        dual_head_count = int(head_counts.get("detected_head", 0)) + int(head_counts.get("fallback_head", 0))
        if dual_head_count > 0 and (
            int(head_counts.get("detected_head", 0)) <= 0 or int(head_counts.get("fallback_head", 0)) <= 0
        ):
            if total_predictions >= 100:
                warnings.append("full test appears to use only one dual-head branch")
            else:
                warnings.append("smoke predictions use only one dual-head branch")

    test_total = int(test["total"].iloc[-1]) if not test.empty and "total" in test.columns else int(len(predictions))
    predicted_classes = _predicted_class_count(pred_count)
    if test_total >= 100:
        if predicted_classes <= 2:
            failures.append(f"hard collapse: predicted_classes={predicted_classes}")
        elif predicted_classes < 7:
            warnings.append(f"prediction bias: predicted_classes={predicted_classes}")
    elif test_total > 0:
        warnings.append(f"small smoke test set; collapse gate is soft: total={test_total}, predicted_classes={predicted_classes}")

    detected_macro = float("nan")
    fallback_macro = float("nan")
    if not fallback.empty and "group" in fallback.columns:
        detected_rows = fallback[fallback["group"].astype(str) == "detected"]
        fallback_rows = fallback[fallback["group"].astype(str) == "fallback"]
        if detected_rows.empty or fallback_rows.empty:
            failures.append("missing detected/fallback metric groups")
        if not detected_rows.empty:
            detected_macro = float(pd.to_numeric(detected_rows["macro_f1"], errors="coerce").iloc[-1])
        if not fallback_rows.empty:
            fallback_macro = float(pd.to_numeric(fallback_rows["macro_f1"], errors="coerce").iloc[-1])
            if math.isfinite(fallback_macro) and fallback_macro < 0.35:
                warnings.append("fallback still weak")

    if any("routed_head" in item for item in failures):
        decision = "D16_V3_RUN_FAIL_NO_ROUTED_HEAD"
    elif any("NaN" in item or "non-finite" in item for item in failures):
        decision = "D16_V3_RUN_FAIL_NAN"
    elif any("hard collapse" in item for item in failures):
        decision = "D16_V3_RUN_FAIL_COLLAPSE"
    elif failures:
        decision = "D16_V3_RUN_FAIL_DATA"
    elif any("only one dual-head branch" in item for item in warnings):
        decision = "D16_V3_RUN_WARN_HEAD_UNUSED"
    elif math.isfinite(fallback_macro) and fallback_macro < 0.35:
        decision = "D16_V3_RUN_WARN_FALLBACK_STILL_WEAK"
    else:
        decision = "D16_V3_RUN_PASS"

    summary = {
        "output_dir": str(output_dir),
        "decision": decision,
        "epoch_count": int(train["epoch"].nunique()) if not train.empty and "epoch" in train.columns else 0,
        "best_val_macro_f1": float(pd.to_numeric(train.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce").max())
        if not train.empty
        else float("nan"),
        "test_macro_f1": float(pd.to_numeric(test.get("macro_f1", pd.Series(dtype=float)), errors="coerce").iloc[-1])
        if not test.empty
        else float("nan"),
        "test_accuracy": float(pd.to_numeric(test.get("accuracy", pd.Series(dtype=float)), errors="coerce").iloc[-1])
        if not test.empty
        else float("nan"),
        "detected_macro_f1": detected_macro,
        "fallback_macro_f1": fallback_macro,
        "predicted_classes": predicted_classes,
        "detected_head_count": int(head_counts.get("detected_head", 0)) if head_counts else 0,
        "fallback_head_count": int(head_counts.get("fallback_head", 0)) if head_counts else 0,
        "single_head_count": int(head_counts.get("single_head", 0)) if head_counts else 0,
        "failures": failures,
        "warnings": warnings,
    }
    (output_dir / "d16_v3_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = check_run(Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    if summary["decision"].startswith("D16_V3_RUN_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
