"""Check D16 v1 run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _finite(df: pd.DataFrame, column: str) -> bool:
    if df.empty or column not in df:
        return False
    vals = pd.to_numeric(df[column], errors="coerce")
    return bool(vals.notna().all() and vals.map(math.isfinite).all())


def check_run(output_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    train = _read_csv(output_dir / "train_log.csv")
    test = _read_csv(output_dir / "test_metrics.csv")
    pred = _read_csv(output_dir / "pred_count.csv")
    fallback = _read_csv(output_dir / "detected_vs_fallback_metrics.csv")
    predictions = _read_csv(output_dir / "predictions.csv")
    confusion = _read_csv(output_dir / "confusion_matrix.csv")
    if train.empty:
        failures.append("missing train_log.csv")
    if test.empty:
        failures.append("missing test_metrics.csv")
    if predictions.empty:
        failures.append("missing predictions.csv")
    if confusion.empty:
        failures.append("missing confusion_matrix.csv")
    if not (output_dir / "checkpoints" / "best.pt").exists():
        failures.append("missing checkpoints/best.pt")
    for col in ["train_loss", "val_macro_f1"]:
        if not _finite(train, col):
            failures.append(f"{col} missing or non-finite")
    if "supcon_loss_total" in train and not _finite(train, "supcon_loss_total"):
        failures.append("supcon_loss_total non-finite")
    if "ce_loss" in train and not _finite(train, "ce_loss"):
        failures.append("ce_loss non-finite")
    if not test.empty:
        if "checkpoint_name" not in test or str(test["checkpoint_name"].iloc[-1]) != "best.pt":
            warnings.append("test_metrics.csv is not explicitly best.pt")
        if not _finite(test, "macro_f1"):
            failures.append("test macro_f1 missing or non-finite")
    pred_classes = 0
    test_total = int(test["total"].iloc[-1]) if not test.empty and "total" in test else 0
    if not pred.empty and "pred_count" in pred:
        pred_classes = int((pd.to_numeric(pred["pred_count"], errors="coerce").fillna(0) > 0).sum())
        if test_total < 100:
            warnings.append(f"small smoke test set; collapse gate not enforced strongly: total={test_total}, predicted_classes={pred_classes}")
        elif pred_classes <= 2:
            failures.append(f"prediction collapse: predicted_classes={pred_classes}")
        elif pred_classes < 7:
            warnings.append(f"prediction bias: predicted_classes={pred_classes}")
    fallback_macro = float("nan")
    if not fallback.empty and "group" in fallback:
        fb = fallback[fallback["group"] == "fallback"]
        if not fb.empty:
            fallback_macro = float(pd.to_numeric(fb["macro_f1"], errors="coerce").iloc[-1])
            if fallback_macro < 0.35:
                warnings.append("fallback still weak")
    supcon_enabled_rows = pd.DataFrame()
    if "lambda_part_supcon_current" in train:
        supcon_enabled_rows = train[pd.to_numeric(train["lambda_part_supcon_current"], errors="coerce").fillna(0) > 0]
    if not supcon_enabled_rows.empty and "supcon_valid_pairs" in supcon_enabled_rows:
        if float(pd.to_numeric(supcon_enabled_rows["supcon_valid_pairs"], errors="coerce").fillna(0).sum()) <= 0:
            failures.append("supcon enabled but no positive pairs")
    if any("supcon enabled but no positive pairs" in item for item in failures):
        decision = "D16_V1_RUN_FAIL_SUPCON_NO_PAIRS"
    elif any("collapse" in item for item in failures):
        decision = "D16_V1_RUN_FAIL_COLLAPSE"
    elif any("non-finite" in item for item in failures):
        decision = "D16_V1_RUN_FAIL_NAN"
    elif failures:
        decision = "D16_V1_RUN_FAIL_DATA"
    elif fallback_macro < 0.35:
        decision = "D16_V1_RUN_WARN_FALLBACK_STILL_WEAK"
    else:
        decision = "D16_V1_RUN_PASS"
    summary = {
        "output_dir": str(output_dir),
        "decision": decision,
        "epoch_count": int(train["epoch"].nunique()) if not train.empty and "epoch" in train else 0,
        "best_val_macro_f1": float(pd.to_numeric(train.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce").max()) if not train.empty else float("nan"),
        "test_macro_f1": float(pd.to_numeric(test.get("macro_f1", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not test.empty else float("nan"),
        "test_accuracy": float(pd.to_numeric(test.get("accuracy", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not test.empty else float("nan"),
        "fallback_macro_f1": fallback_macro,
        "predicted_classes": pred_classes,
        "failures": failures,
        "warnings": warnings,
    }
    (output_dir / "d16_v1_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = check_run(Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    if summary["decision"].startswith("D16_V1_RUN_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
