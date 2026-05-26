"""Check D16 v4 routed fallback-patch run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def finite_col(df: pd.DataFrame, col: str) -> bool:
    if df.empty or col not in df.columns:
        return False
    vals = pd.to_numeric(df[col], errors="coerce")
    return bool(vals.notna().all() and np.isfinite(vals.to_numpy(dtype=float)).all())


def has_nan(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.empty:
        return False
    vals = numeric.to_numpy(dtype=float)
    return bool(np.isnan(vals).any() or (~np.isfinite(vals)).any())


def check_run(output_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    train = read_csv(output_dir / "train_log.csv")
    test = read_csv(output_dir / "test_metrics.csv")
    pred_count = read_csv(output_dir / "pred_count.csv")
    fallback = read_csv(output_dir / "detected_vs_fallback_metrics.csv")
    predictions = read_csv(output_dir / "predictions.csv")
    confusion = read_csv(output_dir / "confusion_matrix.csv")
    per_class = read_csv(output_dir / "per_class_metrics.csv")

    if not (output_dir / "checkpoints" / "best.pt").exists():
        failures.append("missing checkpoints/best.pt")
    if not (output_dir / "checkpoints" / "last.pt").exists():
        failures.append("missing checkpoints/last.pt")
    for name, df in {
        "train_log.csv": train,
        "test_metrics.csv": test,
        "predictions.csv": predictions,
        "confusion_matrix.csv": confusion,
        "per_class_metrics.csv": per_class,
        "detected_vs_fallback_metrics.csv": fallback,
    }.items():
        if df.empty:
            failures.append(f"missing {name}")

    if not test.empty:
        if "checkpoint_name" not in test.columns or str(test["checkpoint_name"].iloc[-1]) != "best.pt":
            failures.append("test_metrics.csv is not from best.pt")
        for col in ["accuracy", "macro_f1", "loss"]:
            if col in test.columns and not finite_col(test, col):
                failures.append(f"test {col} missing or non-finite")
    for col in ["train_loss", "ce_loss"]:
        if col in train.columns and not finite_col(train, col):
            failures.append(f"{col} missing or non-finite")
    if "fallback_token_count_mean" not in train.columns or not finite_col(train, "fallback_token_count_mean"):
        failures.append("fallback_token_count_mean missing or non-finite")
    if has_nan(train) or has_nan(test) or has_nan(fallback):
        failures.append("NaN or non-finite value detected")

    routed_path_missing = predictions.empty or "routed_path" not in predictions.columns
    detected_path_count = 0
    fallback_path_count = 0
    wrong_route_count: Any = ""
    if routed_path_missing:
        failures.append("missing routed_path column in predictions.csv")
    else:
        paths = predictions["routed_path"].astype(str)
        detected_path_count = int((paths == "detected_face_path").sum())
        fallback_path_count = int(paths.isin(["fallback_grid_path", "fallback_transformer_path"]).sum())
        if "landmark_missing_flag" in predictions.columns:
            missing = pd.to_numeric(predictions["landmark_missing_flag"], errors="coerce")
            wrong_route_count = int(
                ((missing == 0) & (paths != "detected_face_path")).sum()
                + ((missing == 1) & (~paths.isin(["fallback_grid_path", "fallback_transformer_path"]))).sum()
            )
            if wrong_route_count:
                failures.append(f"wrong routed_path count={wrong_route_count}")
            detected_count = int((missing == 0).sum())
            fallback_count = int((missing == 1).sum())
        else:
            detected_count = detected_path_count
            fallback_count = fallback_path_count
        if detected_count > 0 and fallback_count > 0 and (detected_path_count == 0 or fallback_path_count == 0):
            warnings.append("test contains both groups but only one routed path appears")

    predicted_classes = 0
    epoch_count = int(train["epoch"].nunique()) if not train.empty and "epoch" in train.columns else 0
    if not pred_count.empty and "pred_count" in pred_count.columns:
        counts = pd.to_numeric(pred_count["pred_count"], errors="coerce").fillna(0)
        predicted_classes = int((counts > 0).sum())
        total = float(counts.sum())
        max_ratio = float(counts.max() / total) if total > 0 else float("nan")
        if predicted_classes <= 2:
            if epoch_count <= 3:
                warnings.append(f"smoke collapse warning: predicted_classes={predicted_classes}")
            else:
                failures.append(f"hard collapse: predicted_classes={predicted_classes}")
        elif predicted_classes < 7 or (math.isfinite(max_ratio) and max_ratio > 0.5):
            warnings.append(f"prediction bias: predicted_classes={predicted_classes}")

    detected_macro = float("nan")
    fallback_macro = float("nan")
    if not fallback.empty and "group" in fallback.columns:
        det = fallback[fallback["group"].astype(str) == "detected"]
        fb = fallback[fallback["group"].astype(str) == "fallback"]
        if det.empty or fb.empty:
            failures.append("missing detected/fallback metric groups")
        if not det.empty:
            detected_macro = float(pd.to_numeric(det["macro_f1"], errors="coerce").iloc[-1])
        if not fb.empty:
            fallback_macro = float(pd.to_numeric(fb["macro_f1"], errors="coerce").iloc[-1])
            if math.isfinite(fallback_macro) and fallback_macro < 0.35:
                warnings.append("fallback still weak")

    if any("routed_path" in item for item in failures):
        decision = "D16_V4_RUN_FAIL_NO_ROUTED_PATH"
    elif any("NaN" in item or "non-finite" in item for item in failures):
        decision = "D16_V4_RUN_FAIL_NAN"
    elif any("collapse" in item for item in failures):
        decision = "D16_V4_RUN_FAIL_COLLAPSE"
    elif failures:
        decision = "D16_V4_RUN_FAIL_DATA"
    elif any("one routed path" in item for item in warnings):
        decision = "D16_V4_RUN_WARN_PATH_UNUSED"
    elif math.isfinite(fallback_macro) and fallback_macro < 0.35:
        decision = "D16_V4_RUN_WARN_FALLBACK_STILL_WEAK"
    else:
        decision = "D16_V4_RUN_PASS"

    summary = {
        "output_dir": str(output_dir),
        "decision": decision,
        "epoch_count": epoch_count,
        "best_val_macro_f1": float(pd.to_numeric(train.get("val_macro_f1", pd.Series(dtype=float)), errors="coerce").max()) if not train.empty else float("nan"),
        "test_macro_f1": float(pd.to_numeric(test.get("macro_f1", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not test.empty else float("nan"),
        "test_accuracy": float(pd.to_numeric(test.get("accuracy", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not test.empty else float("nan"),
        "detected_macro_f1": detected_macro,
        "fallback_macro_f1": fallback_macro,
        "predicted_classes": predicted_classes,
        "detected_path_count": detected_path_count,
        "fallback_path_count": fallback_path_count,
        "wrong_route_count": wrong_route_count,
        "fallback_token_count_mean": float(pd.to_numeric(train.get("fallback_token_count_mean", pd.Series(dtype=float)), errors="coerce").dropna().iloc[-1]) if "fallback_token_count_mean" in train and not pd.to_numeric(train["fallback_token_count_mean"], errors="coerce").dropna().empty else float("nan"),
        "failures": failures,
        "warnings": warnings,
    }
    (output_dir / "d16_v4_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = check_run(Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    if str(summary["decision"]).startswith("D16_V4_RUN_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
