"""Compare D13A 50-epoch runs against resumed 100-epoch extensions.

This is a read-only analysis utility. It does not train, does not modify
checkpoints, and makes no motif or semantic-region claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DEFAULT_EXTENDED_KEYS = {
    "extended_k256": "d13a_edgeaware_lite_localpool_k256_outputs",
    "extended_seed3": "d13a_edgeaware_lite_localpool_k144_seed3_outputs",
    "extended_no_aux": "d13a_edgeaware_lite_localpool_k144_no_aux_outputs",
    "extended_anneal_1to05": "d13a_edgeaware_lite_localpool_k144_anneal_1to05_outputs",
    "extended_compact_balance_x2": "d13a_edgeaware_lite_localpool_k144_compact_balance_x2_outputs",
}


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if not df.empty else None


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def last_value(df: Optional[pd.DataFrame], metric: str) -> Optional[float]:
    if df is None or metric not in df:
        return None
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def best_metric(df: Optional[pd.DataFrame], metric: str) -> tuple[Optional[int], Optional[float]]:
    if df is None or metric not in df:
        return None, None
    vals = pd.to_numeric(df[metric], errors="coerce")
    if vals.dropna().empty:
        return None, None
    idx = vals.idxmax()
    epoch = as_int(df.loc[idx].get("epoch")) if "epoch" in df else None
    return epoch, as_float(vals.loc[idx])


def slope_last(df: Optional[pd.DataFrame], metric: str, n: int = 10) -> Optional[float]:
    if df is None or metric not in df or "epoch" not in df:
        return None
    sub = df[["epoch", metric]].dropna().tail(n)
    if len(sub) < 2:
        return None
    x = pd.to_numeric(sub["epoch"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def pred_entropy(counts: List[int]) -> Optional[float]:
    total = float(sum(counts))
    if total <= 0:
        return None
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    if len(p) == 0:
        return None
    return float(-(p * np.log(p)).sum() / math.log(len(EMOTION_NAMES)))


def load_best_checkpoint_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "checkpoints" / "best.pt"
    if not path.exists():
        return {}
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    metrics = ckpt.get("metrics", {}) or {}
    return {
        "best_checkpoint_epoch": as_int(ckpt.get("epoch")),
        "best_checkpoint_val_macro_f1": as_float(metrics.get("val_macro_f1", metrics.get("macro_f1"))),
        "best_checkpoint_val_accuracy": as_float(metrics.get("val_accuracy", metrics.get("accuracy"))),
    }


def infer_extended_pairs(original_root: Path, extended_root: Path) -> List[tuple[str, Path, Path]]:
    pairs = []
    for key, source_name in DEFAULT_EXTENDED_KEYS.items():
        old_dir = original_root / source_name
        new_dir = extended_root / f"{source_name}_ep100"
        if old_dir.exists() or new_dir.exists():
            pairs.append((key, old_dir, new_dir))
    if pairs:
        return pairs
    for new_dir in sorted(extended_root.glob("*_ep100")):
        source_name = new_dir.name[: -len("_ep100")]
        pairs.append((source_name, original_root / source_name, new_dir))
    return pairs


def summarize_pool(run_dir: Path, suffix: str) -> Dict[str, Any]:
    pool = read_csv(run_dir / "pooling_stats.csv")
    row: Dict[str, Any] = {}
    if pool is None:
        return row
    test = pool[pool["split"].astype(str).str.lower() == "test"] if "split" in pool else pd.DataFrame()
    use = test if not test.empty else pool.tail(1)
    for metric in ["effective_regions", "empty_region_ratio", "assignment_entropy", "assignment_temperature"]:
        if metric in use:
            row[f"{metric}_{suffix}"] = as_float(pd.to_numeric(use[metric], errors="coerce").dropna().iloc[-1]) if not pd.to_numeric(use[metric], errors="coerce").dropna().empty else None
    return row


def summarize_pred(run_dir: Path, suffix: str) -> Dict[str, Any]:
    pred = read_csv(run_dir / "pred_count.csv")
    row: Dict[str, Any] = {}
    if pred is None:
        return row
    test = pred[pred["split"].astype(str).str.lower() == "test"] if "split" in pred else pd.DataFrame()
    use = test.iloc[-1] if not test.empty else pred.iloc[-1]
    counts = []
    for idx, name in enumerate(EMOTION_NAMES):
        counts.append(int(use.get(f"pred_count_{idx}_{name}", 0)))
    total = max(sum(counts), 1)
    max_idx = int(np.argmax(counts)) if counts else 0
    row[f"max_pred_class_{suffix}"] = EMOTION_NAMES[max_idx]
    row[f"max_pred_ratio_{suffix}"] = float(counts[max_idx] / total)
    row[f"classes_predicted_count_{suffix}"] = int(sum(1 for c in counts if c > 0))
    row[f"pred_entropy_{suffix}"] = pred_entropy(counts)
    for idx, name in enumerate(EMOTION_NAMES):
        row[f"pred_count_{name}_{suffix}"] = counts[idx]
    return row


def summarize_metrics(run_dir: Path, suffix: str) -> Dict[str, Any]:
    val = read_csv(run_dir / "val_metrics.csv")
    test = read_csv(run_dir / "test_metrics.csv")
    train = read_csv(run_dir / "train_log.csv")
    best_epoch, best_val = best_metric(val, "val_macro_f1")
    best_val_acc_epoch, best_val_acc = best_metric(val, "val_accuracy")
    best_ckpt = load_best_checkpoint_metrics(run_dir)
    if best_ckpt.get("best_checkpoint_val_macro_f1") is not None:
        best_val = best_ckpt["best_checkpoint_val_macro_f1"]
        best_epoch = best_ckpt["best_checkpoint_epoch"]
    if best_ckpt.get("best_checkpoint_val_accuracy") is not None:
        best_val_acc = best_ckpt["best_checkpoint_val_accuracy"]
    max_epoch = as_int(train["epoch"].max()) if train is not None and "epoch" in train else as_int(val["epoch"].max()) if val is not None and "epoch" in val else None
    test_row = test.iloc[-1] if test is not None else {}
    row: Dict[str, Any] = {
        f"best_val_macro_f1_{suffix}": best_val,
        f"best_val_accuracy_{suffix}": best_val_acc,
        f"best_epoch_{suffix}": best_epoch,
        f"max_logged_epoch_{suffix}": max_epoch,
        f"test_macro_f1_{suffix}": as_float(test_row.get("test_macro_f1")),
        f"test_accuracy_{suffix}": as_float(test_row.get("test_accuracy")),
        f"test_weighted_f1_{suffix}": as_float(test_row.get("test_weighted_f1")),
        f"val_macro_slope_last10_{suffix}": slope_last(val, "val_macro_f1", 10),
        f"train_loss_slope_last10_{suffix}": slope_last(train, "train_loss", 10),
        f"checkpoint_best_exists_{suffix}": (run_dir / "checkpoints" / "best.pt").exists(),
        f"checkpoint_last_exists_{suffix}": (run_dir / "checkpoints" / "last.pt").exists(),
        f"report_exists_{suffix}": (run_dir / "d13a_report.md").exists(),
    }
    for idx, name in enumerate(EMOTION_NAMES):
        row[f"f1_{name}_{suffix}"] = as_float(test_row.get(f"test_f1_{idx}_{name}"))
    return row


def compare_pair(run_key: str, old_dir: Path, new_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_key": run_key,
        "run_name_50": old_dir.name,
        "run_name_100": new_dir.name,
        "old_dir_exists": old_dir.exists(),
        "extended_dir_exists": new_dir.exists(),
        "resume_metadata_exists": (new_dir / "resume_metadata.json").exists(),
    }
    resume_meta = read_json(new_dir / "resume_metadata.json")
    row.update(
        {
            "resumed_from_checkpoint": resume_meta.get("resumed_from_checkpoint"),
            "resume_epoch": resume_meta.get("resume_epoch"),
            "target_max_epoch": resume_meta.get("target_max_epoch"),
        }
    )
    row.update(summarize_metrics(old_dir, "50") if old_dir.exists() else {})
    row.update(summarize_metrics(new_dir, "100") if new_dir.exists() else {})
    row.update(summarize_pool(old_dir, "50") if old_dir.exists() else {})
    row.update(summarize_pool(new_dir, "100") if new_dir.exists() else {})
    row.update(summarize_pred(old_dir, "50") if old_dir.exists() else {})
    row.update(summarize_pred(new_dir, "100") if new_dir.exists() else {})

    row["delta_val_macro"] = None
    if row.get("best_val_macro_f1_50") is not None and row.get("best_val_macro_f1_100") is not None:
        row["delta_val_macro"] = float(row["best_val_macro_f1_100"] - row["best_val_macro_f1_50"])
    row["delta_test_macro"] = None
    if row.get("test_macro_f1_50") is not None and row.get("test_macro_f1_100") is not None:
        row["delta_test_macro"] = float(row["test_macro_f1_100"] - row["test_macro_f1_50"])
    row["delta_effective_regions"] = None
    if row.get("effective_regions_50") is not None and row.get("effective_regions_100") is not None:
        row["delta_effective_regions"] = float(row["effective_regions_100"] - row["effective_regions_50"])
    row["delta_assignment_entropy"] = None
    if row.get("assignment_entropy_50") is not None and row.get("assignment_entropy_100") is not None:
        row["delta_assignment_entropy"] = float(row["assignment_entropy_100"] - row["assignment_entropy_50"])
    row["still_learning_100"] = bool((row.get("val_macro_slope_last10_100") or 0.0) > 0.0)
    row["pooling_healthy_100"] = bool(
        row.get("effective_regions_100") is not None
        and row.get("empty_region_ratio_100") is not None
        and row["empty_region_ratio_100"] <= 0.4
    )
    row["pred_collapse_100"] = bool((row.get("max_pred_ratio_100") or 0.0) >= 0.75)
    for name in EMOTION_NAMES:
        old = row.get(f"f1_{name}_50")
        new = row.get(f"f1_{name}_100")
        row[f"delta_f1_{name}"] = float(new - old) if old is not None and new is not None else None
    return row


def md_table(df: pd.DataFrame, cols: List[str], n: Optional[int] = None) -> str:
    if df.empty:
        return "No data."
    cols = [c for c in cols if c in df.columns]
    use = df[cols].copy()
    if n is not None:
        use = use.head(n)
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(output_dir: Path, comp: pd.DataFrame, per_class: pd.DataFrame) -> None:
    report = output_dir / "d13a_extended_analysis_report.md"
    if comp.empty:
        report.write_text("# D13A Extended Analysis Report\n\nNo extended runs found.\n", encoding="utf-8")
        return
    ranked = comp.sort_values("test_macro_f1_100", ascending=False, na_position="last")
    improved = comp[comp["delta_test_macro"].fillna(-999) > 0.0]
    still = comp[comp["still_learning_100"].fillna(False)]
    k256 = comp[comp["run_key"] == "extended_k256"]
    entropy_note = ""
    if not k256.empty:
        r = k256.iloc[0]
        entropy_note = (
            f"K256 assignment entropy 50->{r.get('assignment_entropy_50')} "
            f"and 100->{r.get('assignment_entropy_100')}; inspect whether the larger bottleneck remains too soft."
        )
    final_decision = "D13A_EXTENDED_NEEDS_REVIEW"
    healthy = ranked[(ranked["pooling_healthy_100"] == True) & (ranked["pred_collapse_100"] == False)]
    if not healthy.empty and not still.empty:
        final_decision = "D13A_EXTENDED_READY_FOR_FINAL_REVIEW"

    lines = [
        "# D13A Extended Analysis Report",
        "",
        "## Context",
        "This report compares selected D13A 50-epoch runs against resumed 100-epoch extensions.",
        "D13A remains a pure GNN hierarchical reduction baseline. Region nodes are a soft learnable bottleneck, not semantic regions or motif slots.",
        "",
        "## Ranking at 100 Epoch",
        md_table(ranked, ["run_key", "run_name_100", "best_epoch_100", "best_val_macro_f1_100", "test_macro_f1_100", "test_accuracy_100", "delta_test_macro", "still_learning_100"], n=20),
        "",
        "## 50 vs 100 Delta",
        md_table(comp.sort_values("delta_test_macro", ascending=False, na_position="last"), ["run_key", "test_macro_f1_50", "test_macro_f1_100", "delta_test_macro", "best_val_macro_f1_50", "best_val_macro_f1_100", "delta_val_macro"], n=20),
        "",
        "## Pooling Changes",
        md_table(comp, ["run_key", "effective_regions_50", "effective_regions_100", "delta_effective_regions", "empty_region_ratio_50", "empty_region_ratio_100", "assignment_entropy_50", "assignment_entropy_100", "delta_assignment_entropy"], n=20),
        "",
        entropy_note,
        "",
        "## Prediction Distribution",
        md_table(comp, ["run_key", "max_pred_class_100", "max_pred_ratio_100", "classes_predicted_count_100", "pred_entropy_100", "pred_collapse_100"], n=20),
        "",
        "## Per-Class Delta",
        md_table(per_class, ["run_key", "delta_f1_Angry", "delta_f1_Disgust", "delta_f1_Fear", "delta_f1_Happy", "delta_f1_Sad", "delta_f1_Surprise", "delta_f1_Neutral"], n=20),
        "",
        "## Interpretation",
        f"- Improved test macro-F1 runs: {', '.join(improved['run_key'].astype(str).tolist()) if not improved.empty else 'none yet'}.",
        f"- Still-learning at epoch 100: {', '.join(still['run_key'].astype(str).tolist()) if not still.empty else 'none detected by late-slope rule'}.",
        "- Do not open D13B from this report alone if the selected run still shows late learning, pooling softness, or prediction bias.",
        "",
        "## Final Decision",
        final_decision,
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_root", default="outputs/d13_hierarchical_reduction")
    parser.add_argument("--extended_root", default="outputs/d13_hierarchical_reduction/extended")
    parser.add_argument("--output_dir", default="outputs/d13_hierarchical_reduction/extended")
    args = parser.parse_args()

    original_root = Path(args.original_root)
    extended_root = Path(args.extended_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [compare_pair(key, old_dir, new_dir) for key, old_dir, new_dir in infer_extended_pairs(original_root, extended_root)]
    comp = pd.DataFrame(rows)
    per_class_cols = ["run_key"] + [f"delta_f1_{name}" for name in EMOTION_NAMES]
    per_class = comp[per_class_cols].copy() if not comp.empty else pd.DataFrame(columns=per_class_cols)
    comp.to_csv(output_dir / "d13a_extended_comparison.csv", index=False)
    per_class.to_csv(output_dir / "d13a_extended_per_class_delta.csv", index=False)
    write_report(output_dir, comp, per_class)
    print(
        json.dumps(
            {
                "num_pairs": int(len(comp)),
                "output_dir": str(output_dir),
                "report": str(output_dir / "d13a_extended_analysis_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

