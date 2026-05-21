"""Compare D14 checkpoint-based references against D15 from-scratch runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


D13C_BASELINES = [
    {
        "track": "D13C_checkpoint_baseline",
        "run_name": "d13c_m8_supcon_l002_control",
        "result_type": "checkpoint_baseline",
        "test_acc": 0.6481,
        "test_macro_f1": 0.6364,
    },
    {
        "track": "D13C_checkpoint_baseline",
        "run_name": "d13c_m16_supcon_l005",
        "result_type": "checkpoint_baseline",
        "test_acc": 0.6420,
        "test_macro_f1": 0.6277,
    },
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _d14_summary(root: Path) -> pd.DataFrame:
    direct = root / "summary" / "d14_performance_summary.csv"
    df = _read_csv(direct)
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "track": "D14_checkpoint_reference",
            "run_name": df.get("run_name"),
            "result_type": "checkpoint_finetune_or_ensemble",
            "test_acc": df.get("test_acc"),
            "test_macro_f1": df.get("test_macro_f1"),
            "checker_decision": df.get("checker_decision"),
        }
    )
    return out


def _d15_summary(root: Path) -> pd.DataFrame:
    direct = root / "summary" / "d15_from_scratch_summary.csv"
    df = _read_csv(direct)
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "track": "D15_from_scratch",
            "run_name": df.get("run_name"),
            "result_type": "from_scratch",
            "test_acc": df.get("test_acc"),
            "test_macro_f1": df.get("test_macro_f1"),
            "checker_decision": df.get("checker_decision"),
        }
    )
    return out


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def compare(d14_root: Path, d15_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    d13 = pd.DataFrame(D13C_BASELINES)
    d14 = _d14_summary(d14_root)
    d15 = _d15_summary(d15_root)
    all_rows = pd.concat([d13, d14, d15], ignore_index=True)
    if not all_rows.empty:
        all_rows["test_acc"] = pd.to_numeric(all_rows["test_acc"], errors="coerce")
        all_rows["test_macro_f1"] = pd.to_numeric(all_rows["test_macro_f1"], errors="coerce")
        all_rows = all_rows.sort_values(["test_acc", "test_macro_f1"], ascending=False)
    all_rows.to_csv(output_dir / "d14_d15_comparison_summary.csv", index=False)

    d15_best = d15.sort_values(["test_acc", "test_macro_f1"], ascending=False).iloc[0].to_dict() if not d15.empty else {}
    d14_best = d14.sort_values(["test_acc", "test_macro_f1"], ascending=False).iloc[0].to_dict() if not d14.empty else {}
    d13_best_acc = max(item["test_acc"] for item in D13C_BASELINES)
    d15_acc = float(d15_best.get("test_acc", np.nan)) if d15_best else np.nan
    d14_acc = float(d14_best.get("test_acc", np.nan)) if d14_best else np.nan
    if np.isfinite(d15_acc) and d15_acc >= 0.70:
        decision = "D15_FROM_SCRATCH_REACHED_0P70"
    elif np.isfinite(d15_acc) and d15_acc >= d13_best_acc:
        decision = "D15_FROM_SCRATCH_MATCHES_CHECKPOINT_BASELINE"
    elif np.isfinite(d14_acc) and np.isfinite(d15_acc) and d14_acc - d15_acc >= 0.03:
        decision = "D14_GT_D15_FROM_SCRATCH_OPTIMIZATION_GAP"
    elif np.isfinite(d15_acc) and d15_acc >= 0.65:
        decision = "D15_FROM_SCRATCH_BASELINE_PASS_BELOW_TARGET"
    else:
        decision = "NEED_ARCHITECTURE_DATA_TRAINING_SHIFT"

    lines = [
        "# D14/D15 Performance Track Comparison",
        "",
        "D14 is checkpoint-based side reference. D15 is the main from-scratch track. These tracks must not be merged as the same evidence type.",
        "",
        "## Ranking",
        _md_table(all_rows),
        "",
        "## Questions",
        f"1. D15 reaches/exceeds D13C checkpoint best: {'yes' if np.isfinite(d15_acc) and d15_acc >= d13_best_acc else 'no or unavailable'}.",
        f"2. D15 reaches 0.65 / 0.68 / 0.70: {d15_acc >= 0.65 if np.isfinite(d15_acc) else 'unavailable'} / {d15_acc >= 0.68 if np.isfinite(d15_acc) else 'unavailable'} / {d15_acc >= 0.70 if np.isfinite(d15_acc) else 'unavailable'}.",
        f"3. D14 upper-bound minus D15 best acc: {d14_acc - d15_acc:.4f}" if np.isfinite(d14_acc) and np.isfinite(d15_acc) else "3. D14 upper-bound minus D15 best acc: unavailable.",
        "4. If D14 is much higher than D15, from-scratch optimization is not enough yet.",
        "5. If D15 is close to D14, the end-to-end recipe is viable.",
        "6. If both are low, shift architecture/data/training.",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "No motif, semantic-region, causal-evidence, or full interpretability claim is made.",
    ]
    (output_dir / "d14_d15_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d14_root", default="outputs/d14_performance")
    parser.add_argument("--d15_root", default="outputs/d15_from_scratch")
    parser.add_argument("--output_dir", default="outputs/performance_track_comparison")
    args = parser.parse_args()
    compare(Path(args.d14_root), Path(args.d15_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
