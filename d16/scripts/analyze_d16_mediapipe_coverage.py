"""Analyze full D16 MediaPipe prior coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_rows(prior_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_dir = prior_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                rows.append(
                    {
                        "split": split,
                        "sample_index": int(data["sample_index"].item()),
                        "label": int(data["label"].item()),
                        "detected": bool(data["detected"].item()),
                        "fallback_type_id": int(data["fallback_type_id"].item()),
                        "landmark_missing_flag": int(data["landmark_missing_flag"].item()),
                        "quality_score": float(data["quality_score"].item()),
                        "path": str(path),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No D16 prior npz files found under {prior_dir}")
    return pd.DataFrame(rows)


def _decision(max_fallback_rate: float, blocked: bool = False) -> str:
    if blocked:
        return "D16_PRIORS_FULL_COVERAGE_BLOCKED"
    if max_fallback_rate <= 0.05:
        return "D16_PRIORS_FULL_COVERAGE_GOOD"
    if max_fallback_rate <= 0.20:
        return "D16_PRIORS_FULL_COVERAGE_ACCEPTABLE_WITH_FALLBACK"
    return "D16_PRIORS_FULL_COVERAGE_RISKY"


def _quality_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("split")["quality_score"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .fillna({"std": 0.0})
    )


def _write_example_figures(df: pd.DataFrame, output_dir: Path, tag: str, rows: pd.DataFrame, max_examples: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = output_dir / "figures" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows.head(max_examples).itertuples(index=False)):
        with np.load(row.path, allow_pickle=False) as data:
            image = data["image_48"]
            face = data["face_mask"]
            parts = data["part_soft_masks"].max(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("image")
        axes[1].imshow(face, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("face")
        axes[2].imshow(parts, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title(tag)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{tag}_{idx:02d}_{row.split}_{int(row.sample_index):06d}.png", dpi=120)
        plt.close(fig)


def analyze(prior_dir: Path, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(prior_dir)
    summary = (
        df.groupby("split")
        .agg(total=("sample_index", "count"), detected=("detected", "sum"), quality_mean=("quality_score", "mean"))
        .reset_index()
    )
    summary["fallback"] = summary["total"] - summary["detected"]
    summary["fallback_rate"] = summary["fallback"] / summary["total"].clip(lower=1)
    summary.to_csv(output_dir / "d16_full_coverage_summary.csv", index=False)

    by_class = (
        df.groupby(["split", "label"])
        .agg(total=("sample_index", "count"), detected=("detected", "sum"), quality_mean=("quality_score", "mean"))
        .reset_index()
    )
    by_class["fallback"] = by_class["total"] - by_class["detected"]
    by_class["fallback_rate"] = by_class["fallback"] / by_class["total"].clip(lower=1)
    by_class.to_csv(output_dir / "d16_full_coverage_by_class.csv", index=False)

    fallback_cases = df[~df["detected"]].sort_values(["split", "label", "sample_index"])
    fallback_cases.to_csv(output_dir / "d16_fallback_cases.csv", index=False)

    quality = _quality_distribution(df)
    quality.to_csv(output_dir / "d16_quality_distribution.csv", index=False)

    detected = df[df["detected"]].copy()
    _write_example_figures(df, output_dir, "fallback", fallback_cases, max_examples=20)
    _write_example_figures(df, output_dir, "low_quality_detected", detected.sort_values("quality_score", ascending=True), max_examples=20)
    _write_example_figures(df, output_dir, "high_quality_detected", detected.sort_values("quality_score", ascending=False), max_examples=20)

    max_rate = float(summary["fallback_rate"].max()) if not summary.empty else 1.0
    decision = _decision(max_rate)
    worst_classes = by_class.sort_values("fallback_rate", ascending=False).head(10)
    fallback_ids = fallback_cases[["split", "sample_index", "label"]].head(200)

    report = [
        "# D16 Full MediaPipe Coverage Analysis",
        "",
        f"- decision: `{decision}`",
        f"- prior_dir: `{prior_dir}`",
        f"- total_samples: {int(summary['total'].sum())}",
        f"- max_split_fallback_rate: {max_rate:.4f}",
        "",
        "## Split Coverage",
        "| split | total | detected | fallback | fallback_rate | quality_mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.split} | {int(row.total)} | {int(row.detected)} | {int(row.fallback)} | {float(row.fallback_rate):.4f} | {float(row.quality_mean):.4f} |"
        )
    report.extend(
        [
            "",
            "## Worst Fallback By Class",
            "| split | label | total | detected | fallback | fallback_rate | quality_mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_classes.itertuples(index=False):
        report.append(
            f"| {row.split} | {int(row.label)} | {int(row.total)} | {int(row.detected)} | {int(row.fallback)} | {float(row.fallback_rate):.4f} | {float(row.quality_mean):.4f} |"
        )
    report.extend(
        [
            "",
            "## Quality Distribution",
            "| split | count | mean | std | min | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality.itertuples(index=False):
        report.append(
            f"| {row.split} | {int(row.count)} | {float(row.mean):.4f} | {float(row.std):.4f} | {float(row.min):.4f} | {float(row.max):.4f} |"
        )
    report.extend(["", "## Fallback Sample IDs"])
    if fallback_ids.empty:
        report.append("- none")
    else:
        for row in fallback_ids.itertuples(index=False):
            report.append(f"- {row.split}/{int(row.sample_index):06d} label={int(row.label)}")
    report.extend(
        [
            "",
            "## Figures",
            "- `figures/fallback/`: up to 20 fallback examples",
            "- `figures/low_quality_detected/`: up to 20 low-quality detected examples",
            "- `figures/high_quality_detected/`: up to 20 high-quality detected examples",
            "",
            "No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        ]
    )
    (output_dir / "d16_coverage_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    summary_payload = {
        "decision": decision,
        "total_samples": int(summary["total"].sum()),
        "max_split_fallback_rate": max_rate,
        "summary": summary.to_dict(orient="records"),
        "worst_classes": worst_classes.to_dict(orient="records"),
    }
    (output_dir / "d16_coverage_analysis_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir) if args.output_dir else prior_dir / "coverage_analysis"
    payload = analyze(prior_dir, output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
