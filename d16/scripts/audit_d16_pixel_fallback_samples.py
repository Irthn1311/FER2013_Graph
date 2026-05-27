"""Audit fallback samples in a D16 MediaPipe pixel-prior directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.mediapipe_pixel_rescue_utils import iter_prior_records, records_to_frame, split_class_summary, split_summary


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---" for _ in columns]) + "|"]
    for row in df[columns].itertuples(index=False):
        vals = []
        for value in row:
            vals.append(f"{float(value):.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best")
    parser.add_argument("--output_dir", default="outputs/d16_analysis/pixel_prior_rescue_audit")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = iter_prior_records(prior_dir, args.splits)
    df = records_to_frame(records)
    fallback = df[df["is_fallback"]].copy() if not df.empty else pd.DataFrame(columns=df.columns)
    by_split = split_summary(df)
    by_class = split_class_summary(df)
    inconsistent = df[
        ((df["detected"].astype(bool)) & (df["landmark_missing_flag"].astype(int) != 0))
        | ((~df["detected"].astype(bool)) & (df["landmark_missing_flag"].astype(int) != 1))
    ].copy() if not df.empty else pd.DataFrame(columns=df.columns)

    fallback[["split", "file_name", "sample_index", "label", "detected", "landmark_missing_flag", "fallback_type_id", "quality_score"]].to_csv(
        output_dir / "fallback_samples.csv", index=False
    )
    by_split.to_csv(output_dir / "fallback_by_split.csv", index=False)
    by_class.to_csv(output_dir / "fallback_by_split_class.csv", index=False)
    inconsistent.to_csv(output_dir / "fallback_consistency_warnings.csv", index=False)

    worst_class = by_class.sort_values(["fallback_rate", "fallback"], ascending=[False, False]).head(20) if not by_class.empty else by_class
    lines = [
        "# D16 Pixel Prior Fallback Audit",
        "",
        f"- prior_dir: `{prior_dir}`",
        "- scope: MediaPipe pixel priors only",
        "- action: audit only; no training, no model edits, no prior overwrite",
        "",
        "## Fallback By Split",
        *_markdown_table(by_split, ["split", "total", "detected", "fallback", "fallback_rate"]),
        "",
        "## Worst Fallback By Split/Class",
        *_markdown_table(worst_class, ["split", "label", "total", "detected", "fallback", "fallback_rate"]),
        "",
        "## Consistency Warnings",
        f"- inconsistent detected vs landmark_missing_flag rows: {len(inconsistent)}",
    ]
    if len(inconsistent) > 0:
        lines.append("- see `fallback_consistency_warnings.csv`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "This report only audits existing D16 MediaPipe pixel priors. It does not train, does not change model architecture, and does not modify the source prior directory.",
        ]
    )
    (output_dir / "D16_PIXEL_FALLBACK_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    print({"prior_dir": str(prior_dir), "output_dir": str(output_dir), "total": int(len(df)), "fallback": int(len(fallback))})


if __name__ == "__main__":
    main()
