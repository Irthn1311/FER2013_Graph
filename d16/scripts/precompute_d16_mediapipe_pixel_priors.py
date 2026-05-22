"""Precompute D16 MediaPipe pixel priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.fer_csv import read_fer_split
from d16.data.mediapipe_priors import (
    FALLBACK_TYPES,
    MICRO_ANCHOR_NAMES,
    PART_NAMES,
    MediaPipeFaceDetector,
    build_priors_for_sample,
    save_npz,
    schema_metadata,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _try_save_example(path: Path, arrays: Dict, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
        axes[0].imshow(arrays["image_48"], cmap="gray")
        axes[0].set_title("image")
        axes[1].imshow(arrays["face_mask"], cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("face")
        axes[2].imshow(arrays["part_soft_masks"].max(axis=0), cmap="magma", vmin=0, vmax=1)
        axes[2].set_title(title)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception:
        return


def process_split(detector: MediaPipeFaceDetector, data_dir: Path, output_dir: Path, split: str, max_samples: int | None, log_every: int) -> Dict:
    df = read_fer_split(data_dir, split, max_samples=max_samples)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    detected_examples = 0
    fallback_examples = 0
    for count, row in enumerate(df.itertuples(index=False), start=1):
        sample_index = int(row.sample_index)
        label = int(row.emotion)
        result = build_priors_for_sample(detector, row.pixels, label=label, sample_index=sample_index)
        save_npz(split_dir / f"{sample_index:06d}.npz", result.arrays)
        rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "label": label,
                "detected": bool(result.detected),
                "fallback_type": result.fallback_type,
                "quality_score": float(result.quality_score),
            }
        )
        if result.detected and detected_examples < 20:
            _try_save_example(output_dir / "figures" / "examples" / f"{split}_detected_{detected_examples:02d}.png", result.arrays, "detected")
            detected_examples += 1
        if not result.detected and fallback_examples < 20:
            _try_save_example(output_dir / "figures" / "examples" / f"{split}_fallback_{fallback_examples:02d}.png", result.arrays, "fallback")
            fallback_examples += 1
        if count % int(log_every) == 0:
            print(f"[D16 precompute] split={split} processed={count} detected={sum(r['detected'] for r in rows)}")
    total = len(rows)
    detected = sum(1 for item in rows if item["detected"])
    fallback = total - detected
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_coverage_rows.csv", index=False)
    return {
        "split": split,
        "total": int(total),
        "detected": int(detected),
        "fallback": int(fallback),
        "fallback_rate": float(fallback / max(total, 1)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/fer13-split")
    parser.add_argument("--output_dir", default="outputs/d16_mediapipe_pixel_priors")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--detection_size", type=int, default=192)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "part_names.json", PART_NAMES)
    _write_json(output_dir / "micro_anchor_names.json", MICRO_ANCHOR_NAMES)
    metadata = schema_metadata()
    metadata["detection_size"] = int(args.detection_size)
    metadata["min_detection_confidence"] = float(args.min_detection_confidence)
    metadata["fallback_types"] = FALLBACK_TYPES
    _write_json(output_dir / "prior_schema.json", metadata)

    detector = MediaPipeFaceDetector(args.detection_size, args.min_detection_confidence)
    summaries = []
    all_rows = []
    try:
        for split in args.splits:
            summary = process_split(
                detector,
                data_dir=data_dir,
                output_dir=output_dir,
                split=split,
                max_samples=args.max_samples_per_split,
                log_every=args.log_every,
            )
            all_rows.extend(summary.pop("rows"))
            summaries.append(summary)
    finally:
        detector.close()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        by_class = rows_df.groupby(["split", "label"]).agg(total=("sample_index", "count"), detected=("detected", "sum")).reset_index()
        by_class["fallback"] = by_class["total"] - by_class["detected"]
        by_class["fallback_rate"] = by_class["fallback"] / by_class["total"].clip(lower=1)
        by_class.to_csv(output_dir / "coverage_by_class.csv", index=False)
    else:
        pd.DataFrame(columns=["split", "label", "total", "detected", "fallback", "fallback_rate"]).to_csv(output_dir / "coverage_by_class.csv", index=False)

    report = [
        "# D16 MediaPipe Pixel Prior Report",
        "",
        "D16 priors are numeric per-sample arrays. Part and anchor names are stored in global JSON metadata.",
        "",
        "| split | total | detected | fallback | fallback_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        report.append(f"| {item['split']} | {item['total']} | {item['detected']} | {item['fallback']} | {item['fallback_rate']:.4f} |")
    report.extend(["", "No failed sample is dropped. Fallback samples carry `landmark_missing_flag=1` and invalid anchor masks."])
    (output_dir / "prior_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()
