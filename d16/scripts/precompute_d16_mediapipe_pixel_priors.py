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
    detected_priors,
    fallback_priors,
    pixels_to_image48,
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


def _build_multi_attempt_priors(detectors: List[MediaPipeFaceDetector], pixels: str, label: int, sample_index: int):
    image48 = pixels_to_image48(pixels)
    for detector in detectors:
        landmark_xy = detector.detect(image48)
        if landmark_xy is not None:
            return detected_priors(image48, landmark_xy=landmark_xy, label=label, sample_index=sample_index)
    return fallback_priors(image48, label=label, sample_index=sample_index)


def _make_detectors(args) -> List[MediaPipeFaceDetector]:
    if args.detection_strategy == "multi_attempt":
        return [
            MediaPipeFaceDetector(192, args.min_detection_confidence, preprocess_mode="raw", padding_pixels=0),
            MediaPipeFaceDetector(256, args.min_detection_confidence, preprocess_mode="histogram_equalize", padding_pixels=0),
            MediaPipeFaceDetector(320, args.min_detection_confidence, preprocess_mode="clahe", padding_pixels=0),
        ]
    return [
        MediaPipeFaceDetector(
            args.detection_size,
            args.min_detection_confidence,
            preprocess_mode=args.preprocess_mode,
            padding_pixels=args.padding_pixels,
        )
    ]


def _close_detectors(detectors: List[MediaPipeFaceDetector]) -> None:
    for detector in detectors:
        detector.close()


def process_split(
    detectors: List[MediaPipeFaceDetector],
    detector_factory,
    data_dir: Path,
    output_dir: Path,
    split: str,
    max_samples: int | None,
    log_every: int,
    detector_recreate_every: int,
) -> Dict:
    df = read_fer_split(data_dir, split, max_samples=max_samples)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    detected_examples = 0
    fallback_examples = 0
    samples_since_recreate = 0
    for count, row in enumerate(df.itertuples(index=False), start=1):
        if detector_recreate_every > 0 and samples_since_recreate >= detector_recreate_every:
            _close_detectors(detectors)
            detectors[:] = detector_factory()
            samples_since_recreate = 0
            print(f"[D16 precompute] recreated detectors split={split} processed={count}", flush=True)
        sample_index = int(row.sample_index)
        label = int(row.emotion)
        if len(detectors) == 1:
            result = build_priors_for_sample(detectors[0], row.pixels, label=label, sample_index=sample_index)
        else:
            result = _build_multi_attempt_priors(detectors, row.pixels, label=label, sample_index=sample_index)
        samples_since_recreate += 1
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
            print(f"[D16 precompute] split={split} processed={count} detected={sum(r['detected'] for r in rows)}", flush=True)
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
    parser.add_argument("--preprocess_mode", default="raw", choices=["raw", "histogram_equalize", "clahe", "contrast_stretch"])
    parser.add_argument("--detection_strategy", default="single", choices=["single", "multi_attempt"])
    parser.add_argument("--padding_pixels", type=int, default=0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--detector_recreate_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "part_names.json", PART_NAMES)
    _write_json(output_dir / "micro_anchor_names.json", MICRO_ANCHOR_NAMES)
    metadata = schema_metadata()
    metadata["detection_size"] = int(args.detection_size)
    metadata["preprocess_mode"] = str(args.preprocess_mode)
    metadata["detection_strategy"] = str(args.detection_strategy)
    metadata["padding_pixels"] = int(args.padding_pixels)
    metadata["min_detection_confidence"] = float(args.min_detection_confidence)
    metadata["detector_recreate_every"] = int(args.detector_recreate_every)
    metadata["fallback_types"] = FALLBACK_TYPES
    _write_json(output_dir / "prior_schema.json", metadata)

    summaries = []
    all_rows = []
    for split in args.splits:
        detectors = _make_detectors(args)
        try:
            summary = process_split(
                detectors,
                detector_factory=lambda: _make_detectors(args),
                data_dir=data_dir,
                output_dir=output_dir,
                split=split,
                max_samples=args.max_samples_per_split,
                log_every=args.log_every,
                detector_recreate_every=int(args.detector_recreate_every),
            )
            all_rows.extend(summary.pop("rows"))
            summaries.append(summary)
        finally:
            _close_detectors(detectors)

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
