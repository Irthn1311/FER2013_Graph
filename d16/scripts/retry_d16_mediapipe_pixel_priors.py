"""Retry MediaPipe detection for fallback D16 pixel priors and merge rescues."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.fer_csv import read_fer_split
from d16.data.mediapipe_pixel_rescue_utils import (
    copy_prior_tree,
    iter_prior_records,
    load_npz,
    merge_extra_keys,
    records_to_frame,
    split_class_summary,
    split_summary,
    validate_prior_arrays,
    write_json,
)
from d16.data.mediapipe_priors import (
    FALLBACK_TYPES,
    MICRO_ANCHOR_NAMES,
    PART_NAMES,
    MediaPipeFaceDetector,
    detected_priors,
    pixels_to_image48,
    save_npz,
    schema_metadata,
)
from d16.data.mediapipe_retry_transforms import build_retry_transform


@dataclass(frozen=True)
class RetryStrategy:
    rank: int
    preprocess_mode: str
    min_confidence: float
    detection_size: int

    @property
    def name(self) -> str:
        return f"{self.rank:04d}_{self.preprocess_mode}_conf{self.min_confidence:g}_size{self.detection_size}"


class RetryFaceDetector:
    def __init__(self, strategy: RetryStrategy, model_path: str | Path | None = None) -> None:
        self.strategy = strategy
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        self.detector = MediaPipeFaceDetector(
            detection_size=int(strategy.detection_size),
            min_detection_confidence=float(strategy.min_confidence),
            preprocess_mode="raw",
            padding_pixels=0,
            **kwargs,
        )

    def detect(self, image48: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        transform = build_retry_transform(image48, self.strategy.preprocess_mode)
        if transform.skipped:
            return None, transform.skip_reason
        if not self.detector.available:
            return None, "mediapipe_unavailable"
        xy_transformed = self.detector.detect(np.asarray(transform.image, dtype=np.float32))
        if xy_transformed is None:
            return None, "no_face_landmarks"
        xy = transform.inverse_xy(xy_transformed)
        xy = np.clip(xy, 0.0, 47.0).astype(np.float32)
        if xy.shape[0] < 100:
            return None, f"too_few_landmarks:{xy.shape[0]}"
        return xy, "detected"

    def close(self) -> None:
        self.detector.close()


def _strategy_grid(args) -> list[RetryStrategy]:
    modes = [m for m in args.preprocess_modes if not (args.disable_rotations and str(m).startswith("rot_"))]
    out: list[RetryStrategy] = []
    rank = 0
    for conf in args.min_confidences:
        for size in args.detection_sizes:
            for mode in modes:
                out.append(RetryStrategy(rank=rank, preprocess_mode=str(mode), min_confidence=float(conf), detection_size=int(size)))
                rank += 1
    return out


def _resolve_data_dir(data_dir: Path) -> Path:
    if (data_dir / "train.csv").exists():
        return data_dir
    candidates = [Path("data"), Path("dataset/fer13-split"), Path("data/dataset/fer13-split")]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            print(f"[D16 pixel rescue] data_dir {data_dir} not found; using {candidate}", flush=True)
            return candidate
    return data_dir


def _read_split_lookup(data_dir: Path, split: str) -> Dict[int, tuple[str, int]]:
    df = read_fer_split(data_dir, split, max_samples=None)
    return {int(row.sample_index): (str(row.pixels), int(row.emotion)) for row in df.itertuples(index=False)}


def _metadata_arrays(success: bool, strategy: Optional[RetryStrategy], mode_id: int, before: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {
        "rescue_success": np.asarray(bool(success), dtype=np.bool_),
        "rescue_strategy_rank": np.asarray(-1 if strategy is None else int(strategy.rank), dtype=np.int64),
        "rescue_confidence": np.asarray(np.nan if strategy is None else float(strategy.min_confidence), dtype=np.float32),
        "rescue_detection_size": np.asarray(-1 if strategy is None else int(strategy.detection_size), dtype=np.int64),
        "rescue_preprocess_mode_id": np.asarray(int(mode_id), dtype=np.int64),
    }
    if strategy is not None:
        out["rescue_strategy_name_present_in_csv"] = np.asarray(True, dtype=np.bool_)
    else:
        out["rescue_strategy_name_present_in_csv"] = np.asarray(False, dtype=np.bool_)
    out["fallback_type_id_before"] = np.asarray(int(np.asarray(before.get("fallback_type_id", np.asarray(-1))).item()), dtype=np.int64)
    out["quality_score_before"] = np.asarray(float(np.asarray(before.get("quality_score", np.asarray(np.nan))).item()), dtype=np.float32)
    return out


def _attempt_rescue(
    image48: np.ndarray,
    label: int,
    sample_index: int,
    strategies: list[RetryStrategy],
    detectors: Dict[int, RetryFaceDetector],
) -> tuple[Optional[Dict[str, np.ndarray]], Optional[RetryStrategy], str]:
    for strategy in strategies:
        detector = detectors[strategy.rank]
        xy, reason = detector.detect(image48)
        if xy is None:
            continue
        result = detected_priors(image48, landmark_xy=xy, label=label, sample_index=sample_index)
        return result.arrays, strategy, reason
    return None, None, "all_strategies_failed"


def _write_retry_report(output_prior_dir: Path, report_path: Path, attempts: pd.DataFrame, original_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    original_split = split_summary(original_df)
    final_split = split_summary(final_df)
    rescued = attempts[attempts["rescue_success"].astype(bool)] if not attempts.empty else attempts
    rescued_by_class = rescued.groupby(["split", "label"]).size().reset_index(name="rescued") if not rescued.empty else pd.DataFrame(columns=["split", "label", "rescued"])
    remaining_by_class = split_class_summary(final_df).rename(columns={"fallback": "remaining_fallback"})
    top_strategies = rescued.groupby(["preprocess_mode", "min_confidence", "detection_size"]).size().reset_index(name="rescued").sort_values("rescued", ascending=False).head(15) if not rescued.empty else pd.DataFrame(columns=["preprocess_mode", "min_confidence", "detection_size", "rescued"])

    def table(df: pd.DataFrame, cols: list[str]) -> list[str]:
        if df.empty:
            return ["_No rows._"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---" for _ in cols]) + "|"]
        for row in df[cols].itertuples(index=False):
            vals = [f"{float(v):.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# D16 Pixel Prior Rescue Report",
        "",
        f"- output_prior_dir: `{output_prior_dir}`",
        "- scope: D16 MediaPipe pixel priors only",
        "- no training, no D16 model edits, no region-mask artifacts",
        "",
        "## Original Fallback",
        *table(original_split, ["split", "total", "detected", "fallback", "fallback_rate"]),
        "",
        "## Final Fallback",
        *table(final_split, ["split", "total", "detected", "fallback", "fallback_rate"]),
        "",
        "## Rescued By Class",
        *table(rescued_by_class.sort_values("rescued", ascending=False).head(20) if not rescued_by_class.empty else rescued_by_class, ["split", "label", "rescued"]),
        "",
        "## Remaining Fallback By Class",
        *table(remaining_by_class.sort_values(["remaining_fallback", "fallback_rate"], ascending=[False, False]).head(20) if not remaining_by_class.empty else remaining_by_class, ["split", "label", "total", "detected", "remaining_fallback", "fallback_rate"]),
        "",
        "## Top Successful Strategies",
        *table(top_strategies, ["preprocess_mode", "min_confidence", "detection_size", "rescued"]),
        "",
        "## Consistency Check",
        f"- attempted fallback samples: {len(attempts)}",
        f"- rescued samples: {int(len(rescued))}",
        f"- remaining fallback samples: {int(final_df['is_fallback'].sum()) if not final_df.empty else 0}",
        "- expected keys/shapes are checked by `check_d16_pixel_prior_rescue.py`.",
        "",
        "## Recommendation",
        "Run training on the rescued prior directory only after the checker passes. This report does not claim motif, causal, semantic-region, or region-mask evidence.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/dataset/fer13-split")
    parser.add_argument("--input_prior_dir", default="outputs/d16_mediapipe_pixel_priors_best")
    parser.add_argument("--output_prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--retry_only_fallback", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--reuse_existing_output", action="store_true", help="Use an existing output prior dir as the already-copied merge base.")
    parser.add_argument("--max_retry_samples_per_split", type=int, default=None)
    parser.add_argument("--min_confidences", nargs="+", type=float, default=[0.35, 0.25, 0.15])
    parser.add_argument("--detection_sizes", nargs="+", type=int, default=[224, 256, 320])
    parser.add_argument("--preprocess_modes", nargs="+", default=["raw", "equalize", "gamma_0_70", "gamma_0_55", "clahe", "sharpen", "pad8", "hflip", "rot_m10", "rot_p10"])
    parser.add_argument("--disable_rotations", action="store_true")
    parser.add_argument("--face_landmarker_model_path", default=None, help="Optional local .task model path for MediaPipe Tasks environments.")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    data_dir = _resolve_data_dir(Path(args.data_dir))
    input_prior_dir = Path(args.input_prior_dir)
    output_prior_dir = Path(args.output_prior_dir)
    copy_prior_tree(input_prior_dir, output_prior_dir, overwrite_output=bool(args.overwrite_output), reuse_existing=bool(args.reuse_existing_output))

    strategies = _strategy_grid(args)
    mode_ids = {mode: idx for idx, mode in enumerate(args.preprocess_modes)}
    detectors = {strategy.rank: RetryFaceDetector(strategy, model_path=args.face_landmarker_model_path) for strategy in strategies}
    original_records = iter_prior_records(input_prior_dir, args.splits)
    original_df = records_to_frame(original_records)
    target_records = [r for r in original_records if (r.is_fallback or not args.retry_only_fallback)]
    if args.max_retry_samples_per_split is not None:
        capped = []
        for split in args.splits:
            capped.extend([r for r in target_records if r.split == split][: int(args.max_retry_samples_per_split)])
        target_records = capped

    attempts: List[Dict] = []
    split_lookup: Dict[str, Dict[int, tuple[str, int]]] = {}
    try:
        for i, record in enumerate(target_records, start=1):
            if record.split not in split_lookup:
                split_lookup[record.split] = _read_split_lookup(data_dir, record.split)
            lookup = split_lookup[record.split]
            if record.sample_index not in lookup:
                raise KeyError(f"Missing sample_index={record.sample_index} in {data_dir / (record.split + '.csv')}")
            pixels, csv_label = lookup[record.sample_index]
            original = load_npz(record.path)
            label = int(record.label if record.label >= 0 else csv_label)
            image48 = pixels_to_image48(pixels)
            rescued_arrays, strategy, reason = _attempt_rescue(image48, label, record.sample_index, strategies, detectors)
            out_path = output_prior_dir / record.split / record.file_name
            if rescued_arrays is not None and strategy is not None:
                merged = merge_extra_keys(original, rescued_arrays, _metadata_arrays(True, strategy, mode_ids.get(strategy.preprocess_mode, -1), original))
                failures = validate_prior_arrays(merged)
                if failures:
                    raise RuntimeError(f"Rescued prior failed schema check for {out_path}: {failures}")
                save_npz(out_path, merged)
            else:
                merged = merge_extra_keys(original, {}, _metadata_arrays(False, None, -1, original))
                save_npz(out_path, merged)

            final = load_npz(out_path)
            attempts.append(
                {
                    "split": record.split,
                    "sample_index": int(record.sample_index),
                    "label": int(label),
                    "original_detected": bool(record.detected),
                    "original_landmark_missing_flag": int(record.landmark_missing_flag),
                    "rescue_success": bool(rescued_arrays is not None),
                    "final_detected": bool(np.asarray(final["detected"]).item()),
                    "final_landmark_missing_flag": int(np.asarray(final["landmark_missing_flag"]).item()),
                    "strategy_rank": -1 if strategy is None else int(strategy.rank),
                    "preprocess_mode": "not_rescued" if strategy is None else strategy.preprocess_mode,
                    "min_confidence": "" if strategy is None else float(strategy.min_confidence),
                    "detection_size": "" if strategy is None else int(strategy.detection_size),
                    "failure_reason": reason,
                    "fallback_type_id_before": "" if record.fallback_type_id is None else int(record.fallback_type_id),
                    "quality_score_before": "" if record.quality_score is None else float(record.quality_score),
                    "npz_path": str(out_path),
                }
            )
            if args.log_every > 0 and i % int(args.log_every) == 0:
                rescued_count = sum(1 for row in attempts if row["rescue_success"])
                print(f"[D16 pixel rescue] processed={i}/{len(target_records)} rescued={rescued_count}", flush=True)
    finally:
        for detector in detectors.values():
            detector.close()

    attempts_df = pd.DataFrame(attempts)
    attempts_df.to_csv(output_prior_dir / "pixel_rescue_attempts.csv", index=False)
    final_df = records_to_frame(iter_prior_records(output_prior_dir, args.splits))
    original_split = split_summary(original_df)
    final_split = split_summary(final_df)
    rescued_df = attempts_df[attempts_df["rescue_success"].astype(bool)] if not attempts_df.empty else attempts_df
    summary = {
        "input_prior_dir": str(input_prior_dir),
        "output_prior_dir": str(output_prior_dir),
        "splits": list(args.splits),
        "retry_only_fallback": bool(args.retry_only_fallback),
        "attempted": int(len(attempts_df)),
        "rescued": int(len(rescued_df)),
        "original_by_split": original_split.to_dict(orient="records"),
        "final_by_split": final_split.to_dict(orient="records"),
        "strategies": [strategy.__dict__ for strategy in strategies],
    }
    write_json(output_prior_dir / "pixel_rescue_summary.json", summary)

    metadata = schema_metadata()
    metadata.update(
        {
            "source_prior_dir": str(input_prior_dir),
            "rescue_pipeline": "d16_mediapipe_pixel_retry_rescue",
            "rescue_preprocess_modes": list(args.preprocess_modes),
            "rescue_min_confidences": [float(x) for x in args.min_confidences],
            "rescue_detection_sizes": [int(x) for x in args.detection_sizes],
            "fallback_types": FALLBACK_TYPES,
        }
    )
    write_json(output_prior_dir / "prior_schema.json", metadata)
    write_json(output_prior_dir / "part_names.json", PART_NAMES)
    write_json(output_prior_dir / "micro_anchor_names.json", MICRO_ANCHOR_NAMES)
    _write_retry_report(output_prior_dir, Path("outputs/d16_analysis/D16_PIXEL_PRIOR_RESCUE_REPORT.md"), attempts_df, original_df, final_df)

    print(json.dumps({"output_prior_dir": str(output_prior_dir), "attempted": int(len(attempts_df)), "rescued": int(len(rescued_df))}, indent=2))


if __name__ == "__main__":
    main()
