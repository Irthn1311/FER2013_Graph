"""Sweep D16 MediaPipe detection preprocessing choices.

This script measures coverage only. It does not train D16 and does not change
model architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.fer_csv import read_fer_split
from d16.data.mediapipe_priors import (
    MICRO_ANCHOR_NAMES,
    PART_NAMES,
    MediaPipeFaceDetector,
    detected_priors,
    fallback_priors,
    image48_to_rgb_uint8,
    pixels_to_image48,
)


@dataclass(frozen=True)
class TuningConfig:
    name: str
    attempts: tuple[dict, ...]


SWEEP_CONFIGS = [
    TuningConfig("baseline_192_raw", ({"detection_size": 192, "preprocess_mode": "raw", "padding_pixels": 0},)),
    TuningConfig("size256_raw", ({"detection_size": 256, "preprocess_mode": "raw", "padding_pixels": 0},)),
    TuningConfig("size320_raw", ({"detection_size": 320, "preprocess_mode": "raw", "padding_pixels": 0},)),
    TuningConfig("size256_equalize", ({"detection_size": 256, "preprocess_mode": "histogram_equalize", "padding_pixels": 0},)),
    TuningConfig("size256_clahe", ({"detection_size": 256, "preprocess_mode": "clahe", "padding_pixels": 0},)),
    TuningConfig("size256_contrast", ({"detection_size": 256, "preprocess_mode": "contrast_stretch", "padding_pixels": 0},)),
    TuningConfig("size256_padding", ({"detection_size": 256, "preprocess_mode": "raw", "padding_pixels": 6},)),
    TuningConfig(
        "multi_attempt",
        (
            {"detection_size": 192, "preprocess_mode": "raw", "padding_pixels": 0},
            {"detection_size": 256, "preprocess_mode": "histogram_equalize", "padding_pixels": 0},
            {"detection_size": 320, "preprocess_mode": "clahe", "padding_pixels": 0},
        ),
    ),
]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def _load_split(data_dir: Path, split: str, sample_mode: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = read_fer_split(data_dir, split, max_samples=None)
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)
    if sample_mode == "full":
        return df.head(max_samples).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pieces = []
    labels = sorted(df["emotion"].unique().tolist())
    per_label = max(int(max_samples) // max(len(labels), 1), 1)
    for label in labels:
        label_df = df[df["emotion"] == label]
        take = min(len(label_df), per_label)
        if take > 0:
            pieces.append(label_df.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(pieces, axis=0) if pieces else df.head(0)
    if len(out) < max_samples:
        remaining = df.drop(index=out.index, errors="ignore")
        take = min(max_samples - len(out), len(remaining))
        if take > 0:
            out = pd.concat([out, remaining.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))], axis=0)
    return out.sort_values(["emotion", "sample_index"]).reset_index(drop=True)


def _make_detectors(config: TuningConfig, min_detection_confidence: float) -> List[MediaPipeFaceDetector]:
    return [
        MediaPipeFaceDetector(
            int(attempt["detection_size"]),
            min_detection_confidence=min_detection_confidence,
            preprocess_mode=str(attempt["preprocess_mode"]),
            padding_pixels=int(attempt.get("padding_pixels", 0)),
        )
        for attempt in config.attempts
    ]


def _detect_with_attempts(image48: np.ndarray, config: TuningConfig, detectors: List[MediaPipeFaceDetector]):
    for attempt_idx, (attempt, detector) in enumerate(zip(config.attempts, detectors)):
        landmark_xy = detector.detect(image48)
        if landmark_xy is not None:
            rgb = image48_to_rgb_uint8(
                image48,
                int(attempt["detection_size"]),
                preprocess_mode=str(attempt["preprocess_mode"]),
                padding_pixels=int(attempt.get("padding_pixels", 0)),
            )
            return landmark_xy, attempt_idx, attempt, rgb
    attempt = config.attempts[-1]
    rgb = image48_to_rgb_uint8(
        image48,
        int(attempt["detection_size"]),
        preprocess_mode=str(attempt["preprocess_mode"]),
        padding_pixels=int(attempt.get("padding_pixels", 0)),
    )
    return None, -1, attempt, rgb


def _mask_sanity(arrays: Dict[str, np.ndarray]) -> tuple[bool, float, str]:
    face = np.asarray(arrays["face_mask"], dtype=np.float32)
    parts = np.asarray(arrays["part_soft_masks"], dtype=np.float32)
    valid_parts = np.asarray(arrays["valid_part_mask"], dtype=np.float32)
    face_area = float(np.mean(face > 0.5))
    mouth_idx = PART_NAMES.index("mouth")
    left_eye_idx = PART_NAMES.index("left_eye")
    right_eye_idx = PART_NAMES.index("right_eye")
    mouth_peak = float(parts[mouth_idx].max()) if valid_parts[mouth_idx] > 0 else 0.0
    eye_peak = max(
        float(parts[left_eye_idx].max()) if valid_parts[left_eye_idx] > 0 else 0.0,
        float(parts[right_eye_idx].max()) if valid_parts[right_eye_idx] > 0 else 0.0,
    )
    valid_ratio = float(valid_parts.mean())
    ok = 0.12 <= face_area <= 0.95 and mouth_peak > 0.05 and eye_peak > 0.05 and valid_ratio > 0.5
    score = float(np.mean([0.12 <= face_area <= 0.95, mouth_peak > 0.05, eye_peak > 0.05, valid_ratio > 0.5]))
    reason = f"face_area={face_area:.3f},mouth_peak={mouth_peak:.3f},eye_peak={eye_peak:.3f},valid_part_ratio={valid_ratio:.3f}"
    return bool(ok), score, reason


def _figure(path: Path, arrays: Dict[str, np.ndarray], rgb: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = arrays["image_48"]
    face = arrays["face_mask"]
    parts = arrays["part_soft_masks"]
    landmarks = arrays["landmark_xy_48"]
    mouth = parts[PART_NAMES.index("mouth")]
    eye_brow = np.maximum.reduce(
        [
            parts[PART_NAMES.index("left_eye")],
            parts[PART_NAMES.index("right_eye")],
            parts[PART_NAMES.index("left_brow")],
            parts[PART_NAMES.index("right_brow")],
        ]
    )
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("48x48")
    axes[1].imshow(rgb)
    axes[1].set_title("detect img")
    axes[2].imshow(image, cmap="gray")
    if len(landmarks) > 0:
        axes[2].scatter(landmarks[:, 0], landmarks[:, 1], s=2, c="lime")
    axes[2].set_title("landmarks")
    axes[3].imshow(face, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("face")
    axes[4].imshow(mouth, cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("mouth")
    axes[5].imshow(eye_brow, cmap="magma", vmin=0, vmax=1)
    axes[5].set_title("eye/brow")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_examples(config_dir: Path, rows: pd.DataFrame, max_examples: int = 20) -> None:
    if rows.empty:
        return
    buckets = {
        "detected": rows[rows["detected"]].head(max_examples),
        "fallback": rows[~rows["detected"]].head(max_examples),
        "low_quality_detected": rows[rows["detected"]].sort_values("mask_sanity_score").head(max_examples),
        "label1": rows[rows["label"] == 1].head(max_examples),
    }
    for bucket, bucket_rows in buckets.items():
        for idx, row in enumerate(bucket_rows.itertuples(index=False)):
            arrays = row.arrays
            rgb = row.rgb
            title = f"{bucket} split={row.split} label={int(row.label)} sample={int(row.sample_index)} detected={bool(row.detected)}"
            _figure(config_dir / "examples" / bucket / f"{bucket}_{idx:02d}_{row.split}_{int(row.sample_index):06d}.png", arrays, rgb, title)


def _output_size_estimate(total_samples: int) -> Dict[str, float]:
    prior_dir = Path("outputs/d16_mediapipe_pixel_priors")
    files = list(prior_dir.glob("*/*.npz")) if prior_dir.exists() else []
    if files:
        sample = files[: min(len(files), 500)]
        avg_bytes = float(sum(p.stat().st_size for p in sample) / max(len(sample), 1))
    else:
        # Rough compressed size estimate from current D16 prior arrays.
        avg_bytes = 32_000.0
    return {
        "avg_npz_bytes": avg_bytes,
        "estimated_full_output_mb": avg_bytes * float(total_samples) / (1024.0 * 1024.0),
    }


def _summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby("split")
        .agg(
            total=("sample_index", "count"),
            detected=("detected", "sum"),
            quality_mean=("quality_score", "mean"),
            sanity_pass_rate=("mask_sanity_pass", "mean"),
            runtime_per_sample_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary["fallback"] = summary["total"] - summary["detected"]
    summary["fallback_rate"] = summary["fallback"] / summary["total"].clip(lower=1)

    by_class = (
        rows.groupby(["split", "label"])
        .agg(total=("sample_index", "count"), detected=("detected", "sum"), quality_mean=("quality_score", "mean"))
        .reset_index()
    )
    by_class["fallback"] = by_class["total"] - by_class["detected"]
    by_class["fallback_rate"] = by_class["fallback"] / by_class["total"].clip(lower=1)

    fallback_cases = rows[~rows["detected"]][["split", "sample_index", "label", "attempt_used", "quality_score", "mask_sanity_reason"]]
    quality = rows.groupby("split")["quality_score"].agg(["count", "mean", "std", "min", "max"]).reset_index().fillna({"std": 0.0})
    return summary, by_class, fallback_cases, quality


def _config_decision(config_name: str, summary: pd.DataFrame, by_class: pd.DataFrame, baseline_rate: Optional[float]) -> str:
    max_split = float(summary["fallback_rate"].max()) if not summary.empty else 1.0
    sanity = float(summary["sanity_pass_rate"].mean()) if "sanity_pass_rate" in summary else 0.0
    improvement = (baseline_rate - max_split) if baseline_rate is not None else 0.0
    if sanity < 0.85:
        return "RISKY_VISUAL_SANITY"
    if baseline_rate is not None and improvement >= 0.03:
        return "CANDIDATE"
    if config_name == "baseline_192_raw":
        return "BASELINE"
    return "NO_MEANINGFUL_IMPROVEMENT"


def _write_config_report(config_dir: Path, config: TuningConfig, summary: pd.DataFrame, by_class: pd.DataFrame, quality: pd.DataFrame, decision: str, metadata: Dict) -> None:
    worst = by_class.sort_values("fallback_rate", ascending=False).head(10)
    lines = [
        f"# D16 MediaPipe Tuning Report - {config.name}",
        "",
        f"- decision: `{decision}`",
        f"- attempts: `{config.attempts}`",
        f"- runtime_per_sample_sec_mean: {metadata.get('runtime_per_sample_sec', 0.0):.5f}",
        f"- estimated_full_output_mb: {metadata.get('estimated_full_output_mb', 0.0):.2f}",
        "",
        "## Coverage",
        "| split | total | detected | fallback | fallback_rate | quality_mean | sanity_pass_rate | runtime_per_sample_sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {int(row.total)} | {int(row.detected)} | {int(row.fallback)} | {float(row.fallback_rate):.4f} | "
            f"{float(row.quality_mean):.4f} | {float(row.sanity_pass_rate):.4f} | {float(row.runtime_per_sample_sec):.5f} |"
        )
    lines.extend(["", "## Worst Class Fallback", "| split | label | total | detected | fallback | fallback_rate | quality_mean |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.split} | {int(row.label)} | {int(row.total)} | {int(row.detected)} | {int(row.fallback)} | {float(row.fallback_rate):.4f} | {float(row.quality_mean):.4f} |"
        )
    lines.extend(["", "## Quality", "| split | count | mean | std | min | max |", "|---|---:|---:|---:|---:|---:|"])
    for row in quality.itertuples(index=False):
        lines.append(f"| {row.split} | {int(row.count)} | {float(row.mean):.4f} | {float(row.std):.4f} | {float(row.min):.4f} | {float(row.max):.4f} |")
    lines.extend(["", "## Visual Examples", "- `examples/detected/`", "- `examples/fallback/`", "- `examples/low_quality_detected/`", "- `examples/label1/`", "", "No motif, semantic-region, causal-evidence, or interpretability claim is made."])
    (config_dir / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_config(
    config: TuningConfig,
    split_frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    min_detection_confidence: float,
    baseline_rate: Optional[float],
    log_every: int,
    detector_recreate_every: int,
) -> Dict:
    config_dir = output_dir / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    detectors = _make_detectors(config, min_detection_confidence=min_detection_confidence)
    rows = []
    start = time.perf_counter()
    samples_since_recreate = 0
    try:
        for split, df in split_frames.items():
            for count, row in enumerate(df.itertuples(index=False), start=1):
                if detector_recreate_every > 0 and samples_since_recreate >= detector_recreate_every:
                    for detector in detectors:
                        detector.close()
                    detectors = _make_detectors(config, min_detection_confidence=min_detection_confidence)
                    samples_since_recreate = 0
                    print(f"[D16 tuning] recreated detectors for config={config.name} split={split} count={count}", flush=True)
                sample_start = time.perf_counter()
                sample_index = int(row.sample_index)
                label = int(row.emotion)
                image48 = pixels_to_image48(row.pixels)
                landmark_xy, attempt_idx, attempt, rgb = _detect_with_attempts(image48, config, detectors)
                samples_since_recreate += 1
                if landmark_xy is None:
                    result = fallback_priors(image48, label=label, sample_index=sample_index)
                else:
                    result = detected_priors(image48, landmark_xy=landmark_xy, label=label, sample_index=sample_index)
                sanity_ok, sanity_score, sanity_reason = _mask_sanity(result.arrays)
                rows.append(
                    {
                        "split": split,
                        "sample_index": sample_index,
                        "label": label,
                        "detected": bool(result.detected),
                        "attempt_used": int(attempt_idx),
                        "attempt_name": "fallback" if attempt_idx < 0 else f"{attempt_idx}:{attempt['detection_size']}_{attempt['preprocess_mode']}",
                        "quality_score": float(result.quality_score),
                        "mask_sanity_pass": bool(sanity_ok) if result.detected else True,
                        "mask_sanity_score": float(sanity_score) if result.detected else 1.0,
                        "mask_sanity_reason": sanity_reason,
                        "runtime_sec": float(time.perf_counter() - sample_start),
                        "arrays": result.arrays,
                        "rgb": rgb,
                    }
                )
                if log_every > 0 and count % int(log_every) == 0:
                    partial = pd.DataFrame([{k: v for k, v in item.items() if k not in ("arrays", "rgb")} for item in rows if item["split"] == split])
                    detected = int(partial["detected"].sum()) if not partial.empty else 0
                    print(f"[D16 tuning] config={config.name} split={split} processed={count} detected={detected}", flush=True)
    finally:
        for detector in detectors:
            detector.close()

    rows_df_full = pd.DataFrame(rows)
    _save_examples(config_dir, rows_df_full, max_examples=20)
    rows_df = rows_df_full.drop(columns=["arrays", "rgb"])
    rows_df.to_csv(config_dir / "coverage_rows.csv", index=False)
    summary, by_class, fallback_cases, quality = _summaries(rows_df)
    summary.to_csv(config_dir / "coverage_summary.csv", index=False)
    by_class.to_csv(config_dir / "coverage_by_class.csv", index=False)
    fallback_cases.to_csv(config_dir / "fallback_cases.csv", index=False)
    quality.to_csv(config_dir / "quality_distribution.csv", index=False)

    total_samples = int(summary["total"].sum()) if not summary.empty else 0
    size_estimate = _output_size_estimate(total_samples)
    runtime_total = float(time.perf_counter() - start)
    runtime_per_sample = runtime_total / max(total_samples, 1)
    decision = _config_decision(config.name, summary, by_class, baseline_rate)
    metadata = {
        "config": config.name,
        "attempts": config.attempts,
        "total_samples": total_samples,
        "runtime_total_sec": runtime_total,
        "runtime_per_sample_sec": runtime_per_sample,
        **size_estimate,
        "decision": decision,
    }
    _write_json(config_dir / "tuning_summary.json", metadata)
    _write_config_report(config_dir, config, summary, by_class, quality, decision, metadata)
    max_class = float(by_class["fallback_rate"].max()) if not by_class.empty else 1.0
    label1 = by_class[by_class["label"] == 1].copy()
    return {
        "config": config.name,
        "train_fallback_rate": float(summary.loc[summary["split"] == "train", "fallback_rate"].iloc[0]) if "train" in set(summary["split"]) else np.nan,
        "val_fallback_rate": float(summary.loc[summary["split"] == "val", "fallback_rate"].iloc[0]) if "val" in set(summary["split"]) else np.nan,
        "test_fallback_rate": float(summary.loc[summary["split"] == "test", "fallback_rate"].iloc[0]) if "test" in set(summary["split"]) else np.nan,
        "max_split_fallback_rate": float(summary["fallback_rate"].max()) if not summary.empty else np.nan,
        "max_class_fallback_rate": max_class,
        "label1_max_fallback_rate": float(label1["fallback_rate"].max()) if not label1.empty else np.nan,
        "quality_mean": float(rows_df["quality_score"].mean()) if not rows_df.empty else np.nan,
        "sanity_pass_rate": float(rows_df.loc[rows_df["detected"], "mask_sanity_pass"].mean()) if rows_df["detected"].any() else 0.0,
        "runtime_per_sample_sec": runtime_per_sample,
        "estimated_full_output_mb": float(size_estimate["estimated_full_output_mb"]),
        "decision": decision,
    }


def _final_decision(results: pd.DataFrame, baseline_name: str = "baseline_192_raw") -> str:
    if results.empty:
        return "MEDIAPIPE_TUNING_RISKY_KEEP_FALLBACK"
    baseline_rows = results[results["config"] == baseline_name]
    baseline = float(baseline_rows["max_split_fallback_rate"].iloc[0]) if not baseline_rows.empty else float(results["max_split_fallback_rate"].max())
    baseline_runtime = float(baseline_rows["runtime_per_sample_sec"].iloc[0]) if not baseline_rows.empty else float(results["runtime_per_sample_sec"].min())
    candidates = results[
        (results["max_split_fallback_rate"] <= baseline - 0.03)
        & (results["sanity_pass_rate"] >= 0.85)
        & (results["runtime_per_sample_sec"] <= baseline_runtime * 2.0)
    ].copy()
    multi = results[(results["config"] == "multi_attempt") & (results["max_split_fallback_rate"] <= baseline - 0.03) & (results["sanity_pass_rate"] >= 0.85)]
    if not multi.empty:
        return "USE_MULTI_ATTEMPT_PIPELINE"
    if candidates.empty:
        return "MEDIAPIPE_TUNING_RISKY_KEEP_FALLBACK"
    best = candidates.sort_values(["max_split_fallback_rate", "runtime_per_sample_sec"]).iloc[0]["config"]
    mapping = {
        "baseline_192_raw": "USE_BASELINE_192_RAW",
        "size256_raw": "USE_SIZE256_RAW",
        "size256_equalize": "USE_SIZE256_EQUALIZE",
        "size256_clahe": "USE_SIZE256_CLAHE",
    }
    return mapping.get(str(best), "MEDIAPIPE_TUNING_RISKY_KEEP_FALLBACK")


def _write_decision_report(output_dir: Path, results: pd.DataFrame, final_decision: str, sample_mode: str, max_samples_per_split: Optional[int]) -> None:
    baseline_current = {"train": 0.1764, "val": 0.1711, "test": 0.1831}
    baseline_rows = results[results["config"] == "baseline_192_raw"]
    runtime_base = float(baseline_rows["runtime_per_sample_sec"].iloc[0]) if not baseline_rows.empty else np.nan
    best = results.sort_values("max_split_fallback_rate").head(1)
    lines = [
        "# D16 MediaPipe Tuning Decision Report",
        "",
        "## 1. Baseline Coverage",
        f"- current full train fallback_rate: {baseline_current['train']:.4f}",
        f"- current full val fallback_rate: {baseline_current['val']:.4f}",
        f"- current full test fallback_rate: {baseline_current['test']:.4f}",
        f"- sweep sample_mode: `{sample_mode}`",
        f"- max_samples_per_split: `{max_samples_per_split}`",
        "",
        "## 2. Sweep Results",
        "| config | train_fallback_rate | val_fallback_rate | test_fallback_rate | max_class_fallback_rate | label1_max_fallback_rate | quality_mean | sanity_pass_rate | runtime_per_sample | runtime_vs_baseline | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results.itertuples(index=False):
        runtime_vs = float(row.runtime_per_sample_sec) / runtime_base if runtime_base and runtime_base > 0 else np.nan
        lines.append(
            f"| {row.config} | {float(row.train_fallback_rate):.4f} | {float(row.val_fallback_rate):.4f} | {float(row.test_fallback_rate):.4f} | "
            f"{float(row.max_class_fallback_rate):.4f} | {float(row.label1_max_fallback_rate):.4f} | {float(row.quality_mean):.4f} | "
            f"{float(row.sanity_pass_rate):.4f} | {float(row.runtime_per_sample_sec):.5f} | {runtime_vs:.2f}x | {row.decision} |"
        )
    lines.extend(["", "## 3. Class Bias"])
    if best.empty:
        lines.append("- no results")
    else:
        lines.append(f"- lowest max split fallback config: `{best.iloc[0]['config']}`")
        lines.append(f"- label 1 max fallback in best config: {float(best.iloc[0]['label1_max_fallback_rate']):.4f}")
        lines.append("- label 1 remains the key class to monitor if its fallback rate stays high.")
    lines.extend(
        [
            "",
            "## 4. Visual Sanity",
            "- Visual sanity is approximated by generated examples and mask sanity pass rate.",
            "- Review `examples/detected`, `examples/fallback`, `examples/low_quality_detected`, and `examples/label1` for top candidates.",
            "",
            "## 5. Runtime Tradeoff",
            "- Runtime is measured as detector plus prior-building time per sample inside this sweep.",
            "- Size 320 and multi-attempt should be accepted only when fallback reduction is meaningful because precompute is offline but still costly.",
            "",
            "## 6. Final Recommendation",
            f"- decision: `{final_decision}`",
            "",
            "Allowed decisions: USE_BASELINE_192_RAW, USE_SIZE256_RAW, USE_SIZE256_EQUALIZE, USE_SIZE256_CLAHE, USE_MULTI_ATTEMPT_PIPELINE, MEDIAPIPE_TUNING_RISKY_KEEP_FALLBACK.",
            "",
            "No full D16 training is launched here. No motif, semantic-region, causal-evidence, or interpretability claim is made.",
        ]
    )
    (output_dir / "D16_MEDIAPIPE_TUNING_DECISION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/fer13-split")
    parser.add_argument("--output_dir", default="outputs/d16_mediapipe_tuning")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--sample_mode", choices=["full", "stratified"], default="stratified")
    parser.add_argument("--max_samples_per_split", type=int, default=700, help="Set <=0 to use all samples in each split.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--configs", nargs="+", default=[cfg.name for cfg in SWEEP_CONFIGS])
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--detector_recreate_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    max_samples_per_split = args.max_samples_per_split
    if max_samples_per_split is not None and int(max_samples_per_split) <= 0:
        max_samples_per_split = None

    available = {cfg.name: cfg for cfg in SWEEP_CONFIGS}
    configs = [available[name] for name in args.configs if name in available]
    missing = [name for name in args.configs if name not in available]
    if missing:
        raise ValueError(f"Unknown tuning configs: {missing}")
    if not _opencv_available() and any(cfg.name in {"size256_clahe", "multi_attempt"} for cfg in configs):
        print("[D16 tuning] OpenCV not available; CLAHE attempts will fall back to histogram equalization.")

    split_frames = {
        split: _load_split(data_dir, split, args.sample_mode, max_samples_per_split, seed=args.seed + idx)
        for idx, split in enumerate(args.splits)
    }
    split_counts = {split: len(df) for split, df in split_frames.items()}
    _write_json(
        output_dir / "tuning_input_manifest.json",
        {
            "data_dir": str(data_dir),
            "splits": args.splits,
            "sample_mode": args.sample_mode,
            "max_samples_per_split": max_samples_per_split,
            "split_counts": split_counts,
            "opencv_available": _opencv_available(),
            "configs": [cfg.name for cfg in configs],
        },
    )

    results = []
    baseline_rate = None
    for config in configs:
        print(f"[D16 tuning] running config={config.name}", flush=True)
        result = _run_config(
            config,
            split_frames=split_frames,
            output_dir=output_dir,
            min_detection_confidence=args.min_detection_confidence,
            baseline_rate=baseline_rate,
            log_every=args.log_every,
            detector_recreate_every=int(args.detector_recreate_every),
        )
        results.append(result)
        if config.name == "baseline_192_raw":
            baseline_rate = float(result["max_split_fallback_rate"])

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "d16_mediapipe_tuning_sweep_results.csv", index=False)
    final_decision = _final_decision(results_df)
    _write_decision_report(output_dir, results_df, final_decision, args.sample_mode, max_samples_per_split)
    _write_json(output_dir / "d16_mediapipe_tuning_decision.json", {"decision": final_decision, "results": results})
    print(json.dumps({"decision": final_decision, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
