"""Shared utilities for D16 MediaPipe pixel-prior rescue scripts."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


EXPECTED_SHAPES = {
    "image_48": (48, 48),
    "face_mask": (48, 48),
    "part_soft_masks": (13, 48, 48),
    "micro_anchor_maps": (12, 48, 48),
    "distance_maps": (12, 48, 48),
    "valid_part_mask": (13,),
    "valid_anchor_mask": (12,),
}

REQUIRED_KEYS = [
    "image_48",
    "face_mask",
    "part_soft_masks",
    "micro_anchor_maps",
    "distance_maps",
    "detected",
    "fallback_type_id",
    "landmark_missing_flag",
    "valid_part_mask",
    "valid_anchor_mask",
    "quality_score",
    "label",
    "sample_index",
]


@dataclass(frozen=True)
class PriorRecord:
    split: str
    path: Path
    file_name: str
    sample_index: int
    label: int
    detected: bool
    landmark_missing_flag: int
    fallback_type_id: int | None
    quality_score: float | None

    @property
    def is_fallback(self) -> bool:
        return bool(self.landmark_missing_flag == 1 or not self.detected)


def scalar(data: Dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def record_from_npz(split: str, path: Path) -> PriorRecord:
    data = load_npz(path)
    detected = bool(scalar(data, "detected", False))
    missing = int(scalar(data, "landmark_missing_flag", 1))
    return PriorRecord(
        split=split,
        path=path,
        file_name=path.name,
        sample_index=int(scalar(data, "sample_index", path.stem)),
        label=int(scalar(data, "label", -1)),
        detected=detected,
        landmark_missing_flag=missing,
        fallback_type_id=None if "fallback_type_id" not in data else int(scalar(data, "fallback_type_id")),
        quality_score=None if "quality_score" not in data else float(scalar(data, "quality_score")),
    )


def iter_prior_records(prior_dir: Path, splits: Iterable[str] = ("train", "val", "test")) -> List[PriorRecord]:
    rows: List[PriorRecord] = []
    for split in splits:
        split_dir = prior_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.npz")):
            rows.append(record_from_npz(split, path))
    return rows


def records_to_frame(records: Iterable[PriorRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": r.split,
                "file_name": r.file_name,
                "sample_index": r.sample_index,
                "label": r.label,
                "detected": bool(r.detected),
                "landmark_missing_flag": int(r.landmark_missing_flag),
                "fallback_type_id": "" if r.fallback_type_id is None else int(r.fallback_type_id),
                "quality_score": "" if r.quality_score is None else float(r.quality_score),
                "is_fallback": bool(r.is_fallback),
                "npz_path": str(r.path),
            }
            for r in records
        ]
    )


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["split", "total", "fallback", "detected", "fallback_rate"])
    out = df.groupby("split").agg(total=("sample_index", "count"), fallback=("is_fallback", "sum"), detected=("detected", "sum")).reset_index()
    out["fallback_rate"] = out["fallback"] / out["total"].clip(lower=1)
    return out


def split_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["split", "label", "total", "fallback", "detected", "fallback_rate"])
    out = df.groupby(["split", "label"]).agg(total=("sample_index", "count"), fallback=("is_fallback", "sum"), detected=("detected", "sum")).reset_index()
    out["fallback_rate"] = out["fallback"] / out["total"].clip(lower=1)
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_prior_tree(input_prior_dir: Path, output_prior_dir: Path, overwrite_output: bool = False, reuse_existing: bool = False) -> None:
    if output_prior_dir.exists():
        if reuse_existing:
            return
        if not overwrite_output:
            raise FileExistsError(f"Output prior dir already exists: {output_prior_dir}. Pass --overwrite_output to replace it.")
        shutil.rmtree(output_prior_dir)
    ignore = shutil.ignore_patterns("coverage_analysis", "inspect", "figures")
    shutil.copytree(input_prior_dir, output_prior_dir, ignore=ignore)


def validate_prior_arrays(data: Dict[str, np.ndarray]) -> List[str]:
    failures: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in data:
            failures.append(f"missing_key:{key}")
    for key, shape in EXPECTED_SHAPES.items():
        if key in data and tuple(np.asarray(data[key]).shape) != tuple(shape):
            failures.append(f"bad_shape:{key}:{tuple(np.asarray(data[key]).shape)}")
    if "detected" in data and "landmark_missing_flag" in data:
        detected = bool(scalar(data, "detected", False))
        missing = int(scalar(data, "landmark_missing_flag", 1))
        if detected and missing != 0:
            failures.append("inconsistent_detected_true_missing_not_zero")
        if (not detected) and missing != 1:
            failures.append("inconsistent_detected_false_missing_not_one")
    return failures


def merge_extra_keys(original: Dict[str, np.ndarray], regenerated: Dict[str, np.ndarray], extra: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(original)
    out.update(regenerated)
    out.update(extra)
    return out
