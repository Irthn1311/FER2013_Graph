"""Inspect D16 prior directory and write shape/coverage checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_KEYS = {
    "image_48": (48, 48),
    "face_mask": (48, 48),
    "part_soft_masks": None,
    "micro_anchor_maps": None,
    "distance_maps": None,
    "landmark_xy_48": None,
    "detected": (),
    "fallback_type_id": (),
    "landmark_missing_flag": (),
    "valid_part_mask": None,
    "valid_anchor_mask": None,
    "quality_score": (),
    "label": (),
    "sample_index": (),
}


def inspect_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(set(REQUIRED_KEYS) - keys)
        shapes = {key: list(data[key].shape) for key in data.files}
        failures = []
        for key, expected in REQUIRED_KEYS.items():
            if key not in data:
                continue
            if expected is not None and tuple(data[key].shape) != tuple(expected):
                failures.append(f"{path.name}:{key} shape {data[key].shape} != {expected}")
        if "part_soft_masks" in data and "valid_part_mask" in data and data["part_soft_masks"].shape[0] != data["valid_part_mask"].shape[0]:
            failures.append(f"{path.name}: part count mismatch")
        if "micro_anchor_maps" in data and "valid_anchor_mask" in data and data["micro_anchor_maps"].shape[0] != data["valid_anchor_mask"].shape[0]:
            failures.append(f"{path.name}: anchor count mismatch")
        return {
            "path": str(path),
            "missing": missing,
            "shapes": shapes,
            "failures": failures,
            "detected": bool(data["detected"].item()) if "detected" in data else False,
            "label": int(data["label"].item()) if "label" in data else -1,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_files_per_split", type=int, default=200)
    args = parser.parse_args()
    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for split_dir in sorted(path for path in prior_dir.iterdir() if path.is_dir() and path.name in ("train", "val", "test")):
        for path in sorted(split_dir.glob("*.npz"))[: int(args.max_files_per_split)]:
            info = inspect_npz(path)
            rows.append({"split": split_dir.name, "path": path.name, "detected": info["detected"], "label": info["label"], "failures": "; ".join(info["failures"] + info["missing"])})
            failures.extend(info["failures"])
            failures.extend([f"{path.name}: missing {key}" for key in info["missing"]])
    pd.DataFrame(rows).to_csv(output_dir / "prior_inspection.csv", index=False)
    summary = {
        "prior_dir": str(prior_dir),
        "checked_files": len(rows),
        "failure_count": len(failures),
        "failures": failures[:100],
        "decision": "D16_PRIOR_INSPECTION_PASS" if not failures else "D16_PRIOR_INSPECTION_FAIL",
    }
    (output_dir / "prior_inspection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
