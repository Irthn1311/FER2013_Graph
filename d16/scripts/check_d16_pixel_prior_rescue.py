"""Validate a rescued D16 MediaPipe pixel-prior directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.mediapipe_pixel_rescue_utils import EXPECTED_SHAPES, REQUIRED_KEYS, iter_prior_records, load_npz, records_to_frame, split_summary, validate_prior_arrays, write_json


def _dtype_ok(key: str, value: np.ndarray) -> bool:
    arr = np.asarray(value)
    if key in ("detected", "rescue_success"):
        return arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer)
    if key in ("label", "sample_index", "landmark_missing_flag", "fallback_type_id"):
        return np.issubdtype(arr.dtype, np.integer)
    if key in EXPECTED_SHAPES or key in ("quality_score", "landmark_xy_48"):
        return np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    return True


def _check_files(prior_dir: Path, reference_prior_dir: Path, splits: list[str]) -> tuple[list[dict], list[str]]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for split in splits:
        current_files = sorted((prior_dir / split).glob("*.npz"))
        ref_files = sorted((reference_prior_dir / split).glob("*.npz"))
        if len(current_files) != len(ref_files):
            failures.append(f"{split}:file_count:{len(current_files)}!={len(ref_files)}")
        ref_by_name = {path.name: path for path in ref_files}
        for path in current_files:
            data = load_npz(path)
            file_failures = validate_prior_arrays(data)
            for key in REQUIRED_KEYS:
                if key in data and not _dtype_ok(key, data[key]):
                    file_failures.append(f"bad_dtype:{key}:{np.asarray(data[key]).dtype}")
            if path.name in ref_by_name:
                ref = load_npz(ref_by_name[path.name])
                for key in ("sample_index", "label"):
                    if key in data and key in ref and int(np.asarray(data[key]).item()) != int(np.asarray(ref[key]).item()):
                        file_failures.append(f"{key}_changed:{int(np.asarray(ref[key]).item())}->{int(np.asarray(data[key]).item())}")
            else:
                file_failures.append("missing_reference_file")
            if file_failures:
                failures.append(f"{split}/{path.name}:{';'.join(file_failures)}")
            rows.append(
                {
                    "split": split,
                    "file_name": path.name,
                    "sample_index": int(np.asarray(data.get("sample_index", np.asarray(-1))).item()),
                    "label": int(np.asarray(data.get("label", np.asarray(-1))).item()),
                    "detected": bool(np.asarray(data.get("detected", np.asarray(False))).item()),
                    "landmark_missing_flag": int(np.asarray(data.get("landmark_missing_flag", np.asarray(1))).item()),
                    "failures": ";".join(file_failures),
                }
            )
    return rows, failures


def _dataset_smoke(prior_dir: Path) -> dict:
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader

        from d16.data.graph_builder import collate_d16_graphs
        from d16.data.pixel_prior_dataset import D16PixelPriorDataset
    except Exception as exc:
        return {"status": "SKIPPED_TORCH_UNAVAILABLE", "reason": str(exc)}

    out = {"status": "PASS", "splits": {}}
    for split in ("train", "val", "test"):
        split_dir = prior_dir / split
        if not split_dir.exists() or not list(split_dir.glob("*.npz")):
            continue
        ds = D16PixelPriorDataset(prior_dir=prior_dir, split=split, graph_mode="face_plus_context", max_samples=4)
        first = ds[0]
        loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False, collate_fn=collate_d16_graphs, num_workers=0)
        batch = next(iter(loader))
        out["splits"][split] = {
            "dataset_len_checked": int(len(ds)),
            "first_sample_index": int(first.sample_index.item()),
            "batch_graphs": int(batch.num_graphs),
            "batch_nodes": int(batch.x_cat.shape[0]),
        }
    return out


def _table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---" for _ in columns]) + "|"]
    for row in df[columns].itertuples(index=False):
        vals = [f"{float(v):.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--reference_prior_dir", default="outputs/d16_mediapipe_pixel_priors_best")
    parser.add_argument("--output_dir", default="outputs/d16_analysis/pixel_prior_rescue_check")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    prior_dir = Path(args.prior_dir)
    reference_prior_dir = Path(args.reference_prior_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = _check_files(prior_dir, reference_prior_dir, args.splits)
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(output_dir / "d16_pixel_prior_rescue_check_rows.csv", index=False)
    final_df = records_to_frame(iter_prior_records(prior_dir, args.splits))
    fallback = split_summary(final_df)
    fallback.to_csv(output_dir / "d16_pixel_prior_rescue_fallback_by_split.csv", index=False)
    smoke = _dataset_smoke(prior_dir)
    status = "PASS" if not failures and smoke.get("status") in ("PASS", "SKIPPED_TORCH_UNAVAILABLE") else "FAIL"
    summary = {
        "status": status,
        "prior_dir": str(prior_dir),
        "reference_prior_dir": str(reference_prior_dir),
        "failure_count": int(len(failures)),
        "failures_first_50": failures[:50],
        "expected_shapes": {k: list(v) for k, v in EXPECTED_SHAPES.items()},
        "fallback_by_split": fallback.to_dict(orient="records"),
        "dataset_smoke": smoke,
    }
    write_json(output_dir / "d16_pixel_prior_rescue_check_summary.json", summary)

    lines = [
        "# D16 Pixel Prior Rescue Check",
        "",
        f"- status: `{status}`",
        f"- prior_dir: `{prior_dir}`",
        f"- reference_prior_dir: `{reference_prior_dir}`",
        "- schema: D16 MediaPipe pixel priors; no region masks",
        "",
        "## Final Fallback",
        *_table(fallback, ["split", "total", "detected", "fallback", "fallback_rate"]),
        "",
        "## Schema/File Check",
        f"- failure_count: {len(failures)}",
    ]
    if failures:
        lines.extend(["- first failures:", *[f"  - `{item}`" for item in failures[:20]]])
    lines.extend(
        [
            "",
            "## Dataset Smoke",
            f"- status: `{smoke.get('status')}`",
            f"- details: `{json.dumps(smoke, default=str)}`",
            "",
            "This checker only validates the rescued pixel-prior directory. It does not train and does not touch D16 model architecture.",
        ]
    )
    (output_dir / "D16_PIXEL_PRIOR_RESCUE_CHECK.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
