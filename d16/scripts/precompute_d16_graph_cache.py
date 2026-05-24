"""Precompute chunked D16 graph tensors from MediaPipe pixel priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_cache_dataset import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    D16GraphCacheDataset,
    compare_graphs,
    graph_to_cache_dict,
)
from d16.data.pixel_prior_dataset import D16PixelPriorDataset  # noqa: E402


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_existing_chunk(path: Path, expected_count: int) -> List[Dict[str, torch.Tensor]] | None:
    if not path.exists():
        return None
    try:
        rows = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(rows, list) or len(rows) != expected_count:
        return None
    return rows


def _save_chunk_atomic(rows: List[Dict[str, torch.Tensor]], chunk_path: Path) -> None:
    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(rows, tmp_path)
        tmp_path.replace(chunk_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _precompute_split(
    prior_dir: Path,
    output_dir: Path,
    split: str,
    graph_mode: str,
    face_threshold: float,
    context_pixels: int,
    chunk_size: int,
    max_samples: int | None,
    resume_existing: bool,
) -> Dict[str, Any]:
    dataset = D16PixelPriorDataset(
        prior_dir,
        split=split,
        graph_mode=graph_mode,
        face_threshold=face_threshold,
        context_pixels=context_pixels,
        max_samples=max_samples,
    )
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[Dict[str, Any]] = []
    for chunk_index, chunk_start in enumerate(range(0, len(dataset), chunk_size)):
        expected_count = min(chunk_size, len(dataset) - chunk_start)
        chunk_path = split_dir / f"chunk_{chunk_index:05d}.pt"
        if resume_existing:
            existing_rows = _load_existing_chunk(chunk_path, expected_count)
            if existing_rows is not None:
                chunks.append({"path": str(chunk_path.relative_to(output_dir)), "start": chunk_start, "count": expected_count})
                continue
            if chunk_path.exists():
                chunk_path.unlink()
        rows = [graph_to_cache_dict(dataset[idx]) for idx in range(chunk_start, chunk_start + expected_count)]
        _save_chunk_atomic(rows, chunk_path)
        chunks.append({"path": str(chunk_path.relative_to(output_dir)), "start": chunk_start, "count": expected_count})
    return {"count": len(dataset), "chunks": chunks}


def _verify_split(
    prior_dir: Path,
    cache_dir: Path,
    split: str,
    graph_mode: str,
    face_threshold: float,
    context_pixels: int,
    max_checks: int,
    max_samples: int | None,
) -> List[str]:
    online = D16PixelPriorDataset(
        prior_dir,
        split=split,
        graph_mode=graph_mode,
        face_threshold=face_threshold,
        context_pixels=context_pixels,
        max_samples=max_samples,
    )
    cached = D16GraphCacheDataset(
        cache_dir,
        split=split,
        graph_mode=graph_mode,
        face_threshold=face_threshold,
        context_pixels=context_pixels,
        max_samples=None,
    )
    failures: List[str] = []
    if len(online) != len(cached):
        failures.append(f"{split}: length mismatch online={len(online)} cached={len(cached)}")
        return failures
    if len(online) == 0:
        return failures
    if max_checks <= 0 or max_checks >= len(online):
        indices = list(range(len(online)))
    else:
        step = max(len(online) // max_checks, 1)
        indices = list(range(0, len(online), step))[:max_checks]
        if (len(online) - 1) not in indices:
            indices.append(len(online) - 1)
    for idx in indices:
        diff = compare_graphs(online[idx], cached[idx])
        failures.extend([f"{split}[{idx}]: {item}" for item in diff])
        if len(failures) >= 50:
            break
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--graph_mode", default="face_plus_context")
    parser.add_argument("--face_threshold", type=float, default=0.15)
    parser.add_argument("--context_pixels", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--verify_max_checks", type=int, default=64)
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--no_resume_existing", action="store_true")
    args = parser.parse_args()

    prior_dir = Path(args.prior_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [str(item) for item in args.splits]

    if not args.verify_only:
        split_meta = {}
        for split in splits:
            split_meta[split] = _precompute_split(
                prior_dir=prior_dir,
                output_dir=output_dir,
                split=split,
                graph_mode=str(args.graph_mode),
                face_threshold=float(args.face_threshold),
                context_pixels=int(args.context_pixels),
                chunk_size=int(args.chunk_size),
                max_samples=args.max_samples_per_split,
                resume_existing=not bool(args.no_resume_existing),
            )
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "prior_dir": str(prior_dir),
            "graph_mode": str(args.graph_mode),
            "face_threshold": float(args.face_threshold),
            "context_pixels": int(args.context_pixels),
            "chunk_size": int(args.chunk_size),
            "splits": split_meta,
        }
        _write_json(output_dir / "metadata.json", metadata)

    failures: List[str] = []
    for split in splits:
        failures.extend(
            _verify_split(
                prior_dir=prior_dir,
                cache_dir=output_dir,
                split=split,
                graph_mode=str(args.graph_mode),
                face_threshold=float(args.face_threshold),
                context_pixels=int(args.context_pixels),
                max_checks=int(args.verify_max_checks),
                max_samples=args.max_samples_per_split,
            )
        )
    summary = {
        "cache_dir": str(output_dir),
        "prior_dir": str(prior_dir),
        "splits": splits,
        "failure_count": len(failures),
        "failures": failures[:100],
        "decision": "D16_GRAPH_CACHE_VERIFY_PASS" if not failures else "D16_GRAPH_CACHE_VERIFY_FAIL",
    }
    _write_json(output_dir / "graph_cache_verify_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
