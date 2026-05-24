"""Repack a D16 graph cache into smaller chunks without rebuilding graphs."""

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

from d16.data.graph_cache_dataset import CACHE_SCHEMA_VERSION, resolve_cache_path  # noqa: E402


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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


def _load_chunk(cache_dir: Path, relative_path: str) -> List[Dict[str, torch.Tensor]]:
    path = resolve_cache_path(cache_dir, relative_path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _flush_rows(output_dir: Path, split: str, rows: List[Dict[str, torch.Tensor]], start: int, chunk_index: int) -> Dict[str, Any]:
    chunk_path = output_dir / split / f"chunk_{chunk_index:05d}.pt"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    _save_chunk_atomic(rows, chunk_path)
    return {"path": str(chunk_path.relative_to(output_dir).as_posix()), "start": int(start), "count": int(len(rows))}


def _rechunk_split(input_dir: Path, output_dir: Path, split: str, split_meta: Dict[str, Any], chunk_size: int) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    pending: List[Dict[str, torch.Tensor]] = []
    pending_start = 0
    written = 0
    chunk_index = 0
    for old_chunk in split_meta.get("chunks", []):
        rows = _load_chunk(input_dir, str(old_chunk["path"]))
        for row in rows:
            if not pending:
                pending_start = written
            pending.append(row)
            written += 1
            if len(pending) >= chunk_size:
                chunks.append(_flush_rows(output_dir, split, pending, pending_start, chunk_index))
                chunk_index += 1
                pending = []
    if pending:
        chunks.append(_flush_rows(output_dir, split, pending, pending_start, chunk_index))
    expected = int(split_meta.get("count", written))
    if written != expected:
        raise RuntimeError(f"{split}: expected {expected} rows, wrote {written}")
    return {"count": written, "chunks": chunks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {metadata.get('schema_version')!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_meta: Dict[str, Any] = {}
    for split in [str(item) for item in args.splits]:
        split_meta[split] = _rechunk_split(input_dir, output_dir, split, metadata["splits"][split], int(args.chunk_size))
    new_metadata = dict(metadata)
    new_metadata["chunk_size"] = int(args.chunk_size)
    new_metadata["source_cache_dir"] = str(input_dir)
    new_metadata["splits"] = split_meta
    _write_json(output_dir / "metadata.json", new_metadata)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chunk_size": int(args.chunk_size),
        "splits": {split: {"count": meta["count"], "num_chunks": len(meta["chunks"])} for split, meta in split_meta.items()},
        "decision": "D16_GRAPH_CACHE_RECHUNK_DONE",
    }
    _write_json(output_dir / "graph_cache_rechunk_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
