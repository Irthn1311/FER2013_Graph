"""Chunked graph-cache dataset for D16.

The cache stores the exact tensors produced by ``build_pixel_graph`` so training
can skip repeated npz loading and graph construction across epochs.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from d16.data.graph_builder import D16GraphData


CACHE_SCHEMA_VERSION = "d16_graph_cache_v1"


def graph_to_cache_dict(graph: D16GraphData) -> Dict[str, torch.Tensor]:
    return {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "pos": graph.pos,
        "y": graph.y,
        "sample_index": graph.sample_index,
        "part_soft": graph.part_soft,
        "face_mask": graph.face_mask,
        "valid_part_mask": graph.valid_part_mask,
        "valid_anchor_mask": graph.valid_anchor_mask,
        "detected": graph.detected,
        "landmark_missing_flag": graph.landmark_missing_flag,
    }


def graph_from_cache_dict(row: Dict[str, torch.Tensor]) -> D16GraphData:
    return D16GraphData(
        x=row["x"],
        edge_index=row["edge_index"],
        pos=row["pos"],
        y=row["y"],
        sample_index=row["sample_index"],
        part_soft=row["part_soft"],
        face_mask=row["face_mask"],
        valid_part_mask=row["valid_part_mask"],
        valid_anchor_mask=row["valid_anchor_mask"],
        detected=row["detected"],
        landmark_missing_flag=row["landmark_missing_flag"],
    )


def compare_graphs(a: D16GraphData, b: D16GraphData) -> List[str]:
    failures: List[str] = []
    for name in (
        "x",
        "edge_index",
        "pos",
        "y",
        "sample_index",
        "part_soft",
        "face_mask",
        "valid_part_mask",
        "valid_anchor_mask",
        "detected",
        "landmark_missing_flag",
    ):
        left = getattr(a, name)
        right = getattr(b, name)
        if left.shape != right.shape:
            failures.append(f"{name}: shape {tuple(left.shape)} != {tuple(right.shape)}")
            continue
        if left.is_floating_point():
            diff = float((left - right).abs().max().item()) if left.numel() else 0.0
            if diff != 0.0:
                failures.append(f"{name}: max_abs_diff={diff}")
        elif not torch.equal(left, right):
            failures.append(f"{name}: tensor mismatch")
    return failures


class D16GraphCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        graph_mode: str = "face_plus_context",
        face_threshold: float = 0.15,
        context_pixels: int = 2,
        max_samples: int | None = None,
        chunk_cache_size: int = 2,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.chunk_cache_size = max(int(chunk_cache_size), 1)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing D16 graph cache metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata(graph_mode, face_threshold, context_pixels)
        split_meta = (self.metadata.get("splits") or {}).get(self.split)
        if not split_meta:
            raise FileNotFoundError(f"Missing split={self.split!r} in graph cache: {metadata_path}")
        self.index: List[tuple[Path, int]] = []
        for chunk in split_meta.get("chunks", []):
            path = self.cache_dir / chunk["path"]
            count = int(chunk["count"])
            for offset in range(count):
                self.index.append((path, offset))
        if max_samples is not None:
            self.index = self.index[: int(max_samples)]
        if not self.index:
            raise FileNotFoundError(f"No cached D16 graphs for split={self.split!r} in {self.cache_dir}")
        self._chunk_cache: OrderedDict[Path, List[Dict[str, torch.Tensor]]] = OrderedDict()

    def _validate_metadata(self, graph_mode: str, face_threshold: float, context_pixels: int) -> None:
        if self.metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported D16 graph cache schema: {self.metadata.get('schema_version')!r}")
        expected = {
            "graph_mode": str(graph_mode),
            "face_threshold": float(face_threshold),
            "context_pixels": int(context_pixels),
        }
        for key, value in expected.items():
            actual = self.metadata.get(key)
            if actual != value:
                raise ValueError(f"D16 graph cache metadata mismatch for {key}: {actual!r} != {value!r}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_chunk(self, path: Path) -> List[Dict[str, torch.Tensor]]:
        cached = self._chunk_cache.get(path)
        if cached is not None:
            self._chunk_cache.move_to_end(path)
            return cached
        rows = torch.load(path, map_location="cpu", weights_only=False)
        self._chunk_cache[path] = rows
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> D16GraphData:
        path, offset = self.index[int(index)]
        rows = self._load_chunk(path)
        return graph_from_cache_dict(rows[offset])
