"""Chunked graph repository for D5 pixel graphs."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from data.graph_config import GraphConfig
from data.graph_resolver import GraphResolver
from data.graph_types import PixelGraphSample, ResolvedPixelGraph, SharedGraphStructure

MANIFEST_FILENAME = "manifest.pt"
SHARED_DIR = "shared"
SHARED_FILENAME = "shared_graph.pt"
CHUNK_PATTERN = "chunk_{idx:03d}.pt"


def torch_load(path: str | Path):
    """Load torch files containing dataclasses across PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GraphRepositoryWriter:
    """Write shared graph structure and per-split graph chunks."""

    def __init__(
        self,
        repo_root: str | Path,
        config: GraphConfig,
        overwrite: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.config = config
        self.overwrite = bool(overwrite)
        self.repo_root.mkdir(parents=True, exist_ok=True)
        (self.repo_root / SHARED_DIR).mkdir(parents=True, exist_ok=True)
        self.manifest: Dict = {
            "version": config.version,
            "chunk_size": int(config.chunk_size),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "config": config.to_dict(),
            "node_feature_names": list(config.node_feature_names),
            "edge_feature_names": list(config.edge_static_feature_names)
            + list(config.edge_dynamic_feature_names),
            "node_dim": int(len(config.node_feature_names)),
            "edge_dim": int(len(config.edge_static_feature_names) + len(config.edge_dynamic_feature_names)),
            "height": int(config.height),
            "width": int(config.width),
            "splits": {},
        }

    def write_shared(self, shared: SharedGraphStructure) -> Path:
        path = self.repo_root / SHARED_DIR / SHARED_FILENAME
        torch.save(shared, path)
        self.manifest["shared"] = str(Path(SHARED_DIR) / SHARED_FILENAME)
        return path

    def open_split(self, split: str) -> "_SplitWriter":
        split_dir = self.repo_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            for old in split_dir.glob("chunk_*.pt"):
                old.unlink()
        return _SplitWriter(
            repo_root=self.repo_root,
            split=split,
            split_dir=split_dir,
            chunk_size=int(self.config.chunk_size),
            manifest=self.manifest,
        )

    def save_manifest(self) -> Path:
        path = self.repo_root / MANIFEST_FILENAME
        torch.save(self.manifest, path)
        return path


class _SplitWriter(AbstractContextManager):
    def __init__(
        self,
        repo_root: Path,
        split: str,
        split_dir: Path,
        chunk_size: int,
        manifest: Dict,
    ) -> None:
        self.repo_root = repo_root
        self.split = split
        self.split_dir = split_dir
        self.chunk_size = int(chunk_size)
        self.manifest = manifest
        self._buf: List[PixelGraphSample] = []
        self._chunk_idx = 0
        self._total = 0
        self._chunk_files: List[str] = []
        self._chunk_counts: List[int] = []

    def add(self, sample: PixelGraphSample) -> None:
        self._buf.append(sample)
        self._total += 1
        if len(self._buf) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        name = CHUNK_PATTERN.format(idx=self._chunk_idx)
        path = self.split_dir / name
        torch.save(self._buf, path)
        rel = path.relative_to(self.repo_root)
        self._chunk_files.append(str(rel).replace("\\", "/"))
        self._chunk_counts.append(len(self._buf))
        self._chunk_idx += 1
        self._buf = []

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._flush()
            self.manifest["splits"][self.split] = {
                "num_samples": int(self._total),
                "num_chunks": int(len(self._chunk_files)),
                "chunk_files": list(self._chunk_files),
                "chunk_counts": list(self._chunk_counts),
            }
        return False


class GraphRepositoryReader:
    """Read a D5 chunked graph repository."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Graph repository not found: {self.repo_root}")
        self._manifest: Optional[Dict] = None
        self._shared: Optional[SharedGraphStructure] = None

    @property
    def manifest(self) -> Dict:
        if self._manifest is None:
            path = self.repo_root / MANIFEST_FILENAME
            if path.exists():
                self._manifest = torch_load(path)
            else:
                self._manifest = {"splits": {}}
        return self._manifest

    def load_shared(self) -> SharedGraphStructure:
        if self._shared is None:
            path = self.repo_root / SHARED_DIR / SHARED_FILENAME
            if not path.exists():
                raise FileNotFoundError(f"Shared graph not found: {path}")
            self._shared = torch_load(path)
        return self._shared

    def chunk_paths(self, split: str) -> List[Path]:
        split_info = self.manifest.get("splits", {}).get(split, {})
        rel_files = split_info.get("chunk_files") or []
        paths = [self.repo_root / rel for rel in rel_files]
        if not paths:
            paths = sorted((self.repo_root / split).glob("chunk_*.pt"))
        return [p for p in paths if p.exists()]

    def load_chunk(self, split: str, chunk_idx: int) -> List[PixelGraphSample]:
        paths = self.chunk_paths(split)
        if chunk_idx < 0 or chunk_idx >= len(paths):
            raise IndexError(f"Chunk index out of range for split={split}: {chunk_idx}")
        return torch_load(paths[chunk_idx])

    def iter_split(self, split: str, start_chunk: int = 0) -> Iterator[PixelGraphSample]:
        for path in self.chunk_paths(split)[int(start_chunk) :]:
            chunk = torch_load(path)
            yield from chunk

    def split_size(self, split: str) -> int:
        info = self.manifest.get("splits", {}).get(split, {})
        if "num_samples" in info:
            return int(info["num_samples"])
        return sum(len(torch_load(path)) for path in self.chunk_paths(split))


class ChunkedGraphDataset(Dataset):
    """Lazy dataset over graph repository chunks.

    If ``resolve`` is true, each item is returned as a ``ResolvedPixelGraph``.
    Otherwise it is returned as the stored ``PixelGraphSample``.
    """

    def __init__(
        self,
        repo_root: str | Path,
        split: str,
        resolve: bool = False,
        chunk_cache_size: int = 0,
        graph_cache_chunks: Optional[int] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.split = split
        self.resolve = bool(resolve)
        self.reader = GraphRepositoryReader(repo_root)
        self.chunk_paths = self.reader.chunk_paths(split)
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunks found for split={split} in {self.repo_root}")
        self.shared = self.reader.load_shared()
        self.resolver = GraphResolver(self.shared)
        if graph_cache_chunks is not None and int(chunk_cache_size or 0) <= 0:
            chunk_cache_size = graph_cache_chunks
        self.chunk_cache_size = max(0, int(chunk_cache_size or 0))
        self._cache: OrderedDict[int, List[PixelGraphSample]] = OrderedDict()
        self._index = self._build_index()

    def _build_index(self) -> List[tuple[int, int]]:
        info = self.reader.manifest.get("splits", {}).get(self.split, {})
        counts = info.get("chunk_counts")
        if not counts or len(counts) != len(self.chunk_paths):
            counts = [len(torch_load(path)) for path in self.chunk_paths]
        index: List[tuple[int, int]] = []
        for chunk_idx, count in enumerate(counts):
            for offset in range(int(count)):
                index.append((chunk_idx, offset))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def chunk_index_groups(self) -> List[List[int]]:
        groups: List[List[int]] = [[] for _ in self.chunk_paths]
        for sample_idx, (chunk_idx, _) in enumerate(self._index):
            groups[int(chunk_idx)].append(sample_idx)
        return groups

    def label_at_index(self, idx: int) -> int:
        chunk_idx, offset = self._index[int(idx)]
        return int(self._get_chunk(chunk_idx)[offset].label)

    def _get_chunk(self, chunk_idx: int) -> List[PixelGraphSample]:
        if self.chunk_cache_size <= 0:
            return torch_load(self.chunk_paths[chunk_idx])
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]
        chunk = torch_load(self.chunk_paths[chunk_idx])
        self._cache[chunk_idx] = chunk
        while len(self._cache) > self.chunk_cache_size:
            self._cache.popitem(last=False)
        return chunk

    def __getitem__(self, idx: int) -> PixelGraphSample | ResolvedPixelGraph:
        chunk_idx, offset = self._index[int(idx)]
        sample = self._get_chunk(chunk_idx)[offset]
        if self.resolve:
            return self.resolver.resolve(sample)
        return sample
