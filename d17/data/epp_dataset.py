"""D17 dataset backed by existing FER2013 prior npz containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from d17.data.epp_graph_builder import EPPGraphData, build_epp_graph
from d17.data.epp_graph_cache import graph_cache_path, load_epp_graph_cache


class EPPPixelDataset(Dataset):
    def __init__(
        self,
        prior_dir: str | Path,
        split: str,
        graph: Dict[str, Any] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.prior_dir = Path(prior_dir)
        self.split = str(split)
        self.split_dir = self.prior_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing D17 prior split dir: {self.split_dir}")
        self.files = sorted(self.split_dir.glob("*.npz"))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {self.split_dir}")
        self.graph_cfg = dict(graph or {})
        cache_cfg = dict((self.graph_cfg.get("cache") or {}))
        self.cache_enabled = bool(cache_cfg.get("enabled", False))
        self.cache_dir = Path(cache_cfg.get("dir")) if cache_cfg.get("dir") else None
        self.cache_strict = bool(cache_cfg.get("strict", True))
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_enabled:
            if self.cache_dir is None:
                raise ValueError("graph.cache.enabled=true but graph.cache.dir is empty")
            split_cache_dir = self.cache_dir / self.split
            if self.cache_strict and not split_cache_dir.exists():
                raise FileNotFoundError(f"Missing D17 graph cache split dir: {split_cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_prior(self, index: int) -> Dict[str, np.ndarray]:
        with np.load(self.files[int(index)], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def __getitem__(self, index: int) -> EPPGraphData:
        prior_file = self.files[int(index)]
        if self.cache_enabled and self.cache_dir is not None:
            cache_file = graph_cache_path(self.cache_dir, self.split, prior_file)
            if cache_file.exists():
                self.cache_hits += 1
                return load_epp_graph_cache(cache_file)
            self.cache_misses += 1
            if self.cache_strict:
                raise FileNotFoundError(f"Missing D17 graph cache file: {cache_file}")
        return build_epp_graph(self._load_prior(index), self.graph_cfg)
