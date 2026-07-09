"""D17 dataset backed by existing FER2013 prior npz containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from d17.data.epp_graph_builder import EPPGraphData, build_epp_graph


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

    def __len__(self) -> int:
        return len(self.files)

    def _load_prior(self, index: int) -> Dict[str, np.ndarray]:
        with np.load(self.files[int(index)], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def __getitem__(self, index: int) -> EPPGraphData:
        return build_epp_graph(self._load_prior(index), self.graph_cfg)

