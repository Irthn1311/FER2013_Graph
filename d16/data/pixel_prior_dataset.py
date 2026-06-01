"""Dataset backed by precomputed D16 MediaPipe pixel priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

from d16.data.graph_builder import D16GraphData, build_pixel_graph


class D16PixelPriorDataset(Dataset):
    def __init__(
        self,
        prior_dir: str | Path,
        split: str = "train",
        graph_mode: str = "face_plus_context",
        face_threshold: float = 0.15,
        context_pixels: int = 2,
        detail_features: Dict[str, Any] | None = None,
        edge_features: Dict[str, Any] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.prior_dir = Path(prior_dir)
        self.split = str(split)
        self.split_dir = self.prior_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing D16 prior split dir: {self.split_dir}")
        self.files = sorted(self.split_dir.glob("*.npz"))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise FileNotFoundError(f"No D16 prior npz files found in {self.split_dir}")
        self.graph_mode = graph_mode
        self.face_threshold = float(face_threshold)
        self.context_pixels = int(context_pixels)
        self.detail_features = dict(detail_features or {})
        self.edge_features = dict(edge_features or {})
        self.part_names = self._read_json("part_names.json")
        self.micro_anchor_names = self._read_json("micro_anchor_names.json")

    def _read_json(self, name: str) -> Any:
        path = self.prior_dir / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> D16GraphData:
        path = self.files[int(index)]
        with np.load(path, allow_pickle=False) as data:
            prior: Dict[str, np.ndarray] = {key: data[key] for key in data.files}
        return build_pixel_graph(
            prior,
            graph_mode=self.graph_mode,
            face_threshold=self.face_threshold,
            context_pixels=self.context_pixels,
            detail_features=self.detail_features,
            edge_features=self.edge_features,
        )
