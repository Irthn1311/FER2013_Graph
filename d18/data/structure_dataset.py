"""D18 dataset backed by existing FER2013 prior npz containers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict
import warnings

import numpy as np
from torch.utils.data import Dataset

from d18.data.structure_graph_builder import D18GraphData, build_structure_graph
from d16.data.mediapipe_priors import pixels_to_image48
from d18.data.structure_graph_cache import (
    evidence_cache_signature,
    evidence_graph_cache_path,
    graph_cache_path,
    load_d18_graph_cache,
)


class StructurePixelDataset(Dataset):
    def __init__(
        self,
        prior_dir: str | Path,
        split: str,
        graph: Dict[str, Any] | None = None,
        max_samples: int | None = None,
        evidence_dir: str | Path | None = None,
    ) -> None:
        self.split = str(split)
        self.graph_cfg = dict(graph or {})
        self.graph_mode = str(self.graph_cfg.get("graph_mode", "structure_guided"))
        self.evidence_only = self.graph_mode == "evidence_only"
        self.prior_dir = Path(prior_dir) if prior_dir else None
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.evidence_rows: list[dict[str, str]] = []
        if self.evidence_only:
            if self.evidence_dir is None:
                raise ValueError("graph_mode=evidence_only requires data.evidence_dir")
            csv_path = self.evidence_dir / f"{self.split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing FER evidence CSV: {csv_path}")
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                self.evidence_rows = list(csv.DictReader(handle))
            self.files = [Path(f"{index:06d}.npz") for index in range(len(self.evidence_rows))]
            self.split_dir = csv_path.parent
        else:
            if self.prior_dir is None:
                raise ValueError("structure-guided D18 dataset requires data.prior_dir")
            self.split_dir = self.prior_dir / self.split
            if not self.split_dir.exists():
                raise FileNotFoundError(f"Missing D18 prior split dir: {self.split_dir}")
            self.files = sorted(self.split_dir.glob("*.npz"))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {self.split_dir}")
        cache_cfg = dict((self.graph_cfg.get("cache") or {}))
        self.cache_enabled = bool(cache_cfg.get("enabled", False))
        self.cache_dir = Path(cache_cfg.get("dir")) if cache_cfg.get("dir") else None
        self.cache_strict = bool(cache_cfg.get("strict", True))
        self.cache_fallback_on_error = bool(cache_cfg.get("fallback_on_error", True))
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        self.cache_namespace = evidence_cache_signature(self.graph_cfg)[:16] if self.evidence_only else None
        if self.cache_enabled:
            if self.cache_dir is None:
                raise ValueError("graph.cache.enabled=true but graph.cache.dir is empty")
            split_cache_dir = (
                self.cache_dir / self.cache_namespace / self.split
                if self.evidence_only
                else self.cache_dir / self.split
            )
            if self.cache_strict and not split_cache_dir.exists():
                raise FileNotFoundError(f"Missing D18 graph cache split dir: {split_cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_prior(self, index: int) -> Dict[str, np.ndarray]:
        if self.evidence_only:
            raise RuntimeError("Evidence-only dataset must not load prior NPZ assets")
        with np.load(self.files[int(index)], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def _load_evidence(self, index: int) -> Dict[str, np.ndarray]:
        row = self.evidence_rows[int(index)]
        return {
            "image_48": pixels_to_image48(row["pixels"]),
            "label": np.asarray(int(row["emotion"]), dtype=np.int64),
            "sample_index": np.asarray(int(index), dtype=np.int64),
        }

    def __getitem__(self, index: int) -> D18GraphData:
        prior_file = self.files[int(index)]
        evidence = self._load_evidence(index) if self.evidence_only else None
        if self.cache_enabled and self.cache_dir is not None:
            cache_file = (
                evidence_graph_cache_path(
                    self.cache_dir,
                    self.split,
                    int(index),
                    evidence["image_48"],
                    int(evidence["label"]),
                    self.graph_cfg,
                )
                if self.evidence_only
                else graph_cache_path(self.cache_dir, self.split, prior_file)
            )
            if cache_file.exists():
                self.cache_hits += 1
                try:
                    return load_d18_graph_cache(cache_file)
                except Exception as exc:
                    self.cache_errors += 1
                    if not self.cache_fallback_on_error:
                        raise
                    warnings.warn(
                        f"Failed to load D18 graph cache {cache_file}; rebuilding graph online for this sample. "
                        f"error={type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    source = evidence if self.evidence_only else self._load_prior(index)
                    return build_structure_graph(source, self.graph_cfg)
            self.cache_misses += 1
            if self.cache_strict:
                raise FileNotFoundError(f"Missing D18 graph cache file: {cache_file}")
        source = evidence if self.evidence_only else self._load_prior(index)
        return build_structure_graph(source, self.graph_cfg)
