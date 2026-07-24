"""Lazy, deterministic dataset backed by verified MediaPipe pixel priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from lap_gnn_tf.graph.builder import D16GraphData, build_pixel_graph
from lap_gnn_tf.priors.mediapipe_priors import fallback_priors


class PixelPriorDataset:
    def __init__(
        self,
        prior_dir: str | Path,
        split: str = "train",
        graph_mode: str = "face_plus_context",
        face_threshold: float = 0.15,
        context_pixels: int = 2,
        detail_features: Dict[str, Any] | None = None,
        edge_features: Dict[str, Any] | None = None,
        anchor_nodes: Dict[str, Any] | None = None,
        node_features: Dict[str, Any] | None = None,
        knn_edges: Dict[str, Any] | None = None,
        prior_usage: str | None = None,
        prior_corruption: Dict[str, Any] | None = None,
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
        self.anchor_nodes = dict(anchor_nodes or {})
        self.node_features = dict(node_features or {})
        self.knn_edges = dict(knn_edges or {})
        self.prior_usage = None if prior_usage is None else str(prior_usage)
        self.prior_corruption = dict(prior_corruption or {})
        self.epoch = 0
        self.part_names = self._read_json("part_names.json")
        self.micro_anchor_names = self._read_json("micro_anchor_names.json")

    def _read_json(self, name: str) -> Any:
        path = self.prior_dir / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def raw_detected(self, index: int) -> bool:
        with np.load(self.files[int(index)], allow_pickle=False) as data:
            return bool(np.asarray(data["detected"]).item())

    def current_corruption_probability(self) -> float:
        return self._current_probability() if self._corruption_enabled() else 0.0

    def current_edge_prior_regularization_probability(self) -> float:
        cfg = self._edge_prior_regularization_cfg()
        return self._scheduled_probability(cfg) if self._edge_prior_regularization_enabled(cfg) else 0.0

    def _edge_prior_regularization_cfg(self) -> Dict[str, Any]:
        return dict((self.edge_features.get("prior_regularization", {}) or {}))

    def _edge_prior_regularization_enabled(self, cfg: Dict[str, Any]) -> bool:
        if not bool(cfg.get("enabled", False)):
            return False
        if bool(cfg.get("train_only", True)) and self.split != "train":
            return False
        return True

    def _scheduled_probability(self, cfg: Dict[str, Any]) -> float:
        probability = float(cfg.get("probability", 0.0) or 0.0)
        for item in cfg.get("schedule") or []:
            if int(self.epoch) >= int(item.get("start_epoch", 1) or 1):
                probability = float(item.get("probability", probability) or 0.0)
        return float(np.clip(probability, 0.0, 1.0))

    def _edge_features_for(self, index: int) -> Dict[str, Any]:
        edge_features = dict(self.edge_features or {})
        cfg = self._edge_prior_regularization_cfg()
        if not self._edge_prior_regularization_enabled(cfg):
            edge_features.pop("prior_regularization", None)
            return edge_features
        cfg["current_epoch"] = int(self.epoch)
        cfg["probability"] = self._scheduled_probability(cfg)
        seed = int(cfg.get("seed", 2719) or 2719)
        cfg["rng_seed"] = int((seed + int(self.epoch) * 1_000_003 + int(index) * 97_531) % (2**32 - 1))
        edge_features["prior_regularization"] = cfg
        return edge_features

    def _knn_edges_for(self, corruption_mode: str | None) -> Dict[str, Any]:
        knn_edges = dict(self.knn_edges or {})
        if knn_edges:
            knn_edges["cache_split"] = self.split
        if corruption_mode is not None:
            knn_edges["cache_enabled"] = False
        return knn_edges

    def _corruption_enabled(self) -> bool:
        cfg = self.prior_corruption
        if not bool(cfg.get("enabled", False)):
            return False
        if bool(cfg.get("train_only", True)) and self.split != "train":
            return False
        return True

    def _current_probability(self) -> float:
        cfg = self.prior_corruption
        probability = float(cfg.get("probability", 0.0) or 0.0)
        for item in cfg.get("schedule") or []:
            if int(self.epoch) >= int(item.get("start_epoch", 1) or 1):
                probability = float(item.get("probability", probability) or 0.0)
        return float(np.clip(probability, 0.0, 1.0))

    def _rng_for(self, index: int) -> np.random.Generator:
        seed = int(self.prior_corruption.get("seed", 137) or 137)
        mixed = (seed + int(self.epoch) * 1_000_003 + int(index) * 97_531) % (2**32 - 1)
        return np.random.default_rng(mixed)

    @staticmethod
    def _copy_prior(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {key: np.array(value, copy=True) for key, value in prior.items()}

    @staticmethod
    def _zero_prior(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = PixelPriorDataset._copy_prior(prior)
        for key in ("face_mask", "part_soft_masks", "micro_anchor_maps"):
            if key in out:
                out[key] = np.zeros_like(out[key], dtype=np.float32)
        if "distance_maps" in out:
            out["distance_maps"] = np.ones_like(out["distance_maps"], dtype=np.float32)
        if "landmark_xy_48" in out:
            out["landmark_xy_48"] = np.zeros((0, 2), dtype=np.float32)
        for key in ("valid_part_mask", "valid_anchor_mask"):
            if key in out:
                out[key] = np.zeros_like(out[key], dtype=np.float32)
        if "quality_score" in out:
            out["quality_score"] = np.asarray(0.0, dtype=np.float32)
        return out

    @staticmethod
    def _attenuate_prior(prior: Dict[str, np.ndarray], strength: float) -> Dict[str, np.ndarray]:
        out = PixelPriorDataset._copy_prior(prior)
        keep = float(np.clip(1.0 - float(strength), 0.0, 1.0))
        for key in ("face_mask", "part_soft_masks", "micro_anchor_maps"):
            if key in out:
                out[key] = (np.asarray(out[key], dtype=np.float32) * keep).astype(np.float32)
        if "distance_maps" in out:
            dist = np.asarray(out["distance_maps"], dtype=np.float32)
            out["distance_maps"] = np.clip(dist * keep + (1.0 - keep), 0.0, 1.0).astype(np.float32)
        if "quality_score" in out:
            out["quality_score"] = np.asarray(float(np.asarray(out["quality_score"]).item()) * keep, dtype=np.float32)
        return out

    @staticmethod
    def _shuffle_prior(base: Dict[str, np.ndarray], donor: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = PixelPriorDataset._copy_prior(base)
        for key in (
            "face_mask",
            "part_soft_masks",
            "micro_anchor_maps",
            "distance_maps",
            "landmark_xy_48",
            "valid_part_mask",
            "valid_anchor_mask",
            "quality_score",
        ):
            if key in donor:
                out[key] = np.array(donor[key], copy=True)
        return out

    @staticmethod
    def _forced_fallback(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        label = int(np.asarray(prior["label"]).item())
        sample_index = int(np.asarray(prior["sample_index"]).item())
        result = fallback_priors(np.asarray(prior["image_48"], dtype=np.float32), label=label, sample_index=sample_index)
        return PixelPriorDataset._copy_prior(result.arrays)

    def _load_prior(self, index: int) -> Dict[str, np.ndarray]:
        with np.load(self.files[int(index)], allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def _sample_corruption_mode(self, rng: np.random.Generator) -> str:
        weights = dict(self.prior_corruption.get("mode_weights") or {})
        if not weights:
            modes = list(self.prior_corruption.get("modes") or ["attenuate_prior"])
            return str(modes[int(rng.integers(0, len(modes)))])
        names = [str(name) for name, value in weights.items() if float(value) > 0.0]
        probs = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def _maybe_corrupt_prior(self, prior: Dict[str, np.ndarray], index: int) -> tuple[Dict[str, np.ndarray], str | None]:
        if not self._corruption_enabled():
            return prior, None
        rng = self._rng_for(index)
        if float(rng.random()) >= self._current_probability():
            return prior, None
        mode = self._sample_corruption_mode(rng)
        if mode == "zero_prior":
            return self._zero_prior(prior), mode
        if mode == "attenuate_prior":
            lo, hi = self.prior_corruption.get("attenuate_strength", [0.35, 0.75])
            return self._attenuate_prior(prior, strength=float(rng.uniform(float(lo), float(hi)))), mode
        if mode == "shuffle_prior":
            if len(self.files) <= 1:
                return prior, None
            donor_idx = int(rng.integers(0, len(self.files) - 1))
            if donor_idx >= int(index):
                donor_idx += 1
            return self._shuffle_prior(prior, self._load_prior(donor_idx)), mode
        if mode == "forced_fallback":
            return self._forced_fallback(prior), mode
        raise ValueError(f"Unsupported D16 prior_corruption mode={mode!r}")

    def __getitem__(self, index: int) -> D16GraphData:
        prior = self._load_prior(int(index))
        prior, corruption_mode = self._maybe_corrupt_prior(prior, int(index))
        graph_mode = "full_with_mask" if corruption_mode == "forced_fallback" else self.graph_mode
        return build_pixel_graph(
            prior,
            graph_mode=graph_mode,
            face_threshold=self.face_threshold,
            context_pixels=self.context_pixels,
            detail_features=self.detail_features,
            edge_features=self._edge_features_for(int(index)),
            anchor_nodes=self.anchor_nodes,
            node_features=self.node_features,
            knn_edges=self._knn_edges_for(corruption_mode),
            prior_usage=self.prior_usage,
        )
