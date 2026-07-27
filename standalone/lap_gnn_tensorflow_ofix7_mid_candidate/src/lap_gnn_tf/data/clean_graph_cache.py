"""Sharded, framework-neutral cache for clean D16 graph objects.

The cache stores the exact NumPy graph arrays produced by ``build_pixel_graph``.
It is deliberately separate from the train-time prior corruption path: callers
may use a clean graph only when the current sample is not corrupted.
"""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from lap_gnn_tf.graph.builder import D16GraphData


CACHE_SCHEMA_VERSION = "tf_clean_graph_cache_v2_records"


def clean_graph_config_payload(graph_config: dict[str, Any]) -> dict[str, Any]:
    """Return only graph settings that affect a clean graph's arrays."""
    graph = json.loads(json.dumps(graph_config))
    graph.pop("prior_corruption", None)
    edge_features = graph.get("edge_features")
    if isinstance(edge_features, dict):
        edge_features.pop("prior_regularization", None)
    return graph


def clean_graph_config_sha256(graph_config: dict[str, Any]) -> str:
    payload = json.dumps(
        clean_graph_config_payload(graph_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_record(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


class CleanGraphCache:
    """Read one split of a sharded clean graph cache with an LRU shard cache."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        expected_graph_config_sha256: str | None = None,
        max_loaded_shards: int = 2,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        complete = self.root / "CACHE_COMPLETE.json"
        index_path = self.split_root / "index.json"
        if not complete.is_file():
            raise FileNotFoundError(f"Missing clean graph cache completion marker: {complete}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing clean graph cache index: {index_path}")
        self.complete = json.loads(complete.read_text(encoding="utf-8"))
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.complete.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported clean graph cache schema: {self.complete.get('schema_version')!r}"
            )
        if self.index.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("Clean graph cache split index schema mismatch")
        actual = self.complete.get("graph_config_sha256")
        if expected_graph_config_sha256 is not None and actual != expected_graph_config_sha256:
            raise ValueError(
                "Clean graph cache graph configuration mismatch: "
                f"{actual!r} != {expected_graph_config_sha256!r}"
            )
        self.node_dim = int(self.complete.get("node_dim", 0))
        self.edge_dim = int(self.complete.get("edge_dim", 0))
        self.sample_count = int(self.index.get("sample_count", 0))
        self.shards = list(self.index.get("shards") or [])
        self.max_loaded_shards = max(int(max_loaded_shards), 1)
        self._thread_handles = threading.local()

    def __len__(self) -> int:
        return self.sample_count

    def _shard_for(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        for shard in self.shards:
            if int(shard["start"]) <= index < int(shard["end"]):
                return shard
        raise IndexError(f"No clean graph cache shard for index={index}")

    def _get_handle(self, shard: dict[str, Any]):
        relative = str(shard["path"])
        handles = getattr(self._thread_handles, "handles", None)
        if handles is None:
            handles = OrderedDict()
            self._thread_handles.handles = handles
        if relative in handles:
            handle = handles.pop(relative)
            handles[relative] = handle
            return handle
        path = self.split_root / relative
        handle = path.open("rb")
        handles[relative] = handle
        while len(handles) > self.max_loaded_shards:
            _, old_handle = handles.popitem(last=False)
            old_handle.close()
        return handle

    def __getitem__(self, index: int) -> D16GraphData:
        index = int(index)
        shard_meta = self._shard_for(index)
        row = index - int(shard_meta["start"])
        offsets = shard_meta.get("offsets") or []
        if len(offsets) != int(shard_meta["samples"]) + 1:
            raise ValueError(f"Invalid clean graph cache offsets in {shard_meta['path']}")
        start = int(offsets[row])
        end = int(offsets[row + 1])
        handle = self._get_handle(shard_meta)
        handle.seek(start)
        payload = handle.read(end - start)
        if len(payload) != end - start:
            raise IOError(f"Short clean graph cache read from {shard_meta['path']}")
        data = _load_record(payload)
        node_feature_names = list(self.complete.get("node_feature_names") or [])
        edge_feature_names = list(self.complete.get("edge_feature_names") or [])
        return D16GraphData(
            x=np.asarray(data["x"], dtype=np.float32),
            edge_index=np.asarray(data["edge_index"], dtype=np.int64),
            edge_attr=np.asarray(data["edge_attr"], dtype=np.float32),
            pos=np.asarray(data["pos"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.int64),
            sample_index=np.asarray(data["sample_index"], dtype=np.int64),
            part_soft=np.asarray(data["part_soft"], dtype=np.float32),
            face_mask=np.asarray(data["face_mask"], dtype=np.float32),
            valid_part_mask=np.asarray(data["valid_part_mask"], dtype=np.float32),
            valid_anchor_mask=np.asarray(data["valid_anchor_mask"], dtype=np.float32),
            detected=np.asarray(data["detected"], dtype=np.bool_),
            landmark_missing_flag=np.asarray(data["landmark_missing_flag"], dtype=np.int64),
            image_48=np.asarray(data["image_48"], dtype=np.float32),
            anchor_mask=np.asarray(data["anchor_mask"], dtype=np.bool_),
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "root": str(self.root),
            "split": self.split,
            "sample_count": self.sample_count,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "record_access": "indexed_seek",
        }
