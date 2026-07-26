"""Correctness-first lazy graph batch pipeline."""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import tensorflow as tf

from lap_gnn_tf.graph.builder import collate_d16_graphs
from lap_gnn_tf.graph.batch import to_tensor_dict
from lap_gnn_tf.priors.loader import PixelPriorDataset


class GraphBatchGenerator:
    def __init__(
        self,
        prior_root: str | Path,
        split: str,
        config: dict,
        batch_size: int = 16,
        seed: int = 42,
        shuffle: bool = False,
        graph_cache_size: int = 64,
        telemetry=None,
        graph_workers: int = 1,
    ):
        graph = config["graph"]
        self.dataset = PixelPriorDataset(
            prior_root,
            split=split,
            graph_mode=graph["graph_mode"],
            face_threshold=graph["face_threshold"],
            context_pixels=graph["context_pixels"],
            detail_features=graph.get("detail_features"),
            edge_features=graph.get("edge_features"),
            anchor_nodes=graph.get("anchor_nodes"),
            node_features=graph.get("node_features"),
            knn_edges=graph.get("knn_edges"),
            prior_usage=graph.get("prior_usage"),
            prior_corruption=graph.get("prior_corruption") if split == "train" else None,
        )
        self.split = split
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.cache_size = max(int(graph_cache_size), 0)
        self.cache: OrderedDict[tuple[int, int], object] = OrderedDict()
        self.telemetry = telemetry
        self.graph_workers = max(int(graph_workers), 1)
        self._cache_lock = threading.Lock()

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _order(self, epoch: int) -> np.ndarray:
        order = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
            rng.shuffle(order)
        return order

    def _graph(self, index: int, epoch: int):
        key = (int(epoch), int(index)) if self.split == "train" else (0, int(index))
        with self._cache_lock:
            if key in self.cache:
                graph = self.cache.pop(key)
                self.cache[key] = graph
                if self.telemetry is not None:
                    self.telemetry.graph_cache_hits += 1
                return graph
        if self.telemetry is not None:
            self.telemetry.graph_cache_misses += 1
        graph = self.dataset[int(index)]
        if self.cache_size:
            with self._cache_lock:
                self.cache[key] = graph
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return graph

    def iter_epoch(self, epoch: int, limit_batches: int | None = None):
        self.dataset.set_epoch(epoch)
        order = self._order(epoch)
        executor = (
            ThreadPoolExecutor(max_workers=self.graph_workers)
            if self.graph_workers > 1
            else None
        )
        try:
            for start in range(0, len(order), self.batch_size):
                batch_number = start // self.batch_size
                if limit_batches is not None and batch_number >= int(limit_batches):
                    break
                started = time.perf_counter()
                indices = [
                    int(index) for index in order[start : start + self.batch_size]
                ]
                if executor is None:
                    graphs = [self._graph(index, epoch) for index in indices]
                else:
                    graphs = list(
                        executor.map(
                            lambda index: self._graph(index, epoch),
                            indices,
                        )
                    )
                batch = to_tensor_dict(collate_d16_graphs(graphs))
                if self.telemetry is not None:
                    self.telemetry.batch_construction_sec.append(
                        time.perf_counter() - started
                    )
                    self.telemetry.sample()
                yield batch
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    @staticmethod
    def output_signature() -> dict[str, tf.TensorSpec]:
        return {
            "node_features": tf.TensorSpec((None, 37), tf.float32),
            "edge_index": tf.TensorSpec((2, None), tf.int64),
            "edge_features": tf.TensorSpec((None, 8), tf.float32),
            "node_types": tf.TensorSpec((None,), tf.int8),
            "node_graph_index": tf.TensorSpec((None,), tf.int64),
            "edge_graph_index": tf.TensorSpec((None,), tf.int64),
            "graph_node_counts": tf.TensorSpec((None,), tf.int64),
            "graph_edge_counts": tf.TensorSpec((None,), tf.int64),
            "labels": tf.TensorSpec((None,), tf.int64),
            "sample_ids": tf.TensorSpec((None,), tf.int64),
            "coordinates": tf.TensorSpec((None, 2), tf.float32),
            "anchor_mask": tf.TensorSpec((None,), tf.bool),
            "part_soft": tf.TensorSpec((None, 13), tf.float32),
            "face_mask": tf.TensorSpec((None,), tf.float32),
            "valid_part_mask": tf.TensorSpec((None, 13), tf.float32),
            "valid_anchor_mask": tf.TensorSpec((None, 12), tf.float32),
            "detected": tf.TensorSpec((None,), tf.bool),
            "landmark_missing_flag": tf.TensorSpec((None,), tf.int64),
            "image_48": tf.TensorSpec((None, 48, 48), tf.float32),
        }

    def as_dataset(
        self,
        epoch: int,
        limit_batches: int | None = None,
        prefetch: int = 0,
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            lambda: self.iter_epoch(epoch, limit_batches=limit_batches),
            output_signature=self.output_signature(),
        )
        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options)
        if int(prefetch) > 0:
            dataset = dataset.prefetch(int(prefetch))
        return dataset
