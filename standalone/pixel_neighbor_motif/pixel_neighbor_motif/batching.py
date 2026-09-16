"""Fast batch collation and dataset pipeline for pixel neighbor motif."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

from pixel_neighbor_motif.dataset import FERPixelDataset


def collate_samples(samples: list[dict]) -> dict[str, tf.Tensor]:
    """Collate a list of sample dicts into a batch of tensors."""
    node_features = np.stack([s["node_features"] for s in samples], axis=0)  # [B, 2304, 5]
    labels = np.array([s["label"] for s in samples], dtype=np.int64)         # [B]
    sample_ids = np.array([s["sample_id"] for s in samples], dtype=np.int64) # [B]
    images = np.stack([s["image_48"] for s in samples], axis=0)              # [B, 48, 48]

    return {
        "node_features": tf.convert_to_tensor(node_features, dtype=tf.float32),
        "labels": tf.convert_to_tensor(labels, dtype=tf.int64),
        "sample_ids": tf.convert_to_tensor(sample_ids, dtype=tf.int64),
        "image_48": tf.convert_to_tensor(images, dtype=tf.float32),
    }


class PixelBatchGenerator:
    """Deterministic batch generator supporting multithreading, LRU caching, and TF dataset integration."""

    def __init__(
        self,
        fer_csv: str | Path,
        split: str,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        cache_size: int = 512,
        workers: int = 2,
        dataset: FERPixelDataset | None = None,
    ):
        self.dataset = dataset if dataset is not None else FERPixelDataset(fer_csv, split)
        self.split = split
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.cache_size = max(int(cache_size), 0)
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.workers = max(int(workers), 1)
        self._cache_lock = threading.Lock()

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _order(self, epoch: int) -> np.ndarray:
        order = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
            rng.shuffle(order)
        return order

    def _get_sample(self, index: int) -> dict:
        key = int(index)
        with self._cache_lock:
            if key in self.cache:
                sample = self.cache.pop(key)
                self.cache[key] = sample
                return sample
        sample = self.dataset[key]
        if self.cache_size:
            with self._cache_lock:
                self.cache[key] = sample
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return sample

    def iter_epoch(self, epoch: int, limit_batches: int | None = None):
        order = self._order(epoch)
        executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        try:
            for start in range(0, len(order), self.batch_size):
                batch_number = start // self.batch_size
                if limit_batches is not None and batch_number >= int(limit_batches):
                    break
                indices = [int(idx) for idx in order[start : start + self.batch_size]]
                if executor is None:
                    samples = [self._get_sample(idx) for idx in indices]
                else:
                    samples = list(executor.map(self._get_sample, indices))
                yield collate_samples(samples)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    @staticmethod
    def output_signature() -> dict[str, tf.TensorSpec]:
        return {
            "node_features": tf.TensorSpec((None, 2304, 5), tf.float32),
            "labels": tf.TensorSpec((None,), tf.int64),
            "sample_ids": tf.TensorSpec((None,), tf.int64),
            "image_48": tf.TensorSpec((None, 48, 48), tf.float32),
        }

    def as_dataset(
        self,
        epoch: int,
        limit_batches: int | None = None,
        prefetch: int = 2,
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
