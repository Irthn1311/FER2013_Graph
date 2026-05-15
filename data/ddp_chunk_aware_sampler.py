"""DDP-safe chunk-aware batch sampling for graph-repository datasets."""

from __future__ import annotations

import random
from statistics import mean
from typing import Dict, Iterator, List, Mapping, Tuple

from torch.utils.data import Sampler


class DDPChunkAwareBatchSampler(Sampler[List[int]]):
    """Yield balanced per-rank batches while preserving graph chunk locality."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        ddp_drop_last_batches: bool = True,
        fixed_batch_size: bool = False,
        drop_incomplete_batches: bool = False,
        carry_over_leftovers: bool = False,
        target_class_repeat_factors: Mapping[int | str, float] | None = None,
    ) -> None:
        if target_class_repeat_factors and not hasattr(dataset, "label_at_index"):
            raise ValueError(
                "data.target_class_repeat_factors requires dataset.label_at_index(idx); "
                "this dataset cannot expose labels for repeat sampling"
            )
        self.dataset = dataset
        self.chunk_indices = dataset.chunk_index_groups()
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.ddp_drop_last_batches = bool(ddp_drop_last_batches)
        self.fixed_batch_size = bool(fixed_batch_size)
        self.drop_incomplete_batches = bool(drop_incomplete_batches)
        self.carry_over_leftovers = bool(carry_over_leftovers)
        self.target_class_repeat_factors = self._normalize_repeat_factors(target_class_repeat_factors or {})
        self.epoch = 0
        self._cached_plans: Dict[int, List[List[List[int]]]] = {}

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {rank}")
        if not self.chunk_indices:
            raise ValueError("dataset.chunk_index_groups() returned no chunks")
        if self.fixed_batch_size and not self.carry_over_leftovers and not self.drop_incomplete_batches:
            raise ValueError(
                "fixed_batch_size=True requires carry_over_leftovers=True or "
                "drop_incomplete_batches=True"
            )

    @staticmethod
    def _normalize_repeat_factors(values: Mapping[int | str, float]) -> Dict[int, float]:
        repeat_factors: Dict[int, float] = {}
        for key, value in values.items():
            cls = int(key)
            factor = float(value)
            if factor <= 0.0:
                raise ValueError(f"target_class_repeat_factors[{cls}] must be > 0, got {factor}")
            if factor != 1.0:
                repeat_factors[cls] = factor
        return repeat_factors

    @property
    def chunk_sizes(self) -> List[int]:
        return [len(indices) for indices in self.chunk_indices]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_indices)

    @property
    def total_samples(self) -> int:
        return sum(self.chunk_sizes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _chunk_order(self, epoch: int) -> List[int]:
        chunk_ids = list(range(self.num_chunks))
        if self.shuffle_chunks:
            random.Random(self.seed + int(epoch)).shuffle(chunk_ids)
        return chunk_ids

    def _chunk_batches(self, chunk_id: int, epoch: int) -> List[List[int]]:
        indices = self._chunk_indices(chunk_id, epoch)
        batches: List[List[int]] = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                batches.append(batch)
        return batches

    def _build_rank_batches_with_drops(self, rank: int, epoch: int) -> Tuple[List[List[int]], int]:
        batches: List[List[int]] = []
        rank_chunk_ids = self._chunk_order(epoch)[rank :: self.num_replicas]
        dropped_samples = 0

        if self.fixed_batch_size and self.carry_over_leftovers:
            buffer: List[int] = []
            for chunk_id in rank_chunk_ids:
                buffer.extend(self._chunk_indices(chunk_id, epoch))
                while len(buffer) >= self.batch_size:
                    batches.append(buffer[: self.batch_size])
                    buffer = buffer[self.batch_size :]
            dropped_samples = len(buffer)
            return batches, dropped_samples

        for chunk_id in rank_chunk_ids:
            chunk_batches = self._chunk_batches(chunk_id, epoch)
            if self.fixed_batch_size:
                full_batches = [batch for batch in chunk_batches if len(batch) == self.batch_size]
                dropped_samples += sum(len(batch) for batch in chunk_batches if len(batch) < self.batch_size)
                batches.extend(full_batches)
            else:
                batches.extend(chunk_batches)
        return batches, dropped_samples

    def _chunk_indices(self, chunk_id: int, epoch: int) -> List[int]:
        indices = list(self.chunk_indices[chunk_id])
        if self.target_class_repeat_factors:
            indices = self._repeat_target_indices(indices, chunk_id, epoch)
        if self.shuffle_within_chunk:
            random.Random(self.seed + int(epoch) * 1_000_003 + int(chunk_id)).shuffle(indices)
        return indices

    def _label_for_index(self, sample_idx: int) -> int:
        return int(self.dataset.label_at_index(int(sample_idx)))

    def _repeat_target_indices(self, indices: List[int], chunk_id: int, epoch: int) -> List[int]:
        repeated: List[int] = []
        for sample_idx in indices:
            label = self._label_for_index(sample_idx)
            factor = self.target_class_repeat_factors.get(label, 1.0)
            full_count = int(factor)
            frac = float(factor) - float(full_count)
            count = max(1, full_count)
            if frac > 0.0:
                rng = random.Random(
                    self.seed
                    + int(epoch) * 1_000_003
                    + int(chunk_id) * 9_176
                    + int(sample_idx) * 37
                    + int(label) * 101
                )
                if rng.random() < frac:
                    count += 1
            repeated.extend([sample_idx] * count)
        return repeated

    def _build_rank_batches(self, rank: int, epoch: int) -> List[List[int]]:
        batches, _ = self._build_rank_batches_with_drops(rank, epoch)
        return batches

    def _plans_for_epoch(self, epoch: int) -> List[List[List[int]]]:
        epoch = int(epoch)
        if epoch in self._cached_plans:
            return self._cached_plans[epoch]

        plans = [self._build_rank_batches(rank, epoch) for rank in range(self.num_replicas)]
        if self.ddp_drop_last_batches:
            min_batches = min(len(rank_batches) for rank_batches in plans)
            plans = [rank_batches[:min_batches] for rank_batches in plans]
        else:
            lengths = {len(rank_batches) for rank_batches in plans}
            if len(lengths) != 1:
                raise RuntimeError(
                    "DDPChunkAwareBatchSampler produced uneven rank lengths with "
                    "ddp_drop_last_batches=False; enable truncation or change the data split"
                )
        self._cached_plans[epoch] = plans
        return plans

    def summary(self, epoch: int = 0) -> Dict[str, object]:
        chunk_order = self._chunk_order(epoch)
        rank_chunk_counts = [
            len(chunk_order[rank :: self.num_replicas])
            for rank in range(self.num_replicas)
        ]
        rank_batches_and_drops = [
            self._build_rank_batches_with_drops(rank, epoch)
            for rank in range(self.num_replicas)
        ]
        before = [len(rank_batches) for rank_batches, _ in rank_batches_and_drops]
        after_plans = self._plans_for_epoch(epoch)
        after = [len(rank_batches) for rank_batches in after_plans]
        chunk_sizes = self.chunk_sizes
        expanded_chunk_sizes = [
            len(self._chunk_indices(chunk_id, epoch))
            for chunk_id in range(self.num_chunks)
        ]
        truncated_batches = [src - dst for src, dst in zip(before, after)]
        dropped_samples = [
            dropped_from_incomplete + sum(len(batch) for batch in rank_batches[after_count:])
            for (rank_batches, dropped_from_incomplete), after_count in zip(rank_batches_and_drops, after)
        ]
        unique_batch_sizes = [
            sorted({len(batch) for batch in rank_batches})
            for rank_batches in after_plans
        ]
        label_histograms = []
        for rank_batches in after_plans:
            hist: Dict[int, int] = {}
            for batch in rank_batches:
                for sample_idx in batch:
                    label = self._label_for_index(sample_idx)
                    hist[label] = hist.get(label, 0) + 1
            label_histograms.append(hist)
        return {
            "num_chunks": self.num_chunks,
            "chunk_size_min": min(chunk_sizes),
            "chunk_size_mean": mean(chunk_sizes),
            "chunk_size_max": max(chunk_sizes),
            "total_samples": self.total_samples,
            "target_class_repeat_factors": dict(self.target_class_repeat_factors),
            "repeated_num_indices_total": int(sum(expanded_chunk_sizes) - self.total_samples),
            "expanded_total_samples": int(sum(expanded_chunk_sizes)),
            "rank_chunk_counts": rank_chunk_counts,
            "batches_before_balance": before,
            "batches_after_balance": after,
            "truncated_batches": truncated_batches,
            "dropped_samples_per_rank": dropped_samples,
            "unique_batch_sizes_per_rank": unique_batch_sizes,
            "per_rank_label_histogram_estimate": label_histograms,
        }

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._plans_for_epoch(self.epoch)[self.rank]

    def __len__(self) -> int:
        return len(self._plans_for_epoch(self.epoch)[self.rank])
