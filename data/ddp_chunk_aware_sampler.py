"""DDP-safe chunk-aware batch sampling for graph-repository datasets."""

from __future__ import annotations

import random
from statistics import mean
from typing import Dict, Iterator, List, Tuple

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
    ) -> None:
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
        if self.shuffle_within_chunk:
            random.Random(self.seed + int(epoch) * 1_000_003 + int(chunk_id)).shuffle(indices)
        return indices

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
        truncated_batches = [src - dst for src, dst in zip(before, after)]
        dropped_samples = [
            dropped_from_incomplete + sum(len(batch) for batch in rank_batches[after_count:])
            for (rank_batches, dropped_from_incomplete), after_count in zip(rank_batches_and_drops, after)
        ]
        unique_batch_sizes = [
            sorted({len(batch) for batch in rank_batches})
            for rank_batches in after_plans
        ]
        return {
            "num_chunks": self.num_chunks,
            "chunk_size_min": min(chunk_sizes),
            "chunk_size_mean": mean(chunk_sizes),
            "chunk_size_max": max(chunk_sizes),
            "total_samples": self.total_samples,
            "rank_chunk_counts": rank_chunk_counts,
            "batches_before_balance": before,
            "batches_after_balance": after,
            "truncated_batches": truncated_batches,
            "dropped_samples_per_rank": dropped_samples,
            "unique_batch_sizes_per_rank": unique_batch_sizes,
        }

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._plans_for_epoch(self.epoch)[self.rank]

    def __len__(self) -> int:
        return len(self._plans_for_epoch(self.epoch)[self.rank])
