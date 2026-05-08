"""Full graph dataset and collate function for D5A."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from torch.utils.data import Dataset, Sampler

from data.graph_repository import ChunkedGraphDataset
from data.graph_types import ResolvedPixelGraph


class FullGraphDataset(Dataset):
    """Return one resolved 48x48 pixel graph per FER-2013 sample."""

    def __init__(
        self,
        repo_root: str | Path,
        split: str,
        chunk_cache_size: int = 0,
        graph_cache_chunks: int | None = None,
    ) -> None:
        self.dataset = ChunkedGraphDataset(
            repo_root=repo_root,
            split=split,
            resolve=True,
            chunk_cache_size=chunk_cache_size,
            graph_cache_chunks=graph_cache_chunks,
        )
        print(
            f"[FullGraphDataset {split}]\n"
            f"chunk_cache_size={self.dataset.chunk_cache_size}\n"
            f"num_chunks={len(self.dataset.chunk_paths)}\n"
            f"num_samples={len(self.dataset)}"
        )
        if self.dataset.chunk_cache_size > 0:
            print(
                f"[FullGraphDataset {split}] chunk cache enabled "
                f"(max_chunks={self.dataset.chunk_cache_size})"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def num_chunks(self) -> int:
        return len(self.dataset.chunk_paths)

    def chunk_index_groups(self) -> List[List[int]]:
        return self.dataset.chunk_index_groups()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(idx)
        graph = self.dataset[sample_idx]
        if not isinstance(graph, ResolvedPixelGraph):
            raise TypeError("FullGraphDataset expected a resolved graph")
        num_nodes = int(graph.node_features.shape[0])
        return {
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "global_index": torch.tensor(sample_idx, dtype=torch.long),
            "graph_id": torch.tensor(int(graph.graph_id), dtype=torch.long),
            "node_features": graph.node_features.float(),
            "x": graph.node_features.float(),
            "edge_index": graph.edge_index.long(),
            "edge_attr": graph.edge_attr.float(),
            "node_mask": torch.ones(num_nodes, dtype=torch.bool),
            "y": torch.tensor(int(graph.label), dtype=torch.long),
            "label": torch.tensor(int(graph.label), dtype=torch.long),
        }


class ChunkAwareBatchSampler(Sampler[List[int]]):
    """Yield batches that stay within one graph-repo chunk when possible."""

    def __init__(
        self,
        dataset: FullGraphDataset,
        batch_size: int,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
    ) -> None:
        self.chunk_indices = dataset.chunk_index_groups()
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.batches_per_epoch = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.chunk_indices
            if indices
        )
        print(
            "[ChunkAwareSampler] enabled=True\n"
            f"num_chunks={len(self.chunk_indices)}\n"
            f"batch_size={self.batch_size}\n"
            f"batches_per_epoch={self.batches_per_epoch}\n"
            f"shuffle_chunks={self.shuffle_chunks}\n"
            f"shuffle_within_chunk={self.shuffle_within_chunk}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        chunk_order = list(range(len(self.chunk_indices)))
        if self.shuffle_chunks:
            random.shuffle(chunk_order)
        for chunk_idx in chunk_order:
            indices = list(self.chunk_indices[chunk_idx])
            if not indices:
                continue
            if self.shuffle_within_chunk:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return self.batches_per_epoch


def collate_fn_full_graph(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate resolved graphs into the D5A batch contract."""

    x = torch.stack([item["node_features"] for item in batch], dim=0)
    edge_attr = torch.stack([item["edge_attr"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    node_mask = torch.stack([item["node_mask"] for item in batch], dim=0)
    graph_id = torch.stack([item["graph_id"] for item in batch], dim=0)
    sample_idx = torch.stack([item["sample_idx"] for item in batch], dim=0)
    return {
        "sample_idx": sample_idx,
        "global_index": sample_idx,
        "graph_id": graph_id,
        "x": x,
        "node_features": x,
        "edge_index": batch[0]["edge_index"].unsqueeze(0).expand(len(batch), -1, -1),
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "y": y,
        "label": y,
    }
