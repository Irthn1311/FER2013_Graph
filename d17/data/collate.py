"""Batch collation for D17 variable-size pixel graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from d17.data.epp_graph_builder import EPPGraphData


@dataclass
class EPPBatch:
    x_cat: torch.Tensor
    edge_index_cat: torch.Tensor
    edge_attr_cat: torch.Tensor
    batch_index: torch.Tensor
    ptr: torch.Tensor
    y: torch.Tensor
    sample_index: torch.Tensor
    pos_cat: torch.Tensor
    detected: torch.Tensor
    landmark_missing_flag: torch.Tensor
    image_48: torch.Tensor
    local_edge_count: torch.Tensor
    knn_edge_count: torch.Tensor
    total_edge_count: torch.Tensor
    node_feature_names: List[str]
    edge_feature_names: List[str]

    @property
    def num_graphs(self) -> int:
        return int(self.y.numel())

    def to(self, device: torch.device | str) -> "EPPBatch":
        return EPPBatch(
            x_cat=self.x_cat.to(device),
            edge_index_cat=self.edge_index_cat.to(device),
            edge_attr_cat=self.edge_attr_cat.to(device),
            batch_index=self.batch_index.to(device),
            ptr=self.ptr.to(device),
            y=self.y.to(device),
            sample_index=self.sample_index.to(device),
            pos_cat=self.pos_cat.to(device),
            detected=self.detected.to(device),
            landmark_missing_flag=self.landmark_missing_flag.to(device),
            image_48=self.image_48.to(device),
            local_edge_count=self.local_edge_count.to(device),
            knn_edge_count=self.knn_edge_count.to(device),
            total_edge_count=self.total_edge_count.to(device),
            node_feature_names=self.node_feature_names,
            edge_feature_names=self.edge_feature_names,
        )


def collate_epp_graphs(graphs: List[EPPGraphData]) -> EPPBatch:
    if not graphs:
        raise ValueError("collate_epp_graphs received an empty graph list")
    xs, edge_indices, edge_attrs, batch_index, pos = [], [], [], [], []
    ptr = [0]
    ys, sample_indices, detected, missing, images = [], [], [], [], []
    local_counts, knn_counts, total_counts = [], [], []
    offset = 0
    for batch_id, graph in enumerate(graphs):
        n = int(graph.x.size(0))
        xs.append(graph.x)
        edge_indices.append(graph.edge_index + offset)
        edge_attrs.append(graph.edge_attr)
        batch_index.append(torch.full((n,), batch_id, dtype=torch.long))
        pos.append(graph.pos)
        ys.append(graph.y)
        sample_indices.append(graph.sample_index)
        detected.append(graph.detected)
        missing.append(graph.landmark_missing_flag)
        images.append(graph.image_48)
        local_counts.append(torch.tensor(graph.local_edge_count, dtype=torch.long))
        knn_counts.append(torch.tensor(graph.knn_edge_count, dtype=torch.long))
        total_counts.append(torch.tensor(graph.total_edge_count, dtype=torch.long))
        offset += n
        ptr.append(offset)
    return EPPBatch(
        x_cat=torch.cat(xs, dim=0).float(),
        edge_index_cat=torch.cat(edge_indices, dim=1).long(),
        edge_attr_cat=torch.cat(edge_attrs, dim=0).float(),
        batch_index=torch.cat(batch_index, dim=0).long(),
        ptr=torch.tensor(ptr, dtype=torch.long),
        y=torch.stack(ys).long(),
        sample_index=torch.stack(sample_indices).long(),
        pos_cat=torch.cat(pos, dim=0).float(),
        detected=torch.stack(detected).bool(),
        landmark_missing_flag=torch.stack(missing).long(),
        image_48=torch.stack(images).float(),
        local_edge_count=torch.stack(local_counts).long(),
        knn_edge_count=torch.stack(knn_counts).long(),
        total_edge_count=torch.stack(total_counts).long(),
        node_feature_names=list(graphs[0].node_feature_names),
        edge_feature_names=list(graphs[0].edge_feature_names),
    )

