"""The two fixed preregistered PyTorch model families for PGM M0."""

from __future__ import annotations

import torch
from torch import nn

from .m0_config import HIDDEN_DIM, K, N_CLASSES, X_DIM


class SparseAggregateMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse_linear = nn.Linear(X_DIM, HIDDEN_DIM)
        self.hidden = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse:
            raise TypeError("M0 M input must remain sparse")
        h = torch.sparse.mm(x, self.sparse_linear.weight.t()) + self.sparse_linear.bias
        h = torch.relu(h)
        h = torch.relu(self.hidden(h))
        return self.classifier(h)


class RelationMessageLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.relation_weight = nn.Parameter(torch.empty(8, HIDDEN_DIM, HIDDEN_DIM))
        for relation in self.relation_weight:
            nn.init.kaiming_uniform_(relation, a=5**0.5)

    def forward(self, h: torch.Tensor, relation: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        if relation.dtype != torch.uint8 or relation.ndim != 3:
            raise TypeError("relation must be uint8 [batch,node,node]")
        transformed = torch.einsum("bsh,roh->brso", h, self.relation_weight)
        incoming = torch.zeros_like(h)
        for r in range(8):
            adjacency = (relation == r).transpose(1, 2).to(h.dtype)
            incoming = incoming + torch.bmm(adjacency, transformed[:, r])
        degree = node_mask.sum(dim=1, keepdim=True).sub(1).clamp_min(1).to(h.dtype).unsqueeze(-1)
        has_incoming = node_mask.unsqueeze(-1) & (node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) > 1)
        message = torch.where(has_incoming, incoming / degree, torch.zeros_like(incoming))
        out = torch.relu(self.self_linear(h) + message)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class MinimalRelationGraphModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(K, HIDDEN_DIM)
        self.message_layers = nn.ModuleList([RelationMessageLayer(), RelationMessageLayer()])
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, component: torch.Tensor, relation: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        h = self.embedding(component.clamp(min=0)) * node_mask.unsqueeze(-1).to(torch.float32)
        for layer in self.message_layers:
            h = layer(h, relation, node_mask)
        count = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / count
        pooled = torch.where(node_mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        return self.classifier(pooled)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
