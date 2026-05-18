"""D13A pure GNN hierarchical pixel-to-region reduction model."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.d13_encoders import EdgeAwarePixelEncoderLiteV2, GINEPixelEncoder
from models.d13_pooling import LocalAssignmentPool


class GraphSAGEBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(h)
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            neigh = h.new_zeros(h.shape)
            neigh.index_add_(0, dst, h.index_select(0, src))
            deg = h.new_zeros((h.shape[0], 1))
            deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            neigh = neigh / deg.clamp_min(1.0)
        h = self.norm(self.self_lin(h) + self.neigh_lin(neigh))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        return h


class RegionGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(GraphSAGEBlock(hidden_dim, dropout=dropout) for _ in range(int(num_layers)))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, edge_index=edge_index)
        return h


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_ids = torch.unique(batch.long(), sorted=True)
        pooled = []
        weights_all = h.new_zeros((h.shape[0],))
        logits = self.score(h).squeeze(-1)
        for out_idx, graph_id in enumerate(graph_ids.tolist()):
            mask = batch.long() == int(graph_id)
            w = torch.softmax(logits[mask], dim=0)
            pooled.append((h[mask] * w.unsqueeze(-1)).sum(dim=0))
            weights_all[mask] = w
        return torch.stack(pooled, dim=0), weights_all


def _pool_mean_max(h: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    graph_ids = torch.unique(batch.long(), sorted=True)
    means = []
    maxes = []
    for graph_id in graph_ids.tolist():
        rows = h[batch.long() == int(graph_id)]
        means.append(rows.mean(dim=0))
        maxes.append(rows.max(dim=0).values)
    return torch.stack(means, dim=0), torch.stack(maxes, dim=0)


class D13HierarchicalReductionModel(nn.Module):
    """Pixel encoder -> local assignment reduction -> region GNN -> classifier."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 7,
        pixel_layers: int = 2,
        region_layers: int = 2,
        dropout: float = 0.1,
        encoder: str | Dict[str, Any] = "edgeaware_lite_v2",
        pooling: str | Dict[str, Any] = "local_assignment",
        readout_attention: bool = True,
        return_region_embeddings: bool = False,
        return_encoder_layers: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.return_region_embeddings = bool(return_region_embeddings)
        self.return_encoder_layers = bool(return_encoder_layers)

        enc_cfg = dict(encoder) if isinstance(encoder, dict) else {"name": encoder}
        enc_name = str(enc_cfg.pop("name", "edgeaware_lite_v2")).lower()
        enc_cfg.setdefault("node_dim", self.node_dim)
        enc_cfg.setdefault("edge_dim", self.edge_dim)
        enc_cfg.setdefault("hidden_dim", self.hidden_dim)
        enc_cfg.setdefault("num_layers", int(pixel_layers))
        enc_cfg.setdefault("dropout", float(dropout))
        if enc_name in {"edgeaware_lite_v2", "edgeaware", "edgeaware_lite"}:
            self.pixel_encoder = EdgeAwarePixelEncoderLiteV2(**enc_cfg)
        elif enc_name in {"gine", "gine_pixel"}:
            self.pixel_encoder = GINEPixelEncoder(**enc_cfg)
        else:
            raise ValueError(f"Unknown D13 encoder: {enc_name}")

        pool_cfg = dict(pooling) if isinstance(pooling, dict) else {"name": pooling}
        pool_name = str(pool_cfg.pop("name", "local_assignment")).lower()
        pool_cfg.setdefault("hidden_dim", self.hidden_dim)
        pool_cfg.setdefault("dropout", float(dropout))
        self.pooling_name = pool_name
        if pool_name in {"local_assignment", "localpool", "local_assignment_pool"}:
            self.reduction = LocalAssignmentPool(**pool_cfg)
        elif pool_name in {"sagpool", "topk", "sagpool/topk"}:
            ratio = float(pool_cfg.get("ratio", 0.0625))
            grid_size = max(1, int(round((2304 * ratio) ** 0.5)))
            self.reduction = LocalAssignmentPool(
                hidden_dim=self.hidden_dim,
                grid_size=grid_size,
                assign_m=int(pool_cfg.get("assign_m", 4)),
                dropout=float(dropout),
            )
        else:
            raise ValueError(f"Unknown D13 pooling: {pool_name}")

        self.region_encoder = RegionGraphEncoder(self.hidden_dim, num_layers=int(region_layers), dropout=dropout)
        self.use_attention = bool(readout_attention)
        if self.use_attention:
            self.attn_readout = AttentionReadout(self.hidden_dim)
            readout_dim = self.hidden_dim * 3
        else:
            readout_dim = self.hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "D13HierarchicalReductionModel":
        return cls(**dict(cfg))

    @staticmethod
    def _dense_to_flat(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch.get("x")
        edge_index = batch.get("edge_index")
        edge_attr = batch.get("edge_attr")
        if x is None or edge_index is None or edge_attr is None:
            raise KeyError("D13 requires batch fields: x, edge_index, edge_attr")
        if x.ndim == 2:
            if "batch" not in batch:
                raise KeyError("Flat D13 batch requires batch vector")
            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch": batch["batch"].long(),
            }
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,F] or [N,F], got {tuple(x.shape)}")
        bsz, num_nodes, feat_dim = x.shape
        if edge_index.ndim == 3:
            base_edge_index = edge_index[0].long()
        elif edge_index.ndim == 2:
            base_edge_index = edge_index.long()
        else:
            raise ValueError(f"edge_index must be [B,2,E] or [2,E], got {tuple(edge_index.shape)}")
        if edge_attr.ndim == 3:
            edge_attr_flat = edge_attr.reshape(bsz * edge_attr.shape[1], edge_attr.shape[2])
        elif edge_attr.ndim == 2:
            edge_attr_flat = edge_attr.unsqueeze(0).expand(bsz, -1, -1).reshape(-1, edge_attr.shape[-1])
        else:
            raise ValueError(f"edge_attr must be [B,E,F] or [E,F], got {tuple(edge_attr.shape)}")
        offsets = (torch.arange(bsz, device=x.device, dtype=torch.long) * num_nodes).view(bsz, 1, 1)
        edge_index_flat = (base_edge_index.to(x.device).unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        batch_vec = torch.arange(bsz, device=x.device, dtype=torch.long).repeat_interleave(num_nodes)
        return {
            "x": x.reshape(bsz * num_nodes, feat_dim),
            "edge_index": edge_index_flat,
            "edge_attr": edge_attr_flat,
            "batch": batch_vec,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        flat = self._dense_to_flat(batch)
        x = flat["x"]
        edge_index = flat["edge_index"]
        edge_attr = flat["edge_attr"]
        pixel_batch = flat["batch"]
        if x.shape[1] < 3:
            raise ValueError("D13 expects x[:, 1:3] to contain normalized pixel positions")
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(f"D13 expected edge_dim={self.edge_dim}, got {edge_attr.shape[1]}")
        pos = x[:, 1:3].contiguous()

        enc = self.pixel_encoder(
            x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_layer_embeddings=self.return_encoder_layers,
        )
        h_pixel = enc["h"]
        h_region, region_edge_index, region_batch, aux = self.reduction(
            h_pixel,
            pos=pos,
            batch=pixel_batch,
        )
        region_pos = aux.pop("region_pos")
        h_region = self.region_encoder(h_region, region_edge_index)
        mean_pool, max_pool = _pool_mean_max(h_region, region_batch)
        out_parts = [mean_pool, max_pool]
        attn_weights = None
        if self.use_attention:
            attn_pool, attn_weights = self.attn_readout(h_region, region_batch)
            out_parts.append(attn_pool)
        logits = self.classifier(torch.cat(out_parts, dim=-1))
        diagnostics = dict(enc.get("diagnostics", {}))
        for key, value in aux.items():
            if torch.is_tensor(value) and value.numel() == 1:
                diagnostics[f"pool_{key}"] = value.detach()
        output: Dict[str, Any] = {
            "logits": logits,
            "aux": aux,
            "diagnostics": diagnostics,
            "h_pixel": h_pixel,
            "h_region": h_region,
            "region_batch": region_batch,
            "region_pos": region_pos,
            "region_edge_index": region_edge_index,
        }
        if attn_weights is not None:
            output["region_attention"] = attn_weights
        if enc.get("layer_embeddings") is not None:
            output["pixel_layer_embeddings"] = enc["layer_embeddings"]
        if not self.return_region_embeddings:
            output.pop("h_region", None)
        return output

