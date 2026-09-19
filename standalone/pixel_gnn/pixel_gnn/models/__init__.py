"""Model registry for Pixel GNN architectures."""

from __future__ import annotations

from pixel_gnn.models.pixel_neighbor_motif import PixelNeighborMotifModel
from pixel_gnn.models.pixel_gnn_only import PixelGNNOnly
from pixel_gnn.models.pixel_motif_dual_scale import PixelMotifDualScaleModel

MODEL_REGISTRY = {
    "pixel_neighbor_motif": PixelNeighborMotifModel,
    "pixel_motif_graph": PixelNeighborMotifModel,
    "pixel_motif_dual_scale": PixelMotifDualScaleModel,
    "pixel_gnn_only": PixelGNNOnly,
}


def build_model(config: dict):
    """Build and return model instance based on model.name in config."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "pixel_neighbor_motif")

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: {name!r}. Available models in registry: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]

    if name in ("pixel_neighbor_motif", "pixel_motif_graph"):
        return model_cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_attention_layers=int(model_cfg.get("num_attention_layers", 1)),
            num_motifs=int(model_cfg.get("num_motifs", 32)),
            use_motif_graph=bool(model_cfg.get("use_motif_graph", True)),
            num_motif_gnn_layers=int(model_cfg.get("num_motif_gnn_layers", 1)),
            temperature=float(model_cfg.get("temperature", 0.1)),
            pooling_type=str(model_cfg.get("pooling_type", "motif")),
            dropout=float(model_cfg.get("dropout", 0.1)),
            num_classes=int(model_cfg.get("num_classes", 7)),
        )
    elif name == "pixel_motif_dual_scale":
        return model_cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            num_attention_layers=int(model_cfg.get("num_attention_layers", 3)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_motifs=int(model_cfg.get("num_motifs", 32)),
            spatial_span=float(model_cfg.get("spatial_span", 0.58)),
            use_motif_graph=bool(model_cfg.get("use_motif_graph", True)),
            num_motif_gnn_layers=int(model_cfg.get("num_motif_gnn_layers", 2)),
            num_motif_heads=int(model_cfg.get("num_motif_heads", 4)),
            ffn_expansion=int(model_cfg.get("ffn_expansion", 2)),
            temperature=float(model_cfg.get("temperature", 0.1)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            num_classes=int(model_cfg.get("num_classes", 7)),
        )
    elif name == "pixel_gnn_only":
        return model_cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            edge_dim=int(model_cfg.get("edge_context_gnn", {}).get("edge_attr_dim", 6)),
            edge_hidden=int(model_cfg.get("edge_context_gnn", {}).get("edge_hidden_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            gnn_dropout=float(model_cfg.get("edge_context_gnn", {}).get("dropout", 0.25)),
            num_classes=int(model_cfg.get("num_classes", 7)),
        )


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "PixelNeighborMotifModel",
    "PixelMotifDualScaleModel",
    "PixelGNNOnly",
]
