"""Runtime feature masking helpers for graph feature ablation runs."""

from __future__ import annotations

from typing import Any, Dict

import torch


DEFAULT_NODE_FEATURE_NAMES = [
    "intensity",
    "x_norm",
    "y_norm",
    "gx",
    "gy",
    "grad_mag",
    "local_contrast",
]
DEFAULT_EDGE_FEATURE_NAMES = [
    "dx",
    "dy",
    "dist",
    "delta_intensity",
    "intensity_similarity",
]


def is_feature_ablation_enabled(feature_ablation_cfg: Dict[str, Any] | None) -> bool:
    return bool((feature_ablation_cfg or {}).get("enabled", False))


def _indices(cfg: Dict[str, Any], key: str, default_dim: int) -> list[int]:
    values = cfg.get(key)
    if values is None:
        return list(range(int(default_dim)))
    indices = [int(value) for value in values]
    if not indices:
        raise ValueError(f"feature_ablation.{key} must not be empty when enabled")
    return indices


def feature_ablation_indices(feature_ablation_cfg: Dict[str, Any] | None) -> tuple[list[int], list[int]]:
    cfg = dict(feature_ablation_cfg or {})
    return (
        _indices(cfg, "node_indices", len(DEFAULT_NODE_FEATURE_NAMES)),
        _indices(cfg, "edge_indices", len(DEFAULT_EDGE_FEATURE_NAMES)),
    )


def apply_feature_ablation(
    batch: Dict[str, Any],
    feature_ablation_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Mask node and edge features on the last dimension without changing graph artifacts."""

    if not is_feature_ablation_enabled(feature_ablation_cfg):
        return batch
    cfg = dict(feature_ablation_cfg or {})
    node_indices, edge_indices = feature_ablation_indices(cfg)
    if "x" not in batch:
        raise KeyError("feature_ablation requires batch['x']")
    if "edge_attr" not in batch:
        raise KeyError("feature_ablation requires batch['edge_attr']")
    x = batch["x"]
    edge_attr = batch["edge_attr"]
    if not torch.is_tensor(x) or not torch.is_tensor(edge_attr):
        raise TypeError("feature_ablation expects tensor batch['x'] and batch['edge_attr']")
    if max(node_indices) >= int(x.shape[-1]) or min(node_indices) < 0:
        raise ValueError(f"node_indices={node_indices} out of range for x last dim={int(x.shape[-1])}")
    if max(edge_indices) >= int(edge_attr.shape[-1]) or min(edge_indices) < 0:
        raise ValueError(f"edge_indices={edge_indices} out of range for edge_attr last dim={int(edge_attr.shape[-1])}")

    node_idx = torch.as_tensor(node_indices, dtype=torch.long, device=x.device)
    edge_idx = torch.as_tensor(edge_indices, dtype=torch.long, device=edge_attr.device)
    masked_x = x.index_select(dim=-1, index=node_idx)
    masked_edge_attr = edge_attr.index_select(dim=-1, index=edge_idx)
    batch["x"] = masked_x
    if "node_features" in batch:
        batch["node_features"] = masked_x
    batch["edge_attr"] = masked_edge_attr
    return batch


def assert_feature_dims(batch: Dict[str, Any], node_dim: int, edge_dim: int) -> None:
    x = batch.get("x", batch.get("node_features"))
    edge_attr = batch.get("edge_attr")
    if x is None or edge_attr is None:
        raise KeyError("Feature dim assertion requires x/node_features and edge_attr")
    actual_node_dim = int(x.shape[-1])
    actual_edge_dim = int(edge_attr.shape[-1])
    if actual_node_dim != int(node_dim):
        raise ValueError(f"batch x last dim={actual_node_dim} does not match model node_dim={int(node_dim)}")
    if actual_edge_dim != int(edge_dim):
        raise ValueError(f"edge_attr last dim={actual_edge_dim} does not match model edge_dim={int(edge_dim)}")


def feature_ablation_summary(
    feature_ablation_cfg: Dict[str, Any] | None,
    model_node_dim: int,
    model_edge_dim: int,
) -> Dict[str, Any]:
    cfg = dict(feature_ablation_cfg or {})
    enabled = is_feature_ablation_enabled(cfg)
    if enabled:
        node_indices, edge_indices = feature_ablation_indices(cfg)
    else:
        node_indices = list(range(int(cfg.get("original_node_dim", len(DEFAULT_NODE_FEATURE_NAMES)))))
        edge_indices = list(range(int(cfg.get("original_edge_dim", len(DEFAULT_EDGE_FEATURE_NAMES)))))
    original_node_dim = int(cfg.get("original_node_dim", len(DEFAULT_NODE_FEATURE_NAMES)))
    original_edge_dim = int(cfg.get("original_edge_dim", len(DEFAULT_EDGE_FEATURE_NAMES)))
    return {
        "enabled": enabled,
        "node_indices": node_indices,
        "edge_indices": edge_indices,
        "original_node_dim": original_node_dim,
        "masked_node_dim": len(node_indices) if enabled else int(model_node_dim),
        "original_edge_dim": original_edge_dim,
        "masked_edge_dim": len(edge_indices) if enabled else int(model_edge_dim),
        "model_node_dim": int(model_node_dim),
        "model_edge_dim": int(model_edge_dim),
    }


def log_feature_ablation(
    feature_ablation_cfg: Dict[str, Any] | None,
    model_node_dim: int,
    model_edge_dim: int,
    prefix: str = "[FeatureAblation]",
) -> None:
    summary = feature_ablation_summary(feature_ablation_cfg, model_node_dim, model_edge_dim)
    print(
        f"{prefix} enabled={summary['enabled']} "
        f"node_indices={summary['node_indices']} edge_indices={summary['edge_indices']} "
        f"original_node_dim={summary['original_node_dim']} masked_node_dim={summary['masked_node_dim']} "
        f"original_edge_dim={summary['original_edge_dim']} masked_edge_dim={summary['masked_edge_dim']} "
        f"model_node_dim={summary['model_node_dim']} model_edge_dim={summary['model_edge_dim']}"
    )
    if summary["masked_node_dim"] != summary["model_node_dim"]:
        raise ValueError(
            f"feature_ablation masked node dim={summary['masked_node_dim']} "
            f"does not match model node_dim={summary['model_node_dim']}"
        )
    if summary["masked_edge_dim"] != summary["model_edge_dim"]:
        raise ValueError(
            f"feature_ablation masked edge dim={summary['masked_edge_dim']} "
            f"does not match model edge_dim={summary['model_edge_dim']}"
        )
