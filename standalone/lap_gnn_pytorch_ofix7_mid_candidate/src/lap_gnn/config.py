"""Portable config loading and locked-baseline validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from lap_gnn.constants import ANCHOR_GROUPS, EDGE_FEATURE_NAMES


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = payload.get("extends")
    if parent:
        parent_path = (path.parent / str(parent)).resolve()
        return _merge(load_config(parent_path), payload)
    return payload


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def scientific_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(cfg[key])
        for key in ("seed", "data", "graph", "model", "loss", "training")
        if key in cfg
    }


def validate_locked_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    graph = cfg.get("graph", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    checks = [
        (data.get("graph_mode") == "face_plus_context", "data.graph_mode"),
        (graph.get("graph_mode") == "face_plus_context", "graph.graph_mode"),
        (float(graph.get("face_threshold", -1)) == 0.15, "graph.face_threshold"),
        (int(graph.get("context_pixels", -1)) == 2, "graph.context_pixels"),
        (graph.get("edge_features", {}).get("features") == EDGE_FEATURE_NAMES, "edge feature order"),
        (graph.get("anchor_nodes", {}).get("groups") == ANCHOR_GROUPS, "anchor order"),
        (int(model.get("hidden_dim", -1)) == 96, "model.hidden_dim"),
        (int(model.get("gnn_layers", -1)) == 3, "model.gnn_layers"),
        (model.get("gnn_type") == "edge_context_gnn", "model.gnn_type"),
        (model.get("readout_type") == "micro_motif_support", "model.readout_type"),
        (training.get("checkpoint_monitor") == "val_macro_f1", "checkpoint monitor"),
        (training.get("scheduler", {}).get("type") == "plateau", "scheduler type"),
        (training.get("scheduler", {}).get("monitor") == "val_loss", "scheduler monitor"),
        (training.get("early_stopping", {}).get("metric") == "val_loss", "early stopping"),
        (int(training.get("max_epochs", -1)) == 90, "max epochs"),
        (int(training.get("batch_size", -1)) == 16, "batch size"),
        (float(training.get("lr", -1)) == 3e-4, "learning rate"),
        (float(training.get("weight_decay", -1)) == 1e-3, "weight decay"),
        (bool(training.get("amp")) is True, "AMP"),
    ]
    errors.extend(label for passed, label in checks if not passed)
    seed = int(cfg.get("seed", training.get("seed", -1)))
    prior_seed = int(graph.get("prior_corruption", {}).get("seed", -1))
    if prior_seed != seed + 7699:
        errors.append("prior seed rule")
    return errors
