"""Configuration loading and locked-baseline validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from lap_gnn_tf.constants import EXPECTED_PARAMETER_COUNT


EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent = load_config(path.parent / str(parent_name))
        cfg = _deep_merge(parent, cfg)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_locked_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    graph = config.get("graph", {})
    training = config.get("training", {})
    checks = {
        "hidden_dim": model.get("hidden_dim") == 96,
        "gnn_layers": model.get("edge_context_gnn", {}).get("num_layers") == 3,
        "edge_dim": model.get("edge_context_gnn", {}).get("edge_attr_dim") == 8,
        "mean_aggregation": model.get("edge_context_gnn", {}).get("aggregation") == "mean",
        "graph_mode": graph.get("graph_mode") == "face_plus_context",
        "anchors": graph.get("anchor_nodes", {}).get("enabled") is True,
        "readout": model.get("readout_type") == "micro_motif_support",
        "optimizer": str(training.get("optimizer", {}).get("type", "")).lower() == "adamw",
        "gradient_execution_mode": training.get("gradient_execution_mode") == "tf_function",
        "optimizer_execution_mode": training.get("optimizer_execution_mode")
        == "restricted_tf_function",
        "grappler_profile": training.get("grappler_profile") == "G1-A",
        "execution_contract": config.get("locked", {}).get(
            "execution_contract_sha256"
        )
        == EXECUTION_CONTRACT_SHA256,
        "scheduler": training.get("scheduler", {}).get("type") == "plateau",
        "checkpoint": (
            training.get("checkpoint_monitor") == "val_accuracy"
            and training.get("final_test_checkpoint") == "best_val_accuracy"
            and training.get("checkpoint_policy", {}).get("type") == "single"
            and training.get("checkpoint_policy", {}).get("monitor")
            == "val_accuracy"
        ),
        "parameter_count": config.get("locked", {}).get("parameter_count") == EXPECTED_PARAMETER_COUNT,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"TensorFlow locked-baseline config drift: {failures}")
    return checks
