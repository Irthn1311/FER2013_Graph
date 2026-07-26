"""Run provenance assembly."""

from __future__ import annotations

import json
from pathlib import Path

from lap_gnn_tf.config import canonical_config_hash
from lap_gnn_tf.resources import environment_manifest


def write_provenance(output_dir: str | Path, config: dict, signatures: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": canonical_config_hash(config),
        "signatures": signatures,
        "environment": environment_manifest(),
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

