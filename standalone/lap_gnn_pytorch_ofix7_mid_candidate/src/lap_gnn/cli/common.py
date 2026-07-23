"""Shared portable CLI handling."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from lap_gnn.config import load_config, validate_locked_config
from lap_gnn.data.fer2013 import inspect_fer_csv
from lap_gnn.priors.schema import PRIOR_SCHEMA_ID


def add_data_arguments(parser: argparse.ArgumentParser, *, output: bool = False) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--fer-csv", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--cache-root")
    if output:
        parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)


def validate_inputs(args) -> tuple[dict, Path]:
    cfg = load_config(args.config)
    failures = validate_locked_config(cfg)
    if failures:
        raise ValueError(f"Config is not the locked OFIX7-mid baseline: {failures}")
    inspect_fer_csv(args.fer_csv)
    prior_root = Path(args.prior_root).resolve()
    if not prior_root.is_dir():
        raise FileNotFoundError(f"Prior root not found: {prior_root}")
    schema_path = prior_root / "prior_schema.json"
    if not schema_path.is_file() or PRIOR_SCHEMA_ID not in schema_path.read_text(encoding="utf-8"):
        raise ValueError(f"Expected prior schema {PRIOR_SCHEMA_ID!r} at {schema_path}")
    for split in ("train", "val", "test"):
        if not (prior_root / split).is_dir():
            raise FileNotFoundError(f"Prior split missing: {prior_root / split}")
    if getattr(args, "cache_root", None):
        cache_root = Path(args.cache_root).resolve()
        if not cache_root.is_dir():
            raise FileNotFoundError(f"Cache root not found: {cache_root}")
    return cfg, prior_root


def write_runtime_config(cfg: dict) -> Path:
    handle, name = tempfile.mkstemp(prefix="lap_gnn_cfg_", suffix=".yaml")
    os.close(handle)
    path = Path(name)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def refuse_output_collision(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing non-empty output directory (resume is disabled): {output_dir}"
        )
