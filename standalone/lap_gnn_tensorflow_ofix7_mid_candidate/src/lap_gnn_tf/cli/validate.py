"""Fail-closed package and data preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lap_gnn_tf.cli.compare_golden import compare
from lap_gnn_tf.config import load_config, validate_locked_config
from lap_gnn_tf.constants import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, PRIOR_SCHEMA_ID, SPLIT_COUNTS
from lap_gnn_tf.data.fer2013 import inspect_fer_csv
from lap_gnn_tf.signatures import scientific_payload_checksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fer-csv")
    parser.add_argument("--prior-root")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checks = validate_locked_config(config)
    result = {
        "config_checks": checks,
        "feature_dimension": len(NODE_FEATURE_NAMES),
        "edge_dimension": len(EDGE_FEATURE_NAMES),
    }
    expected_package = config["locked"]["package_checksum"]
    actual_package = scientific_payload_checksum(args.package_root)
    result["package_checksum"] = actual_package
    if expected_package != actual_package:
        raise ValueError(
            f"Scientific package checksum mismatch: {actual_package} != {expected_package}"
        )
    if args.fer_csv:
        result["fer_csv"] = inspect_fer_csv(args.fer_csv)
    if args.prior_root:
        root = Path(args.prior_root)
        schema = root / "prior_schema.json"
        if not schema.is_file() or PRIOR_SCHEMA_ID not in schema.read_text(encoding="utf-8"):
            raise ValueError(f"Prior schema mismatch: {schema}")
        counts = {split: len(list((root / split).glob("*.npz"))) for split in SPLIT_COUNTS}
        if counts != SPLIT_COUNTS:
            raise ValueError(f"FER split mismatch: {counts}, expected {SPLIT_COUNTS}")
        result["prior_counts"] = counts
    if args.require_gpu and not args.allow_cpu_training:
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("GPU required but TensorFlow sees no GPU")
    if args.golden:
        result["golden"] = compare(args.package_root)
        if not result["golden"]["pass"]:
            raise RuntimeError("Golden parity failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
