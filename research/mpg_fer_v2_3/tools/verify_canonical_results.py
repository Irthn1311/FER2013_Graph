"""Verify frozen compact evidence without loading checkpoints or FER2013 data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

SOURCE_SHA256 = "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
CONFIG_SHA256 = "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2"
CHECKPOINT_SHA256 = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
SEEDS = {0, 1, 42, 43, 123, 3047}
RUNTIME_SAFE_FIELDS = {"num_workers", "output_dir", "resume_path", "segment_number",
                       "segment_soft_limit_hours", "segment_safety_margin_minutes", "run_id"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(package_root: Path) -> dict:
    root = package_root.resolve()
    source = hashlib.sha256()
    for path in sorted((root / "src" / "mpg_fer_v2_3").glob("*.py")):
        source.update(path.name.encode("utf-8"))
        source.update(path.read_bytes())
    _require(source.hexdigest() == SOURCE_SHA256, "Canonical source SHA256 mismatch")
    config = json.loads((root / "v23_config.json").read_text(encoding="utf-8"))
    _require(set(config["runtime_safe_resume_fields"]) == RUNTIME_SAFE_FIELDS,
             "Registered runtime-safe config fields mismatch")
    scientific = {k: v for k, v in config["scientific_config"].items()
                  if k not in RUNTIME_SAFE_FIELDS | {"runtime_safe_resume_fields"}}
    config_hash = hashlib.sha256(json.dumps(
        scientific, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    _require(config_hash == config["config_sha256"] == CONFIG_SHA256,
             "Canonical config SHA256 mismatch")
    compact = root / "MPG_V23_MULTI_SEED_RESULTS"
    checksums = compact / "PACKAGE_SHA256SUMS.txt"
    seen = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*")
        path = (compact / relative).resolve()
        _require(path.is_relative_to(compact.resolve()), "Checksum path escapes compact package")
        _require(relative not in seen, "Duplicate checksum path")
        _require(path.is_file() and _sha(path) == digest,
                 f"Compact checksum mismatch: {relative}")
        seen.add(relative)
    _require(len(seen) == 223, "Expected 223 listed compact checksums")
    actual = {p.relative_to(compact).as_posix() for p in compact.rglob("*") if p.is_file()}
    _require(actual == seen | {"PACKAGE_SHA256SUMS.txt"}, "Unexpected compact package file set")
    with (compact / "multi_seed_registry.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 6 and {int(r["seed"]) for r in rows} == SEEDS,
             "Six-seed registry identity mismatch")
    for row in rows:
        _require(row["source_sha256"] == SOURCE_SHA256, "Registry source mismatch")
        _require(row["evaluation_precision"] == "canonical_fp32_no_autocast_tf32_disabled",
                 "Registry evaluation precision mismatch")
        _require(row["run_status"] == "TRAINING_COMPLETED" and bool(row["run_id"]),
                 "Registry execution identity missing")
        if int(row["seed"]) == 42:
            _require(row["checkpoint_sha256"] == CHECKPOINT_SHA256 and
                     row["config_sha256"] == CONFIG_SHA256, "Seed-42 frozen identity mismatch")
    aggregate = json.loads((compact / "aggregate_statistics.json").read_text(encoding="utf-8"))
    _require(aggregate["n"] == 6 and set(aggregate["seed_set"]) == SEEDS,
             "Aggregate seed identity mismatch")
    for metric, summary in aggregate["metrics"].items():
        values = [float(r[metric]) for r in rows]
        _require(abs(statistics.mean(values) - summary["mean"]) < 1e-12,
                 f"Aggregate mean mismatch: {metric}")
        _require(abs(statistics.stdev(values) - summary["sample_sd"]) < 1e-12,
                 f"Aggregate sample SD mismatch: {metric}")
        _require(all(float(r[metric]) == summary["values_by_seed"][r["seed"]] for r in rows),
                 f"Aggregate per-seed metric mismatch: {metric}")
    values = [float(r["private_tta_acc"]) * 100 for r in rows]
    return {"status": "PASS", "source_sha256": SOURCE_SHA256,
            "config_sha256": CONFIG_SHA256, "compact_checksums_pass": len(seen),
            "compact_files": len(actual), "seeds": sorted(SEEDS),
            "private_tta_mean_percent": statistics.mean(values),
            "private_tta_sample_sd_pp": statistics.stdev(values),
            "checkpoint_weights_or_dataset_loaded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.package_root), indent=2))


if __name__ == "__main__":
    main()
