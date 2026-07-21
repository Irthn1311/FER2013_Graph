"""Run one registered OFIX7-mid replication through the historical trainer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
import numpy as np

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from d16.scripts.prepare_ofix7_mid_final_replication import (
    LOCK_PATH,
    PORTABLE_LOCK_HASH_PATH,
    PORTABLE_LOCK_PATH,
    PORTABLE_REGISTRATION_HASH_PATH,
    PORTABLE_REGISTRATION_PATH,
    REGISTRATION_HASH_PATH,
    REGISTRATION_PATH,
    ROOT,
    json_hash,
    load_json,
    load_yaml,
    normalized_text_sha256,
    relative,
    scientific_normalized_config,
    sha256_file,
    validate_replication_config,
    verify_lock,
)


def refuse_resume(config: dict[str, Any], no_resume: bool) -> None:
    training = config.get("training", {}) or {}
    if not no_resume:
        raise RuntimeError("Replication runner requires --no-resume")
    for value in (config.get("resume"), training.get("resume"), config.get("init_checkpoint"), training.get("init_checkpoint")):
        if value not in (None, False, "", "null"):
            raise RuntimeError("Resume/init checkpoint is prohibited for final replication")


def refuse_contaminated_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory must be nonexistent or empty: {path}")


def resolve_registration_bundle() -> tuple[Path, Path, Path, Path]:
    if REGISTRATION_PATH.exists() and REGISTRATION_HASH_PATH.exists():
        return REGISTRATION_PATH, REGISTRATION_HASH_PATH, LOCK_PATH, LOCK_PATH.with_suffix(".sha256")
    return PORTABLE_REGISTRATION_PATH, PORTABLE_REGISTRATION_HASH_PATH, PORTABLE_LOCK_PATH, PORTABLE_LOCK_HASH_PATH


def registration_identity_sha256(path: Path) -> str:
    if path.resolve() == PORTABLE_REGISTRATION_PATH.resolve():
        return normalized_text_sha256(path)
    return sha256_file(path)


def validate_registered_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registration_path, registration_hash_path, lock_path, lock_hash_path = resolve_registration_bundle()
    portable_mode = registration_path.resolve() == PORTABLE_REGISTRATION_PATH.resolve()
    if not registration_path.exists() or not registration_hash_path.exists():
        raise RuntimeError("Replication registration is missing; run preparation first")
    registration_hash = normalized_text_sha256(registration_path) if portable_mode else sha256_file(registration_path)
    if registration_hash != registration_hash_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("Replication registration hash mismatch")
    registration = load_json(registration_path)
    if not lock_path.exists() or not lock_hash_path.exists():
        raise RuntimeError("Portable candidate lock is missing")
    lock_hash = normalized_text_sha256(lock_path) if portable_mode else sha256_file(lock_path)
    if lock_hash != lock_hash_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("Candidate lock hash mismatch")
    lock = load_json(lock_path)
    semantic_lock_checks = {
        "run_id": lock.get("run_id") == registration.get("locked_candidate_id"),
        "checkpoint_sha256": lock.get("checkpoint_sha256") == registration.get("historical_best_checkpoint_sha256"),
        "seed": int(lock.get("seed", -1)) == int(registration.get("historical_seed", -2)),
        "model_signature": lock.get("model_signature") == registration.get("model_signature"),
        "graph_signature": lock.get("graph_signature") == registration.get("graph_signature"),
        "selector_signature": lock.get("selector_signature") == registration.get("selector_signature"),
        "dataset_signature": lock.get("dataset_signature") == registration.get("dataset_signature"),
        "split_signature": lock.get("split_signature") == registration.get("split_signature"),
        "parameter_count": int(lock.get("parameter_count", -1)) == int(registration.get("parameter_count", -2)),
    }
    failed_lock_checks = [name for name, passed in semantic_lock_checks.items() if not passed]
    if failed_lock_checks:
        raise RuntimeError(f"Candidate lock differs semantically from registration: {failed_lock_checks}")
    cfg = load_yaml(path)
    name = path.name
    registered_names = {Path(item).name for item in registration["replication_config_paths"]}
    seed = int(cfg.get("seed", -1))
    portable_valid = (
        name in registered_names
        and seed in registration["registered_seeds"]
        and json_hash(scientific_normalized_config(cfg)) == registration["normalized_config_sha256"].get(name)
    )
    if not portable_valid:
        raise RuntimeError("Config/scientific normalization is not registered")
    historical_paths_available = LOCK_PATH.exists() and (ROOT / str(lock.get("checkpoint_path", ""))).exists()
    if historical_paths_available:
        verify_lock()
        result = validate_replication_config(path, registration)
        if not result["valid"]:
            raise RuntimeError(f"Unregistered or scientifically changed config: {result}")
    if str((cfg.get("training", {}) or {}).get("checkpoint_monitor")) != "val_macro_f1":
        raise RuntimeError("Historical checkpoint monitor mismatch")
    return cfg, registration, registration_path

def environment_manifest() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "hostname": platform.node(),
    }


def validate_dataset_root(data_root: Path, registration: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    test_labels: list[int] = []
    for split in ("train", "val", "test"):
        files = sorted((data_root / split).glob("*.npz"))
        counts[split] = len(files)
        if split == "test":
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    test_labels.append(int(np.asarray(data["label"]).item()))
    label_sha = __import__("hashlib").sha256(",".join(map(str, test_labels)).encode("utf-8")).hexdigest()
    schema_path = data_root / "prior_schema.json"
    result = {
        "split_counts": counts,
        "dataset_signature": label_sha,
        "split_signature": "SPLIT_EXACT" if counts == registration["expected_split_counts"] else "SPLIT_MISMATCH",
        "prior_schema_sha256": sha256_file(schema_path) if schema_path.exists() else None,
    }
    if counts != registration["expected_split_counts"]:
        raise RuntimeError(f"Dataset split count mismatch: {counts}")
    if label_sha != registration["dataset_signature"]:
        raise RuntimeError("Dataset signature mismatch")
    if result["prior_schema_sha256"] != registration["prior_schema_sha256"]:
        raise RuntimeError("Prior schema signature mismatch")
    return result


def source_manifest(config_path: Path, registration: dict[str, Any], registration_path: Path) -> dict[str, Any]:
    lock_source_path = LOCK_PATH if LOCK_PATH.exists() else PORTABLE_LOCK_PATH
    files = [
        config_path,
        lock_source_path,
        registration_path,
        ROOT / "d16/training/train_d16.py",
        ROOT / "d16/data/pixel_prior_dataset.py",
        ROOT / "d16/data/graph_builder.py",
        ROOT / "d16/models/d16_model.py",
    ]
    return {
        "registration_sha256": registration_identity_sha256(registration_path),
        "repository_commit": registration.get("repository_commit"),
        "files": {relative(path): sha256_file(path) for path in files},
    }


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable, "-B", "d16/training/train_d16.py",
        "--config", str(args.config),
        "--prior_dir", str(args.data_root),
        "--output_dir", str(output_dir),
        "--device", str(args.device),
        "--num_workers", str(args.num_workers),
    ]
    if args.cache_root:
        command.extend(["--graph_cache_dir", str(args.cache_root)])
    else:
        command.append("--disable_graph_cache")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg, registration, registration_path = validate_registered_config(config_path)
    refuse_resume(cfg, args.no_resume)
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset/prior root not found: {args.data_root}")
    if args.cache_root is not None:
        raise RuntimeError("Locked run had no graph cache; --cache-root must be omitted for parity")
    dataset_manifest = validate_dataset_root(args.data_root, registration)
    run_name = str(cfg["run_name"])
    output_dir = args.output_root.resolve() / run_name
    refuse_contaminated_output(output_dir)
    command = build_command(args, output_dir)
    if args.validate_only:
        print(json.dumps({"valid": True, "run_name": run_name, "command": command}, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = output_dir / "replication_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "environment.json").write_text(json.dumps(environment_manifest(), indent=2) + "\n", encoding="utf-8")
    (provenance / "source_hashes.json").write_text(json.dumps(source_manifest(config_path, registration, registration_path), indent=2) + "\n", encoding="utf-8")
    (provenance / "registration.json").write_text(registration_path.read_text(encoding="utf-8"), encoding="utf-8")
    (provenance / "runtime_signatures.json").write_text(json.dumps({
        **dataset_manifest,
        "model_signature": registration["model_signature"],
        "graph_signature": registration["graph_signature"],
        "selector_signature": registration["selector_signature"],
        "feature_signature": registration["feature_signature"],
        "node_type_signature": registration["node_type_signature"],
    }, indent=2) + "\n", encoding="utf-8")
    (provenance / "NO_RESUME.json").write_text(json.dumps({"no_resume": True, "resume_from": None}, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"Historical trainer failed with exit code {result.returncode}")
    required = ["best.pt", "best_val_macro_f1.pt", "best_val_accuracy.pt", "last.pt"]
    missing = [name for name in required if not (output_dir / "checkpoints" / name).exists()]
    if missing:
        raise RuntimeError(f"Completed trainer is missing checkpoints: {missing}")
    import d16.training.train_d16 as trainer
    best = torch.load(output_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    alias = torch.load(output_dir / "checkpoints/best_val_macro_f1.pt", map_location="cpu", weights_only=False)
    if trainer.canonical_model_state_hash(best) != trainer.canonical_model_state_hash(alias):
        raise RuntimeError("best.pt and macro-F1 alias model states differ")
    marker = {
        "status": "COMPLETE",
        "run_name": run_name,
        "seed": int(cfg["seed"]),
        "registration_sha256": registration_identity_sha256(registration_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed": False,
    }
    (output_dir / "REPLICATION_COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
