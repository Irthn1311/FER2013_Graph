"""Prepare the locked OFIX7-mid five-seed replication package.

This script never trains. It derives every replication config from the locked
historical resolved config and writes immutable registration/provenance files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "outputs/d19_analysis/d19_historical_near65_candidate_forensics/17_primary_replication_candidate_lock.json"
LOCK_HASH_PATH = LOCK_PATH.with_suffix(".sha256")
CONFIG_DIR = ROOT / "configs/d16/final_replication"
PREFLIGHT_DIR = ROOT / "outputs/d16_analysis/ofix7_mid_replication_preflight"
RUN_ROOT = ROOT / "outputs/d16_final_replication/ofix7_mid_5seed"
ANALYSIS_ROOT = ROOT / "outputs/d16_analysis/ofix7_mid_5seed_replication_analysis"
REGISTRATION_PATH = PREFLIGHT_DIR / "14_replication_registration.json"
REGISTRATION_HASH_PATH = PREFLIGHT_DIR / "14_replication_registration.sha256"
PORTABLE_LOCK_PATH = CONFIG_DIR / "candidate_lock.json"
PORTABLE_LOCK_HASH_PATH = CONFIG_DIR / "candidate_lock.sha256"
PORTABLE_REGISTRATION_PATH = CONFIG_DIR / "replication_registration.json"
PORTABLE_REGISTRATION_HASH_PATH = CONFIG_DIR / "replication_registration.sha256"
REGISTERED_SEEDS = [42, 1009, 1337, 777, 3407]
EXPECTED_RUN_ID = "d16r_a5b_ofix7_prior_drop_mid_seed42"
REGISTRATION_VERSION = "ofix7-mid-final-replication-v1"
PRIOR_SEED_OFFSET = 7741 - 42
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_locked_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], child))
    else:
        result[prefix] = value
    return result


def feature_order(prior_schema: dict[str, Any], config: dict[str, Any]) -> list[str]:
    graph = config.get("graph", {}) or {}
    detail = graph.get("detail_features", {}) or {}
    node = graph.get("node_features", {}) or {}
    names = ["intensity", "gx", "gy", "x_norm", "y_norm"]
    if bool(node.get("include_face_mask", True)):
        names.append("face_mask")
    if bool(node.get("include_part_soft", node.get("include_part_soft_masks", True))):
        names.extend(f"part_soft_{index}" for index in range(int(prior_schema["part_count"])))
    if bool(node.get("include_distance_maps", True)):
        names.extend(f"distance_map_{index}" for index in range(int(prior_schema["anchor_count"])))
    if bool(node.get("include_landmark_missing_flag", True)):
        names.append("landmark_missing_flag")
    if bool(detail.get("enabled", False)) and bool(detail.get("append_to_x", True)):
        names.extend(str(item) for item in detail.get("features", []))
    return names


def node_semantics(config: dict[str, Any]) -> dict[str, Any]:
    graph = config.get("graph", {}) or {}
    model = config.get("model", {}) or {}
    anchors = graph.get("anchor_nodes", {}) or {}
    micro = model.get("micro_motif_support", {}) or {}
    return {
        "pixel_nodes": "selected real FER pixels passing face mask plus context dilation",
        "context_pixel_nodes": f"real pixels added by {int(graph.get('context_pixels', 0))}-step binary dilation",
        "semantic_anchor_nodes": list(anchors.get("groups", [])),
        "semantic_anchor_count": len(anchors.get("groups", [])),
        "anchors_are_message_passing_nodes": bool(anchors.get("enabled", False)),
        "cls_token": bool(micro.get("use_cls_token", False)),
        "cls_token_scope": "micro_motif_support readout only; not inserted into graph edge_index",
        "motif_tokens_scope": "readout-only tokens derived after graph message passing",
        "graph_node_type_ids": {"pixel": 0, "semantic_anchor": 1},
        "readout_token_type_ids": "internal to MicroMotifSupportReadout; not graph node_type IDs",
    }


def graph_schema(config: dict[str, Any]) -> dict[str, Any]:
    graph = config.get("graph", {}) or {}
    edge = graph.get("edge_features", {}) or {}
    anchors = graph.get("anchor_nodes", {}) or {}
    return {
        "graph_mode": graph.get("graph_mode"),
        "face_threshold": graph.get("face_threshold"),
        "context_pixels": graph.get("context_pixels"),
        "pixel_edges": "directed 8-neighbor adjacency among selected pixel coordinates",
        "anchor_edges": {
            "bidirectional": anchors.get("bidirectional"),
            "anchor_to_anchor": anchors.get("anchor_to_anchor"),
            "connect_global_to_pixels": anchors.get("connect_global_to_pixels"),
            "threshold": anchors.get("connect_threshold"),
            "max_pixels_per_anchor": anchors.get("max_pixels_per_anchor"),
        },
        "edge_feature_order": list(edge.get("features", [])),
        "edge_attr_dim": len(edge.get("features", [])),
        "edge_type_ids": "not stored as a separate tensor; topology distinguishes pixel and anchor edges",
    }


def optimizer_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    return {
        "type": "AdamW",
        "lr": training.get("lr", 3e-4),
        "weight_decay": training.get("weight_decay", 1e-4),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
    }


def scheduler_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    scheduler = copy.deepcopy(training.get("scheduler", {}) or {})
    scheduler["step_location"] = "after checkpoint comparison, before last.pt save"
    return scheduler


def scientific_normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("run_name", None)
    normalized.pop("description", None)
    normalized["seed"] = "<RUN_SEED>"
    training = normalized.setdefault("training", {})
    training["seed"] = "<RUN_SEED>"
    training.pop("dual_validation_checkpoints", None)
    prior = (normalized.setdefault("graph", {})).get("prior_corruption", {}) or {}
    if "seed" in prior:
        prior["seed"] = "<RUN_SEED_PLUS_7699>"
    logging = normalized.get("logging", {}) or {}
    wandb = logging.get("wandb", {}) or {}
    for key in ("name", "group", "tags", "notes"):
        wandb.pop(key, None)
    return normalized


def classify_diff(path: str) -> tuple[str, bool]:
    if path in {"seed", "training.seed", "graph.prior_corruption.seed"}:
        return "SCIENTIFIC", True
    if path == "run_name" or path.startswith("logging.") or path == "description":
        return "NON_SCIENTIFIC", True
    if path.startswith("data.") and path.endswith(("prior_dir", "graph_cache_dir")):
        return "PATH_ONLY", True
    if path.startswith("training.dual_validation_checkpoints"):
        return "CHECKPOINT_INSTRUMENTATION", True
    return "SCIENTIFIC", False


def semantic_diff(source: dict[str, Any], replica: dict[str, Any]) -> list[dict[str, Any]]:
    left, right = flatten(source), flatten(replica)
    rows = []
    for field in sorted(set(left) | set(right)):
        if left.get(field) == right.get(field):
            continue
        classification, authorized = classify_diff(field)
        rows.append({
            "field": field,
            "historical_value": left.get(field),
            "replication_value": right.get(field),
            "classification": classification,
            "authorized": authorized,
        })
    return rows


def make_replication_config(source: dict[str, Any], seed: int) -> dict[str, Any]:
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"Unregistered seed: {seed}")
    cfg = copy.deepcopy(source)
    cfg["run_name"] = f"ofix7_mid_seed{seed}"
    cfg["seed"] = int(seed)
    cfg.setdefault("training", {})["seed"] = int(seed)
    prior = cfg.setdefault("graph", {}).setdefault("prior_corruption", {})
    prior["seed"] = int(seed + PRIOR_SEED_OFFSET)
    cfg["training"]["dual_validation_checkpoints"] = {
        "enabled": True,
        "preserve_best_macro_alias": True,
        "save_validation_predictions": True,
        "tie_break": "earliest_epoch_strict_improvement",
    }
    return cfg


def validate_replication_config(config_path: Path, registration: dict[str, Any] | None = None) -> dict[str, Any]:
    lock = load_json(LOCK_PATH)
    source = load_yaml(resolve_locked_path(lock["resolved_config_path"]))
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", -1))
    expected = make_replication_config(source, seed)
    diffs = semantic_diff(source, cfg)
    unauthorized = [row for row in diffs if not row["authorized"]]
    if cfg != expected:
        expected_flat, actual_flat = flatten(expected), flatten(cfg)
        mismatches = [key for key in sorted(set(expected_flat) | set(actual_flat)) if expected_flat.get(key) != actual_flat.get(key)]
    else:
        mismatches = []
    if registration is not None:
        registered = {Path(item).name for item in registration["replication_config_paths"]}
        if config_path.name not in registered:
            mismatches.append("config_not_in_registration")
        expected_file_hash = (registration.get("config_file_sha256", {}) or {}).get(config_path.name)
        if expected_file_hash is None or sha256_file(config_path) != expected_file_hash:
            mismatches.append("config_file_hash_not_registered")
    return {
        "seed": seed,
        "valid": seed in REGISTERED_SEEDS and not unauthorized and not mismatches,
        "unauthorized": unauthorized,
        "mismatches": mismatches,
        "diffs": diffs,
        "scientific_normalized_sha256": json_hash(scientific_normalized_config(cfg)),
        "file_sha256": sha256_file(config_path),
    }


def verify_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    if not LOCK_PATH.exists():
        raise FileNotFoundError(LOCK_PATH)
    lock = load_json(LOCK_PATH)
    if lock.get("run_id") != EXPECTED_RUN_ID:
        raise RuntimeError(f"Candidate lock identifies {lock.get('run_id')!r}, expected {EXPECTED_RUN_ID!r}")
    paths = {
        "run": resolve_locked_path(lock["canonical_run_path"]),
        "config": resolve_locked_path(lock["resolved_config_path"]),
        "best": resolve_locked_path(lock["checkpoint_path"]),
    }
    paths["last"] = paths["run"] / "checkpoints/last.pt"
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing locked {name}: {path}")
    actual = sha256_file(paths["best"])
    if actual != lock["checkpoint_sha256"]:
        raise RuntimeError(f"Historical best SHA mismatch: {actual} != {lock['checkpoint_sha256']}")
    if LOCK_HASH_PATH.exists() and sha256_file(LOCK_PATH) != LOCK_HASH_PATH.read_text(encoding="utf-8").strip():
        raise RuntimeError("Candidate lock file hash sidecar mismatch")
    return lock, paths


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns) + " |")
    return "\n".join(out)


def write_report(name: str, title: str, body: str) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_DIR / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def registration_payload(lock: dict[str, Any], paths: dict[str, Path], configs: list[Path], source: dict[str, Any]) -> dict[str, Any]:
    prior_schema_path = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue/prior_schema.json"
    prior_schema = load_json(prior_schema_path)
    features = feature_order(prior_schema, source)
    semantics = node_semantics(source)
    graph = graph_schema(source)
    config_rows = [validate_replication_config(path) for path in configs]
    return {
        "registration_version": REGISTRATION_VERSION,
        "locked_candidate_id": lock["run_id"],
        "candidate_lock_path": relative(LOCK_PATH),
        "candidate_lock_sha256": sha256_file(LOCK_PATH),
        "historical_run_path": relative(paths["run"]),
        "historical_config_path": relative(paths["config"]),
        "historical_best_checkpoint_path": relative(paths["best"]),
        "historical_best_checkpoint_sha256": lock["checkpoint_sha256"],
        "historical_last_checkpoint_path": relative(paths["last"]),
        "historical_seed": int(lock["seed"]),
        "registered_seeds": REGISTERED_SEEDS,
        "seed_reporting_order": REGISTERED_SEEDS,
        "prior_corruption_seed_rule": "run_seed + 7699; reproduces historical seed42 -> 7741",
        "replication_config_paths": [relative(path) for path in configs],
        "config_file_sha256": {path.name: sha256_file(path) for path in configs},
        "normalized_config_sha256": {path.name: row["scientific_normalized_sha256"] for path, row in zip(configs, config_rows)},
        "allowed_difference_policy": {
            "seed_fields": ["seed", "training.seed", "graph.prior_corruption.seed"],
            "non_scientific": ["run_name", "description", "logging metadata"],
            "path_only": ["runtime CLI dataset/cache/output paths after signature verification"],
            "checkpoint_instrumentation": ["training.dual_validation_checkpoints"],
        },
        "model_signature": lock["model_signature"],
        "graph_signature": lock["graph_signature"],
        "selector_signature": lock["selector_signature"],
        "feature_order": features,
        "feature_signature": json_hash(features),
        "node_type_semantics": semantics,
        "node_type_signature": json_hash(semantics),
        "graph_schema": graph,
        "prior_schema_path": relative(prior_schema_path),
        "prior_schema_sha256": sha256_file(prior_schema_path),
        "dataset_signature": lock["dataset_signature"],
        "split_signature": lock["split_signature"],
        "expected_split_counts": {"train": 28709, "val": 3589, "test": 3589},
        "class_order": CLASS_NAMES,
        "parameter_count": int(lock["parameter_count"]),
        "optimizer_signature": optimizer_signature(source),
        "scheduler_signature": scheduler_signature(source),
        "early_stopping_signature": source["training"]["early_stopping"],
        "historical_checkpoint_policy": {
            "monitor": source["training"]["checkpoint_monitor"],
            "mode": source["training"]["checkpoint_monitor_mode"],
            "tie_break": "earliest epoch because save occurs only on strict improvement",
            "test_selected": False,
        },
        "dual_checkpoint_policy": {
            "best.pt": "historical val_macro_f1 primary",
            "best_val_macro_f1.pt": "atomic byte copy of best.pt after historical save",
            "best_val_accuracy.pt": "strictly improving validation accuracy, earliest exact tie",
            "last.pt": "historical last checkpoint after scheduler step",
        },
        "checkpoint_policy_comparison_rules": {
            "mean_validation_accuracy_gain_pp_min": 0.50,
            "mean_validation_macro_f1_loss_pp_min": -0.50,
            "accuracy_seed_wins_min": 3,
            "mean_class_f1_loss_pp_max": 3.0,
            "macro_gap_increase_pp_max": 2.0,
            "accuracy_sd_increase_pp_max": 0.50,
            "default_when_any_gate_fails": "VAL_MACRO_F1",
        },
        "test_embargo_policy": "validation-lock refuses filenames/columns containing full-test artifacts; test reveal requires immutable policy lock",
        "future_limit_audit": {
            "status": "DEFERRED_UNTIL_FIVE_BASELINES_AND_POLICY_LOCK",
            "development_seeds": [42, 1009, 1337],
            "heldout_confirmation_seeds": [777, 3407],
            "maximum_alternatives": {"S1": "AdamW + CosineAnnealingLR", "O1": "RAdam + historical plateau"},
            "interaction_cell": "PROHIBITED",
        },
        "repository_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_registration(payload: dict[str, Any]) -> str:
    if REGISTRATION_PATH.exists():
        existing = load_json(REGISTRATION_PATH)
        comparable_old = copy.deepcopy(existing)
        comparable_new = copy.deepcopy(payload)
        comparable_new["created_at_utc"] = comparable_old.get("created_at_utc")
        if comparable_old == comparable_new:
            digest = sha256_file(REGISTRATION_PATH)
            REGISTRATION_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
            return digest
        if comparable_old != comparable_new:
            contaminated = RUN_ROOT.exists() and any(RUN_ROOT.iterdir())
            if contaminated:
                raise RuntimeError("Registration differs after replication output exists; immutable lock refused")
            payload = comparable_new
    REGISTRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRATION_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    digest = sha256_file(REGISTRATION_PATH)
    REGISTRATION_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return digest


def write_portable_registration_bundle() -> None:
    """Mirror immutable, small provenance files into the tracked config package."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PORTABLE_LOCK_PATH.write_bytes(LOCK_PATH.read_bytes())
    PORTABLE_LOCK_HASH_PATH.write_text(sha256_file(PORTABLE_LOCK_PATH) + "\n", encoding="utf-8")
    PORTABLE_REGISTRATION_PATH.write_bytes(REGISTRATION_PATH.read_bytes())
    PORTABLE_REGISTRATION_HASH_PATH.write_text(sha256_file(PORTABLE_REGISTRATION_PATH) + "\n", encoding="utf-8")


def command_reports(configs: list[Path]) -> None:
    ps = [
        "```powershell",
        "conda run -n fer-graph python -B d16/scripts/prepare_ofix7_mid_final_replication.py",
        "conda run -n fer-graph python -B d16/scripts/validate_ofix7_mid_final_replication.py --all --smoke --json-output outputs/d16_analysis/ofix7_mid_replication_preflight/21_validation_summary.json",
    ]
    sh = [
        "```bash",
        "python -B d16/scripts/prepare_ofix7_mid_final_replication.py",
        "python -B d16/scripts/validate_ofix7_mid_final_replication.py --all --smoke --json-output outputs/d16_analysis/ofix7_mid_replication_preflight/21_validation_summary.json",
    ]
    seed42_path = next(path for path in configs if path.stem.endswith("seed42"))
    remaining_paths = [path for path in configs if path != seed42_path]
    for path in [seed42_path]:
        seed = int(path.stem.split("seed")[-1])
        ps.extend([
            f"# seed {seed}",
            f"conda run -n fer-graph python -B d16/scripts/run_ofix7_mid_replication.py --config {relative(path)} --data-root <PRIOR_DIR> --output-root outputs/d16_final_replication/ofix7_mid_5seed --device cuda:0 --num-workers 2 --no-resume",
        ])
        sh.extend([
            f"# seed {seed}",
            f"python -B d16/scripts/run_ofix7_mid_replication.py --config {relative(path)} --data-root <KAGGLE_PRIOR_DIR> --output-root /kaggle/working/outputs/d16_final_replication/ofix7_mid_5seed --device cuda:0 --num-workers 2 --no-resume",
        ])
    ps.extend([
        "conda run -n fer-graph python -B d16/scripts/validate_ofix7_mid_final_replication.py --completed-seed42 outputs/d16_final_replication/ofix7_mid_5seed/ofix7_mid_seed42",
    ])
    sh.extend([
        "python -B d16/scripts/validate_ofix7_mid_final_replication.py --completed-seed42 /kaggle/working/outputs/d16_final_replication/ofix7_mid_5seed/ofix7_mid_seed42",
    ])
    for path in remaining_paths:
        seed = int(path.stem.split("seed")[-1])
        ps.append(f"conda run -n fer-graph python -B d16/scripts/run_ofix7_mid_replication.py --config {relative(path)} --data-root <PRIOR_DIR> --output-root outputs/d16_final_replication/ofix7_mid_5seed --device cuda:0 --num-workers 2 --no-resume")
        sh.append(f"python -B d16/scripts/run_ofix7_mid_replication.py --config {relative(path)} --data-root <KAGGLE_PRIOR_DIR> --output-root /kaggle/working/outputs/d16_final_replication/ofix7_mid_5seed --device cuda:0 --num-workers 2 --no-resume")
    ps.extend([
        "conda run -n fer-graph python -B d16/scripts/analyze_ofix7_mid_5seed.py --stage validation-lock --runs-root outputs/d16_final_replication/ofix7_mid_5seed --output-root outputs/d16_analysis/ofix7_mid_5seed_replication_analysis",
        "conda run -n fer-graph python -B d16/scripts/analyze_ofix7_mid_5seed.py --stage test-reveal --runs-root outputs/d16_final_replication/ofix7_mid_5seed --output-root outputs/d16_analysis/ofix7_mid_5seed_replication_analysis --checkpoint-policy-lock outputs/d16_analysis/ofix7_mid_5seed_replication_analysis/checkpoint_policy_lock.json --prior-dir <PRIOR_DIR> --device cuda:0",
        "```",
    ])
    sh.extend([
        "python -B d16/scripts/analyze_ofix7_mid_5seed.py --stage validation-lock --runs-root /kaggle/working/outputs/d16_final_replication/ofix7_mid_5seed --output-root /kaggle/working/outputs/d16_analysis/ofix7_mid_5seed_replication_analysis",
        "python -B d16/scripts/analyze_ofix7_mid_5seed.py --stage test-reveal --runs-root /kaggle/working/outputs/d16_final_replication/ofix7_mid_5seed --output-root /kaggle/working/outputs/d16_analysis/ofix7_mid_5seed_replication_analysis --checkpoint-policy-lock /kaggle/working/outputs/d16_analysis/ofix7_mid_5seed_replication_analysis/checkpoint_policy_lock.json --prior-dir <KAGGLE_PRIOR_DIR> --device cuda:0",
        "```",
    ])
    write_report("15_powershell_commands.md", "PowerShell Commands", "\n".join(ps))
    write_report("16_kaggle_linux_commands.md", "Kaggle And Linux Commands", "\n".join(sh))


def prepare() -> dict[str, Any]:
    lock, paths = verify_lock()
    source = load_yaml(paths["config"])
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = []
    diff_rows = []
    for seed in REGISTERED_SEEDS:
        path = CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml"
        cfg = make_replication_config(source, seed)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        configs.append(path)
        for row in semantic_diff(source, cfg):
            diff_rows.append({"config": relative(path), **row})
    validations = [validate_replication_config(path) for path in configs]
    unauthorized = [row for result in validations for row in result["unauthorized"]]
    if unauthorized or not all(result["valid"] for result in validations):
        raise RuntimeError(f"Unauthorized replication config differences: {unauthorized}")
    registration = registration_payload(lock, paths, configs, source)
    registration_sha = write_registration(registration)
    write_portable_registration_bundle()
    manifest_rows = [{
        "seed": result["seed"],
        "config_path": relative(path),
        "run_name": load_yaml(path)["run_name"],
        "future_output_dir": relative(RUN_ROOT / load_yaml(path)["run_name"]),
        "config_sha256": result["file_sha256"],
        "scientific_normalized_sha256": result["scientific_normalized_sha256"],
        "unauthorized_diff_count": len(result["unauthorized"]),
    } for path, result in zip(configs, validations)]
    write_csv(PREFLIGHT_DIR / "07_replication_config_manifest.csv", manifest_rows)
    write_csv(PREFLIGHT_DIR / "08_semantic_config_diffs.csv", diff_rows)
    write_report("00_README.md", "OFIX7-Mid Final Replication Preflight", f"Locked candidate: `{lock['run_id']}`.\n\nRegistration SHA-256: `{registration_sha}`.\n\nNo full training is launched by preparation.")
    write_report("01_candidate_lock_validation.md", "Candidate Lock Validation", f"Lock: `{relative(LOCK_PATH)}`\n\nLock SHA-256: `{sha256_file(LOCK_PATH)}`\n\nHistorical best SHA-256: `{sha256_file(paths['best'])}`\n\nCandidate ID and checkpoint hash match the immutable forensic lock.")
    write_report("02_historical_runtime_trace.md", "Historical Runtime Trace", "`D16PixelPriorDataset` -> `build_pixel_graph` -> `collate_d16_graphs` -> `D16Model.from_config` -> `PixelEncoder` -> three `EdgeContextGNN` layers -> `MicroMotifSupportReadout` -> `D16Classifier` -> CE -> AdamW -> ReduceLROnPlateau. Scheduler steps after validation checkpoint comparison; early stopping observes validation loss; best checkpoint observes validation macro-F1.")
    write_report("03_historical_artifact_validation.md", "Historical Artifact Validation", f"Run: `{relative(paths['run'])}`\n\nConfig: `{relative(paths['config'])}`\n\nBest: `{relative(paths['best'])}`\n\nLast: `{relative(paths['last'])}`\n\nStrict load and bounded forward are executed by the preflight validator.")
    write_report("04_architecture_and_node_semantics.md", "Architecture And Node Semantics", "```json\n" + json.dumps(registration["node_type_semantics"], indent=2) + "\n```\n\nSemantic anchors participate in graph message passing. CLS and motif tokens belong to the readout and are not FER pixels or graph-edge nodes.")
    write_report("05_feature_and_graph_schema.md", "Feature And Graph Schema", "Ordered 37-feature schema:\n\n" + "\n".join(f"{index + 1}. `{name}`" for index, name in enumerate(registration["feature_order"])) + "\n\nGraph schema:\n\n```json\n" + json.dumps(registration["graph_schema"], indent=2) + "\n```")
    rng_rows = [
        {"component": "Python random", "seed_source": "run seed via set_seed", "status": "DERIVED"},
        {"component": "NumPy global", "seed_source": "run seed via set_seed", "status": "DERIVED"},
        {"component": "Torch CPU/CUDA/all CUDA devices", "seed_source": "run seed via set_seed", "status": "DERIVED"},
        {"component": "DataLoader shuffle/base worker seeds", "seed_source": "Torch global generator initialized from run seed", "status": "DERIVED_INDIRECTLY"},
        {"component": "prior corruption", "seed_source": "run seed + 7699, then epoch/index mixing", "status": "DERIVED"},
        {"component": "dropout", "seed_source": "Torch RNG", "status": "DERIVED"},
        {"component": "AMP/TF32 kernels", "seed_source": "not random but GPU kernels may be nondeterministic", "status": "WARNING"},
        {"component": "augmentation", "seed_source": "none configured in locked run", "status": "NOT_ACTIVE"},
        {"component": "graph sampling", "seed_source": "none; deterministic selector/edge builder", "status": "NOT_ACTIVE"},
    ]
    write_report("06_seed_and_rng_audit.md", "Seed And RNG Audit", markdown_table(rng_rows))
    write_report("07_replication_config_manifest.md", "Replication Config Manifest", markdown_table(manifest_rows))
    write_report("08_semantic_config_diffs.md", "Semantic Config Diffs", f"Unauthorized scientific differences: **0**.\n\n{markdown_table(diff_rows)}")
    write_report("09_dual_checkpoint_design.md", "Dual Checkpoint Design", "Instrumentation is disabled by default. Registered configs enable it. `best.pt` remains historical val-macro-F1. `best_val_macro_f1.pt` is copied atomically after `best.pt`; `best_val_accuracy.pt` observes existing validation accuracy with strict-improvement/earliest-tie behavior. Scheduler and early stopping are unchanged.")
    write_report("10_checkpoint_rng_neutrality.md", "Checkpoint RNG Neutrality", "The validator performs two CPU trajectories with identical seeds and batches, comparing model, optimizer, scheduler, gradients, logits and RNG state. Final PASS/HOLD is written after smoke validation.")
    write_report("11_runner_design.md", "Thin Runner Design", "The runner validates lock, registration, seed, semantic config parity, clean output and no-resume, then invokes the existing `d16/training/train_d16.py`. It contains no copied model, dataset, graph or training loop.")
    write_report("12_analyzer_and_test_embargo.md", "Analyzer And Test Embargo", "`validation-lock` accepts only training/validation artifacts and rejects test-named files. `test-reveal` requires a policy lock whose SHA-256 sidecar matches before invoking selected-checkpoint evaluation.")
    write_report("13_preflight_smoke_validation.md", "Preflight Smoke Validation", "Pending execution by `validate_ofix7_mid_final_replication.py --all --smoke`. No epoch is permitted.")
    command_reports(configs)
    write_report("17_seed42_completion_gate.md", "Seed42 Completion Gate", "Fresh seed42 is authorized first. The completion validator checks registered config/signatures, no resume, complete history, finite metrics, all four checkpoints, canonical equality of `best.pt` and `best_val_macro_f1.pt`, and checkpoint epochs present in history. Accuracy does not gate the remaining seeds.")
    write_report("18_optimizer_scheduler_deferred_protocol.md", "Deferred Optimizer Scheduler Protocol", "Status: **DEFERRED_UNTIL_FIVE_BASELINES_AND_POLICY_LOCK**. Development seeds: 42, 1009, 1337. Held-out seeds: 777, 3407. At most S1 (AdamW + cosine) and O1 (RAdam + plateau). No combined cell and no executable variant configs are created now.")
    write_report("19_future_paper_and_release_plan.md", "Future Paper And Release Plan", "After final model lock, update architecture/node semantics, 37 features, graph/prior corruption, readout, five-seed results, checkpoint policy, bounded optimizer/scheduler audit, classwise/calibration/sensitivity and limitations. A clean release package is explicitly deferred; no release directory is created in this task.")
    summary = {
        "status": "PREPARED_AWAITING_STRICT_PREFLIGHT",
        "candidate": lock["run_id"],
        "historical_best_sha256": lock["checkpoint_sha256"],
        "registered_seeds": REGISTERED_SEEDS,
        "registration_sha256": registration_sha,
        "five_configs_created": True,
        "unauthorized_scientific_diff_count": 0,
        "full_training_launched": False,
    }
    (PREFLIGHT_DIR / "20_machine_readable_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(prepare(), indent=2))


if __name__ == "__main__":
    main()
