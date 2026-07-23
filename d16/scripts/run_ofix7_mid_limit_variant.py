"""Run one registered OFIX7-mid bounded limit-audit cell."""
from __future__ import annotations

import argparse, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch, yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from d16.scripts.prepare_ofix7_mid_limit_audit import (
    ALL_SEEDS, BASELINE_LOCK, BASELINE_SHA, CONFIG_DIR, DEVELOPMENT_SEEDS,
    HELDOUT_SEEDS, POLICY_LOCK, POLICY_SHA, PREFLIGHT_DIR, REGISTRATION_HASH_PATH,
    REGISTRATION_PATH, RUN_ROOT, config_path, json_hash, load_json, load_yaml,
    make_variant_config, relative, run_name, semantic_diff, sha256_file,
)

# The conditional import syntax above is intentionally avoided at runtime by
# re-binding the expected constants from the preparation module.
from d16.scripts import prepare_ofix7_mid_limit_audit as prep

PORTABLE_REGISTRATION = CONFIG_DIR / "limit_audit_registration.json"
PORTABLE_REGISTRATION_SHA = CONFIG_DIR / "limit_audit_registration.sha256"
PORTABLE_POLICY_LOCK = CONFIG_DIR / "baseline_checkpoint_policy_lock.json"
PORTABLE_BASELINE_LOCK = CONFIG_DIR / "baseline_replication_lock.json"

ROOT_TEST_ARTIFACTS = {
    "test_metrics.csv", "last_test_metrics.csv", "predictions.csv", "last_predictions.csv",
    "confusion_matrix.csv", "confusion_matrix.png", "last_confusion_matrix.csv",
    "last_confusion_matrix.png", "last_per_class_metrics.csv",
    "last_pred_count.csv",
    "last_detected_vs_fallback_metrics.csv", "d16_report.md", "D16_V0_SMALL_TRAIN_REPORT.md",
}


def registration_bundle() -> tuple[Path, Path, Path, Path]:
    if REGISTRATION_PATH.exists() and REGISTRATION_HASH_PATH.exists():
        return REGISTRATION_PATH, REGISTRATION_HASH_PATH, POLICY_LOCK, BASELINE_LOCK
    return PORTABLE_REGISTRATION, PORTABLE_REGISTRATION_SHA, PORTABLE_POLICY_LOCK, PORTABLE_BASELINE_LOCK


def verify_registration() -> tuple[dict[str, Any], str]:
    registration_path, hash_path, policy_path, baseline_path = registration_bundle()
    for path in (registration_path, hash_path, policy_path, baseline_path):
        if not path.exists(): raise FileNotFoundError(path)
    actual = sha256_file(registration_path); expected = hash_path.read_text(encoding="utf-8-sig").strip()
    if actual != expected: raise RuntimeError("Limit-audit registration hash mismatch")
    if sha256_file(policy_path) != prep.POLICY_SHA: raise RuntimeError("Checkpoint-policy portable lock hash mismatch")
    if sha256_file(baseline_path) != prep.BASELINE_SHA: raise RuntimeError("Baseline portable lock hash mismatch")
    registration = load_json(registration_path)
    for source, expected_source_sha in (registration.get("implementation_source_sha256") or {}).items():
        source_path = ROOT / source
        if not source_path.exists() or sha256_file(source_path) != expected_source_sha:
            raise RuntimeError(f"Registered implementation source drift: {source}")
    if registration.get("selected_checkpoint_policy") != "VAL_MACRO_F1":
        raise RuntimeError("Registered checkpoint policy is not VAL_MACRO_F1")
    return registration, actual


def validate_cell_semantics(variant: str, cfg: dict[str, Any]) -> None:
    training = cfg.get("training") or {}
    if training.get("defer_test_evaluation") is not True:
        raise RuntimeError("Test embargo must be enabled")
    if training.get("checkpoint_monitor") != "val_macro_f1":
        raise RuntimeError("Checkpoint policy drift")
    if (training.get("early_stopping") or {}).get("metric") != "val_loss":
        raise RuntimeError("Early-stop policy drift")
    optimizer = training.get("optimizer") or {"type": "adamw"}
    scheduler = training.get("scheduler") or {}
    if variant == "S1":
        if str(optimizer.get("type", "adamw")).lower() != "adamw" or str(scheduler.get("type")).lower() != "cosine":
            raise RuntimeError("S1 must be AdamW plus cosine only")
    elif variant == "O1":
        if str(optimizer.get("type")).lower() != "radam" or str(scheduler.get("type")).lower() != "plateau":
            raise RuntimeError("O1 must be RAdam plus plateau only")
        if optimizer.get("decoupled_weight_decay") is not True:
            raise RuntimeError("BLOCKED_RADAM_WEIGHT_DECAY_SEMANTICS")
    else:
        raise RuntimeError("Unknown limit-audit variant")


def registered_config(variant: str, seed: int, registration: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if variant not in ("S1", "O1") or seed not in ALL_SEEDS: raise RuntimeError("Unregistered variant or seed")
    path = config_path(variant, seed)
    if not path.exists(): raise FileNotFoundError(path)
    key = relative(path)
    if key not in registration["config_paths"]: raise RuntimeError("Config path is not registered")
    if sha256_file(path) != registration["config_file_sha256"][key]: raise RuntimeError("Registered config file hash mismatch")
    cfg = load_yaml(path)
    if json_hash(cfg) != registration["normalized_config_sha256"][key]: raise RuntimeError("Registered normalized config hash mismatch")
    if cfg.get("run_name") != run_name(variant, seed): raise RuntimeError("Run name mismatch")
    if int(cfg.get("seed", -1)) != seed or int((cfg.get("training") or {}).get("seed", -1)) != seed:
        raise RuntimeError("Seed propagation mismatch")
    validate_cell_semantics(variant, cfg)

    return path, cfg


def refuse_resume(cfg: dict[str, Any], no_resume: bool) -> None:
    if not no_resume: raise RuntimeError("Limit-audit runner requires --no-resume")
    training = cfg.get("training") or {}
    for value in (cfg.get("resume"), training.get("resume"), cfg.get("init_checkpoint"), training.get("init_checkpoint")):
        if value not in (None, False, "", "null"): raise RuntimeError("Resume/init checkpoint is prohibited")


def refuse_contaminated_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()): raise RuntimeError(f"Output must be nonexistent or empty: {path}")


def test_artifacts(path: Path) -> list[str]:
    if not path.exists(): return []
    found = []
    for child in path.iterdir():
        if child.is_file() and (child.name in ROOT_TEST_ARTIFACTS or child.name.startswith("best_val_loss_")):
            found.append(child.name)
    summary = path / "d16_train_summary.json"
    if summary.exists():
        payload = load_json(summary)
        forbidden = {"test_accuracy", "test_macro_f1", "last_test_accuracy", "last_test_macro_f1", "final_test_checkpoint"}
        if forbidden.intersection(payload): found.append("d16_train_summary.json:test_result_fields")
        if payload.get("official_test_data_accessed") is not False: found.append("d16_train_summary.json:official_test_data_accessed")
    return sorted(found)


def verify_selection_lock(path: Path, registration_sha: str, variant: str, seed: int, registration: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.exists() or not sidecar.exists(): raise RuntimeError("Held-out seed requires development selection lock and SHA")
    if sha256_file(path) != sidecar.read_text(encoding="utf-8-sig").strip():
        raise RuntimeError("Development selection lock SHA mismatch")
    lock = load_json(path)
    if lock.get("registration_sha256") != registration_sha: raise RuntimeError("Selection lock registration mismatch")
    expected_decision = f"SELECT_{variant}_FOR_HELDOUT"
    if lock.get("decision") != expected_decision or lock.get("selected_variant") != variant:
        raise RuntimeError("Requested held-out cell is not the selected development winner")
    if seed not in HELDOUT_SEEDS: raise RuntimeError("Selection lock is only valid for registered held-out seeds")
    key = relative(config_path(variant, seed))
    if lock.get("heldout_config_sha256", {}).get(str(seed)) != registration["config_file_sha256"][key]:
        raise RuntimeError("Held-out config hash is not locked")
    return lock


def validate_data_root(path: Path) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(path)
    counts = {}
    for split, expected in (("train", 28709), ("val", 3589)):
        split_dir = path / split
        if not split_dir.exists(): raise FileNotFoundError(split_dir)
        counts[split] = sum(1 for _ in split_dir.glob("*.npz"))
        if counts[split] != expected: raise RuntimeError(f"{split} count mismatch: {counts[split]}")
    # Deliberately do not enumerate or read the official test split.
    return {"train_val_counts": counts, "official_test_split_touched": False}


def environment() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "hostname": platform.node(),
    }


def build_command(args: argparse.Namespace, cfg_path: Path, output: Path) -> list[str]:
    return [
        sys.executable, "-B", "d16/training/train_d16.py", "--config", str(cfg_path),
        "--prior_dir", str(args.data_root), "--output_dir", str(output),
        "--device", str(args.device), "--num_workers", str(args.num_workers), "--disable_graph_cache",
    ]


def validate_request(args: argparse.Namespace) -> dict[str, Any]:
    registration, registration_sha = verify_registration()
    cfg_path, cfg = registered_config(args.variant, args.seed, registration)
    refuse_resume(cfg, args.no_resume)
    output = args.output_root.resolve() / run_name(args.variant, args.seed)
    refuse_contaminated_output(output)
    if test_artifacts(output): raise RuntimeError("Development/held-out output contains test artifacts")
    if args.seed in HELDOUT_SEEDS:
        if args.development_selection_lock is None:
            raise RuntimeError("Held-out runner requires --development-selection-lock")
        verify_selection_lock(args.development_selection_lock, registration_sha, args.variant, args.seed, registration)
    elif args.development_selection_lock is not None:
        raise RuntimeError("Development seeds must not consume a held-out selection lock")
    data = validate_data_root(args.data_root)
    return {
        "registration": registration, "registration_sha": registration_sha, "config_path": cfg_path,
        "config": cfg, "output_dir": output, "data": data,
        "command": build_command(args, cfg_path, output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["S1", "O1"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--development-selection-lock", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    context = validate_request(args)
    if args.validate_only:
        print(json.dumps({k:v for k,v in context.items() if k not in {"registration","config"}}, indent=2, default=str))
        return
    output = context["output_dir"]; output.mkdir(parents=True)
    provenance = output / "limit_audit_provenance"; provenance.mkdir()
    (provenance/"registration.json").write_text(json.dumps(context["registration"], indent=2)+"\n", encoding="utf-8")
    (provenance/"environment.json").write_text(json.dumps(environment(), indent=2)+"\n", encoding="utf-8")
    (provenance/"request.json").write_text(json.dumps({
        "variant":args.variant, "seed":args.seed, "registration_sha256":context["registration_sha"],
        "config_sha256":sha256_file(context["config_path"]), "no_resume":True,
        "official_test_split_touched":False,
    }, indent=2)+"\n", encoding="utf-8")
    result = subprocess.run(context["command"], cwd=ROOT, env=os.environ.copy())
    if result.returncode: raise RuntimeError(f"Historical trainer exited {result.returncode}")
    missing = [name for name in ("best.pt","best_val_macro_f1.pt","best_val_accuracy.pt","last.pt")
               if not (output/"checkpoints"/name).exists()]
    if missing: raise RuntimeError(f"Missing checkpoints: {missing}")
    leaked = test_artifacts(output)
    if leaked: raise RuntimeError(f"Test embargo violated: {leaked}")
    from d16.training.train_d16 import canonical_model_state_hash
    best = torch.load(output/"checkpoints/best.pt",map_location="cpu",weights_only=False)
    alias = torch.load(output/"checkpoints/best_val_macro_f1.pt",map_location="cpu",weights_only=False)
    if canonical_model_state_hash(best) != canonical_model_state_hash(alias):
        raise RuntimeError("best.pt does not alias validation macro-F1 checkpoint")
    marker = {
        "status":"COMPLETE_VALIDATION_ONLY","variant":args.variant,"seed":args.seed,
        "registration_sha256":context["registration_sha"],"config_sha256":sha256_file(context["config_path"]),
        "test_evaluation_deferred":True,"resumed":False,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"LIMIT_AUDIT_COMPLETE.json").write_text(json.dumps(marker,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(marker,indent=2))


if __name__ == "__main__": main()

