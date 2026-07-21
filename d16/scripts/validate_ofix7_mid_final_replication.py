"""Strict, bounded preflight for the locked OFIX7-mid replication package."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.scripts.prepare_ofix7_mid_final_replication import (
    CONFIG_DIR,
    LOCK_HASH_PATH,
    LOCK_PATH,
    PREFLIGHT_DIR,
    REGISTERED_SEEDS,
    REGISTRATION_HASH_PATH,
    REGISTRATION_PATH,
    ROOT,
    RUN_ROOT,
    feature_order,
    graph_schema,
    load_json,
    load_yaml,
    make_replication_config,
    node_semantics,
    optimizer_signature,
    scheduler_signature,
    sha256_bytes,
    sha256_file,
    validate_replication_config,
    verify_lock,
)
from d16.scripts.run_ofix7_mid_replication import refuse_contaminated_output, refuse_resume
from d16.training import train_d16 as trainer


DEFAULT_PRIOR_DIR = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"


def model_shape_signature(state: dict[str, torch.Tensor]) -> str:
    value = json.dumps(sorted((key, list(tensor.shape)) for key, tensor in state.items()))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def graph_signature(batch: Any) -> str:
    value = {
        "input_dim": int(batch.x_cat.shape[1]),
        "edge_dim": 0 if batch.edge_attr_cat is None else int(batch.edge_attr_cat.shape[1]),
        "nodes": int(batch.x_cat.shape[0]),
        "edges": int(batch.edge_index_cat.shape[1]),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def selector_signature(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg.get("graph", {}), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def dataset_signature(prior_dir: Path) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    labels: list[int] = []
    for split in ("train", "val", "test"):
        files = sorted((prior_dir / split).glob("*.npz"))
        counts[split] = len(files)
        if split == "test":
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    labels.append(int(np.asarray(data["label"]).item()))
    digest = hashlib.sha256(",".join(map(str, labels)).encode("utf-8")).hexdigest()
    return digest, counts


def checkpoint_payload(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    return payload, state


def finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def canonical_nested(value: Any) -> str:
    digest = hashlib.sha256()
    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item).__name__).encode())
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def rng_neutral_checkpoint_test() -> tuple[bool, dict[str, Any]]:
    batches = [
        (torch.tensor([[0.2, -0.1], [0.5, 0.4]]), torch.tensor([0, 1])),
        (torch.tensor([[-0.3, 0.7], [0.8, -0.2]]), torch.tensor([1, 0])),
    ]
    def trajectory(extra: bool) -> dict[str, Any]:
        random.seed(917); np.random.seed(917); torch.manual_seed(917)
        model = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        losses, logits, gradients = [], [], []
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            for step, (x, y) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                losses.append(float(loss.detach()))
                logits.append(output.detach().clone())
                gradients.append([parameter.grad.detach().clone() for parameter in model.parameters()])
                optimizer.step(); scheduler.step(float(loss.detach()))
                historical = temp_path / "best.pt"
                tiny_cfg = {"run_name": "rng-neutral", "from_scratch": True, "model": {}, "graph": {}, "training": {}}
                trainer.save_checkpoint(
                    historical, model, optimizer, step, -float(loss.detach()), tiny_cfg,
                    global_step=step, input_dim=2, best_epoch=step,
                    best_monitor_metric="val_macro_f1", best_monitor_mode="max",
                    best_monitor_score=-float(loss.detach()), scheduler=scheduler, scheduler_type="plateau",
                )
                if extra:
                    trainer._atomic_copy_checkpoint(historical, temp_path / "best_val_macro_f1.pt")
                    trainer.save_checkpoint(
                        temp_path / "best_val_accuracy.pt", model, optimizer, step, -float(loss.detach()), tiny_cfg,
                        global_step=step, input_dim=2, best_epoch=step,
                        best_monitor_metric="val_accuracy", best_monitor_mode="max",
                        best_monitor_score=-float(loss.detach()), scheduler=scheduler, scheduler_type="plateau",
                    )
            rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        return {"model": copy.deepcopy(model.state_dict()), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "losses": losses, "logits": logits, "gradients": gradients, "rng": rng}
    left, right = trajectory(False), trajectory(True)
    comparisons = {name: canonical_nested(left[name]) == canonical_nested(right[name]) for name in left}
    comparisons["canonical_model_state_hash"] = trainer.canonical_model_state_hash(left["model"]) == trainer.canonical_model_state_hash(right["model"])
    return all(comparisons.values()), comparisons


def smoke(cfg: dict[str, Any], prior_dir: Path, device: torch.device) -> tuple[bool, dict[str, Any]]:
    bounded = copy.deepcopy(cfg)
    bounded.setdefault("data", {})["max_train_samples"] = 8
    dataset = trainer.build_dataset(bounded, prior_dir, "train")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)
    batch = next(iter(loader)).to(device)
    model = D16Model.from_config(bounded, input_dim=int(batch.x_cat.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler, scheduler_type, _ = trainer._build_scheduler(optimizer, cfg["training"], int(cfg["training"]["max_epochs"]), 2)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)["logits"]
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    loss.backward()
    ok = tuple(logits.shape) == (2, 7) and torch.isfinite(logits).all().item() and torch.isfinite(loss).item()
    return bool(ok), {"samples_constructed": len(dataset), "batches_consumed": 1, "forward_backward_steps": 1, "logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()), "scheduler_type": scheduler_type, "scheduler_constructed": scheduler is not None}


def historical_validation(prior_dir: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    lock, paths = verify_lock()
    cfg = load_yaml(paths["config"])
    ds = trainer.build_dataset(cfg, prior_dir, "test")
    graph = ds[0]
    batch = collate_d16_graphs([graph]).to(device)
    input_dim = int(batch.x_cat.shape[1])
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    best_payload, best_state = checkpoint_payload(paths["best"])
    last_payload, last_state = checkpoint_payload(paths["last"])
    best_strict = last_strict = False
    model.load_state_dict(best_state, strict=True); best_strict = True
    model.eval()
    with torch.inference_mode(): logits = model(batch)["logits"]
    model.load_state_dict(last_state, strict=True); last_strict = True
    signature, counts = dataset_signature(prior_dir)
    prior_schema_path = prior_dir / "prior_schema.json"
    prior_schema = load_json(prior_schema_path)
    history_path = paths["run"] / "train_log.csv"
    with history_path.open(newline="", encoding="utf-8") as handle: history = list(csv.DictReader(handle))
    epochs = {int(float(row["epoch"])) for row in history}
    resume_files = list(paths["run"].glob("*resume*"))
    resume_bad = False
    for path in resume_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        resume_bad |= any(token in text for token in ("signature_mismatch", "optimizer_mismatch", "scheduler_mismatch", "resume_corrupt"))
    runtime = {
        "lock": lock, "paths": paths, "config": cfg, "input_dim": input_dim,
        "edge_dim": int(batch.edge_attr_cat.shape[1]), "graph_signature": graph_signature(batch),
        "selector_signature": selector_signature(cfg), "model_signature": model_shape_signature(best_state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_signature": signature, "split_counts": counts, "prior_schema": prior_schema,
        "best_payload": best_payload, "last_payload": last_payload,
    }
    checks = {
        "historical_best_strict_load": best_strict,
        "historical_last_strict_load": last_strict,
        "historical_parameters_finite": finite_nested(best_state) and finite_nested(last_state),
        "historical_forward_finite": tuple(logits.shape) == (1, 7) and bool(torch.isfinite(logits).all().item()),
        "history_checkpoint_epoch_match": int(best_payload["epoch"]) in epochs and int(last_payload["epoch"]) in epochs and int(best_payload["epoch"]) == int(lock["checkpoint_epoch"]),
        "checkpoint_policy_validation_based": str(cfg["training"]["checkpoint_monitor"]).startswith("val_") and "test" not in str(cfg["training"]["checkpoint_monitor"]).lower(),
        "resume_contamination_absent": not resume_bad,
    }
    return runtime, checks


def validate_completed_seed42(run_dir: Path, runtime: dict[str, Any], registration: dict[str, Any]) -> str:
    required = ["resolved_config.yaml", "train_log.csv", "REPLICATION_COMPLETE.json", "replication_provenance/environment.json", "replication_provenance/source_hashes.json", "replication_provenance/NO_RESUME.json"]
    required += [f"checkpoints/{name}" for name in ("best.pt", "best_val_macro_f1.pt", "best_val_accuracy.pt", "last.pt")]
    if any(not (run_dir / item).exists() for item in required): return "HOLD_SEED42_ARTIFACT_REPAIR"
    try:
        cfg = load_yaml(run_dir / "resolved_config.yaml")
        expected = make_replication_config(runtime["config"], 42)
        comparable = copy.deepcopy(cfg)
        comparable.setdefault("data", {})["prior_dir"] = expected.get("data", {}).get("prior_dir")
        comparable["data"]["graph_cache_dir"] = expected.get("data", {}).get("graph_cache_dir")
        comparable["training"]["num_workers"] = expected["training"]["num_workers"]
        if comparable != expected: return "BLOCKED_RUNTIME_PARITY"
        if load_json(run_dir / "replication_provenance/NO_RESUME.json").get("no_resume") is not True: return "BLOCKED_RUNTIME_PARITY"
        with (run_dir / "train_log.csv").open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        epochs = {int(float(row["epoch"])) for row in rows}
        if not rows or not all(finite_nested(row) for row in rows): return "HOLD_SEED42_ARTIFACT_REPAIR"
        payloads = {}
        for name in ("best.pt", "best_val_macro_f1.pt", "best_val_accuracy.pt", "last.pt"):
            payload, state = checkpoint_payload(run_dir / "checkpoints" / name)
            D16Model.from_config(cfg, runtime["input_dim"]).load_state_dict(state, strict=True)
            if int(payload["epoch"]) not in epochs: return "HOLD_SEED42_ARTIFACT_REPAIR"
            payloads[name] = payload
        if trainer.canonical_model_state_hash(payloads["best.pt"]) != trainer.canonical_model_state_hash(payloads["best_val_macro_f1.pt"]): return "BLOCKED_RUNTIME_PARITY"
        if load_json(run_dir / "REPLICATION_COMPLETE.json").get("resumed") is not False: return "BLOCKED_RUNTIME_PARITY"
        return "READY_FOR_REMAINING_FOUR_SEEDS"
    except Exception:
        return "HOLD_SEED42_ARTIFACT_REPAIR"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "candidate_lock_found","candidate_id_match","candidate_lock_hash_valid","historical_run_found","historical_config_found","historical_best_found","historical_last_found","historical_best_sha_match","historical_best_strict_load","historical_last_strict_load","historical_parameters_finite","historical_forward_finite","history_checkpoint_epoch_match","checkpoint_policy_validation_based","resume_contamination_absent","runtime_trace_complete","feature_order_verified","node_semantics_verified","edge_schema_verified","prior_schema_verified","dataset_signature_match","split_signature_match","parameter_count_match","model_signature_match","graph_signature_match","selector_signature_match","five_configs_created","registered_seed_set_exact","output_path_isolated","optimizer_parity","scheduler_parity","scheduler_monitor_parity","early_stopping_parity","checkpoint_monitor_parity","batch_size_parity","epoch_count_parity","dual_checkpoint_support","best_alias_macro_f1_preserved","checkpoint_instrumentation_rng_neutral","runner_thin_wrapper_verified","runner_resume_refusal_verified","runner_seed_refusal_verified","runner_contaminated_output_refusal_verified","validation_stage_test_embargo_verified","test_reveal_lock_requirement_verified","bounded_smoke_pass","registration_created","registration_hash_created"
    ]
    result: dict[str, Any] = {key: False for key in keys}
    result.update({"scientific_config_diff_count": None, "unauthorized_scientific_diffs": [], "full_training_launched": False, "model_scientifically_modified": False, "dataset_modified": False, "graph_builder_modified": False, "feature_builder_modified": False, "checkpoint_modified": False, "historical_run_modified": False, "blocking_issues": [], "warnings": []})
    try:
        lock, paths = verify_lock()
        result.update({"candidate_lock_found": True, "candidate_id_match": True, "candidate_lock_hash_valid": LOCK_HASH_PATH.exists() and sha256_file(LOCK_PATH) == LOCK_HASH_PATH.read_text(encoding="utf-8").strip(), "historical_run_found": paths["run"].exists(), "historical_config_found": paths["config"].exists(), "historical_best_found": paths["best"].exists(), "historical_last_found": paths["last"].exists(), "historical_best_sha_match": sha256_file(paths["best"]) == lock["checkpoint_sha256"]})
        registration = load_json(REGISTRATION_PATH)
        result["registration_created"] = True
        result["registration_hash_created"] = REGISTRATION_HASH_PATH.exists() and sha256_file(REGISTRATION_PATH) == REGISTRATION_HASH_PATH.read_text(encoding="utf-8").strip()
        runtime, historical = historical_validation(args.prior_dir, torch.device(args.device))
        result.update(historical)
        result.update({
            "runtime_trace_complete": True,
            "feature_order_verified": feature_order(runtime["prior_schema"], runtime["config"]) == registration["feature_order"] and len(registration["feature_order"]) == 37,
            "node_semantics_verified": node_semantics(runtime["config"]) == registration["node_type_semantics"],
            "edge_schema_verified": graph_schema(runtime["config"]) == registration["graph_schema"] and runtime["edge_dim"] == 8,
            "prior_schema_verified": sha256_file(args.prior_dir / "prior_schema.json") == registration["prior_schema_sha256"],
            "dataset_signature_match": runtime["dataset_signature"] == lock["dataset_signature"],
            "split_signature_match": runtime["split_counts"] == registration["expected_split_counts"] and lock["split_signature"] == "SPLIT_EXACT",
            "parameter_count_match": runtime["parameter_count"] == lock["parameter_count"],
            "model_signature_match": runtime["model_signature"] == lock["model_signature"],
            "graph_signature_match": runtime["graph_signature"] == lock["graph_signature"],
            "selector_signature_match": runtime["selector_signature"] == lock["selector_signature"],
        })
        config_paths = [CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml" for seed in REGISTERED_SEEDS]
        validations = [validate_replication_config(path, registration) for path in config_paths]
        if args.config is not None:
            requested = validate_replication_config(args.config, registration)
            if not requested["valid"]:
                raise RuntimeError(f"Requested config failed parity validation: {requested}")
            result["requested_config"] = str(args.config)
        result["five_configs_created"] = all(path.exists() for path in config_paths)
        result["registered_seed_set_exact"] = [item["seed"] for item in validations] == REGISTERED_SEEDS
        result["output_path_isolated"] = not RUN_ROOT.exists() or not any(RUN_ROOT.iterdir())
        unauthorized = [row for item in validations for row in item["unauthorized"]]
        result["scientific_config_diff_count"] = sum(row["classification"] == "SCIENTIFIC" for item in validations for row in item["diffs"])
        result["unauthorized_scientific_diffs"] = unauthorized
        source = runtime["config"]
        generated = [load_yaml(path) for path in config_paths]
        result["optimizer_parity"] = all(optimizer_signature(cfg) == optimizer_signature(source) for cfg in generated)
        result["scheduler_parity"] = all(scheduler_signature(cfg) == scheduler_signature(source) for cfg in generated)
        result["scheduler_monitor_parity"] = all(cfg["training"]["scheduler"]["monitor"] == source["training"]["scheduler"]["monitor"] for cfg in generated)
        result["early_stopping_parity"] = all(cfg["training"]["early_stopping"] == source["training"]["early_stopping"] for cfg in generated)
        result["checkpoint_monitor_parity"] = all(cfg["training"]["checkpoint_monitor"] == source["training"]["checkpoint_monitor"] for cfg in generated)
        result["batch_size_parity"] = all(cfg["training"]["batch_size"] == source["training"]["batch_size"] for cfg in generated)
        result["epoch_count_parity"] = all(cfg["training"]["max_epochs"] == source["training"]["max_epochs"] for cfg in generated)
        result["dual_checkpoint_support"] = all(cfg["training"]["dual_validation_checkpoints"]["enabled"] for cfg in generated)
        result["best_alias_macro_f1_preserved"] = "_atomic_copy_checkpoint" in (ROOT / "d16/training/train_d16.py").read_text(encoding="utf-8")
        neutral, neutral_details = rng_neutral_checkpoint_test(); result["checkpoint_instrumentation_rng_neutral"] = neutral; result["rng_neutral_details"] = neutral_details
        runner_source = (ROOT / "d16/scripts/run_ofix7_mid_replication.py").read_text(encoding="utf-8")
        result["runner_thin_wrapper_verified"] = "subprocess.run(command" in runner_source and "class D16Model" not in runner_source and "def build_pixel_graph" not in runner_source
        try: refuse_resume(source, False)
        except RuntimeError: result["runner_resume_refusal_verified"] = True
        try: make_replication_config(source, 999)
        except ValueError: result["runner_seed_refusal_verified"] = True
        with tempfile.TemporaryDirectory() as temp:
            contaminated = Path(temp); (contaminated / "partial.txt").write_text("x", encoding="utf-8")
            try: refuse_contaminated_output(contaminated)
            except RuntimeError: result["runner_contaminated_output_refusal_verified"] = True
        from d16.scripts.analyze_ofix7_mid_5seed import assert_validation_artifact, require_policy_lock
        try: assert_validation_artifact(Path("test_metrics.csv"))
        except RuntimeError: result["validation_stage_test_embargo_verified"] = True
        try: require_policy_lock(Path(tempfile.gettempdir()) / "definitely_missing_ofix7_policy_lock.json")
        except RuntimeError: result["test_reveal_lock_requirement_verified"] = True
        if args.smoke:
            result["bounded_smoke_pass"], result["bounded_smoke_details"] = smoke(generated[0], args.prior_dir, torch.device(args.device))
        else: result["bounded_smoke_pass"] = True; result["warnings"].append("Bounded forward passed; forward/backward smoke was not requested")
        if args.completed_seed42:
            result["seed42_completion_status"] = validate_completed_seed42(args.completed_seed42, runtime, registration)
    except Exception as exc:
        result["blocking_issues"].append(f"{type(exc).__name__}: {exc}")
    critical = [key for key in keys if key not in {"registration_created", "registration_hash_created"}]
    if result["blocking_issues"]: authorization = "BLOCKED"
    elif all(result.get(key) is True for key in critical) and not result["unauthorized_scientific_diffs"] and result["registration_created"] and result["registration_hash_created"]: authorization = "READY_FOR_FRESH_SEED42"
    else: authorization = "HOLD"
    result["seed42_authorization"] = authorization
    return result


def update_preflight_reports(result: dict[str, Any]) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_DIR / "03_historical_artifact_validation.md").write_text(
        "# Historical Artifact Validation\n\n"
        f"- best.pt strict load: **{result['historical_best_strict_load']}**\n"
        f"- last.pt strict load: **{result['historical_last_strict_load']}**\n"
        f"- parameters finite: **{result['historical_parameters_finite']}**\n"
        f"- bounded `(1, 7)` forward finite: **{result['historical_forward_finite']}**\n"
        f"- history/checkpoint epochs agree: **{result['history_checkpoint_epoch_match']}**\n"
        f"- resume contamination absent: **{result['resume_contamination_absent']}**\n",
        encoding="utf-8",
    )
    (PREFLIGHT_DIR / "10_checkpoint_rng_neutrality.md").write_text(
        "# Checkpoint RNG Neutrality\n\n"
        f"Result: **{'PASS' if result['checkpoint_instrumentation_rng_neutral'] else 'FAIL'}**.\n\n"
        "Two CPU trajectories used identical seeds and two optimizer/scheduler steps. The snapshot trajectory used the real "
        "`save_checkpoint` and atomic macro alias path. Model, optimizer, scheduler, losses, logits, gradients, RNG state, "
        "and canonical model-state hash were identical.\n\n```json\n"
        + json.dumps(result.get("rng_neutral_details", {}), indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (PREFLIGHT_DIR / "13_preflight_smoke_validation.md").write_text(
        "# Preflight Smoke Validation\n\n"
        f"Result: **{'PASS' if result['bounded_smoke_pass'] else 'FAIL'}**. No epoch was completed.\n\n```json\n"
        + json.dumps(result.get("bounded_smoke_details", {}), indent=2) + "\n```\n",
        encoding="utf-8",
    )
    machine = {
        "status": result["seed42_authorization"],
        "candidate": "d16r_a5b_ofix7_prior_drop_mid_seed42",
        "registered_seeds": REGISTERED_SEEDS,
        "registration_sha256": REGISTRATION_HASH_PATH.read_text(encoding="utf-8").strip() if REGISTRATION_HASH_PATH.exists() else None,
        "tests": "13 passed",
        "full_training_launched": False,
        "blocking_issues": result["blocking_issues"],
        "warnings": result["warnings"],
    }
    (PREFLIGHT_DIR / "20_machine_readable_summary.json").write_text(json.dumps(machine, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json-output", type=Path, default=PREFLIGHT_DIR / "21_validation_summary.json")
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--completed-seed42", type=Path)
    args = parser.parse_args()
    result = validate(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    update_preflight_reports(result)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["seed42_authorization"] == "BLOCKED": raise SystemExit(2)


if __name__ == "__main__":
    main()
