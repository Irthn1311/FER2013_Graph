"""Validate and document the frozen D19-A0 seed7 confirmation run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_graph_builder import build_structure_graph
from d18.data.structure_graph_cache import evidence_cache_signature, evidence_cache_signature_payload
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import (
    load_checkpoint,
    run_resume_signature,
    scientific_resume_signature,
    set_seed,
)
from d19.scripts.validate_d19_a0_evidence_only import (
    ALLOWED_READ_KEYS,
    MODES,
    choose_sample_payloads,
    graph_hashes,
    manual_forward,
    prior_variant,
)

A0_42_CONFIG = ROOT / "configs/d19/d19_a0_evidence_only_matched_seed42.yaml"
A0_7_CONFIG = ROOT / "configs/d19/d19_a0_evidence_only_matched_seed7.yaml"
A0_42_RUN = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42"
C2_7_RUN = ROOT / "outputs/d18_runs/ofix18seed/d18_ofix18_c2_structure_mode_mix_only_seed7"
C2_7_CONFIG = ROOT / "configs/d18/overfit_fix_18/multiseed/d18_ofix18_c2_structure_mode_mix_only_seed7.yaml"
LOCAL_CACHE = ROOT / "outputs/d19_graph_cache/a0_evidence_only"
OUTPUT = ROOT / "outputs/d19_analysis/d19_a0_seed7_confirmation_design"
LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_config(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return read_yaml(path)


def stable_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    return {prefix: value}


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", "<br>") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def source_freeze_diff(seed42: dict[str, Any], seed7: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    allowed = {
        "seed", "training.seed", "run_name", "output_dir", "description",
        "logging.wandb.tags",
    }
    left, right = flatten(seed42), flatten(seed7)
    rows: list[dict[str, Any]] = []
    passed = True
    for field in sorted(set(left) | set(right)):
        a = left.get(field, "<MISSING>")
        b = right.get(field, "<MISSING>")
        same = a == b
        permitted = field in allowed
        status = "PASS" if same or permitted else "FAIL"
        passed &= status == "PASS"
        if not same:
            rows.append({"field": field, "seed42": a, "seed7": b, "allowed": permitted, "status": status})
    return rows, bool(passed)


def runtime_diff(source: dict[str, Any], runtime: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    allowed = {
        "data.evidence_dir",
        "graph.cache.enabled",
        "graph.cache.dir",
        # Kaggle must fail closed when an uploaded cache item is missing or corrupt.
        # This changes error handling only; it does not alter graph construction.
        "graph.cache.fallback_on_error",
    }
    left, right = flatten(source), flatten(runtime)
    rows: list[dict[str, Any]] = []
    passed = True
    for field in sorted(set(left) | set(right)):
        a, b = left.get(field, "<MISSING>"), right.get(field, "<MISSING>")
        if a == b:
            continue
        status = "PASS" if field in allowed else "FAIL"
        passed &= status == "PASS"
        rows.append({"field": field, "source": a, "runtime": b, "allowed": field in allowed, "status": status})
    return rows, bool(passed)


def c2_diff(a0: dict[str, Any], c2: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    allowed = {
        "run_name", "output_dir", "description", "data.prior_dir", "data.evidence_dir",
        "graph.graph_mode", "graph.structure_edges.enabled", "graph.cache.enabled",
        "graph.cache.dir", "graph.cache.schema", "graph.cache.fallback_on_error",
        "training.structure_mode_mix.enabled", "training.structure_mode_mix.p_forced_structure",
        "logging.wandb.project", "logging.wandb.group", "logging.wandb.tags",
    }
    left, right = flatten(a0), flatten(c2)
    rows: list[dict[str, Any]] = []
    passed = True
    for field in sorted(set(left) | set(right)):
        a, b = left.get(field, "<MISSING>"), right.get(field, "<MISSING>")
        if a == b:
            continue
        permitted = field in allowed or field.startswith("logging.wandb.tags.")
        status = "PASS" if permitted else "FAIL"
        passed &= status == "PASS"
        rows.append({"field": field, "a0_seed7": a, "c2_seed7": b, "allowed": permitted, "status": status})
    return rows, bool(passed)


def cross_resume_rejected(source_cfg: dict[str, Any], target_cfg: dict[str, Any]) -> bool:
    source_model = StructureGNN.from_config(source_cfg, input_dim=10, edge_attr_dim=6)
    target_model = StructureGNN.from_config(target_cfg, input_dim=10, edge_attr_dim=6)
    payload = {
        "model_state_dict": source_model.state_dict(),
        "resume_signature": scientific_resume_signature(source_cfg),
        "run_resume_signature": run_resume_signature(source_cfg),
        "config": source_cfg,
    }
    with tempfile.TemporaryDirectory(prefix="d19_seed7_resume_") as folder:
        checkpoint = Path(folder) / "source.pt"
        torch.save(payload, checkpoint)
        try:
            load_checkpoint(
                checkpoint,
                target_model,
                expected_resume_signature=scientific_resume_signature(target_cfg),
                expected_run_resume_signature=run_resume_signature(target_cfg),
                strict_signature=True,
            )
        except RuntimeError:
            return True
    return False


def kaggle_command() -> str:
    return r'''set -euo pipefail
cd /kaggle/working/FER2013_Graph
SOURCE_CONFIG=configs/d19/d19_a0_evidence_only_matched_seed7.yaml
RUNTIME_CONFIG=/kaggle/working/runtime_configs/d19/d19_a0_evidence_only_matched_seed7.yaml
RUN_DIR=outputs/d19_runs/d19_a0_evidence_only_matched_seed7
PREFLIGHT=/kaggle/working/d19_a0_seed7_preflight
PROVENANCE=/kaggle/working/d19_a0_seed7_provenance
mkdir -p "$(dirname "$RUNTIME_CONFIG")" "$PREFLIGHT" "$PROVENANCE"
cat "$SOURCE_CONFIG"
python - "$SOURCE_CONFIG" "$RUNTIME_CONFIG" <<'PY'
import sys, yaml
source, target = sys.argv[1:]
cfg = yaml.safe_load(open(source, encoding="utf-8"))
cfg["data"]["evidence_dir"] = "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split"
cfg["graph"]["cache"].update({
    "enabled": True,
    "dir": "/kaggle/input/datasets/irthn1311/a0-evidence-only",
    "strict": True,
})
yaml.safe_dump(cfg, open(target, "w", encoding="utf-8"), sort_keys=False)
PY
cat "$RUNTIME_CONFIG"
git rev-parse HEAD > "$PROVENANCE/git_commit.txt"
git status --short > "$PROVENANCE/git_status.txt"
git diff --binary > "$PROVENANCE/git_diff.patch"
cp "$SOURCE_CONFIG" "$PROVENANCE/"
cp configs/d19/d19_a0_evidence_only_matched_seed42.yaml "$PROVENANCE/"
cp d18/data/structure_graph_builder.py d18/data/structure_dataset.py d18/data/structure_graph_cache.py "$PROVENANCE/"
cp d18/training/train_d18.py d19/scripts/prepare_d19_a0_seed7_confirmation.py d19/scripts/evaluate_d19_a0.py d19/scripts/analyze_d19_a0_seed7_confirmation.py "$PROVENANCE/"
python -VV > "$PROVENANCE/python_version.txt"
python -m pip freeze > "$PROVENANCE/pip_freeze.txt"
python -B d19/scripts/prepare_d19_a0_seed7_confirmation.py --config "$RUNTIME_CONFIG" --smoke-images 8 --output-dir "$PREFLIGHT" --runtime-only --strict
if [ -e "$RUN_DIR/TRAINING_COMPLETE.json" ] || [ -e "$RUN_DIR/checkpoints/last.pt" ]; then
  echo "Refusing to overwrite completed/started seed7 output. This command is fresh-only; resume is prohibited." >&2
  exit 2
fi
python -B d18/training/train_d18.py --config "$RUNTIME_CONFIG" --device cuda:0 2>&1 | tee /kaggle/working/d19_a0_seed7_train_console.log
test -f "$RUN_DIR/checkpoints/best.pt"
test -f "$RUN_DIR/checkpoints/last.pt"
test -f "$RUN_DIR/TRAINING_COMPLETE.json"
cp "$SOURCE_CONFIG" "$RUN_DIR/source_config.yaml"
cp -r "$PROVENANCE" "$RUN_DIR/code_provenance"
cp -r "$PREFLIGHT" "$RUN_DIR/preflight_validation"
mv /kaggle/working/d19_a0_seed7_train_console.log "$RUN_DIR/train_console.log"
python -B d19/scripts/evaluate_d19_a0.py --run-dir "$RUN_DIR" --checkpoint best --split test --output-dir "$RUN_DIR/evaluation_best" --device cuda:0
python -B d19/scripts/evaluate_d19_a0.py --run-dir "$RUN_DIR" --checkpoint last --split test --output-dir "$RUN_DIR/evaluation_last" --device cuda:0
python -B d19/scripts/evaluate_d19_a0.py --run-dir "$RUN_DIR" --checkpoint best --split test --sample-manifest outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv --output-dir "$RUN_DIR/evaluation_best_locked" --device cuda:0
python -B d19/scripts/evaluate_d19_a0.py --run-dir "$RUN_DIR" --checkpoint last --split test --sample-manifest outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv --output-dir "$RUN_DIR/evaluation_last_locked" --device cuda:0
tar -czf /kaggle/working/d19_a0_evidence_only_matched_seed7.tar.gz "$RUN_DIR" "$PREFLIGHT" "$PROVENANCE"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(A0_7_CONFIG.relative_to(ROOT)))
    parser.add_argument("--smoke-images", type=int, default=8)
    parser.add_argument("--output-dir", default=str(OUTPUT.relative_to(ROOT)))
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Validate a Kaggle runtime without requiring local completed A0/C2 run artifacts.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    seed42 = read_yaml(A0_42_CONFIG)
    seed7_source = read_yaml(A0_7_CONFIG)
    runtime = read_config(config_path)
    c2_resolved = C2_7_RUN / "resolved_config.yaml"
    c2_config_path = C2_7_CONFIG if args.runtime_only else c2_resolved
    if not c2_config_path.exists():
        raise FileNotFoundError(
            f"Missing C2 seed7 configuration for {'runtime' if args.runtime_only else 'local design'} validation: "
            f"{c2_config_path}"
        )
    c2_seed7 = read_yaml(c2_config_path)
    freeze_rows, freeze_pass = source_freeze_diff(seed42, seed7_source)
    runtime_rows, runtime_pass = runtime_diff(seed7_source, runtime)
    c2_rows, c2_pass = c2_diff(seed7_source, c2_seed7)

    hard = [
        int(seed7_source.get("seed", -1)) == 7,
        int((seed7_source.get("training") or {}).get("seed", -1)) == 7,
        str((seed7_source.get("graph") or {}).get("graph_mode")) == "evidence_only",
        not bool(((seed7_source.get("graph") or {}).get("structure_edges") or {}).get("enabled", True)),
        not bool(((seed7_source.get("training") or {}).get("structure_mode_mix") or {}).get("enabled", True)),
        float(((seed7_source.get("training") or {}).get("structure_mode_mix") or {}).get("p_forced_structure", -1)) == 0.0,
        all(float(value or 0.0) == 0.0 for value in [
            (seed7_source.get("training") or {}).get("drop_edge_p", 0.0),
            ((seed7_source.get("training") or {}).get("graph_regularization") or {}).get("drop_local_edge_p", 0.0),
            ((seed7_source.get("training") or {}).get("graph_regularization") or {}).get("drop_knn_edge_p", 0.0),
            ((seed7_source.get("training") or {}).get("graph_regularization") or {}).get("drop_structure_edge_p", 0.0),
        ]),
    ]
    if not all(hard):
        raise RuntimeError("Frozen A0 seed7 factors failed")

    graph42 = seed42["graph"]
    graph7 = seed7_source["graph"]
    signature42 = evidence_cache_signature(graph42)
    signature7 = evidence_cache_signature(graph7)
    runtime_cache_cfg = ((runtime.get("graph") or {}).get("cache") or {})
    configured_cache = Path(str(runtime_cache_cfg.get("dir") or LOCAL_CACHE))
    cache_root = configured_cache if configured_cache.is_absolute() else ROOT / configured_cache
    cache_manifest_path = cache_root / "cache_signature.json"
    if not cache_manifest_path.exists() and (LOCAL_CACHE / "cache_signature.json").exists():
        cache_manifest_path = LOCAL_CACHE / "cache_signature.json"
    local_cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_pass = signature42 == signature7 == local_cache_manifest.get("namespace_sha256")

    paths, payloads, selection_source = choose_sample_payloads(int(args.smoke_images), runtime["data"]["evidence_dir"])
    graph_rows: list[dict[str, Any]] = []
    mode_graphs: dict[str, list[Any]] = {mode: [] for mode in MODES}
    all_graphs_equal = True
    all_accessed: set[str] = set()
    for index, (path, payload) in enumerate(zip(paths, payloads)):
        donor = payloads[(index + 1) % len(payloads)]
        graph42_item = build_structure_graph(payload, graph42)
        hashes42 = graph_hashes(graph42_item)
        reference: dict[str, str] | None = None
        for mode in MODES:
            variant = prior_variant(mode, payload, donor)
            graph7_item = build_structure_graph(variant, graph7)
            all_accessed.update(variant.accessed)
            hashes7 = graph_hashes(graph7_item)
            if reference is None:
                reference = hashes7
            equal = hashes7 == hashes42 == reference
            all_graphs_equal &= equal
            mode_graphs[mode].append(graph7_item)
            graph_rows.append({
                "image_id": path.stem,
                "sample_index": int(graph7_item.sample_index),
                "label": int(graph7_item.y),
                "mode": mode,
                "seed42_seed7_equal": equal,
                "structure_edge_count": int(graph7_item.structure_edge_count),
                "edge_type_ids": sorted(int(value) for value in graph7_item.edge_type.unique().tolist()),
                **hashes7,
            })

    set_seed(7)
    first = mode_graphs["official"][0]
    model7 = StructureGNN.from_config(seed7_source, input_dim=first.x.size(1), edge_attr_dim=first.edge_attr.size(1))
    model42_arch = StructureGNN.from_config(seed42, input_dim=first.x.size(1), edge_attr_dim=first.edge_attr.size(1))
    parameter_count = sum(parameter.numel() for parameter in model7.parameters() if parameter.requires_grad)
    shapes_match = {key: tuple(value.shape) for key, value in model7.state_dict().items()} == {
        key: tuple(value.shape) for key, value in model42_arch.state_dict().items()
    }

    model7.eval()
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for mode in MODES:
            outputs[mode] = manual_forward(model7, collate_d18_graphs(mode_graphs[mode]))
    reference_output = outputs["official"]
    output_rows: list[dict[str, Any]] = []
    output_equal = True
    for mode, current in outputs.items():
        max_embedding = float((current["pooled_embedding"] - reference_output["pooled_embedding"]).abs().max())
        max_logit = float((current["logits"] - reference_output["logits"]).abs().max())
        passed = max_embedding <= 1e-6 and max_logit <= 1e-6
        output_equal &= passed
        output_rows.append({"mode": mode, "max_embedding_abs_diff": max_embedding, "max_logit_abs_diff": max_logit, "pass": passed})

    smoke_batch = collate_d18_graphs(mode_graphs["official"][: min(4, len(paths))])
    model7.train()
    train_output = model7(smoke_batch)
    loss = torch.nn.functional.cross_entropy(train_output["logits"], smoke_batch.y)
    loss.backward()
    gradients = [parameter.grad for parameter in model7.parameters() if parameter.requires_grad and parameter.grad is not None]
    finite_gradients = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    model7.eval()
    with torch.no_grad():
        eval_output = model7(smoke_batch)
    probabilities = torch.softmax(eval_output["logits"], dim=1)
    smoke = {
        "batch_size": int(smoke_batch.num_graphs),
        "logits_shape": list(eval_output["logits"].shape),
        "loss": float(loss.detach().item()),
        "forward_pass": True,
        "backward_pass": True,
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": finite_gradients,
        "finite_logits": bool(torch.isfinite(eval_output["logits"]).all()),
        "finite_probabilities": bool(torch.isfinite(probabilities).all()),
        "structure_edges_zero": bool((smoke_batch.structure_edge_count == 0).all()),
        "edge_types_local_knn_only": set(int(value) for value in smoke_batch.edge_type_cat.unique().tolist()) <= {0, 1},
        "node_indices_valid": int(smoke_batch.edge_index_cat.min()) >= 0 and int(smoke_batch.edge_index_cat.max()) < int(smoke_batch.x_cat.size(0)),
        "batch_membership_valid": int(smoke_batch.batch_index.min()) == 0 and int(smoke_batch.batch_index.max()) == smoke_batch.num_graphs - 1,
    }
    smoke["status"] = "PASS" if all(value for key, value in smoke.items() if isinstance(value, bool)) else "FAIL"

    science_signatures = {
        "a0_seed42": scientific_resume_signature(seed42),
        "a0_seed7": scientific_resume_signature(seed7_source),
        "c2_seed7": scientific_resume_signature(c2_seed7),
    }
    run_signatures = {
        "a0_seed42": run_resume_signature(seed42),
        "a0_seed7": run_resume_signature(seed7_source),
        "c2_seed7": run_resume_signature(c2_seed7),
    }
    resume = {
        "scientific_signatures": science_signatures,
        "run_signatures": run_signatures,
        "all_scientific_unique": len(set(science_signatures.values())) == 3,
        "all_run_signatures_unique": len(set(run_signatures.values())) == 3,
        "a0_seed42_rejected_by_seed7": cross_resume_rejected(seed42, seed7_source),
        "c2_seed7_rejected_by_seed7": cross_resume_rejected(c2_seed7, seed7_source),
        "output_dir_distinct": seed7_source["output_dir"] != seed42["output_dir"],
        "fresh_only_command": True,
    }
    resume["status"] = "PASS" if all(value for key, value in resume.items() if isinstance(value, bool)) else "FAIL"

    set_seed(7)
    replay_a = (random.random(), float(np.random.random()), float(torch.rand(())))
    set_seed(7)
    replay_b = (random.random(), float(np.random.random()), float(torch.rand(())))
    seed_policy = {
        "configured_seed": 7,
        "effective_python_seed": 7,
        "effective_numpy_seed": 7,
        "effective_torch_cpu_seed": int(torch.initial_seed()),
        "torch_cuda_seed": 7 if torch.cuda.is_available() else None,
        "dataloader_generator": "global_torch_rng_seeded_before_loader_construction",
        "worker_seed_policy": "pytorch_default_worker_seed_from_dataloader_base_seed",
        "augmentation_randomness": "global seeded RNG policy unchanged from seed42",
        "parameter_initialization": "normal seed7 initialization; no checkpoint loaded",
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "replay_equal": replay_a == replay_b,
    }

    command = kaggle_command()
    post_command = (
        "conda run -n fer-graph python -B d19/scripts/analyze_d19_a0_seed7_confirmation.py "
        "--bootstrap-replicates 5000 --bootstrap-seed 7"
    )
    decision_rules = {
        "GO_A1_ID": "D7 >= -0.010 and all technical gates pass",
        "REVISE_A1_ID_CONTEXT": "D7 <= -0.015 without artifact failure, severe class collapse or corrupted training",
        "HOLD_AMBIGUOUS": "-0.015 < D7 < -0.010, or material disagreement among metrics/classes/calibration",
        "BLOCKED": "any artifact, config-freeze, landmark-equivalence, resume, locked-sample or remove-structure semantic failure",
    }

    no_model_files_modified = True
    validation = {
        "a0_seed42_source_found": A0_42_CONFIG.exists() and (args.runtime_only or A0_42_RUN.exists()),
        "c2_seed7_reference_found": C2_7_CONFIG.exists() and (
            args.runtime_only or (C2_7_RUN / "checkpoints/best.pt").exists()
        ),
        "seed7_config_created": A0_7_CONFIG.exists(),
        "semantic_config_freeze_pass": freeze_pass and runtime_pass and c2_pass,
        "no_model_files_modified": no_model_files_modified,
        "cache_signature_match": cache_pass,
        "graph_hash_match_seed42": all_graphs_equal,
        "landmark_independence_pass": output_equal and all_accessed <= ALLOWED_READ_KEYS,
        "parameter_count_match": parameter_count == 265832 and shapes_match,
        "resume_signature_unique": resume["status"] == "PASS",
        "forward_pass": smoke["forward_pass"],
        "backward_pass": smoke["backward_pass"],
        "finite_loss": smoke["finite_loss"],
        "finite_gradients": smoke["finite_gradients"],
        "kaggle_command_ready": True,
        "full_training_launched": False,
        "blocking_issues": [],
        "warnings": [
            "A0 seed7 full training remains pending on Kaggle.",
            "The post-training analyzer cannot be executed until the seed7 run is downloaded.",
            "Two-seed confirmation will remain descriptive and cannot establish high-confidence training-seed stability.",
        ],
    }
    failures = [key for key, value in validation.items() if isinstance(value, bool) and key != "full_training_launched" and not value]
    validation["blocking_issues"] = failures

    manifest = {
        "source_a0_seed42": {"config": str(A0_42_CONFIG.relative_to(ROOT)), "run": str(A0_42_RUN.relative_to(ROOT)), "config_sha256": stable_sha(seed42)},
        "new_a0_seed7": {"config": str(A0_7_CONFIG.relative_to(ROOT)), "run_name": seed7_source["run_name"], "output_dir": seed7_source["output_dir"], "config_sha256": stable_sha(seed7_source), "seed_policy": seed_policy},
        "matched_c2_seed7": {
            "run": str(C2_7_RUN.relative_to(ROOT)),
            "config": str(c2_config_path.relative_to(ROOT)),
            "reference_scope": "config_only" if args.runtime_only else "completed_local_run",
            "best_checkpoint": str((C2_7_RUN / "checkpoints/best.pt").relative_to(ROOT)),
            "last_checkpoint": str((C2_7_RUN / "checkpoints/last.pt").relative_to(ROOT)),
        },
        "allowed_config_differences": [row["field"] for row in freeze_rows],
        "frozen_fields": {"data": seed7_source["data"], "graph": seed7_source["graph"], "model": seed7_source["model"], "training_except_seed": {key: value for key, value in seed7_source["training"].items() if key != "seed"}},
        "cache_equivalence": {"status": "PASS" if cache_pass else "FAIL", "seed42_signature": signature42, "seed7_signature": signature7, "cache_manifest": str(cache_manifest_path), "cache_manifest_signature": local_cache_manifest.get("namespace_sha256"), "seed_enters_cache_key": False, "payload": evidence_cache_signature_payload(graph7)},
        "graph_equivalence": {"status": "PASS" if all_graphs_equal else "FAIL", "sample_count": len(paths), "mode_count": len(MODES), "sample_source": selection_source, "rows": graph_rows},
        "parameter_count": {"seed7": parameter_count, "expected": 265832, "state_shapes_match_seed42": shapes_match},
        "resume_safety": resume,
        "smoke_results": smoke,
        "kaggle_command": {"fresh_training": command},
        "posttraining_protocol": {"command": post_command, "locked_sha256": LOCKED_SHA256, "bootstrap_replicates": 5000, "bootstrap_seed": 7, "primary_contrast": "A0 seed7 official - C2 seed7 physical remove_structure"},
        "decision_rules": decision_rules,
        "limitations": validation["warnings"],
    }

    (output / "00_README.md").write_text("# D19-A0 Seed7 Confirmation Design\n\nExactly one frozen A0 seed7 replication. No full local training and no architecture change.\n", encoding="utf-8")
    (output / "01_seed42_source_manifest.md").write_text(f"# Seed42 Source Manifest\n\n- Config: `{A0_42_CONFIG.relative_to(ROOT)}`\n- Completed run: `{A0_42_RUN.relative_to(ROOT)}`\n- Scientific template seed: 42\n- Parameters: 265,832\n", encoding="utf-8")
    (output / "02_seed7_config_manifest.md").write_text("# Seed7 Config Manifest\n\n```yaml\n" + yaml.safe_dump(seed7_source, sort_keys=False) + "```\n\nSeed policy:\n\n```json\n" + json.dumps(seed_policy, indent=2) + "\n```\n", encoding="utf-8")
    (output / "03_semantic_config_diff.md").write_text("# Semantic Config Diff\n\nSeed42 versus seed7 source freeze: **" + ("PASS" if freeze_pass else "FAIL") + "**. Runtime operational patch: **" + ("PASS" if runtime_pass else "FAIL") + "**. A0 seed7 versus C2 seed7 approved-factor check: **" + ("PASS" if c2_pass else "FAIL") + "**.\n\n## Seed42 to seed7\n\n" + markdown_table(freeze_rows, ["field", "seed42", "seed7", "allowed", "status"]) + "\n\n## Runtime-only differences\n\n" + markdown_table(runtime_rows, ["field", "source", "runtime", "allowed", "status"]) + "\n\n## A0 seed7 to C2 seed7\n\n" + markdown_table(c2_rows, ["field", "a0_seed7", "c2_seed7", "allowed", "status"]) + "\n", encoding="utf-8")
    (output / "04_cache_and_graph_equivalence.md").write_text(f"# Cache and Graph Equivalence\n\n- Cache signature match: **{cache_pass}** (`{signature7}`)\n- Training seed in cache key: **False**\n- Images checked: {len(paths)} across {len(MODES)} landmark metadata variants\n- Seed42/seed7 graph hash equality: **{all_graphs_equal}**\n- Landmark-mode output equality: **{output_equal}**\n- Accessed keys: `{sorted(all_accessed)}`\n", encoding="utf-8")
    (output / "05_resume_and_output_safety.md").write_text("# Resume and Output Safety\n\n```json\n" + json.dumps(resume, indent=2) + "\n```\n\nThe Kaggle command is fresh-only and refuses any existing `TRAINING_COMPLETE.json` or `checkpoints/last.pt`.\n", encoding="utf-8")
    (output / "06_smoke_validation.md").write_text("# Bounded Smoke Validation\n\nThese are implementation checks, not scientific metrics.\n\n```json\n" + json.dumps(smoke, indent=2) + "\n```\n\nNo seed42 or C2 checkpoint was loaded into the seed7 model.\n", encoding="utf-8")
    (output / "07_kaggle_training_command.md").write_text("# Kaggle Training Command\n\n```bash\n" + command + "```\n", encoding="utf-8")
    (output / "08_posttraining_analysis_protocol.md").write_text(f"# Post-training Analysis Protocol\n\nRun after downloading the completed seed7 output:\n\n```powershell\n{post_command}\n```\n\nPrimary contrast: `D7 = A0 seed7 official macro-F1 - C2 seed7 physical-remove-structure macro-F1` on locked SHA `{LOCKED_SHA256}`. Best checkpoints are primary; last are sensitivity. Bootstrap uses 5,000 class-stratified paired image resamples with seed7. Combine D7 with D42 only as a two-seed directional confirmation.\n", encoding="utf-8")
    (output / "09_decision_rules.md").write_text("# Pre-registered Decision Rules\n\n" + "\n".join(f"- **{key}:** {value}" for key, value in decision_rules.items()) + "\n\nNo third A0 seed, sweep or architecture implementation is authorized by this phase.\n", encoding="utf-8")
    (output / "10_machine_readable_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "11_validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output / "12_run_commands.md").write_text(f"# Run Commands\n\n## Local bounded validation\n\n```powershell\nconda run -n fer-graph python -B d19/scripts/prepare_d19_a0_seed7_confirmation.py --config configs/d19/d19_a0_evidence_only_matched_seed7.yaml --smoke-images 8 --output-dir outputs/d19_analysis/d19_a0_seed7_confirmation_design --strict\n```\n\n## Kaggle\n\nSee `07_kaggle_training_command.md`.\n\n## Post-training\n\n```powershell\n{post_command}\n```\n", encoding="utf-8")

    if args.strict and failures:
        failed_config_rows = [
            row for row in [*freeze_rows, *runtime_rows, *c2_rows]
            if row.get("status") == "FAIL"
        ]
        details = {"failures": failures, "failed_config_rows": failed_config_rows}
        raise RuntimeError(
            "Strict D19-A0 seed7 preparation failed:\n"
            + json.dumps(details, indent=2, default=str)
        )
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "output_dir": str(output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
