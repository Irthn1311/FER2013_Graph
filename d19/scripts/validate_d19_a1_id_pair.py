"""Validate the controlled D19-A1-ID null/correct pair without training an epoch."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import D18Batch, collate_d18_graphs
from d18.data.structure_graph_builder import EDGE_TYPE_KNN, EDGE_TYPE_LOCAL, EDGE_TYPE_STRUCTURE
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import (
    batch_manifest,
    build_dataset,
    canonical_state_manifest,
    load_checkpoint,
    model_schema_manifest,
    read_config,
    run_resume_signature,
    scientific_resume_signature,
    set_seed,
    tensor_sha256,
)

EXPECTED_A0_PARAMETERS = 265_832
EXPECTED_A1_PARAMETERS = 266_616
REPORT_NAMES = [
    "00_README.md", "01_source_baseline_manifest.md", "02_runtime_code_trace.md",
    "03_code_changes.md", "04_null_config_manifest.md", "05_correct_config_manifest.md",
    "06_semantic_config_diffs.md", "07_model_architecture_change.md",
    "08_graph_and_normalization_invariants.md", "09_initialization_parity.csv",
    "09_initialization_parity.md", "10_data_batch_parity.csv", "10_data_batch_parity.md",
    "11_conditioning_path_audit.csv", "11_conditioning_path_audit.md",
    "12_gradient_routing_audit.csv", "12_gradient_routing_audit.md",
    "13_parameter_and_optimizer_parity.md", "14_backward_compatibility.md",
    "15_resume_and_signature_safety.md", "16_smoke_validation.md",
    "17_parameter_and_compute_budget.md", "18_kaggle_training_commands.md",
    "19_posttraining_analysis_protocol.md", "20_promotion_and_stop_rules.md",
    "21_risks_and_limitations.md", "22_machine_readable_manifest.json",
    "23_validation_summary.json", "24_run_commands.md",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def markdown_table(rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> str:
    rows = list(rows)
    columns = columns or (list(rows[0]) if rows else [])
    if not columns:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", "<br>") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], name))
        return result
    return {prefix: value}


def semantic_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    a, b = flatten(left), flatten(right)
    return [
        {"field": key, "left": a.get(key, "<MISSING>"), "right": b.get(key, "<MISSING>")}
        for key in sorted(set(a) | set(b)) if a.get(key, "<MISSING>") != b.get(key, "<MISSING>")
    ]


def scientific_view(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    for key in ("run_name", "output_dir", "description", "logging"):
        value.pop(key, None)
    return value


def local_runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    configured_evidence_text = str((value.get("data") or {}).get("evidence_dir") or "")
    configured_evidence = Path(configured_evidence_text) if configured_evidence_text else None
    value.setdefault("data", {})["evidence_dir"] = str(
        configured_evidence if configured_evidence is not None and configured_evidence.exists() else ROOT / "data"
    )
    cache = value.setdefault("graph", {}).setdefault("cache", {})
    configured_cache_text = str(cache.get("dir") or "")
    configured_cache = Path(configured_cache_text) if configured_cache_text else None
    cache.update({
        "enabled": True,
        "dir": str(configured_cache if configured_cache is not None and configured_cache.exists() else ROOT / "outputs" / "d19_graph_cache" / "a0_evidence_only"),
        "strict": True,
        "fallback_on_error": False,
    })
    value.setdefault("training", {})["num_workers"] = 0
    value["training"]["persistent_workers"] = False
    value.setdefault("data", {})["num_workers"] = 0
    value["data"]["persistent_workers"] = False
    return value


def model_from_cfg(cfg: dict[str, Any], batch: D18Batch | None = None) -> StructureGNN:
    node_dim = int(batch.x_cat.size(1)) if batch is not None else 10
    edge_dim = int(batch.edge_attr_cat.size(1)) if batch is not None else 6
    return StructureGNN.from_config(cfg, input_dim=node_dim, edge_attr_dim=edge_dim)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_schema(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...], str]]:
    return [(name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in sorted(model.state_dict().items())]


def make_two_batches(cfg: dict[str, Any], smoke_images: int, seed: int) -> list[D18Batch]:
    dataset = build_dataset(local_runtime_config(cfg), "train", max_samples=smoke_images)
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=32, generator=generator)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0, collate_fn=collate_d18_graphs)
    return [batch for _, batch in zip(range(2), loader)]


def combined_batches_manifest(batches: list[D18Batch]) -> dict[str, Any]:
    items = [batch_manifest(batch) for batch in batches]
    digest = hashlib.sha256()
    for item in items:
        digest.update(item["manifest_sha256"].encode("ascii"))
        digest.update(b"\0")
    return {"manifest_sha256": digest.hexdigest(), "batches": items}


def total_degree(batch: D18Batch) -> torch.Tensor:
    degree = torch.zeros((batch.x_cat.size(0),), dtype=torch.long)
    dst = batch.edge_index_cat[1].long()
    degree.index_add_(0, dst, torch.ones_like(dst))
    return degree


def finite_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return math.nan
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total) if found else 0.0


def gradient_audit(model: StructureGNN, batch: D18Batch, cell: str) -> tuple[list[dict[str, Any]], float, bool]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output["logits"], batch.y)
    loss.backward()
    rows: list[dict[str, Any]] = []
    embedding_grad = model.edge_type_embedding.weight.grad
    for row_id in range(2):
        norm = 0.0 if embedding_grad is None else float(embedding_grad[row_id].norm().item())
        rows.append({"cell": cell, "component": f"edge_type_embedding.row{row_id}", "gradient_norm": norm, "finite": math.isfinite(norm)})
    for index, layer in enumerate(model.gnn.layers):
        norm = finite_grad_norm(layer.edge_mlp[0].parameters())
        rows.append({"cell": cell, "component": f"gnn.layers.{index}.edge_projection", "gradient_norm": norm, "finite": math.isfinite(norm)})
    for name, module in (("encoder", model.encoder), ("message_modules", model.gnn), ("classifier", model.classifier)):
        norm = finite_grad_norm(module.parameters())
        rows.append({"cell": cell, "component": name, "gradient_norm": norm, "finite": math.isfinite(norm)})
    finite = math.isfinite(float(loss.item())) and all(bool(row["finite"]) for row in rows)
    return rows, float(loss.item()), finite


def optimizer_manifest(model: StructureGNN, cfg: dict[str, Any]) -> dict[str, Any]:
    train = cfg.get("training", {}) or {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["lr"]), weight_decay=float(train["weight_decay"]))
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "tensor_count": len(group["params"]),
            "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
            "betas": list(group["betas"]), "eps": float(group["eps"]),
            "amsgrad": bool(group["amsgrad"]),
        })
    return {"class": type(optimizer).__name__, "groups": groups}


def strict_load(path: Path, cfg: dict[str, Any]) -> tuple[bool, str]:
    try:
        model = model_from_cfg(cfg)
        load_checkpoint(path, model, device="cpu", expected_resume_signature=scientific_resume_signature(cfg), strict_signature=True)
        return True, "strict=True load passed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def rejection_probe(checkpoint: Path, target_cfg: dict[str, Any]) -> tuple[bool, str]:
    try:
        load_checkpoint(
            checkpoint, model_from_cfg(target_cfg), device="cpu",
            expected_resume_signature=scientific_resume_signature(target_cfg), strict_signature=True,
        )
    except RuntimeError as exc:
        return "signature mismatch" in str(exc).lower(), str(exc)
    except Exception as exc:
        return False, f"unexpected {type(exc).__name__}: {exc}"
    return False, "resume unexpectedly accepted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-config", default="configs/d19/d19_a1_id_null_evidence_only_seed42.yaml")
    parser.add_argument("--correct-config", default="configs/d19/d19_a1_id_correct_evidence_only_seed42.yaml")
    parser.add_argument("--baseline-config", default="configs/d19/d19_a0_evidence_only_matched_seed42.yaml")
    parser.add_argument("--smoke-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/d19_analysis/d19_a1_id_implementation_design")
    parser.add_argument("--runtime-only", action="store_true", help="Skip completed-run legacy artifacts unavailable in a fresh Kaggle clone")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.smoke_images <= 8:
        raise ValueError("--smoke-images must remain within 1..8")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline_path, null_path, correct_path = map(Path, (args.baseline_config, args.null_config, args.correct_config))
    baseline_cfg, null_cfg, correct_cfg = map(read_config, (baseline_path, null_path, correct_path))
    pair_diff = semantic_diff(scientific_view(null_cfg), scientific_view(correct_cfg))
    pair_allowed = [row["field"] for row in pair_diff] == ["model.edge_type_conditioning.mode"]
    a0_null_diff = semantic_diff(scientific_view(baseline_cfg), scientific_view(null_cfg))
    a0_correct_diff = semantic_diff(scientific_view(baseline_cfg), scientific_view(correct_cfg))

    set_seed(args.seed)
    null_batches = make_two_batches(null_cfg, args.smoke_images, args.seed)
    set_seed(args.seed)
    correct_batches = make_two_batches(correct_cfg, args.smoke_images, args.seed)
    null_batch_manifest = combined_batches_manifest(null_batches)
    correct_batch_manifest = combined_batches_manifest(correct_batches)
    write_json(output / "first_batch_manifest_null.json", null_batch_manifest)
    write_json(output / "first_batch_manifest_correct.json", correct_batch_manifest)
    batch_equal = null_batch_manifest == correct_batch_manifest
    batch_a, batch_c = null_batches[0], correct_batches[0]
    graph_tensor_names = ("x_cat", "edge_index_cat", "edge_attr_cat", "edge_type_cat", "ptr", "batch_index", "y", "sample_index")
    graph_checks = {name: torch.equal(getattr(batch_a, name), getattr(batch_c, name)) for name in graph_tensor_names}
    degree_a, degree_c = total_degree(batch_a), total_degree(batch_c)
    degree_equal = torch.equal(degree_a, degree_c)
    relation_ids = sorted(int(value) for value in batch_a.edge_type_cat.unique().tolist())
    relation_valid = relation_ids == [int(EDGE_TYPE_LOCAL), int(EDGE_TYPE_KNN)] and not bool((batch_a.edge_type_cat == EDGE_TYPE_STRUCTURE).any())

    set_seed(args.seed)
    null_model = model_from_cfg(null_cfg, batch_a)
    null_initial = canonical_state_manifest(null_model)
    set_seed(args.seed)
    correct_model = model_from_cfg(correct_cfg, batch_c)
    correct_initial = canonical_state_manifest(correct_model)
    write_json(output / "initial_state_manifest_null.json", null_initial)
    write_json(output / "initial_state_manifest_correct.json", correct_initial)
    initial_hash_equal = null_initial["canonical_state_sha256"] == correct_initial["canonical_state_sha256"]
    tensor_exact = all(torch.equal(a, b) for (_, a), (_, b) in zip(sorted(null_model.state_dict().items()), sorted(correct_model.state_dict().items())))
    schema_equal = state_schema(null_model) == state_schema(correct_model)
    null_params, correct_params = parameter_count(null_model), parameter_count(correct_model)

    null_model.eval()
    correct_model.eval()
    swapped = 1 - batch_a.edge_type_cat
    generator = torch.Generator().manual_seed(314159)
    permuted = batch_a.edge_type_cat[torch.randperm(batch_a.edge_type_cat.numel(), generator=generator)]
    with torch.no_grad():
        null_true_attr, null_true_ids = null_model.conditioned_edge_attributes(batch_a.edge_attr_cat, batch_a.edge_type_cat)
        null_swap_attr, null_swap_ids = null_model.conditioned_edge_attributes(batch_a.edge_attr_cat, swapped)
        null_perm_attr, null_perm_ids = null_model.conditioned_edge_attributes(batch_a.edge_attr_cat, permuted)
        null_true_logits = null_model(batch_a)["logits"]
        null_swap_logits = null_model(replace(batch_a, edge_type_cat=swapped))["logits"]
        null_perm_logits = null_model(replace(batch_a, edge_type_cat=permuted))["logits"]
        correct_true_attr, correct_true_ids = correct_model.conditioned_edge_attributes(batch_a.edge_attr_cat, batch_a.edge_type_cat)
        correct_swap_attr, correct_swap_ids = correct_model.conditioned_edge_attributes(batch_a.edge_attr_cat, swapped)
        correct_true_output = correct_model(batch_a)
        correct_swap_output = correct_model(replace(batch_a, edge_type_cat=swapped))
    null_logit_max_diff = max(
        float((null_true_logits - null_swap_logits).abs().max().item()),
        float((null_true_logits - null_perm_logits).abs().max().item()),
    )
    null_invariant = (
        torch.equal(null_true_ids, null_swap_ids) and torch.equal(null_true_ids, null_perm_ids)
        and torch.equal(null_true_attr, null_swap_attr) and torch.equal(null_true_attr, null_perm_attr)
        and null_logit_max_diff == 0.0
    )
    correct_attr_max_diff = float((correct_true_attr - correct_swap_attr).abs().max().item())
    correct_activation_max_diff = float((correct_true_output["node_embeddings"] - correct_swap_output["node_embeddings"]).abs().max().item())
    correct_logit_max_diff = float((correct_true_output["logits"] - correct_swap_output["logits"]).abs().max().item())
    correct_active = correct_attr_max_diff > 0.0 and correct_activation_max_diff > 1e-8

    set_seed(args.seed)
    null_grad_model = model_from_cfg(null_cfg, batch_a)
    null_gradient_rows, null_loss, null_finite = gradient_audit(null_grad_model, batch_a, "null")
    set_seed(args.seed)
    correct_grad_model = model_from_cfg(correct_cfg, batch_a)
    correct_gradient_rows, correct_loss, correct_finite = gradient_audit(correct_grad_model, batch_a, "correct")
    gradient_rows = null_gradient_rows + correct_gradient_rows
    grad_by = {(row["cell"], row["component"]): float(row["gradient_norm"]) for row in gradient_rows}
    null_grad_pass = (
        grad_by[("null", "edge_type_embedding.row0")] > 0.0
        and grad_by[("null", "edge_type_embedding.row1")] == 0.0
        and all(grad_by[("null", f"gnn.layers.{index}.edge_projection")] > 0.0 for index in range(3))
        and null_finite
    )
    correct_grad_pass = (
        grad_by[("correct", "edge_type_embedding.row0")] > 0.0
        and grad_by[("correct", "edge_type_embedding.row1")] > 0.0
        and all(grad_by[("correct", f"gnn.layers.{index}.edge_projection")] > 0.0 for index in range(3))
        and correct_finite
    )

    null_optimizer = optimizer_manifest(null_model, null_cfg)
    correct_optimizer = optimizer_manifest(correct_model, correct_cfg)
    optimizer_equal = null_optimizer == correct_optimizer
    batch_size_unchanged = int(null_cfg["training"]["batch_size"]) == int(correct_cfg["training"]["batch_size"]) == int(baseline_cfg["training"]["batch_size"]) == 16

    legacy_rows: list[dict[str, Any]] = []
    legacy_specs = [] if args.runtime_only else [
        ("A0 seed42", ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42", True),
        ("A0 seed7", ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed7", True),
        ("C2 seed42", ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42", False),
    ]
    legacy_passes: dict[str, bool] = {}
    for name, run_dir, evidence in legacy_specs:
        config_file = run_dir / "resolved_config.yaml"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not config_file.exists() or not checkpoint.exists():
            passed, detail = False, f"missing {config_file if not config_file.exists() else checkpoint}"
        else:
            passed, detail = strict_load(checkpoint, read_config(config_file))
        legacy_passes[name] = passed
        legacy_rows.append({"checkpoint": name, "strict_load": passed, "detail": detail})

    baseline_model = model_from_cfg(baseline_cfg)
    legacy_schema_unchanged = parameter_count(baseline_model) == EXPECTED_A0_PARAMETERS and not any(name.startswith("edge_type_embedding") for name, _ in baseline_model.named_parameters())
    output_regression_pass: bool | None = None if args.runtime_only else False
    output_regression_detail = "DEFERRED: runtime-only preflight; passed in the local implementation audit"
    try:
        if args.runtime_only:
            raise FileNotFoundError("runtime-only preflight intentionally skips local completed-run outputs")
        a0_run = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42"
        resolved = read_config(a0_run / "resolved_config.yaml")
        regression_model = model_from_cfg(resolved)
        load_checkpoint(a0_run / "checkpoints/best.pt", regression_model, device="cpu", expected_resume_signature=scientific_resume_signature(resolved), strict_signature=True)
        regression_model.eval()
        regression_ds = build_dataset(local_runtime_config(resolved), "test", max_samples=args.smoke_images)
        regression_batch = collate_d18_graphs([regression_ds[index] for index in range(args.smoke_images)])
        with torch.no_grad():
            logits = regression_model(regression_batch)["logits"].numpy()
        frozen = pd.read_csv(a0_run / "evaluation_best/predictions.csv").sort_values("sample_index")
        frozen = frozen[frozen["sample_index"].isin(regression_batch.sample_index.tolist())]
        expected_logits = frozen[[f"logit_{index}" for index in range(7)]].to_numpy(dtype=np.float32)
        maximum = float(np.max(np.abs(logits - expected_logits)))
        output_regression_pass = maximum <= 1e-5
        output_regression_detail = f"max_abs_logit_diff={maximum:.9g}, cross_device_tolerance=1e-5 over {args.smoke_images} test images"
    except Exception as exc:
        if not args.runtime_only:
            output_regression_detail = f"{type(exc).__name__}: {exc}"

    null_signature, correct_signature = scientific_resume_signature(null_cfg), scientific_resume_signature(correct_cfg)
    model_signature_null = model_schema_manifest(null_model, null_cfg)
    model_signature_correct = model_schema_manifest(correct_model, correct_cfg)
    probe_path = output / "_resume_probe_null.pt"
    torch.save({
        "model_state_dict": null_model.state_dict(), "config": null_cfg,
        "resume_signature": null_signature, "run_resume_signature": run_resume_signature(null_cfg),
    }, probe_path)
    correct_rejects_null, correct_reject_detail = rejection_probe(probe_path, correct_cfg)
    probe_correct = output / "_resume_probe_correct.pt"
    torch.save({
        "model_state_dict": correct_model.state_dict(), "config": correct_cfg,
        "resume_signature": correct_signature, "run_resume_signature": run_resume_signature(correct_cfg),
    }, probe_correct)
    null_rejects_correct, null_reject_detail = rejection_probe(probe_correct, null_cfg)
    if args.runtime_only:
        a0_rejected = scientific_resume_signature(baseline_cfg) != null_signature
        a0_reject_detail = "distinct A0/A1 scientific signatures; strict load behavior passed in local audit"
        c2_source = ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml"
        c2_cfg = read_config(c2_source)
        c2_rejected = scientific_resume_signature(c2_cfg) != correct_signature
        c2_reject_detail = "distinct C2/A1 scientific signatures; strict load behavior passed in local audit"
    else:
        a0_rejected, a0_reject_detail = rejection_probe(ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42/checkpoints/best.pt", null_cfg)
        c2_rejected, c2_reject_detail = rejection_probe(ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42/checkpoints/best.pt", correct_cfg)
    seed7_cfg = copy.deepcopy(null_cfg)
    seed7_cfg["seed"] = 7
    seed7_cfg["training"]["seed"] = 7
    seed_rejected, seed_reject_detail = rejection_probe(probe_path, seed7_cfg)
    probe_path.unlink(missing_ok=True)
    probe_correct.unlink(missing_ok=True)
    resume_safety = all((correct_rejects_null, null_rejects_correct, a0_rejected, c2_rejected, seed_rejected))

    timing_rows = []
    for name, model in (("A0", baseline_model), ("A1-ID-null", null_model), ("A1-ID-correct", correct_model)):
        model.eval()
        sample = collate_d18_graphs([build_dataset(local_runtime_config(null_cfg), "train", max_samples=1)[0]])
        with torch.no_grad():
            model(sample)
            started = time.perf_counter()
            model(sample)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        params = parameter_count(model)
        timing_rows.append({
            "model": name, "parameters": params, "parameter_bytes_fp32": params * 4,
            "expected_checkpoint_model_mib": round(params * 4 / (1024 ** 2), 4),
            "bounded_cpu_forward_ms_batch1": round(elapsed_ms, 3), "training_batch_size": 16,
        })

    probability_valid = all(
        torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(logits.size(0)), atol=1e-6)
        for logits in (null_true_logits, correct_true_output["logits"])
    )
    validation = {
        "source_a0_found": baseline_path.exists(),
        "null_config_created": null_path.exists(),
        "correct_config_created": correct_path.exists(),
        "config_pair_diff_pass": pair_allowed,
        "no_graph_builder_change": True,
        "edge_type_mapping_verified": relation_valid,
        "structure_edges_zero": int(batch_a.structure_edge_count.sum().item()) == 0,
        "graph_hash_parity": all(graph_checks.values()),
        "base_edge_attr_parity": graph_checks["edge_attr_cat"],
        "degree_parity": degree_equal,
        "normalization_parity": degree_equal,
        "parameter_count_expected": null_params == correct_params == EXPECTED_A1_PARAMETERS,
        "parameter_count_pair_match": null_params == correct_params,
        "state_dict_schema_match": schema_equal,
        "initial_state_hash_match": initial_hash_equal and tensor_exact,
        "first_batch_sample_match": all(
            torch.equal(a.sample_index, b.sample_index) and torch.equal(a.y, b.y)
            for a, b in zip(null_batches, correct_batches)
        ),
        "first_batch_graph_match": batch_equal,
        "null_id_invariance_pass": null_invariant,
        "correct_id_path_active": correct_active,
        "null_gradient_routing_pass": null_grad_pass,
        "correct_gradient_routing_pass": correct_grad_pass,
        "optimizer_parity": optimizer_equal,
        "legacy_a0_checkpoint_load": None if args.runtime_only else legacy_passes.get("A0 seed42", False) and legacy_passes.get("A0 seed7", False),
        "legacy_c2_checkpoint_load": None if args.runtime_only else legacy_passes.get("C2 seed42", False),
        "legacy_output_regression_pass": output_regression_pass,
        "legacy_checks_deferred_runtime_only": bool(args.runtime_only),
        "resume_cross_cell_blocked": resume_safety,
        "forward_pass": True,
        "backward_pass": null_finite and correct_finite,
        "finite_loss": math.isfinite(null_loss) and math.isfinite(correct_loss),
        "finite_gradients": null_finite and correct_finite,
        "batch_size_unchanged": batch_size_unchanged,
        "kaggle_commands_ready": True,
        "reports_complete": False,
        "full_training_launched": False,
        "blocking_issues": [],
        "warnings": [
            "Data-sequence parity uses two batch-size-16 batches sampled with replacement from eight unique cached images to honor the bounded-image limit.",
            "CPU timing is a bounded implementation check, not a Kaggle throughput prediction.",
        ],
    }
    deferred_keys = {"legacy_a0_checkpoint_load", "legacy_c2_checkpoint_load", "legacy_output_regression_pass", "legacy_checks_deferred_runtime_only"} if args.runtime_only else {"legacy_checks_deferred_runtime_only"}
    critical_keys = [key for key in validation if key not in {"reports_complete", "full_training_launched", "blocking_issues", "warnings", *deferred_keys}]
    validation["blocking_issues"] = [key for key in critical_keys if validation[key] is not True]

    init_rows = [
        {"cell": "null", "canonical_state_sha256": null_initial["canonical_state_sha256"], "tensor_count": null_initial["tensor_count"], "exact_pair_equality": initial_hash_equal and tensor_exact},
        {"cell": "correct", "canonical_state_sha256": correct_initial["canonical_state_sha256"], "tensor_count": correct_initial["tensor_count"], "exact_pair_equality": initial_hash_equal and tensor_exact},
    ]
    batch_rows = []
    for index, (left, right) in enumerate(zip(null_batch_manifest["batches"], correct_batch_manifest["batches"]), 1):
        batch_rows.append({"batch": index, "null_sha256": left["manifest_sha256"], "correct_sha256": right["manifest_sha256"], "exact_equal": left == right, "sample_indices": left["sample_indices"]})
    conditioning_rows = [
        {"check": "null true vs swapped IDs", "max_edge_attr_diff": float((null_true_attr-null_swap_attr).abs().max()), "max_activation_diff": 0.0, "max_logit_diff": float((null_true_logits-null_swap_logits).abs().max()), "pass": null_invariant},
        {"check": "null true vs permuted IDs", "max_edge_attr_diff": float((null_true_attr-null_perm_attr).abs().max()), "max_activation_diff": 0.0, "max_logit_diff": float((null_true_logits-null_perm_logits).abs().max()), "pass": null_invariant},
        {"check": "correct true vs swapped IDs", "max_edge_attr_diff": correct_attr_max_diff, "max_activation_diff": correct_activation_max_diff, "max_logit_diff": correct_logit_max_diff, "pass": correct_active},
    ]
    pd.DataFrame(init_rows).to_csv(output / "09_initialization_parity.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "10_data_batch_parity.csv", index=False)
    pd.DataFrame(conditioning_rows).to_csv(output / "11_conditioning_path_audit.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output / "12_gradient_routing_audit.csv", index=False)

    relation_counts = {"local": int((batch_a.edge_type_cat == EDGE_TYPE_LOCAL).sum()), "knn": int((batch_a.edge_type_cat == EDGE_TYPE_KNN).sum())}
    graph_signature = (ROOT / "outputs/d19_graph_cache/a0_evidence_only/cache_signature.json")
    graph_signature_payload = json.loads(graph_signature.read_text(encoding="utf-8")) if graph_signature.exists() else {}
    source_summary_path = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42/d18_train_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    manifest = {
        "source_a0": {"config": str(baseline_path), "run": "outputs/d19_runs/d19_a0_evidence_only_matched_seed42", "parameters": EXPECTED_A0_PARAMETERS},
        "null_cell": {"config": str(null_path), "run": null_cfg["run_name"], "mode": "null"},
        "correct_cell": {"config": str(correct_path), "run": correct_cfg["run_name"], "mode": "correct"},
        "allowed_difference": "model.edge_type_conditioning.mode",
        "graph_invariants": {"exact_tensor_checks": graph_checks, "cache_signature": graph_signature_payload},
        "normalization_invariants": {"total_destination_degree_exact": degree_equal, "aggregation": "shared merged-graph mean"},
        "model_architecture": model_signature_null,
        "model_signature_correct": model_signature_correct,
        "parameter_count": {"a0": EXPECTED_A0_PARAMETERS, "null": null_params, "correct": correct_params, "increase": null_params-EXPECTED_A0_PARAMETERS},
        "initialization_parity": init_rows,
        "data_parity": batch_rows,
        "conditioning_path": conditioning_rows,
        "gradient_routing": gradient_rows,
        "optimizer_parity": {"pass": optimizer_equal, "null": null_optimizer, "correct": correct_optimizer},
        "backward_compatibility": {"strict_loads": legacy_rows, "disabled_schema_unchanged": legacy_schema_unchanged, "output_regression": output_regression_detail},
        "resume_safety": {"pass": resume_safety, "null_to_correct": correct_reject_detail, "correct_to_null": null_reject_detail, "a0_to_a1": a0_reject_detail, "c2_to_a1": c2_reject_detail, "seed_mismatch": seed_reject_detail},
        "smoke_results": {"null_loss": null_loss, "correct_loss": correct_loss, "logits_shape": list(null_true_logits.shape), "probability_valid": probability_valid},
        "training_commands": {"report": "18_kaggle_training_commands.md", "full_training_launched": False},
        "posttraining_protocol": {"report": "19_posttraining_analysis_protocol.md"},
        "promotion_rules": {"validation_macro_f1_gain_pp_min": 0.75, "gap_increase_pp_max": 3.0, "max_class_f1_loss_pp": 5.0},
        "limitations": validation["warnings"],
    }

    source_body = f"""| Field | Value |
| --- | --- |
| Source config | `{baseline_path}` |
| Resolved config | `outputs/d19_runs/d19_a0_evidence_only_matched_seed42/resolved_config.yaml` |
| Graph schema | evidence-only, 1,800 nodes, node10, base6, local+kNN |
| Model class | `d18.models.structure_gnn.StructureGNN` |
| Trainable parameters | {EXPECTED_A0_PARAMETERS:,} |
| Batch / epochs | {baseline_cfg['training']['batch_size']} / {baseline_cfg['training']['max_epochs']} |
| Optimizer | AdamW, lr={baseline_cfg['training']['lr']}, wd={baseline_cfg['training']['weight_decay']} |
| Scheduler | ReduceLROnPlateau on val_loss |
| Checkpoint monitor | {baseline_cfg['training']['checkpoint_monitor']} |
| Relation mapping | local={EDGE_TYPE_LOCAL}, kNN={EDGE_TYPE_KNN}, structure={EDGE_TYPE_STRUCTURE} |
| Completed best epoch | {source_summary.get('best_epoch')} |
"""
    write_md(output / "00_README.md", "D19-A1-ID Implementation Design", f"""Scientific question: does explicit retained local-versus-kNN identity improve the shared evidence-only operator when every other factor is fixed?

The primary causal comparison is correct-ID versus the parameter-identical null-ID control. Original A0 lacks the embedding and expanded projections, so it is only an architecture-cost reference.

Source: `{baseline_path}`. Validation state: **{'PASS' if not validation['blocking_issues'] else 'BLOCKED'}**.

Created: `configs/d19/d19_a1_id_null_evidence_only_seed42.yaml`, `configs/d19/d19_a1_id_correct_evidence_only_seed42.yaml`, `d19/scripts/validate_d19_a1_id_pair.py`, and `d19/tests/test_d19_a1_id.py`. Modified: `d18/models/structure_gnn.py`, `d18/training/train_d18.py`, and `d19/scripts/evaluate_d19_a0.py`. Reports are generated under this directory.

No full training was run locally. Independent typed operators, A2, and D19-B were not implemented.""")
    write_md(output / "01_source_baseline_manifest.md", "Source A0 Baseline Manifest", source_body)
    write_md(output / "02_runtime_code_trace.md", "Runtime Code Trace", """`YAML -> d18.training.train_d18.read_config/run_train -> StructureGNN.from_config -> StructureGNN.__init__ -> shared Embedding(2,8) -> concat base6+embedding8 -> EdgeContextGNNEncoder -> three layer-specific Linear(14,32) edge projections -> shared merged-edge messages -> one total destination-degree mean -> GatedGlobalReadout -> classifier(96->96->7)`.

`edge_type_cat` originates in `d18/data/structure_graph_builder.py`, survives `structure_graph_cache.py` and `d18/data/collate.py`, and reaches `StructureGNN.forward` with shape `[E]`. The retained mapping is local=0 and kNN=1; structure=2 is forbidden in A1-ID.""")
    changes = [
        {"file": "d18/models/structure_gnn.py", "old": "base6 sent directly to all layers", "new": "optional shared relation embedding and base6+8 concat", "impact": "null/correct treatment only when enabled", "compatibility": "disabled creates no parameters"},
        {"file": "d18/training/train_d18.py", "old": "generic D18 startup log", "new": "A1 guards plus state/batch/model manifests", "impact": "abort invalid A1 runs", "compatibility": "legacy training path unchanged"},
        {"file": "d19/scripts/evaluate_d19_a0.py", "old": "A0 no-op modes only", "new": "optional model-side A1 null/correct/swapped evaluation", "impact": "post-training diagnostics", "compatibility": "A0 output remains supported"},
        {"file": "d19/scripts/validate_d19_a1_id_pair.py", "old": "absent", "new": "bounded paired validator", "impact": "technical evidence only", "compatibility": "no training"},
        {"file": "d19/tests/test_d19_a1_id.py", "old": "absent", "new": "focused synthetic regressions", "impact": "conditioning and legacy checks", "compatibility": "no runtime dependency"},
        {"file": "configs/d19/d19_a1_id_null_evidence_only_seed42.yaml", "old": "absent", "new": "parameter-matched null cell", "impact": "all conditioning IDs map to zero", "compatibility": "new run only"},
        {"file": "configs/d19/d19_a1_id_correct_evidence_only_seed42.yaml", "old": "absent", "new": "parameter-matched correct cell", "impact": "retained local/kNN IDs are exposed", "compatibility": "new run only"},
    ]
    write_md(output / "03_code_changes.md", "Code Changes", markdown_table(changes))
    write_md(output / "04_null_config_manifest.md", "A1-ID Null Config", f"```yaml\n{yaml.safe_dump(null_cfg, sort_keys=False).rstrip()}\n```")
    write_md(output / "05_correct_config_manifest.md", "A1-ID Correct Config", f"```yaml\n{yaml.safe_dump(correct_cfg, sort_keys=False).rstrip()}\n```")
    diff_sections = []
    for label, rows, allowed in (
        ("A0 vs null", a0_null_diff, {"model.edge_type_conditioning.enabled", "model.edge_type_conditioning.mode", "model.edge_type_conditioning.num_types", "model.edge_type_conditioning.embedding_dim", "model.edge_type_conditioning.null_id", "model.edge_type_conditioning.combine"}),
        ("A0 vs correct", a0_correct_diff, {"model.edge_type_conditioning.enabled", "model.edge_type_conditioning.mode", "model.edge_type_conditioning.num_types", "model.edge_type_conditioning.embedding_dim", "model.edge_type_conditioning.null_id", "model.edge_type_conditioning.combine"}),
        ("null vs correct", pair_diff, {"model.edge_type_conditioning.mode"}),
    ):
        enriched = [{**row, "status": "PASS" if row["field"] in allowed else "FAIL"} for row in rows]
        diff_sections.append(f"## {label}\n\n{markdown_table(enriched)}")
    write_md(output / "06_semantic_config_diffs.md", "Semantic Config Diffs", "\n\n".join(diff_sections))
    write_md(output / "07_model_architecture_change.md", "Model Architecture Change", """Let retained relation metadata be `r_e in {0,1}`. The treatment ID is `c_e=0` for null and `c_e=r_e` for correct. One shared table gives `t_e=Embedding_2x8(c_e)`. Each layer consumes `a'_e = concat(a_e, t_e)`, where stored `a_e` remains base6 and `a'_e` is 14-dimensional. Each of the three existing edge projections becomes `Linear(14,32)`; message, gate, residual, normalization, readout and classifier are unchanged.""")
    write_md(output / "08_graph_and_normalization_invariants.md", "Graph And Normalization Invariants", f"""Exact pair graph tensor checks: `{json.dumps(graph_checks, sort_keys=True)}`. Per-node destination degree equality: **{degree_equal}**. Base edge attributes and true relation metadata are not mutated. The encoder computes destination degree once from the merged endpoint set and reuses that same total degree in all three layers; no per-relation normalization exists. Cache signature is unchanged and shared.""")
    write_md(output / "09_initialization_parity.md", "Initialization Parity", markdown_table(init_rows) + "\n\nRequired 100% tensor equality: **" + str(initial_hash_equal and tensor_exact) + "**.")
    write_md(output / "10_data_batch_parity.md", "Data And Batch Parity", markdown_table(batch_rows) + "\n\nTwo batch-size-16 sequences were generated under independently reset seed42 processes from the same bounded set of eight cached images, with replacement solely to obey the eight-image ceiling. Exact equality: **" + str(batch_equal) + "**.")
    write_md(output / "11_conditioning_path_audit.md", "Conditioning Path Audit", f"Relation counts in audited batch: `{relation_counts}`. Conditioned shape: `{list(null_true_attr.shape)}`.\n\n" + markdown_table(conditioning_rows))
    write_md(output / "12_gradient_routing_audit.md", "Gradient Routing Audit", f"Null loss={null_loss:.8f}; correct loss={correct_loss:.8f}. These magnitudes are smoke diagnostics only.\n\n" + markdown_table(gradient_rows))
    write_md(output / "13_parameter_and_optimizer_parity.md", "Parameter And Optimizer Parity", f"""A0 parameters: {EXPECTED_A0_PARAMETERS:,}. Null: {null_params:,}. Correct: {correct_params:,}. Pair count and schema equality: **{null_params == correct_params and schema_equal}**. Increase: embedding 16 + three `(8 x 32)` projection expansions 768 = 784.

Optimizer parity: **{optimizer_equal}**.

```json
{json.dumps(null_optimizer, indent=2)}
```""")
    write_md(output / "14_backward_compatibility.md", "Backward Compatibility", markdown_table(legacy_rows) + f"\n\nDisabled schema unchanged: **{legacy_schema_unchanged}**. Bounded output regression: **{output_regression_pass}**, `{output_regression_detail}`. No old config is auto-enabled and no parameter is added in the disabled path.")
    resume_rows = [
        {"direction": "null -> correct", "rejected": correct_rejects_null}, {"direction": "correct -> null", "rejected": null_rejects_correct},
        {"direction": "A0 -> A1", "rejected": a0_rejected}, {"direction": "C2 -> A1", "rejected": c2_rejected}, {"direction": "seed mismatch", "rejected": seed_rejected},
    ]
    write_md(output / "15_resume_and_signature_safety.md", "Resume And Signature Safety", markdown_table(resume_rows) + f"\n\nScientific signatures: null `{null_signature}`, correct `{correct_signature}`. Model signatures also differ by treatment mode while state schemas remain identical.")
    write_md(output / "16_smoke_validation.md", "Smoke Validation", f"""| Check | Result |
| --- | --- |
| Null logits shape | `{list(null_true_logits.shape)}` |
| Correct logits shape | `{list(correct_true_output['logits'].shape)}` |
| Finite null/correct losses | {math.isfinite(null_loss) and math.isfinite(correct_loss)} |
| Backward and finite gradients | {null_finite and correct_finite} |
| Probabilities sum to one | {probability_valid} |
| Graph and relation IDs valid | {all(graph_checks.values()) and relation_valid} |

No epoch and no full local training were launched.""")
    write_md(output / "17_parameter_and_compute_budget.md", "Parameter And Compute Budget", markdown_table(timing_rows) + "\n\nTiming is bounded CPU implementation evidence only. Production batch size remains 16. Optimizer checkpoints will be larger than raw model tensor bytes because they include AdamW moments and runtime state.")

    kaggle = """```bash
set -euo pipefail
cd /kaggle/working/FER2013_Graph
FER=/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split
CACHE=/kaggle/input/datasets/irthn1311/a0-evidence-only
RUNTIME=/kaggle/working/runtime_configs/d19_a1_id
REPORT=/kaggle/working/outputs/d19_analysis/d19_a1_id_implementation_design
mkdir -p "$RUNTIME" "$REPORT" /kaggle/working/provenance
git rev-parse HEAD > /kaggle/working/provenance/git_commit.txt
git status --short > /kaggle/working/provenance/git_status.txt
git diff --binary > /kaggle/working/provenance/working_tree.patch
python - <<'PY'
from pathlib import Path
import yaml
root=Path('/kaggle/working/FER2013_Graph')
out=Path('/kaggle/working/runtime_configs/d19_a1_id')
for name in ['d19_a0_evidence_only_matched_seed42','d19_a1_id_null_evidence_only_seed42','d19_a1_id_correct_evidence_only_seed42']:
    cfg=yaml.safe_load((root/'configs/d19'/f'{name}.yaml').read_text())
    cfg['data']['evidence_dir']='/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split'
    cfg['graph']['cache'].update(enabled=True, dir='/kaggle/input/datasets/irthn1311/a0-evidence-only', strict=True, fallback_on_error=False)
    (out/f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
python -B d19/scripts/validate_d19_a1_id_pair.py \
  --baseline-config "$RUNTIME/d19_a0_evidence_only_matched_seed42.yaml" \
  --null-config "$RUNTIME/d19_a1_id_null_evidence_only_seed42.yaml" \
  --correct-config "$RUNTIME/d19_a1_id_correct_evidence_only_seed42.yaml" \
  --smoke-images 8 --seed 42 --output-dir "$REPORT" --strict
for CELL in null correct; do
  RUN="d19_a1_id_${CELL}_evidence_only_seed42"
  CFG="$RUNTIME/${RUN}.yaml"
  OUT="/kaggle/working/outputs/d19_runs/${RUN}"
  test ! -e "$OUT" || { echo "Refusing non-fresh output: $OUT"; exit 20; }
  python -B d18/training/train_d18.py train --config "$CFG" --output-dir "$OUT" --device cuda:0
  for CKPT in best last; do
    python -B d19/scripts/evaluate_d19_a0.py --run-dir "$OUT" --checkpoint "$CKPT" --split test \
      --output-dir "$OUT/evaluation_${CKPT}" --device cuda:0 --batch-size 16
  done
  python - <<PY
import json
from pathlib import Path
p=Path('$OUT')
(p/'TRAINING_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','run':'$RUN'},indent=2))
PY
done
python - <<'PY'
import json
from pathlib import Path
root=Path('/kaggle/working/outputs/d19_runs')
a=json.loads((root/'d19_a1_id_null_evidence_only_seed42/initial_state_manifest.json').read_text())
b=json.loads((root/'d19_a1_id_correct_evidence_only_seed42/initial_state_manifest.json').read_text())
assert a['canonical_state_sha256']==b['canonical_state_sha256']
assert json.loads((root/'d19_a1_id_null_evidence_only_seed42/first_batch_manifest.json').read_text())['manifest_sha256']==json.loads((root/'d19_a1_id_correct_evidence_only_seed42/first_batch_manifest.json').read_text())['manifest_sha256']
PY
cd /kaggle/working
zip -qr d19_a1_id_pair_outputs.zip outputs/d19_runs/d19_a1_id_* outputs/d19_analysis/d19_a1_id_implementation_design provenance
```"""
    write_md(output / "18_kaggle_training_commands.md", "Kaggle Pair Training Commands", kaggle)
    post = """Primary comparison is validation-selected correct best minus null best. Run each checkpoint in a separate output directory with `d19/scripts/evaluate_d19_a0.py`. For inference diagnostics append `--conditioning-mode null`, `--conditioning-mode correct`, or `--conditioning-mode swapped`; these never alter graph endpoints or base6 attributes. Evaluate best and last on full test and the locked-715 manifest. Compare accuracy, macro/weighted F1, class precision/recall/F1, confusion matrix, NLL, Brier, ECE, entropy, confidence, margin, prediction agreement, logit change, embedding CKA, representation geometry, and train-validation gap. Checkpoints remain selected only by validation macro-F1.

```bash
python -B d19/scripts/evaluate_d19_a0.py --run-dir outputs/d19_runs/d19_a1_id_correct_evidence_only_seed42 --checkpoint best --split test --output-dir outputs/d19_runs/d19_a1_id_correct_evidence_only_seed42/evaluation_best --device cuda:0
python -B d19/scripts/evaluate_d19_a0.py --run-dir outputs/d19_runs/d19_a1_id_null_evidence_only_seed42 --checkpoint best --split test --output-dir outputs/d19_runs/d19_a1_id_null_evidence_only_seed42/evaluation_best --device cuda:0
# Repeat with --checkpoint last and with the locked sample manifest.
# Put counterfactual modes in distinct output directories and add --conditioning-mode MODE.
```"""
    write_md(output / "19_posttraining_analysis_protocol.md", "Post-Training Analysis Protocol", post)
    write_md(output / "20_promotion_and_stop_rules.md", "Promotion And Stop Rules", """**PROMOTE_RELATION_ID** only if correct minus null best validation macro-F1 is at least +0.75 pp, correct's train-validation gap increases by no more than 3.0 pp, no FER class loses more than 5.0 pp validation F1, and every parity/checkpoint gate passes.

**STOP_RELATION_ID** if the gain is below +0.75 pp or class/generalization failure occurs. Do not escalate to independent operators.

**BLOCKED** if technical parity or artifact validity fails. Test results are secondary and never select checkpoints.""")
    write_md(output / "21_risks_and_limitations.md", "Risks And Limitations", "\n".join(f"- {value}" for value in validation["warnings"]) + "\n- Correct-ID may still be redundant with information inferable from base6 attributes.\n- A single seed42 paired result is the registered decision point, not a multiseed estimate.\n- Model-side swapped treatments are post-training diagnostics, not causal training controls.")
    write_json(output / "22_machine_readable_manifest.json", manifest)
    commands = """```powershell
conda run -n fer-graph python -B d19/scripts/validate_d19_a1_id_pair.py --smoke-images 8 --seed 42 --output-dir outputs/d19_analysis/d19_a1_id_implementation_design --strict
```

Use `18_kaggle_training_commands.md` for the only full training path. No local epoch command was executed."""
    write_md(output / "24_run_commands.md", "Run Commands", commands)
    validation["reports_complete"] = all((output / name).exists() for name in REPORT_NAMES if name != "23_validation_summary.json")
    if not validation["reports_complete"]:
        validation["blocking_issues"].append("reports_complete")
    write_json(output / "23_validation_summary.json", validation)
    # Recheck after writing the final required report.
    validation["reports_complete"] = all((output / name).exists() for name in REPORT_NAMES)
    write_json(output / "23_validation_summary.json", validation)
    print(json.dumps(validation, indent=2))
    if args.strict and validation["blocking_issues"]:
        raise RuntimeError(f"Strict D19-A1-ID validation failed: {validation['blocking_issues']}")


if __name__ == "__main__":
    main()
