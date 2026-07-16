"""Validate the frozen OFIX18 C0/C2 paired multi-seed design and run bounded smoke tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

from d18.data.collate import D18Batch, collate_d18_graphs
from d18.data.structure_dataset import StructurePixelDataset
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import (
    apply_graph_regularization,
    load_checkpoint,
    run_resume_signature,
    save_checkpoint,
    scientific_resume_signature,
)

SOURCE = {
    "C0": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c0_clean_control_seed42.yaml",
    "C2": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml",
}
NEW_SEEDS = (7, 21, 84, 123)
ALL_SEEDS = (7, 21, 42, 84, 123)
CONFIG_DIR = ROOT / "configs/d18/overfit_fix_18/multiseed"
SEED42_OUTPUTS = {
    str(ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c0_clean_control_seed42"),
    str(ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42"),
}
ALLOWED_TOP_LEVEL = {"seed", "run_name", "output_dir", "description", "logging"}
INTENDED_FACTOR_PATHS = {
    "training.structure_mode_mix.enabled",
    "training.structure_mode_mix.p_forced_structure",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config is not a mapping: {path}")
    return value


def config_path(cell: str, seed: int) -> Path:
    stem = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    return CONFIG_DIR / f"d18_ofix18_{stem}_seed{seed}.yaml"


def normalized_clone(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    for key in ALLOWED_TOP_LEVEL:
        value.pop(key, None)
    value.setdefault("training", {}).pop("seed", None)
    return value


def normalized_factorial(cfg: dict[str, Any]) -> dict[str, Any]:
    value = normalized_clone(cfg)
    mix = value["training"]["structure_mode_mix"]
    mix["enabled"] = "<MODE_MIX_ENABLED>"
    mix["p_forced_structure"] = "<MODE_MIX_P>"
    return value


def semantic_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                rows.append({"path": path, "left": "<MISSING>", "right": right[key]})
            elif key not in right:
                rows.append({"path": path, "left": left[key], "right": "<MISSING>"})
            else:
                rows.extend(semantic_diff(left[key], right[key], path))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            rows.append({"path": prefix, "left": left, "right": right})
        return rows
    if left != right:
        rows.append({"path": prefix, "left": left, "right": right})
    return rows


def config_sha(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_factors(cfg: dict[str, Any]) -> dict[str, Any]:
    train = cfg["training"]
    reg = train["graph_regularization"]
    mix = train["structure_mode_mix"]
    return {
        "global": float(train.get("drop_edge_p", 0.0)),
        "local": float(reg.get("drop_local_edge_p", 0.0)),
        "knn": float(reg.get("drop_knn_edge_p", 0.0)),
        "structure": float(reg.get("drop_structure_edge_p", 0.0)),
        "mode_mix_enabled": bool(mix.get("enabled", False)),
        "p_forced_structure": float(mix.get("p_forced_structure", 0.0)),
        "p_zero_structure": float(mix.get("p_zero_structure", 0.0)),
    }


def validate_configs() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    sources = {cell: load_yaml(path) for cell, path in SOURCE.items()}
    configs: dict[tuple[str, int], dict[str, Any]] = {}
    semantic_rows: list[dict[str, Any]] = []
    outputs: list[str] = []
    science_signatures: list[str] = []
    run_signatures: list[str] = []
    config_signatures: list[str] = []

    source_factor_diff = semantic_diff(normalized_clone(sources["C0"]), normalized_clone(sources["C2"]))
    assert {row["path"] for row in source_factor_diff} == INTENDED_FACTOR_PATHS
    assert normalized_factorial(sources["C0"]) == normalized_factorial(sources["C2"])

    for cell in ("C0", "C2"):
        source = sources[cell]
        for seed in NEW_SEEDS:
            path = config_path(cell, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            cfg = load_yaml(path)
            configs[(cell, seed)] = cfg
            diff = semantic_diff(normalized_clone(source), normalized_clone(cfg))
            semantic_rows.append({
                "cell": cell,
                "seed": seed,
                "config_path": str(path.relative_to(ROOT)),
                "unexpected_diff_count": len(diff),
                "unexpected_diffs": diff,
            })
            assert not diff, f"{path} changes frozen fields: {diff}"
            assert int(cfg["seed"]) == seed
            assert int(cfg["training"]["seed"]) == seed
            assert f"seed{seed}" in str(cfg["run_name"])
            output = str(cfg["output_dir"])
            assert output.startswith("outputs/d18_runs/ofix18_multiseed/")
            assert str((ROOT / output).resolve()) not in SEED42_OUTPUTS
            factors = effective_factors(cfg)
            assert factors["global"] == factors["local"] == factors["knn"] == factors["structure"] == 0.0
            assert factors["p_zero_structure"] == 0.0
            if cell == "C0":
                assert not factors["mode_mix_enabled"] and factors["p_forced_structure"] == 0.0
            else:
                assert factors["mode_mix_enabled"] and factors["p_forced_structure"] == 0.30
            outputs.append(output)
            science_signatures.append(scientific_resume_signature(cfg))
            run_signatures.append(run_resume_signature(cfg))
            config_signatures.append(config_sha(cfg))

    assert len(outputs) == len(set(outputs)) == 8
    assert len(science_signatures) == len(set(science_signatures)) == 8
    assert len(run_signatures) == len(set(run_signatures)) == 8
    assert len(config_signatures) == len(set(config_signatures)) == 8

    output_binding_checked = True
    for cfg in configs.values():
        changed = copy.deepcopy(cfg)
        changed["output_dir"] = str(cfg["output_dir"]) + "_different"
        assert scientific_resume_signature(changed) == scientific_resume_signature(cfg)
        assert run_resume_signature(changed) != run_resume_signature(cfg)

    result = {
        "status": "PASS",
        "source_configs": {cell: str(path.relative_to(ROOT)) for cell, path in SOURCE.items()},
        "new_config_count": len(configs),
        "all_training_seeds": list(ALL_SEEDS),
        "new_training_seeds": list(NEW_SEEDS),
        "semantic_clone_validation": semantic_rows,
        "c0_c2_seed42_factor_diff": source_factor_diff,
        "intended_factor_paths": sorted(INTENDED_FACTOR_PATHS),
        "all_dropedge_zero": True,
        "output_paths_unique": True,
        "seed42_outputs_untouched": True,
        "scientific_signatures_unique": True,
        "run_resume_signatures_unique": True,
        "config_signatures_unique": True,
        "run_signature_binds_output_dir": output_binding_checked,
        "reproducibility_policy": {
            "python_numpy_torch_cuda_seeded": True,
            "dataloader_generator": "global_torch_rng_seeded_before_loader_construction",
            "worker_seed_policy": "pytorch_default_worker_seed_from_dataloader_base_seed",
            "mode_mix_rng": "global_torch_rng",
            "deterministic_algorithms_forced": False,
        },
        "signatures": {
            f"{cell}_seed{seed}": {
                "scientific": scientific_resume_signature(cfg),
                "run_resume": run_resume_signature(cfg),
                "config": config_sha(cfg),
            }
            for (cell, seed), cfg in configs.items()
        },
    }
    return result, configs


def runtime_cfg(cfg: dict[str, Any], prior_dir: Path, cache_dir: Path) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value["data"]["prior_dir"] = str(prior_dir)
    value["graph"]["cache"] = {
        "enabled": True,
        "dir": str(cache_dir),
        "strict": True,
        "fallback_on_error": False,
    }
    return value


def payload_equal(left: D18Batch, right: D18Batch) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in (
            (left.x_cat, right.x_cat),
            (left.pos_cat, right.pos_cat),
            (left.y, right.y),
            (left.sample_index, right.sample_index),
            (left.ptr, right.ptr),
            (left.batch_index, right.batch_index),
        )
    )


def nonstructure_equal(left: D18Batch, right: D18Batch) -> bool:
    left_keep = left.edge_type_cat != 2
    right_keep = right.edge_type_cat != 2
    return all(
        torch.equal(a, b)
        for a, b in (
            (left.edge_index_cat[:, left_keep], right.edge_index_cat[:, right_keep]),
            (left.edge_attr_cat[left_keep], right.edge_attr_cat[right_keep]),
            (left.edge_type_cat[left_keep], right.edge_type_cat[right_keep]),
        )
    )


def validate_batch(batch: D18Batch) -> dict[str, Any]:
    node_count = int(batch.x_cat.size(0))
    assert node_count > 0 and int(batch.edge_index_cat.size(1)) > 0
    assert int(batch.edge_index_cat.min()) >= 0
    assert int(batch.edge_index_cat.max()) < node_count
    assert bool(torch.isfinite(batch.x_cat).all())
    assert bool(torch.isfinite(batch.edge_attr_cat).all())
    return {
        "node_count": node_count,
        "edge_count": int(batch.edge_index_cat.size(1)),
        "node_dim": int(batch.x_cat.size(1)),
        "edge_dim": int(batch.edge_attr_cat.size(1)),
        "indices_valid": True,
        "finite": True,
    }


def run_smoke(
    configs: dict[tuple[str, int], dict[str, Any]],
    prior_dir: Path,
    cache_dir: Path,
    sample_count: int,
    device: torch.device,
) -> dict[str, Any]:
    chosen = {
        cell: runtime_cfg(configs[(cell, 7)], prior_dir, cache_dir)
        for cell in ("C0", "C2")
    }
    dataset = StructurePixelDataset(
        prior_dir=prior_dir,
        split="train",
        graph=chosen["C0"]["graph"],
        max_samples=sample_count,
    )
    base = collate_d18_graphs([dataset[index] for index in range(len(dataset))]).to(device)
    base_info = validate_batch(base)
    assert base_info["node_dim"] == 10 and base_info["edge_dim"] == 6
    loss_fn = torch.nn.CrossEntropyLoss()
    rows: dict[str, Any] = {}

    for cell in ("C0", "C2"):
        cfg = chosen[cell]
        train_cfg = copy.deepcopy(cfg["training"])
        train_cfg["structure_mode_mix"]["p_forced_structure"] = 0.0
        official, official_stats = apply_graph_regularization(base, train_cfg)
        assert payload_equal(base, official)
        assert nonstructure_equal(base, official)
        assert int(official_stats["structure_mode_forced_sample_count"]) == 0
        model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6).to(device)
        model.train()
        logits = model(official)["logits"]
        loss = loss_fn(logits, official.y)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        assert tuple(logits.shape) == (sample_count, 7)
        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)) and gradients_finite
        rows[cell] = {
            "official_logits_shape": list(logits.shape),
            "official_loss": float(loss.detach().cpu()),
            "official_loss_finite": True,
            "backward_gradients_finite": True,
            "node_dim": 10,
            "edge_dim": 6,
        }

        if cell == "C2":
            forced_cfg = copy.deepcopy(cfg["training"])
            forced_cfg["structure_mode_mix"]["enabled"] = True
            forced_cfg["structure_mode_mix"]["p_forced_structure"] = 1.0
            forced, forced_stats = apply_graph_regularization(base, forced_cfg)
            assert int(forced_stats["structure_mode_forced_sample_count"]) == sample_count
            assert int(forced_stats["structure_edges_after_drop"]) == 0
            assert payload_equal(base, forced) and nonstructure_equal(base, forced)
            model.eval()
            with torch.no_grad():
                forced_logits = model(forced)["logits"]
                first = model(base)["logits"]
                second = model(base)["logits"]
            assert tuple(forced_logits.shape) == (sample_count, 7)
            assert bool(torch.isfinite(forced_logits).all())
            assert torch.allclose(first, second, rtol=1e-6, atol=1e-6)
            rows[cell].update({
                "forced_logits_shape": list(forced_logits.shape),
                "forced_logits_finite": True,
                "forced_sample_count": int(forced_stats["structure_mode_forced_sample_count"]),
                "forced_structure_edges_after": int(forced_stats["structure_edges_after_drop"]),
                "same_images_labels_node_order": payload_equal(base, forced),
                "nonstructure_edges_unchanged": nonstructure_equal(base, forced),
                "official_eval_deterministic": True,
                "validation_test_default_mode": "official",
            })

    return {
        "status": "PASS",
        "scope": "bounded smoke only, not experiment results",
        "device": str(device),
        "sample_count": sample_count,
        "base_batch": base_info,
        "configs": rows,
        "structure_dropedge_zero": True,
        "full_training_run": False,
    }


def validate_resume_contract(configs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    cfg = configs[("C0", 7)]
    model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    path = Path(tempfile.gettempdir()) / "d18_ofix18_multiseed_resume_contract.pt"
    try:
        save_checkpoint(path, model, optimizer, None, 1, 0.1, 1, 1.0, 1, 0, 1, cfg)
        load_checkpoint(
            path,
            model,
            optimizer,
            device="cpu",
            expected_resume_signature=scientific_resume_signature(cfg),
            expected_run_resume_signature=run_resume_signature(cfg),
            strict_signature=True,
        )
        cross_seed_rejected = False
        try:
            other = configs[("C0", 21)]
            load_checkpoint(
                path,
                model,
                optimizer,
                device="cpu",
                expected_resume_signature=scientific_resume_signature(other),
                expected_run_resume_signature=run_resume_signature(other),
                strict_signature=True,
            )
        except RuntimeError:
            cross_seed_rejected = True
        same_science_other_output = copy.deepcopy(cfg)
        same_science_other_output["output_dir"] = str(cfg["output_dir"]) + "_other"
        cross_output_rejected = False
        try:
            load_checkpoint(
                path,
                model,
                optimizer,
                device="cpu",
                expected_resume_signature=scientific_resume_signature(same_science_other_output),
                expected_run_resume_signature=run_resume_signature(same_science_other_output),
                strict_signature=True,
            )
        except RuntimeError as exc:
            cross_output_rejected = "Run resume signature mismatch" in str(exc)
        assert cross_seed_rejected and cross_output_rejected
        return {
            "status": "PASS",
            "same_run_resume_accepted": True,
            "cross_seed_resume_rejected": cross_seed_rejected,
            "cross_output_resume_rejected": cross_output_rejected,
        }
    finally:
        path.unlink(missing_ok=True)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    smoke = result.get("smoke")
    lines = [
        "# OFIX18 C0/C2 Multi-Seed Validation and Smoke",
        "",
        f"- Semantic validation: **{result['config_validation']['status']}**",
        f"- Resume contract: **{result['resume_contract']['status']}**",
        f"- Eight configs parse and preserve frozen fields: **True**",
        f"- Unique output/config/scientific/run signatures: **True**",
        f"- Seed42 output targeted: **False**",
        "",
        "## Seed policy",
        "",
        "- Python, NumPy, Torch CPU and CUDA use the configured seed.",
        "- DataLoader shuffle consumes the globally seeded Torch RNG.",
        "- Worker seeds use PyTorch's default DataLoader base-seed policy.",
        "- Mode-mix selection consumes the Torch RNG.",
        "- Deterministic algorithms are not forced, matching seed42.",
    ]
    if smoke:
        lines += [
            "",
            "## Bounded smoke",
            "",
            f"- Status: **{smoke['status']}**",
            f"- Device: {smoke['device']}",
            f"- Images: {smoke['sample_count']}",
            f"- Node/edge dimensions: {smoke['base_batch']['node_dim']}/{smoke['base_batch']['edge_dim']}",
            "- C0 and C2 official forward and backward are finite.",
            "- C2 official and forced branches use identical images, labels and node order.",
            "- Forced branch removes structure edges only; local and kNN edges are unchanged.",
            "- Validation/test default remains deterministic official mode.",
        ]
    lines += ["", "No full local training was run.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--cache_dir", default="outputs/d18_graph_cache/ofix17_structure_reg/base6_shared")
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_c0_c2_multiseed_design")
    parser.add_argument("--sample_count", type=int, default=4, choices=range(2, 9))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()

    config_result, configs = validate_configs()
    result: dict[str, Any] = {
        "status": "PASS",
        "config_validation": config_result,
        "resume_contract": validate_resume_contract(configs),
        "full_training_run": False,
    }
    if not args.skip_smoke:
        device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        result["smoke"] = run_smoke(
            configs,
            Path(args.prior_dir),
            Path(args.cache_dir),
            int(args.sample_count),
            device,
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_and_smoke.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output / "05_validation_and_smoke.md", result)
    print(json.dumps({"status": "PASS", "output": str(output), "full_training_run": False}, indent=2))


if __name__ == "__main__":
    main()
