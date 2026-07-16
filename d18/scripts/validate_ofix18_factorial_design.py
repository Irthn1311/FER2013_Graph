"""Validate D18 OFIX18 factorial configs and bounded graph/model behavior."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import D18Batch, collate_d18_graphs
from d18.data.structure_dataset import StructurePixelDataset
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import (
    apply_graph_regularization, load_checkpoint, save_checkpoint, scientific_resume_signature,
)

PATHS = {
    "C0": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c0_clean_control_seed42.yaml",
    "C1": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c1_structure_dropedge_only_seed42.yaml",
    "C2": ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml",
    "C3_existing": ROOT / "configs/d18/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42.yaml",
}
EXPECTED = {
    "C0": (0.0, False, 0.0),
    "C1": (0.3, False, 0.0),
    "C2": (0.0, True, 0.3),
    "C3_existing": (0.3, True, 0.3),
}


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normal_form(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    for key in ("description", "run_name", "output_dir", "logging"):
        value.pop(key, None)
    value["training"]["graph_regularization"]["drop_structure_edge_p"] = "<DROP_FACTOR>"
    value["training"]["structure_mode_mix"]["enabled"] = "<MIX_FACTOR>"
    value["training"]["structure_mode_mix"]["p_forced_structure"] = "<MIX_P_FACTOR>"
    return value


def validate_configs(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = normal_form(configs["C3_existing"])
    outputs, signatures, matrix = [], {}, {}
    for cell, cfg in configs.items():
        train = cfg["training"]
        reg = train["graph_regularization"]
        mix = train["structure_mode_mix"]
        actual = (
            float(reg["drop_structure_edge_p"]),
            bool(mix["enabled"]),
            float(mix["p_forced_structure"]),
        )
        assert actual == EXPECTED[cell], (cell, actual, EXPECTED[cell])
        assert int(train["seed"]) == 42
        assert float(train["drop_edge_p"]) == 0.0
        assert float(reg["drop_local_edge_p"]) == 0.0
        assert float(reg["drop_knn_edge_p"]) == 0.0
        assert normal_form(cfg) == reference, f"{cell} has an unintended scientific diff"
        outputs.append(str(cfg["output_dir"]))
        signatures[cell] = scientific_resume_signature(cfg)
        matrix[cell] = {
            "drop_structure_edge_p": actual[0],
            "structure_mode_mix_enabled": actual[1],
            "p_forced_structure": actual[2],
        }
    assert len(outputs) == len(set(outputs)), "output collision"
    assert len(signatures) == len(set(signatures.values())), "resume signature collision"
    return {
        "status": "PASS",
        "factorial_matrix": matrix,
        "output_paths_unique": True,
        "resume_signatures_unique": True,
        "resume_signatures": signatures,
        "scientific_invariants_match_ofix17b": True,
    }


def validate_resume_behavior(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model = StructureGNN.from_config(configs["C0"], input_dim=10, edge_attr_dim=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    temp_root = Path(tempfile.gettempdir())
    c0_path = temp_root / "ofix18_c0_resume_validation.pt"
    c3_path = temp_root / "ofix18_c3_resume_validation.pt"
    try:
        save_checkpoint(c0_path, model, optimizer, None, 1, 0.1, 1, 1.0, 1, 0, 1, configs["C0"])
        save_checkpoint(c3_path, model, optimizer, None, 1, 0.1, 1, 1.0, 1, 0, 1, configs["C3_existing"])
        load_checkpoint(
            c0_path, model, optimizer, device="cpu",
            expected_resume_signature=scientific_resume_signature(configs["C0"]),
            strict_signature=True,
        )
        cross_c1_rejected = False
        try:
            load_checkpoint(
                c0_path, model, optimizer, device="cpu",
                expected_resume_signature=scientific_resume_signature(configs["C1"]),
                strict_signature=True,
            )
        except RuntimeError as exc:
            cross_c1_rejected = "Resume signature mismatch" in str(exc)
        cross_c3_rejected = False
        try:
            load_checkpoint(
                c3_path, model, optimizer, device="cpu",
                expected_resume_signature=scientific_resume_signature(configs["C0"]),
                strict_signature=True,
            )
        except RuntimeError as exc:
            cross_c3_rejected = "Resume signature mismatch" in str(exc)
        assert cross_c1_rejected and cross_c3_rejected
        return {
            "status": "PASS",
            "same_config_resume_accepted": True,
            "c0_checkpoint_into_c1_rejected": cross_c1_rejected,
            "c3_checkpoint_into_c0_rejected": cross_c3_rejected,
        }
    finally:
        c0_path.unlink(missing_ok=True)
        c3_path.unlink(missing_ok=True)


def runtime_cfg(cfg: dict[str, Any], prior: Path, cache: Path) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value["data"]["prior_dir"] = str(prior)
    value["graph"]["cache"] = {
        "enabled": True,
        "dir": str(cache),
        "strict": True,
        "fallback_on_error": False,
    }
    return value


def digest(batch: D18Batch) -> str:
    h = hashlib.sha256()
    for tensor in (
        batch.x_cat, batch.edge_index_cat, batch.edge_attr_cat, batch.edge_type_cat,
        batch.structure_relation_id_cat, batch.y, batch.sample_index, batch.ptr,
    ):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def payload_equal(left: D18Batch, right: D18Batch) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in (
            (left.x_cat, right.x_cat), (left.pos_cat, right.pos_cat),
            (left.y, right.y), (left.sample_index, right.sample_index),
            (left.ptr, right.ptr), (left.batch_index, right.batch_index),
        )
    )


def nonstructure_equal(left: D18Batch, right: D18Batch) -> bool:
    lm, rm = left.edge_type_cat != 2, right.edge_type_cat != 2
    return all(
        torch.equal(a, b)
        for a, b in (
            (left.edge_index_cat[:, lm], right.edge_index_cat[:, rm]),
            (left.edge_attr_cat[lm], right.edge_attr_cat[rm]),
            (left.edge_type_cat[lm], right.edge_type_cat[rm]),
        )
    )


def integrity(batch: D18Batch) -> dict[str, Any]:
    n = int(batch.x_cat.size(0))
    assert int(batch.edge_index_cat.min()) >= 0
    assert int(batch.edge_index_cat.max()) < n
    assert bool(torch.isfinite(batch.x_cat).all())
    assert bool(torch.isfinite(batch.edge_attr_cat).all())
    degree = torch.zeros(n, dtype=torch.long, device=batch.x_cat.device)
    degree.scatter_add_(0, batch.edge_index_cat[0], torch.ones_like(batch.edge_index_cat[0]))
    degree.scatter_add_(0, batch.edge_index_cat[1], torch.ones_like(batch.edge_index_cat[1]))
    return {
        "node_count": n,
        "edge_count": int(batch.edge_index_cat.size(1)),
        "isolated_node_count": int((degree == 0).sum()),
        "valid_indices": True,
        "finite_node_features": True,
        "finite_edge_attributes": True,
    }


def run_smoke(
    configs: dict[str, dict[str, Any]],
    prior: Path,
    cache: Path,
    device: torch.device,
    sample_count: int,
    trials: int,
) -> dict[str, Any]:
    cfgs = {key: runtime_cfg(value, prior, cache) for key, value in configs.items()}
    ds = StructurePixelDataset(prior, "train", cfgs["C0"]["graph"], max_samples=sample_count)
    batch = collate_d18_graphs([ds[i] for i in range(len(ds))]).to(device)
    base_check = integrity(batch)
    assert base_check["isolated_node_count"] == 0
    ce = torch.nn.CrossEntropyLoss()
    rows = {}
    torch.manual_seed(42)
    for cell in ("C0", "C1", "C2"):
        cfg = cfgs[cell]
        transformed, stats = apply_graph_regularization(batch, cfg["training"])
        model = StructureGNN.from_config(
            cfg, input_dim=int(batch.x_cat.size(1)), edge_attr_dim=int(batch.edge_attr_cat.size(1))
        ).to(device)
        logits = model(transformed)["logits"]
        loss = ce(logits, transformed.y)
        loss.backward()
        grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert tuple(logits.shape) == (sample_count, 7)
        assert bool(torch.isfinite(loss)) and grads_ok
        check = integrity(transformed)
        assert check["isolated_node_count"] == 0
        if cell in ("C0", "C1"):
            assert int(stats["structure_mode_forced_sample_count"]) == 0
        rows[cell] = {
            "node_dim": int(batch.x_cat.size(1)),
            "edge_dim": int(batch.edge_attr_cat.size(1)),
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "loss_finite": True,
            "backward_finite": True,
            "labels_unchanged": bool(torch.equal(batch.y, transformed.y)),
            "node_payload_unchanged": payload_equal(batch, transformed),
            "official_sample_count": int(stats["structure_mode_official_sample_count"]),
            "forced_sample_count": int(stats["structure_mode_forced_sample_count"]),
        }

    before = after = 0
    local_knn_ok = payload_ok = True
    isolated_max = 0
    torch.manual_seed(4201)
    for _ in range(trials):
        transformed, stats = apply_graph_regularization(batch, cfgs["C1"]["training"])
        before += int(stats["structure_edges_before_drop"])
        after += int(stats["structure_edges_after_drop"])
        local_knn_ok &= nonstructure_equal(batch, transformed)
        payload_ok &= payload_equal(batch, transformed)
        isolated_max = max(isolated_max, integrity(transformed)["isolated_node_count"])
    retention = after / max(before, 1)
    assert 0.66 <= retention <= 0.74
    assert local_knn_ok and payload_ok and isolated_max == 0

    official_cfg = copy.deepcopy(cfgs["C2"]["training"])
    official_cfg["structure_mode_mix"]["p_forced_structure"] = 0.0
    forced_cfg = copy.deepcopy(cfgs["C2"]["training"])
    forced_cfg["structure_mode_mix"]["p_forced_structure"] = 1.0
    c3_forced_cfg = copy.deepcopy(cfgs["C3_existing"]["training"])
    c3_forced_cfg["structure_mode_mix"]["enabled"] = True
    c3_forced_cfg["structure_mode_mix"]["p_forced_structure"] = 1.0
    torch.manual_seed(99)
    official, official_stats = apply_graph_regularization(batch, official_cfg)
    torch.manual_seed(99)
    forced, forced_stats = apply_graph_regularization(batch, forced_cfg)
    torch.manual_seed(99)
    c3_forced, c3_stats = apply_graph_regularization(batch, c3_forced_cfg)
    assert digest(official) == digest(batch)
    assert int(forced_stats["structure_edges_after_drop"]) == 0
    assert nonstructure_equal(batch, forced) and payload_equal(batch, forced)
    assert digest(forced) == digest(c3_forced)

    model = StructureGNN.from_config(
        cfgs["C2"], input_dim=int(batch.x_cat.size(1)), edge_attr_dim=int(batch.edge_attr_cat.size(1))
    ).to(device)
    model.eval()
    with torch.no_grad():
        first = model(batch)["logits"]
        second = model(batch)["logits"]
    eval_max_abs_diff = float((first - second).abs().max().detach().cpu())
    deterministic = bool(torch.allclose(first, second, rtol=1e-6, atol=1e-6))
    assert deterministic, f"official eval logits differ beyond tolerance: max_abs={eval_max_abs_diff}"

    return {
        "status": "PASS",
        "device": str(device),
        "sample_count": sample_count,
        "base_integrity": base_check,
        "per_config": rows,
        "c1": {
            "retention_trials": trials,
            "structure_edges_before_total": before,
            "structure_edges_after_total": after,
            "structure_retention_observed": retention,
            "expected_retention": 0.70,
            "local_knn_edges_unchanged": local_knn_ok,
            "node_features_support_labels_unchanged": payload_ok,
            "max_isolated_node_count": isolated_max,
        },
        "c2": {
            "official_branch_forced_count": int(official_stats["structure_mode_forced_sample_count"]),
            "forced_branch_forced_count": int(forced_stats["structure_mode_forced_sample_count"]),
            "forced_branch_expected_count": sample_count,
            "forced_branch_structure_edges_after": int(forced_stats["structure_edges_after_drop"]),
            "local_knn_edges_unchanged": nonstructure_equal(batch, forced),
            "node_order_and_labels_unchanged": payload_equal(batch, forced),
            "forced_graph_hash": digest(forced),
            "ofix17b_forced_graph_hash": digest(c3_forced),
            "matches_ofix17b_forced_implementation": digest(forced) == digest(c3_forced),
            "ofix17b_forced_count": int(c3_stats["structure_mode_forced_sample_count"]),
        },
        "official_eval_deterministic": deterministic,
        "official_eval_max_abs_diff": eval_max_abs_diff,
        "official_eval_graph_hash_unchanged": digest(official) == digest(batch),
    }


def write_report(path: Path, config_result: dict[str, Any], smoke: dict[str, Any]) -> None:
    lines = [
        "# OFIX18 Factorial Smoke Validation",
        "",
        "Bounded implementation smoke test only; these are not experiment results.",
        "",
        f"- Status: **{smoke['status']}**",
        f"- Device: {smoke['device']}",
        f"- Images: {smoke['sample_count']}",
        f"- Unique output directories: {config_result['output_paths_unique']}",
        f"- Unique resume signatures: {config_result['resume_signatures_unique']}",
        "",
        "| Cell | Node dim | Edge dim | Logits | Loss | Backward finite | Official | Forced |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for cell, row in smoke["per_config"].items():
        lines.append(
            f"| {cell} | {row['node_dim']} | {row['edge_dim']} | {tuple(row['logits_shape'])} | "
            f"{row['loss']:.6f} | {row['backward_finite']} | {row['official_sample_count']} | {row['forced_sample_count']} |"
        )
    c1, c2 = smoke["c1"], smoke["c2"]
    lines += [
        "",
        "## C1 Structure DropEdge",
        "",
        f"- Trials: {c1['retention_trials']}",
        f"- Observed retention: {c1['structure_retention_observed']:.6f}; target approximately 0.70.",
        f"- Local/kNN unchanged: {c1['local_knn_edges_unchanged']}",
        f"- Node support/features/labels unchanged: {c1['node_features_support_labels_unchanged']}",
        f"- Maximum isolated nodes: {c1['max_isolated_node_count']}",
        "",
        "## C2 Mode Mix",
        "",
        f"- Official forced count: {c2['official_branch_forced_count']}",
        f"- Forced branch count: {c2['forced_branch_forced_count']}/{c2['forced_branch_expected_count']}",
        f"- Structure edges after forced transform: {c2['forced_branch_structure_edges_after']}",
        f"- Local/kNN unchanged: {c2['local_knn_edges_unchanged']}",
        f"- Node ordering/labels unchanged: {c2['node_order_and_labels_unchanged']}",
        f"- Matches OFIX17-B forced implementation: {c2['matches_ofix17b_forced_implementation']}",
        f"- Forced graph hash: {c2['forced_graph_hash']}",
        "",
        f"Official eval deterministic within rtol/atol 1e-6: {smoke['official_eval_deterministic']}",
        f"Official eval maximum absolute logit difference: {smoke['official_eval_max_abs_diff']:.9g}",
        f"Official eval graph hash unchanged: {smoke['official_eval_graph_hash_unchanged']}",
        "",
        "No full training was launched.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--cache_dir", default="outputs/d18_graph_cache/ofix17_structure_reg/base6_shared")
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_factorial_design")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample_count", type=int, default=4, choices=range(2, 9))
    parser.add_argument("--retention_trials", type=int, default=64)
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()
    configs = {cell: load(path) for cell, path in PATHS.items()}
    config_result = validate_configs(configs)
    result: dict[str, Any] = {
        "config_validation": config_result,
        "resume_behavior_validation": validate_resume_behavior(configs),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_smoke:
        device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        smoke = run_smoke(
            configs, Path(args.prior_dir), Path(args.cache_dir), device,
            int(args.sample_count), int(args.retention_trials)
        )
        result["smoke_validation"] = smoke
        write_report(output / "05_smoke_validation.md", config_result, smoke)
    (output / "factorial_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output / "factorial_validation.json")}, indent=2))


if __name__ == "__main__":
    main()
