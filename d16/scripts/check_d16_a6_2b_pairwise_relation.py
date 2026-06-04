"""Checker for D16R A6-2b pairwise hard-relation auxiliary heads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.losses.pairwise_hard_relation import pairwise_hard_relation_lambda
from d16.models.d16_model import D16Model
from d16.training.train_d16 import (
    _weighted_ce_loss,
    attach_hard_proto_loss_if_needed,
    attach_pairwise_hard_relation_loss_if_needed,
    build_dataset,
    resolve_device,
)


def _read_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _find_batch_with_pairs(ds, max_scan: int = 5000):
    selected = {}
    for index in range(min(len(ds), int(max_scan))):
        graph = ds[index]
        label = int(graph.y.detach().cpu().view(-1)[0].item())
        if label in {2, 4, 6} and label not in selected:
            selected[label] = graph
        if {2, 4, 6}.issubset(selected):
            return collate_d16_graphs([selected[2], selected[4], selected[6]])
    raise RuntimeError("Could not find Fear, Sad, and Neutral samples for pairwise checker")


def _grad_norm(params) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(total)


def run_check(config_path: Path, prior_dir: Path, output_dir: Path, old_config_path: Path | None, device_name: str) -> Dict[str, Any]:
    cfg = _read_config(config_path)
    loss_cfg = cfg.get("loss", {}) or {}
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool((cfg.get("training", {}) or {}).get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool((cfg.get("training", {}) or {}).get("allow_tf32", True))

    ds = build_dataset(cfg, prior_dir, "train")
    first_batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
    input_dim = int(first_batch.x_cat.size(1))
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    embedding_dim = int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5
    pair_loss = attach_pairwise_hard_relation_loss_if_needed(model, loss_cfg, embedding_dim=embedding_dim)
    if pair_loss is None:
        raise RuntimeError("A6-2b checker expected pairwise hard relation loss to be enabled")
    pair_loss.to(device)
    hard_proto = attach_hard_proto_loss_if_needed(model, loss_cfg, embedding_dim=embedding_dim)
    if hard_proto is not None or hasattr(model, "hard_proto_sep_loss"):
        raise AssertionError("A6-2b must not attach global hard prototype loss")

    batch = _find_batch_with_pairs(ds).to(device)
    model.train()
    out = model(batch)
    logits = out["logits"]
    z = out["z_image"]
    aux = pair_loss(z, batch.y)
    ce_loss, _ = _weighted_ce_loss(logits, batch.y, batch, loss_cfg)
    lambda_epoch_1 = pairwise_hard_relation_lambda(loss_cfg, 1)
    lambda_epoch_15 = pairwise_hard_relation_lambda(loss_cfg, 15)
    lambda_epoch_20 = pairwise_hard_relation_lambda(loss_cfg, 20)
    lambda_epoch_30 = pairwise_hard_relation_lambda(loss_cfg, 30)
    loss = ce_loss + lambda_epoch_30 * aux["loss_pairwise_hard_relation"]
    if not torch.isfinite(loss):
        raise FloatingPointError("A6-2b combined loss is not finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    pair_head_grad_norm = _grad_norm(pair_loss.parameters())
    classifier_grad_norm = _grad_norm(model.classifier.parameters()) if model.classifier is not None else 0.0

    no_pair_y = torch.full_like(batch.y, 3)
    no_pair = pair_loss(z.detach(), no_pair_y)
    no_pair_loss = float(no_pair["loss_pairwise_hard_relation"].detach().cpu().item())
    no_pair_available = int(no_pair["pairwise_available_count"].detach().cpu().item())

    old_ok = True
    old_has_aux_module = False
    if old_config_path is not None:
        old_cfg = _read_config(old_config_path)
        old_ds = build_dataset(old_cfg, prior_dir, "train")
        old_first = next(iter(DataLoader(old_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
        old_model = D16Model.from_config(old_cfg, input_dim=old_first.x_cat.size(1)).to(device)
        old_pair = attach_pairwise_hard_relation_loss_if_needed(
            old_model,
            old_cfg.get("loss", {}) or {},
            embedding_dim=int((old_cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
        )
        old_has_aux_module = old_pair is not None or hasattr(old_model, "pairwise_hard_relation_loss")
        old_batch = next(iter(DataLoader(old_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs))).to(device)
        old_out = old_model(old_batch)
        old_ok = tuple(old_out["logits"].shape) == (2, 7)

    summary = {
        "decision": "A6_2B_PAIRWISE_RELATION_CHECK_PASS",
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "input_dim": int(input_dim),
        "logits_shape": list(logits.shape),
        "z_shape": list(z.shape),
        "pair_names": list(pair_loss.pair_names),
        "pairwise_loss_total": float(aux["loss_pairwise_hard_relation"].detach().cpu().item()),
        "pairwise_loss_fear_sad": float(aux["loss_pairwise_fear_sad"].detach().cpu().item()),
        "pairwise_loss_sad_neutral": float(aux["loss_pairwise_sad_neutral"].detach().cpu().item()),
        "pair_count_fear_sad": int(aux["pair_count_fear_sad"].detach().cpu().item()),
        "pair_count_sad_neutral": int(aux["pair_count_sad_neutral"].detach().cpu().item()),
        "pair_acc_fear_sad": float(aux["pair_acc_fear_sad"].detach().cpu().item()),
        "pair_acc_sad_neutral": float(aux["pair_acc_sad_neutral"].detach().cpu().item()),
        "lambda_epoch_1": float(lambda_epoch_1),
        "lambda_epoch_15": float(lambda_epoch_15),
        "lambda_epoch_20": float(lambda_epoch_20),
        "lambda_epoch_30": float(lambda_epoch_30),
        "no_pair_loss": no_pair_loss,
        "no_pair_available_count": no_pair_available,
        "pair_head_grad_norm": pair_head_grad_norm,
        "classifier_grad_norm": classifier_grad_norm,
        "old_a5b_forward_ok": bool(old_ok),
        "old_a5b_has_pairwise_aux_module": bool(old_has_aux_module),
        "global_hard_proto_attached": bool(hard_proto is not None),
    }
    if tuple(logits.shape) != (int(batch.num_graphs), 7):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if z.ndim != 2 or z.size(0) != int(batch.num_graphs) or z.size(1) != embedding_dim:
        raise AssertionError(f"Unexpected z shape: {tuple(z.shape)}")
    if summary["pair_count_fear_sad"] <= 0 or summary["pair_count_sad_neutral"] <= 0:
        raise AssertionError("Checker batch must contain samples for both configured pairs")
    if abs(no_pair_loss) > 1e-8 or no_pair_available != 0:
        raise AssertionError("No-pair batch should produce zero aux loss and zero available pair count")
    if pair_head_grad_norm <= 0.0:
        raise AssertionError("Expected gradients to flow to pairwise heads")
    if classifier_grad_norm <= 0.0:
        raise AssertionError("Expected gradients to flow to main classifier")
    if not old_ok or old_has_aux_module:
        raise AssertionError("Old A5b config compatibility check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a6_2b_pairwise_relation_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# D16R A6-2b Pairwise Relation Check

## Decision
`{summary['decision']}`

## Forward
- logits_shape: `{summary['logits_shape']}`
- z_shape: `{summary['z_shape']}`
- pair_names: `{summary['pair_names']}`

## Loss
- pairwise_loss_total: `{summary['pairwise_loss_total']:.6f}`
- pairwise_loss_fear_sad: `{summary['pairwise_loss_fear_sad']:.6f}`
- pairwise_loss_sad_neutral: `{summary['pairwise_loss_sad_neutral']:.6f}`
- no_pair_loss: `{summary['no_pair_loss']:.6f}`

## Lambda Schedule
- epoch 1: `{summary['lambda_epoch_1']:.6f}`
- epoch 15: `{summary['lambda_epoch_15']:.6f}`
- epoch 20: `{summary['lambda_epoch_20']:.6f}`
- epoch 30: `{summary['lambda_epoch_30']:.6f}`

## Gradients
- pair_head_grad_norm: `{summary['pair_head_grad_norm']:.6f}`
- classifier_grad_norm: `{summary['classifier_grad_norm']:.6f}`

## Backward Compatibility
- old_a5b_forward_ok: `{summary['old_a5b_forward_ok']}`
- old_a5b_has_pairwise_aux_module: `{summary['old_a5b_has_pairwise_aux_module']}`
- global_hard_proto_attached: `{summary['global_hard_proto_attached']}`
"""
    _write_text(output_dir / "D16R_A6_2B_PAIRWISE_RELATION_CHECK.md", report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    parser.add_argument("--old_config", default="configs/d16/main_branch/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_check(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        old_config_path=Path(args.old_config) if args.old_config else None,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
