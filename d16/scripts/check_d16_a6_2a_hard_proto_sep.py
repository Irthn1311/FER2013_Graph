"""Checker for D16R A6-2a hard prototype separation loss."""

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
from d16.losses.hard_proto_separation import hard_proto_lambda
from d16.models.d16_model import D16Model
from d16.training.train_d16 import attach_hard_proto_loss_if_needed, build_dataset, resolve_device


def _read_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _find_batch_with_hard(loader: DataLoader, hard_ids: set[int]):
    fallback = None
    for batch in loader:
        if fallback is None:
            fallback = batch
        labels = set(int(x) for x in batch.y.detach().cpu().tolist())
        if labels & hard_ids:
            return batch
    if fallback is None:
        raise RuntimeError("No batch available for checker")
    return fallback


def run_check(config_path: Path, prior_dir: Path, output_dir: Path, old_config_path: Path | None, device_name: str) -> Dict[str, Any]:
    cfg = _read_config(config_path)
    loss_cfg = cfg.get("loss", {}) or {}
    hard_cfg = loss_cfg.get("hard_proto_sep", {}) or {}
    hard_ids = set(int(x) for x in hard_cfg.get("hard_class_ids", [0, 2, 4, 6]))
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool((cfg.get("training", {}) or {}).get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool((cfg.get("training", {}) or {}).get("allow_tf32", True))

    ds = build_dataset(cfg, prior_dir, "train")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)
    first_batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
    input_dim = int(first_batch.x_cat.size(1))
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    hard_loss = attach_hard_proto_loss_if_needed(
        model,
        loss_cfg,
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_loss is None:
        raise RuntimeError("A6-2a checker expected hard prototype loss to be enabled")
    hard_loss.to(device)

    batch = _find_batch_with_hard(loader, hard_ids).to(device)
    model.train()
    out = model(batch)
    logits = out["logits"]
    z = out["z_image"]
    z.retain_grad()
    aux = hard_loss(z, batch.y)
    lambda_epoch_1 = hard_proto_lambda(loss_cfg, 1)
    lambda_epoch_10 = hard_proto_lambda(loss_cfg, 10)
    lambda_epoch_15 = hard_proto_lambda(loss_cfg, 15)
    lambda_epoch_20 = hard_proto_lambda(loss_cfg, 20)
    loss = aux["loss_hard_proto_sep"]
    if not torch.isfinite(loss):
        raise FloatingPointError("hard prototype loss is not finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    prototype_grad_norm = float(hard_loss.prototypes.grad.detach().norm().cpu().item()) if hard_loss.prototypes.grad is not None else 0.0
    z_grad_norm = float(z.grad.detach().norm().cpu().item()) if z.grad is not None else 0.0

    no_hard_y = torch.full_like(batch.y, 3)
    no_hard = hard_loss(z.detach(), no_hard_y)
    no_hard_loss = float(no_hard["loss_hard_proto_sep"].detach().cpu().item())
    no_hard_count = int(no_hard["hard_proto_sample_count"].detach().cpu().item())

    old_ok = True
    old_has_aux_module = False
    if old_config_path is not None:
        old_cfg = _read_config(old_config_path)
        old_ds = build_dataset(old_cfg, prior_dir, "train")
        old_first = next(iter(DataLoader(old_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
        old_model = D16Model.from_config(old_cfg, input_dim=old_first.x_cat.size(1)).to(device)
        old_aux = attach_hard_proto_loss_if_needed(
            old_model,
            old_cfg.get("loss", {}) or {},
            embedding_dim=int((old_cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
        )
        old_has_aux_module = old_aux is not None or hasattr(old_model, "hard_proto_sep_loss")
        old_batch = next(iter(DataLoader(old_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs))).to(device)
        old_out = old_model(old_batch)
        old_ok = tuple(old_out["logits"].shape) == (2, 7)

    summary = {
        "decision": "A6_2A_HARD_PROTO_SEP_CHECK_PASS",
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "input_dim": int(input_dim),
        "logits_shape": list(logits.shape),
        "z_shape": list(z.shape),
        "prototype_shape": list(hard_loss.prototypes.shape),
        "hard_proto_sample_count": int(aux["hard_proto_sample_count"].detach().cpu().item()),
        "hard_proto_loss": float(loss.detach().cpu().item()),
        "hard_proto_ce": float(aux["loss_proto_ce"].detach().cpu().item()),
        "hard_proto_margin": float(aux["loss_proto_margin"].detach().cpu().item()),
        "lambda_epoch_1": float(lambda_epoch_1),
        "lambda_epoch_10": float(lambda_epoch_10),
        "lambda_epoch_15": float(lambda_epoch_15),
        "lambda_epoch_20": float(lambda_epoch_20),
        "no_hard_loss": no_hard_loss,
        "no_hard_count": no_hard_count,
        "prototype_grad_norm": prototype_grad_norm,
        "z_grad_norm": z_grad_norm,
        "old_a5b_forward_ok": bool(old_ok),
        "old_a5b_has_aux_module": bool(old_has_aux_module),
    }
    if tuple(logits.shape) != (int(batch.num_graphs), 7):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if z.ndim != 2 or z.size(0) != int(batch.num_graphs):
        raise AssertionError(f"Unexpected z shape: {tuple(z.shape)}")
    if z.size(1) != int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5:
        raise AssertionError(f"Unexpected z dim: {z.size(1)}")
    if int(aux["hard_proto_sample_count"].detach().cpu().item()) <= 0:
        raise AssertionError("Checker batch did not contain hard samples")
    if abs(no_hard_loss) > 1e-8 or no_hard_count != 0:
        raise AssertionError("No-hard batch should produce zero aux loss and zero hard count")
    if prototype_grad_norm <= 0.0 or z_grad_norm <= 0.0:
        raise AssertionError("Expected gradients to flow to prototypes and z embedding")
    if not old_ok or old_has_aux_module:
        raise AssertionError("Old A5b config compatibility check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a6_2a_hard_proto_sep_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# D16R A6-2a Hard Prototype Separation Check

## Decision
`{summary['decision']}`

## Forward
- logits_shape: `{summary['logits_shape']}`
- z_shape: `{summary['z_shape']}`
- prototype_shape: `{summary['prototype_shape']}`

## Loss
- hard_proto_loss: `{summary['hard_proto_loss']:.6f}`
- hard_proto_ce: `{summary['hard_proto_ce']:.6f}`
- hard_proto_margin: `{summary['hard_proto_margin']:.6f}`
- hard_proto_sample_count: `{summary['hard_proto_sample_count']}`
- no_hard_loss: `{summary['no_hard_loss']:.6f}`

## Lambda Schedule
- epoch 1: `{summary['lambda_epoch_1']:.6f}`
- epoch 10: `{summary['lambda_epoch_10']:.6f}`
- epoch 15: `{summary['lambda_epoch_15']:.6f}`
- epoch 20: `{summary['lambda_epoch_20']:.6f}`

## Gradients
- prototype_grad_norm: `{summary['prototype_grad_norm']:.6f}`
- z_grad_norm: `{summary['z_grad_norm']:.6f}`

## Backward Compatibility
- old_a5b_forward_ok: `{summary['old_a5b_forward_ok']}`
- old_a5b_has_aux_module: `{summary['old_a5b_has_aux_module']}`
"""
    _write_text(output_dir / "D16R_A6_2A_HARD_PROTO_SEP_CHECK.md", report)
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
