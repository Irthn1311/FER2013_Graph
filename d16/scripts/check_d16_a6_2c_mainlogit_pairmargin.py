"""Checker for D16R A6-2c main-logit pair-margin loss."""

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
from d16.losses.main_logit_pair_margin import (
    build_main_logit_pair_margin_loss,
    main_logit_pair_margin_lambda,
)
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


def _find_batch_with_active_pairs(ds, max_scan: int = 5000):
    selected = {}
    for index in range(min(len(ds), int(max_scan))):
        graph = ds[index]
        label = int(graph.y.detach().cpu().view(-1)[0].item())
        if label in {2, 4, 6} and label not in selected:
            selected[label] = graph
        if {2, 4, 6}.issubset(selected):
            return collate_d16_graphs([selected[2], selected[4], selected[6]])
    raise RuntimeError("Could not find Fear, Sad, and Neutral samples for A6-2c checker")


def _grad_norm(params) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(total)


def _forward_ok(cfg: Dict[str, Any], prior_dir: Path, device: torch.device, attach_pairwise: bool = False) -> bool:
    ds = build_dataset(cfg, prior_dir, "train")
    first = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=int(first.x_cat.size(1))).to(device)
    if attach_pairwise:
        aux = attach_pairwise_hard_relation_loss_if_needed(
            model,
            cfg.get("loss", {}) or {},
            embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
        )
        if aux is not None:
            aux.to(device)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d16_graphs))).to(device)
    with torch.no_grad():
        out = model(batch)
    return tuple(out["logits"].shape) == (2, 7)


def run_check(
    config_path: Path,
    prior_dir: Path,
    output_dir: Path,
    old_config_path: Path | None,
    old_a6_2b_config_path: Path | None,
    device_name: str,
) -> Dict[str, Any]:
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
    hard_proto = attach_hard_proto_loss_if_needed(model, loss_cfg, embedding_dim=embedding_dim)
    pairwise = attach_pairwise_hard_relation_loss_if_needed(model, loss_cfg, embedding_dim=embedding_dim)
    if hard_proto is not None or hasattr(model, "hard_proto_sep_loss"):
        raise AssertionError("A6-2c must not attach global hard prototype loss")
    if pairwise is not None or hasattr(model, "pairwise_hard_relation_loss"):
        raise AssertionError("A6-2c must not attach pairwise auxiliary heads")
    margin_loss_fn = build_main_logit_pair_margin_loss(loss_cfg)
    if margin_loss_fn is None:
        raise RuntimeError("A6-2c checker expected main-logit pair-margin loss to be enabled")
    margin_loss_fn.to(device)

    batch = _find_batch_with_active_pairs(ds).to(device)
    model.train()
    out = model(batch)
    logits = out["logits"]
    logits.retain_grad()
    ce_loss, _ = _weighted_ce_loss(logits, batch.y, batch, loss_cfg)
    aux = margin_loss_fn(logits, batch.y)
    lambda_epoch_1 = main_logit_pair_margin_lambda(loss_cfg, 1)
    lambda_epoch_15 = main_logit_pair_margin_lambda(loss_cfg, 15)
    lambda_epoch_25 = main_logit_pair_margin_lambda(loss_cfg, 25)
    lambda_epoch_35 = main_logit_pair_margin_lambda(loss_cfg, 35)
    loss = ce_loss + lambda_epoch_35 * aux["main_logit_pair_margin_loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("A6-2c combined loss is not finite")
    model.zero_grad(set_to_none=True)
    loss.backward()
    classifier_grad_norm = _grad_norm(model.classifier.parameters()) if model.classifier is not None else 0.0
    logits_grad_norm = float(logits.grad.detach().float().norm().cpu().item()) if logits.grad is not None else 0.0

    no_pair_y = torch.full_like(batch.y, 3)
    no_pair = margin_loss_fn(logits.detach(), no_pair_y)
    no_pair_loss = float(no_pair["main_logit_pair_margin_loss"].detach().cpu().item())
    no_pair_available = int(no_pair["pair_margin_available_count"].detach().cpu().item())

    old_a5b_ok = True
    if old_config_path is not None and old_config_path.exists():
        old_a5b_ok = _forward_ok(_read_config(old_config_path), prior_dir, device, attach_pairwise=False)
    old_a6_2b_ok = True
    if old_a6_2b_config_path is not None and old_a6_2b_config_path.exists():
        old_a6_2b_ok = _forward_ok(_read_config(old_a6_2b_config_path), prior_dir, device, attach_pairwise=True)

    summary = {
        "decision": "A6_2C_MAINLOGIT_PAIRMARGIN_CHECK_PASS",
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "input_dim": int(input_dim),
        "logits_shape": list(logits.shape),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "pair_margin_loss_total": float(aux["main_logit_pair_margin_loss"].detach().cpu().item()),
        "pair_margin_loss_fear_sad": float(aux["pair_margin_loss_fear_sad"].detach().cpu().item()),
        "pair_margin_loss_sad_neutral": float(aux["pair_margin_loss_sad_neutral"].detach().cpu().item()),
        "pair_margin_loss_neutral_sad": float(aux["pair_margin_loss_neutral_sad"].detach().cpu().item()),
        "pair_count_fear_sad": int(aux["pair_margin_count_fear_sad"].detach().cpu().item()),
        "pair_count_sad_neutral": int(aux["pair_margin_count_sad_neutral"].detach().cpu().item()),
        "pair_count_neutral_sad": int(aux["pair_margin_count_neutral_sad"].detach().cpu().item()),
        "mean_margin_violation_fear_sad": float(aux["mean_margin_violation_fear_sad"].detach().cpu().item()),
        "mean_margin_violation_sad_neutral": float(aux["mean_margin_violation_sad_neutral"].detach().cpu().item()),
        "mean_margin_violation_neutral_sad": float(aux["mean_margin_violation_neutral_sad"].detach().cpu().item()),
        "lambda_epoch_1": float(lambda_epoch_1),
        "lambda_epoch_15": float(lambda_epoch_15),
        "lambda_epoch_25": float(lambda_epoch_25),
        "lambda_epoch_35": float(lambda_epoch_35),
        "no_pair_loss": no_pair_loss,
        "no_pair_available_count": no_pair_available,
        "classifier_grad_norm": classifier_grad_norm,
        "logits_grad_norm": logits_grad_norm,
        "loss_parameter_count": sum(p.numel() for p in margin_loss_fn.parameters()),
        "model_has_hard_proto_module": hasattr(model, "hard_proto_sep_loss"),
        "model_has_pairwise_head_module": hasattr(model, "pairwise_hard_relation_loss"),
        "old_a5b_forward_ok": bool(old_a5b_ok),
        "old_a6_2b_forward_ok": bool(old_a6_2b_ok),
    }
    if tuple(logits.shape) != (int(batch.num_graphs), 7):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not math.isfinite(summary["ce_loss"]) or not math.isfinite(summary["pair_margin_loss_total"]):
        raise AssertionError("CE or pair-margin loss is non-finite")
    if summary["pair_count_fear_sad"] <= 0 or summary["pair_count_sad_neutral"] <= 0 or summary["pair_count_neutral_sad"] <= 0:
        raise AssertionError("Checker batch must contain samples for all active pairs")
    if abs(no_pair_loss) > 1e-8 or no_pair_available != 0:
        raise AssertionError("No-pair batch should produce zero pair-margin loss and zero available pair count")
    if classifier_grad_norm <= 0.0 or logits_grad_norm <= 0.0:
        raise AssertionError("Expected gradients to flow through main logits and classifier")
    if summary["loss_parameter_count"] != 0 or summary["model_has_hard_proto_module"] or summary["model_has_pairwise_head_module"]:
        raise AssertionError("A6-2c must not add auxiliary trainable modules")
    if not old_a5b_ok or not old_a6_2b_ok:
        raise AssertionError("Backward compatibility check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a6_2c_mainlogit_pairmargin_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# D16R A6-2c Main-Logit Pair-Margin Check

## Decision
`{summary['decision']}`

## Forward
- logits_shape: `{summary['logits_shape']}`
- input_dim: `{summary['input_dim']}`

## Loss
- ce_loss: `{summary['ce_loss']:.6f}`
- pair_margin_loss_total: `{summary['pair_margin_loss_total']:.6f}`
- pair_margin_loss_fear_sad: `{summary['pair_margin_loss_fear_sad']:.6f}`
- pair_margin_loss_sad_neutral: `{summary['pair_margin_loss_sad_neutral']:.6f}`
- pair_margin_loss_neutral_sad: `{summary['pair_margin_loss_neutral_sad']:.6f}`
- no_pair_loss: `{summary['no_pair_loss']:.6f}`

## Lambda Schedule
- epoch 1: `{summary['lambda_epoch_1']:.6f}`
- epoch 15: `{summary['lambda_epoch_15']:.6f}`
- epoch 25: `{summary['lambda_epoch_25']:.6f}`
- epoch 35: `{summary['lambda_epoch_35']:.6f}`

## Gradients
- logits_grad_norm: `{summary['logits_grad_norm']:.6f}`
- classifier_grad_norm: `{summary['classifier_grad_norm']:.6f}`

## No Extra Modules
- loss_parameter_count: `{summary['loss_parameter_count']}`
- model_has_hard_proto_module: `{summary['model_has_hard_proto_module']}`
- model_has_pairwise_head_module: `{summary['model_has_pairwise_head_module']}`

## Backward Compatibility
- old_a5b_forward_ok: `{summary['old_a5b_forward_ok']}`
- old_a6_2b_forward_ok: `{summary['old_a6_2b_forward_ok']}`
"""
    _write_text(output_dir / "D16R_A6_2C_MAINLOGIT_PAIRMARGIN_CHECK.md", report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    parser.add_argument("--old_config", default="configs/d16/main_branch/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150.yaml")
    parser.add_argument("--old_a6_2b_config", default="configs/d16/main_branch/d16r_a6_2b_pairwise_relation_a5b_ce_seed42_accmon_150.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_check(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        old_config_path=Path(args.old_config) if args.old_config else None,
        old_a6_2b_config_path=Path(args.old_a6_2b_config) if args.old_a6_2b_config else None,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
