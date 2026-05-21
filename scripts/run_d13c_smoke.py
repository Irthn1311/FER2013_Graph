"""Smoke test D13C diagnostic forward/loss/backward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, save_config
from models.d13c_supcon_model import D13CSupConModel
from training.train_d13c import D13CDiagnosticLoss, _load_init_d13b
from training.trainer import move_to_device, set_seed


def _shape(value: Any):
    return list(value.shape) if torch.is_tensor(value) else None


def run_smoke(config: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    set_seed(int(config.get("training", {}).get("seed", 1)))
    device = resolve_device(config=config)
    loader = build_dataloader(config, split="train", shuffle=False)
    batch = move_to_device(next(iter(loader)), device)
    model = D13CSupConModel.from_config(config.get("model", {})).to(device)
    init_info = _load_init_d13b(model, config.get("model", {}).get("init_d13b_checkpoint"), device, output_dir)
    init_info["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = D13CDiagnosticLoss(config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    out = model(batch)
    loss_dict = criterion(out, batch["y"], batch)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("D13C smoke total loss is non-finite")
    for key in ("logits", "z_image", "z_proj", "slot_embeddings", "slot_attention"):
        value = out.get(key)
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise FloatingPointError(f"D13C smoke {key} missing or non-finite")
    projection_dim = int(config.get("model", {}).get("projection_dim", 64))
    if list(out["z_proj"].shape) != [int(batch["y"].shape[0]), projection_dim]:
        raise AssertionError(f"z_proj shape mismatch: {list(out['z_proj'].shape)} expected [B,{projection_dim}]")
    if list(out["logits"].shape) != [int(batch["y"].shape[0]), 7]:
        raise AssertionError(f"logits shape mismatch: {list(out['logits'].shape)}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    report = {
        "status": "PASS",
        "device": str(device),
        "slot_embeddings_shape": _shape(out["slot_embeddings"]),
        "z_image_shape": _shape(out["z_image"]),
        "z_proj_shape": _shape(out["z_proj"]),
        "logits_shape": _shape(out["logits"]),
        "ce_loss": float(loss_dict["loss_ce"].detach().cpu().item()),
        "supcon_loss": float(loss_dict["loss_supcon"].detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
        "loss_finite": bool(torch.isfinite(loss).detach().cpu().item()),
        "ce_loss_finite": bool(torch.isfinite(loss_dict["loss_ce"]).detach().cpu().item()),
        "supcon_loss_finite": bool(torch.isfinite(loss_dict["loss_supcon"]).detach().cpu().item()),
        "positive_pair_count": float(loss_dict["positive_pair_count"].detach().cpu().item()),
        "valid_supcon_anchor_count": float(loss_dict["valid_supcon_anchor_count"].detach().cpu().item()),
        "embedding_collapse_score": float(loss_dict["embedding_collapse_score"].detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
        "loaded_d13b_keys": int(init_info.get("loaded_d13b_keys", 0)),
        "init_checkpoint_path": init_info.get("init_checkpoint_path", ""),
        "trainable_parameters": int(init_info.get("trainable_parameters", 0)),
        "no_prototype": True,
        "no_motif_level_supcon": True,
        "no_motif_claim": True,
    }
    (output_dir / "d13c_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# D13C Smoke Report", "", f"status: {report['status']}"]
    for key in [
        "slot_embeddings_shape",
        "z_image_shape",
        "z_proj_shape",
        "logits_shape",
        "ce_loss",
        "supcon_loss",
        "total_loss",
        "positive_pair_count",
        "valid_supcon_anchor_count",
        "loaded_d13b_keys",
    ]:
        lines.append(f"- {key}: {report[key]}")
    lines.extend(["", "D13C diagnostic only. No prototype, no motif-level SupCon, and no motif claim."])
    (output_dir / "smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("training", {})["amp"] = False
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    run_smoke(config, args.output_dir)


if __name__ == "__main__":
    main()
