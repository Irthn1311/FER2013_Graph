"""Smoke test D13B diagnostic model forward/loss/backward."""

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
from models.d13b_motif_slot_model import D13BMotifSlotModel
from training.train_d13b import D13BDiagnosticLoss, _load_init_d13a
from training.trainer import move_to_device, set_seed


def _shape(value: Any):
    return list(value.shape) if torch.is_tensor(value) else None


def run_smoke(config: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(config=config)
    loader = build_dataloader(config, split="train", shuffle=False)
    batch = next(iter(loader))
    batch = move_to_device(batch, device)
    model = D13BMotifSlotModel.from_config(config.get("model", {})).to(device)
    init_info = _load_init_d13a(model, config.get("model", {}).get("init_d13a_checkpoint"), device, output_dir)
    criterion = D13BDiagnosticLoss(config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    out = model(batch)
    loss_dict = criterion(out, batch["y"], batch)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("D13B smoke loss is non-finite")
    slot_attention = out["slot_attention"]
    if slot_attention.ndim != 3:
        raise AssertionError(f"slot_attention must be [B,M,K], got {tuple(slot_attention.shape)}")
    if not torch.isfinite(slot_attention).all():
        raise FloatingPointError("slot_attention is non-finite")
    sums = slot_attention.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
        raise AssertionError("slot_attention must sum to 1 over regions per slot")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    report = {
        "status": "PASS",
        "device": str(device),
        "x_shape": _shape(batch["x"]),
        "h_region_shape": _shape(out.get("h_region")),
        "slot_embeddings_shape": _shape(out.get("slot_embeddings")),
        "slot_attention_shape": _shape(slot_attention),
        "logits_shape": _shape(out["logits"]),
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
        "slot_attention_sum_min": float(sums.min().detach().cpu().item()),
        "slot_attention_sum_max": float(sums.max().detach().cpu().item()),
        "effective_slots": float(out["aux"]["effective_slots"].detach().cpu().item()),
        "slot_overlap": float(out["aux"]["slot_overlap"].detach().cpu().item()),
        "loaded_d13a_keys": int(init_info.get("loaded_d13a_keys", 0)),
        "init_checkpoint_path": init_info.get("init_checkpoint_path", ""),
        "loss_terms": {k: float(v.detach().cpu().item()) for k, v in loss_dict.items() if torch.is_tensor(v)},
    }
    (output_dir / "d13b_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# D13B Smoke Report", "", f"status: {report['status']}"]
    for key in [
        "x_shape",
        "h_region_shape",
        "slot_embeddings_shape",
        "slot_attention_shape",
        "logits_shape",
        "loss",
        "grad_norm",
        "effective_slots",
        "slot_overlap",
        "loaded_d13a_keys",
    ]:
        lines.append(f"- {key}: {report[key]}")
    lines.append("")
    lines.append("No motif or semantic-region claim is made.")
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
        config.setdefault("runtime", {})["device"] = args.device
    run_smoke(config, args.output_dir)


if __name__ == "__main__":
    main()
