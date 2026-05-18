"""Smoke test D13A forward, finite loss, region graph bounds, and backward."""

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
from evaluation.d13_diagnostics import compute_assignment_stats
from models.d13_hierarchical_reduction_model import D13HierarchicalReductionModel
from training.train_d13 import D13ReductionLoss
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
    for key in ("x", "edge_index", "edge_attr", "y"):
        if key not in batch:
            raise KeyError(f"D13 smoke requires batch field {key!r}")
    batch = move_to_device(batch, device)
    model = D13HierarchicalReductionModel.from_config(config.get("model", {})).to(device)
    criterion = D13ReductionLoss(config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    out = model(batch)
    loss_dict = criterion(out, batch["y"], batch)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("D13 smoke loss is non-finite")
    region_edge_index = out["region_edge_index"]
    region_batch = out["region_batch"]
    num_region_nodes = int(region_batch.numel())
    if region_edge_index.numel() > 0 and int(region_edge_index.max()) >= num_region_nodes:
        raise AssertionError(
            f"region_edge_index max {int(region_edge_index.max())} >= num_region_nodes {num_region_nodes}"
        )
    if region_batch.min().item() != 0:
        raise AssertionError("region_batch should start at 0")
    if region_batch.max().item() + 1 != batch["y"].numel():
        raise AssertionError("region_batch graph count does not match labels")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    report = {
        "status": "PASS",
        "device": str(device),
        "x_shape": _shape(batch["x"]),
        "edge_index_shape": _shape(batch["edge_index"]),
        "edge_attr_shape": _shape(batch["edge_attr"]),
        "h_pixel_shape": _shape(out.get("h_pixel")),
        "h_region_shape": _shape(out.get("h_region")),
        "logits_shape": _shape(out["logits"]),
        "region_edge_index_shape": _shape(region_edge_index),
        "region_edge_index_max": int(region_edge_index.max().item()) if region_edge_index.numel() else -1,
        "num_region_nodes": num_region_nodes,
        "region_batch_min": int(region_batch.min().item()),
        "region_batch_max": int(region_batch.max().item()),
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
        "loss_terms": {k: float(v.detach().cpu().item()) for k, v in loss_dict.items() if torch.is_tensor(v)},
        "aux_stats": compute_assignment_stats(out.get("aux", {})),
    }
    (output_dir / "d13_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# D13 Smoke Report", "", f"status: {report['status']}"]
    for key in (
        "x_shape",
        "h_pixel_shape",
        "h_region_shape",
        "logits_shape",
        "region_edge_index_shape",
        "region_edge_index_max",
        "num_region_nodes",
        "loss",
        "grad_norm",
    ):
        lines.append(f"- {key}: {report[key]}")
    (output_dir / "d13_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--max_train_batches", type=int, default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("training", {})["amp"] = False
    run_smoke(config, args.output_dir)


if __name__ == "__main__":
    main()

