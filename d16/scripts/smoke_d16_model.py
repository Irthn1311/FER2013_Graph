"""Smoke test D16 model forward/loss/backward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.data.pixel_prior_dataset import D16PixelPriorDataset
from d16.models.d16_model import D16Model


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[D16 smoke] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    dataset = D16PixelPriorDataset(
        args.prior_dir,
        split=args.split,
        graph_mode=graph_cfg.get("graph_mode", data_cfg.get("graph_mode", "face_plus_context")),
        face_threshold=float(graph_cfg.get("face_threshold", 0.15)),
        context_pixels=int(graph_cfg.get("context_pixels", 2)),
        max_samples=int(data_cfg.get("smoke_max_samples", 16)),
    )
    loader = DataLoader(dataset, batch_size=min(int(data_cfg.get("batch_size", 8)), len(dataset)), shuffle=False, collate_fn=collate_d16_graphs)
    batch = next(iter(loader))
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    batch = batch.to(device)
    model = D16Model.from_config(cfg, input_dim=batch.x_cat.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out = model(batch)
    logits = out["logits"]
    loss = F.cross_entropy(logits, batch.y)
    loss.backward()
    optimizer.step()
    peak_memory_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    failures = []
    if list(logits.shape) != [batch.num_graphs, 7]:
        failures.append(f"logits shape {list(logits.shape)} != [{batch.num_graphs}, 7]")
    if not torch.isfinite(loss).item():
        failures.append("loss is not finite")
    summary = {
        "decision": "D16_MODEL_SMOKE_PASS" if not failures else "D16_MODEL_SMOKE_FAIL",
        "device": str(device),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu().item()),
        "num_graphs": batch.num_graphs,
        "x_cat_shape": list(batch.x_cat.shape),
        "peak_memory_reserved_mb": peak_memory_mb,
        "fallback_count_in_batch": int((~batch.detected).sum().detach().cpu().item()),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
