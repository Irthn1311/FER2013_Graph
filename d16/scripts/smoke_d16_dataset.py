"""Smoke test D16 prior dataset and variable-size graph collate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.data.pixel_prior_dataset import D16PixelPriorDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
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
    node_counts = (batch.ptr[1:] - batch.ptr[:-1]).cpu()
    edge_counts = []
    src = batch.edge_index_cat[0].cpu()
    for start, end in zip(batch.ptr[:-1].cpu(), batch.ptr[1:].cpu()):
        edge_counts.append(int(((src >= int(start)) & (src < int(end))).sum().item()))
    edge_counts_t = torch.tensor(edge_counts, dtype=torch.float32)
    failures = []
    if batch.x_cat.ndim != 2:
        failures.append("x_cat must be [N,F]")
    if batch.edge_index_cat.shape[0] != 2:
        failures.append("edge_index_cat must be [2,E]")
    if batch.batch_index.numel() != batch.x_cat.size(0):
        failures.append("batch_index length mismatch")
    if batch.ptr.numel() != batch.y.numel() + 1:
        failures.append("ptr length mismatch")
    if int(node_counts.min().item()) <= 0:
        failures.append("empty graph in batch")
    if batch.valid_part_mask.ndim != 2 or batch.valid_anchor_mask.ndim != 2:
        failures.append("valid masks must be [B,P]/[B,A]")
    summary = {
        "decision": "D16_DATASET_SMOKE_PASS" if not failures else "D16_DATASET_SMOKE_FAIL",
        "num_graphs": int(batch.y.numel()),
        "x_cat_shape": list(batch.x_cat.shape),
        "edge_index_cat_shape": list(batch.edge_index_cat.shape),
        "node_count_min": int(node_counts.min().item()),
        "node_count_mean": float(node_counts.float().mean().item()),
        "node_count_max": int(node_counts.max().item()),
        "edge_count_min": int(edge_counts_t.min().item()),
        "edge_count_mean": float(edge_counts_t.mean().item()),
        "edge_count_max": int(edge_counts_t.max().item()),
        "fallback_count_in_batch": int((~batch.detected).sum().item()),
        "ptr": batch.ptr.tolist(),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
