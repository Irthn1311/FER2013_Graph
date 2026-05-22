"""Minimal D16 evaluation entrypoint for bootstrap wiring."""

from __future__ import annotations

import argparse
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
from d16.models.d16_model import D16Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    dataset = D16PixelPriorDataset(args.prior_dir, split=args.split, graph_mode=graph_cfg.get("graph_mode", "face_plus_context"))
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 8)), shuffle=False, collate_fn=collate_d16_graphs)
    batch = next(iter(loader))
    model = D16Model.from_config(cfg, input_dim=batch.x_cat.size(1))
    with torch.no_grad():
        logits = model(batch)["logits"]
    print({"split": args.split, "logits_shape": list(logits.shape)})


if __name__ == "__main__":
    main()
