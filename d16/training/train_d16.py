"""Minimal D16 training entrypoint.

Full D16 training is intentionally out of scope for the bootstrap phase; this
entrypoint only wires the same dataset/model path used by smoke scripts.
"""

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


def load_config(path: str | Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    dataset = D16PixelPriorDataset(
        args.prior_dir,
        split="train",
        graph_mode=graph_cfg.get("graph_mode", data_cfg.get("graph_mode", "face_plus_context")),
        face_threshold=float(graph_cfg.get("face_threshold", 0.15)),
        context_pixels=int(graph_cfg.get("context_pixels", 2)),
    )
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 8)), shuffle=True, collate_fn=collate_d16_graphs)
    batch = next(iter(loader))
    model = D16Model.from_config(cfg, input_dim=batch.x_cat.size(1))
    out = model(batch)
    print({"logits_shape": list(out["logits"].shape), "note": "bootstrap only, no full training executed"})


if __name__ == "__main__":
    main()
