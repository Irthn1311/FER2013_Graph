"""Evaluate a D17 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d17.data.collate import collate_epp_graphs
from d17.models.epp_gnn import EPPGNN
from d17.training.train_d17 import build_dataset, evaluate, load_checkpoint, read_config, resolve_device, write_eval_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = read_config(run_dir / "resolved_config.yaml")
    device = resolve_device(args.device)
    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=int((cfg.get("training", {}) or {}).get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate_epp_graphs)
    first_batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_epp_graphs)))
    model = EPPGNN.from_config(cfg, input_dim=first_batch.x_cat.size(1), edge_attr_dim=first_batch.edge_attr_cat.size(1)).to(device)
    ckpt = args.checkpoint
    if ckpt in {"best", "last", "best_val_loss"}:
        ckpt = f"{ckpt}.pt"
    load_checkpoint(run_dir / "checkpoints" / ckpt, model, device=device)
    row, detail = evaluate(model, loader, device, torch.nn.CrossEntropyLoss())
    write_eval_outputs(run_dir, f"{args.split}_eval_", row, detail)
    print(json.dumps({"accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "loss": row["loss"]}, indent=2))


if __name__ == "__main__":
    main()

