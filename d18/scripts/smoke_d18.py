"""Smoke-test D18 configs on tiny train/val batches."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_dataset import StructurePixelDataset
from d18.models.structure_gnn import StructureGNN


def read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def smoke_one(config_path: Path, device: torch.device) -> Dict[str, Any]:
    cfg = read_config(config_path)
    data_cfg = cfg.get("data", {}) or {}
    ds = StructurePixelDataset(
        prior_dir=data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"),
        split="train",
        graph=cfg.get("graph", {}) or {},
        max_samples=2,
    )
    val_ds = StructurePixelDataset(
        prior_dir=data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"),
        split="val",
        graph=cfg.get("graph", {}) or {},
        max_samples=2,
    )
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d18_graphs))).to(device)
    val_batch = next(iter(DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_d18_graphs))).to(device)
    model = StructureGNN.from_config(cfg, input_dim=batch.x_cat.size(1), edge_attr_dim=batch.edge_attr_cat.size(1)).to(device)
    out = model(batch)
    logits = out["logits"]
    loss = torch.nn.CrossEntropyLoss()(logits, batch.y)
    with torch.no_grad():
        val_logits = model(val_batch)["logits"]
        val_loss = torch.nn.CrossEntropyLoss()(val_logits, val_batch.y)
    return {
        "config": str(config_path),
        "run_name": cfg.get("run_name", config_path.stem),
        "node_dim": int(batch.x_cat.size(1)),
        "edge_dim": int(batch.edge_attr_cat.size(1)),
        "train_logits_shape": tuple(int(x) for x in logits.shape),
        "val_logits_shape": tuple(int(x) for x in val_logits.shape),
        "loss": float(loss.detach().cpu()),
        "val_loss": float(val_loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item() and torch.isfinite(val_loss).item()),
        "local_edge_count_mean": float(batch.local_edge_count.float().mean().cpu()),
        "knn_edge_count_mean": float(batch.knn_edge_count.float().mean().cpu()),
        "structure_edge_count_mean": float(batch.structure_edge_count.float().mean().cpu()),
        "total_edge_count_mean": float(batch.total_edge_count.float().mean().cpu()),
        "node_feature_names": list(batch.node_feature_names),
        "edge_feature_names": list(batch.edge_feature_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/d18_analysis/ofix16/02_d18_smoke.md")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, Any]] = []
    for text in args.configs:
        rows.append(smoke_one(Path(text), device))
    lines = [
        "# D18 Smoke Test",
        "",
        f"device: `{device}`",
        "",
        "| run | node_dim | edge_dim | logits | loss finite | local edges | kNN edges | structure edges | total edges |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    ok = True
    for row in rows:
        ok = ok and row["node_dim"] == 10 and row["train_logits_shape"] == (2, 7) and row["val_logits_shape"] == (2, 7) and row["loss_finite"]
        lines.append(
            f"| {row['run_name']} | {row['node_dim']} | {row['edge_dim']} | {row['train_logits_shape']} | {row['loss_finite']} | "
            f"{row['local_edge_count_mean']:.1f} | {row['knn_edge_count_mean']:.1f} | {row['structure_edge_count_mean']:.1f} | {row['total_edge_count_mean']:.1f} |"
        )
    lines.extend([
        "",
        "## Feature Schema",
        "",
        f"node features: `{', '.join(rows[0]['node_feature_names']) if rows else ''}`",
        "",
        "Edge schemas:",
    ])
    for row in rows:
        lines.append(f"- `{row['run_name']}`: `{', '.join(row['edge_feature_names'])}`")
    lines.extend(["", f"Smoke status: `{'PASS' if ok else 'FAIL'}`"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output": str(output), "status": "PASS" if ok else "FAIL"})
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
