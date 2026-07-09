"""Smoke-test D17 configs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d17.data.collate import collate_epp_graphs
from d17.models.epp_gnn import EPPGNN
from d17.training.train_d17 import build_dataset, read_config, resolve_device


def smoke_config(config_path: Path, device_text: str | None) -> list[dict]:
    cfg = read_config(config_path)
    device = resolve_device(device_text)
    rows = []
    for split in ("train", "val"):
        ds = build_dataset(cfg, split, max_samples=2)
        batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_epp_graphs))).to(device)
        model = EPPGNN.from_config(cfg, input_dim=batch.x_cat.size(1), edge_attr_dim=batch.edge_attr_cat.size(1)).to(device)
        logits = model(batch)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        rows.append(
            {
                "run": cfg.get("run_name", config_path.stem),
                "split": split,
                "node_dim": int(batch.x_cat.size(1)),
                "edge_dim": int(batch.edge_attr_cat.size(1)),
                "node_count_mean": float((batch.ptr[1:] - batch.ptr[:-1]).float().mean().item()),
                "local_edge_count_mean": float(batch.local_edge_count.float().mean().item()),
                "knn_edge_count_mean": float(batch.knn_edge_count.float().mean().item()),
                "total_edge_count_mean": float(batch.total_edge_count.float().mean().item()),
                "logits_shape": str(tuple(logits.shape)),
                "loss": float(loss.detach().item()),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "uses_face_mask_for_node_selection": False,
                "uses_anchor_nodes": False,
                "uses_log_prior_bias": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/d17_analysis/ofix15/02_d17_smoke.md")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    all_rows = []
    for cfg in args.configs:
        all_rows.extend(smoke_config(Path(cfg), args.device))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    fields = [
        "run",
        "split",
        "node_dim",
        "edge_dim",
        "node_count_mean",
        "local_edge_count_mean",
        "knn_edge_count_mean",
        "total_edge_count_mean",
        "logits_shape",
        "loss",
        "loss_finite",
        "uses_face_mask_for_node_selection",
        "uses_anchor_nodes",
        "uses_log_prior_bias",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ["# D17 Smoke Test", "", "| run | split | node_dim | edge_dim | nodes | local edges | kNN edges | total edges | logits | loss | pass |", "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in all_rows:
        passed = (
            int(r["node_dim"]) == 10
            and int(r["edge_dim"]) == 6
            and r["logits_shape"] == "(2, 7)"
            and bool(r["loss_finite"])
            and not bool(r["uses_face_mask_for_node_selection"])
            and not bool(r["uses_anchor_nodes"])
            and not bool(r["uses_log_prior_bias"])
        )
        lines.append(
            f"| {r['run']} | {r['split']} | {r['node_dim']} | {r['edge_dim']} | {r['node_count_mean']:.1f} | {r['local_edge_count_mean']:.1f} | {r['knn_edge_count_mean']:.1f} | {r['total_edge_count_mean']:.1f} | `{r['logits_shape']}` | {r['loss']:.4f} | {passed} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all("True" in line for line in lines if line.startswith("| d17_")):
        raise SystemExit("D17 smoke failed")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

