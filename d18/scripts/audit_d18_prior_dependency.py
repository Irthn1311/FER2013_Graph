"""Counterfactual prior audit for D18.

D18 graph construction uses priors only for structure-edge topology. The audit mutates prior tensors while keeping image/label/sample id unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.collate import collate_d18_graphs
from d18.data.structure_graph_builder import build_structure_graph
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import build_dataset, evaluate, load_checkpoint, read_config, resolve_device


VARIANTS = ["official", "zero_prior", "shuffle_prior", "forced_fallback"]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _copy(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in prior.items()}


def _zero_prior(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = _copy(prior)
    for key in ("face_mask", "part_soft_masks", "micro_anchor_maps", "distance_maps", "landmark_xy_48", "valid_part_mask", "valid_anchor_mask"):
        if key in out:
            out[key] = np.zeros_like(out[key])
    if "landmark_missing_flag" in out:
        out["landmark_missing_flag"] = np.asarray(1, dtype=np.int64)
    return out


def _shuffle_prior(base: Dict[str, np.ndarray], donor: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = _copy(base)
    for key in ("face_mask", "part_soft_masks", "micro_anchor_maps", "distance_maps", "landmark_xy_48", "valid_part_mask", "valid_anchor_mask"):
        if key in donor:
            out[key] = np.array(donor[key], copy=True)
    return out


class D18PriorAuditDataset(Dataset):
    def __init__(self, prior_dir: Path, split: str, graph_cfg: Dict[str, Any], variant: str, seed: int = 42) -> None:
        self.split_dir = prior_dir / split
        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No files in {self.split_dir}")
        self.graph_cfg = dict(graph_cfg)
        self.variant = str(variant)
        rng = np.random.default_rng(seed)
        self.shuffle_indices = rng.permutation(len(self.files))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        prior = _load_npz(self.files[int(index)])
        if self.variant == "zero_prior":
            prior = _zero_prior(prior)
        elif self.variant == "shuffle_prior":
            prior = _shuffle_prior(prior, _load_npz(self.files[int(self.shuffle_indices[int(index)])]))
        elif self.variant == "forced_fallback":
            prior = _zero_prior(prior)
            if "detected" in prior:
                prior["detected"] = np.asarray(False)
        elif self.variant != "official":
            raise ValueError(f"Unknown variant={self.variant}")
        return build_structure_graph(prior, self.graph_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--graph_cache_dir", default=None)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = read_config(run_dir / "resolved_config.yaml")
    prior_dir = Path(args.prior_dir or (cfg.get("data", {}) or {}).get("prior_dir"))
    cfg.setdefault("data", {})["prior_dir"] = str(prior_dir)
    if args.graph_cache_dir:
        cfg.setdefault("graph", {}).setdefault("cache", {})["dir"] = str(Path(args.graph_cache_dir))
    device = resolve_device(args.device)
    first_ds = build_dataset(cfg, args.split, max_samples=2)
    first_batch = next(iter(DataLoader(first_ds, batch_size=2, collate_fn=collate_d18_graphs)))
    model = StructureGNN.from_config(cfg, input_dim=first_batch.x_cat.size(1), edge_attr_dim=first_batch.edge_attr_cat.size(1)).to(device)
    ckpt = args.checkpoint
    if ckpt in {"best", "last", "best_val_loss"}:
        ckpt = f"{ckpt}.pt"
    load_checkpoint(run_dir / "checkpoints" / ckpt, model, device=device)
    output_dir = Path(args.output_dir or Path("outputs/d18_analysis/ofix16/prior_audit") / run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        ds = D18PriorAuditDataset(prior_dir, args.split, cfg.get("graph", {}) or {}, variant)
        loader = DataLoader(ds, batch_size=int(args.batch_size or (cfg.get("training", {}) or {}).get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate_d18_graphs)
        print(json.dumps({"event": "d18_prior_audit_start", "variant": variant, "split": args.split, "samples": len(ds), "batches": len(loader)}), flush=True)
        start = time.perf_counter()
        if int(args.progress_interval or 0) <= 0:
            row, _ = evaluate(model, loader, device, torch.nn.CrossEntropyLoss())
        else:
            # Reuse the trainer evaluator behavior, but emit lightweight progress by wrapping the dataset loop here.
            # The D18 audit intentionally rebuilds counterfactual graphs online; without this, Kaggle appears idle.
            y_true, y_pred, detected, sample_index = [], [], [], []
            loss_sum = 0.0
            count = 0
            node_counts, local_counts, knn_counts, structure_counts, edge_counts = [], [], [], [], []
            loss_fn = torch.nn.CrossEntropyLoss()
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader, start=1):
                    batch = batch.to(device)
                    logits = model(batch)["logits"]
                    loss = loss_fn(logits, batch.y)
                    pred = logits.argmax(dim=1)
                    bs = int(batch.y.numel())
                    loss_sum += float(loss.item()) * bs
                    count += bs
                    y_true.extend(batch.y.detach().cpu().tolist())
                    y_pred.extend(pred.detach().cpu().tolist())
                    detected.extend(batch.detected.detach().cpu().tolist())
                    sample_index.extend(batch.sample_index.detach().cpu().tolist())
                    node_counts.extend(((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()).tolist())
                    local_counts.extend(batch.local_edge_count.detach().cpu().numpy().tolist())
                    knn_counts.extend(batch.knn_edge_count.detach().cpu().numpy().tolist())
                    structure_counts.extend(batch.structure_edge_count.detach().cpu().numpy().tolist())
                    edge_counts.extend(batch.total_edge_count.detach().cpu().numpy().tolist())
                    if batch_idx == 1 or batch_idx % int(args.progress_interval) == 0 or batch_idx == len(loader):
                        print(json.dumps({"event": "d18_prior_audit_progress", "variant": variant, "batch": batch_idx, "total_batches": len(loader), "elapsed_sec": time.perf_counter() - start}), flush=True)
            from d18.training.train_d18 import metrics_from_predictions
            row = metrics_from_predictions(y_true, y_pred, loss_sum, count)
            row.update({
                "node_count_mean": float(np.mean(node_counts)) if node_counts else float("nan"),
                "local_edge_count_mean": float(np.mean(local_counts)) if local_counts else float("nan"),
                "knn_edge_count_mean": float(np.mean(knn_counts)) if knn_counts else float("nan"),
                "structure_edge_count_mean": float(np.mean(structure_counts)) if structure_counts else float("nan"),
                "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
            })
        rows.append({"variant": variant, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "loss": row["loss"]})
        print(json.dumps({"event": "d18_prior_audit_done", "variant": variant, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "elapsed_sec": time.perf_counter() - start}), flush=True)
    with (output_dir / "prior_counterfactual_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    official = next((r for r in rows if r["variant"] == "official"), rows[0])
    lines = ["# D18 Prior Dependency Audit", "", "| variant | accuracy | macro-F1 | delta macro-F1 vs official |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['accuracy']*100:.2f}% | {r['macro_f1']*100:.2f}% | {(r['macro_f1']-official['macro_f1'])*100:.2f}pp |")
    (output_dir / "PRIOR_DEPENDENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()


