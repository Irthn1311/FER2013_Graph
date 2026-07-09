"""Counterfactual prior audit for D17.

D17 graph construction should ignore prior tensors. The audit intentionally
mutates prior tensors while keeping image/label/sample id unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d17.data.collate import collate_epp_graphs
from d17.data.epp_graph_builder import build_epp_graph
from d17.models.epp_gnn import EPPGNN
from d17.training.train_d17 import build_dataset, evaluate, load_checkpoint, read_config, resolve_device


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


class D17PriorAuditDataset(Dataset):
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
        return build_epp_graph(prior, self.graph_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = read_config(run_dir / "resolved_config.yaml")
    prior_dir = Path(args.prior_dir or (cfg.get("data", {}) or {}).get("prior_dir"))
    device = resolve_device(args.device)
    first_ds = build_dataset(cfg, args.split, max_samples=2)
    first_batch = next(iter(DataLoader(first_ds, batch_size=2, collate_fn=collate_epp_graphs)))
    model = EPPGNN.from_config(cfg, input_dim=first_batch.x_cat.size(1), edge_attr_dim=first_batch.edge_attr_cat.size(1)).to(device)
    ckpt = args.checkpoint
    if ckpt in {"best", "last", "best_val_loss"}:
        ckpt = f"{ckpt}.pt"
    load_checkpoint(run_dir / "checkpoints" / ckpt, model, device=device)
    output_dir = Path(args.output_dir or Path("outputs/d17_analysis/ofix15/prior_audit") / run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        ds = D17PriorAuditDataset(prior_dir, args.split, cfg.get("graph", {}) or {}, variant)
        loader = DataLoader(ds, batch_size=int(args.batch_size or (cfg.get("training", {}) or {}).get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate_epp_graphs)
        row, _ = evaluate(model, loader, device, torch.nn.CrossEntropyLoss())
        rows.append({"variant": variant, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "loss": row["loss"]})
        print(f"{variant}: acc={row['accuracy']*100:.2f}% macro_f1={row['macro_f1']*100:.2f}%")
    with (output_dir / "prior_counterfactual_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    official = next((r for r in rows if r["variant"] == "official"), rows[0])
    lines = ["# D17 Prior Dependency Audit", "", "| variant | accuracy | macro-F1 | delta macro-F1 vs official |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['accuracy']*100:.2f}% | {r['macro_f1']*100:.2f}% | {(r['macro_f1']-official['macro_f1'])*100:.2f}pp |")
    (output_dir / "PRIOR_DEPENDENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

