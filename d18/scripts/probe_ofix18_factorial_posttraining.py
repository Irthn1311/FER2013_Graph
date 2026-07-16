"""Bounded structure-signal probe for OFIX18 factorial best checkpoints.

This script performs inference diagnostics only. It never trains or mutates a
checkpoint or training config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.scripts.audit_ofix18_predecision import probe_checkpoint

RUNS = {
    "C0": ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c0_clean_control_seed42",
    "C1": ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c1_structure_dropedge_only_seed42",
    "C2": ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42",
    "C3": ROOT / "outputs/d18_runs/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_count", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.sample_manifest)
    if len(manifest) != 715:
        raise RuntimeError(f"expected locked 715-image manifest, got {len(manifest)}")
    selected = manifest.iloc[: int(args.probe_count)]
    cache_root = Path(args.graph_cache_dir)
    graphs = []
    for image_id in selected["image_id"]:
        path = cache_root / "test" / f"{int(image_id):06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        graphs.append(load_d18_graph_cache(path))

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows = []
    for cell, run_dir in RUNS.items():
        cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        payload = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        spec = {
            "run_id": run_dir.name,
            "cell": cell,
            "checkpoint_type": "best",
            "checkpoint_path": run_dir / "checkpoints/best.pt",
            "model_family": "d18",
            "cfg": cfg,
        }
        current = probe_checkpoint(spec, graphs, device, int(args.batch_size))
        for row in current:
            row["cell"] = cell
            row["checkpoint_type"] = "best"
            row["checkpoint_epoch"] = int(payload.get("epoch", -1))
        rows.extend(current)
        print(json.dumps({"event": "probe_done", "cell": cell, "rows": len(current)}), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "10_structure_signal_comparison.csv", index=False)
    result = {
        "status": "COMPLETE",
        "cells": sorted(RUNS),
        "checkpoint_policy": "best_only_bounded_probe",
        "probe_count": len(graphs),
        "device": str(device),
        "rows": len(frame),
        "last_checkpoint_probe": "SKIPPED_BOUNDED_RUNTIME",
    }
    (output / "structure_probe_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
