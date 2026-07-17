"""Bounded structure-signal probe for ten OFIX18 C0/C2 best checkpoints."""

from __future__ import annotations

import argparse
import hashlib
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

SEEDS = (7, 21, 42, 84, 123)
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"


def run_name(cell: str, seed: int) -> str:
    stem = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    return f"d18_ofix18_{stem}_seed{seed}"


def run_dir(cell: str, seed: int, new_root: Path, seed42_root: Path) -> Path:
    return (seed42_root if seed == 42 else new_root) / run_name(cell, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--new_run_root", default="outputs/d18_runs/ofix18seed")
    parser.add_argument("--seed42_root", default="outputs/d18_runs/ofix18")
    parser.add_argument(
        "--output_dir",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/structure_probe",
    )
    parser.add_argument("--probe_count", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.sample_manifest)
    if len(manifest) != 715:
        raise RuntimeError(f"expected locked 715-image manifest, got {len(manifest)}")
    sample_hash = hashlib.sha256(
        manifest["sample_index"].to_numpy(dtype="int64").tobytes()
    ).hexdigest()
    if sample_hash != LOCKED_SAMPLE_SHA256:
        raise RuntimeError(f"locked sample hash mismatch: {sample_hash}")
    selected = manifest.iloc[: int(args.probe_count)]
    cache_root = Path(args.graph_cache_dir)
    graphs = []
    for image_id in selected["image_id"]:
        path = cache_root / "test" / f"{int(image_id):06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        graphs.append(load_d18_graph_cache(path))

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    rows = []
    for cell in ("C0", "C2"):
        for seed in SEEDS:
            source = run_dir(cell, seed, Path(args.new_run_root), Path(args.seed42_root))
            cfg = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
            payload = torch.load(
                source / "checkpoints/best.pt", map_location="cpu", weights_only=False
            )
            spec = {
                "run_id": source.name,
                "cell": cell,
                "checkpoint_type": "best",
                "checkpoint_path": source / "checkpoints/best.pt",
                "model_family": "d18",
                "cfg": cfg,
            }
            current = probe_checkpoint(spec, graphs, device, int(args.batch_size))
            for row in current:
                row.update(
                    {
                        "cell": cell,
                        "training_seed": seed,
                        "run_name": source.name,
                        "checkpoint_type": "best",
                        "checkpoint_epoch": int(payload.get("epoch", -1)),
                    }
                )
            rows.extend(current)
            print(
                json.dumps(
                    {"event": "multiseed_probe_done", "cell": cell, "seed": seed, "rows": len(current)}
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty and not bool(numeric.replace([float("inf"), float("-inf")], float("nan")).notna().all().all()):
        # Some unsupported hook fields may be NaN by design; infinities are never acceptable.
        if bool(numeric.isin([float("inf"), float("-inf")]).any().any()):
            raise RuntimeError("non-finite infinity in structure probe")
    frame.to_csv(output / "structure_signal_probe.csv", index=False)
    result = {
        "status": "COMPLETE",
        "cells": ["C0", "C2"],
        "training_seeds": list(SEEDS),
        "checkpoint_policy": "best_only_bounded_probe",
        "probe_count": len(graphs),
        "sample_index_sha256": sample_hash,
        "device": str(device),
        "rows": len(frame),
        "training_or_finetuning": False,
    }
    (output / "structure_probe_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
