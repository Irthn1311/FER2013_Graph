"""Finalize an interrupted OFIX18 audit from persisted expensive artifacts.

Inference only: this script never invokes an optimizer or changes checkpoints,
training configs, builders, or existing audit CSV inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.scripts import audit_ofix18_predecision as audit

REQUIRED_INPUTS = [
    "01_run_manifest.csv", "03_graph_mode_statistics.csv",
    "04_graph_hash_audit.csv", "05_prediction_counterfactuals.csv",
    "05_prediction_counterfactuals_summary.csv", "06_per_sample_sensitivity.csv",
    "07_edge_ablation_matrix.csv", "12_node_support_information.csv",
    "sample_manifest.csv", ".audit_in_progress",
]


def profile_map(specs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    base_cfg = next(s["cfg"] for s in specs if s["graph_profile"] == "base6_structure")
    purified_cfg = next(s["cfg"] for s in specs if s["graph_profile"] == "purified_structure")
    return {
        "base6_structure": {"cfg": base_cfg, "cache_root": PROJECT_ROOT / audit.PROFILE_CACHE["base6_structure"]},
        "purified_structure": {"cfg": purified_cfg, "cache_root": PROJECT_ROOT / audit.PROFILE_CACHE["purified_structure"]},
        "d17_pixel_only": {"cfg": base_cfg, "cache_root": PROJECT_ROOT / audit.PROFILE_CACHE["d17_pixel_only"]},
    }


def graph_for_mode(spec: Mapping[str, Any], profile: Mapping[str, Any], prior_file: Path,
                   donor_file: Path, mode: str) -> Any:
    graph = load_d18_graph_cache(Path(profile["cache_root"]) / "test" / prior_file.name)
    if mode != "official" and spec["model_family"] != "d17":
        base = audit.load_prior(prior_file)
        donor = audit.load_prior(donor_file) if mode == "shuffle_prior" else None
        graph = audit.rebuild_structure_from_cache(
            graph, audit.mode_prior(base, mode, donor),
            profile["cfg"].get("graph", {}) or {},
        )
    return audit.without_structure(graph) if spec["model_family"] == "d17" else graph


@torch.no_grad()
def collect_embeddings(specs: Sequence[Dict[str, Any]], files: Sequence[Path],
                       donor_indices: np.ndarray, profiles: Mapping[str, Dict[str, Any]],
                       device: torch.device, batch_size: int) -> Dict[Tuple[str, str], np.ndarray]:
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    for spec in specs:
        model = audit.load_model(spec, device)
        profile = profiles[spec["graph_profile"]]
        modes = ["official"] if spec["model_family"] == "d17" else audit.MODES
        for mode in modes:
            chunks = []
            for start in range(0, len(files), batch_size):
                stop = min(start + batch_size, len(files))
                graphs = [graph_for_mode(spec, profile, files[i], files[int(donor_indices[i])], mode)
                          for i in range(start, stop)]
                _, z = audit.infer_graphs(model, graphs, device, batch_size)
                chunks.append(z)
            embeddings[(spec["run_id"], mode)] = np.concatenate(chunks, axis=0)
            audit.emit("finalize_embeddings_done", run_id=spec["run_id"], mode=mode, samples=len(files))
        if spec["model_family"] == "d17":
            official = embeddings[(spec["run_id"], "official")]
            for mode in audit.MODES[1:]:
                embeddings[(spec["run_id"], mode)] = official.copy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return embeddings


def collect_probe(specs: Sequence[Dict[str, Any]], files: Sequence[Path],
                  profiles: Mapping[str, Dict[str, Any]], device: torch.device,
                  batch_size: int, probe_count: int) -> pd.DataFrame:
    rows = []
    for spec in specs:
        profile = profiles[spec["graph_profile"]]
        graphs = [graph_for_mode(spec, profile, path, path, "official") for path in files[:probe_count]]
        rows.extend(audit.probe_checkpoint(spec, graphs, device, batch_size))
        audit.emit("finalize_probe_done", run_id=spec["run_id"], samples=len(graphs))
    return pd.DataFrame(rows)


def write_completion_commands(output_dir: Path, args: argparse.Namespace) -> None:
    py = sys.executable
    full = (f'"{py}" -B d18/scripts/audit_ofix18_predecision.py '
            f'--prior_dir "{args.prior_dir}" --output_dir "{args.output_dir}" '
            f'--per_regular_class 110 --batch_size {args.batch_size} '
            f'--probe_count {args.probe_count} --device {args.device}')
    finalize = (f'"{py}" -B d18/scripts/finalize_ofix18_predecision_audit.py '
                f'--prior_dir "{args.prior_dir}" --output_dir "{args.output_dir}" '
                f'--batch_size {args.batch_size} --probe_count {args.probe_count} --device {args.device}')
    lines = [
        "# Exact Run Commands", "",
        "The full command produced all persisted expensive stages before the probe failure. The finalize command resumed from those immutable CSV artifacts.",
        "", "```powershell", full, finalize, "```", "",
    ]
    (output_dir / "16_run_commands.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_predecision_audit")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--probe_count", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(audit.SEED)
    np.random.seed(audit.SEED)
    torch.manual_seed(audit.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(audit.SEED)

    output_dir = PROJECT_ROOT / args.output_dir
    missing = [name for name in REQUIRED_INPUTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot safely finalize; missing persisted inputs: {missing}")
    if (output_dir / "AUDIT_COMPLETE.json").exists():
        raise FileExistsError(f"Audit already complete: {output_dir}")

    specs, failures = audit.prepare_specs()
    if failures or len(specs) != len(audit.RUN_SPECS):
        raise RuntimeError("Checkpoint/config matching failed: " + " | ".join(failures))
    manifest = pd.read_csv(output_dir / "01_run_manifest.csv")
    sample_manifest = pd.read_csv(output_dir / "sample_manifest.csv").sort_values("sample_index")
    files = [Path(path) for path in sample_manifest["path"].tolist()]
    if len(files) < 700 or any(not path.exists() for path in files):
        raise RuntimeError("Persisted stratified sample manifest is incomplete or its priors are unavailable")
    rng = np.random.default_rng(audit.SEED)
    donor_indices = rng.permutation(len(files))
    profiles = profile_map(specs)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    predictions = pd.read_csv(output_dir / "05_prediction_counterfactuals.csv")
    prediction_summary = pd.read_csv(output_dir / "05_prediction_counterfactuals_summary.csv")
    graph_stats = pd.read_csv(output_dir / "03_graph_mode_statistics.csv")
    ablations = pd.read_csv(output_dir / "07_edge_ablation_matrix.csv")

    embeddings = collect_embeddings(specs, files, donor_indices, profiles, device, args.batch_size)
    representations = audit.representation_summary(specs, predictions, embeddings, output_dir)
    probe = collect_probe(specs, files, profiles, device, args.batch_size, min(args.probe_count, len(files)))
    probe.to_csv(output_dir / "08_structure_signal_probe.csv", index=False)
    audit.write_probe_report(probe, output_dir)

    curves = audit.training_curve_audit(specs, output_dir)
    audit.write_training_report(curves, specs, output_dir)
    audit.write_class_detection_report(predictions, ablations, output_dir)
    robustness = audit.checkpoint_robustness_table(prediction_summary, representations, ablations, curves, output_dir)
    audit.write_hypothesis_matrix(robustness, probe, output_dir)
    audit.write_readme(output_dir, specs, failures, len(files), device, True)
    audit.machine_summary(output_dir, manifest, graph_stats, prediction_summary, robustness, failures)
    write_completion_commands(output_dir, args)

    complete = {
        "completed_at": time.time(), "pid": os.getpid(),
        "resume_mode": "finalize_from_persisted_artifacts",
        "files_created": len(list(output_dir.iterdir())),
        "sample_count": len(files), "checkpoint_count": len(specs), "smoke": "PASS",
    }
    (output_dir / "AUDIT_COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    (output_dir / ".audit_in_progress").unlink(missing_ok=True)
    audit.emit("finalize_complete", output_dir=str(output_dir), **complete)


if __name__ == "__main__":
    main()