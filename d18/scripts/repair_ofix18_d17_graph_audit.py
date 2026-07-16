"""Repair D17 graph-stat/hash rows to describe the graph actually seen by EPPGNN."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.scripts import audit_ofix18_predecision as audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_predecision_audit")
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output_dir
    stats_path = out / "03_graph_mode_statistics.csv"
    hashes_path = out / "04_graph_hash_audit.csv"
    supports_path = out / "12_node_support_information.csv"
    manifest_path = out / "sample_manifest.csv"
    for path in (stats_path, hashes_path, supports_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    stats = pd.read_csv(stats_path)
    hashes = pd.read_csv(hashes_path)
    supports = pd.read_csv(supports_path)
    manifest = pd.read_csv(manifest_path).sort_values("sample_index").reset_index(drop=True)
    if len(manifest) != 715:
        raise RuntimeError(f"Expected locked 715-sample manifest, got {len(manifest)}")
    cache_root = PROJECT_ROOT / audit.PROFILE_CACHE["d17_pixel_only"] / "test"
    prior_files = [Path(path) for path in manifest["path"].tolist()]
    donor_indices = np.random.default_rng(audit.SEED).permutation(len(prior_files))
    stat_rows, hash_rows = [], []

    for index, prior_path in enumerate(prior_files):
        prior = audit.load_prior(prior_path)
        donor = audit.load_prior(prior_files[int(donor_indices[index])])
        official = audit.without_structure(load_d18_graph_cache(cache_root / prior_path.name))
        if int(official.sample_index) != int(prior["sample_index"]) or int(official.y) != int(prior["label"]):
            raise RuntimeError(f"Identity mismatch: {prior_path}")
        official_edges = audit.edge_set(official)
        official_by_type = {edge_type: audit.edge_set(official, edge_type) for edge_type in audit.EDGE_NAMES}
        official_nodes = {tuple(row) for row in np.rint((official.pos.numpy() + 1.0) * 47.0 / 2.0).astype(int).tolist()}
        measured = audit.approximate_graph_stats(
            official, np.random.default_rng(audit.SEED + 2 * 100000 + index * 10)
        )
        for mode in audit.MODES:
            mutated = audit.mode_prior(prior, mode, donor)
            graph = replace(
                official,
                detected=torch.tensor(bool(mutated.get("detected", prior["detected"])), dtype=torch.bool),
                landmark_missing_flag=torch.tensor(
                    int(mutated.get("landmark_missing_flag", prior["landmark_missing_flag"])), dtype=torch.long
                ),
            )
            current_nodes = {tuple(row) for row in np.rint((graph.pos.numpy() + 1.0) * 47.0 / 2.0).astype(int).tolist()}
            current_edges = audit.edge_set(graph)
            current_by_type = {edge_type: audit.edge_set(graph, edge_type) for edge_type in audit.EDGE_NAMES}
            row = {
                "graph_profile": "d17_pixel_only",
                "image_id": prior_path.stem,
                "sample_index": int(prior["sample_index"]),
                "true_class": int(prior["label"]),
                "mode": mode,
                "landmark_detected_state": bool(prior["detected"]),
                "mode_detected_state": bool(mutated.get("detected", prior["detected"])),
                "landmark_missing_flag": int(prior["landmark_missing_flag"]),
                "fallback_template_source": f"fallback_type_id={int(prior.get('fallback_type_id', -1))}",
                "number_of_local_edges": int((graph.edge_type == 0).sum()),
                "number_of_knn_edges": int((graph.edge_type == 1).sum()),
                "number_of_structure_edges": 0,
                "local_edge_proportion": float((graph.edge_type == 0).float().mean()),
                "knn_edge_proportion": float((graph.edge_type == 1).float().mean()),
                "structure_edge_proportion": 0.0,
                "node_support_overlap_with_official": audit.jaccard(current_nodes, official_nodes),
                "overall_edge_jaccard_with_official": audit.jaccard(current_edges, official_edges),
                "local_edge_jaccard_with_official": audit.jaccard(current_by_type[0], official_by_type[0]),
                "knn_edge_jaccard_with_official": audit.jaccard(current_by_type[1], official_by_type[1]),
                "structure_edge_jaccard_with_official": audit.jaccard(current_by_type[2], official_by_type[2]),
                "official_structure_edges_retained_pct": 0.0,
                "new_edges_introduced_pct": 0.0,
                **measured,
            }
            stat_rows.append(row)
            keys = ("graph_profile", "image_id", "sample_index", "true_class", "mode", "landmark_detected_state", "landmark_missing_flag")
            hash_rows.append({key: row[key] for key in keys} | audit.graph_hashes(graph))
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(prior_files):
            audit.emit("repair_d17_graph_rows", completed=index + 1, total=len(prior_files))

    d17_stats = pd.DataFrame(stat_rows)[stats.columns]
    d17_hashes = pd.DataFrame(hash_rows)[hashes.columns]
    stats = pd.concat([stats[stats.graph_profile != "d17_pixel_only"], d17_stats], ignore_index=True)
    hashes = pd.concat([hashes[hashes.graph_profile != "d17_pixel_only"], d17_hashes], ignore_index=True)
    stats = stats.sort_values(["graph_profile", "sample_index", "mode"]).reset_index(drop=True)
    hashes = hashes.sort_values(["graph_profile", "sample_index", "mode"]).reset_index(drop=True)
    if len(stats) != 8580 or len(hashes) != 8580:
        raise RuntimeError(f"Unexpected repaired row counts: stats={len(stats)}, hashes={len(hashes)}")
    stats.to_csv(stats_path, index=False)
    hashes.to_csv(hashes_path, index=False)
    audit.write_graph_reports(stats, hashes, supports, out)

    specs, failures = audit.prepare_specs()
    run_manifest = pd.read_csv(out / "01_run_manifest.csv")
    pred_summary = pd.read_csv(out / "05_prediction_counterfactuals_summary.csv")
    robustness = pd.read_csv(out / "13_checkpoint_robustness_table.csv")
    audit.machine_summary(out, run_manifest, stats, pred_summary, robustness, failures)
    complete_path = out / "AUDIT_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.exists() else {}
    complete.update({"d17_graph_rows_repaired": True, "repair_completed_at": time.time()})
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    audit.emit("repair_complete", d17_rows=len(d17_stats), stats_rows=len(stats), hash_rows=len(hashes))


if __name__ == "__main__":
    main()