"""Rigorous read-only predecision audit for D17/D18 OFIX15-OFIX17.

This script never trains and never mutates checkpoints or training configs. It
uses the repository graph builders/models, existing graph caches, and existing
checkpoints to test structure dependence under controlled counterfactuals and
inference-time edge ablations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d17.models.epp_gnn import EPPGNN
from d18.data.collate import collate_d18_graphs
from d18.data.structure_graph_builder import (
    BASE_EDGE_FEATURE_NAMES,
    DEFAULT_RELATIONS,
    D18GraphData,
    NODE_FEATURE_NAMES,
    _edge_attr,
    _edge_metadata,
    _purify_structure_edges,
    _structure_edges,
    _unique_directed_edges,
    compute_detail_score,
    compute_pixel_feature_maps,
    select_node_coords,
)
from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.models.structure_gnn import StructureGNN
from d18.scripts.audit_d18_prior_dependency import _shuffle_prior, _zero_prior


SEED = 42
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MODES = ["official", "zero_prior", "shuffle_prior", "forced_fallback"]
EDGE_NAMES = {0: "local", 1: "knn", 2: "structure"}
ABLATIONS = {
    "A_full_official": "full",
    "B_remove_structure": "no_structure",
    "C_remove_knn": "no_knn",
    "D_remove_local": "no_local",
    "E_keep_only_local": "local_only",
    "F_local_knn_no_structure": "no_structure",
    "G_local_structure_no_knn": "no_knn",
    "H_permute_structure_destinations": "permute_structure",
    "I_degree_matched_random_structure": "degree_swap_structure",
}


RUN_SPECS: List[Dict[str, Any]] = [
    {
        "run_id": "d17_ofix15c_stratified_detail_knn_dropedge_seed42_best",
        "family": "D17_OFIX15",
        "run_dir": "outputs/d17_runs/ofix15/d17_ofix15c_stratified_detail_knn_dropedge_seed42",
        "source_config": "configs/d17/ofix15/d17_ofix15c_stratified_detail_knn_dropedge_seed42.yaml",
        "checkpoint_type": "best",
        "model_family": "d17",
        "graph_profile": "d17_pixel_only",
    },
    {
        "run_id": "d18_structure_edge_dropedge_seed42_best",
        "family": "D18_OFIX16",
        "run_dir": "outputs/d18_runs/ofix16/d18_structure_edge_dropedge_seed42",
        "source_config": "configs/d18/ofix16/d18_structure_edge_dropedge_seed42.yaml",
        "checkpoint_type": "best",
        "model_family": "d18",
        "graph_profile": "base6_structure",
    },
    {
        "run_id": "d18_structure_edge_dropedge_seed42_last",
        "family": "D18_OFIX16",
        "run_dir": "outputs/d18_runs/ofix16/d18_structure_edge_dropedge_seed42",
        "source_config": "configs/d18/ofix16/d18_structure_edge_dropedge_seed42.yaml",
        "checkpoint_type": "last",
        "model_family": "d18",
        "graph_profile": "base6_structure",
    },
    {
        "run_id": "d18_ofix17b_structure_mode_mix_seed42_best",
        "family": "D18_OFIX17_B",
        "run_dir": "outputs/d18_runs/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42",
        "source_config": "configs/d18/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42.yaml",
        "checkpoint_type": "best",
        "model_family": "d18",
        "graph_profile": "base6_structure",
    },
    {
        "run_id": "d18_ofix17b_structure_mode_mix_seed42_last",
        "family": "D18_OFIX17_B",
        "run_dir": "outputs/d18_runs/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42",
        "source_config": "configs/d18/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42.yaml",
        "checkpoint_type": "last",
        "model_family": "d18",
        "graph_profile": "base6_structure",
    },
    {
        "run_id": "d18_ofix17c_purified_structure_seed42_best",
        "family": "D18_OFIX17_C",
        "run_dir": "outputs/d18_runs/ofix17_structure_reg/d18_ofix17c_purified_structure_seed42",
        "source_config": "configs/d18/ofix17_structure_reg/d18_ofix17c_purified_structure_seed42.yaml",
        "checkpoint_type": "best",
        "model_family": "d18",
        "graph_profile": "purified_structure",
    },
]


PROFILE_CACHE = {
    "base6_structure": "outputs/d18_graph_cache/ofix17_structure_reg/base6_shared",
    "purified_structure": "outputs/d18_graph_cache/ofix17_structure_reg/purified_base6",
    "d17_pixel_only": "outputs/d18_graph_cache/ofix17_structure_reg/base6_shared",
}


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, default=json_default), flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_prior(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for item in arrays:
        if isinstance(item, str):
            raw = item.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
            continue
        arr = np.ascontiguousarray(np.asarray(item))
        descriptor = f"{arr.dtype.str}|{arr.shape}".encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(arr.tobytes())
    return digest.hexdigest()


def current_code_signature() -> Tuple[str, str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=PROJECT_ROOT)
        status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
        diff_hash = hashlib.sha256(diff).hexdigest()
        return commit, diff_hash, status
    except Exception as exc:
        return "unavailable", "unavailable", f"git inspection failed: {type(exc).__name__}: {exc}"


def prepare_specs() -> Tuple[List[Dict[str, Any]], List[str]]:
    prepared: List[Dict[str, Any]] = []
    failures: List[str] = []
    commit, diff_hash, status = current_code_signature()
    for raw in RUN_SPECS:
        spec = dict(raw)
        run_dir = PROJECT_ROOT / spec["run_dir"]
        resolved = run_dir / "resolved_config.yaml"
        source_config = PROJECT_ROOT / spec["source_config"]
        checkpoint = run_dir / "checkpoints" / f"{spec['checkpoint_type']}.pt"
        missing = [str(p) for p in (run_dir, resolved, source_config, checkpoint) if not p.exists()]
        if missing:
            failures.append(f"{spec['run_id']}: missing {missing}")
            continue
        cfg = load_yaml(resolved)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        graph_schema_path = run_dir / "graph_schema.json"
        graph_schema = json.loads(graph_schema_path.read_text(encoding="utf-8")) if graph_schema_path.exists() else {}
        monitor_name = str((cfg.get("training") or {}).get("checkpoint_monitor", "unknown"))
        if spec["checkpoint_type"] == "best":
            monitor_value = payload.get("best_score")
        else:
            monitor_value = None
            log_path = run_dir / "train_log.csv"
            if log_path.exists():
                log_frame = pd.read_csv(log_path)
                point = log_frame[log_frame["epoch"] == int(payload.get("epoch", -1))]
                if not point.empty and monitor_name in point.columns:
                    monitor_value = float(point.iloc[-1][monitor_name])
        spec.update(
            run_dir_path=run_dir,
            resolved_config_path=resolved,
            source_config_path=source_config,
            checkpoint_path=checkpoint,
            cfg=cfg,
            checkpoint_epoch=int(payload.get("epoch", -1)),
            best_epoch=int(payload.get("best_epoch", -1)),
            monitor_name=monitor_name,
            monitor_value=monitor_value,
            seed=int((cfg.get("training") or {}).get("seed", cfg.get("seed", SEED))),
            config_hash=sha256_file(resolved),
            checkpoint_hash=sha256_file(checkpoint),
            code_signature=f"training_git_unavailable;current_commit={commit};current_diff_sha256={diff_hash}",
            current_git_status=status,
            graph_schema=graph_schema,
        )
        prepared.append(spec)
        del payload
    return prepared, failures


def select_stratified_files(prior_dir: Path, per_regular_class: int) -> Tuple[List[Path], pd.DataFrame]:
    files = sorted((prior_dir / "test").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No test priors under {prior_dir / 'test'}")
    rows: List[Dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "path": str(path),
                    "image_id": path.stem,
                    "sample_index": int(data["sample_index"]),
                    "true_class": int(data["label"]),
                    "detected_state": bool(data["detected"]),
                    "landmark_missing_flag": int(data["landmark_missing_flag"]),
                    "fallback_type_id": int(data["fallback_type_id"]) if "fallback_type_id" in data.files else -1,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["true_class", "sample_index"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    selected_parts = []
    for class_id in range(len(CLASS_NAMES)):
        group = frame[frame.true_class == class_id]
        n_take = min(int(per_regular_class), len(group))
        chosen = np.sort(rng.choice(group.index.to_numpy(), size=n_take, replace=False))
        selected_parts.append(frame.loc[chosen])
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
    selected_files = [Path(path) for path in selected["path"].tolist()]
    return selected_files, selected


def pixel_backbone(prior: Mapping[str, np.ndarray], graph_cfg: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(prior["image_48"], dtype=np.float32)
    image_norm = image / 255.0 if float(np.nanmax(image)) > 1.0 else image
    image_norm = np.clip(np.nan_to_num(image_norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    maps = compute_pixel_feature_maps(image_norm)
    score = compute_detail_score(maps)
    coords, _ = select_node_coords(score, dict(graph_cfg))
    yy, xx = coords[:, 0], coords[:, 1]
    x_norm = (xx.astype(np.float32) / 47.0) * 2.0 - 1.0
    y_norm = (yy.astype(np.float32) / 47.0) * 2.0 - 1.0
    x = np.stack(
        [
            maps["intensity"][yy, xx],
            maps["gx"][yy, xx],
            maps["gy"][yy, xx],
            x_norm,
            y_norm,
            maps["grad_mag"][yy, xx],
            maps["local_mean_3x3"][yy, xx],
            maps["local_std_3x3"][yy, xx],
            maps["laplacian_abs"][yy, xx],
            maps["center_surround"][yy, xx],
        ],
        axis=1,
    ).astype(np.float32)
    pos = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    return coords, x, pos


def quantize_like_cache(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float16).astype(np.float32)


def mode_prior(base: Dict[str, np.ndarray], mode: str, donor: Dict[str, np.ndarray] | None) -> Dict[str, np.ndarray]:
    if mode == "official":
        return {key: np.array(value, copy=True) for key, value in base.items()}
    if mode == "zero_prior":
        return _zero_prior(base)
    if mode == "shuffle_prior":
        if donor is None:
            raise ValueError("shuffle_prior requires a donor")
        return _shuffle_prior(base, donor)
    if mode == "forced_fallback":
        out = _zero_prior(base)
        out["detected"] = np.asarray(False)
        return out
    raise ValueError(f"Unknown mode={mode}")


def rebuild_structure_from_cache(
    official: D18GraphData,
    prior: Dict[str, np.ndarray],
    graph_cfg: Mapping[str, Any],
) -> D18GraphData:
    cfg = dict(graph_cfg)
    coords, x, pos = pixel_backbone(prior, cfg)
    official_coords = np.rint((official.pos.detach().cpu().numpy() + 1.0) * 47.0 / 2.0).astype(np.int64)
    official_coords = official_coords[:, [1, 0]]
    if not np.array_equal(coords, official_coords):
        raise RuntimeError("Counterfactual reconstruction changed node ordering/support")
    yy, xx = coords[:, 0], coords[:, 1]
    part_masks = np.asarray(prior.get("part_soft_masks"), dtype=np.float32)
    if part_masks.ndim != 3:
        part_node = np.zeros((coords.shape[0], 0), dtype=np.float32)
    else:
        part_node = np.transpose(part_masks[:, yy, xx], (1, 0)).astype(np.float32)
    structure_cfg = dict(cfg.get("structure_edges", {}) or {})
    if bool(structure_cfg.get("force_remove", False)):
        structure = np.zeros((2, 0), dtype=np.int64)
        structure_meta: Dict[Tuple[int, int], Tuple[int, float]] = {}
        purification = {"before": 0.0, "after": 0.0, "kept_compatibility_mean": math.nan, "dropped_compatibility_mean": math.nan}
    else:
        structure, structure_meta = _structure_edges(coords, part_node, structure_cfg)
        before = int(structure.shape[1])
        structure, structure_meta, purification = _purify_structure_edges(x, structure, structure_meta, structure_cfg)
        purification["before"] = float(before)
    official_edges = official.edge_index.detach().cpu().numpy().astype(np.int64)
    official_types = official.edge_type.detach().cpu().numpy().astype(np.int64)
    local = official_edges[:, official_types == 0]
    local_knn = official_edges[:, official_types != 2]
    local_pairs = {tuple(pair) for pair in local.T.tolist()}
    local_knn_pairs = {tuple(pair) for pair in local_knn.T.tolist()}
    total = _unique_directed_edges(np.concatenate([local_knn, structure], axis=1))
    total_pairs = {tuple(pair) for pair in total.T.tolist()}
    edge_type, relation_id = _edge_metadata(total, local_pairs, local_knn_pairs, structure_meta)
    relation_count = len(structure_cfg.get("relations") or DEFAULT_RELATIONS)
    edge_attr, edge_names = _edge_attr(x, pos, total, structure_meta, str(cfg.get("edge_schema", "base6")), relation_count)
    x_q = quantize_like_cache(x)
    pos_q = quantize_like_cache(pos)
    edge_attr_q = quantize_like_cache(edge_attr)
    return D18GraphData(
        x=torch.from_numpy(x_q),
        edge_index=torch.from_numpy(total).long(),
        edge_attr=torch.from_numpy(edge_attr_q),
        pos=torch.from_numpy(pos_q),
        y=torch.tensor(int(np.asarray(prior["label"]).item()), dtype=torch.long),
        sample_index=torch.tensor(int(np.asarray(prior["sample_index"]).item()), dtype=torch.long),
        detected=torch.tensor(bool(np.asarray(prior.get("detected", True)).item()), dtype=torch.bool),
        landmark_missing_flag=torch.tensor(int(np.asarray(prior.get("landmark_missing_flag", 0)).item()), dtype=torch.long),
        image_48=torch.from_numpy(quantize_like_cache(np.asarray(prior["image_48"], dtype=np.float32))),
        edge_type=torch.from_numpy(edge_type).long(),
        structure_relation_id=torch.from_numpy(relation_id).long(),
        node_feature_names=list(NODE_FEATURE_NAMES),
        edge_feature_names=list(edge_names),
        local_edge_count=len(local_pairs),
        knn_edge_count=len(local_knn_pairs - local_pairs),
        structure_edge_count=len(total_pairs - local_knn_pairs),
        total_edge_count=int(total.shape[1]),
        structure_edge_count_before_purification=int(purification["before"]),
        structure_edge_count_after_purification=int(purification["after"]),
        purification_compatibility_kept_mean=float(purification["kept_compatibility_mean"]),
        purification_compatibility_dropped_mean=float(purification["dropped_compatibility_mean"]),
        node_support_mode=str(cfg.get("node_support_mode", "stratified_detail_knn")),
    )


def without_structure(graph: D18GraphData) -> D18GraphData:
    keep = graph.edge_type != 2
    edges = graph.edge_index[:, keep]
    attrs = graph.edge_attr[keep]
    types = graph.edge_type[keep]
    relations = graph.structure_relation_id[keep]
    return replace(
        graph,
        edge_index=edges,
        edge_attr=attrs,
        edge_type=types,
        structure_relation_id=relations,
        structure_edge_count=0,
        total_edge_count=int(edges.size(1)),
        structure_edge_count_before_purification=0,
        structure_edge_count_after_purification=0,
    )


def edge_set(graph: D18GraphData, edge_type: int | None = None) -> set[Tuple[int, int]]:
    edges = graph.edge_index.detach().cpu().numpy()
    if edge_type is not None:
        mask = graph.edge_type.detach().cpu().numpy() == int(edge_type)
        edges = edges[:, mask]
    return {tuple(map(int, pair)) for pair in edges.T.tolist()}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def graph_hashes(graph: D18GraphData) -> Dict[str, str]:
    pos = graph.pos.detach().cpu().numpy()
    coords = np.rint((pos + 1.0) * 47.0 / 2.0).astype(np.int16)
    coords_sorted = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    edges = graph.edge_index.detach().cpu().numpy().astype(np.int32)
    attrs = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    types = graph.edge_type.detach().cpu().numpy().astype(np.int8)
    result = {
        "ordered_node_coordinates_hash": stable_hash(coords),
        "unordered_node_coordinate_set_hash": stable_hash(coords_sorted),
        "edge_index_hash": stable_hash(edges),
        "edge_attr_hash": stable_hash(attrs),
    }
    for edge_type, name in EDGE_NAMES.items():
        result[f"edge_index_{name}_hash"] = stable_hash(edges[:, types == edge_type])
    result["complete_graph_hash"] = stable_hash(
        graph.x.detach().cpu().numpy(),
        coords,
        edges,
        attrs,
        types,
        graph.structure_relation_id.detach().cpu().numpy(),
        np.asarray([int(graph.y), int(graph.sample_index), int(graph.detected), int(graph.landmark_missing_flag)]),
    )
    return result


def approximate_graph_stats(graph: D18GraphData, rng: np.random.Generator) -> Dict[str, Any]:
    n = int(graph.x.size(0))
    directed = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    undirected_pairs = {tuple(sorted((int(a), int(b)))) for a, b in directed.T.tolist() if int(a) != int(b)}
    if undirected_pairs:
        pairs = np.asarray(sorted(undirected_pairs), dtype=np.int64)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adjacency = sp.csr_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n, n))
    else:
        adjacency = sp.csr_matrix((n, n), dtype=np.uint8)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=component_count) if n else np.asarray([], dtype=np.int64)
    largest_ratio = float(component_sizes.max() / n) if n and component_sizes.size else 0.0
    if n:
        largest_label = int(component_sizes.argmax())
        candidates = np.flatnonzero(labels == largest_label)
        source_count = min(4, len(candidates))
        source_indices = np.linspace(0, len(candidates) - 1, source_count, dtype=int) if source_count else np.asarray([], dtype=int)
        sources = candidates[source_indices] if source_count else np.asarray([], dtype=int)
        distances = shortest_path(adjacency, directed=False, unweighted=True, indices=sources) if source_count else np.asarray([])
        finite = distances[np.isfinite(distances) & (distances > 0)] if distances.size else np.asarray([])
        diameter_approx = float(finite.max()) if finite.size else 0.0
        mean_shortest_approx = float(finite.mean()) if finite.size else 0.0
    else:
        diameter_approx = mean_shortest_approx = 0.0
    sample_nodes = rng.choice(n, size=min(64, n), replace=False) if n else np.asarray([], dtype=np.int64)
    clustering_values = []
    for node in sample_nodes.tolist():
        neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        k = len(neighbors)
        if k < 2:
            clustering_values.append(0.0)
            continue
        sub = adjacency[neighbors][:, neighbors]
        clustering_values.append(float(sub.nnz / max(k * (k - 1), 1)))
    return {
        "number_of_nodes": n,
        "total_directed_edge_count": int(directed.shape[1]),
        "total_undirected_edge_count": int(len(undirected_pairs)),
        "mean_degree": float(degree.mean()) if degree.size else 0.0,
        "std_degree": float(degree.std()) if degree.size else 0.0,
        "isolated_node_count": int((degree == 0).sum()),
        "connected_component_count": int(component_count),
        "largest_component_ratio": largest_ratio,
        "diameter_approx": diameter_approx,
        "mean_shortest_path_approx": mean_shortest_approx,
        "clustering_coefficient_sampled": float(np.mean(clustering_values)) if clustering_values else 0.0,
    }


def build_graph_store(
    files: Sequence[Path],
    donor_indices: np.ndarray,
    profiles: Mapping[str, Dict[str, Any]],
    output_dir: Path,
    smoke_only: bool,
) -> Tuple[Dict[Tuple[str, str, int], D18GraphData], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store: Dict[Tuple[str, str, int], D18GraphData] = {}
    stats_rows: List[Dict[str, Any]] = []
    hash_rows: List[Dict[str, Any]] = []
    support_rows: List[Dict[str, Any]] = []
    limit = min(8, len(files)) if smoke_only else len(files)
    rng = np.random.default_rng(SEED)
    profile_items = list(profiles.items())
    for profile_index, (profile_name, profile) in enumerate(profile_items):
        graph_cfg = profile["cfg"].get("graph", {}) or {}
        cache_root = Path(profile["cache_root"])
        for local_index, prior_path in enumerate(files[:limit]):
            prior = load_prior(prior_path)
            donor = load_prior(files[int(donor_indices[local_index])])
            official_cache = cache_root / "test" / prior_path.name
            if not official_cache.exists():
                raise FileNotFoundError(f"Missing exact cache graph: {official_cache}")
            official = load_d18_graph_cache(official_cache)
            if int(official.sample_index) != int(prior["sample_index"]) or int(official.y) != int(prior["label"]):
                raise RuntimeError(f"Cache/prior identity mismatch for {prior_path.name}")
            if profile_name == "d17_pixel_only":
                official = without_structure(official)
            official_by_type = {etype: edge_set(official, etype) for etype in EDGE_NAMES}
            official_nodes = {tuple(row) for row in np.rint((official.pos.numpy() + 1.0) * 47.0 / 2.0).astype(int).tolist()}
            for mode in MODES:
                mutated = mode_prior(prior, mode, donor)
                if profile_name == "d17_pixel_only":
                    graph = replace(
                        official,
                        detected=torch.tensor(bool(mutated.get("detected", prior["detected"])), dtype=torch.bool),
                        landmark_missing_flag=torch.tensor(
                            int(mutated.get("landmark_missing_flag", prior["landmark_missing_flag"])), dtype=torch.long
                        ),
                    )
                else:
                    graph = official if mode == "official" else rebuild_structure_from_cache(official, mutated, graph_cfg)
                key = (profile_name, mode, local_index)
                store[key] = graph
                hashes = graph_hashes(graph)
                current_by_type = {etype: edge_set(graph, etype) for etype in EDGE_NAMES}
                current_nodes = {tuple(row) for row in np.rint((graph.pos.numpy() + 1.0) * 47.0 / 2.0).astype(int).tolist()}
                graph_stats = approximate_graph_stats(graph, np.random.default_rng(SEED + profile_index * 100000 + local_index * 10 + MODES.index(mode)))
                structure_official = official_by_type[2]
                structure_current = current_by_type[2]
                row = {
                    "graph_profile": profile_name,
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
                    "number_of_structure_edges": int((graph.edge_type == 2).sum()),
                    "local_edge_proportion": float((graph.edge_type == 0).float().mean()),
                    "knn_edge_proportion": float((graph.edge_type == 1).float().mean()),
                    "structure_edge_proportion": float((graph.edge_type == 2).float().mean()),
                    "node_support_overlap_with_official": jaccard(current_nodes, official_nodes),
                    "overall_edge_jaccard_with_official": jaccard(edge_set(graph), edge_set(official)),
                    "local_edge_jaccard_with_official": jaccard(current_by_type[0], official_by_type[0]),
                    "knn_edge_jaccard_with_official": jaccard(current_by_type[1], official_by_type[1]),
                    "structure_edge_jaccard_with_official": jaccard(structure_current, structure_official),
                    "official_structure_edges_retained_pct": 100.0 * len(structure_current & structure_official) / max(len(structure_official), 1),
                    "new_edges_introduced_pct": 100.0 * len(edge_set(graph) - edge_set(official)) / max(len(edge_set(graph)), 1),
                    **graph_stats,
                }
                stats_rows.append(row)
                hash_rows.append({k: row[k] for k in ("graph_profile", "image_id", "sample_index", "true_class", "mode", "landmark_detected_state", "landmark_missing_flag")} | hashes)
                x = graph.x.numpy()
                pos = graph.pos.numpy()
                quadrants = [
                    float(np.mean((pos[:, 0] < 0) & (pos[:, 1] < 0))),
                    float(np.mean((pos[:, 0] >= 0) & (pos[:, 1] < 0))),
                    float(np.mean((pos[:, 0] < 0) & (pos[:, 1] >= 0))),
                    float(np.mean((pos[:, 0] >= 0) & (pos[:, 1] >= 0))),
                ]
                support_rows.append(
                    {
                        "graph_profile": profile_name,
                        "sample_index": int(prior["sample_index"]),
                        "true_class": int(prior["label"]),
                        "mode": mode,
                        "coordinate_jaccard_with_official": row["node_support_overlap_with_official"],
                        "spatial_bbox_coverage": float((pos[:, 0].max() - pos[:, 0].min()) * (pos[:, 1].max() - pos[:, 1].min()) / 4.0),
                        "quadrant_top_left": quadrants[0],
                        "quadrant_top_right": quadrants[1],
                        "quadrant_bottom_left": quadrants[2],
                        "quadrant_bottom_right": quadrants[3],
                        "intensity_mean": float(x[:, 0].mean()),
                        "intensity_std": float(x[:, 0].std()),
                        "gradient_magnitude_mean": float(x[:, 5].mean()),
                        "gradient_magnitude_std": float(x[:, 5].std()),
                    }
                )
            if local_index == 0 or (local_index + 1) % 50 == 0 or local_index + 1 == limit:
                emit("graph_store_progress", profile=profile_name, completed=local_index + 1, total=limit)
    stats = pd.DataFrame(stats_rows)
    hashes = pd.DataFrame(hash_rows)
    supports = pd.DataFrame(support_rows)
    stats.to_csv(output_dir / ("smoke_graph_mode_statistics.csv" if smoke_only else "03_graph_mode_statistics.csv"), index=False)
    hashes.to_csv(output_dir / ("smoke_graph_hash_audit.csv" if smoke_only else "04_graph_hash_audit.csv"), index=False)
    supports.to_csv(output_dir / ("smoke_node_support_information.csv" if smoke_only else "12_node_support_information.csv"), index=False)
    return store, stats, hashes, supports


def load_model(spec: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = spec["cfg"]
    if spec["model_family"] == "d17":
        model = EPPGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
    else:
        model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6)
    payload = torch.load(spec["checkpoint_path"], map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def infer_graphs(model: torch.nn.Module, graphs: Sequence[D18GraphData], device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    logits_all, embeddings_all = [], []
    for group in batched(graphs, batch_size):
        batch = collate_d18_graphs(list(group)).to(device)
        out = model(batch)
        logits = out["logits"]
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Non-finite logits encountered")
        logits_all.append(logits.detach().cpu().numpy())
        embeddings_all.append(out["z_image"].detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(embeddings_all, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.clip(denom, 1e-12, None)


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))), axis=1)
    kl_qm = np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1] if i < bins - 1 else confidence <= 1.0)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def aggregate_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    probability_cols = [f"prob_{name}" for name in CLASS_NAMES]
    logit_cols = [f"logit_{name}" for name in CLASS_NAMES]
    for (run_id, mode), group_all in frame.groupby(["run_id", "mode"], sort=False):
        official_all = frame[(frame.run_id == run_id) & (frame["mode"] == "official")].sort_values("sample_index")
        for detection_group, group in [("all", group_all), ("detected", group_all[group_all.detected_state]), ("missing", group_all[~group_all.detected_state])]:
            group = group.sort_values("sample_index")
            if group.empty:
                continue
            y = group.true_class.to_numpy(dtype=int)
            pred = group.predicted_class.to_numpy(dtype=int)
            probs = group[probability_cols].to_numpy(dtype=float)
            logits = group[logit_cols].to_numpy(dtype=float)
            precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(7), zero_division=0)
            cm = confusion_matrix(y, pred, labels=np.arange(7))
            official = official_all[official_all.sample_index.isin(group.sample_index)].sort_values("sample_index")
            if not np.array_equal(group.sample_index.to_numpy(), official.sample_index.to_numpy()):
                raise RuntimeError(f"Official/counterfactual ordering mismatch for {run_id}/{mode}/{detection_group}")
            official_probs = official[probability_cols].to_numpy(dtype=float)
            official_logits = official[logit_cols].to_numpy(dtype=float)
            official_pred = official.predicted_class.to_numpy(dtype=int)
            nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())
            one_hot = np.eye(7)[y]
            row: Dict[str, Any] = {
                "run_id": run_id,
                "mode": mode,
                "detection_group": detection_group,
                "count": len(group),
                "accuracy": float((pred == y).mean()),
                "macro_f1": float(f1.mean()),
                "nll": nll,
                "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
                "ece_15bin": ece_score(y, probs),
                "mean_entropy": float(entropy_np(probs).mean()),
                "agreement_with_official": float((pred == official_pred).mean()),
                "correct_to_wrong": int(np.sum((official_pred == y) & (pred != y))),
                "wrong_to_correct": int(np.sum((official_pred != y) & (pred == y))),
                "mean_js_divergence_vs_official": float(js_divergence_rows(probs, official_probs).mean()),
                "mean_logit_l2_change": float(np.linalg.norm(logits - official_logits, axis=1).mean()),
                "mean_logit_cosine_vs_official": float(cosine_rows(logits, official_logits).mean()),
                "confusion_matrix_json": json.dumps(cm.tolist()),
            }
            for class_id, name in enumerate(CLASS_NAMES):
                row[f"precision_{name}"] = float(precision[class_id])
                row[f"recall_{name}"] = float(recall[class_id])
                row[f"f1_{name}"] = float(f1[class_id])
                row[f"support_{name}"] = int(support[class_id])
            rows.append(row)
    return pd.DataFrame(rows)


def run_predictions(
    specs: Sequence[Dict[str, Any]],
    files: Sequence[Path],
    graph_store: Mapping[Tuple[str, str, int], D18GraphData],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    smoke_only: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    for spec in specs:
        model = load_model(spec, device)
        emit("prediction_checkpoint_start", run_id=spec["run_id"], checkpoint=spec["checkpoint_type"])
        for mode in MODES:
            graphs = []
            for index in range(len(files)):
                graph = graph_store[(spec["graph_profile"], mode, index)]
                if spec["model_family"] == "d17":
                    graph = without_structure(graph)
                graphs.append(graph)
            logits, z = infer_graphs(model, graphs, device, batch_size)
            probs = softmax_np(logits)
            embeddings[(spec["run_id"], mode)] = z
            predictions = probs.argmax(axis=1)
            margins = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
            for index, graph in enumerate(graphs):
                row: Dict[str, Any] = {
                    "run_id": spec["run_id"],
                    "family": spec["family"],
                    "checkpoint_type": spec["checkpoint_type"],
                    "mode": mode,
                    "image_id": files[index].stem,
                    "sample_index": int(graph.sample_index),
                    "true_class": int(graph.y),
                    "detected_state": bool(graph_store[(spec["graph_profile"], "official", index)].detected),
                    "mode_detected_state": bool(graph.detected),
                    "predicted_class": int(predictions[index]),
                    "entropy": float(entropy_np(probs[index : index + 1])[0]),
                    "max_probability": float(probs[index].max()),
                    "margin": float(margins[index]),
                    "correctness": int(predictions[index] == int(graph.y)),
                }
                for class_id, name in enumerate(CLASS_NAMES):
                    row[f"logit_{name}"] = float(logits[index, class_id])
                    row[f"prob_{name}"] = float(probs[index, class_id])
                rows.append(row)
            emit("prediction_mode_done", run_id=spec["run_id"], mode=mode, samples=len(graphs))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    summary = aggregate_prediction_rows(frame)
    frame.to_csv(output_dir / ("smoke_prediction_counterfactuals.csv" if smoke_only else "05_prediction_counterfactuals.csv"), index=False)
    summary.to_csv(output_dir / ("smoke_prediction_counterfactuals_summary.csv" if smoke_only else "05_prediction_counterfactuals_summary.csv"), index=False)
    return frame, summary, embeddings


def probability_sensitivity(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    probability_cols = [f"prob_{name}" for name in CLASS_NAMES]
    logit_cols = [f"logit_{name}" for name in CLASS_NAMES]
    rows = []
    for (run_id, sample_index), group in predictions.groupby(["run_id", "sample_index"]):
        ordered = group.set_index("mode").loc[MODES]
        probs = ordered[probability_cols].to_numpy(float)
        logits = ordered[logit_cols].to_numpy(float)
        divergences = [float(js_divergence_rows(probs[i : i + 1], probs[j : j + 1])[0]) for i in range(len(MODES)) for j in range(i + 1, len(MODES))]
        correct = ordered.correctness.to_numpy(int).astype(bool)
        pred_classes = ordered.predicted_class.to_numpy(int)
        rows.append(
            {
                "run_id": run_id,
                "sample_index": int(sample_index),
                "image_id": ordered.image_id.iloc[0],
                "true_class": int(ordered.true_class.iloc[0]),
                "detected_state": bool(ordered.detected_state.iloc[0]),
                "maximum_probability_js_divergence_across_modes": max(divergences) if divergences else 0.0,
                "mean_logit_variance_across_modes": float(np.var(logits, axis=0).mean()),
                "number_of_distinct_predicted_classes": int(len(set(pred_classes.tolist()))),
                "all_modes_correct": bool(correct.all()),
                "only_official_correct": bool(correct[0] and not correct[1:].any()),
                "official_wrong_counterfactual_correct": bool((not correct[0]) and correct[1:].any()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["sensitivity_rank_group"] = ""
    for run_id, group in frame.groupby("run_id"):
        order = group.sort_values("maximum_probability_js_divergence_across_modes").index
        frame.loc[order[: min(50, len(order))], "sensitivity_rank_group"] = "least_50"
        frame.loc[order[-min(50, len(order)) :], "sensitivity_rank_group"] = "most_50"
    frame.to_csv(output_dir / "06_per_sample_sensitivity.csv", index=False)
    return frame


def recompute_base6_attrs(graph: D18GraphData, edges: np.ndarray) -> np.ndarray:
    attrs, _ = _edge_attr(
        graph.x.detach().cpu().numpy(),
        graph.pos.detach().cpu().numpy(),
        edges,
        {},
        "base6",
        len(DEFAULT_RELATIONS),
    )
    return quantize_like_cache(attrs)


def degree_swap_edges(edges: np.ndarray, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    pairs = [tuple(map(int, pair)) for pair in edges.T.tolist()]
    pair_set = set(pairs)
    if len(pairs) < 2:
        return edges.copy()
    attempts = max(100, 5 * len(pairs))
    for _ in range(attempts):
        i, j = rng.choice(len(pairs), size=2, replace=False)
        a, b = pairs[i]
        c, d = pairs[j]
        proposal_a, proposal_b = (a, d), (c, b)
        if a == d or c == b or proposal_a == proposal_b:
            continue
        if proposal_a in pair_set or proposal_b in pair_set:
            continue
        pair_set.remove(pairs[i])
        pair_set.remove(pairs[j])
        pairs[i], pairs[j] = proposal_a, proposal_b
        pair_set.add(proposal_a)
        pair_set.add(proposal_b)
    return np.asarray(pairs, dtype=np.int64).T


def ablate_graph(graph: D18GraphData, operation: str, seed: int) -> D18GraphData:
    types = graph.edge_type.detach().cpu().numpy().astype(np.int64)
    edges = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    attrs = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    relations = graph.structure_relation_id.detach().cpu().numpy().astype(np.int64)
    if operation == "full":
        return graph
    keep_masks = {
        "no_structure": types != 2,
        "no_knn": types != 1,
        "no_local": types != 0,
        "local_only": types == 0,
    }
    if operation in keep_masks:
        keep = keep_masks[operation]
        new_edges, new_attrs, new_types, new_relations = edges[:, keep], attrs[keep], types[keep], relations[keep]
    elif operation in {"permute_structure", "degree_swap_structure"}:
        structure_mask = types == 2
        structure_edges = edges[:, structure_mask]
        if structure_edges.shape[1] == 0:
            return graph
        rng = np.random.default_rng(seed)
        if operation == "permute_structure":
            destinations = structure_edges[1].copy()
            for _ in range(20):
                shuffled = rng.permutation(destinations)
                if not np.any(shuffled == structure_edges[0]):
                    break
            structure_new = np.stack([structure_edges[0], shuffled], axis=0)
        else:
            structure_new = degree_swap_edges(structure_edges, int(graph.x.size(0)), rng)
        non_structure = types != 2
        new_edges = np.concatenate([edges[:, non_structure], structure_new], axis=1)
        new_types = np.concatenate([types[non_structure], np.full(structure_new.shape[1], 2, dtype=np.int64)])
        new_relations = np.concatenate([relations[non_structure], relations[structure_mask]])
        new_attrs = recompute_base6_attrs(graph, new_edges)
    else:
        raise ValueError(f"Unknown ablation operation={operation}")
    if new_edges.shape[1] == 0:
        raise RuntimeError(f"Ablation {operation} produced an empty graph")
    return replace(
        graph,
        edge_index=torch.from_numpy(new_edges).long(),
        edge_attr=torch.from_numpy(new_attrs).float(),
        edge_type=torch.from_numpy(new_types).long(),
        structure_relation_id=torch.from_numpy(new_relations).long(),
        local_edge_count=int(np.sum(new_types == 0)),
        knn_edge_count=int(np.sum(new_types == 1)),
        structure_edge_count=int(np.sum(new_types == 2)),
        total_edge_count=int(new_edges.shape[1]),
    )


def summarize_single_prediction_set(y: np.ndarray, logits: np.ndarray, reference_pred: np.ndarray) -> Dict[str, Any]:
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(7), zero_division=0)
    one_hot = np.eye(7)[y]
    row: Dict[str, Any] = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_15bin": ece_score(y, probs),
        "mean_entropy": float(entropy_np(probs).mean()),
        "prediction_agreement_with_full": float((pred == reference_pred).mean()),
        "confusion_matrix_json": json.dumps(confusion_matrix(y, pred, labels=np.arange(7)).tolist()),
    }
    for class_id, name in enumerate(CLASS_NAMES):
        row[f"precision_{name}"] = float(precision[class_id])
        row[f"recall_{name}"] = float(recall[class_id])
        row[f"f1_{name}"] = float(f1[class_id])
        row[f"support_{name}"] = int(support[class_id])
    return row


def run_edge_ablations(
    specs: Sequence[Dict[str, Any]],
    files: Sequence[Path],
    graph_store: Mapping[Tuple[str, str, int], D18GraphData],
    predictions: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    logit_cols = [f"logit_{name}" for name in CLASS_NAMES]
    for spec in specs:
        model = load_model(spec, device)
        official_predictions = predictions[(predictions.run_id == spec["run_id"]) & (predictions["mode"] == "official")].sort_values("sample_index")
        y = official_predictions.true_class.to_numpy(int)
        full_pred = official_predictions.predicted_class.to_numpy(int)
        cache_results: Dict[str, np.ndarray] = {"full": official_predictions[logit_cols].to_numpy(float)}
        unique_operations = list(dict.fromkeys(ABLATIONS.values()))
        for operation in unique_operations:
            if operation not in cache_results:
                graphs = []
                for index in range(len(files)):
                    graph = graph_store[(spec["graph_profile"], "official", index)]
                    if spec["model_family"] == "d17":
                        graph = without_structure(graph)
                    graphs.append(ablate_graph(graph, operation, SEED + int(graph.sample_index) * 31 + unique_operations.index(operation)))
                logits, _ = infer_graphs(model, graphs, device, batch_size)
                cache_results[operation] = logits
            emit("edge_ablation_operation_done", run_id=spec["run_id"], operation=operation)
        for label, operation in ABLATIONS.items():
            row = {"run_id": spec["run_id"], "ablation": label, **summarize_single_prediction_set(y, cache_results[operation], full_pred)}
            rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "07_edge_ablation_matrix.csv", index=False)
    return frame


@torch.no_grad()
def probe_checkpoint(
    spec: Dict[str, Any],
    graphs: Sequence[D18GraphData],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    model = load_model(spec, device)
    accum: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for group in batched(graphs, batch_size):
        batch = collate_d18_graphs(list(group)).to(device)
        h = model.encoder(batch.x_cat)
        edge_index = batch.edge_index_cat
        edge_attr = batch.edge_attr_cat
        edge_type = batch.edge_type_cat
        src, dst = edge_index[0].long(), edge_index[1].long()
        degree = h.new_zeros((h.size(0), 1))
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device, dtype=h.dtype))
        for layer_index, layer in enumerate(model.gnn.layers, start=1):
            h_before = h
            edge_emb = layer.edge_mlp(edge_attr.to(dtype=h.dtype))
            vector_gate = torch.sigmoid(layer.gate(edge_emb))
            pre_gate = layer.message(torch.cat([h[src], edge_emb], dim=1))
            post_vector = pre_gate * vector_gate
            scalar_gate = torch.ones((post_vector.size(0), 1), device=device, dtype=h.dtype)
            scalar_gate_module = getattr(layer, "scalar_gate", None)
            if scalar_gate_module is not None:
                scalar_gate = torch.sigmoid(scalar_gate_module(edge_emb))
                if layer.structure_gate_cap is not None:
                    scalar_gate = torch.where((edge_type == 2).view(-1, 1), scalar_gate * float(layer.structure_gate_cap), scalar_gate)
            post_scalar = post_vector * scalar_gate
            aggregate_norms: Dict[str, float] = {}
            for edge_id, edge_name in EDGE_NAMES.items():
                mask = edge_type == edge_id
                if not bool(mask.any()):
                    continue
                bucket = accum[(layer_index, edge_name)]
                bucket["edge_count"].append(float(mask.sum().item()) / max(int(batch.num_graphs), 1))
                bucket["pre_gate_message_norm"].append(float(pre_gate[mask].norm(dim=1).mean().item()))
                bucket["post_vector_gate_message_norm"].append(float(post_vector[mask].norm(dim=1).mean().item()))
                bucket["post_scalar_gate_message_norm"].append(float(post_scalar[mask].norm(dim=1).mean().item()))
                gate_values = vector_gate[mask].reshape(-1)
                bucket["vector_gate_mean"].append(float(gate_values.mean().item()))
                bucket["vector_gate_std"].append(float(gate_values.std(unbiased=False).item()))
                bucket["vector_gate_p05"].append(float(torch.quantile(gate_values, 0.05).item()))
                bucket["vector_gate_p50"].append(float(torch.quantile(gate_values, 0.50).item()))
                bucket["vector_gate_p95"].append(float(torch.quantile(gate_values, 0.95).item()))
                if scalar_gate_module is not None:
                    scalar_values = scalar_gate[mask].reshape(-1)
                    bucket["scalar_gate_mean"].append(float(scalar_values.mean().item()))
                aggregate = h.new_zeros(h.shape)
                aggregate.index_add_(0, dst[mask], post_scalar[mask])
                aggregate = aggregate / degree.clamp_min(1.0)
                aggregate_norms[edge_name] = float(aggregate.norm(dim=1).sum().item())
            total_aggregate_norm = sum(aggregate_norms.values())
            for edge_name, value in aggregate_norms.items():
                accum[(layer_index, edge_name)]["aggregate_message_norm_share"].append(value / max(total_aggregate_norm, 1e-12))
            if spec["model_family"] == "d17":
                h = layer(h, edge_index, edge_attr, dst_degree=degree)
            else:
                h = layer(h, edge_index, edge_attr, dst_degree=degree, edge_type=edge_type)
            change = (h - h_before).norm(dim=1).mean().item()
            for edge_name in aggregate_norms:
                accum[(layer_index, edge_name)]["layer_representation_change_norm"].append(float(change))
        z = model.readout(h, batch.batch_index, batch.num_graphs)
        z_norm = float(z.norm(dim=1).mean().item())
        for layer_index, edge_name in list(accum.keys()):
            if layer_index == len(model.gnn.layers):
                accum[(layer_index, edge_name)]["graph_embedding_norm"].append(z_norm)
    rows = []
    for (layer_index, edge_name), metrics in sorted(accum.items()):
        row = {"run_id": spec["run_id"], "layer": layer_index, "edge_type": edge_name, "probe_sample_count": len(graphs)}
        for key, values in metrics.items():
            row[key] = float(np.mean(values)) if values else math.nan
        rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_structure_probe(
    specs: Sequence[Dict[str, Any]],
    graph_store: Mapping[Tuple[str, str, int], D18GraphData],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    probe_count: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        graphs = []
        for index in range(probe_count):
            graph = graph_store[(spec["graph_profile"], "official", index)]
            graphs.append(without_structure(graph) if spec["model_family"] == "d17" else graph)
        rows.extend(probe_checkpoint(spec, graphs, device, batch_size))
        emit("structure_probe_done", run_id=spec["run_id"], samples=probe_count)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "08_structure_signal_probe.csv", index=False)
    return frame


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    numerator = float(np.sum(cross * cross))
    denominator = math.sqrt(float(np.sum((x.T @ x) ** 2)) * float(np.sum((y.T @ y) ** 2)))
    return numerator / max(denominator, 1e-12)


def representation_quality(z: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    centroids = []
    within = []
    for class_id in range(7):
        subset = z[labels == class_id]
        if not len(subset):
            continue
        centroid = subset.mean(axis=0)
        centroids.append(centroid)
        within.extend(np.linalg.norm(subset - centroid, axis=1).tolist())
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    within_mean = float(np.mean(within)) if within else math.nan
    between_mean = float(np.mean(between)) if between else math.nan
    return between_mean, within_mean, between_mean / max(within_mean, 1e-12)


def representation_summary(
    specs: Sequence[Dict[str, Any]],
    predictions: pd.DataFrame,
    embeddings: Mapping[Tuple[str, str], np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        run_id = spec["run_id"]
        labels = predictions[(predictions.run_id == run_id) & (predictions["mode"] == "official")].sort_values("sample_index").true_class.to_numpy(int)
        official = embeddings[(run_id, "official")]
        for mode in MODES:
            current = embeddings[(run_id, mode)]
            between, within, ratio = representation_quality(current, labels)
            rows.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "paired_cosine_similarity_mean": float(cosine_rows(official, current).mean()),
                    "paired_cosine_similarity_std": float(cosine_rows(official, current).std()),
                    "normalized_l2_distance_mean": float((np.linalg.norm(official - current, axis=1) / np.clip(np.linalg.norm(official, axis=1), 1e-12, None)).mean()),
                    "linear_cka": linear_cka(official, current),
                    "class_centroid_separation": between,
                    "within_class_distance": within,
                    "between_within_ratio": ratio,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "09_representation_similarity.csv", index=False)
    return frame


def training_curve_audit(specs: Sequence[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for spec in specs:
        run_dir = Path(spec["run_dir_path"])
        if str(run_dir) in seen:
            continue
        seen.add(str(run_dir))
        log_path = run_dir / "train_log.csv"
        if not log_path.exists():
            continue
        frame = pd.read_csv(log_path)
        for _, record in frame.iterrows():
            row = {
                "run_id": run_dir.name,
                "family": spec["family"],
                "epoch": int(record["epoch"]),
                "train_loss": record.get("train_loss", math.nan),
                "train_accuracy": record.get("train_accuracy", math.nan),
                "train_macro_f1": record.get("train_macro_f1", math.nan),
                "val_loss": record.get("val_loss", math.nan),
                "val_accuracy": record.get("val_accuracy", math.nan),
                "val_macro_f1": record.get("val_macro_f1", math.nan),
                "learning_rate": record.get("lr", math.nan),
                "checkpoint_selected": int(record["epoch"] == spec["best_epoch"]),
            }
            for key in (
                "structure_drop_fraction_observed",
                "structure_mode_forced_sample_pct",
                "structure_edge_count_mean",
                "raw_gate_mean_structure",
                "effective_gate_mean_structure",
            ):
                row[key] = record.get(key, math.nan)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "10_training_curve_audit.csv", index=False)
    return frame


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    if frame.empty:
        return "_No rows available._"
    view = frame[list(columns)].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "MISSING" if pd.isna(x) else f"{x:.{digits}f}")
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def write_manifest(specs: Sequence[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        cfg = spec["cfg"]
        graph = cfg.get("graph", {}) or {}
        model = cfg.get("model", {}) or {}
        training = cfg.get("training", {}) or {}
        schema = spec["graph_schema"]
        rows.append(
            {
                "run_id": spec["run_id"],
                "family": spec["family"],
                "config_path": str(Path(spec["resolved_config_path"]).relative_to(PROJECT_ROOT)),
                "checkpoint_path": str(Path(spec["checkpoint_path"]).relative_to(PROJECT_ROOT)),
                "checkpoint_type": spec["checkpoint_type"],
                "checkpoint_epoch": spec["checkpoint_epoch"],
                "monitor_name": spec["monitor_name"],
                "monitor_value": spec["monitor_value"],
                "seed": spec["seed"],
                "code_signature": spec["code_signature"],
                "node_schema": json.dumps(schema.get("node_feature_names", NODE_FEATURE_NAMES)),
                "edge_schema": str(graph.get("edge_schema", "base6")),
                "node_selection_mode": str(graph.get("node_support_mode", "unknown")),
                "readout_mode": "gated_global_mean_max_weighted",
                "structure_regularization": json.dumps(training.get("graph_regularization", {}), sort_keys=True),
                "structure_mode_mix": json.dumps(training.get("structure_mode_mix", {}), sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "01_run_manifest.csv", index=False)
    return frame


def flatten_dict(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            out.update(flatten_dict(child, name))
        else:
            out[name] = child
    return out


def config_category(key: str) -> str:
    routes = [
        ("node support", ["graph.node_support", "graph.target_node", "graph.bins"]),
        ("node features", ["node_feature", "model.input"]),
        ("local edges", ["local_edge"]),
        ("kNN edges", ["graph.knn"]),
        ("structure edges", ["graph.structure"]),
        ("edge attributes", ["edge_schema", "edge_attr_dim", "edge_feature"]),
        ("edge gates", ["scalar_edge_gate", "edge_gate"]),
        ("readout", ["readout"]),
        ("loss", ["loss", "penalty_lambda"]),
        ("graph corruption", ["drop_edge", "graph_regularization", "structure_mode_mix"]),
        ("optimizer/training schedule", ["training.lr", "weight_decay", "scheduler", "early_stopping", "max_epochs", "checkpoint_monitor", "amp"]),
    ]
    for category, needles in routes:
        if any(needle in key for needle in needles):
            return category
    return "other runtime/config"


def write_config_diff(specs: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    representatives = [
        next(s for s in specs if s["family"] == "D17_OFIX15"),
        next(s for s in specs if s["family"] == "D18_OFIX16"),
        next(s for s in specs if s["family"] == "D18_OFIX17_B"),
    ]
    names = ["D17_OFIX15", "D18_OFIX16", "D18_OFIX17_B"]
    flats = [flatten_dict(s["cfg"]) for s in representatives]
    keys = sorted(set().union(*(set(x) for x in flats)))
    grouped: Dict[str, List[List[Any]]] = defaultdict(list)
    for key in keys:
        values = [flat.get(key, "MISSING") for flat in flats]
        if len({json.dumps(v, sort_keys=True, default=str) for v in values}) > 1:
            grouped[config_category(key)].append([key, *values])
    lines = [
        "# Exact Semantic Config Diff",
        "",
        "The comparison uses each run's `resolved_config.yaml`, not directory names. Cache paths are runtime I/O details; graph semantics are described separately.",
        "",
    ]
    categories = ["node support", "node features", "local edges", "kNN edges", "structure edges", "edge attributes", "edge gates", "readout", "loss", "graph corruption", "optimizer/training schedule", "other runtime/config"]
    for category in categories:
        lines += [f"## {category}", ""]
        rows = grouped.get(category, [])
        if not rows:
            lines += ["No semantic difference recorded in resolved configs.", ""]
            continue
        lines += ["| key | D17 OFIX15 | D18 OFIX16 | D18 OFIX17-B |", "|---|---|---|---|"]
        for key, *values in rows:
            rendered = [json.dumps(v, sort_keys=True, default=str).replace("|", "\\|") for v in values]
            lines.append(f"| `{key}` | {rendered[0]} | {rendered[1]} | {rendered[2]} |")
        lines.append("")
    lines += [
        "## OFIX16 vs OFIX17-B confounds",
        "",
        "- Official graph schema, node support, node features, local edges, kNN edges, structure-edge construction, readout, optimizer, scheduler, and checkpoint monitor are matched.",
        "- OFIX16 uses global `training.drop_edge_p=0.15`; OFIX17-B sets it to 0 and instead drops only structure edges with probability 0.3.",
        "- OFIX17-B additionally enables per-sample structure-mode mixing (`p_forced_structure=0.3`), so structure-only DropEdge and mode mixing are not isolated from each other.",
        "- Best/last checkpoints occur at different epochs. Best-vs-last is therefore analyzed explicitly rather than treated as interchangeable.",
        "- The exact training git commit was not preserved in local artifacts. Current repo signature is recorded, but it is not asserted to be the training code state.",
        "",
    ]
    (output_dir / "02_exact_config_diff.md").write_text("\n".join(lines), encoding="utf-8")


def write_graph_reports(stats: pd.DataFrame, hashes: pd.DataFrame, supports: pd.DataFrame, output_dir: Path) -> None:
    summary = stats.groupby(["graph_profile", "mode"], as_index=False).agg(
        samples=("sample_index", "count"),
        nodes_mean=("number_of_nodes", "mean"),
        local_edges_mean=("number_of_local_edges", "mean"),
        knn_edges_mean=("number_of_knn_edges", "mean"),
        structure_edges_mean=("number_of_structure_edges", "mean"),
        components_mean=("connected_component_count", "mean"),
        largest_component_ratio_mean=("largest_component_ratio", "mean"),
        diameter_approx_mean=("diameter_approx", "mean"),
        shortest_path_approx_mean=("mean_shortest_path_approx", "mean"),
        node_overlap_mean=("node_support_overlap_with_official", "mean"),
        edge_jaccard_mean=("overall_edge_jaccard_with_official", "mean"),
        structure_jaccard_mean=("structure_edge_jaccard_with_official", "mean"),
    )
    class_summary = stats.groupby(["graph_profile", "mode", "true_class"], as_index=False).agg(
        samples=("sample_index", "count"),
        structure_edges_mean=("number_of_structure_edges", "mean"),
        edge_jaccard_mean=("overall_edge_jaccard_with_official", "mean"),
        largest_component_ratio_mean=("largest_component_ratio", "mean"),
    )
    zero_forced = hashes.pivot_table(index=["graph_profile", "sample_index"], columns="mode", values=["edge_index_hash", "edge_attr_hash", "complete_graph_hash"], aggfunc="first")
    zero_forced_edge_equal = float((zero_forced[("edge_index_hash", "zero_prior")] == zero_forced[("edge_index_hash", "forced_fallback")]).mean())
    zero_forced_attr_equal = float((zero_forced[("edge_attr_hash", "zero_prior")] == zero_forced[("edge_attr_hash", "forced_fallback")]).mean())
    zero_forced_complete_equal = float((zero_forced[("complete_graph_hash", "zero_prior")] == zero_forced[("complete_graph_hash", "forced_fallback")]).mean())
    hash_unique = hashes.groupby(["graph_profile", "mode"], as_index=False).agg(
        samples=("sample_index", "count"),
        unique_node_hashes=("unordered_node_coordinate_set_hash", "nunique"),
        unique_edge_hashes=("edge_index_hash", "nunique"),
        unique_structure_hashes=("edge_index_structure_hash", "nunique"),
        unique_complete_hashes=("complete_graph_hash", "nunique"),
    )
    fallback = hashes[hashes.landmark_missing_flag == 1]
    fallback_identical = fallback.groupby(["graph_profile", "mode"])["edge_index_hash"].agg(["count", "nunique"]).reset_index()
    lines = [
        "# Graph Mode Statistics Summary",
        "",
        f"Deterministic stratified sample size: {stats.sample_index.nunique()} images. All graph rows preserve the same image IDs and labels.",
        "",
        "## Per-mode topology",
        "",
        markdown_table(summary, summary.columns),
        "",
        "## Measured mode identities",
        "",
        f"- `zero_prior` vs `forced_fallback` edge topology equality: {zero_forced_edge_equal*100:.2f}%.",
        f"- `zero_prior` vs `forced_fallback` edge-attribute equality: {zero_forced_attr_equal*100:.2f}%.",
        f"- Complete-object equality: {zero_forced_complete_equal*100:.2f}% (the modes intentionally differ in detected/missing metadata even when model inputs are equal).",
        "- Shuffle effects are reported through node-support, edge-topology, and edge-attribute hashes rather than inferred from code alone.",
        "",
        "## Unique hashes by mode",
        "",
        markdown_table(hash_unique, hash_unique.columns),
        "",
        "## Natural fallback graph identity",
        "",
        markdown_table(fallback_identical, fallback_identical.columns),
        "",
        "## Per-class summary",
        "",
        markdown_table(class_summary, class_summary.columns),
        "",
    ]
    (output_dir / "03_graph_mode_statistics_summary.md").write_text("\n".join(lines), encoding="utf-8")
    support_summary = supports.groupby(["graph_profile", "mode", "true_class"], as_index=False).mean(numeric_only=True)
    support_lines = [
        "# Node Support Information",
        "",
        "Node selection is computed from pixel detail only. Facial-region priors are not injected into any model for this analysis. The table uses coordinates, quadrants, intensity, and gradient statistics only.",
        "",
        markdown_table(support_summary, support_summary.columns),
        "",
    ]
    (output_dir / "12_node_support_information.md").write_text("\n".join(support_lines), encoding="utf-8")


def write_training_report(curves: pd.DataFrame, specs: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run_id, group in curves.groupby("run_id"):
        best_idx = group.val_macro_f1.idxmax()
        best = group.loc[best_idx]
        last = group.sort_values("epoch").iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "epochs_logged": len(group),
                "official_val_peak_epoch": int(best.epoch),
                "peak_val_macro_f1": float(best.val_macro_f1),
                "peak_train_macro_f1": float(best.train_macro_f1),
                "peak_train_val_gap_pp": 100.0 * float(best.train_macro_f1 - best.val_macro_f1),
                "last_epoch": int(last.epoch),
                "last_val_macro_f1": float(last.val_macro_f1),
                "last_train_val_gap_pp": 100.0 * float(last.train_macro_f1 - last.val_macro_f1),
            }
        )
    summary = pd.DataFrame(rows)
    lines = [
        "# Training Curve Audit",
        "",
        markdown_table(summary, summary.columns),
        "",
        "## Direct answers",
        "",
        "- Underfitting and overfitting are reported through the measured train/validation macro-F1 levels and gaps above; no missing train macro-F1 was inferred from accuracy.",
        "- The official validation peak epoch is the `argmax(val_macro_f1)` from each existing train log.",
        "- Best-vs-last robustness and accuracy are compared in `13_checkpoint_robustness_table.csv`; checkpoint conclusions are not transferred across epochs.",
        "",
    ]
    (output_dir / "10_training_curve_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_probe_report(probe: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Structure Signal Probe",
        "",
        "The probe reuses each trained layer in evaluation mode and computes its exact edge embedding, vector gate, message, and destination aggregation on official graphs. No parameters or forward outputs are changed.",
        "",
        "Exact attribution is available for pre/post vector-gate message norms and aggregate norm shares by edge type because D18 retains `edge_type`. Scalar-gate attribution is `MISSING` for checkpoints whose scalar gate is disabled.",
        "",
        "The classifier is a nonlinear MLP after concatenated mean/max/gated pooling, so an exact additive class-logit decomposition by edge type is not identifiable. The report therefore records graph-embedding norm and does not invent classifier contributions.",
        "",
        markdown_table(probe, [c for c in probe.columns if c in {"run_id", "layer", "edge_type", "probe_sample_count", "pre_gate_message_norm", "post_vector_gate_message_norm", "post_scalar_gate_message_norm", "vector_gate_mean", "aggregate_message_norm_share", "layer_representation_change_norm", "graph_embedding_norm"}]),
        "",
    ]
    (output_dir / "08_structure_signal_probe.md").write_text("\n".join(lines), encoding="utf-8")


def write_class_detection_report(predictions: pd.DataFrame, ablations: pd.DataFrame, output_dir: Path) -> None:
    official = predictions[predictions["mode"] == "official"]
    detection_counts = official.drop_duplicates(["run_id", "sample_index"]).groupby(["run_id", "detected_state", "true_class"]).size().reset_index(name="count")
    per_class = ablations[["run_id", "ablation"] + [f"f1_{name}" for name in CLASS_NAMES]].copy()
    lines = [
        "# Class and Detection Audit",
        "",
        "## Detected vs landmark-missing class counts",
        "",
        markdown_table(detection_counts, detection_counts.columns),
        "",
        "## Per-class F1 under edge ablations",
        "",
        markdown_table(per_class, per_class.columns),
        "",
        "Structure benefit/damage is the difference between `A_full_official` and `B_remove_structure` for the same checkpoint. OFIX17-B recovery is compared checkpoint-to-checkpoint on the identical sampled image IDs; the raw values above are retained so no class claim relies on percentages alone.",
        "",
    ]
    (output_dir / "11_class_and_detection_audit.md").write_text("\n".join(lines), encoding="utf-8")


def checkpoint_robustness_table(
    predictions_summary: pd.DataFrame,
    representations: pd.DataFrame,
    ablations: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    all_group = predictions_summary[predictions_summary.detection_group == "all"]
    for run_id, group in all_group.groupby("run_id"):
        by_mode = group.set_index("mode")
        official = by_mode.loc["official"]
        values = [float(by_mode.loc[mode, "macro_f1"]) for mode in MODES[1:]]
        rep = representations[(representations.run_id == run_id) & (representations["mode"] != "official")]
        ablation = ablations[(ablations.run_id == run_id)].set_index("ablation")
        base_name = run_id.rsplit("_", 1)[0] if run_id.endswith(("_best", "_last")) else run_id
        curve = curves[curves.run_id == base_name]
        if curve.empty:
            gap = math.nan
        else:
            checkpoint_type = "last" if run_id.endswith("_last") else "best"
            selected_epoch = int(curve.epoch.max()) if checkpoint_type == "last" else int(curve.loc[curve.val_macro_f1.idxmax(), "epoch"])
            point = curve[curve.epoch == selected_epoch]
            gap = 100.0 * float(point.train_macro_f1.iloc[0] - point.val_macro_f1.iloc[0]) if not point.empty else math.nan
        rows.append(
            {
                "run_id": run_id,
                "official_macro": float(official.macro_f1),
                "zero_macro": float(by_mode.loc["zero_prior", "macro_f1"]),
                "shuffle_macro": float(by_mode.loc["shuffle_prior", "macro_f1"]),
                "forced_macro": float(by_mode.loc["forced_fallback", "macro_f1"]),
                "robust_min": min(values),
                "robust_avg": float(np.mean(values)),
                "official_to_zero_drop": float(official.macro_f1 - by_mode.loc["zero_prior", "macro_f1"]),
                "official_to_shuffle_drop": float(official.macro_f1 - by_mode.loc["shuffle_prior", "macro_f1"]),
                "official_to_forced_drop": float(official.macro_f1 - by_mode.loc["forced_fallback", "macro_f1"]),
                "official_ECE": float(official.ece_15bin),
                "forced_ECE": float(by_mode.loc["forced_fallback", "ece_15bin"]),
                "mean_representation_similarity": float(rep.paired_cosine_similarity_mean.mean()),
                "structure_edge_ablation_drop": float(ablation.loc["A_full_official", "macro_f1"] - ablation.loc["B_remove_structure", "macro_f1"]),
                "train_val_gap": gap,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "13_checkpoint_robustness_table.csv", index=False)
    return frame


def write_hypothesis_matrix(robustness: pd.DataFrame, probe: pd.DataFrame, output_dir: Path) -> None:
    b_last = robustness[robustness.run_id == "d18_ofix17b_structure_mode_mix_seed42_last"]
    b_row = b_last.iloc[0] if not b_last.empty else None
    structure_probe = probe[(probe.run_id == "d18_ofix17b_structure_mode_mix_seed42_last") & (probe.edge_type == "structure")]
    structure_share = float(structure_probe.aggregate_message_norm_share.mean()) if not structure_probe.empty else math.nan
    lines = [
        "# Hypothesis Evidence Matrix",
        "",
        "This file synthesizes measured evidence only. It intentionally contains no OFIX18/OFIX19 architecture recommendation.",
        "",
        "| hypothesis | supporting evidence | contradicting evidence | unresolved observations | confidence |",
        "|---|---|---|---|---|",
    ]
    if b_row is None:
        h1_support = h1_contra = h2_support = h2_contra = h3_support = h3_contra = h4_support = h4_contra = "MISSING"
    else:
        h1_support = f"OFIX17-B last robust_min={b_row.robust_min:.4f}; representation invariance and pixel-only comparator are tabulated."
        h1_contra = f"Official macro={b_row.official_macro:.4f}; structure-ablation drop={b_row.structure_edge_ablation_drop:.4f}."
        h2_support = f"Measured structure aggregate norm share={structure_share:.4f}; small ablation drop would support ignoring."
        h2_contra = "Non-zero structure message share or class-specific ablation damage contradicts complete ignoring."
        h3_support = "Hash audit directly measures zero/forced equality and fallback topology collisions."
        h3_contra = "Prediction/representation changes under shuffled topology contradict a fully shared graph explanation."
        h4_support = "Training curves and best-vs-last official/robustness trade-offs quantify averaging/underfit behavior."
        h4_contra = "Stable class separation or selective edge-family use contradicts simple indiscriminate averaging."
    rows = [
        ("H1 stronger pixel evidence with structure guidance", h1_support, h1_contra, "Causal training-time attribution remains unavailable without retraining; inference probes are associative.", "medium"),
        ("H2 robustness mainly from ignoring structure", h2_support, h2_contra, "Message norm is not equivalent to class-logit causal contribution because readout/classifier are nonlinear.", "medium"),
        ("H3 counterfactuals retain shared support/topology", h3_support, h3_contra, "Complete-object hash includes metadata; topology and model-input hashes must be interpreted separately.", "high"),
        ("H4 graph-distribution averaging causes official loss", h4_support, h4_contra, "No retrained single-factor control isolates mode mixing from structure-only DropEdge.", "medium"),
    ]
    for hypothesis, support, contra, unresolved, confidence in rows:
        lines.append(f"| {hypothesis} | {support} | {contra} | {unresolved} | {confidence} |")
    lines.append("")
    (output_dir / "14_hypothesis_evidence_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    output_dir: Path,
    specs: Sequence[Dict[str, Any]],
    failures: Sequence[str],
    sample_count: int,
    device: torch.device,
    smoke_passed: bool,
) -> None:
    lines = [
        "# OFIX15-OFIX17 Predecision Diagnostic Audit",
        "",
        "## Purpose",
        "",
        "Distinguish H1-H4 using existing D17/D18 checkpoints without training or changing model behavior.",
        "",
        "## Included checkpoints",
        "",
    ]
    lines.extend([f"- `{spec['run_id']}`: `{Path(spec['checkpoint_path']).relative_to(PROJECT_ROOT)}` (epoch {spec['checkpoint_epoch']})." for spec in specs])
    lines += [
        "",
        "## Environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- Device: `{device}`",
        f"- Deterministic seed: {SEED}",
        f"- Stratified audit images: {sample_count}",
        f"- Smoke status: {'PASS' if smoke_passed else 'FAIL'}",
        "",
        "## Commands",
        "",
        "See `16_run_commands.md` for exact commands.",
        "",
        "## Unavailable or bounded evidence",
        "",
        "- Exact training git commit/code state was not preserved in the copied run artifacts; current repo HEAD/diff signature and checkpoint/config hashes are recorded instead.",
        "- Exact additive classifier-logit attribution by edge type is unavailable because pooled features pass through a nonlinear MLP.",
        "- The structure-message probe uses a deterministic bounded subset; sample count is recorded in the CSV.",
    ]
    if failures:
        lines.extend([f"- {failure}" for failure in failures])
    else:
        lines.append("- No requested checkpoint/config artifact was unavailable.")
    lines += [
        "",
        "No conclusion in this package exceeds measured evidence, and no next-architecture recommendation is included.",
        "",
    ]
    (output_dir / "00_README.md").write_text("\n".join(lines), encoding="utf-8")


def write_commands(output_dir: Path, args: argparse.Namespace) -> None:
    python = sys.executable
    command = (
        f'"{python}" -B d18/scripts/audit_ofix18_predecision.py '
        f'--prior_dir "{args.prior_dir}" --output_dir "{args.output_dir}" '
        f'--per_regular_class {args.per_regular_class} --batch_size {args.batch_size} '
        f'--probe_count {args.probe_count} --device {args.device}'
    )
    smoke = command + " --smoke-only"
    lines = ["# Exact Run Commands", "", "```powershell", smoke, command, "```", ""]
    (output_dir / "16_run_commands.md").write_text("\n".join(lines), encoding="utf-8")


def machine_summary(
    output_dir: Path,
    manifest: pd.DataFrame,
    graph_stats: pd.DataFrame,
    predictions_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    failures: Sequence[str],
) -> None:
    payload = {
        "schema_version": 1,
        "seed": SEED,
        "sample_count": int(graph_stats.sample_index.nunique()),
        "checkpoints": manifest.to_dict(orient="records"),
        "graph_mode_summary": graph_stats.groupby(["graph_profile", "mode"], as_index=False).mean(numeric_only=True).to_dict(orient="records"),
        "prediction_summary": predictions_summary.to_dict(orient="records"),
        "checkpoint_robustness": robustness.to_dict(orient="records"),
        "failures_and_skips": list(failures),
        "artifact_paths": sorted(str(path.relative_to(PROJECT_ROOT)) for path in output_dir.glob("*")),
    }
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(child) for key, child in value.items()}
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, tuple):
            return [clean(child) for child in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    (output_dir / "15_machine_readable_summary.json").write_text(
        json.dumps(clean(payload), indent=2, default=json_default, allow_nan=False), encoding="utf-8"
    )


def smoke_validate(
    specs: Sequence[Dict[str, Any]],
    files: Sequence[Path],
    donor_indices: np.ndarray,
    profiles: Mapping[str, Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    store, _, hashes, _ = build_graph_store(files, donor_indices, profiles, output_dir, smoke_only=True)
    for profile_name in profiles:
        for index in range(min(8, len(files))):
            zero = store[(profile_name, "zero_prior", index)]
            forced = store[(profile_name, "forced_fallback", index)]
            if graph_hashes(zero)["edge_index_hash"] != graph_hashes(forced)["edge_index_hash"]:
                raise RuntimeError(f"Expected zero/forced topology equality failed for {profile_name}/{files[index].name}")
            if int(zero.y) != int(forced.y) or int(zero.sample_index) != int(forced.sample_index):
                raise RuntimeError("Counterfactual changed label or sample identity")
    smoke_specs = list(specs)
    smoke_files = files[: min(8, len(files))]
    smoke_predictions, _, _ = run_predictions(smoke_specs, smoke_files, store, device, batch_size, output_dir, smoke_only=True)
    expected = len(smoke_specs) * len(smoke_files) * len(MODES)
    if len(smoke_predictions) != expected:
        raise RuntimeError(f"Smoke prediction row count {len(smoke_predictions)} != {expected}")
    emit("smoke_pass", samples=len(smoke_files), checkpoints=len(smoke_specs), modes=len(MODES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_predecision_audit")
    parser.add_argument("--per_regular_class", type=int, default=110, help="Classes with enough examples; rare disgust uses all available samples.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--probe_count", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)

    output_dir = PROJECT_ROOT / args.output_dir
    complete_marker = output_dir / "AUDIT_COMPLETE.json"
    progress_marker = output_dir / ".audit_in_progress"
    if complete_marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit report: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not progress_marker.exists():
        raise FileExistsError(f"Refusing to overwrite an existing non-audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_marker.write_text(json.dumps({"started_at": time.time(), "pid": os.getpid()}), encoding="utf-8")
    prior_dir = PROJECT_ROOT / args.prior_dir
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")

    specs, failures = prepare_specs()
    requested = len(RUN_SPECS)
    if len(specs) != requested:
        raise RuntimeError("Requested checkpoint/config matching failed: " + " | ".join(failures))
    files, sample_manifest = select_stratified_files(prior_dir, args.per_regular_class)
    if len(files) < 700:
        raise RuntimeError(f"Stratified audit requires at least 700 images, selected {len(files)}")
    sample_manifest.to_csv(output_dir / "sample_manifest.csv", index=False)
    rng = np.random.default_rng(SEED)
    donor_indices = rng.permutation(len(files))
    profiles = {
        "base6_structure": {"cfg": next(s["cfg"] for s in specs if s["graph_profile"] == "base6_structure"), "cache_root": PROJECT_ROOT / PROFILE_CACHE["base6_structure"]},
        "purified_structure": {"cfg": next(s["cfg"] for s in specs if s["graph_profile"] == "purified_structure"), "cache_root": PROJECT_ROOT / PROFILE_CACHE["purified_structure"]},
        "d17_pixel_only": {"cfg": next(s["cfg"] for s in specs if s["graph_profile"] == "base6_structure"), "cache_root": PROJECT_ROOT / PROFILE_CACHE["d17_pixel_only"]},
    }

    manifest = write_manifest(specs, output_dir)
    write_config_diff(specs, output_dir)
    write_commands(output_dir, args)
    smoke_validate(specs, files, donor_indices, profiles, output_dir, device, args.batch_size)
    if args.smoke_only:
        write_readme(output_dir, specs, failures, len(files), device, True)
        return

    emit("full_audit_start", samples=len(files), checkpoints=len(specs), device=str(device))
    graph_store, graph_stats, graph_hashes_frame, supports = build_graph_store(files, donor_indices, profiles, output_dir, smoke_only=False)
    write_graph_reports(graph_stats, graph_hashes_frame, supports, output_dir)
    predictions, prediction_summary, embeddings = run_predictions(specs, files, graph_store, device, args.batch_size, output_dir, smoke_only=False)
    sensitivity = probability_sensitivity(predictions, output_dir)
    ablations = run_edge_ablations(specs, files, graph_store, predictions, device, args.batch_size, output_dir)
    probe_count = min(args.probe_count, len(files))
    probe = run_structure_probe(specs, graph_store, device, args.batch_size, output_dir, probe_count)
    write_probe_report(probe, output_dir)
    representations = representation_summary(specs, predictions, embeddings, output_dir)
    curves = training_curve_audit(specs, output_dir)
    write_training_report(curves, specs, output_dir)
    write_class_detection_report(predictions, ablations, output_dir)
    robustness = checkpoint_robustness_table(prediction_summary, representations, ablations, curves, output_dir)
    write_hypothesis_matrix(robustness, probe, output_dir)
    write_readme(output_dir, specs, failures, len(files), device, True)
    machine_summary(output_dir, manifest, graph_stats, prediction_summary, robustness, failures)
    emit(
        "audit_complete",
        output_dir=str(output_dir),
        files_created=len(list(output_dir.iterdir())),
        checkpoints=[spec["run_id"] for spec in specs],
        unavailable=failures,
        smoke="PASS",
    )
    complete_marker.write_text(
        json.dumps({"completed_at": time.time(), "files_created": len(list(output_dir.iterdir()))}, indent=2),
        encoding="utf-8",
    )
    progress_marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
