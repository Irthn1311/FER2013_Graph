"""Read-only diagnostics for the D19 pre-implementation review.

This script never updates model/config/checkpoint files and never performs an
optimizer step. It reconstructs D18 graph families for the locked 715-image set
and runs bounded inference on frozen C2 checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import D18Batch, collate_d18_graphs
from d18.data.structure_graph_builder import (
    DEFAULT_RELATIONS,
    GROUP_INDICES,
    NODE_FEATURE_NAMES,
    _group_score,
    _knn_edges,
    _local_edges,
    _select_group_nodes,
    _unique_directed_edges,
    compute_detail_score,
    compute_pixel_feature_maps,
)
from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.models.structure_gnn import StructureGNN


SEEDS = (7, 21, 42, 84, 123)
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
CLASS_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pair)) for pair in edges.T.tolist()}


def reverse_fraction(pairs: set[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    return float(sum((v, u) in pairs for u, v in pairs) / len(pairs))


def relation_structure_edges(
    coords: np.ndarray,
    part_node: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], set[int]], int]:
    """Reproduce the builder loop while retaining multi-relation collisions."""
    if not bool(cfg.get("enabled", True)):
        return set(), {}, 0
    max_nodes = int(cfg.get("max_nodes_per_group", 32) or 32)
    targets_per_source = int(cfg.get("targets_per_source", 4) or 4)
    min_score = float(cfg.get("min_score", 0.05))
    bidirectional = bool(cfg.get("bidirectional", True))
    relations = [(str(a), str(b)) for a, b in (cfg.get("relations") or DEFAULT_RELATIONS)]
    group_scores = {name: _group_score(part_node, name) for rel in relations for name in rel}
    group_nodes = {
        name: _select_group_nodes(score, max_nodes, min_score)
        for name, score in group_scores.items()
    }
    pos = coords.astype(np.float32) / 47.0
    raw_count = 0
    memberships: dict[tuple[int, int], set[int]] = defaultdict(set)
    for rel_id, (src_group, dst_group) in enumerate(relations, start=1):
        for src in group_nodes.get(src_group, np.zeros((0,), dtype=np.int64)).tolist():
            dst_nodes = group_nodes.get(dst_group, np.zeros((0,), dtype=np.int64))
            candidates = dst_nodes[dst_nodes != int(src)]
            if candidates.size == 0:
                continue
            dist = np.sum((pos[candidates] - pos[int(src)]) ** 2, axis=1)
            chosen = candidates[np.argsort(dist, kind="mergesort")[: min(targets_per_source, candidates.size)]]
            for dst in chosen.tolist():
                memberships[(int(src), int(dst))].add(rel_id)
                raw_count += 1
                if bidirectional:
                    memberships[(int(dst), int(src))].add(rel_id)
                    raw_count += 1
    return set(memberships), dict(memberships), raw_count


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str})
    if len(frame) != 715:
        raise RuntimeError(f"Expected locked 715-image manifest, got {len(frame)}")
    digest = hashlib.sha256(frame["sample_index"].to_numpy(dtype="int64").tobytes()).hexdigest()
    if digest != LOCKED_SAMPLE_SHA256:
        raise RuntimeError(f"Locked sample hash mismatch: {digest}")
    frame["image_id"] = frame["image_id"].str.zfill(6)
    return frame


def topology_audit(
    manifest: pd.DataFrame,
    prior_root: Path,
    cache_root: Path,
    graph_cfg: dict[str, Any],
    output: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    feature_samples: list[np.ndarray] = []
    class_coverage: list[dict[str, Any]] = []
    knn_cfg = dict(graph_cfg.get("knn_edges") or {})
    structure_cfg = dict(graph_cfg.get("structure_edges") or {})
    for row_i, row in manifest.iterrows():
        image_id = str(row["image_id"]).zfill(6)
        graph = load_d18_graph_cache(cache_root / "test" / f"{image_id}.npz")
        prior_path = prior_root / "test" / f"{image_id}.npz"
        with np.load(prior_path, allow_pickle=False) as data:
            image = data["image_48"].astype(np.float32)
            part_masks = data["part_soft_masks"].astype(np.float32)
        image = image / 255.0 if float(np.max(image)) > 1.0 else image
        maps = compute_pixel_feature_maps(image)
        detail = compute_detail_score(maps)
        coords = np.rint((graph.pos.numpy() + 1.0) * 47.0 / 2.0).astype(np.int64)
        yy, xx = coords[:, 1], coords[:, 0]
        # graph.pos stores x,y while builder helpers consume y,x.
        coords_yx = np.stack([yy, xx], axis=1)
        x_norm = (xx.astype(np.float32) / 47.0) * 2.0 - 1.0
        y_norm = (yy.astype(np.float32) / 47.0) * 2.0 - 1.0
        x_exact = np.stack(
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
        local_raw = _local_edges(coords_yx)
        # Cache node features are float16. The original graph was constructed
        # from these FP32 maps, so kNN must be reconstructed from x_exact.
        knn_raw = _knn_edges(x_exact, NODE_FEATURE_NAMES, knn_cfg)
        local = edge_set(_unique_directed_edges(local_raw))
        knn = edge_set(_unique_directed_edges(knn_raw))
        part_node = np.transpose(part_masks[:, yy, xx], (1, 0)).astype(np.float32)
        structure, relation_memberships, structure_raw_count = relation_structure_edges(
            coords_yx, part_node, structure_cfg
        )
        union = local | knn | structure
        local_knn = local & knn
        local_structure = local & structure
        knn_structure = knn & structure
        triple = local & knn & structure
        final_pairs = edge_set(graph.edge_index.numpy())
        final_duplicate_count = int(graph.edge_index.shape[1] - len(final_pairs))
        selected_mask = np.zeros((48, 48), dtype=bool)
        selected_mask[yy, xx] = True
        grad = maps["grad_mag"]
        high_grad = grad >= np.quantile(grad, 0.90)
        sampled_idx = np.linspace(0, graph.x.shape[0] - 1, 100, dtype=np.int64)
        feature_samples.append(x_exact[sampled_idx])
        selected_detail = detail[selected_mask]
        omitted_detail = detail[~selected_mask]
        knn_arr = np.asarray(list(knn), dtype=np.int64)
        if knn_arr.size:
            delta = coords_yx[knn_arr[:, 0]] - coords_yx[knn_arr[:, 1]]
            knn_pixel_dist = np.sqrt(np.sum(delta.astype(np.float32) ** 2, axis=1))
        else:
            knn_pixel_dist = np.zeros((0,), dtype=np.float32)
        degree_total = np.bincount(graph.edge_index[1].numpy(), minlength=graph.x.shape[0])
        type_np = graph.edge_type.numpy()
        type_degrees = {
            name: np.bincount(
                graph.edge_index[1].numpy()[type_np == type_id], minlength=graph.x.shape[0]
            )
            for name, type_id in (("local", 0), ("knn", 1), ("structure", 2))
        }
        multiple_relation_pairs = sum(len(ids) > 1 for ids in relation_memberships.values())
        result = {
            "sample_index": int(row["sample_index"]),
            "image_id": image_id,
            "true_class": int(row["true_class"]),
            "detected": bool(str(row["detected_state"]).lower() == "true"),
            "node_count": int(graph.x.shape[0]),
            "retained_pixel_fraction": float(graph.x.shape[0] / 2304.0),
            "raw_local_count": len(local),
            "raw_knn_count": len(knn),
            "raw_structure_count": len(structure),
            "raw_structure_preunique_count": structure_raw_count,
            "local_knn_overlap": len(local_knn),
            "local_structure_overlap": len(local_structure),
            "knn_structure_overlap": len(knn_structure),
            "three_way_overlap": len(triple),
            "union_count": len(union),
            "cached_final_count": int(graph.edge_index.shape[1]),
            "cached_final_duplicate_count": final_duplicate_count,
            "cached_union_exact_match": final_pairs == union,
            "local_self_loops": sum(u == v for u, v in local),
            "knn_self_loops": sum(u == v for u, v in knn),
            "structure_self_loops": sum(u == v for u, v in structure),
            "local_reverse_fraction": reverse_fraction(local),
            "knn_reverse_fraction": reverse_fraction(knn),
            "structure_reverse_fraction": reverse_fraction(structure),
            "multi_relation_endpoint_count": multiple_relation_pairs,
            "final_local_count": int((type_np == 0).sum()),
            "final_knn_count": int((type_np == 1).sum()),
            "final_structure_count": int((type_np == 2).sum()),
            "total_degree_mean": float(degree_total.mean()),
            "total_degree_std": float(degree_total.std()),
            "local_degree_mean": float(type_degrees["local"].mean()),
            "knn_degree_mean": float(type_degrees["knn"].mean()),
            "structure_degree_mean": float(type_degrees["structure"].mean()),
            "selected_detail_mean": float(selected_detail.mean()),
            "omitted_detail_mean": float(omitted_detail.mean()),
            "high_gradient_recall": float((selected_mask & high_grad).sum() / max(high_grad.sum(), 1)),
            "knn_pixel_distance_mean": float(knn_pixel_dist.mean()),
            "knn_pixel_distance_median": float(np.median(knn_pixel_dist)),
            "knn_long_range_gt8px_fraction": float((knn_pixel_dist > 8.0).mean()),
        }
        rows.append(result)
        class_coverage.append(
            {
                "true_class": int(row["true_class"]),
                "selected_detail_mean": result["selected_detail_mean"],
                "omitted_detail_mean": result["omitted_detail_mean"],
                "high_gradient_recall": result["high_gradient_recall"],
                "knn_long_range_gt8px_fraction": result["knn_long_range_gt8px_fraction"],
            }
        )
        if (row_i + 1) % 50 == 0:
            print(json.dumps({"event": "d19_topology_progress", "done": row_i + 1, "total": len(manifest)}), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "10_edge_overlap_per_image.csv", index=False)
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    summary_rows = []
    for column in numeric_columns:
        values = frame[column].astype(float)
        summary_rows.append(
            {
                "metric": column,
                "count": int(values.notna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "10_edge_overlap_summary.csv", index=False)
    class_frame = pd.DataFrame(class_coverage).groupby("true_class").agg(["mean", "std", "count"])
    class_frame.columns = ["_".join(map(str, col)) for col in class_frame.columns]
    class_frame.reset_index().to_csv(output / "12_node_coverage_by_class.csv", index=False)

    features = np.concatenate(feature_samples, axis=0).astype(np.float64)
    corr = np.corrcoef(features, rowvar=False)
    pd.DataFrame(corr, index=NODE_FEATURE_NAMES, columns=NODE_FEATURE_NAMES).to_csv(
        output / "12_node_feature_correlation.csv"
    )
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
    write_json(
        output / "12_node_graph_bottleneck_summary.json",
        {
            "locked_sample_count": len(frame),
            "node_feature_sample_count": int(features.shape[0]),
            "node_feature_effective_rank_of_10": effective_rank,
            "node_count_unique": sorted(int(x) for x in frame["node_count"].unique()),
            "retained_pixel_fraction_mean": float(frame["retained_pixel_fraction"].mean()),
            "high_gradient_recall_mean": float(frame["high_gradient_recall"].mean()),
            "knn_long_range_gt8px_fraction_mean": float(frame["knn_long_range_gt8px_fraction"].mean()),
            "deterministic_construction": True,
            "augmentation_configured_in_d18_dataset": False,
        },
    )


def stratified_subset(manifest: pd.DataFrame, per_class: int) -> pd.DataFrame:
    chunks = []
    for class_id in range(7):
        group = manifest[manifest["true_class"].astype(int) == class_id].sort_values("sample_index")
        if len(group) < per_class:
            raise RuntimeError(f"Class {class_id} has only {len(group)} locked samples")
        positions = np.linspace(0, len(group) - 1, per_class, dtype=np.int64)
        chunks.append(group.iloc[positions])
    return pd.concat(chunks, ignore_index=True).sort_values("sample_index").reset_index(drop=True)


def c2_run_dir(seed: int) -> Path:
    root = ROOT / ("outputs/d18_runs/ofix18" if seed == 42 else "outputs/d18_runs/ofix18seed")
    return root / f"d18_ofix18_c2_structure_mode_mix_only_seed{seed}"


def filter_batch(batch: D18Batch, remove_structure: bool) -> D18Batch:
    if not remove_structure:
        return batch
    keep = batch.edge_type_cat.long() != 2
    return D18Batch(
        x_cat=batch.x_cat,
        edge_index_cat=batch.edge_index_cat[:, keep],
        edge_attr_cat=batch.edge_attr_cat[keep],
        batch_index=batch.batch_index,
        ptr=batch.ptr,
        y=batch.y,
        sample_index=batch.sample_index,
        pos_cat=batch.pos_cat,
        detected=batch.detected,
        landmark_missing_flag=batch.landmark_missing_flag,
        image_48=batch.image_48,
        edge_type_cat=batch.edge_type_cat[keep],
        structure_relation_id_cat=batch.structure_relation_id_cat[keep],
        local_edge_count=batch.local_edge_count,
        knn_edge_count=batch.knn_edge_count,
        structure_edge_count=torch.zeros_like(batch.structure_edge_count),
        total_edge_count=batch.total_edge_count - batch.structure_edge_count,
        structure_edge_count_before_purification=batch.structure_edge_count_before_purification,
        structure_edge_count_after_purification=batch.structure_edge_count_after_purification,
        purification_compatibility_kept_mean=batch.purification_compatibility_kept_mean,
        purification_compatibility_dropped_mean=batch.purification_compatibility_dropped_mean,
        node_feature_names=batch.node_feature_names,
        edge_feature_names=batch.edge_feature_names,
    )


def pool_components(model: StructureGNN, h: torch.Tensor, batch: D18Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = h.shape[1]
    z = model.readout(h, batch.batch_index, batch.num_graphs)
    return z, z[:, :hidden], z[:, hidden : 2 * hidden], z[:, 2 * hidden :]


def effective_rank(matrix: np.ndarray) -> float:
    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    values = singular * singular
    if float(values.sum()) <= 1e-12:
        return 0.0
    p = values / values.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.astype(np.float64).mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / max(denominator, 1e-12))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    values = []
    for class_id in range(7):
        tp = int(((y == class_id) & (pred == class_id)).sum())
        fp = int(((y != class_id) & (pred == class_id)).sum())
        fn = int(((y == class_id) & (pred != class_id)).sum())
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


def graph_separation(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in range(7)])
    within = np.mean([np.mean(np.sum((embeddings[labels == c] - centroids[c]) ** 2, axis=1)) for c in range(7)])
    pairwise = []
    for i in range(7):
        for j in range(i + 1, 7):
            pairwise.append(float(np.sum((centroids[i] - centroids[j]) ** 2)))
    between = float(np.mean(pairwise))
    result = {
        "class_centroid_separation": float(math.sqrt(max(between, 0.0)) / max(math.sqrt(max(within, 0.0)), 1e-12)),
        "within_between_ratio": float(within / max(between, 1e-12)),
        "covariance_effective_rank": effective_rank(embeddings),
        "graph_representation_variance": float(np.var(embeddings, axis=0, ddof=1).mean()),
    }
    for class_id, name in enumerate(CLASS_NAMES):
        other = np.delete(centroids, class_id, axis=0).mean(axis=0)
        class_within = float(np.mean(np.sum((embeddings[labels == class_id] - centroids[class_id]) ** 2, axis=1)))
        result[f"class_separation_{name}"] = float(np.linalg.norm(centroids[class_id] - other) / max(math.sqrt(class_within), 1e-12))
    return result


def node_metrics(h: torch.Tensor, edge_index: torch.Tensor, ptr: torch.Tensor) -> dict[str, float]:
    pair_cos, variances, ranks, energies = [], [], [], []
    for graph_id in range(ptr.numel() - 1):
        start, end = int(ptr[graph_id]), int(ptr[graph_id + 1])
        local_h = h[start:end]
        count = min(128, local_h.shape[0])
        ids = torch.linspace(0, local_h.shape[0] - 1, count, device=h.device).long()
        sample = local_h[ids]
        normalized = torch.nn.functional.normalize(sample, dim=1)
        cosine = normalized @ normalized.T
        pair_cos.append(float((cosine.sum() - count).item() / max(count * (count - 1), 1)))
        variances.append(float(local_h.var(dim=0, unbiased=False).mean().item()))
        ranks.append(effective_rank(sample.detach().cpu().numpy()))
        edge_mask = (edge_index[0] >= start) & (edge_index[0] < end)
        graph_edges = edge_index[:, edge_mask]
        if graph_edges.numel():
            diff = h[graph_edges[0]] - h[graph_edges[1]]
            numerator = diff.pow(2).sum(dim=1).mean()
            centered = local_h - local_h.mean(dim=0, keepdim=True)
            denominator = centered.pow(2).sum(dim=1).mean().clamp_min(1e-12)
            energies.append(float((numerator / denominator).item()))
    return {
        "mean_pairwise_node_cosine": float(np.mean(pair_cos)),
        "node_representation_variance": float(np.mean(variances)),
        "node_covariance_effective_rank": float(np.mean(ranks)),
        "normalized_dirichlet_energy": float(np.mean(energies)),
    }


def collect_message_scale(layer, h: torch.Tensor, batch: D18Batch, seed: int, layer_id: int) -> list[dict[str, Any]]:
    edge_index = batch.edge_index_cat
    edge_attr = batch.edge_attr_cat.to(dtype=h.dtype)
    src, dst = edge_index[0].long(), edge_index[1].long()
    edge_emb = layer.edge_mlp(edge_attr)
    gate = torch.sigmoid(layer.gate(edge_emb))
    messages = layer.message(torch.cat([h[src], edge_emb], dim=1)) * gate
    total_degree = h.new_zeros((h.shape[0], 1))
    total_degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
    result = []
    for name, type_id in (("local", 0), ("knn", 1), ("structure", 2)):
        mask = batch.edge_type_cat.long() == type_id
        if not bool(mask.any()):
            continue
        family_dst = dst[mask]
        family_msg = messages[mask]
        aggregate = h.new_zeros(h.shape)
        aggregate.index_add_(0, family_dst, family_msg)
        own_degree = h.new_zeros((h.shape[0], 1))
        own_degree.index_add_(0, family_dst, torch.ones((family_dst.numel(), 1), device=h.device, dtype=h.dtype))
        current = aggregate / total_degree.clamp_min(1.0)
        own = aggregate / own_degree.clamp_min(1.0)
        active = own_degree[:, 0] > 0
        result.append(
            {
                "training_seed": seed,
                "layer": layer_id,
                "family": name,
                "edge_count": float(mask.sum().item() / max(batch.num_graphs, 1)),
                "mean_dst_degree": float(own_degree[active].mean().item()),
                "mean_total_degree_on_active_dst": float(total_degree[active].mean().item()),
                "message_l2_per_edge": float(family_msg.norm(dim=1).mean().item()),
                "aggregate_l2_current_full_degree": float(current[active].norm(dim=1).mean().item()),
                "aggregate_l2_own_degree": float(own[active].norm(dim=1).mean().item()),
                "own_to_current_scale_ratio": float(own[active].norm(dim=1).mean().item() / max(current[active].norm(dim=1).mean().item(), 1e-12)),
            }
        )
    return result


def layerwise_audit(
    manifest: pd.DataFrame,
    cache_root: Path,
    output: Path,
    device: torch.device,
    per_class: int,
    batch_size: int,
) -> None:
    subset = stratified_subset(manifest, per_class)
    subset.to_csv(output / "09_layerwise_subset_manifest.csv", index=False)
    graphs = [load_d18_graph_cache(cache_root / "test" / f"{str(image_id).zfill(6)}.npz") for image_id in subset["image_id"]]
    labels = subset["true_class"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = c2_run_dir(seed)
        cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        model = StructureGNN.from_config(cfg, input_dim=10, edge_attr_dim=6).to(device)
        payload = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        if seed == 42:
            grouped = defaultdict(int)
            for name, parameter in model.named_parameters():
                component = name.split(".", 1)[0]
                grouped[component] += parameter.numel()
            for component, count in sorted(grouped.items()):
                parameter_rows.append({"component": component, "parameter_count": count})
            parameter_rows.append({"component": "total", "parameter_count": sum(p.numel() for p in model.parameters())})
        mode_layers: dict[str, dict[str, list[np.ndarray]]] = {
            mode: {layer: [] for layer in ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "classifier_input")}
            for mode in ("official", "remove_structure")
        }
        mode_node_values: dict[str, dict[str, list[dict[str, float]]]] = {
            mode: defaultdict(list) for mode in ("official", "remove_structure")
        }
        readout_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
        with torch.inference_mode():
            for start in range(0, len(graphs), batch_size):
                raw_batch = collate_d18_graphs(graphs[start : start + batch_size]).to(device)
                for mode in ("official", "remove_structure"):
                    batch = filter_batch(raw_batch, mode == "remove_structure")
                    h = model.encoder(batch.x_cat)
                    layer_states = [("input_projection", h)]
                    dst = batch.edge_index_cat[1].long()
                    degree = h.new_zeros((h.shape[0], 1))
                    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device, dtype=h.dtype))
                    for layer_id, layer in enumerate(model.gnn.layers, start=1):
                        if mode == "official":
                            message_rows.extend(collect_message_scale(layer, h, batch, seed, layer_id))
                        h = layer(
                            h,
                            batch.edge_index_cat,
                            batch.edge_attr_cat,
                            dst_degree=degree,
                            edge_type=batch.edge_type_cat,
                        )
                        layer_states.append((f"gnn_layer_{layer_id}", h))
                    for layer_name, state in layer_states:
                        z, _, _, _ = pool_components(model, state, batch)
                        mode_layers[mode][layer_name].append(z.detach().cpu().numpy())
                        mode_node_values[mode][layer_name].append(node_metrics(state, batch.edge_index_cat, batch.ptr))
                    z, z_mean, z_max, z_gated = pool_components(model, h, batch)
                    classifier_input = model.classifier[0](z)
                    mode_layers[mode]["classifier_input"].append(classifier_input.detach().cpu().numpy())
                    if mode == "official":
                        variants = {
                            "full": z,
                            "zero_mean": torch.cat([torch.zeros_like(z_mean), z_max, z_gated], dim=1),
                            "zero_max": torch.cat([z_mean, torch.zeros_like(z_max), z_gated], dim=1),
                            "zero_gated": torch.cat([z_mean, z_max, torch.zeros_like(z_gated)], dim=1),
                        }
                        for name, variant in variants.items():
                            readout_predictions[name].append(model.classifier(variant).argmax(dim=1).cpu().numpy())

        arrays = {
            mode: {layer: np.concatenate(parts, axis=0) for layer, parts in layers.items()}
            for mode, layers in mode_layers.items()
        }
        for mode, layers in arrays.items():
            for layer_name, embedding in layers.items():
                metrics = graph_separation(embedding, labels)
                if layer_name in mode_node_values[mode]:
                    entries = mode_node_values[mode][layer_name]
                    for key in entries[0]:
                        metrics[key] = float(np.mean([entry[key] for entry in entries]))
                for metric, value in metrics.items():
                    metric_rows.append(
                        {"training_seed": seed, "mode": mode, "layer": layer_name, "metric": metric, "value": value}
                    )
        layer_names = ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "classifier_input")
        for i, left in enumerate(layer_names):
            for right in layer_names[i + 1 :]:
                cka_rows.append(
                    {"training_seed": seed, "comparison": "inter_layer_official", "left": left, "right": right, "linear_cka": linear_cka(arrays["official"][left], arrays["official"][right])}
                )
        for layer_name in layer_names:
            cka_rows.append(
                {"training_seed": seed, "comparison": "official_vs_remove", "left": layer_name, "right": layer_name, "linear_cka": linear_cka(arrays["official"][layer_name], arrays["remove_structure"][layer_name])}
            )
        for variant, pieces in readout_predictions.items():
            pred = np.concatenate(pieces)
            readout_rows.append(
                {"training_seed": seed, "variant": variant, "sample_count": len(labels), "accuracy": float((pred == labels).mean()), "macro_f1": macro_f1(labels, pred)}
            )
        print(json.dumps({"event": "d19_layerwise_seed_done", "seed": seed, "device": str(device)}), flush=True)

    pd.DataFrame(metric_rows).to_csv(output / "09_layerwise_information_audit.csv", index=False)
    pd.DataFrame(cka_rows).to_csv(output / "09_layerwise_cka.csv", index=False)
    pd.DataFrame(message_rows).to_csv(output / "10_message_scale_probe.csv", index=False)
    pd.DataFrame(readout_rows).to_csv(output / "08_readout_ablation.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "15_current_parameter_count.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv")
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--cache_dir", default="outputs/d18_graph_cache/ofix17_structure_reg/base6_shared")
    parser.add_argument("--config", default="configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml")
    parser.add_argument("--output_dir", default="outputs/d19_analysis/d19_preimplementation_review")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer_samples_per_class", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_topology", action="store_true")
    parser.add_argument("--skip_layers", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if not args.skip_topology:
        topology_audit(manifest, Path(args.prior_dir), Path(args.cache_dir), cfg["graph"], output)
    if not args.skip_layers:
        device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        layerwise_audit(manifest, Path(args.cache_dir), output, device, args.layer_samples_per_class, args.batch_size)
    topology_complete = (output / "10_edge_overlap_per_image.csv").exists()
    layerwise_complete = (output / "09_layerwise_information_audit.csv").exists()
    write_json(
        output / "diagnostic_execution.json",
        {
            "status": "COMPLETE",
            "locked_sample_count": len(manifest),
            "locked_sample_sha256": LOCKED_SAMPLE_SHA256,
            "topology_audit_complete": topology_complete,
            "layerwise_audit_complete": layerwise_complete,
            "topology_audit_executed_this_invocation": not args.skip_topology,
            "layerwise_audit_executed_this_invocation": not args.skip_layers,
            "training_or_finetuning_performed": False,
            "model_code_modified": False,
        },
    )


if __name__ == "__main__":
    main()
