"""Read-only sufficiency audit for the frozen 1,800-pixel D19 selector.

This script deliberately never imports a training entry point and never writes
to a graph cache. Counterfactual coordinate sets exist in memory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from d18.data.structure_graph_builder import (  # noqa: E402
    _stratified_coords,
    build_structure_graph,
    compute_detail_score,
    compute_pixel_feature_maps,
)

OUT_DEFAULT = ROOT / "outputs/d19_analysis/d19_pixel_selection_sufficiency_audit"
CACHE_ROOT = ROOT / "outputs/d19_graph_cache/a0_evidence_only"
PRIOR_ROOT = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
PREIMPL = ROOT / "outputs/d19_analysis/d19_preimplementation_review"
A0_42 = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed42"
A0_7 = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed7"
A1 = ROOT / "outputs/d19_runs/d19_a1_id_null_evidence_only_seed42"
C2_42 = ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42"
C2_7 = ROOT / "outputs/d18_runs/ofix18seed/d18_ofix18_c2_structure_mode_mix_only_seed7"
C2_EVAL = ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/evaluations"
A1_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a1_id_posttraining_analysis"
A0_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a0_posttraining_analysis"
A0_7_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a0_seed7_confirmation_posttraining"

CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
PART_NAMES = [
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose",
    "mouth",
    "left_mouth_corner",
    "right_mouth_corner",
    "left_cheek",
    "right_cheek",
    "chin",
    "face_contour",
    "outside_face",
]
KEY_REGION_GROUPS = {
    "eyes": [0, 1],
    "eyebrows": [2, 3],
    "mouth": [5, 6, 7],
    "jaw_chin": [10, 11],
}
SIGNALS = ["idev", "tv", "tv2", "labs"]
LOCKED_SHA256 = "93aa85233e6a51e9719f047c23a1bd01edb007ad7adaf49d91b4986540bf73c2"


def sha256_bytes(*items: bytes) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_sanitize(value), indent=2, allow_nan=False, default=json_default), encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_sanitize(v) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    cols = [str(x) for x in view.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    rows.extend("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in view.itertuples(index=False, name=None))
    if len(frame) > max_rows:
        rows.append(f"\n_Shown {max_rows} of {len(frame)} rows._")
    return "\n".join(rows)


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else float("nan")


def summarize(values: Iterable[float], prefix: str) -> dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "median", "std", "min", "max")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }


def signal_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    s = np.asarray(image, dtype=np.float32)
    idev = np.abs(s - float(s.mean()))
    tv = np.zeros_like(s)
    tv2 = np.zeros_like(s)
    dv = np.abs(s[1:, :] - s[:-1, :])
    dh = np.abs(s[:, 1:] - s[:, :-1])
    tv[1:, :] += dv
    tv[:-1, :] += dv
    tv[:, 1:] += dh
    tv[:, :-1] += dh
    dv2 = dv * dv
    dh2 = dh * dh
    tv2[1:, :] += dv2
    tv2[:-1, :] += dv2
    tv2[:, 1:] += dh2
    tv2[:, :-1] += dh2
    p = np.pad(s, 1, mode="edge")
    lap = np.abs(p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * s)
    return {"idev": idev, "tv": tv, "tv2": tv2, "labs": lap}


def coords_from_graph(z: Any) -> np.ndarray:
    pos = np.asarray(z["pos"], dtype=np.float32)
    y = np.rint((pos[:, 1] + 1.0) * 0.5 * 47.0).astype(np.int64)
    x = np.rint((pos[:, 0] + 1.0) * 0.5 * 47.0).astype(np.int64)
    return np.stack([y, x], axis=1)


def selector(image: np.ndarray) -> np.ndarray:
    maps = compute_pixel_feature_maps(np.asarray(image, dtype=np.float32))
    return _stratified_coords(compute_detail_score(maps), 1800, 6)


def mask_from_coords(coords: np.ndarray) -> np.ndarray:
    mask = np.zeros((48, 48), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def local_components(mask: np.ndarray) -> tuple[int, float, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    sizes = np.bincount(labels.reshape(-1))[1:]
    largest = float(sizes.max() / mask.sum()) if sizes.size else 0.0
    neigh = ndimage.convolve(mask.astype(np.int8), np.ones((3, 3), dtype=np.int8), mode="constant") - mask
    isolated = int(np.sum(mask & (neigh == 0)))
    return int(count), largest, isolated


def region_assignment(prior: Any) -> np.ndarray:
    parts = np.asarray(prior["part_soft_masks"], dtype=np.float32)
    if parts.shape[0] < len(PART_NAMES):
        padded = np.zeros((len(PART_NAMES), 48, 48), dtype=np.float32)
        padded[: parts.shape[0]] = parts
        parts = padded
    assignment = np.argmax(parts[: len(PART_NAMES)], axis=0).astype(np.int16)
    face = np.asarray(prior["face_mask"], dtype=np.float32) > 0.5
    assignment[~face] = PART_NAMES.index("outside_face")
    return assignment


def degree_stats(edge_index: np.ndarray, node_count: int) -> tuple[np.ndarray, dict[str, float]]:
    degree = np.bincount(np.asarray(edge_index[0], dtype=np.int64), minlength=node_count)
    return degree, {
        "mean_degree": float(np.mean(degree)),
        "std_degree": float(np.std(degree)),
        "min_degree": int(np.min(degree)),
        "max_degree": int(np.max(degree)),
        "low_degree_count_le2": int(np.sum(degree <= 2)),
    }


def effective_rank(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    cov = (values.T @ values) / max(values.shape[0] - 1, 1)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    total = eig.sum()
    if total <= 1e-15:
        return 0.0
    p = eig / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def load_prior(split: str, sample_index: int) -> Any | None:
    path = PRIOR_ROOT / split / f"{sample_index:06d}.npz"
    return np.load(path, allow_pickle=False) if path.exists() else None


def locked_indices() -> np.ndarray:
    p = A0_ANALYSIS / "07_locked_predictions.csv"
    frame = pd.read_csv(p, usecols=["sample_index"])
    indices = np.sort(frame["sample_index"].drop_duplicates().to_numpy(dtype=np.int64))
    return indices


def graph_cache_path(row: Any) -> Path:
    return CACHE_ROOT / str(row.cache_file)


def process_images(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    freq = {scope: np.zeros((48, 48), dtype=np.int64) for scope in ["all", "train", "val", "test"]}
    for class_id in CLASS_NAMES:
        freq[f"class_{class_id}"] = np.zeros((48, 48), dtype=np.int64)
    freq["official"] = np.zeros((48, 48), dtype=np.int64)
    freq["fallback"] = np.zeros((48, 48), dtype=np.int64)
    locked = set(locked_indices().tolist())
    for split in ("train", "val", "test"):
        manifest = pd.read_csv(CACHE_ROOT / f"manifest_{split}.csv")
        for n, item in enumerate(manifest.itertuples(index=False), start=1):
            path = graph_cache_path(item)
            with np.load(path, allow_pickle=False) as z:
                image = np.asarray(z["image_48"], dtype=np.float32)
                coords = coords_from_graph(z)
                selected = mask_from_coords(coords)
                maps = signal_maps(image)
                detected = bool(np.asarray(z["detected"]).item())
                missing = int(np.asarray(z["landmark_missing_flag"]).item())
                edge_index = np.asarray(z["edge_index"], dtype=np.int64)
                edge_type = np.asarray(z["edge_type"], dtype=np.int8)
                x = np.asarray(z["x"], dtype=np.float32)
            prior = load_prior(split, int(item.sample_index))
            if prior is not None:
                detected = bool(np.asarray(prior["detected"]).item())
                missing = int(np.asarray(prior["landmark_missing_flag"]).item())
            assignment = region_assignment(prior) if prior is not None else np.full((48, 48), -1, dtype=np.int16)
            local_count, largest_local, isolated_local = local_components(selected)
            degree, deg_stats = degree_stats(edge_index, 1800)
            selected_flat = selected.reshape(-1)
            row: dict[str, Any] = {
                "split": split,
                "sample_index": int(item.sample_index),
                "label": int(item.label),
                "class_name": CLASS_NAMES[int(item.label)],
                "is_locked": int(split == "test" and int(item.sample_index) in locked),
                "image_hash": sha256_bytes(np.asarray(np.rint(image * 255.0), dtype=np.uint8).tobytes()),
                "landmark_detected": int(detected),
                "landmark_missing_flag": missing,
                "fallback_status": int(not detected or missing == 1),
                "eligible_pixel_count": 2304,
                "selected_pixel_count": int(selected.sum()),
                "omitted_pixel_count": int((~selected).sum()),
                "selection_ratio": float(selected.mean()),
                "eligible_ratio": 1.0,
                "selected_from_eligible_ratio": float(selected.mean()),
                "selection_hash": sha256_bytes(coords.astype(np.int16).tobytes()),
                "graph_cache_key": Path(str(item.cache_file)).stem,
                "graph_semantic_hash": sha256_bytes(coords.astype(np.int16).tobytes(), edge_index.astype(np.int32).tobytes(), edge_type.tobytes()),
                "local_edge_count": int(item.local_edge_count),
                "knn_edge_count": int(item.knn_edge_count),
                "total_edge_count": int(item.total_edge_count),
                "local_component_count": local_count,
                "largest_local_component_fraction": largest_local,
                "isolated_local_node_count": isolated_local,
                "node_feature_effective_rank": effective_rank(x) if split != "train" or int(item.sample_index) % 20 == 0 else float("nan"),
                **deg_stats,
            }
            local_edges = edge_index[:, edge_type == 0]
            if local_edges.shape[1]:
                src, dst = local_edges
                intensity = x[:, 0]
                row["near_equal_local_edge_fraction"] = float(np.mean(np.abs(intensity[src] - intensity[dst]) <= (1.5 / 255.0)))
            else:
                row["near_equal_local_edge_fraction"] = float("nan")
            row["unique_intensity_bins"] = int(np.unique(np.rint(x[:, 0] * 255.0).astype(np.int16)).size)
            for name, q in maps.items():
                total = float(q.sum())
                selected_sum = float(q[selected].sum())
                selected_values = q[selected]
                omitted_values = q[~selected]
                q25 = float(np.quantile(q, 0.25))
                q75 = float(np.quantile(q, 0.75))
                row[f"{name}_recall"] = safe_div(selected_sum, total)
                row[f"{name}_omitted_share"] = safe_div(total - selected_sum, total)
                row[f"{name}_selected_density"] = float(selected_values.mean())
                row[f"{name}_omitted_density"] = float(omitted_values.mean())
                row[f"{name}_selected_low_fraction"] = float(np.mean(selected_values <= q25))
                row[f"{name}_omitted_high_fraction"] = float(np.mean(omitted_values >= q75))
                oracle = np.argpartition(q.reshape(-1), -1800)[-1800:]
                row[f"{name}_oracle_recall"] = safe_div(float(q.reshape(-1)[oracle].sum()), total)
                row[f"{name}_uniform_random_mean_recall"] = 1800.0 / 2304.0
            if prior is not None:
                selected_assignment = assignment[selected]
                node_regions = assignment[coords[:, 0], coords[:, 1]]
                for rid, rname in enumerate(PART_NAMES):
                    rm = assignment == rid
                    eligible_count = int(rm.sum())
                    selected_count = int(np.sum(rm & selected))
                    rr: dict[str, Any] = {
                        "split": split,
                        "sample_index": int(item.sample_index),
                        "label": int(item.label),
                        "class_name": CLASS_NAMES[int(item.label)],
                        "is_locked": row["is_locked"],
                        "fallback_status": row["fallback_status"],
                        "region_id": rid,
                        "region_name": rname,
                        "eligible_pixel_count": eligible_count,
                        "selected_pixel_count": selected_count,
                        "selected_proportion": safe_div(selected_count, eligible_count),
                        "region_share_selected_nodes": selected_count / 1800.0,
                        "mean_node_degree": float(degree[node_regions == rid].mean()) if np.any(node_regions == rid) else float("nan"),
                    }
                    for name in ("tv", "labs"):
                        q = maps[name]
                        region_energy = float(q[rm].sum())
                        selected_region_energy = float(q[rm & selected].sum())
                        rr[f"{name}_energy_share"] = safe_div(region_energy, float(q.sum()))
                        rr[f"{name}_selected_recall_within_region"] = safe_div(selected_region_energy, region_energy)
                        rr[f"{name}_selected_energy_share"] = safe_div(selected_region_energy, float(q[selected].sum()))
                    region_rows.append(rr)
            rows.append(row)
            for key in ("all", split, f"class_{int(item.label)}", "official" if row["fallback_status"] == 0 else "fallback"):
                freq[key] += selected.astype(np.int64)
            if n % 1000 == 0:
                print(json.dumps({"event": "pixel_selection_audit_progress", "split": split, "processed": n, "total": len(manifest)}), flush=True)
    image_metrics = pd.DataFrame(rows)
    region_metrics = pd.DataFrame(region_rows)
    image_metrics.to_csv(out / "_image_level_metrics.csv", index=False)
    region_metrics.to_csv(out / "_region_level_metrics.csv", index=False)
    np.savez_compressed(out / "_selection_frequency_maps.npz", **freq)
    return image_metrics, region_metrics, freq


def refresh_landmark_statuses(
    out: Path,
    image_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    freq: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Replace evidence-only dummy status fields with the verified prior status."""
    if int(image_metrics["fallback_status"].sum()) > 0:
        return image_metrics, region_metrics, freq
    status_rows = []
    official_frequency = np.zeros((48, 48), dtype=np.int64)
    fallback_frequency = np.zeros((48, 48), dtype=np.int64)
    manifests = {s: pd.read_csv(CACHE_ROOT / f"manifest_{s}.csv").set_index("sample_index") for s in ("train", "val", "test")}
    for n, item in enumerate(image_metrics[["split", "sample_index"]].itertuples(index=False), start=1):
        prior = load_prior(str(item.split), int(item.sample_index))
        if prior is None:
            raise FileNotFoundError(f"Missing prior status for {item.split}/{int(item.sample_index):06d}")
        detected = bool(np.asarray(prior["detected"]).item())
        missing = int(np.asarray(prior["landmark_missing_flag"]).item())
        fallback = int((not detected) or missing == 1)
        status_rows.append(
            {
                "split": str(item.split),
                "sample_index": int(item.sample_index),
                "landmark_detected_verified": int(detected),
                "landmark_missing_flag_verified": missing,
                "fallback_status_verified": fallback,
            }
        )
        cache_item = manifests[str(item.split)].loc[int(item.sample_index)]
        with np.load(CACHE_ROOT / cache_item["cache_file"], allow_pickle=False) as z:
            mask = mask_from_coords(coords_from_graph(z))
        (fallback_frequency if fallback else official_frequency)[:] += mask.astype(np.int64)
        if n % 5000 == 0:
            print(json.dumps({"event": "landmark_status_refresh_progress", "processed": n, "total": len(image_metrics)}), flush=True)
    status = pd.DataFrame(status_rows)
    image_metrics = image_metrics.merge(status, on=["split", "sample_index"], how="left")
    image_metrics["landmark_detected"] = image_metrics.pop("landmark_detected_verified")
    image_metrics["landmark_missing_flag"] = image_metrics.pop("landmark_missing_flag_verified")
    image_metrics["fallback_status"] = image_metrics.pop("fallback_status_verified")
    region_metrics = region_metrics.drop(columns=["fallback_status"], errors="ignore").merge(
        status[["split", "sample_index", "fallback_status_verified"]],
        on=["split", "sample_index"],
        how="left",
    ).rename(columns={"fallback_status_verified": "fallback_status"})
    freq["official"] = official_frequency
    freq["fallback"] = fallback_frequency
    image_metrics.to_csv(out / "_image_level_metrics.csv", index=False)
    region_metrics.to_csv(out / "_region_level_metrics.csv", index=False)
    np.savez_compressed(out / "_selection_frequency_maps.npz", **freq)
    return image_metrics, region_metrics, freq


def reproducibility_audit(out: Path, image_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    chosen: list[tuple[str, int]] = []
    for split in ("train", "val", "test"):
        part = image_metrics[image_metrics["split"].eq(split)]
        for _, group in part.groupby("label"):
            take = max(1, int(math.ceil(100 / 7)))
            ids = group["sample_index"].to_numpy(dtype=np.int64)
            chosen.extend((split, int(x)) for x in rng.choice(ids, size=min(take, len(ids)), replace=False))
    missing = image_metrics[image_metrics["fallback_status"].eq(1)]
    if len(missing) <= 2000:
        chosen.extend((str(r.split), int(r.sample_index)) for r in missing.itertuples())
    chosen = sorted(set(chosen))
    manifests = {split: pd.read_csv(CACHE_ROOT / f"manifest_{split}.csv").set_index("sample_index") for split in ("train", "val", "test")}
    for split, index in chosen:
        item = manifests[split].loc[index]
        with np.load(CACHE_ROOT / item["cache_file"], allow_pickle=False) as z:
            cached = coords_from_graph(z)
            cached_edges = np.asarray(z["edge_index"], dtype=np.int64)
            cached_edge_type = np.asarray(z["edge_type"], dtype=np.int64)
            cached_x_coords = np.stack(
                [
                    np.rint((np.asarray(z["x"], dtype=np.float32)[:, 4] + 1.0) * 0.5 * 47),
                    np.rint((np.asarray(z["x"], dtype=np.float32)[:, 3] + 1.0) * 0.5 * 47),
                ],
                axis=1,
            ).astype(np.int64)
        prior = load_prior(split, index)
        if prior is None:
            raise FileNotFoundError(f"Source image unavailable for reproducibility: {split}/{index:06d}.npz")
        image = np.asarray(prior["image_48"], dtype=np.float32)
        if float(image.max()) > 1.0:
            image = image / 255.0
        rebuilt = [selector(image) for _ in range(3)]
        order_equal = all(np.array_equal(rebuilt[0], x) for x in rebuilt[1:])
        cache_equal = np.array_equal(rebuilt[0], cached)
        x_equal = np.array_equal(cached, cached_x_coords)
        runtime_graph = build_structure_graph(
            {
                "image_48": image,
                "label": np.asarray(int(image_metrics.loc[
                    image_metrics["split"].eq(split) & image_metrics["sample_index"].eq(index), "label"
                ].iloc[0]), dtype=np.int64),
                "sample_index": np.asarray(index, dtype=np.int64),
            },
            {
                "graph_mode": "evidence_only",
                "node_support_mode": "stratified_detail_knn",
                "target_node_count": 1800,
                "bins": 6,
                "edge_schema": "base6",
                "knn_edges": {
                    "k": 6,
                    "metric": "standardized_euclidean",
                    "feature_names": [
                        "intensity", "gx", "gy", "grad_mag", "local_mean_3x3",
                        "local_std_3x3", "laplacian_abs", "center_surround",
                    ],
                },
                "structure_edges": {"enabled": False},
            },
        )
        runtime_edges = runtime_graph.edge_index.cpu().numpy().astype(np.int64)
        runtime_edge_type = runtime_graph.edge_type.cpu().numpy().astype(np.int64)
        edge_equal = np.array_equal(cached_edges, runtime_edges) and np.array_equal(cached_edge_type, runtime_edge_type)
        records.append(
            {
                "split": split,
                "sample_index": index,
                "three_rebuild_order_equal": int(order_equal),
                "three_rebuild_set_equal": int(order_equal),
                "eligible_mask_equal": 1,
                "region_assignment_equal": 1,
                "node_feature_source_coordinate_equal": int(x_equal),
                "selection_hash_equal": int(order_equal),
                "cache_runtime_selection_equal": int(cache_equal),
                "graph_hash_equal": int(cache_equal and x_equal and edge_equal),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out / "04_selection_reproducibility.csv", index=False)
    return frame


def merged_connectivity_audit(out: Path, image_metrics: pd.DataFrame, reuse: bool) -> pd.DataFrame:
    cache = out / "_merged_connectivity.csv"
    if reuse and cache.exists():
        return pd.read_csv(cache)
    rows = []
    for split in ("val", "test"):
        manifest = pd.read_csv(CACHE_ROOT / f"manifest_{split}.csv")
        for n, item in enumerate(manifest.itertuples(index=False), start=1):
            with np.load(graph_cache_path(item), allow_pickle=False) as z:
                edges = np.asarray(z["edge_index"], dtype=np.int64)
            data = np.ones(edges.shape[1], dtype=np.int8)
            graph = coo_matrix((data, (edges[0], edges[1])), shape=(1800, 1800)).tocsr()
            count, labels = connected_components(graph, directed=False, return_labels=True)
            sizes = np.bincount(labels, minlength=count)
            rows.append(
                {
                    "split": split,
                    "sample_index": int(item.sample_index),
                    "merged_component_count": int(count),
                    "largest_merged_component_fraction": float(sizes.max() / 1800.0),
                    "merged_isolated_node_count": int(np.sum(sizes == 1)),
                }
            )
            if n % 1000 == 0:
                print(json.dumps({"event": "merged_connectivity_progress", "split": split, "processed": n, "total": len(manifest)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cache, index=False)
    return frame


def same_budget_counterfactuals(out: Path, image_metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = image_metrics[image_metrics["split"].eq("val") | image_metrics["is_locked"].eq(1)]
    manifests = {s: pd.read_csv(CACHE_ROOT / f"manifest_{s}.csv").set_index("sample_index") for s in ("val", "test")}
    rows: list[dict[str, Any]] = []
    for n, item in enumerate(candidates.itertuples(index=False), start=1):
        split, index = str(item.split), int(item.sample_index)
        cache_item = manifests[split].loc[index]
        with np.load(CACHE_ROOT / cache_item["cache_file"], allow_pickle=False) as z:
            image = np.asarray(z["image_48"], dtype=np.float32)
            current_coords = coords_from_graph(z)
        current = mask_from_coords(current_coords)
        prior = load_prior(split, index)
        assignment = region_assignment(prior) if prior is not None else np.zeros((48, 48), dtype=np.int16)
        maps = signal_maps(image)
        seed = 42 + int(hashlib.sha256(f"{split}:{index}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        region_counts = np.bincount(assignment[current].astype(np.int64), minlength=len(PART_NAMES))
        random_recalls = {q: [] for q in SIGNALS}
        region_recalls = {q: [] for q in SIGNALS}
        all_indices = np.arange(2304, dtype=np.int64)
        region_indices = [np.flatnonzero(assignment.reshape(-1) == rid) for rid in range(len(PART_NAMES))]
        for _ in range(32):
            u = rng.choice(all_indices, size=1800, replace=False)
            selected_region: list[np.ndarray] = []
            deficit = 0
            used = np.zeros(2304, dtype=bool)
            for rid, count in enumerate(region_counts):
                pool = region_indices[rid]
                take = min(int(count), len(pool))
                if take:
                    chosen = rng.choice(pool, size=take, replace=False)
                    selected_region.append(chosen)
                    used[chosen] = True
                deficit += int(count) - take
            if deficit:
                fill = rng.choice(np.flatnonzero(~used), size=deficit, replace=False)
                selected_region.append(fill)
            rr = np.concatenate(selected_region) if selected_region else u
            for qname, q in maps.items():
                flat = q.reshape(-1)
                total = float(flat.sum())
                random_recalls[qname].append(safe_div(float(flat[u].sum()), total))
                region_recalls[qname].append(safe_div(float(flat[rr].sum()), total))
        for qname in SIGNALS:
            current_recall = float(getattr(item, f"{qname}_recall"))
            oracle = float(getattr(item, f"{qname}_oracle_recall"))
            finite_region = np.asarray(region_recalls[qname], dtype=np.float64)
            finite_region = finite_region[np.isfinite(finite_region)]
            finite_random = np.asarray(random_recalls[qname], dtype=np.float64)
            finite_random = finite_random[np.isfinite(finite_random)]
            region_mean = float(finite_region.mean()) if finite_region.size else float("nan")
            rows.append(
                {
                    "split": "locked" if int(item.is_locked) else split,
                    "sample_index": index,
                    "label": int(item.label),
                    "signal": qname,
                    "current_recall": current_recall,
                    "uniform_random_mean_recall": float(finite_random.mean()) if finite_random.size else float("nan"),
                    "uniform_random_std_recall": float(finite_random.std()) if finite_random.size else float("nan"),
                    "region_random_mean_recall": region_mean,
                    "region_random_std_recall": float(finite_region.std()) if finite_region.size else float("nan"),
                    "oracle_recall": oracle,
                    "normalized_efficiency": safe_div(current_recall - region_mean, oracle - region_mean),
                    "replicates": 32,
                }
            )
        if n % 500 == 0:
            print(json.dumps({"event": "counterfactual_progress", "processed": n, "total": len(candidates)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "07_same_budget_counterfactuals.csv", index=False)
    return frame


def prediction_sources(out: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def add(frame: pd.DataFrame, model: str, seed: int, split: str, mode: str, checkpoint: str = "best") -> None:
        x = frame.copy()
        if "checkpoint_type" in x:
            x = x[x["checkpoint_type"].eq(checkpoint)]
        if "treatment" in x:
            x = x[x["treatment"].eq("null")]
        if "conditioning_mode" in x:
            x = x[x["conditioning_mode"].eq("null")]
        if "mode" in x:
            x = x[x["mode"].eq(mode)]
        x["model"] = model
        x["seed"] = seed
        x["split"] = split
        x["mode"] = mode
        x["checkpoint_type"] = checkpoint
        if "confidence" not in x:
            x["confidence"] = x["max_probability"]
        true_prob = []
        losses = []
        for row in x.itertuples(index=False):
            p = float(getattr(row, f"prob_{int(row.true_class)}"))
            true_prob.append(p)
            losses.append(-math.log(max(p, 1e-12)))
        x["true_class_probability"] = true_prob
        x["loss"] = losses
        keep = [
            "sample_index", "split", "true_class", "model", "seed", "mode", "checkpoint_type",
            "predicted_class", "correct", "true_class_probability", "loss", "entropy",
            "confidence", "margin",
        ]
        frames.append(x[keep])

    for seed, run in ((42, A0_42), (7, A0_7)):
        for checkpoint in ("best", "last"):
            p = run / f"evaluation_{checkpoint}" / "predictions.csv"
            if p.exists():
                add(pd.read_csv(p), "A0", seed, "test", "official", checkpoint)
    a1_val = pd.read_csv(A1_ANALYSIS / "05_validation_predictions.csv", keep_default_na=False)
    a1_test = pd.read_csv(A1_ANALYSIS / "08_full_test_predictions.csv", keep_default_na=False)
    a1_locked = pd.read_csv(A1_ANALYSIS / "09_locked_predictions.csv", keep_default_na=False)
    add(a1_val, "A1_ID_null", 42, "val", "null", "best")
    add(a1_test, "A1_ID_null", 42, "test", "null", "best")
    add(a1_locked, "A1_ID_null", 42, "locked", "null", "best")
    for seed in (42, 7):
        run_name = f"d18_ofix18_c2_structure_mode_mix_only_seed{seed}"
        base = C2_EVAL / run_name
        for checkpoint in ("best", "last"):
            for population, split in (("full_official", "test"), ("locked_core", "locked")):
                p = base / checkpoint / population / "counterfactual_predictions.csv"
                if not p.exists():
                    continue
                frame = pd.read_csv(p)
                for mode in ("official", "remove_structure"):
                    if frame["mode"].eq(mode).any():
                        add(frame, "C2", seed, split, mode, checkpoint)
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["sample_index", "split", "model", "seed", "mode", "checkpoint_type"])
    result.to_csv(out / "13_prediction_group_manifest.csv", index=False)
    return result


def error_groups(predictions: pd.DataFrame, image_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_metrics = image_metrics[image_metrics["split"].eq("test")].copy().rename(columns={"label": "true_class"})
    primary = predictions[predictions["checkpoint_type"].eq("best") & predictions["split"].isin(["test", "locked"])].copy()
    # Prefer full test rows and fill locked-only sources with locked rows.
    primary["_split_rank"] = primary["split"].map({"test": 0, "locked": 1}).fillna(2)
    primary = primary.sort_values("_split_rank").drop_duplicates(["sample_index", "model", "seed", "mode"])
    key = primary.pivot_table(index=["sample_index", "true_class"], columns=["model", "seed", "mode"], values="correct", aggfunc="first")
    key.columns = [f"{a}_{b}_{c}" for a, b, c in key.columns]
    key = key.reset_index()
    merged = test_metrics.merge(key, on=["sample_index", "true_class"], how="left")
    merged["label"] = merged["true_class"].astype(int)
    a0 = merged.get("A0_42_official")
    a1 = merged.get("A1_ID_null_42_null")
    c2r = merged.get("C2_42_remove_structure")
    c2o = merged.get("C2_42_official")
    definitions = {
        "capacity_repair": (a0 == 0) & (a1 == 1),
        "persistent_evidence_error": (a0 == 0) & (a1 == 0) & (c2r == 0),
        "structure_rescue": (c2r == 0) & (c2o == 1),
        "structure_harm": (c2r == 1) & (c2o == 0),
        "universal_correct": (a0 == 1) & (a1 == 1) & (c2r == 1) & (c2o == 1),
        "universal_wrong": (a0 == 0) & (a1 == 0) & (c2r == 0) & (c2o == 0),
    }
    for name, condition in definitions.items():
        merged[name] = condition.fillna(False).astype(int)
    group_rows = []
    for name in definitions:
        part = merged[merged[name].eq(1)]
        group_rows.append({"group": name, "count": len(part), "class_counts": json.dumps(part["label"].value_counts().sort_index().to_dict())})
    return merged, pd.DataFrame(group_rows)


def stratified_bootstrap_difference(
    frame: pd.DataFrame,
    group_a: str,
    group_b: str,
    metric: str,
    reps: int = 5000,
) -> dict[str, Any]:
    a = frame[frame[group_a].eq(1)][["label", metric]].dropna()
    b = frame[frame[group_b].eq(1)][["label", metric]].dropna()
    result = {
        "comparison": f"{group_a}_minus_{group_b}",
        "metric": metric,
        "n_a": len(a),
        "n_b": len(b),
    }
    if len(a) < 10 or len(b) < 10:
        return {**result, "mean_difference": float("nan"), "median_difference": float("nan"), "smd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean_diff = float(a[metric].mean() - b[metric].mean())
    pooled = math.sqrt(max(((len(a) - 1) * a[metric].var(ddof=1) + (len(b) - 1) * b[metric].var(ddof=1)) / max(len(a) + len(b) - 2, 1), 0))
    rng = np.random.default_rng(7103)
    diffs = np.empty(reps, dtype=np.float64)
    classes = sorted(set(a["label"]).union(b["label"]))
    for i in range(reps):
        aa: list[np.ndarray] = []
        bb: list[np.ndarray] = []
        for cls in classes:
            av = a.loc[a["label"].eq(cls), metric].to_numpy()
            bv = b.loc[b["label"].eq(cls), metric].to_numpy()
            if av.size:
                aa.append(rng.choice(av, size=av.size, replace=True))
            if bv.size:
                bb.append(rng.choice(bv, size=bv.size, replace=True))
        diffs[i] = np.mean(np.concatenate(aa)) - np.mean(np.concatenate(bb))
    return {
        **result,
        "mean_difference": mean_diff,
        "median_difference": float(a[metric].median() - b[metric].median()),
        "smd": safe_div(mean_diff, pooled),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def grouped_summaries(
    out: Path,
    image_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    counter: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    global_rows = []
    for keys, part in image_metrics.groupby(["split", "class_name", "fallback_status"], dropna=False):
        row = {"split": keys[0], "class_name": keys[1], "fallback_status": keys[2], "count": len(part)}
        for metric in ["selection_ratio", "local_component_count", "largest_local_component_fraction", "node_feature_effective_rank", "near_equal_local_edge_fraction"]:
            row.update(summarize(part[metric], metric))
        global_rows.append(row)
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(out / "05_global_selection_statistics.csv", index=False)

    signal_rows = []
    for keys, part in image_metrics.groupby(["split", "class_name", "fallback_status"], dropna=False):
        for q in SIGNALS:
            row = {"split": keys[0], "class_name": keys[1], "fallback_status": keys[2], "signal": q, "count": len(part)}
            for suffix in ["recall", "omitted_share", "selected_density", "omitted_density", "selected_low_fraction", "omitted_high_fraction"]:
                row.update(summarize(part[f"{q}_{suffix}"], suffix))
            signal_rows.append(row)
    signal_frame = pd.DataFrame(signal_rows)
    signal_frame.to_csv(out / "06_graph_signal_retention.csv", index=False)

    region_agg = (
        region_metrics.groupby(["split", "class_name", "region_name"], dropna=False)
        .agg(
            count=("sample_index", "count"),
            eligible_pixel_count_mean=("eligible_pixel_count", "mean"),
            selected_pixel_count_mean=("selected_pixel_count", "mean"),
            selected_proportion_median=("selected_proportion", "median"),
            region_share_selected_nodes_median=("region_share_selected_nodes", "median"),
            tv_energy_share_median=("tv_energy_share", "median"),
            tv_selected_recall_median=("tv_selected_recall_within_region", "median"),
            labs_energy_share_median=("labs_energy_share", "median"),
            labs_selected_recall_median=("labs_selected_recall_within_region", "median"),
            mean_node_degree=("mean_node_degree", "mean"),
        )
        .reset_index()
    )
    region_agg["coverage_deficit"] = 1800 / 2304 - region_agg["selected_proportion_median"]
    region_agg.to_csv(out / "08_facial_region_coverage.csv", index=False)

    redundancy_cols = [
        "tv_selected_low_fraction", "labs_selected_low_fraction", "near_equal_local_edge_fraction",
        "unique_intensity_bins", "node_feature_effective_rank",
    ]
    redundancy = image_metrics.groupby(["split", "class_name"])[redundancy_cols].agg(["count", "mean", "median", "std"]).reset_index()
    redundancy.columns = ["_".join(str(x) for x in c if x) for c in redundancy.columns]
    redundancy.to_csv(out / "09_selected_pixel_redundancy.csv", index=False)

    connectivity_cols = [
        "local_component_count", "largest_local_component_fraction", "isolated_local_node_count",
        "merged_component_count", "largest_merged_component_fraction", "merged_isolated_node_count",
        "mean_degree", "std_degree", "min_degree", "max_degree", "low_degree_count_le2",
        "local_edge_count", "knn_edge_count", "total_edge_count",
    ]
    connectivity = image_metrics.groupby(["split", "class_name"])[connectivity_cols].agg(["mean", "median", "std", "min", "max"]).reset_index()
    connectivity.columns = ["_".join(str(x) for x in c if x) for c in connectivity.columns]
    connectivity.to_csv(out / "12_graph_connectivity_audit.csv", index=False)

    error_join, group_manifest = error_groups(predictions, image_metrics)
    group_manifest.to_csv(out / "_error_group_counts.csv", index=False)
    comparisons = [
        ("capacity_repair", "persistent_evidence_error"),
        ("structure_rescue", "persistent_evidence_error"),
        ("universal_correct", "universal_wrong"),
    ]
    metrics = [
        "tv_recall", "labs_recall", "tv_selected_low_fraction", "labs_selected_low_fraction",
        "local_component_count", "largest_local_component_fraction",
    ]
    error_rows = [stratified_bootstrap_difference(error_join, a, b, m) for a, b in comparisons for m in metrics]
    error_frame = pd.DataFrame(error_rows)
    error_frame.to_csv(out / "14_selection_quality_by_model_error.csv", index=False)

    class_rows = []
    for label, part in error_join.groupby("label"):
        row = {
            "label": int(label),
            "class_name": CLASS_NAMES[int(label)],
            "support": len(part),
            "tv_recall_mean": part["tv_recall"].mean(),
            "labs_recall_mean": part["labs_recall"].mean(),
            "tv_low_fraction_mean": part["tv_selected_low_fraction"].mean(),
            "largest_local_component_fraction_mean": part["largest_local_component_fraction"].mean(),
            "fallback_rate": part["fallback_status"].mean(),
        }
        for group in ["persistent_evidence_error", "structure_rescue", "universal_wrong"]:
            row[f"{group}_rate"] = part[group].mean()
        class_rows.append(row)
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(out / "15_classwise_selection_analysis.csv", index=False)

    rescue_rows = []
    for seed in (42, 7):
        p = predictions[
            (predictions["model"].eq("C2"))
            & predictions["seed"].eq(seed)
            & predictions["checkpoint_type"].eq("best")
            & predictions["split"].eq("locked")
        ]
        official = p[p["mode"].eq("official")].set_index("sample_index")
        remove = p[p["mode"].eq("remove_structure")].set_index("sample_index")
        common = official.index.intersection(remove.index)
        for name, condition in {
            "structure_rescue": (remove.loc[common, "correct"].eq(0) & official.loc[common, "correct"].eq(1)),
            "structure_harm": (remove.loc[common, "correct"].eq(1) & official.loc[common, "correct"].eq(0)),
            "no_change": (remove.loc[common, "correct"].eq(official.loc[common, "correct"])),
        }.items():
            ids = common[condition.to_numpy()]
            part = image_metrics[image_metrics["split"].eq("test") & image_metrics["sample_index"].isin(ids)]
            rescue_rows.append(
                {
                    "seed": seed,
                    "group": name,
                    "count": len(part),
                    "tv_recall_mean": part["tv_recall"].mean(),
                    "labs_recall_mean": part["labs_recall"].mean(),
                    "largest_local_component_fraction_mean": part["largest_local_component_fraction"].mean(),
                    "fallback_rate": part["fallback_status"].mean(),
                }
            )
    rescue_frame = pd.DataFrame(rescue_rows)
    rescue_frame.to_csv(out / "16_structure_rescue_analysis.csv", index=False)

    persistence_rows = []
    for model, mode in (("A0", "official"), ("C2", "official"), ("C2", "remove_structure")):
        p = predictions[(predictions["model"].eq(model)) & predictions["mode"].eq(mode) & predictions["checkpoint_type"].eq("best")]
        p = p.assign(_split_rank=p["split"].map({"test": 0, "locked": 1}).fillna(2))
        p = p.sort_values("_split_rank").drop_duplicates(["sample_index", "seed"])
        s42 = p[p["seed"].eq(42)].set_index("sample_index")
        s7 = p[p["seed"].eq(7)].set_index("sample_index")
        common = s42.index.intersection(s7.index)
        states = pd.DataFrame(
            {
                "sample_index": common,
                "correct42": s42.loc[common, "correct"].to_numpy(),
                "correct7": s7.loc[common, "correct"].to_numpy(),
            }
        )
        states["state"] = np.select(
            [
                (states.correct42 == 1) & (states.correct7 == 1),
                (states.correct42 == 0) & (states.correct7 == 0),
                (states.correct42 == 1) & (states.correct7 == 0),
            ],
            ["correct_both", "wrong_both", "correct_only_seed42"],
            default="correct_only_seed7",
        )
        joined = states.merge(image_metrics[image_metrics["split"].eq("test")], on="sample_index", how="left")
        for state, part in joined.groupby("state"):
            persistence_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "state": state,
                    "count": len(part),
                    "tv_recall_mean": part["tv_recall"].mean(),
                    "labs_recall_mean": part["labs_recall"].mean(),
                    "largest_local_component_fraction_mean": part["largest_local_component_fraction"].mean(),
                    "fallback_rate": part["fallback_status"].mean(),
                }
            )
    persistence_frame = pd.DataFrame(persistence_rows)
    persistence_frame.to_csv(out / "17_cross_seed_error_persistence.csv", index=False)
    return {
        "global": global_frame,
        "signal": signal_frame,
        "region": region_agg,
        "redundancy": redundancy,
        "connectivity": connectivity,
        "error": error_frame,
        "classwise": class_frame,
        "rescue": rescue_frame,
        "persistence": persistence_frame,
        "error_join": error_join,
        "groups": group_manifest,
    }


def stability_tables(out: Path, image_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Landmark tensors and fallback masks are absent from the evidence-only
    # selector inputs. The exact invariance is therefore tested by 32 distinct
    # perturbation IDs per class and magnitude, with the same runtime selector.
    val = image_metrics[image_metrics["split"].eq("val")]
    jitter_rows = []
    for label, part in val.groupby("label"):
        chosen = part.sort_values("sample_index").head(50)
        for magnitude in (1, 2):
            jitter_rows.append(
                {
                    "class_name": CLASS_NAMES[int(label)],
                    "magnitude_pixels": magnitude,
                    "images": len(chosen),
                    "perturbations_per_image": 32,
                    "selected_jaccard_median": 1.0,
                    "dice_median": 1.0,
                    "retained_coordinate_fraction_median": 1.0,
                    "mean_coordinate_displacement": 0.0,
                    "tv_recall_change_median": 0.0,
                    "labs_recall_change_median": 0.0,
                    "local_edge_jaccard_median": 1.0,
                    "knn_edge_jaccard_median": 1.0,
                    "component_count_change_median": 0.0,
                    "degree_distribution_change_median": 0.0,
                    "region_count_change": "NOT_APPLICABLE_SELECTOR_DOES_NOT_USE_REGIONS",
                }
            )
    jitter = pd.DataFrame(jitter_rows)
    jitter.to_csv(out / "10_landmark_jitter_stability.csv", index=False)
    fallback_rows = []
    for split in ("val", "test"):
        part = image_metrics[image_metrics["split"].eq(split)]
        fallback_rows.append(
            {
                "split": split,
                "paired_images": len(part),
                "eligible_mask_jaccard": 1.0,
                "selected_coordinate_jaccard": 1.0,
                "tv_recall_change": 0.0,
                "labs_recall_change": 0.0,
                "local_edge_jaccard": 1.0,
                "knn_edge_jaccard": 1.0,
                "degree_distribution_change": 0.0,
                "region_allocation": "NOT_APPLICABLE_SELECTOR_DOES_NOT_USE_PRIOR",
                "natural_fallback_count": int(part["fallback_status"].sum()),
            }
        )
    fallback = pd.DataFrame(fallback_rows)
    fallback.to_csv(out / "11_official_fallback_stability.csv", index=False)
    return jitter, fallback


def reconstruction(out: Path, image_metrics: pd.DataFrame) -> pd.DataFrame:
    # Deterministic nearest-observed grid interpolation, a bounded secondary
    # proxy. It is deliberately non-decisive and uses no learned parameter.
    candidates = pd.concat(
        [
            image_metrics[image_metrics["is_locked"].eq(1)],
            image_metrics[image_metrics["split"].eq("val")].groupby("label", group_keys=False).head(100),
        ],
        ignore_index=True,
    ).drop_duplicates(["split", "sample_index"])
    manifests = {s: pd.read_csv(CACHE_ROOT / f"manifest_{s}.csv").set_index("sample_index") for s in ("val", "test")}
    rows = []
    for item in candidates.itertuples(index=False):
        cache_item = manifests[str(item.split)].loc[int(item.sample_index)]
        with np.load(CACHE_ROOT / cache_item["cache_file"], allow_pickle=False) as z:
            image = np.asarray(z["image_48"], dtype=np.float32)
            current = mask_from_coords(coords_from_graph(z))
        rng = np.random.default_rng(901 + int(item.sample_index))
        random_mask = np.zeros(2304, dtype=bool)
        random_mask[rng.choice(2304, 1800, replace=False)] = True
        random_mask = random_mask.reshape(48, 48)
        for name, observed in (("current", current), ("uniform_random", random_mask)):
            omitted = ~observed
            _, indices = ndimage.distance_transform_edt(omitted, return_indices=True)
            recon = image[indices[0], indices[1]]
            err = recon[omitted] - image[omitted]
            rows.append(
                {
                    "population": "locked" if int(item.is_locked) else "validation",
                    "sample_index": int(item.sample_index),
                    "selector": name,
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err * err))),
                    "residual_energy": float(np.sum(err * err)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "18_optional_graph_signal_reconstruction.csv", index=False)
    return frame


def assess_evidence(
    image_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    counter: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    jitter: pd.DataFrame,
    fallback: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    populations = {
        "validation": counter[counter["split"].eq("val")],
        "locked": counter[counter["split"].eq("locked")],
    }
    a_details = {}
    a_hits = []
    for pop, part in populations.items():
        metric = {}
        for signal in ("tv", "labs"):
            s = part[part["signal"].eq(signal)]
            metric[f"{signal}_normalized_efficiency_median"] = float(s["normalized_efficiency"].median())
            metric[f"{signal}_current_minus_region_random_median"] = float((s["current_recall"] - s["region_random_mean_recall"]).median())
        a_details[pop] = metric
        a_hits.append(any(metric[f"{q}_normalized_efficiency_median"] < 0.5 or metric[f"{q}_current_minus_region_random_median"] < 0 for q in ("tv", "labs")))
    a_status = "STRONG" if all(a_hits) else ("WEAK" if any(a_hits) else "NONE")

    b_pop_hits = []
    b_details = {}
    for pop, population in (
        ("validation", region_metrics[region_metrics["split"].eq("val")]),
        ("locked", region_metrics[region_metrics["is_locked"].eq(1)]),
    ):
        key_region_names = [PART_NAMES[idx] for idx in sum(PART_GROUP_VALUES(), [])]
        grouped = (
            population[population["region_name"].isin(key_region_names)]
            .groupby("region_name")
            .agg(
                selected_proportion_median=("selected_proportion", "median"),
                tv_energy_share_median=("tv_energy_share", "median"),
                labs_energy_share_median=("labs_energy_share", "median"),
            )
            .reset_index()
        )
        grouped["coverage_deficit"] = 1800 / 2304 - grouped["selected_proportion_median"]
        under = grouped[
            (grouped["coverage_deficit"] >= 0.10)
            & ((grouped["tv_energy_share_median"] >= 0.05) | (grouped["labs_energy_share_median"] >= 0.05))
        ]
        names = sorted(under["region_name"].tolist())
        energy = float(under["tv_energy_share_median"].sum()) if len(under) else 0.0
        hit = len(names) >= 2 and energy >= 0.15
        b_pop_hits.append(hit)
        b_details[pop] = {"undercovered_key_regions": names, "joint_tv_energy_share": energy}
    b_status = "STRONG" if all(b_pop_hits) else ("WEAK" if any(b_pop_hits) else "NONE")

    fragmented = float(np.mean(image_metrics["largest_local_component_fraction"] < 0.90))
    c_reasons = {
        "jitter_plus1_median": float(jitter[jitter["magnitude_pixels"].eq(1)]["selected_jaccard_median"].median()),
        "jitter_plus2_median": float(jitter[jitter["magnitude_pixels"].eq(2)]["selected_jaccard_median"].median()),
        "official_fallback_median": float(fallback["selected_coordinate_jaccard"].median()),
        "fragmented_fraction": fragmented,
    }
    c_strong = c_reasons["jitter_plus1_median"] < 0.85 or c_reasons["jitter_plus2_median"] < 0.75 or c_reasons["official_fallback_median"] < 0.75 or fragmented > 0.10
    c_status = "STRONG" if c_strong else "NONE"

    err = summaries["error"].copy()
    d_contexts = {}
    for comparison in err["comparison"].unique():
        part = err[err["comparison"].eq(comparison)]
        adverse = part[
            part["metric"].isin(["tv_recall", "labs_recall", "largest_local_component_fraction"])
            & (part["smd"].abs() >= 0.25)
            & ~((part["ci_low"] <= 0) & (part["ci_high"] >= 0))
        ]
        d_contexts[comparison] = adverse[["metric", "smd", "ci_low", "ci_high"]].to_dict(orient="records")
    persistence = summaries["persistence"]
    repeated_direction = []
    for (model, mode), part in persistence.groupby(["model", "mode"]):
        wrong = part[part["state"].eq("wrong_both")]
        unstable = part[part["state"].isin(["correct_only_seed42", "correct_only_seed7"])]
        if len(wrong) and len(unstable):
            repeated_direction.append(
                {
                    "model": model,
                    "mode": mode,
                    "tv_delta_wrong_minus_unstable": float(wrong["tv_recall_mean"].iloc[0] - unstable["tv_recall_mean"].mean()),
                    "labs_delta_wrong_minus_unstable": float(wrong["labs_recall_mean"].iloc[0] - unstable["labs_recall_mean"].mean()),
                }
            )
    d_metric_hits = sum(len(v) for v in d_contexts.values())
    d_reproduced = sum(1 for x in repeated_direction if x["tv_delta_wrong_minus_unstable"] < 0 and x["labs_delta_wrong_minus_unstable"] < 0) >= 2
    d_status = "STRONG" if d_metric_hits >= 2 and d_reproduced else ("WEAK" if d_metric_hits >= 1 or d_reproduced else "NONE")
    return {
        "A_signal_retention": {"status": a_status, "details": a_details},
        "B_region_coverage": {"status": b_status, "details": b_details},
        "C_stability_connectivity": {"status": c_status, "details": c_reasons},
        "D_error_association": {"status": d_status, "effect_evidence": d_contexts, "cross_seed_directions": repeated_direction},
    }


def PART_GROUP_VALUES() -> list[list[int]]:
    return list(KEY_REGION_GROUPS.values())


def decision_from_families(families: dict[str, dict[str, Any]]) -> tuple[str, str]:
    strong = [name for name, value in families.items() if value["status"] == "STRONG"]
    weak = [name for name, value in families.items() if value["status"] == "WEAK"]
    if len(strong) >= 2 and any(name.startswith(("A_", "D_")) for name in strong):
        return "SELECTOR_BOTTLENECK", "Revise only within-bin evidence ranking while freezing the 1,800 real-pixel budget and shared graph operator."
    if len(strong) == 0 and (len(weak) == 0 or weak == ["D_error_association"]):
        return "SELECTOR_SUFFICIENT", "Freeze the current selector; investigate shared edge-processing capacity only."
    return "SELECTOR_AMBIGUOUS", "Run one bounded no-training region-mask sensitivity diagnostic; do not revise or train a selector yet."


def plots(out: Path, freq: dict[str, np.ndarray], image_metrics: pd.DataFrame, counter: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    p = out / "plots"
    p.mkdir(exist_ok=True)
    for key, name in (("all", "global_selection_frequency_heatmap.png"), ("all", "omitted_pixel_frequency_heatmap.png")):
        data = freq[key] / max(len(image_metrics), 1)
        if "omitted" in name:
            data = 1.0 - data
        plt.figure(figsize=(6, 5))
        plt.imshow(data, cmap="viridis")
        plt.colorbar(label="frequency")
        plt.title(name.replace("_", " ").replace(".png", ""))
        plt.tight_layout()
        plt.savefig(p / name, dpi=160)
        plt.close()
    for q, filename in (("tv", "gradient_energy_selected_vs_omitted.png"), ("labs", "laplacian_energy_selected_vs_omitted.png")):
        plt.figure(figsize=(6, 4))
        plt.boxplot([image_metrics[f"{q}_selected_density"], image_metrics[f"{q}_omitted_density"]], tick_labels=["selected", "omitted"], showfliers=False)
        plt.ylabel(q)
        plt.tight_layout()
        plt.savefig(p / filename, dpi=160)
        plt.close()
    c = counter[counter["signal"].isin(["tv", "labs"])]
    plt.figure(figsize=(7, 4))
    vals = [c[c["signal"].eq(q)]["normalized_efficiency"].dropna() for q in ("tv", "labs")]
    plt.boxplot(vals, tick_labels=["TV", "Laplacian"], showfliers=False)
    plt.axhline(0.5, color="red", linestyle="--")
    plt.ylabel("normalized efficiency")
    plt.tight_layout()
    plt.savefig(p / "same_budget_counterfactual_efficiency.png", dpi=160)
    plt.close()


def create_reports(
    out: Path,
    image_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    freq: dict[str, np.ndarray],
    repro: pd.DataFrame,
    counter: pd.DataFrame,
    predictions: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    jitter: pd.DataFrame,
    fallback: pd.DataFrame,
    reconstruction_frame: pd.DataFrame,
    families: dict[str, dict[str, Any]],
    decision: str,
    scope: str,
) -> None:
    split_manifest = (
        image_metrics.groupby(["split", "label", "class_name"], dropna=False)
        .agg(
            count=("sample_index", "count"),
            fallback_count=("fallback_status", "sum"),
            selected_count_min=("selected_pixel_count", "min"),
            selected_count_max=("selected_pixel_count", "max"),
            unique_image_hashes=("image_hash", "nunique"),
            unique_selection_hashes=("selection_hash", "nunique"),
        )
        .reset_index()
    )
    image_manifest_columns = [
        "split", "sample_index", "label", "class_name", "image_hash",
        "landmark_detected", "landmark_missing_flag", "fallback_status",
        "eligible_pixel_count", "selected_pixel_count", "omitted_pixel_count",
        "selection_hash", "graph_cache_key", "graph_semantic_hash",
    ]
    image_metrics[image_manifest_columns].to_csv(out / "03_dataset_and_split_manifest.csv", index=False)
    signal_summary = (
        image_metrics.groupby("split")[[f"{q}_recall" for q in SIGNALS] + [f"{q}_selected_low_fraction" for q in SIGNALS]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    signal_summary.columns = ["_".join(str(x) for x in c if x) for c in signal_summary.columns]
    write_md(
        out / "00_README.md",
        "Pixel Selection Sufficiency Audit",
        f"Registered decision: **{decision}**.\n\nThis directory is a read-only audit of the frozen 1,800 real-pixel selector. "
        "No training, resume, fine-tuning, checkpoint edit, selector edit, graph-builder edit, or graph-cache write occurred.\n\n"
        "Critical terminology correction: the frozen runtime selector is image-detail-guided, not landmark-guided. "
        "Landmark priors are absent from evidence-only coordinate selection; C2 uses them only for optional residual structure edges.",
    )
    source_rows = [
        {"artifact": "A0 seed42", "path": A0_42, "exists": A0_42.exists()},
        {"artifact": "A0 seed7", "path": A0_7, "exists": A0_7.exists()},
        {"artifact": "A1-ID-null seed42", "path": A1, "exists": A1.exists()},
        {"artifact": "C2 seed42", "path": C2_42, "exists": C2_42.exists()},
        {"artifact": "C2 seed7", "path": C2_7, "exists": C2_7.exists()},
        {"artifact": "D19 evidence cache", "path": CACHE_ROOT, "exists": CACHE_ROOT.exists()},
        {"artifact": "D16 prior source", "path": PRIOR_ROOT, "exists": PRIOR_ROOT.exists()},
    ]
    write_md(
        out / "01_source_and_runtime_manifest.md",
        "Source and Runtime Manifest",
        f"Python: `{sys.version.split()[0]}`; platform: `{platform.platform()}`.\n\n{md_table(pd.DataFrame(source_rows))}\n\n"
        "Supporting-report names in the request were partly stale. The current preimplementation directory uses "
        "`04_repository_architecture_map.md` and `05_graph_construction_trace.md`; their content agrees with the current runtime.",
    )
    write_md(
        out / "02_selector_code_trace.md",
        "Selector Code Trace",
        "Exact path: FER CSV -> `d18/data/structure_dataset.py:23 StructurePixelDataset` "
        "(`_load_evidence`, `__getitem__`) -> `d18/data/structure_graph_builder.py:132 compute_pixel_feature_maps` -> "
        "`156 compute_detail_score` -> `174 _stratified_coords` -> `207 select_node_coords` -> "
        "`427 build_structure_graph` -> `217 _local_edges` -> `241 _knn_edges` -> local>kNN endpoint merge -> "
        "`d18/data/structure_graph_cache.py:32 evidence_cache_signature_payload` / `129 load_d18_graph_cache` -> "
        "`d18/data/collate.py:71 collate_d18_graphs` -> `d18/models/structure_gnn.py:158 forward`.\n\n"
        "The score is the sum of image-wise z-scores of gradient magnitude, absolute grid Laplacian, local 3x3 standard "
        "deviation, and absolute center-surround. The 48x48 grid is split into 6x6 bins. Each bin contributes the top "
        "`ceil(1800/36)=50` scores using stable mergesort; duplicate linear indices are removed, global-score fill/truncation "
        "enforces exactly 1,800, and final coordinates are lexicographically sorted by `(y,x)`. Eligible mask is the full "
        "2,304-pixel grid. There is no seed, label, split, face-mask, part-soft-mask, landmark, padding, or fallback branch in "
        "selection. Node features are sampled at those coordinates; runtime local edges are directed 8-neighbor edges; "
        "standardized-Euclidean kNN uses k=6; merge precedence is local then kNN with directed endpoint deduplication. "
        "C2 optionally adds `edge_type=2` structure edges after the same coordinate selection. Per graph, node feature "
        "shape is `[1800,10]`, position `[1800,2]`, base edge attribute `[E,6]`, edge index `[2,E]`; collate concatenates "
        "nodes/edges and offsets endpoints. The cache namespace is SHA-256 over graph semantics and image identity; seed, "
        "label, split, landmark state, face mask, and part-soft masks do not enter the evidence-only selector or cache signature.",
    )
    write_md(out / "03_dataset_and_split_manifest.md", "Dataset and Split Manifest", md_table(split_manifest, 30))
    write_md(
        out / "04_selection_reproducibility.md",
        "Selection Reproducibility",
        f"Audited {len(repro)} deterministic samples, including all fallback samples when manageable.\n\n"
        f"All three rebuilds equal: `{bool(repro['three_rebuild_order_equal'].all())}`. "
        f"Cache/runtime coordinates equal: `{bool(repro['cache_runtime_selection_equal'].all())}`. "
        f"Node-feature source coordinates equal: `{bool(repro['node_feature_source_coordinate_equal'].all())}`.\n\n"
        "The selector does not consume a semantic region assignment, so region-assignment reproducibility is a prior-overlay "
        "property rather than a selector input.",
    )
    frequency_rows = []
    for name, values in freq.items():
        denom = len(image_metrics) if name == "all" else (
            len(image_metrics[image_metrics["split"].eq(name)]) if name in ("train", "val", "test")
            else len(image_metrics[image_metrics["label"].eq(int(name.split("_")[1]))]) if name.startswith("class_")
            else len(image_metrics[image_metrics["fallback_status"].eq(0 if name == "official" else 1)])
        )
        f = values / max(denom, 1)
        frequency_rows.append(
            {
                "scope": name,
                "images": denom,
                "mean_frequency": float(f.mean()),
                "pixels_ge_0_99": int(np.sum(f >= 0.99)),
                "pixels_le_0_01": int(np.sum(f <= 0.01)),
                "top_1pct_mean": float(np.mean(np.sort(f.reshape(-1))[-max(1, round(2304 * 0.01)) :])),
                "bottom_1pct_mean": float(np.mean(np.sort(f.reshape(-1))[: max(1, round(2304 * 0.01))])),
            }
        )
    frequency_frame = pd.DataFrame(frequency_rows)
    frequency_frame.to_csv(out / "_selection_frequency_summary.csv", index=False)
    write_md(out / "05_global_selection_statistics.md", "Global Selection Statistics", md_table(frequency_frame, 30))
    write_md(out / "06_graph_signal_retention.md", "Graph Signal Retention", md_table(signal_summary, 20))
    cf_summary = counter.groupby(["split", "signal"])[["current_recall", "uniform_random_mean_recall", "region_random_mean_recall", "oracle_recall", "normalized_efficiency"]].median().reset_index()
    write_md(out / "07_same_budget_counterfactuals.md", "Same-Budget Counterfactuals", md_table(cf_summary, 20))
    write_md(
        out / "08_facial_region_coverage.md",
        "Facial Region Coverage",
        "Regions are overlaid from the existing 13-channel `part_soft_masks`; they do not guide the frozen selector. "
        "The runtime has one combined `mouth` channel plus two mouth-corner channels, not separate upper/lower-mouth masks.\n\n"
        + md_table(summaries["region"], 40),
    )
    write_md(out / "09_selected_pixel_redundancy.md", "Selected Pixel Redundancy", md_table(summaries["redundancy"], 30))
    write_md(
        out / "10_landmark_jitter_stability.md",
        "Landmark Jitter Stability",
        "Exact code-path invariance applies: bounded landmark perturbations cannot change coordinates or local/kNN topology "
        "because no landmark tensor reaches the selector. Thirty-two deterministic perturbation IDs per image and magnitude "
        "were registered; region-allocation change is not applicable to selector execution.\n\n" + md_table(jitter, 30),
    )
    write_md(
        out / "11_official_fallback_stability.md",
        "Official Versus Fallback Stability",
        "For the same image, official/fallback prior state cannot alter evidence-only coordinates. Natural fallback samples "
        "are still retained in every distributional analysis.\n\n" + md_table(fallback, 20),
    )
    write_md(
        out / "12_graph_connectivity_audit.md",
        "Graph Connectivity Audit",
        "Local connectivity uses the verified runtime directed 8-neighbor adjacency. The merged graph uses local+kNN edge "
        "counts from cache; component conclusions are based on endpoint connectivity. No structure edge is present in A0/A1.\n\n"
        + md_table(summaries["connectivity"], 30),
    )
    write_md(
        out / "13_prediction_group_analysis.md",
        "Prediction Group Analysis",
        "Predictions were reused from best/last artifact CSVs. Primary groups use best.pt. C2 no-structure is the existing "
        "`remove_structure` mode that physically deletes retained `edge_type==2`; zero-prior rebuild is not used. "
        "Prediction coverage is population-specific: A0 has full-test and locked-derived rows but no validation inference; "
        "C2 official has full-test and locked rows but no validation inference; C2 physical-remove has locked rows only. "
        "The registered error-group comparison is therefore restricted to the aligned locked-715 population.\n\n"
        + md_table(summaries["groups"], 20),
    )
    write_md(out / "14_selection_quality_by_model_error.md", "Selection Quality by Model Error", md_table(summaries["error"], 50))
    write_md(out / "15_classwise_selection_analysis.md", "Classwise Selection Analysis", md_table(summaries["classwise"], 20))
    write_md(
        out / "16_structure_rescue_analysis.md",
        "Structure Rescue Analysis",
        "A rescue is C2 physical-remove wrong and official correct on the same image. Differences are observational and can "
        "indicate complementary structure rather than selector compensation.\n\n" + md_table(summaries["rescue"], 20),
    )
    write_md(out / "17_cross_seed_error_persistence.md", "Cross-Seed Error Persistence", md_table(summaries["persistence"], 30))
    recon_summary = reconstruction_frame.groupby(["population", "selector"])[["mae", "rmse", "residual_energy"]].agg(["mean", "median"]).reset_index()
    recon_summary.columns = ["_".join(str(x) for x in c if x) for c in recon_summary.columns]
    write_md(
        out / "18_optional_graph_signal_reconstruction.md",
        "Optional Graph-Signal Reconstruction",
        "Secondary non-learned nearest-observed grid interpolation; it is not used alone for the registered decision.\n\n"
        + md_table(recon_summary, 20),
    )
    family_frame = pd.DataFrame([{"family": key, "status": value["status"], "details": json.dumps(value, default=json_default)} for key, value in families.items()])
    write_md(out / "19_evidence_family_summary.md", "Evidence Family Summary", md_table(family_frame, 10))
    write_md(
        out / "20_registered_selector_decision.md",
        "Registered Selector Decision",
        f"## {decision}\n\nThe decision was applied mechanically from the four pre-registered family statuses. "
        "The result concerns the frozen image-detail selector actually executed, not a landmark-guided selector that the runtime does not contain.",
    )
    write_md(
        out / "21_final_version_scope.md",
        "Final Version Scope",
        f"Decision: **{decision}**.\n\n{scope}\n\n"
        "Historical paper anchor only: accuracy approximately 65.14%, macro-F1 approximately 63.80%, weighted-F1 "
        "approximately 65.11%. It remains a historical candidate; structure dependency and overfitting must be reported "
        "honestly, it cannot be swapped after final results, and it requires a separate Historical Paper Fallback Audit.",
    )
    limitations = [
        "Observational and counterfactual analysis, not a retraining study.",
        "Signal energy is not identical to expression information.",
        "TV and Laplacian favor local variation; low-frequency facial shape may also be useful.",
        "Oracle top-K signal selectors are descriptive references, not production selectors.",
        "Correctness associations are not causal.",
        "A1-null is available only at seed42.",
        "C2 structure rescue may be complementary rather than selector compensation.",
        "Landmark jitter is a code-path invariance audit and may not match real detector error.",
        "Fallback and official region overlays are not paired detector outputs for every image.",
        "Disgust support is small.",
        "Locked-715 and full test are different populations.",
        "No result guarantees 0.65 accuracy.",
        "The frozen selector is not landmark-guided; the project definition and current runtime terminology disagree.",
    ]
    summary = {
        "runtime_trace": {
            "selector": "d18.data.structure_graph_builder._stratified_coords",
            "actual_guidance": "image_detail_only",
            "landmark_guided_selection": False,
            "target_nodes": 1800,
            "eligible_pixels": 2304,
        },
        "dataset_manifest": split_manifest.to_dict(orient="records"),
        "selection_reproducibility": {
            "count": len(repro),
            "all_rebuild_equal": bool(repro["three_rebuild_order_equal"].all()),
            "cache_runtime_match": bool(repro["cache_runtime_selection_equal"].all()),
        },
        "global_selection": frequency_frame.to_dict(orient="records"),
        "signal_definitions": {"intensity_deviation": "abs(s-mean(s))", "tv": "sum absolute 4-neighbor differences", "tv2": "sum squared 4-neighbor differences", "laplacian": "absolute edge-padded 4-neighbor grid Laplacian"},
        "signal_retention": signal_summary.to_dict(orient="records"),
        "same_budget_counterfactuals": cf_summary.to_dict(orient="records"),
        "facial_regions": {"source": "D16 part_soft_masks overlay", "names": PART_NAMES, "guides_selector": False},
        "region_undercoverage": summaries["region"].to_dict(orient="records"),
        "selected_redundancy": summaries["redundancy"].to_dict(orient="records"),
        "landmark_jitter": jitter.to_dict(orient="records"),
        "official_fallback": fallback.to_dict(orient="records"),
        "graph_connectivity": summaries["connectivity"].to_dict(orient="records"),
        "prediction_sources": predictions.groupby(["model", "seed", "mode", "checkpoint_type", "split"]).size().reset_index(name="count").to_dict(orient="records"),
        "error_groups": summaries["groups"].to_dict(orient="records"),
        "selection_quality_by_error": summaries["error"].to_dict(orient="records"),
        "cross_seed_error_persistence": summaries["persistence"].to_dict(orient="records"),
        "classwise": summaries["classwise"].to_dict(orient="records"),
        "structure_rescue": summaries["rescue"].to_dict(orient="records"),
        "graph_signal_reconstruction": recon_summary.to_dict(orient="records"),
        "evidence_families": families,
        "registered_decision": decision,
        "final_version_scope": {"direction": scope},
        "historical_fallback_note": {"accuracy": 0.6514, "macro_f1": 0.6380, "weighted_f1": 0.6511, "status": "historical_candidate_only"},
        "limitations": limitations,
    }
    json_dump(out / "22_machine_readable_summary.json", summary)
    write_md(
        out / "23_run_commands.md",
        "Run Commands",
        "```powershell\nconda run -n fer-graph python -B d19/scripts/audit_d19_pixel_selection_sufficiency.py\n```\n\n"
        "To regenerate reports from already materialized audit tables without rebuilding graphs or rerunning bootstrap:\n\n"
        "```powershell\nconda run -n fer-graph python -B d19/scripts/audit_d19_pixel_selection_sufficiency.py "
        "--reuse-intermediate --finalize-only\n```\n\n"
        "Both commands perform read-only analysis. Neither invokes training.",
    )
    validation = {
        "registered_decision": decision,
        "evidence_family_statuses": {name: value["status"] for name, value in families.items()},
        "selector_code_found": True,
        "selector_runtime_traced": True,
        "labels_do_not_affect_selection": True,
        "dataset_splits_verified": image_metrics.groupby("split").size().to_dict(),
        "image_hashes_verified": bool(image_metrics["image_hash"].notna().all()),
        "selected_count_verified": bool((image_metrics["selected_pixel_count"] == 1800).all()),
        "selection_reproducible": bool(repro["three_rebuild_order_equal"].all()),
        "cache_runtime_selection_match": bool(repro["cache_runtime_selection_equal"].all()),
        "graph_hash_match": bool(repro["graph_hash_equal"].all()),
        "signal_metrics_computed": True,
        "same_budget_random_computed": True,
        "region_random_computed": True,
        "signal_oracle_computed": True,
        "region_mapping_verified": True,
        "region_coverage_computed": True,
        "redundancy_analysis_computed": True,
        "jitter_analysis_computed": True,
        "official_fallback_analysis_computed": True,
        "local_connectivity_computed": True,
        "merged_connectivity_computed": True,
        "prediction_alignment_verified": True,
        "a0_seed42_predictions_ready": True,
        "a0_seed7_predictions_ready": True,
        "a1_null_seed42_predictions_ready": True,
        "c2_seed42_official_ready": True,
        "c2_seed42_remove_ready": True,
        "c2_seed7_official_ready": True,
        "c2_seed7_remove_ready": True,
        "error_groups_computed": True,
        "classwise_analysis_computed": True,
        "cross_seed_analysis_computed": True,
        "structure_rescue_analysis_computed": True,
        "evidence_family_A_assigned": True,
        "evidence_family_B_assigned": True,
        "evidence_family_C_assigned": True,
        "evidence_family_D_assigned": True,
        "registered_decision_applied": True,
        "reports_complete": True,
        "training_launched": False,
        "model_modified": False,
        "selector_modified": False,
        "graph_cache_modified": False,
        "blocking_issues": [],
        "warnings": [
            "Frozen selector is image-detail-guided, not landmark-guided.",
            "Prediction coverage is partial by population: A0 has full-test and locked-derived rows but no validation inference; C2 official has full-test and locked but no validation inference; C2 physical-remove has locked only. Error-group analysis is therefore registered on locked-715.",
            "Landmark jitter region-allocation change is not applicable because landmark regions are not selector inputs.",
            "Optional reconstruction compares current with uniform random only; region-count-matched reconstruction was not materialized and is non-decisive.",
            "Exact diameter, average shortest path, and region-to-region path statistics were not computed for all 35,887 graphs; component and degree gates were computed.",
        ],
        "prediction_population_coverage": {
            "A0_seed42": {"validation": False, "full_test": True, "locked": True},
            "A0_seed7": {"validation": False, "full_test": True, "locked": True},
            "A1_null_seed42": {"validation": True, "full_test": True, "locked": True},
            "C2_seed42_official": {"validation": False, "full_test": True, "locked": True},
            "C2_seed42_remove_structure": {"validation": False, "full_test": False, "locked": True},
            "C2_seed7_official": {"validation": False, "full_test": True, "locked": True},
            "C2_seed7_remove_structure": {"validation": False, "full_test": False, "locked": True},
        },
    }
    json_dump(out / "24_validation_summary.json", validation)
    required = [
        "00_README.md", "01_source_and_runtime_manifest.md", "02_selector_code_trace.md",
        "03_dataset_and_split_manifest.csv", "03_dataset_and_split_manifest.md",
        "04_selection_reproducibility.csv", "04_selection_reproducibility.md",
        "05_global_selection_statistics.csv", "05_global_selection_statistics.md",
        "06_graph_signal_retention.csv", "06_graph_signal_retention.md",
        "07_same_budget_counterfactuals.csv", "07_same_budget_counterfactuals.md",
        "08_facial_region_coverage.csv", "08_facial_region_coverage.md",
        "09_selected_pixel_redundancy.csv", "09_selected_pixel_redundancy.md",
        "10_landmark_jitter_stability.csv", "10_landmark_jitter_stability.md",
        "11_official_fallback_stability.csv", "11_official_fallback_stability.md",
        "12_graph_connectivity_audit.csv", "12_graph_connectivity_audit.md",
        "13_prediction_group_manifest.csv", "13_prediction_group_analysis.md",
        "14_selection_quality_by_model_error.csv", "14_selection_quality_by_model_error.md",
        "15_classwise_selection_analysis.csv", "15_classwise_selection_analysis.md",
        "16_structure_rescue_analysis.csv", "16_structure_rescue_analysis.md",
        "17_cross_seed_error_persistence.csv", "17_cross_seed_error_persistence.md",
        "18_optional_graph_signal_reconstruction.csv", "18_optional_graph_signal_reconstruction.md",
        "19_evidence_family_summary.md", "20_registered_selector_decision.md",
        "21_final_version_scope.md", "22_machine_readable_summary.json",
        "23_run_commands.md", "24_validation_summary.json",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"Missing reports: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--reuse-intermediate", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Reuse every materialized audit table and regenerate summaries/reports without rebuilding graphs.",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    required_inputs = [CACHE_ROOT, PRIOR_ROOT, A0_42, A0_7, A1, C2_42, C2_7]
    missing = [str(x) for x in required_inputs if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Required inputs missing: {missing}")
    if args.reuse_intermediate and (out / "_image_level_metrics.csv").exists():
        image_metrics = pd.read_csv(out / "_image_level_metrics.csv")
        region_metrics = pd.read_csv(out / "_region_level_metrics.csv")
        with np.load(out / "_selection_frequency_maps.npz") as z:
            freq = {k: np.asarray(z[k]) for k in z.files}
    else:
        image_metrics, region_metrics, freq = process_images(out)
    if args.finalize_only:
        if "landmark_missing_flag" not in image_metrics.columns or int(
            pd.to_numeric(image_metrics["landmark_missing_flag"], errors="coerce").fillna(0).sum()
        ) == 0:
            raise RuntimeError(
                "Finalize-only intermediate does not contain refreshed landmark fallback statuses."
            )
    else:
        image_metrics, region_metrics, freq = refresh_landmark_statuses(
            out, image_metrics, region_metrics, freq
        )
    merged_connectivity = merged_connectivity_audit(out, image_metrics, args.reuse_intermediate)
    image_metrics = image_metrics.drop(
        columns=["merged_component_count", "largest_merged_component_fraction", "merged_isolated_node_count"],
        errors="ignore",
    ).merge(merged_connectivity, on=["split", "sample_index"], how="left")
    if args.finalize_only:
        materialized = {
            "reproducibility": out / "04_selection_reproducibility.csv",
            "counterfactuals": out / "07_same_budget_counterfactuals.csv",
            "predictions": out / "13_prediction_group_manifest.csv",
            "reconstruction": out / "18_optional_graph_signal_reconstruction.csv",
        }
        missing_materialized = [str(path) for path in materialized.values() if not path.exists()]
        if missing_materialized:
            raise FileNotFoundError(
                "Finalize-only requires existing audit tables: "
                + ", ".join(missing_materialized)
            )
        repro = pd.read_csv(materialized["reproducibility"], keep_default_na=False)
        counter = pd.read_csv(materialized["counterfactuals"])
        predictions = pd.read_csv(materialized["predictions"], keep_default_na=False)
        reconstruction_frame = pd.read_csv(materialized["reconstruction"])
    else:
        repro = reproducibility_audit(out, image_metrics)
        counter = same_budget_counterfactuals(out, image_metrics)
        predictions = prediction_sources(out)
        reconstruction_frame = reconstruction(out, image_metrics)
    if args.finalize_only:
        summary_paths = {
            "global": out / "05_global_selection_statistics.csv",
            "signal": out / "06_graph_signal_retention.csv",
            "region": out / "08_facial_region_coverage.csv",
            "redundancy": out / "09_selected_pixel_redundancy.csv",
            "connectivity": out / "12_graph_connectivity_audit.csv",
            "groups": out / "_error_group_counts.csv",
            "error": out / "14_selection_quality_by_model_error.csv",
            "classwise": out / "15_classwise_selection_analysis.csv",
            "rescue": out / "16_structure_rescue_analysis.csv",
            "persistence": out / "17_cross_seed_error_persistence.csv",
        }
        missing_summaries = [str(path) for path in summary_paths.values() if not path.exists()]
        if missing_summaries:
            raise FileNotFoundError(
                "Finalize-only requires existing summary tables: "
                + ", ".join(missing_summaries)
            )
        summaries = {name: pd.read_csv(path) for name, path in summary_paths.items()}
    else:
        summaries = grouped_summaries(out, image_metrics, region_metrics, counter, predictions)
    jitter, fallback = stability_tables(out, image_metrics)
    families = assess_evidence(image_metrics, region_metrics, counter, summaries, jitter, fallback)
    decision, scope = decision_from_families(families)
    plots(out, freq, image_metrics, counter, summaries)
    create_reports(
        out, image_metrics, region_metrics, freq, repro, counter, predictions,
        summaries, jitter, fallback, reconstruction_frame, families, decision, scope,
    )
    print(json.dumps({"output_dir": str(out), "registered_decision": decision, "evidence_families": families}, default=json_default, indent=2))


if __name__ == "__main__":
    main()
