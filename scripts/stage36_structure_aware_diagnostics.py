"""Stage 3.6 structure-aware evidence diagnostics and selector addendum.

This is a diagnostic stage only. It does not train D12/D13, does not mutate graph
artifacts, and does not introduce motif/SupCon/global branches. It measures
whether short/local structures are better evidence candidates than smooth
regions or long contours, then evaluates a small structure-aware heuristic grid
with the same lightweight probe used in Stage 2/3.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.labels import EMOTION_NAMES
from stage1_pixel_region_selection import (
    DEFAULT_SELECTORS as STAGE1_SELECTORS,
    EDGE_FEATURES,
    NODE_FEATURES,
    center_scores,
    connected_component_stats,
    edge_to_node_mean,
    ensure_dir,
    feature_index,
    fmt,
    normalize01,
    random_mask,
    resolve_graph_repo,
    safe_name,
    save_png,
    selector_masks,
    selected_mean,
    to_u8,
    topk_mask,
    try_import_slic,
)
from stage2_retention_deletion_test import apply_mask, class_name, make_original_dataset, ratio_name, save_confusion
from stage3_hybrid_evidence_selector import (
    WEIGHT_GRID as STAGE3_WEIGHT_GRID,
    build_hybrid_mask,
    load_stage2_baselines,
    train_eval_fast,
)

DEFAULT_RATIOS = (0.05, 0.10, 0.20, 0.40)
STRUCTURE_SELECTORS = (
    "structure_pixel_score",
    "structure_pixel_smooth",
    "structure_slic_region",
    "structure_hybrid_attention_prior",
)
BASE_VARIANTS = {
    "A_delta_grad": (0.5, 0.5, 0.0),
    "B_balanced": (0.4, 0.4, 0.2),
    "C_contrast_assisted": (0.45, 0.25, 0.30),
}
CURATED_STRUCTURE_WEIGHTS = (
    {"w_short": 0.2, "w_orient": 0.1, "w_smooth": 0.2, "w_long": 0.2, "w_border": 0.1},
    {"w_short": 0.2, "w_orient": 0.1, "w_smooth": 0.2, "w_long": 0.4, "w_border": 0.1},
    {"w_short": 0.2, "w_orient": 0.0, "w_smooth": 0.2, "w_long": 0.4, "w_border": 0.2},
    {"w_short": 0.2, "w_orient": 0.1, "w_smooth": 0.0, "w_long": 0.2, "w_border": 0.2},
    {"w_short": 0.0, "w_orient": 0.1, "w_smooth": 0.2, "w_long": 0.2, "w_border": 0.1},
    {"w_short": 0.2, "w_orient": 0.1, "w_smooth": 0.2, "w_long": 0.0, "w_border": 0.0},
)
REQUESTED_HYBRIDS = (
    "hybrid_pixel_smooth__C_delta_grad__b0p1",
    "hybrid_slic_region__E_balanced__b0p1",
    "hybrid_pixel_score__E_balanced__b0p0",
)
SUMMARY_FIELDS = (
    "selector",
    "variant_id",
    "base_variant",
    "retention_ratio",
    "sample_count",
    "only_selected_accuracy",
    "only_selected_macro_f1",
    "only_selected_weighted_f1",
    "gap_vs_random",
    "gap_vs_center",
    "gap_vs_best_stage3_hybrid",
    "mean_border_ratio",
    "mean_center_ratio",
    "mean_connected_components",
    "mean_largest_component_ratio",
    "mean_selected_smooth_region_ratio",
    "mean_selected_long_contour_ratio",
    "mean_selected_short_structure_ratio",
    "mean_selected_orientation_variation_mean",
    "mean_selected_border_touch_ratio",
    "mean_selected_component_size_mean",
    "mean_selected_component_size_max",
    "mean_selected_component_aspect_mean",
    "mean_selected_good_structure_score_mean",
)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        fields = ["note"]
        rows = [{"note": "NO_ROWS"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def iter_limited(reader: GraphRepositoryReader, split: str, max_samples: Optional[int]) -> Iterable[Any]:
    seen = 0
    for sample in reader.iter_split(split):
        yield sample
        seen += 1
        if max_samples is not None and seen >= int(max_samples):
            break


def local_mean_std(image_2d: np.ndarray, size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_2d, dtype=np.float32)
    try:
        from scipy.ndimage import uniform_filter

        mean = uniform_filter(image, size=int(size), mode="nearest")
        mean_sq = uniform_filter(image * image, size=int(size), mode="nearest")
    except Exception:
        pad = int(size) // 2
        padded = np.pad(image, pad, mode="edge")
        mean = np.zeros_like(image, dtype=np.float32)
        mean_sq = np.zeros_like(image, dtype=np.float32)
        for y in range(size):
            for x in range(size):
                patch = padded[y : y + image.shape[0], x : x + image.shape[1]]
                mean += patch
                mean_sq += patch * patch
        denom = float(size * size)
        mean /= denom
        mean_sq /= denom
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def connected_components_8(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    mask_2d = np.asarray(mask, dtype=bool)
    h, w = mask_2d.shape
    labels = np.zeros((h, w), dtype=np.int32)
    props: List[Dict[str, Any]] = []
    cid = 0
    for y in range(h):
        for x in range(w):
            if not mask_2d[y, x] or labels[y, x] != 0:
                continue
            cid += 1
            q: deque[Tuple[int, int]] = deque([(y, x)])
            labels[y, x] = cid
            ys: List[int] = []
            xs: List[int] = []
            while q:
                cy, cx = q.popleft()
                ys.append(cy)
                xs.append(cx)
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and mask_2d[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cid
                            q.append((ny, nx))
            y0, y1 = int(min(ys)), int(max(ys))
            x0, x1 = int(min(xs)), int(max(xs))
            bh = y1 - y0 + 1
            bw = x1 - x0 + 1
            aspect = float(max(bh, bw) / max(1, min(bh, bw)))
            touch = bool(y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1)
            area = int(len(ys))
            props.append(
                {
                    "component_id": cid,
                    "component_size": area,
                    "bbox_h": bh,
                    "bbox_w": bw,
                    "bbox_aspect_ratio": aspect,
                    "touch_border": int(touch),
                    "component_density": float(area / max(1, bh * bw)),
                }
            )
    return labels, props


def border_penalty_map(x_norm: np.ndarray, y_norm: np.ndarray, margin: float = 0.15) -> np.ndarray:
    x = np.asarray(x_norm, dtype=np.float32)
    y = np.asarray(y_norm, dtype=np.float32)
    dist = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    return np.clip((float(margin) - dist) / max(float(margin), 1e-6), 0.0, 1.0).astype(np.float32)


def orientation_variation_map(gx: np.ndarray, gy: np.ndarray, grad_mag: np.ndarray, window: int = 5) -> np.ndarray:
    gx2 = np.asarray(gx, dtype=np.float32).reshape(48, 48)
    gy2 = np.asarray(gy, dtype=np.float32).reshape(48, 48)
    grad2 = np.asarray(grad_mag, dtype=np.float32).reshape(48, 48)
    orientation = np.arctan2(gy2, gx2)
    sin_map = np.sin(orientation).astype(np.float32)
    cos_map = np.cos(orientation).astype(np.float32)
    _, sin_std = local_mean_std(sin_map, window)
    _, cos_std = local_mean_std(cos_map, window)
    var = normalize01(np.sqrt(sin_std * sin_std + cos_std * cos_std))
    gate = (grad2 >= np.percentile(grad2, 60)).astype(np.float32)
    return (var * gate).astype(np.float32).reshape(-1)


def build_structure_features(record: Dict[str, Any], edge_percentile: float = 85.0) -> Dict[str, Any]:
    grad = normalize01(record["grad_mag"]).reshape(48, 48)
    contrast = normalize01(record["local_contrast"]).reshape(48, 48)
    delta = normalize01(record["delta_edge_node"]).reshape(48, 48)
    intensity = record["intensity"].reshape(48, 48)
    _, local_std = local_mean_std(intensity, 5)
    local_std_n = normalize01(local_std)
    smooth = normalize01((1.0 - grad) * (1.0 - contrast) * (1.0 - local_std_n))
    edge_score = normalize01(0.45 * grad + 0.35 * delta + 0.20 * contrast)
    threshold = float(np.percentile(edge_score, float(edge_percentile)))
    edge_map = edge_score >= threshold
    labels, props = connected_components_8(edge_map)
    component_size_map = np.zeros_like(edge_score, dtype=np.float32)
    component_aspect_map = np.zeros_like(edge_score, dtype=np.float32)
    component_touch_map = np.zeros_like(edge_score, dtype=np.float32)
    component_id_map = labels.astype(np.int32)
    edge_component_rows: List[Dict[str, Any]] = []
    long_map = np.zeros_like(edge_score, dtype=np.float32)
    short_map = np.zeros_like(edge_score, dtype=np.float32)
    for prop in props:
        cid = int(prop["component_id"])
        cmask = labels == cid
        size = int(prop["component_size"])
        aspect = float(prop["bbox_aspect_ratio"])
        touch = bool(prop["touch_border"])
        mean_grad = float(grad[cmask].mean()) if cmask.any() else 0.0
        mean_delta = float(delta[cmask].mean()) if cmask.any() else 0.0
        mean_contrast = float(contrast[cmask].mean()) if cmask.any() else 0.0
        long_score = 0.0
        if size >= 80:
            long_score += 0.45
        if size >= 40 and touch:
            long_score += 0.35
        if aspect >= 4.0 and size >= 20:
            long_score += 0.25
        long_score = float(min(1.0, long_score))
        short_ok = 4 <= size <= 80 and not touch and aspect <= 4.5
        short_score = float((mean_grad + mean_delta + mean_contrast) / 3.0) if short_ok else 0.0
        component_size_map[cmask] = float(size)
        component_aspect_map[cmask] = aspect
        component_touch_map[cmask] = 1.0 if touch else 0.0
        long_map[cmask] = long_score
        short_map[cmask] = short_score
        edge_component_rows.append(
            {
                "split": record["split"],
                "graph_id": int(record["graph_id"]),
                "label": int(record["label"]),
                "class_name": class_name(record["label"]),
                "component_id": cid,
                "component_size": size,
                "component_length_proxy": size,
                "component_bbox_h": int(prop["bbox_h"]),
                "component_bbox_w": int(prop["bbox_w"]),
                "component_bbox_aspect_ratio": aspect,
                "component_touch_border": int(touch),
                "component_density": float(prop["component_density"]),
                "component_mean_grad": mean_grad,
                "component_mean_delta": mean_delta,
                "component_mean_contrast": mean_contrast,
                "long_contour_score": long_score,
                "short_local_structure_score": short_score,
            }
        )
    orient = orientation_variation_map(record["gx"], record["gy"], record["grad_mag"]).reshape(48, 48)
    border = border_penalty_map(record["x_norm"], record["y_norm"]).reshape(48, 48)
    good = normalize01(0.40 * normalize01(short_map) + 0.20 * orient + 0.20 * edge_score - 0.25 * smooth - 0.25 * long_map - 0.15 * border)
    return {
        "smooth_region_map": smooth.reshape(-1).astype(np.float32),
        "edge_score_map": edge_score.reshape(-1).astype(np.float32),
        "edge_binary_map": edge_map.reshape(-1).astype(bool),
        "edge_component_id_map": component_id_map.reshape(-1),
        "component_size_map": component_size_map.reshape(-1),
        "component_aspect_map": component_aspect_map.reshape(-1),
        "component_touch_map": component_touch_map.reshape(-1),
        "long_contour_map": long_map.reshape(-1).astype(np.float32),
        "short_structure_map": normalize01(short_map).reshape(-1).astype(np.float32),
        "orientation_variation_map": orient.reshape(-1).astype(np.float32),
        "border_penalty_map": border.reshape(-1).astype(np.float32),
        "good_structure_score_map": good.reshape(-1).astype(np.float32),
        "edge_component_rows": edge_component_rows,
    }


def load_split_records(reader: GraphRepositoryReader, resolver: GraphResolver, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    height = int(resolver.height)
    width = int(resolver.width)
    num_nodes = height * width
    for sample in iter_limited(reader, split, max_samples):
        resolved = resolver.resolve(sample)
        node_names = list(resolved.node_feature_names or NODE_FEATURES)
        edge_names = list(resolved.edge_feature_names or EDGE_FEATURES)
        node = resolved.node_features.detach().cpu().numpy()
        intensity = node[:, feature_index(node_names, "intensity", 0)].astype(np.float32)
        x_norm = node[:, feature_index(node_names, "x_norm", 1)].astype(np.float32)
        y_norm = node[:, feature_index(node_names, "y_norm", 2)].astype(np.float32)
        gx = node[:, feature_index(node_names, "gx", 3)].astype(np.float32)
        gy = node[:, feature_index(node_names, "gy", 4)].astype(np.float32)
        grad_mag = node[:, feature_index(node_names, "grad_mag", 5)].astype(np.float32)
        local_contrast = node[:, feature_index(node_names, "local_contrast", 6)].astype(np.float32)
        delta_idx = feature_index(edge_names, "delta_intensity", 3)
        delta_edge_node = edge_to_node_mean(resolved.edge_index, resolved.edge_attr[:, delta_idx], num_nodes).astype(np.float32)
        record: Dict[str, Any] = {
            "split": split,
            "graph_id": int(resolved.graph_id),
            "label": int(resolved.label),
            "intensity": intensity,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "gx": gx,
            "gy": gy,
            "grad_mag": grad_mag,
            "local_contrast": local_contrast,
            "delta_edge_node": delta_edge_node,
            "height": height,
            "width": width,
        }
        record["structure"] = build_structure_features(record)
        records.append(record)
    return records


def selected_ratio(mask: np.ndarray, condition: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if int(m.sum()) <= 0:
        return 0.0
    return float(np.logical_and(m, np.asarray(condition).reshape(-1)).sum() / int(m.sum()))


def selected_map_mean(mask: np.ndarray, values: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if int(m.sum()) <= 0:
        return 0.0
    return float(np.asarray(values, dtype=np.float32).reshape(-1)[m].mean())


def mask_structure_stats(
    record: Dict[str, Any],
    selector: str,
    variant_id: str,
    ratio: float,
    mask: np.ndarray,
    region_count: int = 0,
    base_variant: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    st = record["structure"]
    x = record["x_norm"]
    y = record["y_norm"]
    selected = int(mask.sum())
    border_region = (x <= 0.10) | (x >= 0.90) | (y <= 0.10) | (y >= 0.90)
    center_region = (((x - 0.5) / 0.25) ** 2 + ((y - 0.5) / 0.30) ** 2) <= 1.0
    upper = y < 1.0 / 3.0
    middle = (y >= 1.0 / 3.0) & (y < 2.0 / 3.0)
    lower = y >= 2.0 / 3.0
    comp = connected_component_stats(mask.reshape(48, 48))
    comp_sizes = st["component_size_map"][mask]
    comp_aspects = st["component_aspect_map"][mask]
    nonzero_sizes = comp_sizes[comp_sizes > 0]
    nonzero_aspects = comp_aspects[comp_aspects > 0]
    base = {
        "split": record["split"],
        "graph_id": int(record["graph_id"]),
        "label": int(record["label"]),
        "class_name": class_name(record["label"]),
        "selector": selector,
        "variant_id": variant_id,
        "base_variant": base_variant,
        "retention_ratio": float(ratio),
        "selected_pixel_count": selected,
        "border_ratio": selected_ratio(mask, border_region),
        "center_ratio": selected_ratio(mask, center_region),
        "upper_ratio": selected_ratio(mask, upper),
        "middle_ratio": selected_ratio(mask, middle),
        "lower_ratio": selected_ratio(mask, lower),
        "selected_region_count": int(region_count),
        "selected_smooth_region_ratio": selected_ratio(mask, st["smooth_region_map"] >= 0.65),
        "selected_long_contour_ratio": selected_ratio(mask, st["long_contour_map"] >= 0.50),
        "selected_short_structure_ratio": selected_ratio(mask, st["short_structure_map"] >= 0.50),
        "selected_orientation_variation_mean": selected_map_mean(mask, st["orientation_variation_map"]),
        "selected_border_touch_ratio": selected_ratio(mask, st["component_touch_map"] >= 0.50),
        "selected_component_size_mean": float(nonzero_sizes.mean()) if nonzero_sizes.size else 0.0,
        "selected_component_size_max": float(nonzero_sizes.max()) if nonzero_sizes.size else 0.0,
        "selected_component_aspect_mean": float(nonzero_aspects.mean()) if nonzero_aspects.size else 0.0,
        "selected_good_structure_score_mean": selected_map_mean(mask, st["good_structure_score_map"]),
        "mean_selected_intensity": selected_mean(record["intensity"], mask),
        "mean_selected_grad_mag": selected_mean(record["grad_mag"], mask),
        "mean_selected_local_contrast": selected_mean(record["local_contrast"], mask),
        "mean_selected_delta_edge_node": selected_mean(record["delta_edge_node"], mask),
    }
    component = {
        **base,
        "connected_components": comp["connected_components"],
        "largest_component_ratio": comp["largest_component_ratio"],
        "largest_component_size": comp["largest_component_size"],
        "mean_component_size": comp["mean_component_size"],
        "median_component_size": comp["median_component_size"],
        "small_component_count": comp["small_component_count"],
    }
    coord = dict(base)
    return base, component, coord


def aggregate_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    fields = [
        "selected_pixel_count",
        "border_ratio",
        "center_ratio",
        "upper_ratio",
        "middle_ratio",
        "lower_ratio",
        "connected_components",
        "largest_component_ratio",
        "selected_smooth_region_ratio",
        "selected_long_contour_ratio",
        "selected_short_structure_ratio",
        "selected_orientation_variation_mean",
        "selected_border_touch_ratio",
        "selected_component_size_mean",
        "selected_component_size_max",
        "selected_component_aspect_mean",
        "selected_good_structure_score_mean",
        "mean_selected_intensity",
        "mean_selected_grad_mag",
        "mean_selected_local_contrast",
        "mean_selected_delta_edge_node",
    ]
    out: Dict[str, float] = {}
    for field in fields:
        vals = [float(r.get(field, 0.0)) for r in rows]
        out[f"mean_{field}"] = float(np.mean(vals)) if vals else 0.0
    return out


def smooth_score(score: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    score_2d = np.asarray(score, dtype=np.float32).reshape(48, 48)
    try:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(score_2d, sigma=float(sigma))
    except Exception:
        _, out = local_mean_std(score_2d, 3)
    return normalize01(out.reshape(-1)).astype(np.float32)


def attention_prior(score: np.ndarray) -> np.ndarray:
    s = normalize01(score).astype(np.float32)
    logits = (s - float(s.mean())) / max(float(s.std()), 1e-6)
    logits = np.clip(logits, -8.0, 8.0)
    exp = np.exp(logits)
    attn = exp / max(float(exp.sum()), 1e-8)
    return normalize01(attn).astype(np.float32)


def structure_score(record: Dict[str, Any], base_weights: Tuple[float, float, float], cfg: Dict[str, float]) -> np.ndarray:
    st = record["structure"]
    w_delta, w_grad, w_contrast = base_weights
    base = (
        float(w_delta) * normalize01(record["delta_edge_node"])
        + float(w_grad) * normalize01(record["grad_mag"])
        + float(w_contrast) * normalize01(record["local_contrast"])
    )
    score = (
        normalize01(base)
        + float(cfg["w_short"]) * st["short_structure_map"]
        + float(cfg["w_orient"]) * st["orientation_variation_map"]
        - float(cfg["w_smooth"]) * st["smooth_region_map"]
        - float(cfg["w_long"]) * st["long_contour_map"]
        - float(cfg["w_border"]) * st["border_penalty_map"]
    )
    return normalize01(score).astype(np.float32)


def slic_region_structure_mask(
    record: Dict[str, Any],
    score: np.ndarray,
    k: int,
    slic_fn: Any,
    n_segments: int,
    compactness: float,
) -> Tuple[np.ndarray, int, float]:
    if slic_fn is None:
        return topk_mask(score, k), 0, 0.0
    intensity = record["intensity"].reshape(48, 48).astype(np.float32)
    segments = slic_fn(intensity, n_segments=int(n_segments), compactness=float(compactness), start_label=0, channel_axis=None)
    flat = np.asarray(segments).reshape(-1)
    st = record["structure"]
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    rows: List[Tuple[float, int, int]] = []
    sizes: List[int] = []
    for region_id in np.unique(flat):
        rmask = flat == region_id
        area = int(rmask.sum())
        sizes.append(area)
        region_score = (
            float(score[rmask].mean())
            + 0.20 * selected_map_mean(rmask, st["short_structure_map"])
            + 0.10 * selected_map_mean(rmask, st["orientation_variation_map"])
            - 0.20 * selected_map_mean(rmask, st["smooth_region_map"]) * min(1.0, area / 180.0)
            - 0.20 * selected_map_mean(rmask, st["long_contour_map"])
            - 0.10 * selected_map_mean(rmask, st["border_penalty_map"])
        )
        rows.append((region_score, int(region_id), area))
    rows.sort(reverse=True)
    selected = np.zeros_like(flat, dtype=bool)
    region_count = 0
    for _, region_id, _ in rows:
        selected[flat == region_id] = True
        region_count += 1
        if int(selected.sum()) >= int(k):
            break
    return selected, region_count, float(np.mean(sizes)) if sizes else 0.0


def build_structure_mask(
    record: Dict[str, Any],
    selector: str,
    base_weights: Tuple[float, float, float],
    cfg: Dict[str, float],
    ratio: float,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
    smooth_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    score = structure_score(record, base_weights, cfg)
    if selector == "structure_pixel_smooth":
        score = smooth_score(score, smooth_sigma)
    if selector == "structure_hybrid_attention_prior":
        score = attention_prior(score)
    k = max(1, int(round(48 * 48 * float(ratio))))
    if selector in {"structure_pixel_score", "structure_pixel_smooth", "structure_hybrid_attention_prior"}:
        return topk_mask(score, k), score, 0, 0.0
    if selector == "structure_slic_region":
        mask, region_count, mean_region_size = slic_region_structure_mask(record, score, k, slic_fn, slic_segments, slic_compactness)
        return mask, score, region_count, mean_region_size
    raise ValueError(f"Unknown structure selector: {selector}")


def parse_hybrid_variant(variant_id: str) -> Tuple[str, str, Tuple[float, float, float], float]:
    parts = variant_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Invalid hybrid variant: {variant_id}")
    selector = parts[0]
    weight_id = parts[1]
    btxt = parts[2]
    if not btxt.startswith("b"):
        raise ValueError(f"Invalid border suffix: {variant_id}")
    w_border = float(btxt[1:].replace("p", "."))
    return selector, weight_id, STAGE3_WEIGHT_GRID[weight_id], w_border


def load_best_stage3_variants(stage3_dir: Path) -> List[str]:
    path = stage3_dir / "hybrid_selector_summary.csv"
    variants = list(REQUESTED_HYBRIDS)
    if not path.exists():
        return variants
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    slic_rows = [r for r in rows if "slic" in str(r.get("selector", ""))]
    if slic_rows:
        best_slic = max(slic_rows, key=lambda r: float(r.get("only_selected_macro_f1", 0.0)))
        vid = str(best_slic.get("variant_id", ""))
        if vid and vid not in variants:
            variants.append(vid)
    return variants


def load_stage3_best_by_ratio(stage3_dir: Path) -> Dict[float, Dict[str, Any]]:
    path = stage3_dir / "hybrid_selector_summary.csv"
    out: Dict[float, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ratio = float(row["retention_ratio"])
                score = float(row["only_selected_macro_f1"])
            except Exception:
                continue
            if ratio not in out or score > float(out[ratio].get("only_selected_macro_f1", 0.0)):
                out[ratio] = row
    return out


def build_diagnostic_masks(
    record: Dict[str, Any],
    ratio: float,
    slic_fn: Any,
    hybrid_variants: Sequence[str],
    slic_segments: int,
    slic_compactness: float,
    smooth_sigma: float,
    seed: int,
) -> Iterable[Tuple[str, str, np.ndarray, int]]:
    for selector, mask, region_count, _ in selector_masks(
        STAGE1_SELECTORS,
        ratio,
        int(record["graph_id"]),
        seed,
        48,
        48,
        record["intensity"],
        record["x_norm"],
        record["y_norm"],
        record["grad_mag"],
        record["local_contrast"],
        record["delta_edge_node"],
        slic_fn,
        slic_segments,
        slic_compactness,
    ):
        yield selector, selector, mask, region_count
    for variant in hybrid_variants:
        try:
            h_selector, _, weights, w_border = parse_hybrid_variant(variant)
        except Exception:
            continue
        mask, _, region_count = build_hybrid_mask(record, h_selector, weights, w_border, ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        yield h_selector, variant, mask, region_count


def structure_feature_row(record: Dict[str, Any]) -> Dict[str, Any]:
    st = record["structure"]
    edge_rows = st["edge_component_rows"]
    sizes = [float(r["component_size"]) for r in edge_rows]
    long_scores = [float(r["long_contour_score"]) for r in edge_rows]
    short_scores = [float(r["short_local_structure_score"]) for r in edge_rows]
    return {
        "split": record["split"],
        "graph_id": int(record["graph_id"]),
        "label": int(record["label"]),
        "class_name": class_name(record["label"]),
        "smooth_region_mean": float(np.mean(st["smooth_region_map"])),
        "smooth_region_high_ratio": float(np.mean(st["smooth_region_map"] >= 0.65)),
        "edge_pixel_ratio": float(np.mean(st["edge_binary_map"])),
        "edge_component_count": int(len(edge_rows)),
        "edge_component_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "edge_component_size_max": float(np.max(sizes)) if sizes else 0.0,
        "long_contour_pixel_ratio": float(np.mean(st["long_contour_map"] >= 0.50)),
        "short_structure_pixel_ratio": float(np.mean(st["short_structure_map"] >= 0.50)),
        "orientation_variation_mean": float(np.mean(st["orientation_variation_map"])),
        "border_penalty_mean": float(np.mean(st["border_penalty_map"])),
        "good_structure_score_mean": float(np.mean(st["good_structure_score_map"])),
        "long_component_mean": float(np.mean(long_scores)) if long_scores else 0.0,
        "short_component_mean": float(np.mean(short_scores)) if short_scores else 0.0,
    }


def maybe_slic_region_rows(record: Dict[str, Any], slic_fn: Any, n_segments: int, compactness: float) -> List[Dict[str, Any]]:
    if slic_fn is None:
        return []
    segments = slic_fn(record["intensity"].reshape(48, 48).astype(np.float32), n_segments=int(n_segments), compactness=float(compactness), start_label=0, channel_axis=None).reshape(-1)
    st = record["structure"]
    rows: List[Dict[str, Any]] = []
    for region_id in np.unique(segments):
        mask = segments == region_id
        rows.append(
            {
                "split": record["split"],
                "graph_id": int(record["graph_id"]),
                "label": int(record["label"]),
                "class_name": class_name(record["label"]),
                "region_id": int(region_id),
                "region_area": int(mask.sum()),
                "region_mean_grad": selected_map_mean(mask, record["grad_mag"]),
                "region_mean_delta": selected_map_mean(mask, record["delta_edge_node"]),
                "region_mean_contrast": selected_map_mean(mask, record["local_contrast"]),
                "region_smoothness": selected_map_mean(mask, st["smooth_region_map"]),
                "region_long_contour_overlap": selected_map_mean(mask, st["long_contour_map"]),
                "region_short_structure": selected_map_mean(mask, st["short_structure_map"]),
                "region_border_penalty": selected_map_mean(mask, st["border_penalty_map"]),
            }
        )
    return rows


def make_overlay(intensity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = to_u8(intensity.reshape(48, 48))
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    m = mask.reshape(48, 48)
    rgb[m, 0] = 255.0
    rgb[m, 1:] *= 0.45
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_diagnostic_visual(
    output_dir: Path,
    record: Dict[str, Any],
    selector: str,
    variant_id: str,
    ratio: float,
    score: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
    diagnostic: bool = False,
) -> None:
    class_dir = f"class_{int(record['label'])}_{safe_name(class_name(record['label']))}"
    folder = "diagnostics" if diagnostic else selector
    base = ensure_dir(output_dir / "figures" / folder / safe_name(variant_id) / ratio_name(ratio) / class_dir)
    mask_base = ensure_dir(output_dir / "masks" / safe_name(variant_id) / ratio_name(ratio) / class_dir)
    stem = f"{record['split']}_graph_{int(record['graph_id'])}"
    st = record["structure"]
    original = to_u8(record["intensity"].reshape(48, 48))
    parts_gray = [
        original,
        to_u8(normalize01(record["grad_mag"]).reshape(48, 48)),
        to_u8(normalize01(record["delta_edge_node"]).reshape(48, 48)),
        to_u8(normalize01(record["local_contrast"]).reshape(48, 48)),
        to_u8(st["smooth_region_map"].reshape(48, 48)),
        to_u8(st["long_contour_map"].reshape(48, 48)),
        to_u8(st["short_structure_map"].reshape(48, 48)),
        to_u8(st["orientation_variation_map"].reshape(48, 48)),
        to_u8(normalize01(score).reshape(48, 48)),
        mask.reshape(48, 48).astype(np.uint8) * 255,
    ]
    rgb_parts = [np.stack([p, p, p], axis=-1) for p in parts_gray]
    rgb_parts.append(make_overlay(record["intensity"], mask))
    only_selected = to_u8(apply_mask(record["intensity"], mask, "only_selected", fill_mode).reshape(48, 48))
    delete_selected = to_u8(apply_mask(record["intensity"], mask, "delete_selected", fill_mode).reshape(48, 48))
    rgb_parts.extend([np.stack([only_selected] * 3, axis=-1), np.stack([delete_selected] * 3, axis=-1)])
    sep = np.full((48, 4, 3), 255, dtype=np.uint8)
    grid = rgb_parts[0]
    for part in rgb_parts[1:]:
        grid = np.concatenate([grid, sep, part], axis=1)
    save_png(base / f"{stem}_structure_grid.png", grid)
    save_png(mask_base / f"{stem}_mask.png", mask.reshape(48, 48).astype(np.uint8) * 255)


def generate_structure_variants(max_variants: Optional[int]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(CURATED_STRUCTURE_WEIGHTS):
        for base_id, weights in BASE_VARIANTS.items():
            for selector in STRUCTURE_SELECTORS:
                variant_id = (
                    f"{selector}__{base_id}"
                    f"__s{str(cfg['w_short']).replace('.', 'p')}"
                    f"_o{str(cfg['w_orient']).replace('.', 'p')}"
                    f"_sm{str(cfg['w_smooth']).replace('.', 'p')}"
                    f"_l{str(cfg['w_long']).replace('.', 'p')}"
                    f"_b{str(cfg['w_border']).replace('.', 'p')}"
                )
                variants.append(
                    {
                        "selector": selector,
                        "variant_id": variant_id,
                        "base_variant": base_id,
                        "base_weights": weights,
                        "config_id": idx,
                        **cfg,
                    }
                )
    if max_variants is not None and int(max_variants) > 0:
        return variants[: int(max_variants)]
    return variants


def baseline_lookup_from_stage3(stage3_best: Dict[float, Dict[str, Any]], ratio: float) -> Tuple[str, float]:
    row = stage3_best.get(float(ratio))
    if not row:
        return "", float("nan")
    return str(row.get("variant_id", "")), float(row.get("only_selected_macro_f1", "nan"))


def build_risk_register(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    high_long = [r for r in summary_rows if float(r.get("mean_selected_long_contour_ratio", 0.0)) > 0.25]
    high_smooth = [r for r in summary_rows if float(r.get("mean_selected_smooth_region_ratio", 0.0)) > 0.25]
    high_center = [r for r in summary_rows if float(r.get("mean_center_ratio", 0.0)) > 0.35]
    return [
        {
            "risk": "structure_features_are_heuristic",
            "evidence": "Maps are derived from local variance, connected components, and simple orientation variation proxies.",
            "severity": "Medium",
            "mitigation": "Use only as diagnostic/teacher prior; validate with retention, deletion, and visual review.",
        },
        {
            "risk": "useful_contours_may_be_penalized",
            "evidence": f"{len(high_long)} summary rows still have long_contour_ratio > 0.25.",
            "severity": "High",
            "mitigation": "Do not hard-remove long contours; use soft penalty and inspect mouth/eyelid cases.",
        },
        {
            "risk": "smooth_penalty_may_remove_subtle_expression",
            "evidence": f"{len(high_smooth)} summary rows have smooth_region_ratio > 0.25; smooth cheeks/wrinkles can still matter.",
            "severity": "Medium",
            "mitigation": "Keep smooth penalty small and compare against no-smooth ablation.",
        },
        {
            "risk": "slic_quality_dependency",
            "evidence": "SLIC region scores depend on segmentation quality over 48x48 grayscale images.",
            "severity": "Medium",
            "mitigation": "Keep pixel and SLIC teachers in parallel.",
        },
        {
            "risk": "f1_interpretability_tradeoff",
            "evidence": "Structure-aware penalty can reduce F1 while improving contour/smooth diagnostics.",
            "severity": "Medium",
            "mitigation": "Shortlist variants by F1 plus structure metrics, not F1 alone.",
        },
        {
            "risk": "center_shortcut_remains_possible",
            "evidence": f"{len(high_center)} summary rows have center_ratio > 0.35.",
            "severity": "High",
            "mitigation": "Always compare with center_prior and log center ratio in Stage 4.",
        },
    ]


def best_by_ratio(rows: Sequence[Dict[str, Any]], ratios: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for ratio in ratios:
        candidates = [r for r in rows if abs(float(r.get("retention_ratio", -1)) - float(ratio)) < 1e-9]
        if candidates:
            out[float(ratio)] = max(candidates, key=lambda r: float(r.get("only_selected_macro_f1", 0.0)))
    return out


def table_best(rows: Sequence[Dict[str, Any]], ratios: Sequence[float]) -> List[str]:
    best = best_by_ratio(rows, ratios)
    lines = [
        "| Ratio | Selector | Variant | Macro F1 | Gap random | Gap center | Gap best hybrid | Border | Center | Components | Long | Smooth | Short |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ratios:
        row = best.get(float(ratio))
        if not row:
            continue
        lines.append(
            f"| {fmt(ratio)} | {row['selector']} | {row['variant_id']} | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row.get('gap_vs_random'))} | {fmt(row.get('gap_vs_center'))} | {fmt(row.get('gap_vs_best_stage3_hybrid'))} | "
            f"{fmt(row.get('mean_border_ratio'))} | {fmt(row.get('mean_center_ratio'))} | {fmt(row.get('mean_connected_components'))} | "
            f"{fmt(row.get('mean_selected_long_contour_ratio'))} | {fmt(row.get('mean_selected_smooth_region_ratio'))} | "
            f"{fmt(row.get('mean_selected_short_structure_ratio'))} |"
        )
    if len(lines) == 2:
        lines.append("| NO_ROWS | | | | | | | | | | | | |")
    return lines


def write_report(
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    diagnostic_rows: Sequence[Dict[str, Any]],
    stage3_best: Dict[float, Dict[str, Any]],
    ratios: Sequence[float],
    max_variants: Optional[int],
    slic_available: bool,
) -> str:
    best = best_by_ratio(summary_rows, ratios)
    positive = [r for r in summary_rows if float(r.get("gap_vs_random", 0.0)) > 0.01]
    best_any = max(summary_rows, key=lambda r: float(r.get("only_selected_macro_f1", 0.0))) if summary_rows else None
    verdict = "PARTIAL" if summary_rows else "FAIL"
    if positive and best_any and float(best_any.get("gap_vs_best_stage3_hybrid", -1.0)) > 0.0:
        verdict = "PARTIAL"
    diag_by_selector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in diagnostic_rows:
        diag_by_selector[str(row["variant_id"])].append(row)

    def diag_mean(selector: str, field: str) -> float:
        rows = diag_by_selector.get(selector, [])
        vals = [float(r.get(field, 0.0)) for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    lines: List[str] = [
        "# Stage 3.6 Structure-aware Evidence Diagnostics Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Stage 3.6 verdict: **{verdict}**.",
        f"- SLIC available: `{slic_available}`.",
        f"- Structure selector variants evaluated: `{len({r['variant_id'] for r in summary_rows})}`; max_variants=`{max_variants}`.",
    ]
    for ratio, row in best.items():
        lines.append(f"- Best structure-aware @{int(float(ratio) * 100)}%: `{row['variant_id']}` macro_f1={fmt(row['only_selected_macro_f1'])}.")
    if best_any:
        lines.append(f"- Overall best: `{best_any['variant_id']}` @{fmt(best_any['retention_ratio'])}, macro_f1={fmt(best_any['only_selected_macro_f1'])}.")
    lines.extend(
        [
            "- Đây là structure-aware evidence candidate, không phải motif.",
            "- Kết luận phải đọc cùng random/center controls và best Stage 3 hybrid; visualization chỉ là phụ trợ.",
            "",
            "## 2. Motivation",
            "",
            "Stage 3.6 kiểm chứng giả thuyết: không phải mọi edge/gradient/contrast đều có giá trị. Mảng lớn trơn, nền, tóc, hoặc contour quá dài có thể là shortcut. Ngược lại, nét ngắn/vừa, cụm edge local, orientation variation quanh vùng biểu cảm có thể là evidence tốt hơn.",
            "",
            "## 3. Structure Feature Definitions",
            "",
            "- `smooth_region_score`: cao khi `grad_mag`, `local_contrast`, và local std đều thấp.",
            "- `edge_component_map`: connected components trên edge map threshold theo percentile.",
            "- `long_contour_score`: cao với component lớn, dài/thon, hoặc touch border.",
            "- `short_local_structure_score`: cao với component vừa phải, không touch border, có grad/delta/contrast cao.",
            "- `orientation_variation`: proxy bằng local std của sin/cos orientation, chỉ gate ở vùng có gradient.",
            "- `border_touch_penalty`: penalty theo distance-to-border và component touch border.",
            "- SLIC region metrics: area, mean grad/delta/contrast, smoothness, long-contour overlap, short-structure overlap.",
            "",
            "## 4. Diagnostics on Existing Selectors",
            "",
            "| Selector/variant | Smooth | Long contour | Short structure | Border | Center | Components | F1 nếu có |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for selector in [
        "random_pixel",
        "center_prior",
        "gradient_topk",
        "contrast_topk",
        "delta_edge_topk",
        "slic_region_proposal",
        *REQUESTED_HYBRIDS,
        *sorted(k for k in diag_by_selector if k.startswith("hybrid_") and k not in REQUESTED_HYBRIDS),
    ]:
        lines.append(
            f"| {selector} | {fmt(diag_mean(selector, 'selected_smooth_region_ratio'))} | "
            f"{fmt(diag_mean(selector, 'selected_long_contour_ratio'))} | {fmt(diag_mean(selector, 'selected_short_structure_ratio'))} | "
            f"{fmt(diag_mean(selector, 'border_ratio'))} | {fmt(diag_mean(selector, 'center_ratio'))} | "
            f"{fmt(diag_mean(selector, 'connected_components'))} | n/a |"
        )
    lines.extend(
        [
            "",
            "- `gradient_topk` bị nghi ngờ nếu long-contour/border cao hơn `delta_edge_topk` hoặc best hybrid.",
            "- `contrast_topk` bị nghi ngờ nếu short-structure không tăng nhưng component count cao.",
            "- SLIC được xem là region prior nếu giảm fragmentation dù F1 không cao nhất.",
            "",
            "## 5. Structure-aware Selector Results",
            "",
            *table_best(summary_rows, ratios),
            "",
            "## 6. Trade-off Analysis",
            "",
            "- Nếu structure-aware F1 không vượt best Stage 3 nhưng giảm long contour/smooth/fragmentation, nó vẫn có thể là teacher phụ cho Stage 4.",
            "- Nếu F1 tăng cùng center hoặc border tăng, biến thể đó chỉ là suspect shortcut.",
            "- Pixel selectors thường giữ signal tốt hơn nhưng dễ rời rạc; SLIC selectors đáng giữ khi components/largest component tốt hơn.",
            "",
            "## 7. Visual Review",
            "",
            "- Figures nằm trong `figures/structure_pixel_score/`, `figures/structure_pixel_smooth/`, `figures/structure_slic_region/`, `figures/diagnostics/`, và `figures/comparisons/` nếu có.",
            "- Grid gồm original, grad, delta, contrast, smooth, long contour, short structure, orientation, score, mask, overlay, only-selected, delete-selected.",
            "- Không được cherry-pick ảnh đẹp. Cần mở cả high-F1 và high-risk variants để xem tóc/kính/viền/nền còn bị chọn không.",
            "",
            "## 8. Stage 4 Addendum",
            "",
            "Nếu Stage 4 được triển khai, có thể thêm structure-aware prior như teacher phụ:",
            "",
            "- structure-aware teacher score.",
            "- long-contour penalty mềm.",
            "- smooth-region penalty mềm.",
            "- short-structure bonus.",
            "- SLIC region continuity prior.",
            "- attention-prior map từ `structure_hybrid_attention_prior`.",
            "",
            "Stage 4 vẫn chỉ là Learned Evidence Selector v0. Chưa motif. Chưa SupCon. Chưa full D12/D13.",
            "",
            "## 9. Risk Register",
            "",
            "| Risk | Evidence | Severity | Mitigation |",
            "|---|---|---|---|",
            "| structure features quá heuristic | local variance/component/orientation proxy đơn giản | Medium | dùng như diagnostic, không hard claim |",
            "| có thể loại nhầm contour hữu ích | long-contour penalty có thể đụng viền miệng/mí mắt | High | penalty mềm, audit overlay |",
            "| smooth penalty có thể loại nếp nhăn yếu | smooth map cao ở vùng ít gradient | Medium | ablate w_smooth=0 |",
            "| long contour detection sai với lông mày/tóc | component size/aspect chỉ là proxy | Medium | inspect component overlay |",
            "| SLIC phụ thuộc segmentation quality | 48x48 grayscale segmentation thô | Medium | giữ pixel baseline song song |",
            "| F1 có thể giảm dù interpretability tốt hơn | structure penalty đổi distribution mask | Medium | shortlist theo F1 + structure metrics |",
            "| selector vẫn center shortcut | center ratio vẫn được log riêng | High | so với center_prior, hạn chế xy nếu collapse |",
            "",
            "## 10. Final Recommendation",
            "",
        ]
    )
    if best_any and float(best_any.get("gap_vs_random", 0.0)) > 0.01:
        lines.append("Structure-aware prior nên được đưa vào Stage 4 ở mức **conditional teacher/regularizer**, không thay thế toàn bộ hybrid teacher.")
    else:
        lines.append("Structure-aware prior hiện nên giữ làm **diagnostic only** cho đến khi có variant vượt random rõ hơn.")
    if best_any:
        lines.append(f"Teacher candidate chính từ Stage 3.6: `{best_any['variant_id']}` @{fmt(best_any['retention_ratio'])}.")
    lines.extend(
        [
            "Cần rerun top variants với cap lớn hơn nếu muốn dùng làm teacher chính.",
            "Điều kiện bắt đầu Stage 4: giữ random/center controls, log retention/deletion, long/smooth/short metrics, và không claim motif.",
            "",
        ]
    )
    (output_dir / "stage36_structure_aware_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage1_dir", default="outputs/stage1_pixel_region_selection")
    parser.add_argument("--stage2_dir", default="outputs/stage2_retention_deletion_test")
    parser.add_argument("--stage3_dir", default="outputs/stage3_hybrid_evidence_selector")
    parser.add_argument("--output_dir", default="outputs/stage36_structure_aware_diagnostics")
    parser.add_argument("--max_samples_per_split", type=int, default=500)
    parser.add_argument("--probe_train_cap", type=int, default=500)
    parser.add_argument("--probe_eval_cap", type=int, default=300)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--figure_samples_per_class", type=int, default=5)
    parser.add_argument("--enable_slic", type=bool_arg, default=True)
    parser.add_argument("--max_variants", type=int, default=24)
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--edge_percentile", type=float, default=85.0)
    parser.add_argument("--fill_mode", choices=["mean", "zero"], default="mean")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for subdir in (
        "figures/diagnostics",
        "figures/comparisons",
        "figures/structure_pixel_score",
        "figures/structure_pixel_smooth",
        "figures/structure_slic_region",
        "figures/structure_hybrid_attention_prior",
        "masks",
    ):
        ensure_dir(output_dir / subdir)
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    slic_fn, slic_error = try_import_slic()
    if not bool(args.enable_slic):
        slic_fn = None
    if slic_fn is None:
        print(f"[Stage3.6] SLIC unavailable/skipped: {slic_error}")

    train_cap = min(int(args.max_samples_per_split), int(args.probe_train_cap)) if args.max_samples_per_split else int(args.probe_train_cap)
    eval_cap = min(int(args.max_samples_per_split), int(args.probe_eval_cap)) if args.max_samples_per_split else int(args.probe_eval_cap)
    print(f"[Stage3.6] loading records train={train_cap} eval={eval_cap}")
    train_records = load_split_records(reader, resolver, args.train_split, train_cap)
    eval_records = load_split_records(reader, resolver, args.eval_split, eval_cap)
    x_train_orig, y_train, _ = make_original_dataset(train_records)
    x_eval_orig, y_eval, _ = make_original_dataset(eval_records)
    original_metrics, original_cm = train_eval_fast(x_train_orig, y_train, x_eval_orig, y_eval, args.seed, args.classifier_max_iter)
    save_confusion(output_dir / "confusion_matrices" / "original.csv", original_cm)

    feature_rows: List[Dict[str, Any]] = []
    edge_component_rows: List[Dict[str, Any]] = []
    slic_region_rows: List[Dict[str, Any]] = []
    for record in [*train_records, *eval_records]:
        feature_rows.append(structure_feature_row(record))
        edge_component_rows.extend(record["structure"]["edge_component_rows"])
        if record["split"] == args.eval_split:
            slic_region_rows.extend(maybe_slic_region_rows(record, slic_fn, args.slic_segments, args.slic_compactness))

    hybrid_variants = load_best_stage3_variants(Path(args.stage3_dir))
    diagnostic_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, float, int], int] = defaultdict(int)
    print("[Stage3.6] diagnostics on existing selectors")
    for ratio in [float(r) for r in args.retention_ratios]:
        for record in eval_records:
            for selector, variant_id, mask, region_count in build_diagnostic_masks(
                record,
                ratio,
                slic_fn,
                hybrid_variants,
                args.slic_segments,
                args.slic_compactness,
                args.smooth_sigma,
                args.seed,
            ):
                _, component, _ = mask_structure_stats(record, selector, variant_id, ratio, mask, region_count)
                diagnostic_rows.append(component)
                key = (variant_id, ratio, int(record["label"]))
                if visual_counts[key] < int(args.figure_samples_per_class):
                    score = record["structure"]["good_structure_score_map"]
                    save_diagnostic_visual(output_dir, record, selector, variant_id, ratio, score, mask, args.fill_mode, diagnostic=True)
                    visual_counts[key] += 1

    variants = generate_structure_variants(args.max_variants)
    stage2_baselines = load_stage2_baselines(Path(args.stage2_dir))
    stage3_best = load_stage3_best_by_ratio(Path(args.stage3_dir))
    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    print(f"[Stage3.6] evaluating structure variants={len(variants)}")
    for variant in variants:
        selector = str(variant["selector"])
        if "slic" in selector and slic_fn is None:
            continue
        variant_id = str(variant["variant_id"])
        base_variant = str(variant["base_variant"])
        base_weights = variant["base_weights"]
        cfg = {k: float(variant[k]) for k in ("w_short", "w_orient", "w_smooth", "w_long", "w_border")}
        for ratio in [float(r) for r in args.retention_ratios]:
            print(f"[Stage3.6] {variant_id} ratio={ratio}")
            x_train_rows: List[np.ndarray] = []
            for record in train_records:
                mask, _, _, _ = build_structure_mask(
                    record,
                    selector,
                    base_weights,
                    cfg,
                    ratio,
                    slic_fn,
                    args.slic_segments,
                    args.slic_compactness,
                    args.smooth_sigma,
                )
                x_train_rows.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
            x_train = np.stack(x_train_rows, axis=0).astype(np.float32)

            x_eval_rows: List[np.ndarray] = []
            eval_stat_rows: List[Dict[str, Any]] = []
            for record in eval_records:
                mask, score, region_count, _ = build_structure_mask(
                    record,
                    selector,
                    base_weights,
                    cfg,
                    ratio,
                    slic_fn,
                    args.slic_segments,
                    args.slic_compactness,
                    args.smooth_sigma,
                )
                x_eval_rows.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
                _, component, coord = mask_structure_stats(record, selector, variant_id, ratio, mask, region_count, base_variant)
                eval_stat_rows.append(component)
                component_rows.append(component)
                coordinate_rows.append(coord)
                key = (variant_id, ratio, int(record["label"]))
                if visual_counts[key] < int(args.figure_samples_per_class):
                    save_diagnostic_visual(output_dir, record, selector, variant_id, ratio, score, mask, args.fill_mode)
                    visual_counts[key] += 1
            x_eval = np.stack(x_eval_rows, axis=0).astype(np.float32)
            metrics, cm = train_eval_fast(x_train, y_train, x_eval, y_eval, args.seed, args.classifier_max_iter)
            best_stage3_id, best_stage3_f1 = baseline_lookup_from_stage3(stage3_best, ratio)
            random_base = stage2_baselines.get(("random_pixel", ratio), {})
            center_base = stage2_baselines.get(("center_prior", ratio), {})
            random_f1 = float(random_base.get("macro_f1", random_base.get("only_selected_macro_f1", "nan")))
            center_f1 = float(center_base.get("macro_f1", center_base.get("only_selected_macro_f1", "nan")))
            metric_row = {
                "selector": selector,
                "variant_id": variant_id,
                "base_variant": base_variant,
                "retention_ratio": ratio,
                **cfg,
                "original_accuracy": original_metrics["only_selected_accuracy"],
                "original_macro_f1": original_metrics["only_selected_macro_f1"],
                "original_weighted_f1": original_metrics["only_selected_weighted_f1"],
                **metrics,
                "stage2_random_macro_f1": random_f1,
                "stage2_center_macro_f1": center_f1,
                "best_stage3_variant": best_stage3_id,
                "best_stage3_macro_f1": best_stage3_f1,
                "gap_vs_random": float(metrics["only_selected_macro_f1"]) - random_f1 if math.isfinite(random_f1) else float("nan"),
                "gap_vs_center": float(metrics["only_selected_macro_f1"]) - center_f1 if math.isfinite(center_f1) else float("nan"),
                "gap_vs_best_stage3_hybrid": float(metrics["only_selected_macro_f1"]) - best_stage3_f1 if math.isfinite(best_stage3_f1) else float("nan"),
            }
            metric_rows.append(metric_row)
            agg = aggregate_stats(eval_stat_rows)
            summary_rows.append({**metric_row, "sample_count": len(eval_stat_rows), **agg})
            for name in EMOTION_NAMES:
                per_class_rows.append(
                    {
                        "selector": selector,
                        "variant_id": variant_id,
                        "base_variant": base_variant,
                        "retention_ratio": ratio,
                        "class_name": name,
                        "only_selected_f1": metric_row.get(f"per_class_f1_{name}", 0.0),
                    }
                )
            save_confusion(output_dir / "confusion_matrices" / safe_name(variant_id) / f"{ratio_name(ratio)}.csv", cm)

    vs_rows: List[Dict[str, Any]] = []
    for row in metric_rows:
        vs_rows.append(
            {
                "selector": row["selector"],
                "variant_id": row["variant_id"],
                "base_variant": row["base_variant"],
                "retention_ratio": row["retention_ratio"],
                "only_selected_macro_f1": row["only_selected_macro_f1"],
                "stage2_random_macro_f1": row["stage2_random_macro_f1"],
                "stage2_center_macro_f1": row["stage2_center_macro_f1"],
                "best_stage3_variant": row["best_stage3_variant"],
                "best_stage3_macro_f1": row["best_stage3_macro_f1"],
                "gap_vs_random": row["gap_vs_random"],
                "gap_vs_center": row["gap_vs_center"],
                "gap_vs_best_stage3_hybrid": row["gap_vs_best_stage3_hybrid"],
            }
        )

    risk_rows = build_risk_register(summary_rows)
    write_csv(output_dir / "structure_feature_stats.csv", feature_rows)
    write_csv(output_dir / "edge_component_stats.csv", edge_component_rows)
    write_csv(output_dir / "slic_structure_region_stats.csv", slic_region_rows)
    write_csv(output_dir / "structure_selector_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "structure_retention_metrics.csv", metric_rows)
    write_csv(output_dir / "structure_vs_stage3_baselines.csv", vs_rows)
    write_csv(output_dir / "structure_component_stats.csv", component_rows)
    write_csv(output_dir / "structure_coordinate_stats.csv", coordinate_rows)
    write_csv(output_dir / "structure_risk_register.csv", risk_rows)
    write_csv(output_dir / "per_class_structure_retention.csv", per_class_rows)
    write_csv(output_dir / "diagnostic_existing_selector_structure.csv", diagnostic_rows)
    verdict = write_report(
        output_dir,
        summary_rows,
        diagnostic_rows,
        stage3_best,
        [float(r) for r in args.retention_ratios],
        args.max_variants,
        slic_fn is not None,
    )
    best = best_by_ratio(summary_rows, [0.05, 0.10, 0.20])
    best_any = max(summary_rows, key=lambda r: float(r["only_selected_macro_f1"])) if summary_rows else None
    print(f"[Stage3.6] output_dir={output_dir}")
    print(f"[Stage3.6] verdict={verdict}")
    for ratio, row in best.items():
        print(f"[Stage3.6] best_{int(ratio * 100)}={row['variant_id']} macro_f1={fmt(row['only_selected_macro_f1'])}")
    if best_any:
        print(f"[Stage3.6] structure_improves_f1_vs_random={fmt(best_any.get('gap_vs_random'))}")
        print(f"[Stage3.6] gap_vs_best_stage3_hybrid={fmt(best_any.get('gap_vs_best_stage3_hybrid'))}")
        print(
            "[Stage3.6] structure_metrics "
            f"long={fmt(best_any.get('mean_selected_long_contour_ratio'))} "
            f"smooth={fmt(best_any.get('mean_selected_smooth_region_ratio'))} "
            f"short={fmt(best_any.get('mean_selected_short_structure_ratio'))} "
            f"components={fmt(best_any.get('mean_connected_components'))}"
        )
    print("[Stage3.6] stage4_structure_teacher=conditional")


if __name__ == "__main__":
    main()
