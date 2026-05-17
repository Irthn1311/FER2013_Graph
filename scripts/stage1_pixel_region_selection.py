"""Stage 1 controlled pixel/region selection baselines for FER full graphs.

This script is intentionally read-only for graph artifacts and model code. It
loads existing graph chunks, builds deterministic selector masks, writes
selection statistics, and saves visual evidence for a small sample set.
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

SPLITS = ("train", "val", "test")
DEFAULT_SELECTORS = (
    "random_pixel",
    "center_prior",
    "gradient_topk",
    "contrast_topk",
    "delta_edge_topk",
    "slic_region_proposal",
)
NODE_FEATURES = (
    "intensity",
    "x_norm",
    "y_norm",
    "gx",
    "gy",
    "grad_mag",
    "local_contrast",
)
EDGE_FEATURES = (
    "dx",
    "dy",
    "dist",
    "delta_intensity",
    "intensity_similarity",
)
RETENTION_FIELDS = (
    "split",
    "graph_id",
    "label",
    "class_name",
    "selector",
    "retention_ratio_target",
    "retention_ratio_actual",
    "selected_pixel_count",
    "border_ratio",
    "center_ratio",
    "upper_ratio",
    "middle_ratio",
    "lower_ratio",
    "left_ratio",
    "x_center_ratio",
    "right_ratio",
    "connected_components",
    "largest_component_ratio",
    "mean_component_size",
    "median_component_size",
    "selected_region_count",
    "mean_selected_intensity",
    "mean_selected_grad_mag",
    "mean_selected_local_contrast",
    "mean_selected_delta_edge_node",
)
COMPONENT_FIELDS = (
    "split",
    "graph_id",
    "label",
    "class_name",
    "selector",
    "retention_ratio_target",
    "selected_pixel_count",
    "connected_components",
    "largest_component_size",
    "largest_component_ratio",
    "mean_component_size",
    "median_component_size",
    "small_component_count",
    "selected_region_count",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def class_name(label: int) -> str:
    label = int(label)
    return EMOTION_NAMES[label] if 0 <= label < len(EMOTION_NAMES) else f"label_{label}"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


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


class CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        ensure_dir(path.parent)
        self.path = path
        self.fields = list(fields)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow(row)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class MeanAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def add(self, row: Dict[str, Any], fields: Sequence[str]) -> None:
        self.count += 1
        for field in fields:
            try:
                value = float(row.get(field, 0.0))
            except Exception:
                value = 0.0
            if math.isfinite(value):
                self.sums[field] += value

    def row(self, prefix: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
        out = dict(prefix)
        out["sample_count"] = int(self.count)
        for field in fields:
            out[f"mean_{field}"] = self.sums.get(field, 0.0) / max(1, self.count)
        return out


def resolve_graph_repo(path_arg: Optional[str]) -> Path:
    candidates: List[Path] = []
    if path_arg:
        candidates.append(Path(path_arg))
    candidates.extend(
        [
            PROJECT_ROOT / "artifacts" / "graph-repo" / "graph_repo",
            PROJECT_ROOT / "artifacts" / "graph_repo",
            PROJECT_ROOT / "artifacts" / "graph_repo_local",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No graph repo found. Checked: {[str(c) for c in candidates]}")


def iter_limited(reader: GraphRepositoryReader, split: str, max_samples: Optional[int]) -> Iterable[Any]:
    seen = 0
    for sample in reader.iter_split(split):
        yield sample
        seen += 1
        if max_samples is not None and seen >= int(max_samples):
            break


def feature_index(names: Sequence[str], name: str, fallback: int) -> int:
    return int(names.index(name)) if name in names else int(fallback)


def edge_to_node_mean(edge_index: torch.Tensor, edge_values: torch.Tensor, num_nodes: int) -> np.ndarray:
    edge_index = edge_index.detach().cpu().long()
    values = edge_values.detach().cpu().float()
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros((num_nodes,), dtype=torch.float32)
    cnt = torch.zeros((num_nodes,), dtype=torch.float32)
    out.index_add_(0, src, values)
    out.index_add_(0, dst, values)
    one = torch.ones_like(values)
    cnt.index_add_(0, src, one)
    cnt.index_add_(0, dst, one)
    return (out / cnt.clamp_min(1.0)).numpy()


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = int(scores.size)
    k = max(0, min(int(k), n))
    mask = np.zeros((n,), dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    idx = np.argpartition(scores, n - k)[n - k :]
    mask[idx] = True
    return mask


def random_mask(num_nodes: int, k: int, seed: int) -> np.ndarray:
    k = max(0, min(int(k), int(num_nodes)))
    mask = np.zeros((num_nodes,), dtype=bool)
    if k <= 0:
        return mask
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(num_nodes, size=k, replace=False)
    mask[idx] = True
    return mask


def center_scores(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    dx = (x_norm - 0.5) / 0.28
    dy = (y_norm - 0.5) / 0.34
    dist2 = dx * dx + dy * dy
    return np.exp(-0.5 * dist2).astype(np.float32)


def connected_component_stats(mask_2d: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask_2d, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: List[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q: deque[Tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            sizes.append(size)
    selected = int(mask.sum())
    largest = int(max(sizes) if sizes else 0)
    return {
        "connected_components": int(len(sizes)),
        "largest_component_size": largest,
        "largest_component_ratio": float(largest / selected) if selected else 0.0,
        "mean_component_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_component_size": float(np.median(sizes)) if sizes else 0.0,
        "small_component_count": int(sum(1 for size in sizes if size <= 3)),
    }


def ratio_in(mask: np.ndarray, region: np.ndarray) -> float:
    selected = int(mask.sum())
    if selected <= 0:
        return 0.0
    return float(np.logical_and(mask, region).sum() / selected)


def selected_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 0:
        return 0.0
    return float(np.asarray(values, dtype=np.float32)[mask].mean())


def mask_metrics(
    *,
    split: str,
    graph_id: int,
    label: int,
    selector: str,
    ratio_target: float,
    mask: np.ndarray,
    height: int,
    width: int,
    intensity: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    grad_mag: np.ndarray,
    local_contrast: np.ndarray,
    delta_edge_node: np.ndarray,
    selected_region_count: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    num_nodes = int(mask.size)
    selected = int(mask.sum())
    border_region = (x_norm <= 0.10) | (x_norm >= 0.90) | (y_norm <= 0.10) | (y_norm >= 0.90)
    center_region = (((x_norm - 0.5) / 0.25) ** 2 + ((y_norm - 0.5) / 0.30) ** 2) <= 1.0
    upper = y_norm < (1.0 / 3.0)
    middle = (y_norm >= (1.0 / 3.0)) & (y_norm < (2.0 / 3.0))
    lower = y_norm >= (2.0 / 3.0)
    left = x_norm < (1.0 / 3.0)
    x_center = (x_norm >= (1.0 / 3.0)) & (x_norm < (2.0 / 3.0))
    right = x_norm >= (2.0 / 3.0)
    component = connected_component_stats(mask.reshape(height, width))
    base = {
        "split": split,
        "graph_id": int(graph_id),
        "label": int(label),
        "class_name": class_name(label),
        "selector": selector,
        "retention_ratio_target": float(ratio_target),
        "retention_ratio_actual": float(selected / max(1, num_nodes)),
        "selected_pixel_count": selected,
        "border_ratio": ratio_in(mask, border_region),
        "center_ratio": ratio_in(mask, center_region),
        "upper_ratio": ratio_in(mask, upper),
        "middle_ratio": ratio_in(mask, middle),
        "lower_ratio": ratio_in(mask, lower),
        "left_ratio": ratio_in(mask, left),
        "x_center_ratio": ratio_in(mask, x_center),
        "right_ratio": ratio_in(mask, right),
        "connected_components": component["connected_components"],
        "largest_component_ratio": component["largest_component_ratio"],
        "mean_component_size": component["mean_component_size"],
        "median_component_size": component["median_component_size"],
        "selected_region_count": int(selected_region_count),
        "mean_selected_intensity": selected_mean(intensity, mask),
        "mean_selected_grad_mag": selected_mean(grad_mag, mask),
        "mean_selected_local_contrast": selected_mean(local_contrast, mask),
        "mean_selected_delta_edge_node": selected_mean(delta_edge_node, mask),
    }
    component_row = {
        "split": split,
        "graph_id": int(graph_id),
        "label": int(label),
        "class_name": class_name(label),
        "selector": selector,
        "retention_ratio_target": float(ratio_target),
        "selected_pixel_count": selected,
        "connected_components": component["connected_components"],
        "largest_component_size": component["largest_component_size"],
        "largest_component_ratio": component["largest_component_ratio"],
        "mean_component_size": component["mean_component_size"],
        "median_component_size": component["median_component_size"],
        "small_component_count": component["small_component_count"],
        "selected_region_count": int(selected_region_count),
    }
    return base, component_row


def try_import_slic() -> Tuple[Any, Optional[str]]:
    try:
        from skimage.segmentation import slic

        return slic, None
    except Exception as exc:
        return None, str(exc)


def slic_mask(
    intensity_2d: np.ndarray,
    grad_2d: np.ndarray,
    contrast_2d: np.ndarray,
    delta_2d: np.ndarray,
    k: int,
    slic_fn: Any,
    n_segments: int,
    compactness: float,
) -> Tuple[np.ndarray, int, float]:
    segments = slic_fn(
        intensity_2d.astype(np.float32),
        n_segments=int(n_segments),
        compactness=float(compactness),
        start_label=0,
        channel_axis=None,
    )
    flat_segments = np.asarray(segments).reshape(-1)
    score_map = (
        normalize01(grad_2d.reshape(-1))
        + normalize01(contrast_2d.reshape(-1))
        + normalize01(delta_2d.reshape(-1))
    ) / 3.0
    region_scores: List[Tuple[float, int, int]] = []
    for region_id in np.unique(flat_segments):
        region_mask = flat_segments == region_id
        region_scores.append((float(score_map[region_mask].mean()), int(region_id), int(region_mask.sum())))
    region_scores.sort(reverse=True)
    selected = np.zeros_like(flat_segments, dtype=bool)
    selected_regions = 0
    for _, region_id, _ in region_scores:
        selected[flat_segments == region_id] = True
        selected_regions += 1
        if int(selected.sum()) >= int(k):
            break
    sizes = [size for _, _, size in region_scores]
    return selected, selected_regions, float(np.mean(sizes)) if sizes else 0.0


def to_u8(image: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    ensure_dir(path.parent)
    Image.fromarray(image).save(path)


def make_overlay(intensity_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    base = to_u8(intensity_2d)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    mask = mask_2d[..., None].astype(np.float32)
    out = rgb * (1.0 - 0.55 * mask) + red * (0.55 * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_grid(original: np.ndarray, mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    original_rgb = np.stack([original, original, original], axis=-1)
    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    sep = np.full((original.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([original_rgb, sep, mask_rgb, sep, overlay], axis=1)


def save_visuals(
    output_dir: Path,
    selector: str,
    ratio: float,
    label: int,
    graph_id: int,
    split: str,
    intensity_2d: np.ndarray,
    mask_2d: np.ndarray,
) -> None:
    ratio_name = f"ratio_{int(round(float(ratio) * 100)):02d}"
    class_dir = f"class_{int(label)}_{safe_name(class_name(label))}"
    base = ensure_dir(output_dir / "figures" / selector / ratio_name / class_dir)
    stem = f"{split}_graph_{int(graph_id)}"
    original = to_u8(intensity_2d)
    mask_u8 = (mask_2d.astype(np.uint8) * 255)
    overlay = make_overlay(intensity_2d, mask_2d)
    only_selected = original.copy()
    only_selected[~mask_2d] = 0
    delete_selected = original.copy()
    delete_selected[mask_2d] = int(round(float(original.mean())))
    save_png(base / f"{stem}_original.png", original)
    save_png(base / f"{stem}_mask.png", mask_u8)
    save_png(base / f"{stem}_overlay.png", overlay)
    save_png(base / f"{stem}_comparison.png", make_grid(original, mask_u8, overlay))
    save_png(base / f"{stem}_only_selected.png", only_selected)
    save_png(base / f"{stem}_delete_selected.png", delete_selected)


def update_histograms(
    hist: Dict[Tuple[str, float, str, str, int], int],
    selector: str,
    ratio: float,
    class_label: str,
    mask: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    bins: int = 10,
) -> None:
    if int(mask.sum()) <= 0:
        return
    for axis, values in (("x", x_norm), ("y", y_norm)):
        idx = np.floor(np.clip(values[mask], 0.0, 0.999999) * bins).astype(int)
        counts = np.bincount(idx, minlength=bins)
        for bin_index, count in enumerate(counts):
            hist[(selector, float(ratio), class_label, axis, int(bin_index))] += int(count)


def choose_visual_sample(
    visual_counts: Dict[int, int],
    label: int,
    figure_samples_per_class: int,
) -> bool:
    if visual_counts[int(label)] < int(figure_samples_per_class):
        visual_counts[int(label)] += 1
        return True
    return False


def selector_masks(
    selectors: Sequence[str],
    ratio: float,
    graph_id: int,
    seed: int,
    height: int,
    width: int,
    intensity: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    grad_mag: np.ndarray,
    local_contrast: np.ndarray,
    delta_edge_node: np.ndarray,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
) -> Iterable[Tuple[str, np.ndarray, int, Dict[str, Any]]]:
    num_nodes = int(intensity.size)
    k = max(1, int(round(num_nodes * float(ratio))))
    for selector in selectors:
        extra: Dict[str, Any] = {}
        if selector == "random_pixel":
            mask = random_mask(num_nodes, k, seed + int(graph_id) * 1009 + int(round(ratio * 10000)))
            yield selector, mask, 0, extra
        elif selector == "center_prior":
            yield selector, topk_mask(center_scores(x_norm, y_norm), k), 0, extra
        elif selector == "gradient_topk":
            yield selector, topk_mask(grad_mag, k), 0, extra
        elif selector == "contrast_topk":
            yield selector, topk_mask(local_contrast, k), 0, extra
        elif selector == "delta_edge_topk":
            yield selector, topk_mask(delta_edge_node, k), 0, extra
        elif selector == "slic_region_proposal":
            if slic_fn is None:
                continue
            mask, region_count, mean_region_size = slic_mask(
                intensity.reshape(height, width),
                grad_mag.reshape(height, width),
                local_contrast.reshape(height, width),
                delta_edge_node.reshape(height, width),
                k,
                slic_fn,
                slic_segments,
                slic_compactness,
            )
            extra["mean_slic_region_size"] = mean_region_size
            yield selector, mask, region_count, extra
        else:
            raise ValueError(f"Unknown selector: {selector}")


def build_hist_rows(hist: Dict[Tuple[str, float, str, str, int], int]) -> List[Dict[str, Any]]:
    totals: Dict[Tuple[str, float, str, str], int] = defaultdict(int)
    for selector, ratio, class_label, axis, bin_index in hist:
        totals[(selector, ratio, class_label, axis)] += hist[(selector, ratio, class_label, axis, bin_index)]
    rows: List[Dict[str, Any]] = []
    for key, count in sorted(hist.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        selector, ratio, class_label, axis, bin_index = key
        total = totals[(selector, ratio, class_label, axis)]
        rows.append(
            {
                "selector": selector,
                "retention_ratio_target": ratio,
                "class_name": class_label,
                "axis": axis,
                "bin_index": bin_index,
                "bin_start": bin_index / 10.0,
                "bin_end": (bin_index + 1) / 10.0,
                "selected_count": count,
                "total_selected": total,
                "fraction": float(count / total) if total else 0.0,
            }
        )
    return rows


SUMMARY_METRICS = (
    "retention_ratio_actual",
    "selected_pixel_count",
    "border_ratio",
    "center_ratio",
    "upper_ratio",
    "middle_ratio",
    "lower_ratio",
    "left_ratio",
    "x_center_ratio",
    "right_ratio",
    "connected_components",
    "largest_component_ratio",
    "mean_component_size",
    "median_component_size",
    "selected_region_count",
    "mean_selected_intensity",
    "mean_selected_grad_mag",
    "mean_selected_local_contrast",
    "mean_selected_delta_edge_node",
)


def summarize_accumulators(
    summary_acc: Dict[Tuple[str, float], MeanAccumulator],
    class_acc: Dict[Tuple[str, float, int], MeanAccumulator],
    skipped: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    for (selector, ratio), acc in sorted(summary_acc.items()):
        row = acc.row({"selector": selector, "retention_ratio_target": ratio, "status": "OK"}, SUMMARY_METRICS)
        summary_rows.append(row)
    for selector, reason in sorted(skipped.items()):
        summary_rows.append(
            {
                "selector": selector,
                "retention_ratio_target": "ALL",
                "status": "SKIPPED",
                "reason": reason,
                "sample_count": 0,
            }
        )
    class_rows: List[Dict[str, Any]] = []
    for (selector, ratio, label), acc in sorted(class_acc.items()):
        row = acc.row(
            {
                "selector": selector,
                "retention_ratio_target": ratio,
                "label": int(label),
                "class_name": class_name(label),
                "status": "OK",
            },
            SUMMARY_METRICS,
        )
        class_rows.append(row)
    return summary_rows, class_rows


def numeric(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def selector_level(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            grouped[str(row["selector"])].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for selector, items in grouped.items():
        out[selector] = {
            "border": float(np.mean([numeric(r, "mean_border_ratio") for r in items])),
            "center": float(np.mean([numeric(r, "mean_center_ratio") for r in items])),
            "components": float(np.mean([numeric(r, "mean_connected_components") for r in items])),
            "largest": float(np.mean([numeric(r, "mean_largest_component_ratio") for r in items])),
            "grad": float(np.mean([numeric(r, "mean_mean_selected_grad_mag") for r in items])),
            "contrast": float(np.mean([numeric(r, "mean_mean_selected_local_contrast") for r in items])),
            "delta": float(np.mean([numeric(r, "mean_mean_selected_delta_edge_node") for r in items])),
            "retention": float(np.mean([numeric(r, "mean_retention_ratio_actual") for r in items])),
        }
    return out


def rank_selectors(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    levels = selector_level(rows)
    if not levels:
        return []
    grads = np.array([v["grad"] for v in levels.values()], dtype=np.float32)
    contrasts = np.array([v["contrast"] for v in levels.values()], dtype=np.float32)
    deltas = np.array([v["delta"] for v in levels.values()], dtype=np.float32)
    borders = np.array([v["border"] for v in levels.values()], dtype=np.float32)
    comps = np.array([v["components"] for v in levels.values()], dtype=np.float32)

    def norm(value: float, arr: np.ndarray, invert: bool = False) -> float:
        if float(arr.max()) <= float(arr.min()):
            score = 0.5
        else:
            score = (float(value) - float(arr.min())) / (float(arr.max()) - float(arr.min()))
        return 1.0 - score if invert else score

    ranked = []
    for selector, values in levels.items():
        score = (
            norm(values["grad"], grads)
            + norm(values["contrast"], contrasts)
            + norm(values["delta"], deltas)
            + norm(values["border"], borders, invert=True)
            + norm(values["components"], comps, invert=True)
        ) / 5.0
        ranked.append((selector, float(score)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def report_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Selector | Ratio | N | Border | Center | Components | Largest comp | Grad | Contrast | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "OK":
            continue
        lines.append(
            "| {selector} | {ratio} | {n} | {border} | {center} | {components} | {largest} | {grad} | {contrast} | {delta} |".format(
                selector=row["selector"],
                ratio=fmt(row["retention_ratio_target"]),
                n=int(row.get("sample_count", 0)),
                border=fmt(row.get("mean_border_ratio")),
                center=fmt(row.get("mean_center_ratio")),
                components=fmt(row.get("mean_connected_components")),
                largest=fmt(row.get("mean_largest_component_ratio")),
                grad=fmt(row.get("mean_mean_selected_grad_mag")),
                contrast=fmt(row.get("mean_mean_selected_local_contrast")),
                delta=fmt(row.get("mean_mean_selected_delta_edge_node")),
            )
        )
    if len(lines) == 2:
        lines.append("| NO_ROWS | | | | | | | | | |")
    return lines


def write_report(
    output_dir: Path,
    graph_repo: Path,
    summary_rows: Sequence[Dict[str, Any]],
    class_rows: Sequence[Dict[str, Any]],
    skipped: Dict[str, str],
    max_samples_per_split: Optional[int],
    selectors: Sequence[str],
    retention_ratios: Sequence[float],
) -> str:
    ranked = rank_selectors(summary_rows)
    levels = selector_level(summary_rows)
    slic_status = "OK" if "slic_region_proposal" in levels else ("SKIPPED" if "slic_region_proposal" in skipped else "NOT_REQUESTED")
    verdict = "PASS"
    if max_samples_per_split is not None or skipped:
        verdict = "PARTIAL"
    if not levels:
        verdict = "FAIL"

    best_names = [name for name, _ in ranked[:3]]
    shortcut_center = sorted(levels.items(), key=lambda item: item[1]["center"], reverse=True)[:2]
    shortcut_border = sorted(levels.items(), key=lambda item: item[1]["border"], reverse=True)[:2]
    random_level = levels.get("random_pixel", {})

    lines: List[str] = [
        "# Stage 1 Pixel/Region Selection Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Stage 1 verdict: **{verdict}**.",
        f"- Graph repo: `{graph_repo}`.",
        f"- Samples per split: `{max_samples_per_split if max_samples_per_split is not None else 'FULL'}`.",
        f"- Selectors requested: `{', '.join(selectors)}`.",
        f"- Retention ratios: `{', '.join(fmt(r) for r in retention_ratios)}`.",
    ]
    if best_names:
        lines.append(f"- Selector hợp lý nhất theo proxy enrichment + ít shortcut hơn: `{', '.join(best_names)}`.")
    else:
        lines.append("- Chưa có selector OK để xếp hạng.")
    if shortcut_center:
        lines.append(
            "- Shortcut center rõ nhất: "
            + ", ".join(f"`{name}` center={fmt(values['center'])}" for name, values in shortcut_center)
            + "."
        )
    if shortcut_border:
        lines.append(
            "- Border/edge shortcut cao nhất: "
            + ", ".join(f"`{name}` border={fmt(values['border'])}" for name, values in shortcut_border)
            + "."
        )
    if slic_status == "OK":
        lines.append("- SLIC chạy được; dùng như region-contiguity baseline trước learned selector v0.")
    elif slic_status == "SKIPPED":
        lines.append(f"- SLIC bị skip rõ reason: `{skipped.get('slic_region_proposal')}`.")
    else:
        lines.append("- SLIC không được request trong lần chạy này.")
    lines.append("- Chưa nên dùng learned selector v0 như kết luận cuối; trước hết dùng các baseline này cho retention test.")
    lines.extend(
        [
            "",
            "## 2. Selector Comparison",
            "",
            "Bảng dưới là average theo các sample đã quét. Đây là proxy/debug metric, không phải accuracy phân loại.",
            "",
            *report_table(summary_rows),
            "",
            "Diễn giải nhanh:",
            "",
            "- `random_pixel` là lower-bound bắt buộc; nếu selector khác không vượt random về grad/contrast/delta hoặc chỉ tăng center/border, chưa đủ evidence.",
            "- `center_prior` chủ yếu đo positional shortcut, không được xem là semantic evidence.",
            "- `gradient_topk`, `contrast_topk`, `delta_edge_topk` đo feature-driven evidence nhưng có rủi ro chọn tóc/viền/nền.",
            "- `slic_region_proposal` nếu chạy được thì kiểm tra region liền mạch hơn pixel top-k, nhưng vẫn cần xem overlay.",
            "",
            "## 3. Coordinate Shortcut Analysis",
            "",
        ]
    )
    for name, values in sorted(levels.items()):
        lines.append(
            f"- `{name}`: center={fmt(values['center'])}, border={fmt(values['border'])}, "
            f"avg_components={fmt(values['components'])}, largest_component={fmt(values['largest'])}."
        )
    if random_level:
        lines.append(
            f"- Random baseline center={fmt(random_level.get('center'))}, border={fmt(random_level.get('border'))}; "
            "các selector khác nên được so với mốc này thay vì đọc riêng lẻ."
        )
    lines.extend(
        [
            "",
            "Center-prior nếu có center_ratio cao là expected behavior, nhưng đó cũng là bằng chứng cần kiểm soát positional shortcut vì FER-2013 mặt thường centered.",
            "Các selector gradient/contrast/delta nếu vừa tăng feature score vừa tăng border_ratio thì chưa thể kết luận chúng đang chọn vùng biểu cảm.",
            "",
            "## 4. Background/Edge Shortcut Analysis",
            "",
        ]
    )
    for name, values in sorted(levels.items(), key=lambda item: item[1]["border"], reverse=True):
        lines.append(
            f"- `{name}`: border={fmt(values['border'])}, grad={fmt(values['grad'])}, "
            f"contrast={fmt(values['contrast'])}, delta={fmt(values['delta'])}."
        )
    lines.extend(
        [
            "",
            "Border_ratio cao, component count cao, hoặc largest_component_ratio thấp là dấu hiệu mask rời rạc/edge-heavy. Cần xem figures để phân biệt facial edge với tóc/kính/viền nền.",
            "",
            "## 5. Per-class Pattern Analysis",
            "",
        ]
    )
    if class_rows:
        for row in class_rows:
            if row.get("status") != "OK":
                continue
            if str(row.get("retention_ratio_target")) not in {"0.1", "0.10"}:
                continue
            lines.append(
                f"- `{row['selector']}` / `{row['class_name']}` @10%: "
                f"center={fmt(row.get('mean_center_ratio'))}, border={fmt(row.get('mean_border_ratio'))}, "
                f"grad={fmt(row.get('mean_mean_selected_grad_mag'))}, delta={fmt(row.get('mean_mean_selected_delta_edge_node'))}."
            )
    else:
        lines.append("- Chưa có per-class row.")
    lines.append("")
    lines.append("Nếu khác biệt per-class nhỏ hoặc sample cap thấp, kết luận đúng là **chưa đủ evidence**, không phải không có pattern.")
    lines.extend(
        [
            "",
            "## 6. SLIC Analysis",
            "",
        ]
    )
    if slic_status == "OK":
        slic = levels.get("slic_region_proposal", {})
        grad = levels.get("gradient_topk", {})
        lines.append(
            f"- SLIC avg_components={fmt(slic.get('components'))}, largest_component={fmt(slic.get('largest'))}, "
            f"border={fmt(slic.get('border'))}."
        )
        if grad:
            lines.append(
                f"- Gradient top-k avg_components={fmt(grad.get('components'))}, largest_component={fmt(grad.get('largest'))}, "
                f"border={fmt(grad.get('border'))}."
            )
        lines.append("- Nếu SLIC có component ít hơn và largest component lớn hơn, nó nên được giữ làm region baseline cho retention test.")
    elif slic_status == "SKIPPED":
        lines.append(f"- SLIC skipped vì thiếu hoặc lỗi `skimage`: `{skipped.get('slic_region_proposal')}`.")
        lines.append("- Cài/enable `scikit-image` nếu muốn kiểm tra region proposal.")
    else:
        lines.append("- SLIC không chạy trong lần này.")
    lines.extend(
        [
            "",
            "## 7. Decision for Stage 2",
            "",
            f"- Decision: **{verdict}**.",
            "- Nếu dùng cho retention test, ưu tiên selector có feature enrichment tốt nhưng không border/center quá cực đoan.",
        ]
    )
    if best_names:
        lines.append(f"- Candidate retention selectors: `{', '.join(best_names)}`.")
    lines.extend(
        [
            "- Luôn giữ `random_pixel` và `center_prior` làm control.",
            "- Nếu selector feature-driven có border_ratio/component count quá cao, quay lại Stage 1 với SLIC hoặc smoothing/region constraint trước learned selector.",
            "- Learned selector v0 chỉ nên bắt đầu sau khi overlay + coordinate/component stats xác nhận không collapse vào center/border shortcut.",
            "",
            "## Output Index",
            "",
            "- `selector_summary.csv`",
            "- `selector_per_class_stats.csv`",
            "- `selector_coordinate_histograms.csv`",
            "- `selector_component_stats.csv`",
            "- `selector_retention_metrics.csv`",
            "- `figures/`",
        ]
    )
    (output_dir / "stage1_selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 controlled pixel/region selector baselines.")
    parser.add_argument("--graph_repo", "--graph_repo_path", dest="graph_repo", default=None)
    parser.add_argument("--output_dir", default="outputs/stage1_pixel_region_selection")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--figure_samples_per_class", type=int, default=5)
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS), choices=list(DEFAULT_SELECTORS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slic_segments", type=int, default=96)
    parser.add_argument("--slic_compactness", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph_repo = resolve_graph_repo(args.graph_repo)
    output_dir = ensure_dir(Path(args.output_dir))
    ensure_dir(output_dir / "figures")
    selectors = list(args.selectors)
    ratios = [float(r) for r in args.retention_ratios]
    reader = GraphRepositoryReader(graph_repo)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    height = int(shared.height)
    width = int(shared.width)
    num_nodes = height * width
    slic_fn, slic_error = try_import_slic()
    skipped: Dict[str, str] = {}
    if "slic_region_proposal" in selectors and slic_fn is None:
        skipped["slic_region_proposal"] = slic_error or "skimage.segmentation.slic unavailable"

    summary_acc: Dict[Tuple[str, float], MeanAccumulator] = defaultdict(MeanAccumulator)
    class_acc: Dict[Tuple[str, float, int], MeanAccumulator] = defaultdict(MeanAccumulator)
    hist: Dict[Tuple[str, float, str, str, int], int] = defaultdict(int)
    visual_counts: Dict[int, int] = defaultdict(int)
    scanned = defaultdict(int)

    with CsvSink(output_dir / "selector_retention_metrics.csv", RETENTION_FIELDS) as retention_sink, CsvSink(
        output_dir / "selector_component_stats.csv", COMPONENT_FIELDS
    ) as component_sink:
        for split in SPLITS:
            for sample in iter_limited(reader, split, args.max_samples_per_split):
                resolved = resolver.resolve(sample)
                label = int(resolved.label)
                scanned[split] += 1
                node_names = list(resolved.node_feature_names or NODE_FEATURES)
                edge_names = list(resolved.edge_feature_names or EDGE_FEATURES)
                node = resolved.node_features.detach().cpu().numpy()
                intensity_idx = feature_index(node_names, "intensity", 0)
                x_idx = feature_index(node_names, "x_norm", 1)
                y_idx = feature_index(node_names, "y_norm", 2)
                grad_idx = feature_index(node_names, "grad_mag", 5)
                contrast_idx = feature_index(node_names, "local_contrast", 6)
                delta_idx = feature_index(edge_names, "delta_intensity", 3)
                intensity = node[:, intensity_idx].astype(np.float32)
                x_norm = node[:, x_idx].astype(np.float32)
                y_norm = node[:, y_idx].astype(np.float32)
                grad_mag = node[:, grad_idx].astype(np.float32)
                local_contrast = node[:, contrast_idx].astype(np.float32)
                delta_edge_node = edge_to_node_mean(resolved.edge_index, resolved.edge_attr[:, delta_idx], num_nodes)
                visualize = choose_visual_sample(visual_counts, label, args.figure_samples_per_class)

                for ratio in ratios:
                    for selector, mask, region_count, extra in selector_masks(
                        selectors,
                        ratio,
                        int(resolved.graph_id),
                        args.seed,
                        height,
                        width,
                        intensity,
                        x_norm,
                        y_norm,
                        grad_mag,
                        local_contrast,
                        delta_edge_node,
                        slic_fn,
                        args.slic_segments,
                        args.slic_compactness,
                    ):
                        row, component_row = mask_metrics(
                            split=split,
                            graph_id=int(resolved.graph_id),
                            label=label,
                            selector=selector,
                            ratio_target=ratio,
                            mask=mask,
                            height=height,
                            width=width,
                            intensity=intensity,
                            x_norm=x_norm,
                            y_norm=y_norm,
                            grad_mag=grad_mag,
                            local_contrast=local_contrast,
                            delta_edge_node=delta_edge_node,
                            selected_region_count=region_count,
                        )
                        if extra.get("mean_slic_region_size") is not None:
                            row["mean_slic_region_size"] = extra["mean_slic_region_size"]
                            component_row["mean_slic_region_size"] = extra["mean_slic_region_size"]
                        retention_sink.write(row)
                        component_sink.write(component_row)
                        summary_acc[(selector, ratio)].add(row, SUMMARY_METRICS)
                        class_acc[(selector, ratio, label)].add(row, SUMMARY_METRICS)
                        update_histograms(hist, selector, ratio, "ALL", mask, x_norm, y_norm)
                        update_histograms(hist, selector, ratio, class_name(label), mask, x_norm, y_norm)
                        if visualize:
                            save_visuals(
                                output_dir,
                                selector,
                                ratio,
                                label,
                                int(resolved.graph_id),
                                split,
                                intensity.reshape(height, width),
                                mask.reshape(height, width),
                            )

    summary_rows, class_rows = summarize_accumulators(summary_acc, class_acc, skipped)
    write_csv(output_dir / "selector_summary.csv", summary_rows)
    write_csv(output_dir / "selector_per_class_stats.csv", class_rows)
    write_csv(output_dir / "selector_coordinate_histograms.csv", build_hist_rows(hist))
    verdict = write_report(
        output_dir,
        graph_repo,
        summary_rows,
        class_rows,
        skipped,
        args.max_samples_per_split,
        selectors,
        ratios,
    )
    ranked = rank_selectors(summary_rows)
    print(f"[Stage1] output_dir={output_dir}")
    print(f"[Stage1] scanned={dict(scanned)}")
    print(f"[Stage1] verdict={verdict}")
    print("[Stage1] top_selectors=" + ", ".join(name for name, _ in ranked[:3]))
    if skipped:
        print(f"[Stage1] skipped={skipped}")


if __name__ == "__main__":
    main()
