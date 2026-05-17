"""Stage 3 hybrid region-constrained evidence selector v0.

This is a controlled selector probe: it does not train D12/D13, does not mutate
graph artifacts, and does not introduce learned selector modules. Hybrid masks
combine Stage 1/2 evidence features with optional smoothing and SLIC continuity,
then use the same lightweight image classifier probe as Stage 2.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.labels import EMOTION_NAMES
from stage1_pixel_region_selection import (
    connected_component_stats,
    ensure_dir,
    fmt,
    normalize01,
    resolve_graph_repo,
    safe_name,
    save_png,
    to_u8,
    try_import_slic,
)
from stage2_retention_deletion_test import (
    apply_mask,
    class_name,
    load_split_records,
    make_original_dataset,
    ratio_name,
    save_confusion,
    train_eval_classifier,
)

DEFAULT_HYBRID_SELECTORS = (
    "hybrid_pixel_score",
    "hybrid_pixel_smooth",
    "hybrid_slic_region",
    "hybrid_slic_region_center_control",
)
DEFAULT_RATIOS = (0.05, 0.10, 0.20, 0.40)
WEIGHT_GRID = {
    "A_delta": (1.0, 0.0, 0.0),
    "B_grad": (0.0, 1.0, 0.0),
    "C_delta_grad": (0.5, 0.5, 0.0),
    "D_delta_grad_contrast": (0.5, 0.3, 0.2),
    "E_balanced": (0.4, 0.4, 0.2),
}
DEFAULT_BORDER_WEIGHTS = (0.0, 0.1, 0.2)
SUMMARY_FIELDS = (
    "selector",
    "variant_id",
    "weight_id",
    "retention_ratio",
    "w_delta",
    "w_grad",
    "w_contrast",
    "w_border",
    "sample_count",
    "only_selected_accuracy",
    "only_selected_macro_f1",
    "only_selected_weighted_f1",
    "gap_vs_random",
    "gap_vs_center",
    "gap_vs_delta_edge_topk",
    "gap_vs_gradient_topk",
    "gap_vs_slic_region_proposal",
    "mean_border_ratio",
    "mean_center_ratio",
    "mean_upper_ratio",
    "mean_middle_ratio",
    "mean_lower_ratio",
    "mean_connected_components",
    "mean_largest_component_ratio",
    "mean_selected_pixel_count",
    "mean_selected_intensity",
    "mean_selected_grad_mag",
    "mean_selected_local_contrast",
    "mean_selected_delta_edge_node",
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


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = int(scores.size)
    k = max(1, min(int(k), n))
    mask = np.zeros((n,), dtype=bool)
    if k >= n:
        mask[:] = True
        return mask
    idx = np.argpartition(scores, n - k)[n - k :]
    mask[idx] = True
    return mask


def border_penalty(x_norm: np.ndarray, y_norm: np.ndarray, margin: float = 0.15) -> np.ndarray:
    dist = np.minimum.reduce([x_norm, 1.0 - x_norm, y_norm, 1.0 - y_norm])
    return np.clip((float(margin) - dist) / max(float(margin), 1e-6), 0.0, 1.0).astype(np.float32)


def hybrid_score(record: Dict[str, Any], weights: Tuple[float, float, float], w_border: float) -> np.ndarray:
    w_delta, w_grad, w_contrast = weights
    score = (
        float(w_delta) * normalize01(record["delta_edge_node"])
        + float(w_grad) * normalize01(record["grad_mag"])
        + float(w_contrast) * normalize01(record["local_contrast"])
    )
    if float(w_border) > 0:
        score = score - float(w_border) * border_penalty(record["x_norm"], record["y_norm"])
    return normalize01(score).astype(np.float32)


def smooth_score(score: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    score_2d = np.asarray(score, dtype=np.float32).reshape(48, 48)
    try:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(score_2d, sigma=float(sigma))
    except Exception:
        padded = np.pad(score_2d, 1, mode="edge")
        out = np.zeros_like(score_2d)
        for dy in range(3):
            for dx in range(3):
                out += padded[dy : dy + 48, dx : dx + 48]
        out /= 9.0
    return normalize01(out.reshape(-1)).astype(np.float32)


def slic_region_mask(
    intensity: np.ndarray,
    score: np.ndarray,
    k: int,
    slic_fn: Any,
    n_segments: int,
    compactness: float,
    score_mode: str,
) -> Tuple[np.ndarray, int]:
    if slic_fn is None:
        return topk_mask(score, k), 0
    segments = slic_fn(
        intensity.reshape(48, 48).astype(np.float32),
        n_segments=int(n_segments),
        compactness=float(compactness),
        start_label=0,
        channel_axis=None,
    ).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    rows: List[Tuple[float, int, int]] = []
    for region_id in np.unique(segments):
        region_mask = segments == region_id
        values = score[region_mask]
        if score_mode == "percentile":
            region_score = float(np.percentile(values, 80))
        else:
            region_score = float(values.mean())
        rows.append((region_score, int(region_id), int(region_mask.sum())))
    rows.sort(reverse=True)
    selected = np.zeros_like(score, dtype=bool)
    region_count = 0
    for _, region_id, _ in rows:
        selected[segments == region_id] = True
        region_count += 1
        if int(selected.sum()) >= int(k):
            break
    return selected, region_count


def build_hybrid_mask(
    record: Dict[str, Any],
    selector: str,
    weights: Tuple[float, float, float],
    w_border: float,
    ratio: float,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
    smooth_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    score = hybrid_score(record, weights, w_border)
    if selector == "hybrid_pixel_smooth":
        score = smooth_score(score, smooth_sigma)
    k = max(1, int(round(48 * 48 * float(ratio))))
    if selector == "hybrid_pixel_score" or selector == "hybrid_pixel_smooth":
        return topk_mask(score, k), score, 0
    if selector == "hybrid_slic_region":
        mask, region_count = slic_region_mask(record["intensity"], score, k, slic_fn, slic_segments, slic_compactness, "percentile")
        return mask, score, region_count
    if selector == "hybrid_slic_region_center_control":
        mask, region_count = slic_region_mask(record["intensity"], score, k, slic_fn, slic_segments, slic_compactness, "mean")
        return mask, score, region_count
    raise ValueError(f"Unknown hybrid selector: {selector}")


def selected_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) <= 0:
        return 0.0
    return float(np.asarray(values, dtype=np.float32)[mask].mean())


def mask_stat_row(
    record: Dict[str, Any],
    selector: str,
    variant_id: str,
    weight_id: str,
    ratio: float,
    weights: Tuple[float, float, float],
    w_border: float,
    mask: np.ndarray,
    region_count: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    x = record["x_norm"]
    y = record["y_norm"]
    selected = int(mask.sum())
    border = (x <= 0.10) | (x >= 0.90) | (y <= 0.10) | (y >= 0.90)
    center = (((x - 0.5) / 0.25) ** 2 + ((y - 0.5) / 0.30) ** 2) <= 1.0
    upper = y < 1.0 / 3.0
    middle = (y >= 1.0 / 3.0) & (y < 2.0 / 3.0)
    lower = y >= 2.0 / 3.0
    comp = connected_component_stats(mask.reshape(48, 48))

    def ratio_in(region: np.ndarray) -> float:
        return float(np.logical_and(mask, region).sum() / max(1, selected))

    base = {
        "split": record["split"],
        "graph_id": int(record["graph_id"]),
        "label": int(record["label"]),
        "class_name": class_name(record["label"]),
        "selector": selector,
        "variant_id": variant_id,
        "weight_id": weight_id,
        "retention_ratio": float(ratio),
        "w_delta": weights[0],
        "w_grad": weights[1],
        "w_contrast": weights[2],
        "w_border": float(w_border),
        "selected_pixel_count": selected,
        "border_ratio": ratio_in(border),
        "center_ratio": ratio_in(center),
        "upper_ratio": ratio_in(upper),
        "middle_ratio": ratio_in(middle),
        "lower_ratio": ratio_in(lower),
        "selected_region_count": int(region_count),
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


def make_dataset(
    records: Sequence[Dict[str, Any]],
    selector: str,
    weights: Tuple[float, float, float],
    w_border: float,
    ratio: float,
    fill_mode: str,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
    smooth_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    for record in records:
        mask, _, _ = build_hybrid_mask(record, selector, weights, w_border, ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        x_rows.append(apply_mask(record["intensity"], mask, "only_selected", fill_mode))
        y_rows.append(int(record["label"]))
    return np.stack(x_rows, axis=0).astype(np.float32), np.asarray(y_rows, dtype=np.int64)


def save_score_visual(
    output_dir: Path,
    record: Dict[str, Any],
    selector: str,
    variant_id: str,
    ratio: float,
    score: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
) -> None:
    class_dir = f"class_{int(record['label'])}_{safe_name(class_name(record['label']))}"
    base = ensure_dir(output_dir / "figures" / variant_id / ratio_name(ratio) / class_dir)
    mask_base = ensure_dir(output_dir / "masks" / variant_id / ratio_name(ratio) / class_dir)
    stem = f"{record['split']}_graph_{int(record['graph_id'])}"
    original = to_u8(record["intensity"].reshape(48, 48))
    score_u8 = to_u8(normalize01(score).reshape(48, 48))
    mask_u8 = mask.reshape(48, 48).astype(np.uint8) * 255
    overlay = np.stack([original, original, original], axis=-1).astype(np.float32)
    overlay[mask.reshape(48, 48), 0] = 255
    overlay[mask.reshape(48, 48), 1:] *= 0.45
    only_selected = to_u8(apply_mask(record["intensity"], mask, "only_selected", fill_mode).reshape(48, 48))
    delete_selected = to_u8(apply_mask(record["intensity"], mask, "delete_selected", fill_mode).reshape(48, 48))
    sep = np.full((48, 4, 3), 255, dtype=np.uint8)
    parts = [
        np.stack([original, original, original], axis=-1),
        np.stack([score_u8, score_u8, score_u8], axis=-1),
        np.stack([mask_u8, mask_u8, mask_u8], axis=-1),
        np.clip(overlay, 0, 255).astype(np.uint8),
        np.stack([only_selected, only_selected, only_selected], axis=-1),
        np.stack([delete_selected, delete_selected, delete_selected], axis=-1),
    ]
    grid = parts[0]
    for part in parts[1:]:
        grid = np.concatenate([grid, sep, part], axis=1)
    save_png(base / f"{stem}_comparison.png", grid)
    save_png(mask_base / f"{stem}_mask.png", mask_u8)


def load_stage2_baselines(stage2_dir: Path) -> Dict[Tuple[str, float], Dict[str, Any]]:
    path = stage2_dir / "retention_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        try:
            out[(str(row["selector"]), float(row["retention_ratio"]))] = row
        except Exception:
            continue
    return out


def gap(row: Dict[str, Any], baselines: Dict[Tuple[str, float], Dict[str, Any]], baseline: str) -> float:
    base = baselines.get((baseline, float(row["retention_ratio"])))
    if not base:
        return float("nan")
    return float(row["only_selected_macro_f1"]) - float(base.get("macro_f1", base.get("only_selected_macro_f1", 0.0)))


def per_class_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in metric_rows:
        for name in EMOTION_NAMES:
            rows.append(
                {
                    "selector": row["selector"],
                    "variant_id": row["variant_id"],
                    "weight_id": row["weight_id"],
                    "retention_ratio": row["retention_ratio"],
                    "class_name": name,
                    "only_selected_f1": row.get(f"per_class_f1_{name}", 0.0),
                }
            )
    return rows


def best_by_ratio(rows: Sequence[Dict[str, Any]], ratios: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for ratio in ratios:
        candidates = [r for r in rows if abs(float(r["retention_ratio"]) - float(ratio)) < 1e-9]
        if candidates:
            out[float(ratio)] = max(candidates, key=lambda r: float(r["only_selected_macro_f1"]))
    return out


def write_report(
    output_dir: Path,
    metric_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    per_class: Sequence[Dict[str, Any]],
    ratios: Sequence[float],
    max_train: Optional[int],
    max_eval: Optional[int],
    slic_available: bool,
) -> str:
    best = best_by_ratio(metric_rows, ratios)
    best_any = max(metric_rows, key=lambda r: float(r["only_selected_macro_f1"])) if metric_rows else None
    positive = [r for r in baseline_rows if float(r.get("gap_vs_random", 0.0)) > 0.01]
    verdict = "PARTIAL" if metric_rows else "FAIL"
    if positive and max_train is None and max_eval is None:
        verdict = "PASS"
    slic_rows = [r for r in metric_rows if "slic" in r["selector"]]
    pixel_rows = [r for r in metric_rows if "pixel" in r["selector"]]
    best_slic = max(slic_rows, key=lambda r: float(r["only_selected_macro_f1"])) if slic_rows else None
    best_pixel = max(pixel_rows, key=lambda r: float(r["only_selected_macro_f1"])) if pixel_rows else None

    lines: List[str] = [
        "# Stage 3 Hybrid Region-Constrained Evidence Selector Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Stage 3 verdict: **{verdict}**.",
        f"- Train/eval sample cap: `{max_train if max_train is not None else 'FULL'}` / `{max_eval if max_eval is not None else 'FULL'}`.",
        f"- SLIC available: `{slic_available}`.",
    ]
    for ratio in ratios:
        row = best.get(float(ratio))
        if row:
            lines.append(
                f"- Best hybrid @{int(float(ratio) * 100)}%: `{row['variant_id']}` macro_f1={fmt(row['only_selected_macro_f1'])}."
            )
    if best_any:
        lines.append(
            f"- Overall best: `{best_any['variant_id']}` @{fmt(best_any['retention_ratio'])}, macro_f1={fmt(best_any['only_selected_macro_f1'])}."
        )
    if best_slic and best_pixel:
        lines.append(
            f"- Best SLIC hybrid={fmt(best_slic['only_selected_macro_f1'])}; best pixel hybrid={fmt(best_pixel['only_selected_macro_f1'])}."
        )
    if positive:
        best_gap = max(positive, key=lambda r: float(r["gap_vs_random"]))
        lines.append(f"- Best gap vs random: `{best_gap['variant_id']}` gap={fmt(best_gap['gap_vs_random'])}.")
    else:
        lines.append("- No hybrid has a robust positive gap vs random in this capped probe.")
    lines.extend(
        [
            "",
            "## 2. Hybrid Formula & Weight Grid",
            "",
            "Pixel score:",
            "",
            "`score = norm(delta_edge_node) * w_delta + norm(grad_mag) * w_grad + norm(local_contrast) * w_contrast - border_penalty * w_border`",
            "",
            "Final score is min-max normalized per image before selection. `border_penalty` is a soft penalty based on distance to image border; it does not add a center prior.",
            "",
            "| Weight ID | w_delta | w_grad | w_contrast |",
            "|---|---:|---:|---:|",
        ]
    )
    for weight_id, weights in WEIGHT_GRID.items():
        lines.append(f"| {weight_id} | {weights[0]} | {weights[1]} | {weights[2]} |")
    lines.extend(
        [
            "",
            "Selectors: `hybrid_pixel_score`, `hybrid_pixel_smooth`, `hybrid_slic_region`, `hybrid_slic_region_center_control`.",
            "",
            "## 3. Retention Performance",
            "",
            "| Ratio | Best Variant | Macro F1 | Gap vs Random | Gap vs Center | Gap vs Delta | Gap vs Gradient |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio, row in best.items():
        lines.append(
            f"| {fmt(ratio)} | {row['variant_id']} | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row.get('gap_vs_random'))} | {fmt(row.get('gap_vs_center'))} | "
            f"{fmt(row.get('gap_vs_delta_edge_topk'))} | {fmt(row.get('gap_vs_gradient_topk'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. Region Continuity Analysis",
            "",
            "| Variant | Ratio | Components | Largest Component | Border | Center |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary_rows, key=lambda r: float(r["only_selected_macro_f1"]), reverse=True)[:20]:
        lines.append(
            f"| {row['variant_id']} | {fmt(row['retention_ratio'])} | {fmt(row['mean_connected_components'])} | "
            f"{fmt(row['mean_largest_component_ratio'])} | {fmt(row['mean_border_ratio'])} | {fmt(row['mean_center_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "SLIC variants should reduce fragmentation. Pixel variants can win F1 but may remain noisy; compare component count before using them as pseudo-labels.",
            "",
            "## 5. Shortcut Analysis",
            "",
        ]
    )
    high_center = sorted(summary_rows, key=lambda r: float(r["mean_center_ratio"]), reverse=True)[:5]
    high_border = sorted(summary_rows, key=lambda r: float(r["mean_border_ratio"]), reverse=True)[:5]
    lines.append("- Highest center ratios: " + ", ".join(f"`{r['variant_id']}`={fmt(r['mean_center_ratio'])}" for r in high_center) + ".")
    lines.append("- Highest border ratios: " + ", ".join(f"`{r['variant_id']}`={fmt(r['mean_border_ratio'])}" for r in high_border) + ".")
    lines.append("- A hybrid is suspect if its F1 gain comes with high center or border ratio relative to Stage 2 controls.")
    lines.extend(
        [
            "",
            "## 6. Per-class Analysis",
            "",
        ]
    )
    for name in EMOTION_NAMES:
        candidates = [r for r in per_class if abs(float(r["retention_ratio"]) - 0.10) < 1e-9 and r["class_name"] == name]
        if candidates:
            row = max(candidates, key=lambda r: float(r["only_selected_f1"]))
            lines.append(f"- `{name}` @10% best: `{row['variant_id']}` F1={fmt(row['only_selected_f1'])}.")
    lines.extend(
        [
            "",
            "## 7. Visual Review",
            "",
            "- Figures are stored as `original | score map | selected mask | overlay | only_selected | delete_selected`.",
            "- Review high-F1 variants and high-border variants side-by-side; do not accept a hybrid only because a few masks look clean.",
            "- If overlays concentrate on hair, glasses, image border, or background texture, keep the variant as diagnostic only.",
            "",
            "## 8. Decision for Stage 4",
            "",
            f"- Decision: **{verdict}**.",
        ]
    )
    if best_any:
        lines.append(f"- Recommended learned selector seed: `{best_any['variant_id']}` only if visual review does not show border/center shortcut.")
    if best_slic:
        lines.append(f"- Keep SLIC-region hybrid as continuity regularizer/baseline; best SLIC variant is `{best_slic['variant_id']}`.")
    lines.extend(
        [
            "- Stage 4 should learn from hybrid scores with random and center controls preserved.",
            "- If best variants do not beat Stage 2 delta/gradient baselines, return to Stage 1/2 for SLIC tuning or feature normalization.",
            "",
            "## Output Index",
            "",
            "- `hybrid_selector_summary.csv`",
            "- `hybrid_selector_component_stats.csv`",
            "- `hybrid_selector_coordinate_stats.csv`",
            "- `hybrid_retention_metrics.csv`",
            "- `hybrid_vs_stage2_baselines.csv`",
            "- `figures/`",
            "- `masks/`",
        ]
    )
    (output_dir / "stage3_hybrid_selector_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 hybrid region-constrained evidence selector.")
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage1_dir", default="outputs/stage1_pixel_region_selection")
    parser.add_argument("--stage2_dir", default="outputs/stage2_retention_deletion_test")
    parser.add_argument("--output_dir", default="outputs/stage3_hybrid_evidence_selector")
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_HYBRID_SELECTORS), choices=list(DEFAULT_HYBRID_SELECTORS))
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--border_weights", nargs="+", type=float, default=list(DEFAULT_BORDER_WEIGHTS))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--max_train_samples", type=int, default=800)
    parser.add_argument("--max_eval_samples", type=int, default=400)
    parser.add_argument("--classifier_max_iter", type=int, default=300)
    parser.add_argument("--figure_samples_per_class", type=int, default=5)
    parser.add_argument("--fill_mode", choices=["mean", "zero"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slic_segments", type=int, default=96)
    parser.add_argument("--slic_compactness", type=float, default=0.2)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    return parser


def train_eval_fast(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, seed: int, max_iter: int) -> Tuple[Dict[str, Any], np.ndarray]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=int(max_iter),
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        ),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_eval)
    per_class = f1_score(y_eval, pred, average=None, labels=list(range(len(EMOTION_NAMES))), zero_division=0)
    row: Dict[str, Any] = {
        "classifier": f"StandardScaler+SGDClassifier(log_loss,max_iter={int(max_iter)})",
        "num_train": int(len(y_train)),
        "num_eval": int(len(y_eval)),
        "only_selected_accuracy": float(accuracy_score(y_eval, pred)),
        "only_selected_macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        "only_selected_weighted_f1": float(f1_score(y_eval, pred, average="weighted", zero_division=0)),
    }
    for idx, name in enumerate(EMOTION_NAMES):
        row[f"per_class_f1_{name}"] = float(per_class[idx])
    return row, confusion_matrix(y_eval, pred, labels=list(range(len(EMOTION_NAMES))))


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    slic_fn, slic_error = try_import_slic()
    slic_available = slic_fn is not None
    selectors = list(args.selectors)
    if not slic_available:
        print(f"[Stage3] SLIC unavailable: {slic_error}")
        selectors = [s for s in selectors if "slic" not in s]

    print("[Stage3] loading records")
    train_records = load_split_records(reader, resolver, args.train_split, args.max_train_samples)
    eval_records = load_split_records(reader, resolver, args.eval_split, args.max_eval_samples)
    x_train_orig, y_train, _ = make_original_dataset(train_records)
    x_eval_orig, y_eval, _ = make_original_dataset(eval_records)
    original_metrics, _, original_cm = train_eval_classifier(x_train_orig, y_train, x_eval_orig, y_eval, args.seed)
    save_confusion(output_dir / "confusion_matrices" / "original.csv", original_cm)

    baselines = load_stage2_baselines(Path(args.stage2_dir))
    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, float, int], int] = defaultdict(int)

    for selector in selectors:
        for weight_id, weights in WEIGHT_GRID.items():
            for w_border in [float(v) for v in args.border_weights]:
                variant_id = f"{selector}__{weight_id}__b{str(w_border).replace('.', 'p')}"
                for ratio in [float(r) for r in args.retention_ratios]:
                    print(f"[Stage3] {variant_id} ratio={ratio}")
                    x_train_rows: List[np.ndarray] = []
                    for record in train_records:
                        mask, _, _ = build_hybrid_mask(record, selector, weights, w_border, ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
                        x_train_rows.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
                    x_train = np.stack(x_train_rows, axis=0).astype(np.float32)

                    x_eval_rows: List[np.ndarray] = []
                    eval_stat_rows: List[Dict[str, Any]] = []
                    for record in eval_records:
                        mask, score, region_count = build_hybrid_mask(record, selector, weights, w_border, ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
                        x_eval_rows.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
                        base, component, coord = mask_stat_row(record, selector, variant_id, weight_id, ratio, weights, w_border, mask, region_count)
                        eval_stat_rows.append({**component})
                        component_rows.append(component)
                        coordinate_rows.append(coord)
                        key = (variant_id, ratio, int(record["label"]))
                        if visual_counts[key] < int(args.figure_samples_per_class):
                            save_score_visual(output_dir, record, selector, variant_id, ratio, score, mask, args.fill_mode)
                            visual_counts[key] += 1
                    x_eval = np.stack(x_eval_rows, axis=0).astype(np.float32)
                    metrics, cm = train_eval_fast(x_train, y_train, x_eval, y_eval, args.seed, args.classifier_max_iter)
                    metric_row = {
                        "selector": selector,
                        "variant_id": variant_id,
                        "weight_id": weight_id,
                        "retention_ratio": ratio,
                        "w_delta": weights[0],
                        "w_grad": weights[1],
                        "w_contrast": weights[2],
                        "w_border": w_border,
                        "original_accuracy": original_metrics["accuracy"],
                        "original_macro_f1": original_metrics["macro_f1"],
                        "original_weighted_f1": original_metrics["weighted_f1"],
                        **metrics,
                    }
                    for baseline in ("random_pixel", "center_prior", "delta_edge_topk", "gradient_topk", "contrast_topk", "slic_region_proposal"):
                        metric_row[f"gap_vs_{baseline}"] = gap(metric_row, baselines, baseline)
                    metric_row["gap_vs_random"] = metric_row.get("gap_vs_random_pixel")
                    metric_row["gap_vs_center"] = metric_row.get("gap_vs_center_prior")
                    metric_rows.append(metric_row)
                    agg = aggregate_stats(eval_stat_rows)
                    summary_rows.append({**metric_row, "sample_count": len(eval_stat_rows), **agg})
                    save_confusion(output_dir / "confusion_matrices" / variant_id / f"{ratio_name(ratio)}.csv", cm)

    baseline_rows: List[Dict[str, Any]] = []
    for row in metric_rows:
        baseline_rows.append(
            {
                "selector": row["selector"],
                "variant_id": row["variant_id"],
                "weight_id": row["weight_id"],
                "retention_ratio": row["retention_ratio"],
                "only_selected_macro_f1": row["only_selected_macro_f1"],
                "stage2_random_macro_f1": baselines.get(("random_pixel", float(row["retention_ratio"])), {}).get("macro_f1", ""),
                "stage2_center_macro_f1": baselines.get(("center_prior", float(row["retention_ratio"])), {}).get("macro_f1", ""),
                "stage2_delta_macro_f1": baselines.get(("delta_edge_topk", float(row["retention_ratio"])), {}).get("macro_f1", ""),
                "stage2_gradient_macro_f1": baselines.get(("gradient_topk", float(row["retention_ratio"])), {}).get("macro_f1", ""),
                "stage2_slic_macro_f1": baselines.get(("slic_region_proposal", float(row["retention_ratio"])), {}).get("macro_f1", ""),
                "gap_vs_random": row.get("gap_vs_random_pixel"),
                "gap_vs_center": row.get("gap_vs_center_prior"),
                "gap_vs_delta_edge_topk": row.get("gap_vs_delta_edge_topk"),
                "gap_vs_gradient_topk": row.get("gap_vs_gradient_topk"),
                "gap_vs_slic_region_proposal": row.get("gap_vs_slic_region_proposal"),
            }
        )
    per_class = per_class_rows(metric_rows)
    write_csv(output_dir / "hybrid_retention_metrics.csv", metric_rows)
    write_csv(output_dir / "hybrid_selector_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "hybrid_selector_component_stats.csv", component_rows)
    write_csv(output_dir / "hybrid_selector_coordinate_stats.csv", coordinate_rows)
    write_csv(output_dir / "hybrid_vs_stage2_baselines.csv", baseline_rows)
    write_csv(output_dir / "hybrid_per_class_f1.csv", per_class)
    verdict = write_report(
        output_dir,
        metric_rows,
        summary_rows,
        baseline_rows,
        per_class,
        [float(r) for r in args.retention_ratios],
        args.max_train_samples,
        args.max_eval_samples,
        slic_available,
    )
    best = best_by_ratio(metric_rows, [0.05, 0.10, 0.20])
    print(f"[Stage3] output_dir={output_dir}")
    print(f"[Stage3] verdict={verdict}")
    for ratio, row in best.items():
        print(f"[Stage3] best_{int(ratio * 100)}={row['variant_id']} macro_f1={fmt(row['only_selected_macro_f1'])}")
    if any("slic" in r["selector"] for r in metric_rows):
        print("[Stage3] slic_region_hybrid=OK")
    else:
        print("[Stage3] slic_region_hybrid=SKIPPED")


if __name__ == "__main__":
    main()
