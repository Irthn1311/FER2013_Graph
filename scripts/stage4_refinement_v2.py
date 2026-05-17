"""Stage 4 refinement v2 for learned evidence selector v0.

Scope is intentionally narrow: no Stage 5, no part grouping, no motif bank, no
SupCon, no D12/D13 training. This script reruns two Stage 4 candidates with a
small set of refinements: soft teacher, light long/border penalty, and
connected-component postprocessing.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    selected_mean,
    to_u8,
    topk_mask,
    try_import_slic,
)
from stage2_retention_deletion_test import class_name, make_original_dataset, ratio_name, save_confusion
from stage4_learned_evidence_selector import (
    ExperimentConfig,
    build_control_mask,
    build_model,
    build_teacher,
    center_prior_map,
    feature_stack,
    input_feature_names,
    mask_structure_stats,
    score_correlations,
    total_variation,
)
from stage4_refinement_run import apply_mask_refined, eval_probe, local_mean_fill, train_probe
from stage36_structure_aware_diagnostics import load_split_records, selected_map_mean


CONTROLS = (
    "random_pixel",
    "center_prior",
    "gradient_topk",
    "delta_edge_topk",
    "main_hybrid_teacher",
    "structure_aux_teacher",
)


@dataclass
class V2Variant:
    name: str
    group: str
    arch: str
    input_variant: str
    target_ratio: float
    regularized: bool
    soft_teacher: bool
    long_border_penalty: bool
    postprocess_cc: bool
    base_variant_name: str = ""


class V2Dataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], input_variant: str, target_ratio: float, slic_fn: Any, args: argparse.Namespace) -> None:
        self.records = list(records)
        self.input_variant = input_variant
        self.target_ratio = float(target_ratio)
        self.slic_fn = slic_fn
        self.args = args

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        teacher_score, teacher_mask, _ = build_teacher(
            record,
            "main_hybrid",
            self.target_ratio,
            self.slic_fn,
            self.args.slic_segments,
            self.args.slic_compactness,
            self.args.smooth_sigma,
        )
        st = record["structure"]
        reg = np.stack(
            [
                st["smooth_region_map"].reshape(48, 48),
                st["long_contour_map"].reshape(48, 48),
                st["border_penalty_map"].reshape(48, 48),
                center_prior_map(record).reshape(48, 48),
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "x": torch.from_numpy(feature_stack(record, self.input_variant)),
            "teacher_score": torch.from_numpy(teacher_score.reshape(1, 48, 48).astype(np.float32)),
            "teacher_mask": torch.from_numpy(teacher_mask.reshape(1, 48, 48).astype(np.float32)),
            "reg": torch.from_numpy(reg),
        }


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


def v2_variants() -> List[V2Variant]:
    pixel_base = "pixel_mlp_no_xy_r0p1_baseline_rerun"
    tiny_base = "tiny_conv_struct_reg_r0p1_baseline_rerun"
    return [
        V2Variant(pixel_base, "pixel_mlp", "pixel_mlp", "no_xy_basic", 0.10, False, False, False, False),
        V2Variant("pixel_mlp_no_xy_r0p1_soft_teacher", "pixel_mlp", "pixel_mlp", "no_xy_basic", 0.10, False, True, False, False),
        V2Variant("pixel_mlp_no_xy_r0p1_postprocess_cc", "pixel_mlp", "pixel_mlp", "no_xy_basic", 0.10, False, False, False, True, pixel_base),
        V2Variant("pixel_mlp_no_xy_r0p1_long_border_penalty_light", "pixel_mlp", "pixel_mlp", "no_xy_basic", 0.10, False, False, True, False),
        V2Variant(tiny_base, "tiny_conv_struct", "tiny_conv", "structure_augmented_no_xy", 0.10, True, False, False, False),
        V2Variant("tiny_conv_struct_reg_r0p1_soft_teacher", "tiny_conv_struct", "tiny_conv", "structure_augmented_no_xy", 0.10, True, True, False, False),
        V2Variant("tiny_conv_struct_reg_r0p1_postprocess_cc", "tiny_conv_struct", "tiny_conv", "structure_augmented_no_xy", 0.10, True, False, False, True, tiny_base),
        V2Variant("tiny_conv_struct_reg_r0p1_long_border_penalty_light", "tiny_conv_struct", "tiny_conv", "structure_augmented_no_xy", 0.10, True, False, True, False),
    ]


def rank_loss(score: torch.Tensor, teacher: torch.Tensor, pairs: int = 256) -> torch.Tensor:
    b, _, h, w = score.shape
    n = h * w
    score_f = score.reshape(b, n)
    teacher_f = teacher.reshape(b, n)
    losses = []
    for i in range(b):
        high = torch.topk(teacher_f[i], k=min(pairs, n // 4), largest=True).indices
        low = torch.topk(teacher_f[i], k=min(pairs, n // 4), largest=False).indices
        k = min(high.numel(), low.numel())
        if k <= 0:
            continue
        diff = score_f[i, high[:k]] - score_f[i, low[:k]]
        losses.append(F.softplus(-diff).mean())
    if not losses:
        return score.mean() * 0.0
    return torch.stack(losses).mean()


def compute_v2_loss(logits: torch.Tensor, batch: Dict[str, torch.Tensor], variant: V2Variant, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    score = torch.sigmoid(logits)
    teacher_score = batch["teacher_score"].to(logits.device)
    teacher_mask = batch["teacher_mask"].to(logits.device)
    reg = batch["reg"].to(logits.device)
    smooth_map = reg[:, 0:1]
    long_map = reg[:, 1:2]
    border_map = reg[:, 2:3]
    center_map = reg[:, 3:4]
    target = teacher_score if variant.soft_teacher else teacher_mask
    teacher_loss = F.binary_cross_entropy_with_logits(logits, target) + F.mse_loss(score, teacher_score)
    rloss = rank_loss(score, teacher_score) if variant.soft_teacher else score.mean() * 0.0
    sparse_loss = torch.abs(score.mean() - float(variant.target_ratio))
    tv_loss = total_variation(score)
    long_loss = (score * long_map).mean()
    border_loss = (score * border_map).mean()
    smooth_region_loss = (score * smooth_map).mean()
    center_loss = torch.relu((score * center_map).mean() / (score.mean() + 1e-6) - 0.42)
    lambda_long = float(args.lambda_long_light if variant.long_border_penalty else args.lambda_long)
    lambda_border = float(args.lambda_border_light if variant.long_border_penalty else args.lambda_border)
    structure_mult = 1.0 if variant.regularized or variant.long_border_penalty else 0.0
    loss = (
        float(args.lambda_teacher) * teacher_loss
        + float(args.lambda_rank) * rloss
        + float(args.lambda_sparse) * sparse_loss
        + float(args.lambda_smooth) * tv_loss
        + lambda_long * long_loss * structure_mult
        + lambda_border * border_loss * structure_mult
        + float(args.lambda_smooth_region) * smooth_region_loss * structure_mult
        + float(args.lambda_center) * center_loss
    )
    logs = {
        "loss": float(loss.detach().cpu()),
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "rank_loss": float(rloss.detach().cpu()),
        "sparsity_loss": float(sparse_loss.detach().cpu()),
        "smoothness_loss": float(tv_loss.detach().cpu()),
        "long_loss": float(long_loss.detach().cpu()),
        "border_loss": float(border_loss.detach().cpu()),
        "smooth_region_loss": float(smooth_region_loss.detach().cpu()),
        "center_loss": float(center_loss.detach().cpu()),
        "score_mean": float(score.mean().detach().cpu()),
        "score_std": float(score.std().detach().cpu()),
        "score_entropy": float((-(score * torch.log(score + 1e-6) + (1 - score) * torch.log(1 - score + 1e-6))).mean().detach().cpu()),
    }
    return loss, logs


def train_variant(variant: V2Variant, train_records: Sequence[Dict[str, Any]], val_records: Sequence[Dict[str, Any]], slic_fn: Any, args: argparse.Namespace, device: torch.device, output_dir: Path) -> Tuple[torch.nn.Module, List[Dict[str, Any]]]:
    input_dim = len(input_feature_names(variant.input_variant))
    model = build_model(variant.arch, input_dim).to(device)
    train_ds = V2Dataset(train_records, variant.input_variant, variant.target_ratio, slic_fn, args)
    val_ds = V2Dataset(val_records, variant.input_variant, variant.target_ratio, slic_fn, args)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rows: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_acc: Dict[str, List[float]] = defaultdict(list)
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(batch["x"].to(device))
            loss, logs = compute_v2_loss(logits, batch, variant, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in logs.items():
                train_acc[k].append(v)
        model.eval()
        val_acc: Dict[str, List[float]] = defaultdict(list)
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["x"].to(device))
                _, logs = compute_v2_loss(logits, batch, variant, args)
                for k, v in logs.items():
                    val_acc[k].append(v)
        row = {
            "experiment": variant.name,
            "epoch": epoch,
            "arch": variant.arch,
            "input_variant": variant.input_variant,
            "target_ratio": variant.target_ratio,
            "soft_teacher": int(variant.soft_teacher),
            "long_border_penalty": int(variant.long_border_penalty),
            "postprocess_cc": int(variant.postprocess_cc),
            "train_loss": float(np.mean(train_acc["loss"])),
            "val_loss": float(np.mean(val_acc["loss"])),
            "val_teacher_loss": float(np.mean(val_acc["teacher_loss"])),
            "val_score_mean": float(np.mean(val_acc["score_mean"])),
            "val_score_std": float(np.mean(val_acc["score_std"])),
            "val_score_entropy": float(np.mean(val_acc["score_entropy"])),
        }
        rows.append(row)
        val_key = row["val_teacher_loss"]
        if val_key < best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    torch.save({"model_state": model.state_dict(), "variant": variant.__dict__, "input_dim": input_dim}, ckpt_dir / f"{safe_name(variant.name)}.pt")
    return model, rows


def predict_score(model: torch.nn.Module, record: Dict[str, Any], input_variant: str, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(feature_stack(record, input_variant)[None]).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy().reshape(-1)
    return normalize01(score).astype(np.float32)


def components_4(mask_2d: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask_2d, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < 48 and 0 <= nx < 48 and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pts)
    out: List[np.ndarray] = []
    for pts in comps:
        c = np.zeros_like(mask, dtype=bool)
        for y, x in pts:
            c[y, x] = True
        out.append(c)
    return out


def postprocess_mask(score: np.ndarray, ratio: float, min_size: int = 5, keep_top: int = 8) -> np.ndarray:
    n = int(score.size)
    k = max(1, int(round(n * float(ratio))))
    raw = topk_mask(score, k).reshape(48, 48)
    comps = components_4(raw)
    scored: List[Tuple[float, int, np.ndarray]] = []
    for comp in comps:
        size = int(comp.sum())
        if size < int(min_size):
            continue
        scored.append((float(score.reshape(48, 48)[comp].mean()), size, comp))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = np.zeros((48, 48), dtype=bool)
    for _, _, comp in scored[: int(keep_top)]:
        selected |= comp
    if int(selected.sum()) < max(1, int(0.70 * k)):
        selected = raw.copy()
    idx_keep = np.flatnonzero(selected.reshape(-1))
    final = np.zeros(n, dtype=bool)
    if idx_keep.size >= k:
        vals = score[idx_keep]
        chosen = idx_keep[np.argpartition(vals, idx_keep.size - k)[idx_keep.size - k :]]
        final[chosen] = True
    else:
        final[idx_keep] = True
        missing = k - int(final.sum())
        if missing > 0:
            rest = np.flatnonzero(~final)
            vals = score[rest]
            chosen = rest[np.argpartition(vals, rest.size - missing)[rest.size - missing :]]
            final[chosen] = True
    return final


def local_mean_apply(image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32).reshape(-1)
    fill = local_mean_fill(img)
    out = img.copy()
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if mode == "only_selected":
        out[~m] = fill[~m]
    elif mode == "delete_selected":
        out[m] = fill[m]
    return out.astype(np.float32)


def make_source_mask(source: str, source_type: str, variant: Optional[V2Variant], models: Dict[str, torch.nn.Module], record: Dict[str, Any], ratio: float, args: argparse.Namespace, slic_fn: Any, device: torch.device) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    teacher_score, _, _ = build_teacher(record, "main_hybrid", ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
    if source_type == "control":
        mask, score, region_count = build_control_mask(record, source, ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma, args.seed)
        return mask, score, region_count, teacher_score
    assert variant is not None
    base_name = variant.base_variant_name if variant.postprocess_cc else variant.name
    model = models[base_name]
    score = predict_score(model, record, variant.input_variant, device)
    if variant.postprocess_cc:
        mask = postprocess_mask(score, ratio, args.min_component_size, args.keep_top_components)
    else:
        mask = topk_mask(score, max(1, int(round(score.size * float(ratio)))))
    return mask, score, 0, teacher_score


def eval_sources(variants: Sequence[V2Variant], models: Dict[str, torch.nn.Module], train_records: Sequence[Dict[str, Any]], eval_records: Sequence[Dict[str, Any]], y_train: np.ndarray, y_eval: np.ndarray, args: argparse.Namespace, slic_fn: Any, device: torch.device, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sources: List[Tuple[str, str, Optional[V2Variant]]] = [(c, "control", None) for c in CONTROLS]
    sources.extend((v.name, "learned", v) for v in variants)
    x_train_original, _, _ = make_original_dataset(train_records)
    x_eval_original, _, _ = make_original_dataset(eval_records)
    original_probe = train_probe(x_train_original, y_train, args.seed, args.classifier_max_iter)
    original_metrics, original_cm = eval_probe(original_probe, x_eval_original, y_eval)
    metric_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, int, str], int] = defaultdict(int)
    save_confusion(output_dir / "confusion_matrices" / "fixed_original_probe" / "original.csv", original_cm)
    for ratio in [float(r) for r in args.retention_ratios]:
        metric_rows.append({"protocol": "fixed_original_probe", "source_name": "original", "source_type": "control", "retention_ratio": ratio, "eval_mode": "original", **original_metrics})
    for source_name, source_type, variant in sources:
        for ratio in [float(r) for r in args.retention_ratios]:
            x_train_only: List[np.ndarray] = []
            x_train_delete: List[np.ndarray] = []
            for record in train_records:
                mask, _, _, _ = make_source_mask(source_name, source_type, variant, models, record, ratio, args, slic_fn, device)
                x_train_only.append(local_mean_apply(record["intensity"], mask, "only_selected"))
                x_train_delete.append(local_mean_apply(record["intensity"], mask, "delete_selected"))
            x_eval_only: List[np.ndarray] = []
            x_eval_delete: List[np.ndarray] = []
            stat_rows: List[Dict[str, Any]] = []
            sample_cache: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
            for record in eval_records:
                mask, score, region_count, teacher_score = make_source_mask(source_name, source_type, variant, models, record, ratio, args, slic_fn, device)
                x_eval_only.append(local_mean_apply(record["intensity"], mask, "only_selected"))
                x_eval_delete.append(local_mean_apply(record["intensity"], mask, "delete_selected"))
                _, comp, coord = mask_structure_stats(record, source_type, source_name, ratio, mask, region_count, variant.input_variant if variant else "")
                corr = score_correlations(record, score)
                teacher_sim = float(np.corrcoef(score.reshape(-1), teacher_score.reshape(-1))[0, 1]) if np.std(score) > 1e-8 and np.std(teacher_score) > 1e-8 else 0.0
                comp.update(corr)
                comp["teacher_similarity"] = teacher_sim
                coord.update(corr)
                coord["teacher_similarity"] = teacher_sim
                stat_rows.append(comp)
                component_rows.append(comp)
                coordinate_rows.append(coord)
                structure_rows.append(comp)
                if source_type == "learned":
                    sample_cache.append((record, mask, score, teacher_score))
            agg = aggregate_stats(stat_rows)
            only_probe = train_probe(np.stack(x_train_only), y_train, args.seed, args.classifier_max_iter)
            for protocol, probe in (("fixed_original_probe", original_probe), ("fixed_only_selected_probe", only_probe)):
                for mode, x_eval in (("only_selected", np.stack(x_eval_only)), ("delete_selected", np.stack(x_eval_delete))):
                    metrics, cm = eval_probe(probe, x_eval, y_eval)
                    metric_rows.append({"protocol": protocol, "source_name": source_name, "source_type": source_type, "retention_ratio": ratio, "eval_mode": mode, **metrics, **agg})
                    save_confusion(output_dir / "confusion_matrices" / protocol / safe_name(source_name) / f"{ratio_name(ratio)}_{mode}.csv", cm)
            if source_type == "learned":
                save_visuals_for_source(output_dir, source_name, ratio, sample_cache, args, visual_counts, visual_rows)
    return metric_rows, component_rows, coordinate_rows, structure_rows, visual_rows


def aggregate_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    fields = [
        "border_ratio",
        "center_ratio",
        "connected_components",
        "largest_component_ratio",
        "selected_long_contour_ratio",
        "selected_smooth_region_ratio",
        "selected_short_structure_ratio",
        "selected_orientation_variation_mean",
        "corr_center_prior",
        "corr_grad_mag",
        "corr_delta_edge_node",
        "teacher_similarity",
    ]
    out: Dict[str, float] = {}
    for field in fields:
        vals = [float(r.get(field, 0.0)) for r in rows]
        out[f"mean_{field}"] = float(np.mean(vals)) if vals else 0.0
    return out


def save_visual_grid(path: Path, record: Dict[str, Any], mask: np.ndarray, score: np.ndarray, teacher_score: np.ndarray) -> None:
    st = record["structure"]
    original = to_u8(record["intensity"].reshape(48, 48))
    mask2 = mask.reshape(48, 48)
    overlay = np.stack([original, original, original], axis=-1).astype(np.float32)
    overlay[mask2, 0] = 255.0
    overlay[mask2, 1:] *= 0.45
    only = to_u8(local_mean_apply(record["intensity"], mask, "only_selected").reshape(48, 48))
    delete = to_u8(local_mean_apply(record["intensity"], mask, "delete_selected").reshape(48, 48))
    grays = [
        original,
        to_u8(score.reshape(48, 48)),
        to_u8(teacher_score.reshape(48, 48)),
        mask2.astype(np.uint8) * 255,
        only,
        delete,
        to_u8(normalize01(record["grad_mag"]).reshape(48, 48)),
        to_u8(normalize01(record["delta_edge_node"]).reshape(48, 48)),
        to_u8(st["long_contour_map"].reshape(48, 48)),
        to_u8(st["short_structure_map"].reshape(48, 48)),
    ]
    parts = [np.stack([g, g, g], axis=-1) for g in grays]
    parts.insert(4, np.clip(overlay, 0, 255).astype(np.uint8))
    sep = np.full((48, 4, 3), 255, dtype=np.uint8)
    grid = parts[0]
    for part in parts[1:]:
        grid = np.concatenate([grid, sep, part], axis=1)
    save_png(path, grid)


def risk_flags(row: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if float(row.get("border_ratio", 0.0)) > 0.38:
        flags.append("high_border")
    if float(row.get("center_ratio", 0.0)) > 0.30:
        flags.append("high_center")
    if float(row.get("selected_long_contour_ratio", 0.0)) > 0.28:
        flags.append("high_long_contour")
    if float(row.get("connected_components", 0.0)) > 25:
        flags.append("high_fragmentation")
    if float(row.get("teacher_similarity", 1.0)) < 0.50:
        flags.append("low_teacher_similarity")
    if "high_border" in flags and "high_long_contour" in flags:
        flags.append("possible_background")
        flags.append("possible_hair_or_glasses")
    return flags


def save_visuals_for_source(output_dir: Path, source_name: str, ratio: float, samples: Sequence[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], args: argparse.Namespace, counts: Dict[Tuple[str, int, str], int], visual_rows: List[Dict[str, Any]]) -> None:
    stat_samples: List[Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = []
    for record, mask, score, teacher in samples:
        _, comp, _ = mask_structure_stats(record, "learned", source_name, ratio, mask, 0, "")
        comp["teacher_similarity"] = float(np.corrcoef(score.reshape(-1), teacher.reshape(-1))[0, 1]) if np.std(score) > 1e-8 and np.std(teacher) > 1e-8 else 0.0
        risk_score = float(comp["border_ratio"]) + float(comp["selected_long_contour_ratio"]) + min(1.0, float(comp["connected_components"]) / 30.0)
        stat_samples.append((risk_score, record, mask, score, teacher, comp))
    high = sorted(stat_samples, key=lambda x: x[0], reverse=True)
    best = sorted(stat_samples, key=lambda x: x[0])
    random = stat_samples[:: max(1, len(stat_samples) // 50)]
    pools = [("high_risk", high, args.high_risk_samples_per_class), ("best_looking", best, args.high_risk_samples_per_class), ("random", random, args.figure_samples_per_class)]
    for bucket, pool, limit in pools:
        for _, record, mask, score, teacher, comp in pool:
            key = (source_name, int(record["label"]), bucket)
            if counts[key] >= int(limit):
                continue
            class_dir = f"class_{int(record['label'])}_{safe_name(class_name(record['label']))}"
            base = ensure_dir(output_dir / "figures" / safe_name(source_name) / bucket / ratio_name(ratio) / class_dir)
            mask_dir = ensure_dir(output_dir / "masks" / safe_name(source_name) / ratio_name(ratio) / class_dir)
            stem = f"{record['split']}_graph_{int(record['graph_id'])}"
            image_path = base / f"{stem}_comparison.png"
            overlay_path = mask_dir / f"{stem}_mask.png"
            save_visual_grid(image_path, record, mask, score, teacher)
            save_png(overlay_path, mask.reshape(48, 48).astype(np.uint8) * 255)
            flags = risk_flags(comp)
            visual_rows.append(
                {
                    "experiment": source_name,
                    "split": record["split"],
                    "graph_id": int(record["graph_id"]),
                    "label": int(record["label"]),
                    "class_name": class_name(record["label"]),
                    "ratio": ratio,
                    "bucket": bucket,
                    "image_path": str(image_path),
                    "overlay_path": str(overlay_path),
                    "risk_flags": ";".join(flags),
                }
            )
            counts[key] += 1


def compare_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(r["protocol"], r["source_name"], float(r["retention_ratio"]), r["eval_mode"]): r for r in metric_rows}
    rows: List[Dict[str, Any]] = []
    for r in metric_rows:
        if r["eval_mode"] != "only_selected":
            continue
        protocol = r["protocol"]
        ratio = float(r["retention_ratio"])
        random_row = lookup.get((protocol, "random_pixel", ratio, "only_selected"), {})
        center_row = lookup.get((protocol, "center_prior", ratio, "only_selected"), {})
        hybrid_row = lookup.get((protocol, "main_hybrid_teacher", ratio, "only_selected"), {})
        delete_row = lookup.get((protocol, r["source_name"], ratio, "delete_selected"), {})
        original = lookup.get(("fixed_original_probe", "original", ratio, "original"), {})
        base = float(original.get("macro_f1", "nan")) if protocol == "fixed_original_probe" else float(r["macro_f1"])
        delete_f1 = float(delete_row.get("macro_f1", "nan"))
        rows.append(
            {
                "protocol": protocol,
                "source_name": r["source_name"],
                "source_type": r["source_type"],
                "retention_ratio": ratio,
                "only_selected_macro_f1": r["macro_f1"],
                "delete_selected_macro_f1": delete_f1,
                "deletion_drop": base - delete_f1 if math.isfinite(base) and math.isfinite(delete_f1) else "nan",
                "gap_vs_random": float(r["macro_f1"]) - float(random_row.get("macro_f1", "nan")),
                "gap_vs_center": float(r["macro_f1"]) - float(center_row.get("macro_f1", "nan")),
                "gap_vs_main_hybrid_teacher": float(r["macro_f1"]) - float(hybrid_row.get("macro_f1", "nan")),
                "border_ratio": r.get("mean_border_ratio", ""),
                "center_ratio": r.get("mean_center_ratio", ""),
                "components": r.get("mean_connected_components", ""),
                "largest_component_ratio": r.get("mean_largest_component_ratio", ""),
                "long_contour_ratio": r.get("mean_selected_long_contour_ratio", ""),
                "smooth_region_ratio": r.get("mean_selected_smooth_region_ratio", ""),
                "short_structure_ratio": r.get("mean_selected_short_structure_ratio", ""),
                "orientation_variation_mean": r.get("mean_selected_orientation_variation_mean", ""),
                "corr_center_prior": r.get("mean_corr_center_prior", ""),
                "corr_grad": r.get("mean_corr_grad_mag", ""),
                "corr_delta": r.get("mean_corr_delta_edge_node", ""),
                "teacher_similarity": r.get("mean_teacher_similarity", ""),
            }
        )
    return rows


def per_class_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in metric_rows:
        for name in EMOTION_NAMES:
            rows.append(
                {
                    "protocol": r["protocol"],
                    "source_name": r["source_name"],
                    "source_type": r["source_type"],
                    "retention_ratio": r["retention_ratio"],
                    "eval_mode": r["eval_mode"],
                    "class_name": name,
                    "f1": r.get(f"per_class_f1_{name}", 0.0),
                }
            )
    return rows


def best_row(rows: Sequence[Dict[str, Any]], protocol: str, ratio: float) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if r["source_type"] == "learned" and r["protocol"] == protocol and abs(float(r["retention_ratio"]) - ratio) < 1e-9]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["only_selected_macro_f1"]))


def write_report(output_dir: Path, rows: Sequence[Dict[str, Any]], visual_rows: Sequence[Dict[str, Any]]) -> str:
    fo20 = best_row(rows, "fixed_original_probe", 0.20)
    fos20 = best_row(rows, "fixed_only_selected_probe", 0.20)
    fos10 = best_row(rows, "fixed_only_selected_probe", 0.10)
    ready = False
    if fo20 and fos20:
        ready = (
            (float(fo20["deletion_drop"]) >= 0.02 or float(fo20["gap_vs_random"]) >= 0.02)
            and float(fos20["only_selected_macro_f1"]) >= 0.26
            and float(fo20["gap_vs_center"]) > 0
            and float(fo20["components"]) < 18
            and float(fo20["long_contour_ratio"]) < 0.20
            and 0.20 <= float(fo20["center_ratio"]) <= 0.28
        )
    verdict = "READY_FOR_STAGE_5_DIAGNOSTIC" if ready else "STILL_NOT_READY_REFINE_OR_FREEZE"
    lines: List[str] = [
        "# Stage 4 Refinement v2 Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Verdict: **{verdict}**.",
        "- Scope: Stage 4 only. No motif, no SupCon, no part grouping.",
    ]
    if fo20:
        lines.append(f"- Best fixed-original @20: `{fo20['source_name']}` F1={fmt(fo20['only_selected_macro_f1'])}, drop={fmt(fo20['deletion_drop'])}, components={fmt(fo20['components'])}, long={fmt(fo20['long_contour_ratio'])}.")
    if fos20:
        lines.append(f"- Best fixed-only-selected @20: `{fos20['source_name']}` F1={fmt(fos20['only_selected_macro_f1'])}, drop={fmt(fos20['deletion_drop'])}.")
    lines.append(f"- Visual audit images indexed: `{len(visual_rows)}`.")
    lines.extend(
        [
            "",
            "## 2. Experiment Scope",
            "",
            "Only 8 small variants were run: pixel MLP baseline/soft/postprocess/penalty and tiny-conv structure baseline/soft/postprocess/penalty.",
            "",
            "## 3. Fixed-original Deletion Results",
            "",
            "| Ratio | Variant | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap hybrid | Border | Center | Components | Long | Smooth | Short |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in (0.10, 0.20):
        for r in sorted([x for x in rows if x["protocol"] == "fixed_original_probe" and x["source_type"] == "learned" and abs(float(x["retention_ratio"]) - ratio) < 1e-9], key=lambda x: float(x["only_selected_macro_f1"]), reverse=True):
            lines.append(f"| {fmt(ratio)} | {r['source_name']} | {fmt(r['only_selected_macro_f1'])} | {fmt(r['delete_selected_macro_f1'])} | {fmt(r['deletion_drop'])} | {fmt(r['gap_vs_random'])} | {fmt(r['gap_vs_center'])} | {fmt(r['gap_vs_main_hybrid_teacher'])} | {fmt(r['border_ratio'])} | {fmt(r['center_ratio'])} | {fmt(r['components'])} | {fmt(r['long_contour_ratio'])} | {fmt(r['smooth_region_ratio'])} | {fmt(r['short_structure_ratio'])} |")
    lines.extend(
        [
            "",
            "## 4. Fixed-only-selected Results",
            "",
            "| Ratio | Variant | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap hybrid | Border | Center | Components | Long | Smooth | Short |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in (0.10, 0.20):
        for r in sorted([x for x in rows if x["protocol"] == "fixed_only_selected_probe" and x["source_type"] == "learned" and abs(float(x["retention_ratio"]) - ratio) < 1e-9], key=lambda x: float(x["only_selected_macro_f1"]), reverse=True):
            lines.append(f"| {fmt(ratio)} | {r['source_name']} | {fmt(r['only_selected_macro_f1'])} | {fmt(r['delete_selected_macro_f1'])} | {fmt(r['deletion_drop'])} | {fmt(r['gap_vs_random'])} | {fmt(r['gap_vs_center'])} | {fmt(r['gap_vs_main_hybrid_teacher'])} | {fmt(r['border_ratio'])} | {fmt(r['center_ratio'])} | {fmt(r['components'])} | {fmt(r['long_contour_ratio'])} | {fmt(r['smooth_region_ratio'])} | {fmt(r['short_structure_ratio'])} |")
    lines.extend(
        [
            "",
            "## 5. Retention vs Structure Trade-off",
            "",
            "- PASS requires fixed-original @20 to improve deletion/gap while reducing components below 18 and long contour below 0.20.",
            "- If postprocess improves components but hurts fixed-original F1/drop, it remains diagnostic only.",
            "- Soft teacher is useful only if it lowers fragmentation without losing retention.",
            "",
            "## 6. Candidate Comparison",
            "",
            "See CSV `refinement_v2_vs_controls.csv` for exact rows. The decision is based on fixed-original first, fixed-only-selected second.",
            "",
            "## 7. Visual Audit Summary",
            "",
            f"- Generated/indexed visual audit rows: `{len(visual_rows)}`.",
            "- Risk flags include high_border, high_center, high_long_contour, high_fragmentation, low_teacher_similarity, possible_background, possible_hair_or_glasses.",
            "- Manual review is still required before any future part grouping diagnostic.",
            "",
            "## 8. Decision",
            "",
        ]
    )
    if verdict.startswith("READY"):
        lines.append("A. READY_FOR_STAGE_5_DIAGNOSTIC, but only as diagnostic part grouping. No motif/SupCon.")
    else:
        lines.append("B. STILL_NOT_READY_REFINE_OR_FREEZE. Do not start Stage 5.")
    lines.extend(
        [
            "",
            "## 9. Next Step",
            "",
            "- If NOT READY: freeze learned selector or do one final very narrow refinement. Do not open a larger grid.",
            "- If STOP is chosen later: use hybrid/structure heuristic as selector baseline.",
            "",
            "## 10. What Not To Claim",
            "",
            "- Không motif.",
            "- Không causal nếu deletion chưa đủ.",
            "- Không semantic part.",
            "- Không Q1 claim.",
            "",
        ]
    )
    (output_dir / "stage4_refinement_v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage4_dir", default="outputs/stage4_learned_evidence_selector")
    parser.add_argument("--output_dir", default="outputs/stage4_learned_evidence_selector/refinement_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--probe_train_cap", type=int, default=2000)
    parser.add_argument("--probe_eval_cap", type=int, default=1000)
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=[0.10, 0.20])
    parser.add_argument("--figure_samples_per_class", type=int, default=5)
    parser.add_argument("--high_risk_samples_per_class", type=int, default=3)
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--min_component_size", type=int, default=5)
    parser.add_argument("--keep_top_components", type=int, default=8)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda_teacher", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.05)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.02)
    parser.add_argument("--lambda_long", type=float, default=0.02)
    parser.add_argument("--lambda_border", type=float, default=0.02)
    parser.add_argument("--lambda_long_light", type=float, default=0.02)
    parser.add_argument("--lambda_border_light", type=float, default=0.02)
    parser.add_argument("--lambda_smooth_region", type=float, default=0.01)
    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4V2] SLIC unavailable: {slic_error}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"[Stage4V2] device={device}")
    print("[Stage4V2] loading records")
    train_records = load_split_records(reader, resolver, "train", int(args.max_train_samples))
    val_records = load_split_records(reader, resolver, "val", int(args.max_val_samples))
    probe_train = train_records[: min(len(train_records), int(args.probe_train_cap))]
    probe_eval = val_records[: min(len(val_records), int(args.probe_eval_cap))]
    _, y_train, _ = make_original_dataset(probe_train)
    _, y_eval, _ = make_original_dataset(probe_eval)
    variants = v2_variants()
    models: Dict[str, torch.nn.Module] = {}
    train_rows: List[Dict[str, Any]] = []
    print("[Stage4V2] training trainable variants")
    for variant in variants:
        if variant.postprocess_cc:
            continue
        print(f"[Stage4V2] train {variant.name}")
        model, rows = train_variant(variant, train_records, val_records, slic_fn, args, device, output_dir)
        models[variant.name] = model
        train_rows.extend(rows)
    print("[Stage4V2] evaluating")
    metric_rows, comp_rows, coord_rows, struct_rows, visual_rows = eval_sources(variants, models, probe_train, probe_eval, y_train, y_eval, args, slic_fn, device, output_dir)
    compare = compare_rows(metric_rows)
    per_class = per_class_rows(metric_rows)
    write_csv(output_dir / "refinement_v2_train_log.csv", train_rows)
    write_csv(output_dir / "refinement_v2_probe_metrics.csv", metric_rows)
    write_csv(output_dir / "refinement_v2_vs_controls.csv", compare)
    write_csv(output_dir / "refinement_v2_component_stats.csv", comp_rows)
    write_csv(output_dir / "refinement_v2_coordinate_stats.csv", coord_rows)
    write_csv(output_dir / "refinement_v2_structure_stats.csv", struct_rows)
    write_csv(output_dir / "refinement_v2_per_class_f1.csv", per_class)
    write_csv(output_dir / "refinement_v2_visual_audit_index.csv", visual_rows)
    verdict = write_report(output_dir, compare, visual_rows)
    fo20 = best_row(compare, "fixed_original_probe", 0.20)
    fos20 = best_row(compare, "fixed_only_selected_probe", 0.20)
    print(f"[Stage4V2] output_dir={output_dir}")
    print(f"[Stage4V2] verdict={verdict}")
    if fo20:
        print(f"[Stage4V2] fixed_original_best20={fo20['source_name']} drop={fmt(fo20['deletion_drop'])} f1={fmt(fo20['only_selected_macro_f1'])}")
        print(f"[Stage4V2] structure20 components={fmt(fo20['components'])} long={fmt(fo20['long_contour_ratio'])} center={fmt(fo20['center_ratio'])} border={fmt(fo20['border_ratio'])}")
    if fos20:
        print(f"[Stage4V2] fixed_only_best20={fos20['source_name']} f1={fmt(fos20['only_selected_macro_f1'])} drop={fmt(fos20['deletion_drop'])}")
    print("[Stage4V2] stage5=no" if not verdict.startswith("READY") else "[Stage4V2] stage5=diagnostic_only")


if __name__ == "__main__":
    main()
