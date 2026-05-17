"""Stage 4 learned evidence selector v0.

This stage trains a tiny pixel evidence selector against deterministic hybrid
teacher targets. It does not train D12/D13, does not build motif banks, does not
enable SupCon, does not add global branches, and does not mutate graph
artifacts. Evaluation uses the same lightweight retention/deletion probe family
as Stage 2/3.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    DEFAULT_SELECTORS as STAGE1_SELECTORS,
    center_scores,
    connected_component_stats,
    ensure_dir,
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
from stage3_hybrid_evidence_selector import WEIGHT_GRID as STAGE3_WEIGHT_GRID, build_hybrid_mask, train_eval_fast
from stage36_structure_aware_diagnostics import (
    BASE_VARIANTS as STRUCTURE_BASE_VARIANTS,
    build_structure_mask,
    load_split_records,
    mask_structure_stats,
    selected_map_mean,
)


DEFAULT_RATIOS = (0.05, 0.10, 0.20, 0.40)
CONTROL_SELECTORS = (
    "random_pixel",
    "center_prior",
    "gradient_topk",
    "delta_edge_topk",
    "contrast_topk",
    "slic_region_proposal",
    "main_hybrid_teacher",
    "structure_aux_teacher",
)
TEACHER_VARIANTS = {
    "main_hybrid": "hybrid_pixel_score__E_balanced__b0p0",
    "sparse_hybrid": "hybrid_pixel_smooth__C_delta_grad__b0p1",
    "slic_hybrid": "hybrid_slic_region__E_balanced__b0p1",
}
STRUCTURE_TEACHER_VARIANTS = {
    "best_structure": "structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1",
    "structure_sparse": "structure_pixel_smooth__A_delta_grad__s0p2_o0p1_sm0p2_l0p4_b0p1",
    "structure_pixel": "structure_pixel_score__C_contrast_assisted__s0p2_o0p1_sm0p2_l0p2_b0p1",
}


@dataclass
class ExperimentConfig:
    name: str
    selector_arch: str
    input_variant: str
    teacher: str
    target_ratio: float
    use_structure_regularizers: bool


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


def parse_hybrid_variant(variant_id: str) -> Tuple[str, str, Tuple[float, float, float], float]:
    parts = variant_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Invalid hybrid variant: {variant_id}")
    selector = parts[0]
    weight_id = parts[1]
    border = float(parts[2][1:].replace("p", "."))
    return selector, weight_id, STAGE3_WEIGHT_GRID[weight_id], border


def parse_structure_variant(variant_id: str) -> Tuple[str, str, Tuple[float, float, float], Dict[str, float]]:
    parts = variant_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Invalid structure variant: {variant_id}")
    selector = parts[0]
    base_variant = parts[1]
    cfg_text = parts[2]
    cfg: Dict[str, float] = {}
    for token in cfg_text.split("_"):
        if token.startswith("sm"):
            cfg["w_smooth"] = float(token[2:].replace("p", "."))
        elif token.startswith("s"):
            cfg["w_short"] = float(token[1:].replace("p", "."))
        elif token.startswith("o"):
            cfg["w_orient"] = float(token[1:].replace("p", "."))
        elif token.startswith("l"):
            cfg["w_long"] = float(token[1:].replace("p", "."))
        elif token.startswith("b"):
            cfg["w_border"] = float(token[1:].replace("p", "."))
    for key in ("w_short", "w_orient", "w_smooth", "w_long", "w_border"):
        cfg.setdefault(key, 0.0)
    return selector, base_variant, STRUCTURE_BASE_VARIANTS[base_variant], cfg


def input_feature_names(input_variant: str) -> Tuple[str, ...]:
    if input_variant == "no_xy_basic":
        return ("intensity", "grad_mag", "local_contrast", "delta_edge_node")
    if input_variant == "with_xy":
        return ("intensity", "x_norm", "y_norm", "grad_mag", "local_contrast", "delta_edge_node")
    if input_variant == "structure_augmented_no_xy":
        return (
            "intensity",
            "grad_mag",
            "local_contrast",
            "delta_edge_node",
            "smooth_region_score",
            "long_contour_score",
            "short_local_structure_score",
            "orientation_variation",
            "border_penalty",
        )
    if input_variant == "structure_augmented_with_xy":
        return (
            "intensity",
            "x_norm",
            "y_norm",
            "grad_mag",
            "local_contrast",
            "delta_edge_node",
            "smooth_region_score",
            "long_contour_score",
            "short_local_structure_score",
            "orientation_variation",
            "border_penalty",
        )
    raise ValueError(f"Unknown input_variant: {input_variant}")


def feature_stack(record: Dict[str, Any], input_variant: str) -> np.ndarray:
    st = record["structure"]
    values: Dict[str, np.ndarray] = {
        "intensity": record["intensity"],
        "x_norm": record["x_norm"],
        "y_norm": record["y_norm"],
        "gx": record["gx"],
        "gy": record["gy"],
        "grad_mag": normalize01(record["grad_mag"]),
        "local_contrast": normalize01(record["local_contrast"]),
        "delta_edge_node": normalize01(record["delta_edge_node"]),
        "smooth_region_score": st["smooth_region_map"],
        "long_contour_score": st["long_contour_map"],
        "short_local_structure_score": st["short_structure_map"],
        "orientation_variation": st["orientation_variation_map"],
        "border_penalty": st["border_penalty_map"],
    }
    maps = [np.asarray(values[name], dtype=np.float32).reshape(48, 48) for name in input_feature_names(input_variant)]
    return np.stack(maps, axis=0).astype(np.float32)


def center_prior_map(record: Dict[str, Any]) -> np.ndarray:
    return normalize01(center_scores(record["x_norm"], record["y_norm"])).astype(np.float32)


def build_teacher(record: Dict[str, Any], teacher: str, target_ratio: float, slic_fn: Any, slic_segments: int, slic_compactness: float, smooth_sigma: float) -> Tuple[np.ndarray, np.ndarray, str]:
    variant = TEACHER_VARIANTS.get(teacher, teacher)
    if variant in STRUCTURE_TEACHER_VARIANTS:
        variant = STRUCTURE_TEACHER_VARIANTS[variant]
    if variant.startswith("hybrid_"):
        selector, _, weights, w_border = parse_hybrid_variant(variant)
        mask, score, _ = build_hybrid_mask(record, selector, weights, w_border, target_ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        return normalize01(score).astype(np.float32), mask.astype(np.float32), variant
    if variant.startswith("structure_"):
        selector, _, weights, cfg = parse_structure_variant(variant)
        mask, score, _, _ = build_structure_mask(record, selector, weights, cfg, target_ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        return normalize01(score).astype(np.float32), mask.astype(np.float32), variant
    raise ValueError(f"Unknown teacher: {teacher}")


class EvidenceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        input_variant: str,
        teacher: str,
        target_ratio: float,
        slic_fn: Any,
        slic_segments: int,
        slic_compactness: float,
        smooth_sigma: float,
    ) -> None:
        self.records = list(records)
        self.input_variant = input_variant
        self.teacher = teacher
        self.target_ratio = float(target_ratio)
        self.slic_fn = slic_fn
        self.slic_segments = int(slic_segments)
        self.slic_compactness = float(slic_compactness)
        self.smooth_sigma = float(smooth_sigma)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        teacher_score, teacher_mask, _ = build_teacher(record, self.teacher, self.target_ratio, self.slic_fn, self.slic_segments, self.slic_compactness, self.smooth_sigma)
        st = record["structure"]
        x = feature_stack(record, self.input_variant)
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
            "x": torch.from_numpy(x),
            "teacher_score": torch.from_numpy(teacher_score.reshape(1, 48, 48).astype(np.float32)),
            "teacher_mask": torch.from_numpy(teacher_mask.reshape(1, 48, 48).astype(np.float32)),
            "reg": torch.from_numpy(reg),
        }


class PixelMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        out = self.net(z).reshape(b, h, w, 1).permute(0, 3, 1, 2)
        return out


class TinyConvSelector(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(selector_arch: str, input_dim: int) -> nn.Module:
    if selector_arch == "pixel_mlp":
        return PixelMLP(input_dim)
    if selector_arch == "tiny_conv":
        return TinyConvSelector(input_dim)
    raise ValueError(f"Unknown selector_arch: {selector_arch}")


def total_variation(score: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(score[:, :, :, 1:] - score[:, :, :, :-1]).mean()
    dy = torch.abs(score[:, :, 1:, :] - score[:, :, :-1, :]).mean()
    return dx + dy


def compute_loss(logits: torch.Tensor, batch: Dict[str, torch.Tensor], args: argparse.Namespace, target_ratio: float, use_structure_regularizers: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    score = torch.sigmoid(logits)
    teacher_score = batch["teacher_score"].to(logits.device)
    teacher_mask = batch["teacher_mask"].to(logits.device)
    reg = batch["reg"].to(logits.device)
    smooth_map = reg[:, 0:1]
    long_map = reg[:, 1:2]
    border_map = reg[:, 2:3]
    center_map = reg[:, 3:4]
    bce = F.binary_cross_entropy_with_logits(logits, teacher_mask)
    mse = F.mse_loss(score, teacher_score)
    teacher_loss = bce + 0.50 * mse
    sparse_loss = torch.abs(score.mean() - float(target_ratio))
    tv_loss = total_variation(score)
    region_loss = tv_loss
    long_loss = (score * long_map).mean()
    smooth_region_loss = (score * smooth_map).mean()
    border_loss = (score * border_map).mean()
    center_loss = torch.relu((score * center_map).mean() / (score.mean() + 1e-6) - 0.42)
    multiplier = 1.0 if use_structure_regularizers else 0.0
    loss = (
        float(args.lambda_teacher) * teacher_loss
        + float(args.lambda_sparse) * sparse_loss
        + float(args.lambda_smooth) * tv_loss
        + float(args.lambda_region) * region_loss * multiplier
        + float(args.lambda_long) * long_loss * multiplier
        + float(args.lambda_smooth_region) * smooth_region_loss * multiplier
        + float(args.lambda_center) * center_loss
        + float(args.lambda_border) * border_loss * multiplier
    )
    logs = {
        "loss": float(loss.detach().cpu()),
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "sparsity_loss": float(sparse_loss.detach().cpu()),
        "smoothness_loss": float(tv_loss.detach().cpu()),
        "region_loss": float(region_loss.detach().cpu()),
        "long_loss": float(long_loss.detach().cpu()),
        "smooth_region_loss": float(smooth_region_loss.detach().cpu()),
        "center_loss": float(center_loss.detach().cpu()),
        "border_loss": float(border_loss.detach().cpu()),
        "score_mean": float(score.mean().detach().cpu()),
    }
    return loss, logs


def train_one_experiment(
    exp: ExperimentConfig,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    slic_fn: Any,
    output_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    input_dim = len(input_feature_names(exp.input_variant))
    model = build_model(exp.selector_arch, input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_ds = EvidenceDataset(train_records, exp.input_variant, exp.teacher, exp.target_ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
    val_ds = EvidenceDataset(val_records, exp.input_variant, exp.teacher, exp.target_ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = int(args.patience)
    bad_epochs = 0
    rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_acc: Dict[str, List[float]] = defaultdict(list)
        for batch in train_loader:
            x = batch["x"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, logs = compute_loss(logits, batch, args, exp.target_ratio, exp.use_structure_regularizers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in logs.items():
                train_acc[key].append(value)
        model.eval()
        val_acc: Dict[str, List[float]] = defaultdict(list)
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["x"].to(device))
                _, logs = compute_loss(logits, batch, args, exp.target_ratio, exp.use_structure_regularizers)
                for key, value in logs.items():
                    val_acc[key].append(value)
        train_loss = float(np.mean(train_acc["loss"])) if train_acc["loss"] else float("nan")
        val_loss = float(np.mean(val_acc["loss"])) if val_acc["loss"] else float("nan")
        val_teacher = float(np.mean(val_acc["teacher_loss"])) if val_acc["teacher_loss"] else val_loss
        row: Dict[str, Any] = {
            "experiment": exp.name,
            "epoch": epoch,
            "selector_arch": exp.selector_arch,
            "input_variant": exp.input_variant,
            "teacher": exp.teacher,
            "target_ratio": exp.target_ratio,
            "use_structure_regularizers": int(exp.use_structure_regularizers),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_score_mean": float(np.mean(train_acc["score_mean"])) if train_acc["score_mean"] else float("nan"),
            "val_score_mean": float(np.mean(val_acc["score_mean"])) if val_acc["score_mean"] else float("nan"),
            "val_teacher_loss": val_teacher,
            "val_sparsity_loss": float(np.mean(val_acc["sparsity_loss"])) if val_acc["sparsity_loss"] else float("nan"),
            "val_smoothness_loss": float(np.mean(val_acc["smoothness_loss"])) if val_acc["smoothness_loss"] else float("nan"),
            "val_center_loss": float(np.mean(val_acc["center_loss"])) if val_acc["center_loss"] else float("nan"),
            "val_border_loss": float(np.mean(val_acc["border_loss"])) if val_acc["border_loss"] else float("nan"),
        }
        rows.append(row)
        if val_teacher < best_val:
            best_val = val_teacher
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ensure_dir(output_dir / "checkpoints")
    torch.save(
        {
            "model_state": model.state_dict(),
            "experiment": exp.__dict__,
            "input_dim": input_dim,
            "feature_names": input_feature_names(exp.input_variant),
            "best_val_teacher_loss": best_val,
        },
        output_dir / "checkpoints" / f"{safe_name(exp.name)}.pt",
    )
    return model, rows, {"best_val_teacher_loss": best_val, "epochs_ran": len(rows)}


def predict_score(model: nn.Module, record: Dict[str, Any], input_variant: str, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(feature_stack(record, input_variant)[None]).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    return normalize01(score.reshape(-1)).astype(np.float32)


def score_correlations(record: Dict[str, Any], score: np.ndarray) -> Dict[str, float]:
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if float(a.std()) <= 1e-8 or float(b.std()) <= 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "corr_center_prior": corr(score, center_prior_map(record)),
        "corr_grad_mag": corr(score, record["grad_mag"]),
        "corr_delta_edge_node": corr(score, record["delta_edge_node"]),
    }


def save_visual(
    output_dir: Path,
    experiment: str,
    record: Dict[str, Any],
    ratio: float,
    teacher_score: np.ndarray,
    predicted_score: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
) -> None:
    class_dir = f"class_{int(record['label'])}_{safe_name(class_name(record['label']))}"
    base = ensure_dir(output_dir / "figures" / safe_name(experiment) / class_dir)
    mask_dir = ensure_dir(output_dir / "masks" / safe_name(experiment) / ratio_name(ratio) / class_dir)
    stem = f"{record['split']}_graph_{int(record['graph_id'])}_{ratio_name(ratio)}"
    original = to_u8(record["intensity"].reshape(48, 48))
    st = record["structure"]
    mask2 = mask.reshape(48, 48)
    overlay = np.stack([original, original, original], axis=-1).astype(np.float32)
    overlay[mask2, 0] = 255.0
    overlay[mask2, 1:] *= 0.45
    only_selected = to_u8(apply_mask(record["intensity"], mask, "only_selected", fill_mode).reshape(48, 48))
    delete_selected = to_u8(apply_mask(record["intensity"], mask, "delete_selected", fill_mode).reshape(48, 48))
    parts_gray = [
        original,
        original,
        to_u8(normalize01(teacher_score).reshape(48, 48)),
        to_u8(normalize01(predicted_score).reshape(48, 48)),
        mask2.astype(np.uint8) * 255,
        to_u8(normalize01(record["grad_mag"]).reshape(48, 48)),
        to_u8(normalize01(record["delta_edge_node"]).reshape(48, 48)),
        to_u8(st["long_contour_map"].reshape(48, 48)),
        to_u8(st["smooth_region_map"].reshape(48, 48)),
        to_u8(st["short_structure_map"].reshape(48, 48)),
    ]
    rgb_parts = [np.stack([p, p, p], axis=-1) for p in parts_gray]
    rgb_parts.insert(5, np.clip(overlay, 0, 255).astype(np.uint8))
    rgb_parts.extend([np.stack([only_selected] * 3, axis=-1), np.stack([delete_selected] * 3, axis=-1)])
    sep = np.full((48, 4, 3), 255, dtype=np.uint8)
    grid = rgb_parts[0]
    for part in rgb_parts[1:]:
        grid = np.concatenate([grid, sep, part], axis=1)
    save_png(base / f"{stem}_comparison.png", grid)
    save_png(mask_dir / f"{stem}_mask.png", mask2.astype(np.uint8) * 255)


def build_control_mask(
    record: Dict[str, Any],
    control: str,
    ratio: float,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
    smooth_sigma: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if control in STAGE1_SELECTORS:
        for selector, mask, region_count, _ in selector_masks(
            [control],
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
            score = {
                "random_pixel": np.zeros(2304, dtype=np.float32),
                "center_prior": center_prior_map(record),
                "gradient_topk": normalize01(record["grad_mag"]),
                "contrast_topk": normalize01(record["local_contrast"]),
                "delta_edge_topk": normalize01(record["delta_edge_node"]),
                "slic_region_proposal": record["structure"]["good_structure_score_map"],
            }.get(selector, np.zeros(2304, dtype=np.float32))
            return mask, score, region_count
    if control == "main_hybrid_teacher":
        score, _, variant = build_teacher(record, "main_hybrid", ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        selector, _, weights, w_border = parse_hybrid_variant(variant)
        mask, score, region_count = build_hybrid_mask(record, selector, weights, w_border, ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        return mask, score, region_count
    if control == "structure_aux_teacher":
        variant = STRUCTURE_TEACHER_VARIANTS["best_structure"]
        selector, _, weights, cfg = parse_structure_variant(variant)
        mask, score, region_count, _ = build_structure_mask(record, selector, weights, cfg, ratio, slic_fn, slic_segments, slic_compactness, smooth_sigma)
        return mask, score, region_count
    raise ValueError(f"Unknown control: {control}")


def evaluate_mask_source(
    source_name: str,
    source_type: str,
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    ratios: Sequence[float],
    args: argparse.Namespace,
    slic_fn: Any,
    output_dir: Path,
    model: Optional[nn.Module] = None,
    input_variant: str = "",
    teacher_name: str = "main_hybrid",
    device: Optional[torch.device] = None,
    save_figures: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    retention_rows: List[Dict[str, Any]] = []
    deletion_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, int], int] = defaultdict(int)

    def make_mask(record: Dict[str, Any], ratio: float) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        teacher_score, _, _ = build_teacher(record, teacher_name, ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma)
        if source_type == "learned":
            assert model is not None and device is not None
            score = predict_score(model, record, input_variant, device)
            mask = topk_mask(score, max(1, int(round(2304 * float(ratio)))))
            return mask, score, 0, teacher_score
        mask, score, region_count = build_control_mask(record, source_name, ratio, slic_fn, args.slic_segments, args.slic_compactness, args.smooth_sigma, args.seed)
        return mask, score, region_count, teacher_score

    for ratio in ratios:
        x_train_only: List[np.ndarray] = []
        x_train_delete: List[np.ndarray] = []
        for record in train_records:
            mask, _, _, _ = make_mask(record, float(ratio))
            x_train_only.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
            x_train_delete.append(apply_mask(record["intensity"], mask, "delete_selected", args.fill_mode))
        x_eval_only: List[np.ndarray] = []
        x_eval_delete: List[np.ndarray] = []
        eval_stats: List[Dict[str, Any]] = []
        for record in eval_records:
            mask, score, region_count, teacher_score = make_mask(record, float(ratio))
            x_eval_only.append(apply_mask(record["intensity"], mask, "only_selected", args.fill_mode))
            x_eval_delete.append(apply_mask(record["intensity"], mask, "delete_selected", args.fill_mode))
            _, comp, coord = mask_structure_stats(record, source_type, source_name, float(ratio), mask, region_count, input_variant)
            corr = score_correlations(record, score)
            teacher_sim = float(np.corrcoef(score.reshape(-1), teacher_score.reshape(-1))[0, 1]) if np.std(score) > 1e-8 and np.std(teacher_score) > 1e-8 else 0.0
            comp.update(corr)
            comp["teacher_similarity"] = teacher_sim
            coord.update(corr)
            coord["teacher_similarity"] = teacher_sim
            component_rows.append(comp)
            coordinate_rows.append(coord)
            structure_rows.append(comp)
            eval_stats.append(comp)
            if save_figures:
                key = (source_name, int(record["label"]))
                if visual_counts[key] < int(args.figure_samples_per_class):
                    save_visual(output_dir, source_name, record, float(ratio), teacher_score, score, mask, args.fill_mode)
                    visual_counts[key] += 1
        metrics, cm = train_eval_fast(
            np.stack(x_train_only, axis=0).astype(np.float32),
            y_train,
            np.stack(x_eval_only, axis=0).astype(np.float32),
            y_eval,
            args.seed,
            args.classifier_max_iter,
        )
        del_metrics, del_cm = train_eval_fast(
            np.stack(x_train_delete, axis=0).astype(np.float32),
            y_train,
            np.stack(x_eval_delete, axis=0).astype(np.float32),
            y_eval,
            args.seed,
            args.classifier_max_iter,
        )
        agg = aggregate_stats(eval_stats)
        row = {
            "experiment": source_name,
            "source_type": source_type,
            "retention_ratio": float(ratio),
            **metrics,
            **agg,
        }
        del_row = {
            "experiment": source_name,
            "source_type": source_type,
            "retention_ratio": float(ratio),
            "delete_selected_accuracy": del_metrics["only_selected_accuracy"],
            "delete_selected_macro_f1": del_metrics["only_selected_macro_f1"],
            "delete_selected_weighted_f1": del_metrics["only_selected_weighted_f1"],
            **{f"delete_per_class_f1_{name}": del_metrics.get(f"per_class_f1_{name}", 0.0) for name in EMOTION_NAMES},
        }
        retention_rows.append(row)
        deletion_rows.append(del_row)
        save_confusion(output_dir / "confusion_matrices" / safe_name(source_name) / f"{ratio_name(ratio)}_only_selected.csv", cm)
        save_confusion(output_dir / "confusion_matrices" / safe_name(source_name) / f"{ratio_name(ratio)}_delete_selected.csv", del_cm)
    return retention_rows, deletion_rows, component_rows, coordinate_rows, structure_rows


def aggregate_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    fields = [
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


def build_experiments(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.experiment_suite == "single":
        return [
            ExperimentConfig(
                name=f"{args.selector_arch}__{args.input_variant}__{args.teacher}__r{str(args.target_ratio).replace('.', 'p')}",
                selector_arch=args.selector_arch,
                input_variant=args.input_variant,
                teacher=args.teacher,
                target_ratio=float(args.target_ratio),
                use_structure_regularizers=bool_arg(args.use_structure_regularizers),
            )
        ]
    return [
        ExperimentConfig("tiny_conv__no_xy_basic__main_hybrid__r0p10", "tiny_conv", "no_xy_basic", "main_hybrid", 0.10, False),
        ExperimentConfig("tiny_conv__with_xy__main_hybrid__r0p10", "tiny_conv", "with_xy", "main_hybrid", 0.10, False),
        ExperimentConfig("tiny_conv__structure_augmented_no_xy__main_hybrid__r0p10", "tiny_conv", "structure_augmented_no_xy", "main_hybrid", 0.10, False),
        ExperimentConfig("tiny_conv__structure_augmented_no_xy__main_hybrid__r0p20", "tiny_conv", "structure_augmented_no_xy", "main_hybrid", 0.20, False),
        ExperimentConfig("tiny_conv__structure_augmented_no_xy__main_hybrid__regularized__r0p10", "tiny_conv", "structure_augmented_no_xy", "main_hybrid", 0.10, True),
        ExperimentConfig("pixel_mlp__no_xy_basic__main_hybrid__r0p10", "pixel_mlp", "no_xy_basic", "main_hybrid", 0.10, False),
    ]


def add_gaps(retention_rows: List[Dict[str, Any]], deletion_rows: List[Dict[str, Any]], original_macro_f1: float) -> List[Dict[str, Any]]:
    lookup = {(r["experiment"], float(r["retention_ratio"])): r for r in retention_rows}
    deletion_lookup = {(r["experiment"], float(r["retention_ratio"])): r for r in deletion_rows}
    rows: List[Dict[str, Any]] = []
    for row in retention_rows:
        exp = row["experiment"]
        ratio = float(row["retention_ratio"])
        random_row = lookup.get(("random_pixel", ratio), {})
        center_row = lookup.get(("center_prior", ratio), {})
        hybrid_row = lookup.get(("main_hybrid_teacher", ratio), {})
        structure_row = lookup.get(("structure_aux_teacher", ratio), {})
        delete_row = deletion_lookup.get((exp, ratio), {})
        delete_f1 = float(delete_row.get("delete_selected_macro_f1", "nan"))
        rows.append(
            {
                "experiment": exp,
                "source_type": row.get("source_type", ""),
                "retention_ratio": ratio,
                "only_selected_macro_f1": row["only_selected_macro_f1"],
                "random_macro_f1": random_row.get("only_selected_macro_f1", ""),
                "center_macro_f1": center_row.get("only_selected_macro_f1", ""),
                "main_hybrid_teacher_macro_f1": hybrid_row.get("only_selected_macro_f1", ""),
                "structure_teacher_macro_f1": structure_row.get("only_selected_macro_f1", ""),
                "gap_vs_random": float(row["only_selected_macro_f1"]) - float(random_row.get("only_selected_macro_f1", "nan")),
                "gap_vs_center": float(row["only_selected_macro_f1"]) - float(center_row.get("only_selected_macro_f1", "nan")),
                "gap_vs_main_hybrid_teacher": float(row["only_selected_macro_f1"]) - float(hybrid_row.get("only_selected_macro_f1", "nan")),
                "gap_vs_structure_teacher": float(row["only_selected_macro_f1"]) - float(structure_row.get("only_selected_macro_f1", "nan")),
                "delete_selected_macro_f1": delete_f1,
                "deletion_drop_vs_original": original_macro_f1 - delete_f1 if math.isfinite(delete_f1) else "nan",
                "border_ratio": row.get("mean_border_ratio", ""),
                "center_ratio": row.get("mean_center_ratio", ""),
                "components": row.get("mean_connected_components", ""),
                "long_contour_ratio": row.get("mean_selected_long_contour_ratio", ""),
                "smooth_region_ratio": row.get("mean_selected_smooth_region_ratio", ""),
                "short_structure_ratio": row.get("mean_selected_short_structure_ratio", ""),
            }
        )
    return rows


def per_class_rows(retention_rows: Sequence[Dict[str, Any]], deletion_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    del_lookup = {(r["experiment"], float(r["retention_ratio"])): r for r in deletion_rows}
    rows: List[Dict[str, Any]] = []
    for row in retention_rows:
        drow = del_lookup.get((row["experiment"], float(row["retention_ratio"])), {})
        for name in EMOTION_NAMES:
            rows.append(
                {
                    "experiment": row["experiment"],
                    "source_type": row.get("source_type", ""),
                    "retention_ratio": row["retention_ratio"],
                    "class_name": name,
                    "only_selected_f1": row.get(f"per_class_f1_{name}", 0.0),
                    "delete_selected_f1": drow.get(f"delete_per_class_f1_{name}", 0.0),
                }
            )
    return rows


def best_learned_by_ratio(rows: Sequence[Dict[str, Any]], ratios: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    learned = [r for r in rows if r.get("source_type") == "learned"]
    for ratio in ratios:
        candidates = [r for r in learned if abs(float(r["retention_ratio"]) - float(ratio)) < 1e-9]
        if candidates:
            out[float(ratio)] = max(candidates, key=lambda r: float(r["only_selected_macro_f1"]))
    return out


def write_report(
    output_dir: Path,
    train_rows: Sequence[Dict[str, Any]],
    retention_rows: Sequence[Dict[str, Any]],
    deletion_rows: Sequence[Dict[str, Any]],
    vs_rows: Sequence[Dict[str, Any]],
    ratios: Sequence[float],
    experiments: Sequence[ExperimentConfig],
    args: argparse.Namespace,
) -> str:
    best = best_learned_by_ratio(retention_rows, ratios)
    learned_vs = [r for r in vs_rows if r.get("source_type") == "learned"]
    positive_random = [r for r in learned_vs if float(r.get("gap_vs_random", -999)) > 0.01]
    positive_center = [r for r in learned_vs if float(r.get("gap_vs_center", -999)) > 0.01]
    pass_candidates: List[str] = []
    for exp_name in sorted({str(r["experiment"]) for r in learned_vs}):
        exp_rows = {float(r["retention_ratio"]): r for r in learned_vs if str(r["experiment"]) == exp_name}
        r10 = exp_rows.get(0.10)
        r20 = exp_rows.get(0.20)
        if not r10 or not r20:
            continue
        if (
            float(r10.get("gap_vs_random", -999)) > 0.01
            and float(r20.get("gap_vs_random", -999)) > 0.01
            and float(r10.get("gap_vs_center", -999)) > 0.01
            and float(r20.get("gap_vs_center", -999)) > 0.01
            and float(r10.get("gap_vs_main_hybrid_teacher", -999)) >= 0.0
            and float(r20.get("gap_vs_main_hybrid_teacher", -999)) >= 0.0
            and float(r10.get("deletion_drop_vs_original", -999)) > 0.01
            and float(r20.get("deletion_drop_vs_original", -999)) > 0.01
            and float(r10.get("center_ratio", 1.0)) < 0.35
            and float(r20.get("center_ratio", 1.0)) < 0.35
        ):
            pass_candidates.append(exp_name)
    verdict = "FAIL"
    if positive_random and positive_center:
        verdict = "PARTIAL"
    if pass_candidates:
        verdict = "PASS"
    lines: List[str] = [
        "# Stage 4 Learned Evidence Selector v0 Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Stage 4 verdict: **{verdict}**.",
        f"- Experiments trained: `{len(experiments)}`.",
        "- Learned selector is evaluated as evidence candidate only; not motif, not causal proof.",
    ]
    for ratio, row in best.items():
        lines.append(
            f"- Best learned @{int(ratio * 100)}%: `{row['experiment']}` macro_f1={fmt(row['only_selected_macro_f1'])}, "
            f"border={fmt(row.get('mean_border_ratio'))}, center={fmt(row.get('mean_center_ratio'))}, components={fmt(row.get('mean_connected_components'))}."
        )
    lines.extend(
        [
            "",
            "## 2. Setup",
            "",
            f"- Selector suite: `{args.experiment_suite}`.",
            f"- Epochs max: `{args.epochs}`, batch size `{args.batch_size}`.",
            f"- Train/val caps: `{args.max_train_samples}` / `{args.max_val_samples}`.",
            f"- Probe train/eval caps: `{args.probe_train_cap}` / `{args.probe_eval_cap}`.",
            f"- Retention ratios: `{', '.join(str(r) for r in ratios)}`.",
            "- Controls: random, center, gradient, delta, contrast, SLIC, main hybrid teacher, structure-aware teacher.",
            "",
            "| Experiment | Arch | Input | Teacher | Target ratio | Structure regularizers |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for exp in experiments:
        lines.append(f"| {exp.name} | {exp.selector_arch} | {exp.input_variant} | {exp.teacher} | {fmt(exp.target_ratio)} | {int(exp.use_structure_regularizers)} |")
    lines.extend(
        [
            "",
            "## 3. Training Behavior",
            "",
            "| Experiment | Last epoch | Train loss | Val loss | Val teacher loss | Val score mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_exp[str(row["experiment"])].append(row)
    for exp_name, rows in by_exp.items():
        last = rows[-1]
        lines.append(
            f"| {exp_name} | {last['epoch']} | {fmt(last['train_loss'])} | {fmt(last['val_loss'])} | "
            f"{fmt(last['val_teacher_loss'])} | {fmt(last['val_score_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. Retention Results",
            "",
            "| Experiment | Ratio | Macro F1 | Gap random | Gap center | Gap hybrid teacher | Border | Center | Components | Long | Smooth | Short |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in learned_vs:
        lines.append(
            f"| {row['experiment']} | {fmt(row['retention_ratio'])} | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row['gap_vs_random'])} | {fmt(row['gap_vs_center'])} | {fmt(row['gap_vs_main_hybrid_teacher'])} | "
            f"{fmt(row['border_ratio'])} | {fmt(row['center_ratio'])} | {fmt(row['components'])} | "
            f"{fmt(row['long_contour_ratio'])} | {fmt(row['smooth_region_ratio'])} | {fmt(row['short_structure_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Deletion Results",
            "",
            "- Deletion is reported as a diagnostic, not causal proof. If drop is weak, do not claim causal evidence.",
            "",
            "| Experiment | Ratio | Delete macro F1 | Drop vs original |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in [r for r in vs_rows if r.get("source_type") == "learned"]:
        lines.append(f"| {row['experiment']} | {fmt(row['retention_ratio'])} | {fmt(row['delete_selected_macro_f1'])} | {fmt(row['deletion_drop_vs_original'])} |")
    lines.extend(
        [
            "",
            "## 6. Structure & Shortcut Analysis",
            "",
            "- Center shortcut is checked through center ratio and correlation with center prior.",
            "- Border/background shortcut is checked through border ratio and long-contour ratio.",
            "- Fragmentation is checked through connected component count and largest component ratio.",
            "- Structure regularizers are intentionally weak; they should not be interpreted as hard rules.",
            "",
            "## 7. Per-class Analysis",
            "",
            "- Per-class rows are written to `per_class_learned_selector_metrics.csv`.",
            "- If Happy/Surprise remain strong while Angry/Sad/Neutral follow center controls, Stage 5 must keep class-wise diagnostics.",
            "",
            "## 8. Visual Review",
            "",
            "- Figures include original, teacher score, predicted score, mask, overlay, only-selected, delete-selected, grad, delta, long/smooth/short maps.",
            "- Do not cherry-pick. Review both high-F1 and high-risk samples.",
            "- Do not call selected regions motifs.",
            "",
            "## 9. Comparison with Stage 3 and Stage 3.6",
            "",
            "- `learned_selector_vs_controls.csv` contains gaps vs random, center, main hybrid teacher, and structure teacher.",
            "- If learned selector is below teacher but has lower fragmentation/long-contour risk, it is a trade-off, not a strict win.",
            "",
            "## 10. Decision",
            "",
        ]
    )
    if verdict == "PASS":
        lines.append("A. Stage 4 PASS: learned selector is stable enough to start Stage 5 part grouping, still without SupCon/motif bank.")
    elif verdict == "PARTIAL":
        lines.append("B. Stage 4 PARTIAL: learned selector has retention signal but deletion/structure evidence is not strong enough for a full Stage 5 claim yet.")
    else:
        lines.append("C. Stage 4 FAIL: return to hybrid/structure heuristic and do not proceed.")
    lines.extend(
        [
            "",
            "## 11. Stage 5 Recommendation nếu có",
            "",
            "- Stage 5 should not start as motif learning yet.",
            "- If attempted, it must be a narrow part-grouping diagnostic from selected masks/components/SLIC regions only.",
            "- No SupCon/motif bank unless a follow-up confirms deletion and visual stability.",
            "",
            "## 12. What Not To Claim",
            "",
            "- Không claim motif.",
            "- Không claim causal nếu deletion yếu.",
            "- Không claim semantic facial part nếu chưa có validation.",
            "- Không claim Q1 contribution.",
            "- Không claim selector đã hiểu mắt/miệng nếu chỉ có F1 probe.",
            "",
        ]
    )
    (output_dir / "stage4_learned_selector_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage3_dir", default="outputs/stage3_hybrid_evidence_selector")
    parser.add_argument("--stage36_dir", default="outputs/stage36_structure_aware_diagnostics")
    parser.add_argument("--output_dir", default="outputs/stage4_learned_evidence_selector")
    parser.add_argument("--selector_arch", choices=["pixel_mlp", "tiny_conv"], default="tiny_conv")
    parser.add_argument("--input_variant", choices=["no_xy_basic", "with_xy", "structure_augmented_no_xy", "structure_augmented_with_xy"], default="structure_augmented_no_xy")
    parser.add_argument("--teacher", default="main_hybrid")
    parser.add_argument("--target_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--probe_train_cap", type=int, default=2000)
    parser.add_argument("--probe_eval_cap", type=int, default=1000)
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--figure_samples_per_class", type=int, default=5)
    parser.add_argument("--experiment_suite", choices=["minimal", "single"], default="minimal")
    parser.add_argument("--use_structure_regularizers", type=bool_arg, default=True)
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--fill_mode", choices=["mean", "zero"], default="mean")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    parser.add_argument("--lambda_teacher", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.02)
    parser.add_argument("--lambda_region", type=float, default=0.02)
    parser.add_argument("--lambda_long", type=float, default=0.02)
    parser.add_argument("--lambda_smooth_region", type=float, default=0.01)
    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--lambda_border", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "checkpoints")
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4] SLIC unavailable: {slic_error}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Stage4] device={device}")
    print("[Stage4] loading records")
    train_records_full = load_split_records(reader, resolver, "train", int(args.max_train_samples))
    val_records_full = load_split_records(reader, resolver, "val", int(args.max_val_samples))
    train_records = train_records_full
    val_records = val_records_full
    probe_train_records = train_records_full[: min(len(train_records_full), int(args.probe_train_cap))]
    probe_eval_records = val_records_full[: min(len(val_records_full), int(args.probe_eval_cap))]
    x_train_orig, y_train_probe, _ = make_original_dataset(probe_train_records)
    x_eval_orig, y_eval_probe, _ = make_original_dataset(probe_eval_records)
    original_metrics, original_cm = train_eval_fast(x_train_orig, y_train_probe, x_eval_orig, y_eval_probe, args.seed, args.classifier_max_iter)
    save_confusion(output_dir / "confusion_matrices" / "original.csv", original_cm)

    experiments = build_experiments(args)
    train_log_rows: List[Dict[str, Any]] = []
    retention_rows: List[Dict[str, Any]] = []
    deletion_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []

    print("[Stage4] evaluating controls")
    for control in CONTROL_SELECTORS:
        r_rows, d_rows, c_rows, co_rows, s_rows = evaluate_mask_source(
            control,
            "control",
            probe_train_records,
            probe_eval_records,
            y_train_probe,
            y_eval_probe,
            [float(r) for r in args.retention_ratios],
            args,
            slic_fn,
            output_dir,
            teacher_name=args.teacher,
            save_figures=True,
        )
        retention_rows.extend(r_rows)
        deletion_rows.extend(d_rows)
        component_rows.extend(c_rows)
        coordinate_rows.extend(co_rows)
        structure_rows.extend(s_rows)

    for exp in experiments:
        print(f"[Stage4] training {exp.name}")
        model, logs, meta = train_one_experiment(exp, train_records, val_records, args, slic_fn, output_dir, device)
        for row in logs:
            row.update(meta)
        train_log_rows.extend(logs)
        print(f"[Stage4] evaluating {exp.name}")
        r_rows, d_rows, c_rows, co_rows, s_rows = evaluate_mask_source(
            exp.name,
            "learned",
            probe_train_records,
            probe_eval_records,
            y_train_probe,
            y_eval_probe,
            [float(r) for r in args.retention_ratios],
            args,
            slic_fn,
            output_dir,
            model=model,
            input_variant=exp.input_variant,
            teacher_name=exp.teacher,
            device=device,
            save_figures=True,
        )
        retention_rows.extend(r_rows)
        deletion_rows.extend(d_rows)
        component_rows.extend(c_rows)
        coordinate_rows.extend(co_rows)
        structure_rows.extend(s_rows)

    vs_rows = add_gaps(retention_rows, deletion_rows, float(original_metrics["only_selected_macro_f1"]))
    per_class = per_class_rows(retention_rows, deletion_rows)
    write_csv(output_dir / "learned_selector_train_log.csv", train_log_rows)
    write_csv(output_dir / "learned_selector_eval_metrics.csv", retention_rows)
    write_csv(output_dir / "learned_selector_retention_metrics.csv", retention_rows)
    write_csv(output_dir / "learned_selector_deletion_metrics.csv", deletion_rows)
    write_csv(output_dir / "learned_selector_vs_controls.csv", vs_rows)
    write_csv(output_dir / "learned_selector_component_stats.csv", component_rows)
    write_csv(output_dir / "learned_selector_coordinate_stats.csv", coordinate_rows)
    write_csv(output_dir / "learned_selector_structure_stats.csv", structure_rows)
    write_csv(output_dir / "per_class_learned_selector_metrics.csv", per_class)
    verdict = write_report(output_dir, train_log_rows, retention_rows, deletion_rows, vs_rows, [float(r) for r in args.retention_ratios], experiments, args)
    best = best_learned_by_ratio(retention_rows, [0.05, 0.10, 0.20])
    print(f"[Stage4] output_dir={output_dir}")
    print(f"[Stage4] verdict={verdict}")
    for ratio, row in best.items():
        print(f"[Stage4] best_{int(ratio * 100)}={row['experiment']} macro_f1={fmt(row['only_selected_macro_f1'])}")
    learned_vs = [r for r in vs_rows if r.get("source_type") == "learned"]
    if learned_vs:
        best_any = max(learned_vs, key=lambda r: float(r["only_selected_macro_f1"]))
        print(f"[Stage4] learned_vs_random={fmt(best_any['gap_vs_random'])}")
        print(f"[Stage4] learned_vs_center={fmt(best_any['gap_vs_center'])}")
        print(f"[Stage4] learned_vs_main_hybrid_teacher={fmt(best_any['gap_vs_main_hybrid_teacher'])}")
        print(
            "[Stage4] mask_stats "
            f"center={fmt(best_any['center_ratio'])} "
            f"border={fmt(best_any['border_ratio'])} "
            f"components={fmt(best_any['components'])} "
            f"long={fmt(best_any['long_contour_ratio'])} "
            f"smooth={fmt(best_any['smooth_region_ratio'])}"
        )
    print("[Stage4] next_step=analyze report before Stage5; do not claim motif")


if __name__ == "__main__":
    main()
