"""Visualize MotifDiscoveryModule assignment maps and audit CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_existing_path, resolve_path, resolve_device  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402
from utils.motif_audit import (  # noqa: E402
    audit_motif_outputs,
    compute_border_center_mass,
    compute_clean_candidate_scores,
    compute_motif_area,
    compute_outer_border_mass,
    compute_soft_region_masses,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _normalize_image(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _plot_maps_grid(maps: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> None:
    num_motifs = maps.shape[0]
    cols = int(math.ceil(math.sqrt(num_motifs)))
    rows = int(math.ceil(num_motifs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(np.nanmax(maps)), 1e-8)
    for ax in axes:
        ax.axis("off")
    for idx in range(num_motifs):
        axes[idx].imshow(maps[idx], cmap="magma", vmin=0.0, vmax=vmax)
        axes[idx].set_title(f"k={idx} s={scores[idx]:.2f}", fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overlay_grid(image: np.ndarray, maps: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> None:
    num_motifs = maps.shape[0]
    cols = int(math.ceil(math.sqrt(num_motifs)))
    rows = int(math.ceil(num_motifs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(np.nanmax(maps)), 1e-8)
    for ax in axes:
        ax.axis("off")
    for idx in range(num_motifs):
        axes[idx].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].imshow(maps[idx], cmap="magma", alpha=0.62, vmin=0.0, vmax=vmax)
        axes[idx].set_title(f"k={idx} s={scores[idx]:.2f}", fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_top_maps(image: np.ndarray, maps: np.ndarray, scores: np.ndarray, out_path: Path, title: str, top_k: int = 6) -> None:
    order = np.argsort(scores)[-int(top_k) :][::-1]
    fig, axes = plt.subplots(len(order), 3, figsize=(7.2, 2.0 * len(order)))
    axes = np.asarray(axes).reshape(len(order), 3)
    vmax = max(float(np.nanmax(maps[order])), 1e-8)
    for row, motif_idx in enumerate(order):
        axes[row, 0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title("image", fontsize=8)
        axes[row, 1].imshow(maps[motif_idx], cmap="magma", vmin=0.0, vmax=vmax)
        axes[row, 1].set_title(f"motif {motif_idx}", fontsize=8)
        axes[row, 2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].imshow(maps[motif_idx], cmap="magma", alpha=0.65, vmin=0.0, vmax=vmax)
        axes[row, 2].set_title(f"score {scores[motif_idx]:.3f}", fontsize=8)
        for col in range(3):
            axes[row, col].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_selected_overlay(
    image: np.ndarray,
    maps: np.ndarray,
    selection_weights: np.ndarray,
    out_path: Path,
    title: str,
    top_m: int = 8,
) -> None:
    order = np.argsort(selection_weights)[-int(top_m) :][::-1]
    weighted = (maps[order] * selection_weights[order, None, None]).sum(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].axis("off")
    axes[0].set_title("image", fontsize=8)
    axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(weighted, cmap="magma", alpha=0.68, vmin=0.0, vmax=max(float(np.nanmax(weighted)), 1e-8))
    axes[1].axis("off")
    label = ", ".join(f"{idx}:{selection_weights[idx]:.2f}" for idx in order[: min(4, len(order))])
    axes[1].set_title(f"selected {label}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_clean_overlay(
    image: np.ndarray,
    maps: np.ndarray,
    clean_scores: np.ndarray,
    out_path: Path,
    title: str,
    top_m: int = 6,
) -> None:
    order = np.argsort(clean_scores)[-int(top_m) :][::-1]
    weighted = (maps[order] * clean_scores[order, None, None]).sum(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].axis("off")
    axes[0].set_title("image", fontsize=8)
    axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(weighted, cmap="viridis", alpha=0.68, vmin=0.0, vmax=max(float(np.nanmax(weighted)), 1e-8))
    axes[1].axis("off")
    label = ", ".join(f"{idx}:{clean_scores[idx]:.2f}" for idx in order[: min(4, len(order))])
    axes[1].set_title(f"clean {label}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_rows(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_visualize(
    config,
    split: str = "val",
    max_samples: int = 8,
    max_batches: int | None = None,
    checkpoint: str | Path | None = None,
    tag: str | None = None,
) -> Path:
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config=config)
    model_cfg = dict(config["model"])
    print(f"[Device] selected={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[Device] gpu={torch.cuda.get_device_name(device.index or torch.cuda.current_device())}")
    feature_ablation_cfg = dict(config.get("feature_ablation", {}) or {})
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 7)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
        prefix="[FeatureAblation visualize]",
    )
    model = build_model(model_cfg).to(device)
    if checkpoint is not None:
        ckpt_path = resolve_existing_path(checkpoint)
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded motif discovery checkpoint: {ckpt_path}")
    model.eval()
    loader = build_dataloader(config, split=split, shuffle=False)
    paths = config.get("paths", {})
    output_root = resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    figure_name = "motif_discovery" if not tag else f"motif_discovery_{str(tag).strip()}"
    out_dir = _ensure_dir(output_root / "figures" / figure_name)

    summary_rows: list[dict] = []
    pair_rows: list[dict] = []
    selected_rows: list[dict] = []
    clean_rows: list[dict] = []
    region_clean_rows: list[dict] = []
    saved = 0
    height = int(config.get("model", {}).get("image_size", 48))
    width = int(config.get("model", {}).get("image_size", 48))

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if saved >= int(max_samples):
            break
        batch = move_to_device(batch, device)
        batch = apply_feature_ablation(batch, feature_ablation_cfg)
        assert_feature_dims(
            batch,
            node_dim=int(model_cfg.get("node_dim", 7)),
            edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
        out = model(batch)
        maps = out["motif_assignment_maps"]
        scores = out["motif_scores"]
        centers = out["motif_centers"]
        audit = out["motif_audit"]
        pair_map = audit["pairwise_map_sim"]
        pair_emb = audit["pairwise_emb_sim"]
        pair_dist = audit["pairwise_center_dist"]
        bsz, num_motifs = maps.shape[:2]
        model_cfg = dict(config.get("model", {}) or {})
        audit_cfg = dict(config.get("audit", {}) or {})
        loss_cfg = dict(config.get("motif_loss", config.get("motif", {}).get("loss", {})) or {})
        criterion = MotifDiscoveryStage1Loss(
            {
                **loss_cfg,
                "height": int(model_cfg.get("image_size", 48)),
                "width": int(model_cfg.get("image_size", 48)),
            }
        ).to(device)
        loss_metrics = criterion(out, batch)
        map_threshold = float(audit_cfg.get("redundant_map_threshold", model_cfg.get("map_sim_threshold", 0.80)))
        emb_threshold = float(audit_cfg.get("redundant_emb_threshold", model_cfg.get("emb_sim_threshold", 0.80)))
        center_threshold = float(audit_cfg.get("redundant_center_threshold_norm", model_cfg.get("center_distance_threshold", 0.15)))
        audit_min_area = float(audit_cfg.get("min_effective_area", loss_cfg.get("min_effective_area", model_cfg.get("min_effective_area", 40))))
        audit_max_area = float(audit_cfg.get("max_effective_area", loss_cfg.get("max_effective_area", model_cfg.get("max_effective_area", 400))))

        for i in range(bsz):
            if saved >= int(max_samples):
                break
            graph_id = int(batch["graph_id"][i].detach().cpu())
            sample_maps = _np(maps[i])
            sample_scores = _np(scores[i])
            if "selection_weights" in out:
                sample_selection = _np(out["selection_weights"][i])
            else:
                sample_selection = _np(torch.softmax(scores[i], dim=0))
            image = _normalize_image(_np(batch["x"][i, :, 0]).reshape(height, width))
            sample_audit = audit_motif_outputs(
                maps[i : i + 1],
                out["motif_embeddings"][i : i + 1],
                motif_centers=centers[i : i + 1],
                border_width=int(model_cfg.get("border_width", 4)),
                map_sim_threshold=map_threshold,
                emb_sim_threshold=emb_threshold,
                center_distance_threshold=center_threshold,
                min_effective_area=audit_min_area,
                max_effective_area=audit_max_area,
                outer_border_width=int(loss_cfg.get("outer_border_width", model_cfg.get("border_width", 4))),
            )
            row = {
                "sample_idx": saved,
                "graph_id": graph_id,
                "mean_pairwise_map_sim": float(sample_audit["mean_pairwise_map_sim"].cpu()),
                "max_pairwise_map_sim": float(sample_audit["max_pairwise_map_sim"].cpu()),
                "mean_pairwise_emb_sim": float(sample_audit["mean_pairwise_emb_sim"].cpu()),
                "max_pairwise_emb_sim": float(sample_audit["max_pairwise_emb_sim"].cpu()),
                "mean_center_dist": float(sample_audit["mean_center_dist"].cpu()),
                "min_center_dist": float(sample_audit["min_center_dist"].cpu()),
                "redundant_pair_count": float(sample_audit["redundant_pair_count"].cpu()),
                "redundant_pair_ratio": float(sample_audit["redundant_pair_ratio"].cpu()),
                "border_mass_mean": float(sample_audit["border_mass_mean"].cpu()),
                "outer_border_mass_mean": float(sample_audit["outer_border_mass_mean"].cpu()),
                "center_mass_mean": float(sample_audit["center_mass_mean"].cpu()),
                "center_minus_border": float(sample_audit["center_minus_border"].cpu()),
                "border_dominance": float(sample_audit["border_dominance"].cpu()),
                "selected_border_mass": float(loss_metrics["selected_border_mass_mean"].detach().cpu()),
                "selected_outer_border_mass": float(loss_metrics["selected_outer_border_mass_mean"].detach().cpu()),
                "selected_foreground_mass": float(loss_metrics["selected_foreground_mass_mean"].detach().cpu()),
                "selection_entropy": float(loss_metrics["selection_entropy"].detach().cpu()),
                "selection_effective_count": float(loss_metrics["selection_effective_count"].detach().cpu()),
                "selected_top_m": float(loss_metrics["selected_top_m"].detach().cpu()),
                "effective_area_mean": float(sample_audit["effective_area_mean"].cpu()),
                "effective_area_min": float(sample_audit["effective_area_min"].cpu()),
                "effective_area_max": float(sample_audit["effective_area_max"].cpu()),
                "area_in_range_ratio": float(sample_audit["area_in_range_ratio"].cpu()),
                "area_over_max_ratio": float(sample_audit["area_over_max_ratio"].cpu()),
                "area_under_min_ratio": float(sample_audit["area_under_min_ratio"].cpu()),
                "effective_motif_count": float(sample_audit["effective_motif_count"].cpu()),
                "effective_motif_ratio": float(sample_audit["effective_motif_ratio"].cpu()),
            }
            summary_rows.append(row)

            sample_border, _ = compute_border_center_mass(maps[i : i + 1], border_width=int(model_cfg.get("border_width", 4)))
            sample_outer = compute_outer_border_mass(
                maps[i : i + 1],
                outer_border_width=int(loss_cfg.get("outer_border_width", model_cfg.get("border_width", 4))),
            )
            foreground = criterion._foreground_prior(
                batch={key: value[i : i + 1] if torch.is_tensor(value) and value.shape[:1] == batch["x"].shape[:1] else value for key, value in batch.items()},
                bsz=1,
                height=height,
                width=width,
                device=maps.device,
                dtype=maps.dtype,
            )
            foreground_anchor = foreground / foreground.max(dim=1, keepdim=True).values.clamp_min(criterion.eps)
            sample_foreground = (maps[i : i + 1].flatten(2) * foreground_anchor.unsqueeze(1)).sum(dim=2)
            sample_area = compute_motif_area(maps[i : i + 1])
            clean = compute_clean_candidate_scores(
                border_mass=sample_border,
                outer_border_mass=sample_outer,
                foreground_mass=sample_foreground,
                effective_area=sample_area,
                clean_border_threshold=float(loss_cfg.get("clean_border_threshold", 0.30)),
                clean_outer_threshold=float(loss_cfg.get("clean_outer_threshold", 0.40)),
                clean_foreground_threshold=float(loss_cfg.get("clean_foreground_threshold", 0.25)),
                clean_tau=float(loss_cfg.get("clean_tau", 0.05)),
                clean_area_tau=float(loss_cfg.get("clean_area_tau", 50.0)),
                clean_min_effective_area=float(loss_cfg.get("clean_min_effective_area", audit_min_area)),
                clean_max_effective_area=float(loss_cfg.get("clean_max_effective_area", audit_cfg.get("max_effective_area", 512))),
            )
            sample_clean = _np(clean["clean_score"][0])
            region = compute_soft_region_masses(
                maps[i : i + 1],
                region_tau=float(loss_cfg.get("region_tau", 0.05)),
            )
            sample_upper = _np(region["upper_mass"][0])
            sample_middle = _np(region["middle_mass"][0])
            sample_lower = _np(region["lower_mass"][0])
            region_stack = np.stack([sample_upper, sample_middle, sample_lower], axis=1)
            dominant_names = np.asarray(["upper", "middle", "lower"])
            upper_clean = float((clean["clean_score"] * region["upper_mass"]).sum(dim=1).detach().cpu())
            middle_clean = float((clean["clean_score"] * region["middle_mass"]).sum(dim=1).detach().cpu())
            lower_clean = float((clean["clean_score"] * region["lower_mass"]).sum(dim=1).detach().cpu())
            selection_tensor = torch.as_tensor(sample_selection, device=maps.device, dtype=maps.dtype).view(1, -1)
            row["selected_border_mass"] = float((selection_tensor * sample_border).sum(dim=1).detach().cpu())
            row["selected_outer_border_mass"] = float((selection_tensor * sample_outer).sum(dim=1).detach().cpu())
            row["selected_foreground_mass"] = float((selection_tensor * sample_foreground).sum(dim=1).detach().cpu())
            row["selection_entropy"] = float(
                (-(selection_tensor * selection_tensor.clamp_min(1e-8).log()).sum(dim=1)).detach().cpu()
            )
            row["selection_effective_count"] = float(np.exp(row["selection_entropy"]))
            row["selected_top_m"] = float(config.get("model", {}).get("selection_top_m", 0))
            row["clean_score_mean"] = float(clean["clean_score"].mean().detach().cpu())
            row["clean_score_max"] = float(clean["clean_score"].max().detach().cpu())
            row["clean_candidate_count"] = float(clean["clean_candidate_count"].detach().cpu())
            row["hard_clean_candidate_count"] = float(clean["hard_clean_candidate_count"].detach().cpu())
            row["upper_clean_count"] = upper_clean
            row["middle_clean_count"] = middle_clean
            row["lower_clean_count"] = lower_clean
            row["min_region_clean_count"] = min(upper_clean, middle_clean, lower_clean)
            row["mean_region_clean_count"] = (upper_clean + middle_clean + lower_clean) / 3.0
            for motif_idx in range(num_motifs):
                dominant_region = str(dominant_names[int(np.argmax(region_stack[motif_idx]))])
                region_clean_rows.append(
                    {
                        "sample_idx": saved,
                        "graph_id": graph_id,
                        "motif_id": motif_idx,
                        "clean_score": float(clean["clean_score"][0, motif_idx].detach().cpu()),
                        "upper_mass": float(sample_upper[motif_idx]),
                        "middle_mass": float(sample_middle[motif_idx]),
                        "lower_mass": float(sample_lower[motif_idx]),
                        "dominant_region": dominant_region,
                        "selection_weight": float(sample_selection[motif_idx]),
                    }
                )
                clean_rows.append(
                    {
                        "sample_idx": saved,
                        "graph_id": graph_id,
                        "motif_id": motif_idx,
                        "clean_score": float(clean["clean_score"][0, motif_idx].detach().cpu()),
                        "clean_border_score": float(clean["clean_border_score"][0, motif_idx].detach().cpu()),
                        "clean_outer_score": float(clean["clean_outer_score"][0, motif_idx].detach().cpu()),
                        "clean_foreground_score": float(clean["clean_foreground_score"][0, motif_idx].detach().cpu()),
                        "clean_area_low_score": float(clean["clean_area_low_score"][0, motif_idx].detach().cpu()),
                        "clean_area_high_score": float(clean["clean_area_high_score"][0, motif_idx].detach().cpu()),
                        "border_mass": float(sample_border[0, motif_idx].detach().cpu()),
                        "outer_border_mass": float(sample_outer[0, motif_idx].detach().cpu()),
                        "foreground_mass": float(sample_foreground[0, motif_idx].detach().cpu()),
                        "effective_area": float(sample_area[0, motif_idx].detach().cpu()),
                        "selection_weight": float(sample_selection[motif_idx]),
                    }
                )
                selected_rows.append(
                    {
                        "sample_idx": saved,
                        "graph_id": graph_id,
                        "motif_idx": motif_idx,
                        "motif_score": float(sample_scores[motif_idx]),
                        "selection_weight": float(sample_selection[motif_idx]),
                        "selected_rank": int(np.where(np.argsort(sample_selection)[::-1] == motif_idx)[0][0] + 1),
                        "border_mass": float(sample_border[0, motif_idx].detach().cpu()),
                        "outer_border_mass": float(sample_outer[0, motif_idx].detach().cpu()),
                        "foreground_mass": float(sample_foreground[0, motif_idx].detach().cpu()),
                        "effective_area": float(sample_area[0, motif_idx].detach().cpu()),
                    }
                )

            pm = _np(pair_map[i])
            pe = _np(pair_emb[i])
            pd = _np(pair_dist[i])
            for left in range(num_motifs):
                for right in range(left + 1, num_motifs):
                    redundant = (
                        pm[left, right] > map_threshold
                        and pe[left, right] > emb_threshold
                        and pd[left, right] < center_threshold
                    )
                    pair_rows.append(
                        {
                            "sample_idx": saved,
                            "graph_id": graph_id,
                            "motif_i": left,
                            "motif_j": right,
                            "map_cosine": float(pm[left, right]),
                            "embedding_cosine": float(pe[left, right]),
                            "center_distance": float(pd[left, right]),
                            "redundant": int(redundant),
                        }
                    )

            if saved < int(max_samples):
                stem = f"motif_sample_{saved:03d}"
                title = f"id={graph_id}"
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(out_dir / f"{stem}_image.png", dpi=160)
                plt.close(fig)
                _plot_maps_grid(sample_maps, sample_scores, out_dir / f"{stem}_maps.png", title)
                _plot_overlay_grid(image, sample_maps, sample_scores, out_dir / f"{stem}_overlay.png", title)
                _plot_top_maps(image, sample_maps, sample_scores, out_dir / f"{stem}_top_maps.png", title)
                selected_path = out_dir / f"{stem}_selected_motif_overlay.png"
                _plot_selected_overlay(
                    image,
                    sample_maps,
                    sample_selection,
                    selected_path,
                    title,
                    top_m=int(model_cfg.get("selection_top_m", 8)),
                )
                clean_path = out_dir / f"{stem}_clean_motif_overlay.png"
                _plot_clean_overlay(
                    image,
                    sample_maps,
                    sample_clean,
                    clean_path,
                    title,
                    top_m=int(model_cfg.get("selection_top_m", 6)),
                )
                if saved == 0:
                    _plot_selected_overlay(
                        image,
                        sample_maps,
                        sample_selection,
                        out_dir / "selected_motif_overlay.png",
                        title,
                        top_m=int(model_cfg.get("selection_top_m", 8)),
                    )
                    _plot_clean_overlay(
                        image,
                        sample_maps,
                        sample_clean,
                        out_dir / "clean_motif_overlay.png",
                        title,
                        top_m=int(model_cfg.get("selection_top_m", 6)),
                    )
            saved += 1

    _write_rows(
        summary_rows,
        out_dir / "motif_audit_summary.csv",
        [
            "sample_idx",
            "graph_id",
            "mean_pairwise_map_sim",
            "max_pairwise_map_sim",
            "mean_pairwise_emb_sim",
            "max_pairwise_emb_sim",
            "mean_center_dist",
            "min_center_dist",
            "redundant_pair_count",
            "redundant_pair_ratio",
            "border_mass_mean",
            "outer_border_mass_mean",
            "center_mass_mean",
            "center_minus_border",
            "border_dominance",
            "selected_border_mass",
            "selected_outer_border_mass",
            "selected_foreground_mass",
            "selection_entropy",
            "selection_effective_count",
            "selected_top_m",
            "effective_area_mean",
            "effective_area_min",
            "effective_area_max",
            "area_in_range_ratio",
            "area_over_max_ratio",
            "area_under_min_ratio",
            "effective_motif_count",
            "effective_motif_ratio",
            "clean_score_mean",
            "clean_score_max",
            "clean_candidate_count",
            "hard_clean_candidate_count",
            "upper_clean_count",
            "middle_clean_count",
            "lower_clean_count",
            "min_region_clean_count",
            "mean_region_clean_count",
        ],
    )
    _write_rows(
        pair_rows,
        out_dir / "motif_pairwise_similarity.csv",
        ["sample_idx", "graph_id", "motif_i", "motif_j", "map_cosine", "embedding_cosine", "center_distance", "redundant"],
    )
    _write_rows(
        selected_rows,
        out_dir / "selected_motif_weights.csv",
        [
            "sample_idx",
            "graph_id",
            "motif_idx",
            "motif_score",
            "selection_weight",
            "selected_rank",
            "border_mass",
            "outer_border_mass",
            "foreground_mass",
            "effective_area",
        ],
    )
    _write_rows(
        clean_rows,
        out_dir / "motif_clean_scores.csv",
        [
            "sample_idx",
            "graph_id",
            "motif_id",
            "clean_score",
            "clean_border_score",
            "clean_outer_score",
            "clean_foreground_score",
            "clean_area_low_score",
            "clean_area_high_score",
            "border_mass",
            "outer_border_mass",
            "foreground_mass",
            "effective_area",
            "selection_weight",
        ],
    )
    _write_rows(
        region_clean_rows,
        out_dir / "motif_region_clean_scores.csv",
        [
            "sample_idx",
            "graph_id",
            "motif_id",
            "clean_score",
            "upper_mass",
            "middle_mass",
            "lower_mass",
            "dominant_region",
            "selection_weight",
        ],
    )
    print(f"Motif discovery figures and CSVs: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8m_motif_discovery_debug.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = str(args.output_dir)
        config.setdefault("output", {})["dir"] = str(args.output_dir)
    if sys.platform.startswith("win") and args.num_workers is None:
        config.setdefault("data", {})["num_workers"] = 0
        config.setdefault("data", {})["persistent_workers"] = False
        config.setdefault("data", {})["prefetch_factor"] = None
    run_visualize(
        config,
        split=args.split,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        checkpoint=args.checkpoint,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
