"""Prepare and summarize D13A visual pooling / assignment audits.

This utility is read-only with respect to trained models. It creates visual
traceability artifacts for D13A reduction outputs and summarizes a manually
filled review sheet. It makes no motif, semantic-region, or causal-evidence
claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device
from data.labels import EMOTION_NAMES
from models.d13_hierarchical_reduction_model import D13HierarchicalReductionModel
from training.trainer import move_to_device


LEGACY_AUDIT_SHEET_NAME = "d13a_k256_visual_pooling_audit_sheet.csv"
LEGACY_AUDIT_FILLED_NAME = "d13a_k256_visual_pooling_audit_sheet_filled.csv"


def _sanitize_run_name(value: Optional[str], fallback: str) -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _audit_prefix(run_name: str) -> str:
    return f"d13a_{_sanitize_run_name(run_name, 'run')}_visual_pooling"


def _audit_names(run_name: str) -> Dict[str, str]:
    prefix = _audit_prefix(run_name)
    return {
        "sheet": f"{prefix}_audit_sheet.csv",
        "filled": f"{prefix}_audit_sheet_filled.csv",
        "instructions": f"{prefix}_audit_instructions.md",
        "index": f"{prefix}_audit_index.md",
        "summary": f"{prefix}_audit_summary.csv",
        "area": f"{prefix}_area_summary.csv",
        "risk": f"{prefix}_risk_cases.csv",
        "report": f"{prefix}_audit_report.md",
    }


def _resolve_run_name(args: argparse.Namespace, path: Optional[Path] = None) -> str:
    fallback = path.name if path is not None else "run"
    return _sanitize_run_name(getattr(args, "run_name", None), fallback)


def _load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "For the current local layout, K256 ep100 is often under "
            "outputs/d13_hierarchical_reduction/extended/"
            "d13a_edgeaware_lite_localpool_k256_outputs_ep100_outputs/checkpoints/best.pt"
        )
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _extract_dense_x(batch: Dict[str, torch.Tensor], graph_idx: int, pixel_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = batch["x"]
    if x.ndim == 3:
        return x[graph_idx].detach().cpu()
    if x.ndim == 2:
        if pixel_indices is None:
            raise ValueError("Flat batch needs pixel_indices to extract a graph image")
        return x.index_select(0, pixel_indices.detach().cpu().long()).detach().cpu()
    raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")


def _sample_graph_id(batch: Dict[str, torch.Tensor], graph_idx: int, fallback: str) -> str:
    for key in ("graph_id", "idx", "index"):
        if key in batch:
            value = batch[key]
            try:
                if torch.is_tensor(value):
                    return str(value.detach().cpu().view(-1)[graph_idx].item())
                return str(value[graph_idx])
            except Exception:
                pass
    return fallback


def _dominant_assignment(payload: Dict[str, torch.Tensor], num_nodes: int = 48 * 48) -> np.ndarray:
    anchors = payload["anchor_index"].detach().cpu()
    weights = payload["weights"].detach().cpu()
    dominant = anchors.gather(1, weights.argmax(dim=1, keepdim=True)).squeeze(1).numpy()
    if dominant.shape[0] != num_nodes:
        out = np.full((num_nodes,), -1, dtype=np.int64)
        pix = payload["pixel_index"].detach().cpu().numpy()
        out[pix] = dominant
        dominant = out
    return dominant.reshape(48, 48)


def _pixel_entropy_map(payload: Dict[str, torch.Tensor], num_nodes: int = 48 * 48) -> Tuple[np.ndarray, float]:
    weights = payload["weights"].detach().cpu().float()
    denom = max(float(math.log(max(weights.shape[1], 2))), 1e-6)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1) / denom
    ent_np = entropy.numpy()
    if ent_np.shape[0] != num_nodes:
        out = np.full((num_nodes,), np.nan, dtype=np.float32)
        pix = payload["pixel_index"].detach().cpu().numpy()
        out[pix] = ent_np
        ent_np = out
    return ent_np.reshape(48, 48), float(np.nanmean(ent_np))


def _region_area(payload: Dict[str, torch.Tensor], grid_size: int) -> np.ndarray:
    anchors = payload["anchor_index"].detach().cpu().numpy().reshape(-1)
    weights = payload["weights"].detach().cpu().numpy().reshape(-1)
    area = np.zeros((grid_size * grid_size,), dtype=np.float32)
    np.add.at(area, anchors, weights)
    return area


def _project_region_to_pixels(payload: Dict[str, torch.Tensor], region_importance: np.ndarray, num_nodes: int = 48 * 48) -> np.ndarray:
    anchors = payload["anchor_index"].detach().cpu().numpy()
    weights = payload["weights"].detach().cpu().numpy()
    pixel_values = (region_importance[anchors] * weights).sum(axis=1)
    if pixel_values.shape[0] != num_nodes:
        out = np.zeros((num_nodes,), dtype=np.float32)
        pix = payload["pixel_index"].detach().cpu().numpy()
        out[pix] = pixel_values
        pixel_values = out
    return pixel_values.reshape(48, 48).astype(np.float32)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _region_importance_for_graph(
    out: Dict[str, Any],
    graph_idx: int,
    grid_size: int,
) -> Tuple[np.ndarray, str]:
    k = grid_size * grid_size
    start = graph_idx * k
    end = start + k
    attention = out.get("region_attention")
    if torch.is_tensor(attention) and attention.numel() >= end:
        return _tensor_to_numpy(attention[start:end].float()), "attention_readout_weight"
    h_region = out.get("h_region")
    if torch.is_tensor(h_region) and h_region.shape[0] >= end:
        return _tensor_to_numpy(h_region[start:end].float().norm(dim=1)), "region_embedding_norm"
    return np.ones((k,), dtype=np.float32), "uniform_fallback"


def _overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = _normalize_map(image)
    cmap = plt.get_cmap("magma")(_normalize_map(heat))[..., :3]
    gray = np.stack([base, base, base], axis=-1)
    return np.clip((1.0 - alpha) * gray + alpha * cmap, 0.0, 1.0)


def _top_region_overlay(image: np.ndarray, region_importance: np.ndarray, grid_size: int, top_k: int = 12) -> np.ndarray:
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)
    top = np.argsort(region_importance)[-int(top_k) :]
    heat.reshape(-1)[top] = region_importance[top]
    heat = np.kron(_normalize_map(heat), np.ones((48 // grid_size, 48 // grid_size), dtype=np.float32))
    if heat.shape != (48, 48):
        heat = np.resize(heat, (48, 48))
    return _overlay_heatmap(image, heat, alpha=0.60)


def _save_sample_figure(
    fig_path: Path,
    image: np.ndarray,
    entropy_map: np.ndarray,
    area: np.ndarray,
    region_importance: np.ndarray,
    projected: np.ndarray,
    grid_size: int,
    title: str,
    importance_source: str,
) -> None:
    area_map = area.reshape(grid_size, grid_size)
    imp_map = region_importance.reshape(grid_size, grid_size)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    axes = axes.reshape(-1)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original 48x48")
    axes[1].imshow(entropy_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Assignment softness")
    axes[2].imshow(area_map, cmap="viridis")
    axes[2].set_title(f"Region area ({grid_size}x{grid_size})")
    axes[3].imshow(_top_region_overlay(image, region_importance, grid_size), interpolation="nearest")
    axes[3].set_title(f"Top regions ({importance_source})")
    axes[4].imshow(_overlay_heatmap(image, projected), interpolation="nearest")
    axes[4].set_title("Pixel-projected importance")
    axes[5].imshow(imp_map, cmap="magma")
    axes[5].set_title("Region importance grid")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _load_sample_id_order(path: Optional[str]) -> List[str]:
    if not path:
        return []
    sheet_path = Path(path)
    if not sheet_path.exists():
        raise FileNotFoundError(f"Sample id sheet not found: {sheet_path}")
    df = pd.read_csv(sheet_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Sample id sheet needs a sample_id column: {sheet_path}")
    return [str(v) for v in df["sample_id"].dropna().astype(str).tolist()]


def _select_samples(
    candidates: List[Dict[str, Any]],
    samples_per_class: int,
    sample_id_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if sample_id_order:
        by_id = {str(row["sample_id"]): row for row in candidates}
        selected = []
        missing = []
        label_counts = {idx: 0 for idx in range(len(EMOTION_NAMES))}
        for sample_id in sample_id_order:
            if sample_id in by_id:
                row = by_id[sample_id]
                label = int(row["label"])
                if label_counts[label] < samples_per_class:
                    selected.append(row)
                    label_counts[label] += 1
            else:
                missing.append(sample_id)
        if missing:
            warnings.warn(
                f"Could not find {len(missing)}/{len(sample_id_order)} requested sample_ids; "
                f"missing examples include: {missing[:8]}"
            )
        expected = min(len(sample_id_order), samples_per_class * len(EMOTION_NAMES))
        if len(selected) < expected:
            warnings.warn(f"Selected {len(selected)} requested samples; expected up to {expected}.")
        return selected

    selected: List[Dict[str, Any]] = []
    rng = np.random.default_rng(13)
    for label in range(len(EMOTION_NAMES)):
        rows = [r for r in candidates if int(r["label"]) == label]
        if len(rows) < samples_per_class:
            warnings.warn(f"Class {EMOTION_NAMES[label]} has only {len(rows)} candidates; requested {samples_per_class}")
        correct_high = sorted([r for r in rows if r["correct"]], key=lambda r: r["confidence"], reverse=True)
        correct_low = sorted([r for r in rows if r["correct"]], key=lambda r: r["confidence"])
        wrong_high = sorted([r for r in rows if not r["correct"]], key=lambda r: r["confidence"], reverse=True)
        picked: List[Dict[str, Any]] = []
        buckets = [correct_high, correct_low, wrong_high]
        while len(picked) < min(samples_per_class, len(rows)) and any(buckets):
            for bucket in buckets:
                while bucket and bucket[0]["sample_id"] in {p["sample_id"] for p in picked}:
                    bucket.pop(0)
                if bucket and len(picked) < samples_per_class:
                    picked.append(bucket.pop(0))
        if len(picked) < min(samples_per_class, len(rows)):
            remaining = [r for r in rows if r["sample_id"] not in {p["sample_id"] for p in picked}]
            order = rng.permutation(len(remaining))
            for idx in order:
                if len(picked) >= samples_per_class:
                    break
                picked.append(remaining[int(idx)])
        selected.extend(picked[:samples_per_class])
    return selected


@torch.no_grad()
def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    run_name = _resolve_run_name(args, output_dir)
    names = _audit_names(run_name)
    figures_dir = output_dir / "figures"
    masks_dir = output_dir / "masks"
    figures_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("data", {})["batch_size"] = int(args.batch_size or config.get("data", {}).get("batch_size", 16))
    device = resolve_device(args.device, config)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    model = D13HierarchicalReductionModel.from_config(config.get("model", {})).to(device)
    ckpt = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    if hasattr(model.reduction, "set_save_visualization"):
        model.reduction.set_save_visualization(True)
    model.eval()

    grid_size = int(config.get("model", {}).get("pooling", {}).get("grid_size", 16))
    candidates: List[Dict[str, Any]] = []
    global_index = 0
    requested_sample_ids = _load_sample_id_order(getattr(args, "sample_id_sheet", None))
    requested_sample_id_set = set(requested_sample_ids)
    max_candidates_per_class = max(int(args.samples_per_class) * 4, int(args.samples_per_class) + 3)
    candidate_counts = {idx: 0 for idx in range(len(EMOTION_NAMES))}

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        out = model(batch)
        probs = torch.softmax(out["logits"], dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        payloads = out.get("aux", {}).get("assignment_maps", [])
        if not payloads:
            raise RuntimeError("No assignment_maps returned. The reduction visualization flag did not reach LocalAssignmentPool.")
        y = batch["y"].detach().cpu()
        for graph_idx, payload in enumerate(payloads):
            label = int(y[graph_idx].item())
            pred_i = int(pred[graph_idx].detach().cpu().item())
            conf_i = float(conf[graph_idx].detach().cpu().item())
            sample_id = _sample_graph_id(batch, graph_idx, fallback=f"{args.split}_{global_index:06d}")
            if requested_sample_id_set:
                if str(sample_id) not in requested_sample_id_set:
                    global_index += 1
                    continue
            elif candidate_counts[label] >= max_candidates_per_class:
                global_index += 1
                continue
            x_graph = _extract_dense_x(batch, graph_idx, payload.get("pixel_index"))
            image = x_graph[:, 0].reshape(48, 48).numpy().astype(np.float32)
            region_importance, importance_source = _region_importance_for_graph(out, graph_idx, grid_size)
            area = _region_area(payload, grid_size)
            entropy_map, entropy_mean = _pixel_entropy_map(payload)
            dominant = _dominant_assignment(payload)
            projected = _project_region_to_pixels(payload, region_importance)
            candidates.append(
                {
                    "sample_id": str(sample_id),
                    "split": args.split,
                    "batch_idx": batch_idx,
                    "graph_idx": graph_idx,
                    "global_index": global_index,
                    "label": label,
                    "pred": pred_i,
                    "confidence": conf_i,
                    "correct": bool(label == pred_i),
                    "image": image,
                    "assignment_map": dominant,
                    "entropy_map": entropy_map,
                    "assignment_entropy": entropy_mean,
                    "region_area": area,
                    "region_importance": region_importance.astype(np.float32),
                    "pixel_projected_importance": projected.astype(np.float32),
                    "importance_source": importance_source,
                    "effective_regions_sample": float(area.sum() ** 2 / max(float(np.square(area).sum()), 1e-8)),
                    "empty_region_ratio_sample": float(np.mean(area <= 1e-6)),
                    "topk_anchor_index": _tensor_to_numpy(payload["anchor_index"].short()),
                    "topk_weights": _tensor_to_numpy(payload["weights"].float()).astype(np.float16),
                }
            )
            candidate_counts[label] += 1
            global_index += 1
        if requested_sample_id_set and len(candidates) >= len(requested_sample_id_set):
            break
        if not requested_sample_id_set and all(count >= max_candidates_per_class for count in candidate_counts.values()):
            break

    selected = _select_samples(candidates, int(args.samples_per_class), requested_sample_ids)
    rows = []
    index_lines = [
        f"# D13A {run_name} Visual Pooling Audit Index",
        "",
        "| sample_id | label | pred | confidence | correct | figure | review_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item_idx, item in enumerate(selected):
        label_name = EMOTION_NAMES[int(item["label"])]
        pred_name = EMOTION_NAMES[int(item["pred"])]
        status = "correct" if item["correct"] else "wrong"
        safe_id = str(item["sample_id"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
        file_stem = f"sample_{safe_id}_label_{label_name}_pred_{pred_name}_status_{status}"
        fig_path = figures_dir / f"{file_stem}.png"
        sample_mask_dir = masks_dir / safe_id
        sample_mask_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_mask_dir / "assignment_map.npy", item["assignment_map"])
        np.save(sample_mask_dir / "assignment_entropy_map.npy", item["entropy_map"].astype(np.float32))
        np.save(sample_mask_dir / "region_area.npy", item["region_area"].astype(np.float32))
        np.save(sample_mask_dir / "region_importance.npy", item["region_importance"].astype(np.float32))
        np.save(sample_mask_dir / "pixel_projected_importance.npy", item["pixel_projected_importance"].astype(np.float32))
        np.save(sample_mask_dir / "assignment_topk_anchor_index.npy", item["topk_anchor_index"])
        np.save(sample_mask_dir / "assignment_topk_weights.npy", item["topk_weights"])
        title = (
            f"id={item['sample_id']} true={label_name} pred={pred_name} "
            f"conf={item['confidence']:.3f} correct={item['correct']} "
            f"entropy={item['assignment_entropy']:.3f}"
        )
        _save_sample_figure(
            fig_path,
            item["image"],
            item["entropy_map"],
            item["region_area"],
            item["region_importance"],
            item["pixel_projected_importance"],
            grid_size,
            title,
            item["importance_source"],
        )
        metadata = {
            "sample_id": item["sample_id"],
            "split": item["split"],
            "label": label_name,
            "label_index": int(item["label"]),
            "pred": pred_name,
            "pred_index": int(item["pred"]),
            "confidence": float(item["confidence"]),
            "correct": bool(item["correct"]),
            "figure_path": str(fig_path.relative_to(output_dir)),
            "mask_paths": {
                "assignment_map": str((sample_mask_dir / "assignment_map.npy").relative_to(output_dir)),
                "assignment_entropy_map": str((sample_mask_dir / "assignment_entropy_map.npy").relative_to(output_dir)),
                "region_area": str((sample_mask_dir / "region_area.npy").relative_to(output_dir)),
                "region_importance": str((sample_mask_dir / "region_importance.npy").relative_to(output_dir)),
                "pixel_projected_importance": str((sample_mask_dir / "pixel_projected_importance.npy").relative_to(output_dir)),
                "assignment_topk_anchor_index": str((sample_mask_dir / "assignment_topk_anchor_index.npy").relative_to(output_dir)),
                "assignment_topk_weights": str((sample_mask_dir / "assignment_topk_weights.npy").relative_to(output_dir)),
            },
            "assignment_entropy": float(item["assignment_entropy"]),
            "effective_regions_sample": float(item["effective_regions_sample"]),
            "empty_region_ratio_sample": float(item["empty_region_ratio_sample"]),
            "region_importance_proxy": item["importance_source"],
            "notes_auto": (
                "Top-region panel uses a proxy importance score, not causal evidence. "
                "Region nodes are soft bottleneck nodes, not semantic facial regions."
            ),
        }
        metadata_path = sample_mask_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        row = {
            "sample_id": item["sample_id"],
            "split": item["split"],
            "label": label_name,
            "pred": pred_name,
            "confidence": float(item["confidence"]),
            "correct": bool(item["correct"]),
            "figure_path": str(fig_path.relative_to(output_dir)),
            "metadata_path": str(metadata_path.relative_to(output_dir)),
            "review_status": "NEEDS_MANUAL_REVIEW",
            "visual_status": "",
            "face_coverage_score": "",
            "region_traceability_score": "",
            "assignment_interpretability_score": "",
            "hair_glasses_risk": "",
            "background_border_risk": "",
            "center_shortcut_risk": "",
            "main_visible_area": "",
            "region_softness_issue": "",
            "notes": "",
        }
        rows.append(row)
        index_lines.append(
            f"| {item['sample_id']} | {label_name} | {pred_name} | {item['confidence']:.3f} | "
            f"{item['correct']} | [{fig_path.name}]({fig_path.relative_to(output_dir).as_posix()}) | NEEDS_MANUAL_REVIEW |"
        )

    sheet = pd.DataFrame(rows)
    sheet.to_csv(output_dir / names["sheet"], index=False)
    (output_dir / names["index"]).write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    _write_instructions(output_dir / names["instructions"], run_name=run_name, grid_size=grid_size)
    prepare_summary = {
        "mode": "prepare",
        "run_name": run_name,
        "split": args.split,
        "samples_requested_per_class": int(args.samples_per_class),
        "sample_id_sheet": str(args.sample_id_sheet) if getattr(args, "sample_id_sheet", None) else None,
        "sample_id_reuse_requested": bool(requested_sample_ids),
        "sample_id_reuse_found": int(len(selected)) if requested_sample_ids else None,
        "samples_saved": int(len(sheet)),
        "candidate_counts_by_label": {EMOTION_NAMES[k]: int(v) for k, v in candidate_counts.items()},
        "output_dir": str(output_dir),
        "importance_proxy": sorted(set(item["importance_source"] for item in selected)),
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(prepare_summary, indent=2), encoding="utf-8")
    print(json.dumps(prepare_summary, indent=2))


def _write_instructions(path: Path, run_name: str, grid_size: int) -> None:
    path.write_text(
        f"""# D13A {run_name} Visual Pooling Audit Instructions

This review is about traceability and visual reliability of D13A reduction/assignment only.
Do not evaluate motif quality, do not label region nodes as semantic facial parts, and do not treat overlays as causal evidence.
This run uses a {grid_size}x{grid_size} region grid. Region nodes are soft learnable bottleneck nodes.

## visual_status
- PASS: overlay/region importance is mostly on the face, traceable back to pixels, not dominated by hair/glasses/background/border, and assignment map is readable.
- PARTIAL: some reasonable face coverage but mixed with non-face support, or assignment is soft/diffuse but still inspectable.
- FAIL: overlay is dominated by hair/glasses/background/border, is too diffuse to interpret, or shows an obvious center shortcut.

## Scores
- face_coverage_score: 0 = mostly outside face, 1 = partial/noisy face coverage, 2 = clear face coverage.
- region_traceability_score: 0 = no meaningful region-to-pixel trace, 1 = traceable but noisy, 2 = clear region/pixel support.
- assignment_interpretability_score: 0 = too soft/chaotic, 1 = soft but readable, 2 = spatially interpretable.
- hair_glasses_risk: 0 = negligible, 1 = present but not dominant, 2 = dominant or serious concern.
- background_border_risk: 0 = negligible, 1 = present but not dominant, 2 = dominant background/border/face contour.
- center_shortcut_risk: 0 = negligible, 1 = suspicious, 2 = clear center shortcut.
- region_softness_issue: 0 = no major issue, 1 = somewhat soft but usable, 2 = too soft/diffuse for motif foundation.

## main_visible_area suggestions
mouth, eyes, eyebrows, nose_cheek, forehead, face_contour, full_face, hair, glasses, background, border, mixed, unclear.

Final review should judge whether this D13A reduction is visually reliable enough for a D13B diagnostic run. It must not claim motif discovery.
""",
        encoding="utf-8",
    )


def _read_review_sheet(audit_dir: Path, review_sheet: Optional[str], run_name: str) -> Tuple[pd.DataFrame, Path]:
    if review_sheet:
        path = Path(review_sheet)
    else:
        names = _audit_names(run_name)
        candidates = [
            audit_dir / names["filled"],
            audit_dir / names["sheet"],
            audit_dir / LEGACY_AUDIT_FILLED_NAME,
            audit_dir / LEGACY_AUDIT_SHEET_NAME,
        ]
        candidates.extend(sorted(audit_dir.glob("d13a_*_visual_pooling_audit_sheet_filled.csv")))
        candidates.extend(sorted(audit_dir.glob("d13a_*_visual_pooling_audit_sheet.csv")))
        path = next((p for p in candidates if p.exists()), audit_dir / names["sheet"])
    if not path.exists():
        raise FileNotFoundError(f"Review sheet not found: {path}")
    return pd.read_csv(path), path


def _numeric_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _rate(count: int, total: int) -> float:
    return float(count / max(total, 1))


def _write_md_table(df: pd.DataFrame, cols: List[str], n: Optional[int] = None) -> str:
    if df.empty:
        return "No data."
    available = [c for c in cols if c in df.columns]
    use = df[available].copy()
    if n is not None:
        use = use.head(n)
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def summarize(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    run_name = _resolve_run_name(args, audit_dir)
    names = _audit_names(run_name)
    df, sheet_path = _read_review_sheet(audit_dir, args.review_sheet, run_name)
    status = df.get("visual_status", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
    review_status = df.get("review_status", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
    reviewed = (review_status != "NEEDS_MANUAL_REVIEW") & (status != "")
    total = int(len(df))
    reviewed_count = int(reviewed.sum())
    pass_count = int((status == "PASS").sum())
    partial_count = int((status == "PARTIAL").sum())
    fail_count = int((status == "FAIL").sum())
    summary = {
        "run_name": run_name,
        "total_samples": total,
        "reviewed_samples": reviewed_count,
        "unreviewed_samples": total - reviewed_count,
        "pass_count": pass_count,
        "partial_count": partial_count,
        "fail_count": fail_count,
        "pass_rate": _rate(pass_count, total),
        "partial_rate": _rate(partial_count, total),
        "fail_rate": _rate(fail_count, total),
        "pass_partial_count": pass_count + partial_count,
        "pass_partial_rate": _rate(pass_count + partial_count, total),
        "avg_face_coverage_score": _numeric_mean(df, "face_coverage_score"),
        "avg_region_traceability_score": _numeric_mean(df, "region_traceability_score"),
        "avg_assignment_interpretability_score": _numeric_mean(df, "assignment_interpretability_score"),
        "avg_hair_glasses_risk": _numeric_mean(df, "hair_glasses_risk"),
        "avg_background_border_risk": _numeric_mean(df, "background_border_risk"),
        "avg_center_shortcut_risk": _numeric_mean(df, "center_shortcut_risk"),
        "avg_region_softness_issue": _numeric_mean(df, "region_softness_issue"),
        "review_sheet": str(sheet_path),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(audit_dir / names["summary"], index=False)

    area_rows = []
    if "main_visible_area" in df:
        for area, group in df.fillna({"main_visible_area": "unclear"}).groupby("main_visible_area", dropna=False):
            st = group.get("visual_status", pd.Series([""] * len(group))).fillna("").astype(str).str.upper()
            area_rows.append(
                {
                    "main_visible_area": area if str(area) else "unclear",
                    "count": int(len(group)),
                    "rate": _rate(int(len(group)), total),
                    "pass_count": int((st == "PASS").sum()),
                    "partial_count": int((st == "PARTIAL").sum()),
                    "fail_count": int((st == "FAIL").sum()),
                    "avg_face_coverage_score": _numeric_mean(group, "face_coverage_score"),
                    "avg_region_traceability_score": _numeric_mean(group, "region_traceability_score"),
                    "avg_assignment_interpretability_score": _numeric_mean(group, "assignment_interpretability_score"),
                    "avg_hair_glasses_risk": _numeric_mean(group, "hair_glasses_risk"),
                    "avg_background_border_risk": _numeric_mean(group, "background_border_risk"),
                    "avg_center_shortcut_risk": _numeric_mean(group, "center_shortcut_risk"),
                    "avg_region_softness_issue": _numeric_mean(group, "region_softness_issue"),
                }
            )
    area_df = pd.DataFrame(area_rows)
    area_df.to_csv(audit_dir / names["area"], index=False)

    risk_mask = status == "FAIL"
    for col in ["hair_glasses_risk", "background_border_risk", "center_shortcut_risk", "region_softness_issue"]:
        if col in df:
            risk_mask = risk_mask | (pd.to_numeric(df[col], errors="coerce") >= 2)
    for col in ["region_traceability_score", "assignment_interpretability_score"]:
        if col in df:
            risk_mask = risk_mask | (pd.to_numeric(df[col], errors="coerce") == 0)
    risk_cols = [
        "sample_id", "label", "pred", "confidence", "correct", "figure_path", "visual_status",
        "face_coverage_score", "region_traceability_score", "assignment_interpretability_score",
        "hair_glasses_risk", "background_border_risk", "center_shortcut_risk", "main_visible_area",
        "region_softness_issue", "notes",
    ]
    risk_df = df.loc[risk_mask, [c for c in risk_cols if c in df.columns]].copy()
    risk_df.to_csv(audit_dir / names["risk"], index=False)

    decision = _visual_decision(summary, run_name)
    _write_summary_report(audit_dir, summary_df, area_df, risk_df, decision, incomplete=(reviewed_count < total), run_name=run_name, names=names)
    print(json.dumps({"audit_dir": str(audit_dir), "decision": decision, **summary}, indent=2))


def _run_family(run_name: str) -> str:
    low = run_name.lower()
    if "anneal" in low:
        return "ANNEAL"
    if "k144" in low or "baseline" in low:
        return "K144"
    if "k256" in low:
        return "K256"
    return "D13A"


def _visual_decision(summary: Dict[str, Any], run_name: str) -> str:
    family = _run_family(run_name)
    if summary["reviewed_samples"] != summary["total_samples"]:
        return "VISUAL_POOLING_AUDIT_INCOMPLETE_KEEP_D13B_LOCKED"
    gates = [
        summary["pass_partial_rate"] >= 0.60,
        (summary["avg_face_coverage_score"] or 0.0) >= 1.2,
        (summary["avg_region_traceability_score"] or 0.0) >= 1.2,
        (summary["avg_assignment_interpretability_score"] or 0.0) >= 1.2,
        (summary["avg_hair_glasses_risk"] or 9.0) < 1.0,
        (summary["avg_background_border_risk"] or 9.0) < 1.0,
        (summary["avg_center_shortcut_risk"] or 9.0) < 1.0,
        (summary["avg_region_softness_issue"] or 9.0) < 1.2,
    ]
    if all(gates):
        return f"{family}_VISUAL_POOLING_ACCEPTABLE_FOR_D13B_DIAGNOSTIC"
    if summary["pass_partial_rate"] >= 0.40 and summary["fail_rate"] < 0.50:
        if family == "ANNEAL":
            return "ANNEAL_VISUAL_POOLING_PARTIAL_USE_WITH_CAUTION"
        if family == "K144":
            return "K144_VISUAL_POOLING_PARTIAL_USE_WITH_CAUTION"
        if family == "K256":
            return "K256_VISUAL_POOLING_PARTIAL_USE_WITH_CAUTION_COMPARE_K144"
        return "D13A_VISUAL_POOLING_PARTIAL_USE_WITH_CAUTION"
    if family == "K144":
        return "K144_VISUAL_POOLING_UNRELIABLE_AUDIT_ANNEAL"
    if family == "ANNEAL":
        return "ANNEAL_VISUAL_POOLING_UNRELIABLE_KEEP_D13B_LOCKED"
    if family == "K256":
        return "K256_VISUAL_POOLING_UNRELIABLE_AUDIT_K144_OR_ANNEAL"
    return "D13A_VISUAL_POOLING_UNRELIABLE"


def _write_summary_report(
    audit_dir: Path,
    summary_df: pd.DataFrame,
    area_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    decision: str,
    incomplete: bool,
    run_name: str,
    names: Dict[str, str],
) -> None:
    s = summary_df.iloc[0].to_dict()
    family = _run_family(run_name)
    score_context = {
        "K256": [
            "- test_macro_f1 = 0.5866; test_acc = 0.6227.",
            "- effective_regions = 246.28 / 256; empty_region_ratio = 0.0.",
            "- assignment_entropy is approximately 1.0.",
        ],
        "K144": [
            "- test_macro_f1 = 0.5829; test_acc = 0.6166.",
            "- assignment_entropy is approximately 0.8589 in the extended analysis.",
        ],
        "ANNEAL": [
            "- test_macro_f1 = 0.5813; test_acc = 0.6141.",
            "- assignment_entropy is approximately 0.8765 in the extended analysis.",
        ],
    }.get(family, [])
    lines = [
        f"# D13A {run_name} Visual Pooling / Assignment Audit Report",
        "",
        "## 1. Context",
        "- D13A is a pure GNN hierarchical reduction baseline.",
        f"- Audited run: `{run_name}`.",
        *score_context,
        "- Region nodes are soft learnable bottleneck nodes, not semantic regions.",
        "",
        "## 2. Audit Goal",
        "Check traceability and visual reliability before any D13B diagnostic. No motif claim is made.",
        "",
        "## 3. Visual Audit Summary",
        _write_md_table(summary_df, [
            "total_samples", "reviewed_samples", "unreviewed_samples", "pass_count", "partial_count",
            "fail_count", "pass_partial_count", "pass_partial_rate",
        ]),
        "",
        "## 4. Traceability and Interpretability",
        _write_md_table(summary_df, [
            "avg_face_coverage_score", "avg_region_traceability_score",
            "avg_assignment_interpretability_score", "avg_region_softness_issue",
        ]),
        "",
        "## 5. Risk Analysis",
        _write_md_table(summary_df, [
            "avg_hair_glasses_risk", "avg_background_border_risk", "avg_center_shortcut_risk",
        ]),
        f"Risk cases written to `{names['risk']}` ({len(risk_df)} rows).",
        "",
        "## 6. Area Distribution",
        _write_md_table(area_df, [
            "main_visible_area", "count", "rate", "pass_count", "partial_count", "fail_count",
            "avg_face_coverage_score", "avg_region_traceability_score",
            "avg_assignment_interpretability_score",
        ]),
        "",
        "## 7. Gate Check",
        "- reviewed_samples == total_samples",
        "- pass_partial_rate >= 0.60",
        "- avg_face_coverage_score >= 1.2",
        "- avg_region_traceability_score >= 1.2",
        "- avg_assignment_interpretability_score >= 1.2",
        "- avg_hair_glasses_risk < 1.0",
        "- avg_background_border_risk < 1.0",
        "- avg_center_shortcut_risk < 1.0",
        "- avg_region_softness_issue < 1.2",
        "",
        "## Final Decision",
        decision,
        "",
    ]
    if incomplete:
        lines.insert(-2, "Manual review is incomplete, so D13B remains locked.")
    (audit_dir / names["report"]).write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D13A visual pooling / assignment audit utility")
    parser.add_argument("--mode", required=True, choices=["prepare", "summarize"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--samples_per_class", type=int, default=15)
    parser.add_argument("--audit_dir", default=None)
    parser.add_argument("--review_sheet", default=None)
    parser.add_argument("--run_name", default=None, help="Run label used for audit file prefixes")
    parser.add_argument("--sample_id_sheet", default=None, help="Optional audit sheet whose sample_id column should be reused")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.config or not args.checkpoint or not args.output_dir:
            parser.error("--mode prepare requires --config, --checkpoint, and --output_dir")
        prepare(args)
    else:
        if not args.audit_dir:
            parser.error("--mode summarize requires --audit_dir")
        summarize(args)


if __name__ == "__main__":
    main()
