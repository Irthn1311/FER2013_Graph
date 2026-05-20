"""Prepare and summarize D13B visual slot audits.

D13B remains diagnostic only. These artifacts inspect slot candidates for
traceability and shortcut risk; they do not claim motifs, semantic regions, or
causal evidence.
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
from models.d13b_motif_slot_model import D13BMotifSlotModel
from training.trainer import move_to_device


SHEET_COLUMNS = [
    "sample_id",
    "split",
    "label",
    "pred",
    "confidence",
    "correct",
    "run_name",
    "num_slots",
    "figure_path",
    "metadata_path",
    "review_status",
    "visual_status",
    "slot_traceability_score",
    "slot_diversity_visual_score",
    "slot_face_coverage_score",
    "slot_assignment_readability_score",
    "mouth_only_risk",
    "center_shortcut_risk",
    "hair_glasses_risk",
    "background_border_risk",
    "dominant_slot_area",
    "multi_region_support",
    "slot_collapse_visual",
    "notes",
]

RISK_COLUMNS = [
    "sample_id",
    "label",
    "pred",
    "confidence",
    "correct",
    "figure_path",
    "visual_status",
    "slot_traceability_score",
    "slot_diversity_visual_score",
    "slot_face_coverage_score",
    "slot_assignment_readability_score",
    "mouth_only_risk",
    "center_shortcut_risk",
    "hair_glasses_risk",
    "background_border_risk",
    "dominant_slot_area",
    "multi_region_support",
    "slot_collapse_visual",
    "notes",
]


def _sanitize(value: str | None, fallback: str = "run") -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _prefix(run_name: str) -> str:
    return f"d13b_{_sanitize(run_name)}_visual_slot"


def _names(run_name: str) -> Dict[str, str]:
    p = _prefix(run_name)
    return {
        "sheet": f"{p}_audit_sheet.csv",
        "filled": f"{p}_audit_sheet_filled.csv",
        "instructions": f"{p}_audit_instructions.md",
        "index": f"{p}_audit_index.md",
        "summary": f"{p}_audit_summary.csv",
        "area": f"{p}_area_summary.csv",
        "risk": f"{p}_risk_cases.csv",
        "report": f"{p}_audit_report.md",
    }


def _load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = _normalize(image)
    gray = np.stack([base, base, base], axis=-1)
    color = plt.get_cmap("magma")(_normalize(heat))[..., :3]
    return np.clip((1.0 - alpha) * gray + alpha * color, 0.0, 1.0)


def _resize_grid(grid: np.ndarray, height: int = 48, width: int = 48) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float32)
    scale_y = max(1, int(math.ceil(height / g.shape[0])))
    scale_x = max(1, int(math.ceil(width / g.shape[1])))
    out = np.kron(g, np.ones((scale_y, scale_x), dtype=np.float32))
    return out[:height, :width]


def _extract_image(batch: Dict[str, torch.Tensor], idx: int) -> np.ndarray:
    x = batch["x"][idx].detach().cpu()
    return x[:, 0].reshape(48, 48).numpy().astype(np.float32)


def _value_at(batch: Dict[str, torch.Tensor], key: str, idx: int, fallback: int) -> int:
    value = batch.get(key)
    if torch.is_tensor(value):
        return int(value.detach().cpu().view(-1)[idx].item())
    return int(fallback)


def _read_sample_ids(path: str | None) -> List[str]:
    if not path:
        return []
    sheet = Path(path)
    if not sheet.exists():
        raise FileNotFoundError(f"sample_id_sheet not found: {sheet}")
    df = pd.read_csv(sheet)
    if "sample_id" not in df.columns:
        raise ValueError(f"sample_id_sheet requires sample_id column: {sheet}")
    return [str(v) for v in df["sample_id"].dropna().astype(str).tolist()]


def _slot_sample_stats(slot_attention: np.ndarray, slot_embeddings: np.ndarray | None = None) -> Dict[str, float]:
    eps = 1e-8
    attn = np.asarray(slot_attention, dtype=np.float32)
    m, k = attn.shape
    entropy = -(attn * np.log(np.clip(attn, eps, None))).sum(axis=1) / max(math.log(max(k, 2)), eps)
    flat = attn / np.clip(np.linalg.norm(attn, axis=1, keepdims=True), eps, None)
    sim = flat @ flat.T
    off = sim[~np.eye(m, dtype=bool)]
    if slot_embeddings is not None:
        mass = np.linalg.norm(slot_embeddings.astype(np.float32), axis=1)
    else:
        mass = attn.sum(axis=1)
    mass = np.clip(mass, eps, None)
    dist = mass / np.clip(mass.sum(), eps, None)
    effective = float(np.exp(-(dist * np.log(np.clip(dist, eps, None))).sum()))
    slot_area = (attn > (1.0 / max(float(k), 1.0))).sum(axis=1)
    return {
        "effective_slots_sample": effective,
        "slot_overlap_sample": float(off.mean()) if off.size else 0.0,
        "slot_entropy_sample": float(entropy.mean()),
        "slot_dominance_sample": float(dist.max()),
        "slot_area_mean": float(slot_area.mean()),
        "slot_area_min": float(slot_area.min()),
        "slot_area_max": float(slot_area.max()),
    }


def _project_assignment(
    payload: Dict[str, torch.Tensor],
    slot_attention: np.ndarray,
    num_nodes: int = 48 * 48,
) -> Tuple[np.ndarray, str]:
    anchors = _to_numpy(payload["anchor_index"].long())
    weights = _to_numpy(payload["weights"].float())
    maps = []
    for slot_idx in range(slot_attention.shape[0]):
        values = (slot_attention[slot_idx][anchors] * weights).sum(axis=1)
        if values.shape[0] != num_nodes:
            out = np.zeros((num_nodes,), dtype=np.float32)
            pix = _to_numpy(payload["pixel_index"].long())
            out[pix] = values
            values = out
        maps.append(values.reshape(48, 48).astype(np.float32))
    return np.stack(maps, axis=0), "assignment_projected"


def _project_grid(slot_attention: np.ndarray) -> Tuple[np.ndarray, str]:
    k = int(slot_attention.shape[1])
    grid = int(round(math.sqrt(k)))
    if grid * grid != k:
        raise ValueError(f"Cannot grid-upscale slot attention with K={k}")
    maps = np.stack([_resize_grid(row.reshape(grid, grid)) for row in slot_attention], axis=0)
    return maps.astype(np.float32), "grid_upscale"


def _combined_map(slot_maps: np.ndarray, method: str = "max") -> np.ndarray:
    if method == "sum_normalized":
        arr = slot_maps.sum(axis=0)
    else:
        arr = slot_maps.max(axis=0)
    return _normalize(arr).astype(np.float32)


def _slot_centers(slot_maps: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32)
    centers = []
    for item in slot_maps:
        w = np.clip(item.astype(np.float32), 0.0, None)
        denom = float(w.sum())
        if denom <= 1e-8:
            centers.append([np.nan, np.nan])
        else:
            centers.append([float((xx * w).sum() / denom), float((yy * w).sum() / denom)])
    return np.asarray(centers, dtype=np.float32)


def _save_figure(
    path: Path,
    image: np.ndarray,
    combined: np.ndarray,
    slot_maps: np.ndarray,
    slot_attention: np.ndarray,
    slot_importance: np.ndarray,
    centers: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    num_slots = int(slot_attention.shape[0])
    top_n = min(8, num_slots)
    top = np.argsort(slot_importance)[-top_n:][::-1]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Original 48x48")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_overlay(image, combined))
    ax.set_title("Combined slot importance")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(slot_attention, cmap="viridis", aspect="auto")
    ax.set_title("Slot attention M x K")
    ax.set_xlabel("region")
    ax.set_ylabel("slot")
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(centers[:, 0], centers[:, 1], c=np.arange(num_slots), cmap="tab20", s=45)
    ax.set_xlim(0, 47)
    ax.set_ylim(47, 0)
    ax.set_title("Slot centers")
    ax.grid(True, alpha=0.25)
    for pos, slot_idx in enumerate(top):
        row = 1 + pos // 4
        col = pos % 4
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(_overlay(image, slot_maps[slot_idx]))
        ax.set_title(f"slot {slot_idx} imp={slot_importance[slot_idx]:.3f}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[3, :])
    text = "\n".join(
        [
            f"run={metadata['run_name']} sample={metadata['sample_id']} split={metadata['split']}",
            f"label={metadata['label']} pred={metadata['pred']} conf={metadata['confidence']:.3f} correct={metadata['correct']}",
            f"effective_slots={metadata.get('effective_slots_sample')} overlap={metadata.get('slot_overlap_sample')} entropy={metadata.get('slot_entropy_sample')} dominance={metadata.get('slot_dominance_sample')}",
            f"projection={metadata['projection_method']} combined={metadata['combined_slot_map_method']}",
            "Visual diagnostic only: slot maps are not motif, semantic-region, or causal evidence.",
        ]
    )
    ax.text(0.01, 0.96, text, va="top", ha="left", fontsize=10, family="monospace")
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_model(config: Dict[str, Any], checkpoint: str | Path, device: torch.device) -> D13BMotifSlotModel:
    model = D13BMotifSlotModel.from_config(config.get("model", {})).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    if hasattr(model.reduction, "set_save_visualization"):
        model.reduction.set_save_visualization(True)
    model.eval()
    return model


def _selected_from_candidates(
    candidates: List[Dict[str, Any]],
    sample_ids: List[str],
    samples_per_class: int,
) -> List[Dict[str, Any]]:
    if sample_ids:
        by_id = {str(row["sample_id"]): row for row in candidates}
        counts = {idx: 0 for idx in range(len(EMOTION_NAMES))}
        selected = []
        missing = []
        for sample_id in sample_ids:
            row = by_id.get(str(sample_id))
            if row is None:
                missing.append(str(sample_id))
                continue
            label = int(row["label_index"])
            if counts[label] < samples_per_class:
                selected.append(row)
                counts[label] += 1
        if missing:
            warnings.warn(f"Could not match {len(missing)} requested sample_ids; examples: {missing[:8]}")
        if not selected:
            warnings.warn("sample_id_sheet matched no samples; falling back to stratified random sampling.")
        else:
            return selected

    rng = np.random.default_rng(13)
    selected = []
    for label in range(len(EMOTION_NAMES)):
        rows = [r for r in candidates if int(r["label_index"]) == label]
        if len(rows) < samples_per_class:
            warnings.warn(f"Class {EMOTION_NAMES[label]} has {len(rows)} samples; requested {samples_per_class}")
        order = rng.permutation(len(rows))
        for idx in order[: min(samples_per_class, len(rows))]:
            selected.append(rows[int(idx)])
    return selected


def _metric_context(checkpoint: str | Path) -> Dict[str, Any]:
    run_dir = Path(checkpoint).resolve().parents[1] if len(Path(checkpoint).parts) >= 2 else Path(checkpoint).parent
    context: Dict[str, Any] = {}
    test_path = run_dir / "test_metrics.csv"
    check_path = run_dir / "d13b_diagnostic_check_summary.json"
    if test_path.exists():
        row = pd.read_csv(test_path).iloc[-1].to_dict()
        context.update(
            {
                "test_macro_f1": float(row.get("test_macro_f1", row.get("macro_f1", np.nan))),
                "test_acc": float(row.get("test_accuracy", row.get("accuracy", np.nan))),
                "test_weighted_f1": float(row.get("test_weighted_f1", np.nan)),
            }
        )
    if check_path.exists():
        try:
            check = json.loads(check_path.read_text(encoding="utf-8"))
            for key in ["effective_slots_mean", "slot_overlap_mean", "slot_entropy_mean", "slot_dominance_mean"]:
                if key in check:
                    context[key] = check[key]
        except Exception:
            pass
    return context


@torch.no_grad()
def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    run_name = _sanitize(args.run_name, output_dir.name)
    names = _names(run_name)
    figures_dir = output_dir / "figures"
    masks_dir = output_dir / "masks"
    figures_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    config.setdefault("data", {})["num_workers"] = int(args.num_workers)
    config.setdefault("data", {})["chunk_cache_size"] = int(args.chunk_cache_size)
    device = resolve_device(args.device, config)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    model = _load_model(config, args.checkpoint, device)

    requested_ids = _read_sample_ids(args.sample_id_sheet)
    requested_set = set(requested_ids)
    samples_per_class = int(args.samples_per_class)
    candidates: List[Dict[str, Any]] = []
    counts = {idx: 0 for idx in range(len(EMOTION_NAMES))}
    max_scan = None if requested_set else samples_per_class * 8

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        slot_attention = _to_numpy(out["slot_attention"].float())
        slot_embeddings = _to_numpy(out["slot_embeddings"].float()) if torch.is_tensor(out.get("slot_embeddings")) else None
        readout = out.get("aux", {}).get("slot_readout_weights")
        readout_np = _to_numpy(readout.float()) if torch.is_tensor(readout) else None
        payloads = out.get("aux", {}).get("assignment_maps", [])
        y = batch["y"].detach().cpu().long()

        for i in range(int(logits.shape[0])):
            sample_id = str(_value_at(batch, "graph_id", i, fallback=_value_at(batch, "sample_idx", i, i)))
            label = int(y[i].item())
            if requested_set and sample_id not in requested_set:
                continue
            if not requested_set and counts[label] >= max_scan:
                continue
            image = _extract_image(batch, i)
            attn_i = slot_attention[i]
            emb_i = slot_embeddings[i] if slot_embeddings is not None else None
            if i < len(payloads):
                slot_maps, projection_method = _project_assignment(payloads[i], attn_i)
            else:
                slot_maps, projection_method = _project_grid(attn_i)
            combined = _combined_map(slot_maps, args.combined_method)
            slot_importance = readout_np[i] if readout_np is not None else attn_i.max(axis=1)
            stats = _slot_sample_stats(attn_i, emb_i)
            centers = _slot_centers(slot_maps)
            pred_i = int(pred[i].detach().cpu().item())
            conf_i = float(conf[i].detach().cpu().item())
            candidates.append(
                {
                    "sample_id": sample_id,
                    "split": args.split,
                    "label_index": label,
                    "pred_index": pred_i,
                    "confidence": conf_i,
                    "correct": bool(label == pred_i),
                    "image": image,
                    "slot_attention": attn_i.astype(np.float32),
                    "slot_pixel_maps": slot_maps.astype(np.float32),
                    "combined_slot_map": combined.astype(np.float32),
                    "slot_importance": slot_importance.astype(np.float32),
                    "slot_centers": centers.astype(np.float32),
                    "projection_method": projection_method,
                    "stats": stats,
                }
            )
            counts[label] += 1
        if requested_set and len({str(c["sample_id"]) for c in candidates}) >= len(requested_set):
            break
        if not requested_set and all(v >= int(max_scan or 0) for v in counts.values()):
            break

    selected = _selected_from_candidates(candidates, requested_ids, samples_per_class)
    rows = []
    index_lines = [
        f"# D13B {run_name} Visual Slot Audit Index",
        "",
        "| sample_id | label | pred | confidence | correct | figure | review_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in selected:
        sample_id = str(item["sample_id"])
        safe_id = sample_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        label_name = EMOTION_NAMES[int(item["label_index"])]
        pred_name = EMOTION_NAMES[int(item["pred_index"])]
        correct_flag = 1 if item["correct"] else 0
        fig_name = f"sample_{safe_id}_label_{label_name}_pred_{pred_name}_correct_{correct_flag}.png"
        fig_path = figures_dir / fig_name
        sample_dir = masks_dir / f"sample_{safe_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / "slot_attention.npy", item["slot_attention"])
        np.save(sample_dir / "slot_pixel_maps.npy", item["slot_pixel_maps"])
        np.save(sample_dir / "combined_slot_map.npy", item["combined_slot_map"])
        np.save(sample_dir / "slot_importance.npy", item["slot_importance"])
        np.save(sample_dir / "slot_centers.npy", item["slot_centers"])
        (sample_dir / "slot_stats.json").write_text(json.dumps(item["stats"], indent=2), encoding="utf-8")
        metadata = {
            "sample_id": sample_id,
            "split": item["split"],
            "label": label_name,
            "label_index": int(item["label_index"]),
            "pred": pred_name,
            "pred_index": int(item["pred_index"]),
            "confidence": float(item["confidence"]),
            "correct": bool(item["correct"]),
            "run_name": run_name,
            "num_slots": int(item["slot_attention"].shape[0]),
            "figure_path": str(fig_path.relative_to(output_dir)),
            "slot_attention_path": str((sample_dir / "slot_attention.npy").relative_to(output_dir)),
            "slot_pixel_maps_path": str((sample_dir / "slot_pixel_maps.npy").relative_to(output_dir)),
            "combined_slot_map_path": str((sample_dir / "combined_slot_map.npy").relative_to(output_dir)),
            "projection_method": item["projection_method"],
            "combined_slot_map_method": args.combined_method,
            **item["stats"],
            "notes_auto": "Visual slot audit only; no motif, semantic-region, or causal-evidence claim.",
        }
        metadata_path = sample_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _save_figure(
            fig_path,
            item["image"],
            item["combined_slot_map"],
            item["slot_pixel_maps"],
            item["slot_attention"],
            item["slot_importance"],
            item["slot_centers"],
            metadata,
        )
        row = {
            "sample_id": sample_id,
            "split": item["split"],
            "label": label_name,
            "pred": pred_name,
            "confidence": float(item["confidence"]),
            "correct": bool(item["correct"]),
            "run_name": run_name,
            "num_slots": int(item["slot_attention"].shape[0]),
            "figure_path": str(fig_path.relative_to(output_dir)),
            "metadata_path": str(metadata_path.relative_to(output_dir)),
            "review_status": "NEEDS_MANUAL_REVIEW",
            "visual_status": "",
            "slot_traceability_score": "",
            "slot_diversity_visual_score": "",
            "slot_face_coverage_score": "",
            "slot_assignment_readability_score": "",
            "mouth_only_risk": "",
            "center_shortcut_risk": "",
            "hair_glasses_risk": "",
            "background_border_risk": "",
            "dominant_slot_area": "",
            "multi_region_support": "",
            "slot_collapse_visual": "",
            "notes": "",
        }
        rows.append(row)
        index_lines.append(
            f"| {sample_id} | {label_name} | {pred_name} | {item['confidence']:.3f} | {item['correct']} | "
            f"[{fig_name}]({Path(row['figure_path']).as_posix()}) | NEEDS_MANUAL_REVIEW |"
        )

    pd.DataFrame(rows, columns=SHEET_COLUMNS).to_csv(output_dir / names["sheet"], index=False)
    (output_dir / names["index"]).write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    _write_instructions(output_dir / names["instructions"], run_name)
    summary = {
        "mode": "prepare",
        "run_name": run_name,
        "split": args.split,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "samples_per_class": samples_per_class,
        "samples_saved": int(len(rows)),
        "sample_id_sheet": str(args.sample_id_sheet) if args.sample_id_sheet else None,
        "sample_ids_reused": bool(requested_ids),
        "projection_methods": sorted({str(r["projection_method"]) for r in selected}),
        "metric_context": _metric_context(args.checkpoint),
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
        "no_causal_claim": True,
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _write_instructions(path: Path, run_name: str) -> None:
    path.write_text(
        f"""# D13B {run_name} Visual Slot Audit Instructions

This is a visual slot audit for D13B diagnostic slot candidates only.
Do not claim motif discovery, semantic regions, or causal evidence.

## visual_status
- PASS: slot maps traceable, cover facial areas, multiple slots show distinguishable support, not dominated by mouth/center/background.
- PARTIAL: traceable but somewhat mouth-heavy/center-heavy, slots partly repeated, or assignment/slot maps somewhat diffuse.
- FAIL: slot maps unreadable, severe collapse, mostly mouth-only/center shortcut/background/border/hair/glasses.

## Scores
- slot_traceability_score: 0 = cannot trace slot to pixel/region; 1 = traceable but blurry/noisy; 2 = clearly traceable.
- slot_diversity_visual_score: 0 = slots nearly identical/collapsed; 1 = somewhat different but overlapping; 2 = several slots focus on visibly different areas.
- slot_face_coverage_score: 0 = mostly outside face; 1 = partly face but noisy; 2 = mostly facial support.
- slot_assignment_readability_score: 0 = too diffuse/hard to read; 1 = diffuse but readable; 2 = clear and spatially interpretable.

## Risks
- mouth_only_risk: 0 = negligible; 1 = mildly mouth-heavy; 2 = dominant mouth-only/lower-face shortcut.
- center_shortcut_risk: 0 = negligible; 1 = mildly center-heavy; 2 = clear center shortcut.
- hair_glasses_risk: 0 = negligible; 1 = present but not dominant; 2 = dominant or severe.
- background_border_risk: 0 = negligible; 1 = present but not dominant; 2 = dominant background/border/face contour.

## dominant_slot_area suggestions
mouth, eyes, eyebrows, nose_cheek, forehead, face_contour, full_face, hair, glasses, background, border, mixed, unclear.

## multi_region_support
0 = slot mostly single vague blob or unreadable.
1 = some multi-region support but weak.
2 = multiple meaningful facial regions visible.

## slot_collapse_visual
0 = no obvious collapse.
1 = mild repeated slot maps.
2 = severe slot collapse / all slots same area.
""",
        encoding="utf-8",
    )


def _read_sheet(audit_dir: Path, review_sheet: str | None, run_name: str) -> Tuple[pd.DataFrame, Path, bool]:
    names = _names(run_name)
    if review_sheet:
        path = Path(review_sheet)
        explicit = True
    else:
        filled = audit_dir / names["filled"]
        path = filled if filled.exists() else audit_dir / names["sheet"]
        explicit = filled.exists()
    if not path.exists():
        raise FileNotFoundError(f"Review sheet not found: {path}")
    return pd.read_csv(path), path, explicit


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _rate(count: int, total: int) -> float:
    return float(count / max(int(total), 1))


def _status_counts(df: pd.DataFrame) -> Dict[str, int]:
    status = df.get("visual_status", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    return {
        "pass_count": int((status == "PASS").sum()),
        "partial_count": int((status == "PARTIAL").sum()),
        "fail_count": int((status == "FAIL").sum()),
        "reviewed_samples": int(status.isin(["PASS", "PARTIAL", "FAIL"]).sum()),
    }


def _summary_row(df: pd.DataFrame, incomplete: bool) -> Dict[str, Any]:
    total = int(len(df))
    counts = _status_counts(df)
    reviewed = counts["reviewed_samples"]
    row = {
        "total_samples": total,
        "reviewed_samples": reviewed,
        "unreviewed_samples": total - reviewed,
        **{k: counts[k] for k in ["pass_count", "partial_count", "fail_count"]},
        "pass_rate": _rate(counts["pass_count"], total),
        "partial_rate": _rate(counts["partial_count"], total),
        "fail_rate": _rate(counts["fail_count"], total),
        "pass_partial_count": counts["pass_count"] + counts["partial_count"],
        "pass_partial_rate": _rate(counts["pass_count"] + counts["partial_count"], total),
        "avg_slot_traceability_score": _mean(df, "slot_traceability_score"),
        "avg_slot_diversity_visual_score": _mean(df, "slot_diversity_visual_score"),
        "avg_slot_face_coverage_score": _mean(df, "slot_face_coverage_score"),
        "avg_slot_assignment_readability_score": _mean(df, "slot_assignment_readability_score"),
        "avg_mouth_only_risk": _mean(df, "mouth_only_risk"),
        "avg_center_shortcut_risk": _mean(df, "center_shortcut_risk"),
        "avg_hair_glasses_risk": _mean(df, "hair_glasses_risk"),
        "avg_background_border_risk": _mean(df, "background_border_risk"),
        "avg_multi_region_support": _mean(df, "multi_region_support"),
        "avg_slot_collapse_visual": _mean(df, "slot_collapse_visual"),
        "audit_incomplete": bool(incomplete or reviewed < total),
    }
    return row


def _gate(row: Dict[str, Any]) -> str:
    if row.get("audit_incomplete") or int(row.get("reviewed_samples", 0)) != int(row.get("total_samples", 0)):
        return "VISUAL_SLOT_AUDIT_INCOMPLETE_KEEP_D13C_LOCKED"
    checks = [
        row.get("pass_partial_rate", 0.0) >= 0.60,
        row.get("avg_slot_traceability_score", 0.0) >= 1.2,
        row.get("avg_slot_diversity_visual_score", 0.0) >= 1.2,
        row.get("avg_slot_face_coverage_score", 0.0) >= 1.2,
        row.get("avg_slot_assignment_readability_score", 0.0) >= 1.2,
        row.get("avg_mouth_only_risk", 9.0) < 1.2,
        row.get("avg_center_shortcut_risk", 9.0) < 1.0,
        row.get("avg_hair_glasses_risk", 9.0) < 1.0,
        row.get("avg_background_border_risk", 9.0) < 1.0,
        row.get("avg_slot_collapse_visual", 9.0) < 1.0,
    ]
    if all(checks):
        return "D13B_VISUAL_SLOT_ACCEPTABLE_FOR_D13C_DIAGNOSTIC"
    if row.get("pass_partial_rate", 0.0) >= 0.45 and row.get("avg_slot_traceability_score", 0.0) >= 1.0:
        return "D13B_VISUAL_SLOT_PARTIAL_USE_WITH_CAUTION"
    return "D13B_VISUAL_SLOT_UNRELIABLE_STOP_D13C"


def _area_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for area, group in df.groupby(df.get("dominant_slot_area", pd.Series(dtype=str)).fillna("").replace("", "unreviewed")):
        counts = _status_counts(group)
        rows.append(
            {
                "dominant_slot_area": area,
                "count": int(len(group)),
                "rate": _rate(len(group), total),
                "pass_count": counts["pass_count"],
                "partial_count": counts["partial_count"],
                "fail_count": counts["fail_count"],
                "avg_slot_traceability_score": _mean(group, "slot_traceability_score"),
                "avg_slot_diversity_visual_score": _mean(group, "slot_diversity_visual_score"),
                "avg_mouth_only_risk": _mean(group, "mouth_only_risk"),
                "avg_center_shortcut_risk": _mean(group, "center_shortcut_risk"),
                "avg_slot_collapse_visual": _mean(group, "slot_collapse_visual"),
            }
        )
    return pd.DataFrame(rows)


def _risk_cases(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    status_fail = work.get("visual_status", pd.Series(dtype=str)).astype(str).str.upper().eq("FAIL")
    risk = status_fail.copy()
    thresholds = {
        "mouth_only_risk": 2,
        "center_shortcut_risk": 2,
        "hair_glasses_risk": 2,
        "background_border_risk": 2,
        "slot_traceability_score": 0,
        "slot_diversity_visual_score": 0,
        "slot_assignment_readability_score": 0,
        "slot_collapse_visual": 2,
    }
    for col, threshold in thresholds.items():
        vals = pd.to_numeric(work.get(col, pd.Series(index=work.index, dtype=float)), errors="coerce")
        if col.startswith("slot_") and col not in {"slot_collapse_visual"}:
            risk = risk | vals.eq(threshold)
        else:
            risk = risk | vals.ge(threshold)
    return work.loc[risk, [c for c in RISK_COLUMNS if c in work.columns]]


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def summarize(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    run_name = _sanitize(args.run_name, audit_dir.name)
    names = _names(run_name)
    df, sheet_path, explicit_filled = _read_sheet(audit_dir, args.review_sheet, run_name)
    incomplete = not explicit_filled
    summary = _summary_row(df, incomplete)
    decision = _gate(summary)
    summary["decision"] = decision
    summary["run_name"] = run_name
    summary_df = pd.DataFrame([summary])
    area_df = _area_summary(df)
    risk_df = _risk_cases(df)
    summary_df.to_csv(audit_dir / names["summary"], index=False)
    area_df.to_csv(audit_dir / names["area"], index=False)
    risk_df.to_csv(audit_dir / names["risk"], index=False)

    prep = {}
    prep_path = audit_dir / "prepare_summary.json"
    if prep_path.exists():
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
    ctx = prep.get("metric_context", {})
    lines = [
        f"# D13B Visual Slot Audit Report: {run_name}",
        "",
        "## 1. Context",
        "- D13B diagnostic only.",
        f"- run_name: `{run_name}`.",
        f"- test_macro_f1: {ctx.get('test_macro_f1', 'unknown')}.",
        f"- test_acc: {ctx.get('test_acc', 'unknown')}.",
        f"- num_slots: {df['num_slots'].iloc[0] if 'num_slots' in df and len(df) else 'unknown'}.",
        f"- effective_slots: {ctx.get('effective_slots_mean', 'unknown')}.",
        f"- slot_overlap: {ctx.get('slot_overlap_mean', 'unknown')}.",
        f"- slot_entropy: {ctx.get('slot_entropy_mean', 'unknown')}.",
        "- D13A base: K144 ep100 visual base with caution unless this is a score-control run.",
        "- No motif claim.",
        "",
        "## 2. Audit Goal",
        "Check slot traceability, diversity, shortcut risk before D13C diagnostic.",
        "",
        "## 3. Manual/AI-assisted Visual Audit Summary",
        _md_table(summary_df),
        "",
        "## 4. Slot Traceability and Diversity",
        f"- avg_slot_traceability_score: {summary.get('avg_slot_traceability_score')}",
        f"- avg_slot_diversity_visual_score: {summary.get('avg_slot_diversity_visual_score')}",
        f"- avg_slot_face_coverage_score: {summary.get('avg_slot_face_coverage_score')}",
        f"- avg_slot_assignment_readability_score: {summary.get('avg_slot_assignment_readability_score')}",
        f"- avg_multi_region_support: {summary.get('avg_multi_region_support')}",
        f"- avg_slot_collapse_visual: {summary.get('avg_slot_collapse_visual')}",
        "",
        "## 5. Shortcut/Risk Analysis",
        f"- avg_mouth_only_risk: {summary.get('avg_mouth_only_risk')}",
        f"- avg_center_shortcut_risk: {summary.get('avg_center_shortcut_risk')}",
        f"- avg_hair_glasses_risk: {summary.get('avg_hair_glasses_risk')}",
        f"- avg_background_border_risk: {summary.get('avg_background_border_risk')}",
        "",
        "Risk cases:",
        _md_table(risk_df, max_rows=30),
        "",
        "## 6. Area Distribution",
        _md_table(area_df),
        "",
        "## 7. Gate Check",
        decision,
        "",
        "Do not output OPEN_D13C_FULL, OPEN_SUPCON, or motif claims from this report.",
        "",
    ]
    (audit_dir / names["report"]).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit_dir": str(audit_dir), "review_sheet": str(sheet_path), "decision": decision}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or summarize D13B visual slot audits.")
    parser.add_argument("--mode", choices=["prepare", "summarize"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir")
    parser.add_argument("--audit_dir")
    parser.add_argument("--review_sheet")
    parser.add_argument("--samples_per_class", type=int, default=15)
    parser.add_argument("--sample_id_sheet")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--environment", choices=["local", "kaggle"], default="local")
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chunk_cache_size", type=int, default=2)
    parser.add_argument("--combined_method", choices=["max", "sum_normalized"], default="max")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "prepare":
        for attr in ("config", "checkpoint", "output_dir"):
            if getattr(args, attr) is None:
                parser.error(f"--{attr} is required for prepare")
        prepare(args)
    else:
        if args.audit_dir is None:
            parser.error("--audit_dir is required for summarize")
        summarize(args)


if __name__ == "__main__":
    main()
