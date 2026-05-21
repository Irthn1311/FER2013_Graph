"""D13D slot evidence diagnostics for D13C candidates.

This script runs deletion/control/stability probes on D13C slot candidates.
It is diagnostic-only: no training, no full D13C, no prototype, no motif-level
SupCon, no motif claim, and no causal-evidence claim.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage_d13c_post_visual_slot_audit import (
    EMOTION_NAMES,
    _combined_map,
    _extract_image,
    _load_checkpoint,
    _normalize,
    _overlay,
    _project_assignment,
    _project_grid,
    _read_sample_ids,
    _selected_from_candidates,
    _slot_centers,
    _slot_sample_stats,
    _to_numpy,
    _value_at,
)


HARD_CLASSES = {"Angry", "Disgust", "Fear", "Sad"}
CONDITIONS = [
    "original",
    "delete_top1_slot",
    "delete_top3_slots",
    "delete_random1_slot",
    "delete_random3_slots",
    "delete_center_slot_control",
    "delete_mouth_like_slot_control",
    "delete_low_importance_slot",
]
RANDOM_REPEATS = 5


def _sanitize(value: str | None, fallback: str = "run") -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _safe_sample_id(sample_id: Any) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_model(config: Dict[str, Any], checkpoint: str | Path, device: Any) -> Any:
    from models.d13c_supcon_model import D13CSupConModel

    model = D13CSupConModel.from_config(config.get("model", {})).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    state = ckpt.get("model_state_dict", ckpt)
    clean_state = {str(k)[7:] if str(k).startswith("module.") else str(k): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    if hasattr(model.reduction, "set_save_visualization"):
        model.reduction.set_save_visualization(True)
    model.eval()
    return model


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.nanmax(x)
    exp = np.exp(x)
    return (exp / np.clip(exp.sum(), 1e-12, None)).astype(np.float32)


def _entropy(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _slot_mass(mask: np.ndarray, region: np.ndarray) -> float:
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, None)
    total = float(work.sum())
    if total <= 1e-8:
        return 0.0
    return float(work[region].sum() / total)


def _regions() -> Dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:48, 0:48]
    return {
        "mouth": (xx >= 14) & (xx <= 34) & (yy >= 29) & (yy <= 41),
        "center": (xx >= 16) & (xx <= 32) & (yy >= 16) & (yy <= 32),
        "border": (xx < 4) | (xx >= 44) | (yy < 4) | (yy >= 44),
        "eyes": (xx >= 8) & (xx <= 40) & (yy >= 13) & (yy <= 23),
        "nose_cheek": (xx >= 11) & (xx <= 37) & (yy >= 21) & (yy <= 32),
        "forehead": (xx >= 13) & (xx <= 35) & (yy >= 5) & (yy <= 14),
    }


REGIONS = _regions()


def _dominant_area(mask: np.ndarray) -> str:
    norm = _normalize(mask)
    masses = {name: _slot_mass(norm, region) for name, region in REGIONS.items()}
    if masses["border"] >= 0.38:
        return "border"
    facial = {k: v for k, v in masses.items() if k not in {"border", "center"}}
    ordered = sorted(facial.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] < 0.16:
        return "mixed"
    if len(ordered) > 1 and ordered[1][1] >= ordered[0][1] * 0.82:
        return "mixed"
    return ordered[0][0]


def _mouth_slot(slot_maps: np.ndarray) -> int:
    masses = [_slot_mass(item, REGIONS["mouth"]) for item in slot_maps]
    return int(np.argmax(masses))


def _center_slot(slot_maps: np.ndarray) -> int:
    centers = _slot_centers(slot_maps)
    dist = np.linalg.norm(centers - np.asarray([[23.5, 23.5]], dtype=np.float32), axis=1)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    return int(np.argmin(dist))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(x, y) / denom)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if float(x.std()) <= 1e-8 or float(y.std()) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _load_config_runtime(args: argparse.Namespace) -> Tuple[Dict[str, Any], Any, Any, Any]:
    import torch
    from common import apply_cli_overrides, build_dataloader, load_config, resolve_device

    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    config.setdefault("data", {})["num_workers"] = int(args.num_workers)
    config.setdefault("data", {})["chunk_cache_size"] = int(args.chunk_cache_size)
    device = resolve_device(args.device, config)
    model = _load_model(config, args.checkpoint, device)
    return config, model, device, torch


def _classify_slots(model: Any, slots: Any, delete_slots: Iterable[int] = ()) -> Tuple[Any, Any, Any]:
    import torch

    work = slots.clone()
    delete = sorted({int(i) for i in delete_slots if 0 <= int(i) < int(work.shape[1])})
    if delete:
        work[:, delete, :] = 0.0
    slot_mean = work.mean(dim=1)
    slot_max = work.max(dim=1).values
    slot_attn_pool, slot_readout_weights = model.slot_readout(work)
    z_image = torch.cat([slot_mean, slot_max, slot_attn_pool], dim=-1)
    logits = model.classifier(z_image)
    return logits, z_image, slot_readout_weights


def _condition_row(
    sample: Dict[str, Any],
    condition: str,
    repeat_id: int,
    deleted_slots: List[int],
    logits: np.ndarray,
) -> Dict[str, Any]:
    prob = _softmax_np(logits)
    original_prob = np.asarray(sample["original_prob"], dtype=np.float32)
    label_idx = int(sample["label_index"])
    original_pred_idx = int(sample["pred_index"])
    cond_pred_idx = int(np.argmax(prob))
    return {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "original_pred": sample["pred"],
        "condition": condition,
        "repeat_id": int(repeat_id),
        "deleted_slots": ";".join(str(i) for i in deleted_slots),
        "original_true_prob": float(original_prob[label_idx]),
        "condition_true_prob": float(prob[label_idx]),
        "drop_true_prob": float(original_prob[label_idx] - prob[label_idx]),
        "original_pred_prob": float(original_prob[original_pred_idx]),
        "condition_original_pred_prob": float(prob[original_pred_idx]),
        "drop_original_pred_prob": float(original_prob[original_pred_idx] - prob[original_pred_idx]),
        "condition_pred": EMOTION_NAMES[cond_pred_idx],
        "prediction_changed": bool(cond_pred_idx != original_pred_idx),
        "confidence": float(prob[cond_pred_idx]),
        "logit_true_class": float(logits[label_idx]),
        "entropy": _entropy(prob),
        "notes": "D13D diagnostic deletion/control probe only; not causal evidence.",
    }


def _sample_figure(path: Path, sample: Dict[str, Any], deletion_rows: List[Dict[str, Any]]) -> None:
    slot_maps = sample["slot_pixel_maps"]
    top = int(sample["top_slot_ids"][0])
    rng = int(sample.get("random1_slot", top))
    center = int(sample["center_slot_id"])
    mouth = int(sample["mouth_slot_id"])
    rows = pd.DataFrame(deletion_rows)
    if "condition" in rows.columns:
        focus = rows[rows["condition"].isin(["delete_top1_slot", "delete_random1_slot", "delete_center_slot_control", "delete_mouth_like_slot_control", "delete_low_importance_slot"])]
    else:
        focus = pd.DataFrame()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 4)
    panels = [
        ("Original", sample["image"]),
        (f"top slot {top}", slot_maps[top]),
        (f"random slot {rng}", slot_maps[rng]),
        (f"center slot {center}", slot_maps[center]),
        (f"mouth slot {mouth}", slot_maps[mouth]),
    ]
    for idx, (title, arr) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        if idx == 0:
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(_overlay(sample["image"], arr))
        ax.set_title(title)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, 1:])
    if not focus.empty:
        labels = focus["condition"].astype(str).tolist()
        vals = pd.to_numeric(focus["drop_true_prob"], errors="coerce").to_numpy()
        ax.barh(labels, vals)
        ax.set_xlabel("drop_true_prob")
    ax.set_title(f"{sample['sample_id']} label={sample['label']} pred={sample['pred']} conf={sample['confidence']:.3f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _augment_batch(batch: Dict[str, Any], kind: str, torch_mod: Any) -> Dict[str, Any]:
    out = dict(batch)
    x = batch["x"].clone()
    pix = x[..., 0].reshape(x.shape[0], 48, 48)
    if kind == "hflip":
        pix_aug = torch_mod.flip(pix, dims=[2])
    elif kind == "brightness":
        pix_aug = torch_mod.clamp(pix * 1.08 + 0.02, 0.0, 1.0)
    elif kind == "noise":
        gen = torch_mod.Generator(device=pix.device)
        gen.manual_seed(1307)
        noise = torch_mod.randn(pix.shape, generator=gen, device=pix.device, dtype=pix.dtype) * 0.025
        pix_aug = torch_mod.clamp(pix + noise, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown augmentation: {kind}")
    x[..., 0] = pix_aug.reshape(x.shape[0], -1)
    out["x"] = x
    return out


def _project_outputs(out: Dict[str, Any], index: int, combined_method: str = "max") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    slot_attention = _to_numpy(out["slot_attention"].float())[index]
    payloads = out.get("aux", {}).get("assignment_maps", [])
    if index < len(payloads):
        slot_maps, _ = _project_assignment(payloads[index], slot_attention)
    else:
        slot_maps, _ = _project_grid(slot_attention)
    combined = _combined_map(slot_maps, combined_method)
    return slot_attention.astype(np.float32), slot_maps.astype(np.float32), combined.astype(np.float32)


def _prepare_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# D13D Evidence Diagnostic Prepare Report",
        "",
        f"- run_name: `{summary['run_name']}`",
        f"- split: `{summary['split']}`",
        f"- samples_saved: {summary['samples_saved']}",
        f"- projection_methods: {', '.join(summary['projection_methods'])}",
        "- diagnostic only; no motif claim; no causal-evidence claim.",
        "",
    ]
    (output_dir / "d13d_prepare_report.md").write_text("\n".join(lines), encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    import torch
    from common import build_dataloader
    from training.trainer import move_to_device

    output_dir = Path(args.output_dir)
    run_name = _sanitize(args.run_name, output_dir.name)
    for sub in ["metadata", "masks", "figures"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    config, model, device, _ = _load_config_runtime(args)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    requested_ids = _read_sample_ids(args.sample_id_sheet)
    requested_set = set(requested_ids)
    max_scan = None if requested_set else int(args.samples_per_class) * 8
    counts = {idx: 0 for idx in range(len(EMOTION_NAMES))}
    candidates: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out = model(batch)
            probs = torch.softmax(out["logits"], dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            slot_attention_np = _to_numpy(out["slot_attention"].float())
            slot_embeddings_np = _to_numpy(out["slot_embeddings"].float())
            z_image_np = _to_numpy(out["z_image"].float())
            z_proj_np = _to_numpy(out["z_proj"].float())
            logits_np = _to_numpy(out["logits"].float())
            probs_np = _to_numpy(probs.float())
            readout = out.get("aux", {}).get("slot_readout_weights")
            readout_np = _to_numpy(readout.float()) if torch.is_tensor(readout) else None
            payloads = out.get("aux", {}).get("assignment_maps", [])
            y = batch["y"].detach().cpu().long()
            for i in range(int(out["logits"].shape[0])):
                sample_id = str(_value_at(batch, "graph_id", i, fallback=_value_at(batch, "sample_idx", i, i)))
                label = int(y[i].item())
                if requested_set and sample_id not in requested_set:
                    continue
                if not requested_set and counts[label] >= max_scan:
                    continue
                attn_i = slot_attention_np[i]
                if i < len(payloads):
                    slot_maps, projection_method = _project_assignment(payloads[i], attn_i)
                else:
                    slot_maps, projection_method = _project_grid(attn_i)
                importance = readout_np[i] if readout_np is not None else np.linalg.norm(slot_embeddings_np[i], axis=1)
                top_ids = np.argsort(importance)[::-1].astype(np.int64)
                pred_i = int(pred[i].detach().cpu().item())
                candidates.append(
                    {
                        "sample_id": sample_id,
                        "split": args.split,
                        "label_index": label,
                        "label": EMOTION_NAMES[label],
                        "pred_index": pred_i,
                        "pred": EMOTION_NAMES[pred_i],
                        "confidence": float(conf[i].detach().cpu().item()),
                        "correct": bool(label == pred_i),
                        "image": _extract_image(batch, i),
                        "original_logits": logits_np[i].astype(np.float32),
                        "original_prob": probs_np[i].astype(np.float32),
                        "slot_attention": attn_i.astype(np.float32),
                        "slot_embeddings": slot_embeddings_np[i].astype(np.float32),
                        "z_image": z_image_np[i].astype(np.float32),
                        "z_proj": z_proj_np[i].astype(np.float32),
                        "slot_pixel_maps": slot_maps.astype(np.float32),
                        "combined_slot_map": _combined_map(slot_maps, args.combined_method),
                        "slot_importance": importance.astype(np.float32),
                        "top_slot_ids": top_ids,
                        "center_slot_id": _center_slot(slot_maps),
                        "mouth_slot_id": _mouth_slot(slot_maps),
                        "projection_method": projection_method,
                        "stats": _slot_sample_stats(attn_i, slot_embeddings_np[i]),
                    }
                )
                counts[label] += 1
            if requested_set and len({str(c["sample_id"]) for c in candidates}) >= len(requested_set):
                break
            if not requested_set and all(v >= int(max_scan or 0) for v in counts.values()):
                break

    selected = _selected_from_candidates(candidates, requested_ids, int(args.samples_per_class))
    rows = []
    for item in selected:
        sid = str(item["sample_id"])
        safe = _safe_sample_id(sid)
        sample_mask_dir = output_dir / "masks" / f"sample_{safe}"
        sample_meta_dir = output_dir / "metadata" / f"sample_{safe}"
        sample_mask_dir.mkdir(parents=True, exist_ok=True)
        sample_meta_dir.mkdir(parents=True, exist_ok=True)
        for key in [
            "original_logits",
            "original_prob",
            "slot_attention",
            "slot_embeddings",
            "slot_pixel_maps",
            "combined_slot_map",
            "slot_importance",
            "top_slot_ids",
            "z_image",
            "z_proj",
        ]:
            np.save(sample_mask_dir / f"{key}.npy", item[key])
        metadata = {
            "sample_id": sid,
            "split": item["split"],
            "label": item["label"],
            "label_index": int(item["label_index"]),
            "pred": item["pred"],
            "pred_index": int(item["pred_index"]),
            "confidence": float(item["confidence"]),
            "correct": bool(item["correct"]),
            "run_name": run_name,
            "projection_method": item["projection_method"],
            "top_slot_ids": [int(v) for v in item["top_slot_ids"].tolist()],
            "center_slot_id": int(item["center_slot_id"]),
            "mouth_slot_id": int(item["mouth_slot_id"]),
            "dominant_area_top1": _dominant_area(item["slot_pixel_maps"][int(item["top_slot_ids"][0])]),
            "mouth_mass_top1": _slot_mass(item["slot_pixel_maps"][int(item["top_slot_ids"][0])], REGIONS["mouth"]),
            "center_mass_top1": _slot_mass(item["slot_pixel_maps"][int(item["top_slot_ids"][0])], REGIONS["center"]),
            "mask_dir": str(sample_mask_dir.relative_to(output_dir)),
            "metadata_path": str((sample_meta_dir / "metadata.json").relative_to(output_dir)),
            **item["stats"],
            "notes": "D13D diagnostic importance proxy only; not motif or causal evidence.",
        }
        (sample_meta_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        fig_path = output_dir / "figures" / f"sample_{safe}_prepare.png"
        _sample_figure(fig_path, item, [])
        rows.append(
            {
                "sample_id": sid,
                "split": item["split"],
                "label": item["label"],
                "pred": item["pred"],
                "confidence": float(item["confidence"]),
                "correct": bool(item["correct"]),
                "top1_slot": int(item["top_slot_ids"][0]),
                "top3_slots": ";".join(str(int(v)) for v in item["top_slot_ids"][:3]),
                "center_slot": int(item["center_slot_id"]),
                "mouth_slot": int(item["mouth_slot_id"]),
                "dominant_area_top1": metadata["dominant_area_top1"],
                "mouth_mass_top1": metadata["mouth_mass_top1"],
                "center_mass_top1": metadata["center_mass_top1"],
                "projection_method": item["projection_method"],
                "metadata_path": metadata["metadata_path"],
                "mask_dir": metadata["mask_dir"],
                "figure_path": str(fig_path.relative_to(output_dir)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "d13d_sample_index.csv", index=False)
    prep = {
        "mode": "prepare",
        "run_name": run_name,
        "split": args.split,
        "samples_saved": int(len(rows)),
        "samples_per_class": int(args.samples_per_class),
        "sample_id_sheet": str(args.sample_id_sheet) if args.sample_id_sheet else None,
        "projection_methods": sorted({str(r["projection_method"]) for r in selected}),
        "no_motif_claim": True,
        "no_causal_claim": True,
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
    _prepare_report(output_dir, prep)
    print(json.dumps(prep, indent=2))


def _load_sample_from_index(audit_dir: Path, row: pd.Series) -> Dict[str, Any]:
    meta = json.loads((audit_dir / str(row["metadata_path"])).read_text(encoding="utf-8"))
    mask_dir = audit_dir / str(row["mask_dir"])
    sample = dict(meta)
    for key in [
        "original_logits",
        "original_prob",
        "slot_attention",
        "slot_embeddings",
        "slot_pixel_maps",
        "combined_slot_map",
        "slot_importance",
        "top_slot_ids",
    ]:
        sample[key] = np.load(mask_dir / f"{key}.npy")
    fig = audit_dir / str(row.get("figure_path", ""))
    if fig.exists():
        sample["image"] = plt.imread(fig)[:, :, 0] if plt.imread(fig).ndim == 3 else plt.imread(fig)
    else:
        sample["image"] = np.zeros((48, 48), dtype=np.float32)
    return sample


def _collect_run_samples(args: argparse.Namespace, sample_ids: set[str]) -> Tuple[Dict[str, Dict[str, Any]], Any, Any, Any]:
    import torch
    from common import build_dataloader
    from training.trainer import move_to_device

    config, model, device, torch_mod = _load_config_runtime(args)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    samples: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out = model(batch)
            probs = torch.softmax(out["logits"], dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            slot_attention_np = _to_numpy(out["slot_attention"].float())
            slot_embeddings_t = out["slot_embeddings"].float()
            slot_embeddings_np = _to_numpy(slot_embeddings_t)
            readout = out.get("aux", {}).get("slot_readout_weights")
            readout_np = _to_numpy(readout.float()) if torch.is_tensor(readout) else None
            payloads = out.get("aux", {}).get("assignment_maps", [])
            y = batch["y"].detach().cpu().long()
            for i in range(int(out["logits"].shape[0])):
                sid = str(_value_at(batch, "graph_id", i, fallback=_value_at(batch, "sample_idx", i, i)))
                if sid not in sample_ids:
                    continue
                attn_i = slot_attention_np[i]
                if i < len(payloads):
                    slot_maps, _ = _project_assignment(payloads[i], attn_i)
                else:
                    slot_maps, _ = _project_grid(attn_i)
                importance = readout_np[i] if readout_np is not None else np.linalg.norm(slot_embeddings_np[i], axis=1)
                top_ids = np.argsort(importance)[::-1].astype(np.int64)
                pred_i = int(pred[i].detach().cpu().item())
                samples[sid] = {
                    "sample_id": sid,
                    "batch": batch,
                    "batch_index": i,
                    "label_index": int(y[i].item()),
                    "label": EMOTION_NAMES[int(y[i].item())],
                    "pred_index": pred_i,
                    "pred": EMOTION_NAMES[pred_i],
                    "confidence": float(conf[i].detach().cpu().item()),
                    "slot_embeddings_t": slot_embeddings_t[i : i + 1],
                    "slot_embeddings": slot_embeddings_np[i],
                    "slot_attention": attn_i,
                    "slot_pixel_maps": slot_maps,
                    "combined_slot_map": _combined_map(slot_maps),
                    "slot_importance": importance,
                    "top_slot_ids": top_ids,
                    "center_slot_id": _center_slot(slot_maps),
                    "mouth_slot_id": _mouth_slot(slot_maps),
                    "original_logits": _to_numpy(out["logits"][i].float()),
                    "original_prob": _to_numpy(probs[i].float()),
                    "image": _extract_image(batch, i),
                }
            if len(samples) >= len(sample_ids):
                break
    return samples, model, device, torch_mod


def run(args: argparse.Namespace) -> None:
    import torch

    audit_dir = Path(args.audit_dir)
    index = pd.read_csv(audit_dir / "d13d_sample_index.csv")
    sample_ids = set(index["sample_id"].astype(str).tolist())
    args.split = str(index["split"].iloc[0]) if "split" in index.columns and len(index) else "test"
    samples, model, device, torch_mod = _collect_run_samples(args, sample_ids)
    deletion_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    risk_fig_dir = audit_dir / "figures" / "risk_cases"
    rng = np.random.default_rng(1307)

    for _, idx_row in index.iterrows():
        sid = str(idx_row["sample_id"])
        sample = samples[sid]
        slots_t = sample["slot_embeddings_t"]
        top_ids = [int(v) for v in sample["top_slot_ids"].tolist()]
        center_slot = int(sample["center_slot_id"])
        mouth_slot = int(sample["mouth_slot_id"])
        low_slot = int(np.argsort(sample["slot_importance"])[0])
        all_slots = np.arange(int(slots_t.shape[1]))
        random1 = int(rng.choice(all_slots))
        random3 = [int(v) for v in rng.choice(all_slots, size=min(3, len(all_slots)), replace=False).tolist()]
        sample["random1_slot"] = random1
        condition_specs = [
            ("original", 0, []),
            ("delete_top1_slot", 0, [top_ids[0]]),
            ("delete_top3_slots", 0, top_ids[:3]),
            ("delete_center_slot_control", 0, [center_slot]),
            ("delete_mouth_like_slot_control", 0, [mouth_slot]),
            ("delete_low_importance_slot", 0, [low_slot]),
        ]
        sample_rows: List[Dict[str, Any]] = []
        for condition, repeat_id, delete in condition_specs:
            logits, _, _ = _classify_slots(model, slots_t, delete)
            row = _condition_row(sample, condition, repeat_id, delete, _to_numpy(logits[0].float()))
            deletion_rows.append(row)
            sample_rows.append(row)
        for repeat_id in range(RANDOM_REPEATS):
            r1 = int(rng.choice(all_slots))
            r3 = [int(v) for v in rng.choice(all_slots, size=min(3, len(all_slots)), replace=False).tolist()]
            for condition, delete in [("delete_random1_slot", [r1]), ("delete_random3_slots", r3)]:
                logits, _, _ = _classify_slots(model, slots_t, delete)
                row = _condition_row(sample, condition, repeat_id, delete, _to_numpy(logits[0].float()))
                deletion_rows.append(row)
                sample_rows.append(row)
        sample["center_slot_id"] = center_slot
        sample["mouth_slot_id"] = mouth_slot
        _sample_figure(risk_fig_dir / f"sample_{_safe_sample_id(sid)}_deletion.png", sample, sample_rows)

    # Stability pass over weak augmentations.
    with torch.no_grad():
        for sid, sample in samples.items():
            batch = sample["batch"]
            i = int(sample["batch_index"])
            for aug in ["hflip", "brightness", "noise"]:
                aug_batch = _augment_batch(batch, aug, torch_mod)
                out_aug = model(aug_batch)
                prob_aug = torch.softmax(out_aug["logits"], dim=1)
                pred_aug = int(prob_aug.argmax(dim=1)[i].detach().cpu().item())
                _, aug_maps, aug_combined = _project_outputs(out_aug, i)
                orig_maps = sample["slot_pixel_maps"]
                orig_combined = sample["combined_slot_map"]
                if aug == "hflip":
                    compare_maps = np.flip(orig_maps, axis=2)
                    compare_combined = np.flip(orig_combined, axis=1)
                else:
                    compare_maps = orig_maps
                    compare_combined = orig_combined
                sims = [_cosine(compare_maps[j], aug_maps[j]) for j in range(min(compare_maps.shape[0], aug_maps.shape[0]))]
                top_orig = int(sample["top_slot_ids"][0])
                aug_importance = _to_numpy(out_aug.get("aux", {}).get("slot_readout_weights")[i].float())
                top_aug = int(np.argsort(aug_importance)[::-1][0])
                stability_rows.append(
                    {
                        "sample_id": sid,
                        "label": sample["label"],
                        "augmentation": aug,
                        "original_pred": sample["pred"],
                        "aug_pred": EMOTION_NAMES[pred_aug],
                        "prediction_consistent": bool(pred_aug == int(sample["pred_index"])),
                        "top_slot_consistent": bool(top_aug == top_orig),
                        "mean_slot_map_similarity": float(np.nanmean(sims)),
                        "combined_map_similarity": _corr(compare_combined, aug_combined),
                        "notes": "Weak augmentation stability diagnostic only; not causal evidence.",
                    }
                )

    deletion = pd.DataFrame(deletion_rows)
    stability = pd.DataFrame(stability_rows)
    deletion.to_csv(audit_dir / "d13d_deletion_results.csv", index=False)
    stability.to_csv(audit_dir / "d13d_stability_results.csv", index=False)
    per_class = _per_class_summary(index, deletion, stability)
    per_class.to_csv(audit_dir / "d13d_per_class_slot_summary.csv", index=False)
    print(json.dumps({"audit_dir": str(audit_dir), "deletion_rows": len(deletion), "stability_rows": len(stability)}, indent=2))


def _condition_mean(deletion: pd.DataFrame, condition: str, metric: str = "drop_true_prob") -> pd.Series:
    work = deletion[deletion["condition"] == condition]
    return pd.to_numeric(work[metric], errors="coerce")


def _per_sample_condition(deletion: pd.DataFrame, condition: str) -> pd.DataFrame:
    work = deletion[deletion["condition"] == condition].copy()
    if work.empty:
        return pd.DataFrame(columns=["sample_id", "drop_true_prob", "prediction_changed"])
    return (
        work.groupby("sample_id", as_index=False)
        .agg(drop_true_prob=("drop_true_prob", "mean"), prediction_changed=("prediction_changed", "mean"))
    )


def _per_class_summary(index: pd.DataFrame, deletion: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in index.groupby("label"):
        ids = set(group["sample_id"].astype(str))
        d = deletion[deletion["sample_id"].astype(str).isin(ids)]
        s = stability[stability["sample_id"].astype(str).isin(ids)]
        top1 = _condition_mean(d, "delete_top1_slot")
        top3 = _condition_mean(d, "delete_top3_slots")
        rows.append(
            {
                "label": label,
                "num_samples": int(len(group)),
                "hard_class": bool(label in HARD_CLASSES),
                "top_slot_distribution": ";".join(f"{k}:{v}" for k, v in group["top1_slot"].astype(str).value_counts().to_dict().items()),
                "dominant_area_distribution": ";".join(f"{k}:{v}" for k, v in group["dominant_area_top1"].astype(str).value_counts().to_dict().items()),
                "avg_top1_drop_true_prob": float(top1.mean()) if not top1.empty else np.nan,
                "avg_top3_drop_true_prob": float(top3.mean()) if not top3.empty else np.nan,
                "prediction_change_rate_top1": float(pd.to_numeric(d[d["condition"] == "delete_top1_slot"]["prediction_changed"], errors="coerce").mean()),
                "stability_mean": float(pd.to_numeric(s["mean_slot_map_similarity"], errors="coerce").mean()) if not s.empty else np.nan,
                "prediction_consistency": float(pd.to_numeric(s["prediction_consistent"], errors="coerce").mean()) if not s.empty else np.nan,
                "mouth_risk_proxy": float(pd.to_numeric(group["mouth_mass_top1"], errors="coerce").mean()),
                "center_risk_proxy": float(pd.to_numeric(group["center_mass_top1"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary_metrics(deletion: pd.DataFrame) -> Dict[str, Any]:
    top1 = _per_sample_condition(deletion, "delete_top1_slot")
    top3 = _per_sample_condition(deletion, "delete_top3_slots")
    rand1 = _per_sample_condition(deletion, "delete_random1_slot")
    rand3 = _per_sample_condition(deletion, "delete_random3_slots")
    low = _per_sample_condition(deletion, "delete_low_importance_slot")
    center = _per_sample_condition(deletion, "delete_center_slot_control")
    mouth = _per_sample_condition(deletion, "delete_mouth_like_slot_control")

    def avg(df: pd.DataFrame, col: str = "drop_true_prob") -> float:
        return float(pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").mean())

    def rate(df: pd.DataFrame) -> float:
        return float(pd.to_numeric(df.get("prediction_changed", pd.Series(dtype=float)), errors="coerce").mean())

    return {
        "avg_top1_drop_true_prob": avg(top1),
        "avg_top3_drop_true_prob": avg(top3),
        "avg_random1_drop_true_prob_mean": avg(rand1),
        "avg_random3_drop_true_prob_mean": avg(rand3),
        "avg_low_importance_drop": avg(low),
        "avg_center_control_drop": avg(center),
        "avg_mouth_control_drop": avg(mouth),
        "top1_vs_random1_gap": avg(top1) - avg(rand1),
        "top3_vs_random3_gap": avg(top3) - avg(rand3),
        "top1_vs_low_gap": avg(top1) - avg(low),
        "top1_prediction_change_rate": rate(top1),
        "random1_prediction_change_rate": rate(rand1),
        "center_prediction_change_rate": rate(center),
        "mouth_prediction_change_rate": rate(mouth),
    }


def _stability_summary(stability: pd.DataFrame) -> Dict[str, Any]:
    return {
        "avg_prediction_consistency": float(pd.to_numeric(stability["prediction_consistent"], errors="coerce").mean()),
        "avg_top_slot_consistency": float(pd.to_numeric(stability["top_slot_consistent"], errors="coerce").mean()),
        "avg_slot_map_similarity": float(pd.to_numeric(stability["mean_slot_map_similarity"], errors="coerce").mean()),
        "avg_combined_map_similarity": float(pd.to_numeric(stability["combined_map_similarity"], errors="coerce").mean()),
    }


def _risk_cases(index: pd.DataFrame, deletion: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    top1 = _per_sample_condition(deletion, "delete_top1_slot").rename(columns={"drop_true_prob": "top1_drop"})
    rand1 = _per_sample_condition(deletion, "delete_random1_slot").rename(columns={"drop_true_prob": "random1_drop"})
    center = _per_sample_condition(deletion, "delete_center_slot_control").rename(columns={"drop_true_prob": "center_drop"})
    mouth = _per_sample_condition(deletion, "delete_mouth_like_slot_control").rename(columns={"drop_true_prob": "mouth_drop"})
    stab = (
        stability.groupby("sample_id", as_index=False)
        .agg(
            prediction_consistency=("prediction_consistent", "mean"),
            mean_slot_map_similarity=("mean_slot_map_similarity", "mean"),
            combined_map_similarity=("combined_map_similarity", "mean"),
        )
    )
    work = index[["sample_id", "label", "pred", "top1_slot", "center_slot", "mouth_slot", "dominant_area_top1"]].copy()
    for df in [top1[["sample_id", "top1_drop"]], rand1[["sample_id", "random1_drop"]], center[["sample_id", "center_drop"]], mouth[["sample_id", "mouth_drop"]], stab]:
        work = work.merge(df, on="sample_id", how="left")
    reasons = []
    for _, row in work.iterrows():
        r = []
        if row.get("top1_drop", 0) <= row.get("random1_drop", 0):
            r.append("top_drop_not_above_random")
        if row.get("center_drop", 0) >= row.get("top1_drop", 0):
            r.append("center_control_ge_top")
        if row.get("mouth_drop", 0) >= row.get("top1_drop", 0):
            r.append("mouth_control_ge_top")
        if row.get("prediction_consistency", 1) < 0.85:
            r.append("weak_aug_prediction_flip")
        if row.get("mean_slot_map_similarity", 1) < 0.35:
            r.append("low_slot_map_stability")
        reasons.append(";".join(r))
    work["risk_reason"] = reasons
    return work[work["risk_reason"].astype(str) != ""].copy()


def _plot_outputs(audit_dir: Path, deletion: pd.DataFrame, stability: pd.DataFrame, per_class: pd.DataFrame) -> None:
    fig_dir = audit_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    conds = ["delete_top1_slot", "delete_top3_slots", "delete_random1_slot", "delete_random3_slots", "delete_low_importance_slot", "delete_center_slot_control", "delete_mouth_like_slot_control"]
    data = [pd.to_numeric(deletion[deletion["condition"] == c]["drop_true_prob"], errors="coerce").dropna().to_numpy() for c in conds]
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=conds, vert=True)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("drop_true_prob")
    plt.title("Deletion drop distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "deletion_drop_boxplot.png", dpi=150)
    plt.close()

    means = [float(np.nanmean(d)) if len(d) else np.nan for d in data]
    plt.figure(figsize=(10, 5))
    plt.bar(conds, means)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean drop_true_prob")
    plt.title("Top vs random/control deletion drop")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_vs_random_drop_bar.png", dpi=150)
    plt.close()

    rates = [pd.to_numeric(deletion[deletion["condition"] == c]["prediction_changed"], errors="coerce").mean() for c in conds]
    plt.figure(figsize=(10, 5))
    plt.bar(conds, rates)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("prediction change rate")
    plt.title("Prediction change rate by condition")
    plt.tight_layout()
    plt.savefig(fig_dir / "prediction_change_rate_bar.png", dpi=150)
    plt.close()

    heat = per_class.set_index("label")[["avg_top1_drop_true_prob", "avg_top3_drop_true_prob", "prediction_change_rate_top1", "mouth_risk_proxy", "center_risk_proxy"]]
    plt.figure(figsize=(8, 5))
    plt.imshow(heat.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="value")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.title("Per-class deletion/risk summary")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_drop_heatmap.png", dpi=150)
    plt.close()

    stab = stability.groupby("augmentation")[["prediction_consistent", "top_slot_consistent", "mean_slot_map_similarity", "combined_map_similarity"]].mean(numeric_only=True)
    stab.plot(kind="bar", figsize=(9, 5))
    plt.title("Weak augmentation stability")
    plt.tight_layout()
    plt.savefig(fig_dir / "stability_similarity_bar.png", dpi=150)
    plt.close()

    control = ["delete_top1_slot", "delete_center_slot_control", "delete_mouth_like_slot_control"]
    vals = [pd.to_numeric(deletion[deletion["condition"] == c]["drop_true_prob"], errors="coerce").mean() for c in control]
    plt.figure(figsize=(7, 4))
    plt.bar(control, vals)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("mean drop_true_prob")
    plt.title("Top vs center/mouth control")
    plt.tight_layout()
    plt.savefig(fig_dir / "risk_control_drop_bar.png", dpi=150)
    plt.close()


def _decision(del_summary: Dict[str, Any], stab_summary: Dict[str, Any], risk_df: pd.DataFrame, total_samples: int) -> str:
    shortcut_rate = 0.0 if total_samples <= 0 else float(risk_df["risk_reason"].astype(str).str.contains("center_control_ge_top|mouth_control_ge_top", regex=True).mean())
    checks = [
        del_summary.get("top1_vs_random1_gap", 0.0) > 0,
        del_summary.get("top3_vs_random3_gap", 0.0) > 0,
        del_summary.get("top1_vs_low_gap", 0.0) > 0,
        stab_summary.get("avg_prediction_consistency", 0.0) >= 0.85,
        stab_summary.get("avg_slot_map_similarity", 0.0) >= 0.35,
        shortcut_rate < 0.5,
    ]
    if all(checks):
        return "D13D_EVIDENCE_DIAGNOSTIC_SUPPORTS_SLOT_CANDIDATE"
    if shortcut_rate >= 0.65:
        return "D13D_EVIDENCE_FAIL_SHORTCUT_DOMINATED"
    if sum(bool(v) for v in checks) >= 4:
        return "D13D_EVIDENCE_PARTIAL_NEEDS_MANUAL_REVIEW"
    return "D13D_EVIDENCE_WEAK_KEEP_AS_ATTENTION_DIAGNOSTIC_ONLY"


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def summarize(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    index = pd.read_csv(audit_dir / "d13d_sample_index.csv")
    deletion = pd.read_csv(audit_dir / "d13d_deletion_results.csv")
    stability = pd.read_csv(audit_dir / "d13d_stability_results.csv")
    per_class_path = audit_dir / "d13d_per_class_slot_summary.csv"
    per_class = pd.read_csv(per_class_path) if per_class_path.exists() else _per_class_summary(index, deletion, stability)
    del_summary = _summary_metrics(deletion)
    stab_summary = _stability_summary(stability)
    risk = _risk_cases(index, deletion, stability)
    decision = _decision(del_summary, stab_summary, risk, int(len(index)))
    del_summary["decision"] = decision
    stab_summary["decision"] = decision
    pd.DataFrame([del_summary]).to_csv(audit_dir / "d13d_deletion_summary.csv", index=False)
    pd.DataFrame([stab_summary]).to_csv(audit_dir / "d13d_stability_summary.csv", index=False)
    risk.to_csv(audit_dir / "d13d_risk_cases.csv", index=False)
    _plot_outputs(audit_dir, deletion, stability, per_class)
    lines = [
        "# D13D Slot Evidence Diagnostic Report",
        "",
        "## 1. Context",
        "- Candidate: `d13c_m16_supcon_l005` unless this audit dir belongs to a control run.",
        "- This is evidence diagnostic only.",
        "- No motif claim.",
        "- No causal evidence claim.",
        "",
        "## 2. Deletion Test Summary",
        _md_table(pd.DataFrame([del_summary])),
        "",
        "## 3. Control Tests",
        f"- center_control_drop: {del_summary.get('avg_center_control_drop')}",
        f"- mouth_control_drop: {del_summary.get('avg_mouth_control_drop')}",
        "If center/mouth control is comparable to top deletion on many samples, treat this as shortcut risk.",
        "",
        "## 4. Stability",
        _md_table(pd.DataFrame([stab_summary])),
        "",
        "## 5. Per-class Behavior",
        _md_table(per_class, max_rows=10),
        "",
        "## 6. Risk Cases",
        _md_table(risk, max_rows=30),
        "",
        "## 7. Gate Check",
        decision,
        "",
        "Forbidden outputs: MOTIF_DISCOVERED, SEMANTIC_REGION_DISCOVERED, CAUSAL_EVIDENCE_CONFIRMED, FULL_INTERPRETABILITY_CLAIM.",
        "",
    ]
    (audit_dir / "d13d_evidence_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit_dir": str(audit_dir), "decision": decision, "risk_cases": int(len(risk))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run D13D slot evidence diagnostics.")
    parser.add_argument("--mode", choices=["prepare", "run", "summarize"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir")
    parser.add_argument("--audit_dir")
    parser.add_argument("--samples_per_class", type=int, default=15)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--sample_id_sheet")
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
    elif args.mode == "run":
        for attr in ("audit_dir", "config", "checkpoint"):
            if getattr(args, attr) is None:
                parser.error(f"--{attr} is required for run")
        run(args)
    else:
        if args.audit_dir is None:
            parser.error("--audit_dir is required for summarize")
        summarize(args)


if __name__ == "__main__":
    main()
