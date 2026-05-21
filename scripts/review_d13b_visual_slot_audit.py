"""AI-assisted heuristic review helper for D13B visual slot audits.

This script fills the D13B visual slot audit sheet from saved slot masks and
metadata. The output is diagnostic-only: it does not claim motifs, semantic
regions, facial landmarks, or causal evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REVIEW_STATUS = "AI_ASSISTED_REVIEW"
REVIEW_COLUMNS = [
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
    *REVIEW_COLUMNS,
]


def _sanitize(value: str | None, fallback: str = "run") -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _prefix(run_name: str) -> str:
    return f"d13b_{_sanitize(run_name)}_visual_slot"


def _find_sheet(audit_dir: Path, run_name: str, output_sheet: Optional[Path]) -> Path:
    candidates: List[Path] = []
    prefix = _prefix(run_name)
    if output_sheet is not None:
        out_resolved = output_sheet.resolve()
    else:
        out_resolved = None
    candidates.extend(
        [
            audit_dir / f"{prefix}_audit_sheet.csv",
            audit_dir / f"{prefix}_audit_sheet_filled.csv",
        ]
    )
    candidates.extend(sorted(audit_dir.glob("d13b_*_visual_slot_audit_sheet.csv")))
    candidates.extend(sorted(audit_dir.glob("*audit_sheet.csv")))
    for path in candidates:
        if not path.exists():
            continue
        if out_resolved is not None and path.resolve() == out_resolved:
            continue
        return path
    raise FileNotFoundError(f"Could not find base audit sheet in {audit_dir}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_relative(audit_dir: Path, value: Any) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return audit_dir / path


def _load_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _safe_sample_dir(audit_dir: Path, sample_id: Any) -> Path:
    safe_id = str(sample_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return audit_dir / "masks" / f"sample_{safe_id}"


def _load_sample_payload(audit_dir: Path, row: pd.Series) -> Dict[str, Any]:
    metadata_path = _resolve_relative(audit_dir, row.get("metadata_path"))
    metadata = _read_json(metadata_path) if metadata_path else {}
    sample_dir = metadata_path.parent if metadata_path else _safe_sample_dir(audit_dir, row.get("sample_id"))

    def meta_path(key: str, fallback: str) -> Optional[Path]:
        return _resolve_relative(audit_dir, metadata.get(key)) or (sample_dir / fallback)

    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "slot_attention": _load_array(meta_path("slot_attention_path", "slot_attention.npy")),
        "slot_pixel_maps": _load_array(meta_path("slot_pixel_maps_path", "slot_pixel_maps.npy")),
        "combined_slot_map": _load_array(meta_path("combined_slot_map_path", "combined_slot_map.npy")),
        "slot_importance": _load_array(sample_dir / "slot_importance.npy"),
        "slot_stats": _read_json(sample_dir / "slot_stats.json"),
    }


def _normalize(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    y = np.where(finite, x, 0.0)
    lo = float(y[finite].min())
    hi = float(y[finite].max())
    if hi - lo < 1e-8:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - lo) / (hi - lo)).astype(np.float32)


def _as_maps(payload: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    maps = payload.get("slot_pixel_maps")
    if maps is not None:
        maps = np.asarray(maps, dtype=np.float32)
        if maps.ndim == 2:
            maps = maps[None, :, :]
        if maps.ndim == 3:
            return maps, _combined_from_maps(maps, payload.get("combined_slot_map"))
    attn = payload.get("slot_attention")
    if attn is None:
        return None, None
    attn = np.asarray(attn, dtype=np.float32)
    if attn.ndim != 2:
        return None, None
    k = int(attn.shape[1])
    grid = int(round(math.sqrt(k)))
    if grid * grid != k:
        return None, None
    scale = int(math.ceil(48 / grid))
    projected = []
    for slot in attn:
        item = np.kron(slot.reshape(grid, grid), np.ones((scale, scale), dtype=np.float32))[:48, :48]
        projected.append(item)
    maps = np.stack(projected, axis=0).astype(np.float32)
    return maps, _combined_from_maps(maps, payload.get("combined_slot_map"))


def _combined_from_maps(maps: np.ndarray, combined: Optional[np.ndarray]) -> np.ndarray:
    if combined is not None:
        arr = np.asarray(combined, dtype=np.float32)
        if arr.shape == maps.shape[-2:]:
            return arr
    return np.nanmax(maps, axis=0).astype(np.float32)


def _entropy01(values: np.ndarray) -> float:
    flat = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(flat.sum())
    if total <= 1e-12:
        return 1.0
    prob = flat / total
    nz = prob[prob > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(max(len(prob), 2)))


def _mass(mask: np.ndarray, region: np.ndarray) -> float:
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, None)
    total = float(work.sum())
    if total <= 1e-8:
        return 0.0
    return float(work[region].sum() / total)


def _regions(height: int = 48, width: int = 48) -> Dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    border = (xx < 4) | (xx >= width - 4) | (yy < 4) | (yy >= height - 4)
    face = (xx >= 6) & (xx <= 41) & (yy >= 5) & (yy <= 43)
    mouth = (xx >= 14) & (xx <= 34) & (yy >= 29) & (yy <= 41)
    eyes = (xx >= 8) & (xx <= 40) & (yy >= 13) & (yy <= 23)
    nose_cheek = (xx >= 11) & (xx <= 37) & (yy >= 21) & (yy <= 32)
    forehead = (xx >= 13) & (xx <= 35) & (yy >= 5) & (yy <= 14)
    center = (xx >= 16) & (xx <= 32) & (yy >= 16) & (yy <= 32)
    upper_side = ((yy <= 16) & ((xx <= 12) | (xx >= 35))) | ((yy <= 9) & (xx >= 8) & (xx <= 40))
    return {
        "border": border,
        "face": face,
        "mouth": mouth,
        "eyes": eyes,
        "nose_cheek": nose_cheek,
        "forehead": forehead,
        "center": center,
        "upper_side": upper_side,
    }


REGIONS = _regions()


def _region_profile(mask: np.ndarray) -> Dict[str, float]:
    return {name: _mass(mask, region) for name, region in REGIONS.items()}


def _dominant_area(mask: np.ndarray, diffuse: bool = False) -> str:
    if diffuse:
        return "unclear"
    profile = _region_profile(mask)
    if profile["border"] >= 0.38:
        return "border"
    areas = {
        "mouth": profile["mouth"],
        "eyes": profile["eyes"],
        "nose_cheek": profile["nose_cheek"],
        "forehead": profile["forehead"],
    }
    ordered = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    best, best_value = ordered[0]
    second_value = ordered[1][1] if len(ordered) > 1 else 0.0
    if best_value < 0.18:
        return "mixed" if profile["face"] >= 0.55 else "unclear"
    if second_value >= best_value * 0.82 and second_value >= 0.16:
        return "mixed"
    return best


def _cosine_similarity_mean(maps: np.ndarray) -> float:
    flat = np.stack([_normalize(item).reshape(-1) for item in maps], axis=0)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    if int(valid.sum()) < 2:
        return 1.0
    flat = flat[valid] / np.clip(norms[valid], 1e-8, None)
    sim = flat @ flat.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    return float(np.nanmean(off)) if off.size else 1.0


def _top_overlap_mean(maps: np.ndarray, q: float = 0.80) -> float:
    masks = []
    for item in maps:
        norm = _normalize(item)
        threshold = float(np.quantile(norm, q))
        mask = norm >= max(threshold, 1e-6)
        if mask.any():
            masks.append(mask)
    if len(masks) < 2:
        return 1.0
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = float(np.logical_and(masks[i], masks[j]).sum())
            union = float(np.logical_or(masks[i], masks[j]).sum())
            vals.append(inter / max(union, 1.0))
    return float(np.mean(vals)) if vals else 1.0


def _spread_score(mask: np.ndarray) -> Tuple[float, float]:
    norm = _normalize(mask)
    dynamic_range = float(np.nanmax(norm) - np.nanmin(norm)) if np.isfinite(norm).any() else 0.0
    entropy = _entropy01(norm)
    top_area = float((norm >= max(float(np.quantile(norm, 0.85)), 1e-6)).mean())
    concentration = 1.0 - entropy
    return max(concentration, 1.0 - top_area), dynamic_range


def _risk_score(value: float, mild: float, severe: float) -> int:
    if value >= severe:
        return 2
    if value >= mild:
        return 1
    return 0


def _score_traceability(maps: np.ndarray, combined: np.ndarray) -> int:
    if not np.isfinite(maps).all() or not np.isfinite(combined).all():
        return 0
    concentration, dynamic_range = _spread_score(combined)
    if dynamic_range < 1e-6:
        return 0
    if concentration >= 0.25 or dynamic_range >= 0.35:
        return 2
    if concentration >= 0.12 or dynamic_range >= 0.15:
        return 1
    return 0


def _score_readability(combined: np.ndarray) -> int:
    concentration, dynamic_range = _spread_score(combined)
    entropy = _entropy01(_normalize(combined))
    if dynamic_range < 1e-6 or entropy > 0.97:
        return 0
    if concentration >= 0.22 and entropy <= 0.92:
        return 2
    if concentration >= 0.10:
        return 1
    return 0


def _score_diversity(maps: np.ndarray) -> Tuple[int, float, float]:
    if maps.shape[0] < 2:
        return 0, 1.0, 1.0
    cosine = _cosine_similarity_mean(maps)
    overlap = _top_overlap_mean(maps)
    if cosine >= 0.94 or overlap >= 0.70:
        return 0, cosine, overlap
    if cosine <= 0.72 and overlap <= 0.42:
        return 2, cosine, overlap
    return 1, cosine, overlap


def _score_face_coverage(combined: np.ndarray) -> Tuple[int, Dict[str, float]]:
    profile = _region_profile(_normalize(combined))
    if profile["border"] >= 0.42:
        return 0, profile
    if profile["face"] >= 0.62 and profile["border"] <= 0.25:
        return 2, profile
    if profile["face"] >= 0.45:
        return 1, profile
    return 0, profile


def _multi_region_support(maps: np.ndarray) -> Tuple[int, List[str]]:
    areas = []
    for item in maps:
        norm = _normalize(item)
        if _spread_score(norm)[1] < 1e-6:
            continue
        area = _dominant_area(norm)
        if area not in {"unclear", "border"}:
            areas.append(area)
    unique = sorted(set(areas))
    if len(unique) >= 3:
        return 2, unique
    if len(unique) >= 2:
        return 1, unique
    return 0, unique


def _visual_status(review: Dict[str, Any]) -> str:
    severe_count = sum(
        int(review[key]) == 2
        for key in ["mouth_only_risk", "center_shortcut_risk", "background_border_risk", "slot_collapse_visual"]
    )
    if (
        int(review["slot_traceability_score"]) == 0
        or int(review["slot_diversity_visual_score"]) == 0
        or int(review["slot_assignment_readability_score"]) == 0
        or int(review["slot_collapse_visual"]) == 2
        or int(review["background_border_risk"]) == 2
        or (int(review["mouth_only_risk"]) == 2 and int(review["center_shortcut_risk"]) == 2)
    ):
        return "FAIL"
    if (
        int(review["slot_traceability_score"]) >= 2
        and int(review["slot_diversity_visual_score"]) >= 2
        and int(review["slot_face_coverage_score"]) >= 1
        and int(review["slot_assignment_readability_score"]) >= 1
        and int(review["mouth_only_risk"]) <= 1
        and int(review["center_shortcut_risk"]) <= 1
        and int(review["background_border_risk"]) <= 1
        and int(review["slot_collapse_visual"]) == 0
    ):
        return "PASS"
    if int(review["slot_traceability_score"]) >= 1 and int(review["slot_assignment_readability_score"]) >= 1 and severe_count <= 1:
        return "PARTIAL"
    return "FAIL"


def _review_row(audit_dir: Path, row: pd.Series) -> Dict[str, Any]:
    payload = _load_sample_payload(audit_dir, row)
    maps, combined_raw = _as_maps(payload)
    if maps is None or combined_raw is None:
        return {
            "review_status": REVIEW_STATUS,
            "visual_status": "FAIL",
            "slot_traceability_score": 0,
            "slot_diversity_visual_score": 0,
            "slot_face_coverage_score": 0,
            "slot_assignment_readability_score": 0,
            "mouth_only_risk": 0,
            "center_shortcut_risk": 0,
            "hair_glasses_risk": 0,
            "background_border_risk": 0,
            "dominant_slot_area": "unclear",
            "multi_region_support": 0,
            "slot_collapse_visual": 2,
            "notes": "Missing/unreadable slot maps; AI-assisted heuristic review; requires human confirmation for final motif interpretation.",
        }

    maps = np.asarray(maps, dtype=np.float32)
    combined = _normalize(combined_raw)
    norm_maps = np.stack([_normalize(item) for item in maps], axis=0)

    traceability = _score_traceability(norm_maps, combined)
    readability = _score_readability(combined)
    diversity, cosine, overlap = _score_diversity(norm_maps)
    face_coverage, profile = _score_face_coverage(combined)
    mouth_risk = _risk_score(profile["mouth"], mild=0.28, severe=0.43)
    center_risk = _risk_score(profile["center"], mild=0.28, severe=0.45)
    border_risk = _risk_score(profile["border"], mild=0.25, severe=0.42)
    hair_glasses_risk = 1 if profile["upper_side"] >= 0.34 else 0
    collapse = 2 if diversity == 0 else (1 if diversity == 1 and (cosine >= 0.86 or overlap >= 0.55) else 0)
    dominant = _dominant_area(combined, diffuse=readability == 0)
    multi_region, regions = _multi_region_support(norm_maps)

    review: Dict[str, Any] = {
        "review_status": REVIEW_STATUS,
        "slot_traceability_score": traceability,
        "slot_diversity_visual_score": diversity,
        "slot_face_coverage_score": face_coverage,
        "slot_assignment_readability_score": readability,
        "mouth_only_risk": mouth_risk,
        "center_shortcut_risk": center_risk,
        "hair_glasses_risk": hair_glasses_risk,
        "background_border_risk": border_risk,
        "dominant_slot_area": dominant,
        "multi_region_support": multi_region,
        "slot_collapse_visual": collapse,
    }
    review["visual_status"] = _visual_status(review)

    reason = [
        f"cosine={cosine:.3f}",
        f"top_overlap={overlap:.3f}",
        f"face_mass={profile['face']:.3f}",
        f"mouth_mass={profile['mouth']:.3f}",
        f"center_mass={profile['center']:.3f}",
        f"border_mass={profile['border']:.3f}",
    ]
    if regions:
        reason.append("slot_regions=" + ",".join(regions))
    reason.append("heuristic face coverage only, not a landmark detector")
    reason.append("heuristic/no explicit hair-glasses detector")
    reason.append("AI-assisted heuristic review; requires human confirmation for final motif interpretation.")
    review["notes"] = "; ".join(reason)
    return review


def _has_review_status(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(str(value).strip())


def _merge_existing_reviews(df: pd.DataFrame, existing: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
    if "sample_id" not in df.columns or "sample_id" not in existing.columns:
        return df
    existing_by_id = {str(row["sample_id"]): row for _, row in existing.iterrows()}
    out = df.copy()
    for idx, row in out.iterrows():
        old = existing_by_id.get(str(row["sample_id"]))
        if old is None or overwrite or not _has_review_status(old.get("review_status")):
            continue
        for col in REVIEW_COLUMNS:
            if col in old.index:
                out.at[idx, col] = old[col]
    return out


def _high_risk_mask(df: pd.DataFrame) -> pd.Series:
    status = df.get("visual_status", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    risk = status.eq("FAIL")
    for col in ["mouth_only_risk", "center_shortcut_risk", "hair_glasses_risk", "background_border_risk", "slot_collapse_visual"]:
        vals = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
        risk = risk | vals.ge(2)
    for col in ["slot_traceability_score", "slot_diversity_visual_score", "slot_assignment_readability_score"]:
        vals = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
        risk = risk | vals.eq(0)
    return risk


def _write_todo(output_sheet: Path, df: pd.DataFrame, run_name: str) -> Path:
    todo_path = output_sheet.with_name(output_sheet.stem + "_review_todo.md")
    risk_df = df.loc[_high_risk_mask(df)].copy()
    lines = [
        f"# D13B Visual Slot Review TODO: {run_name}",
        "",
        "These cases were flagged by AI-assisted heuristic review. They need human confirmation before any final motif interpretation.",
        "",
        "| sample_id | visual_status | dominant_slot_area | risks | figure_path | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in risk_df.iterrows():
        risks = []
        for col in ["mouth_only_risk", "center_shortcut_risk", "hair_glasses_risk", "background_border_risk", "slot_collapse_visual"]:
            try:
                if float(row.get(col, 0)) >= 2:
                    risks.append(col)
            except Exception:
                pass
        fig = str(row.get("figure_path", ""))
        notes = str(row.get("notes", "")).replace("|", "/")[:220]
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('visual_status', '')} | {row.get('dominant_slot_area', '')} | "
            f"{', '.join(risks) or 'score/readability gate'} | {fig} | {notes} |"
        )
    if risk_df.empty:
        lines.append("| none | PASS/PARTIAL only |  |  |  |  |")
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def run(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    run_name = _sanitize(args.run_name, audit_dir.name)
    output_sheet = Path(args.output_sheet)
    base_sheet = _find_sheet(audit_dir, run_name, output_sheet)
    df = pd.read_csv(base_sheet)
    for col in SHEET_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[SHEET_COLUMNS + [c for c in df.columns if c not in SHEET_COLUMNS]]
    for col in REVIEW_COLUMNS:
        df[col] = df[col].astype("object")
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    existing = pd.read_csv(output_sheet) if output_sheet.exists() else pd.DataFrame()
    reviewed_rows = 0
    preserved_rows = 0
    for idx, row in df.iterrows():
        if (
            args.preserve_existing_reviews
            and not args.overwrite
            and not existing.empty
            and "sample_id" in existing.columns
        ):
            old = existing.loc[existing["sample_id"].astype(str) == str(row.get("sample_id"))]
            if not old.empty and _has_review_status(old.iloc[0].get("review_status")):
                for col in REVIEW_COLUMNS:
                    if col in old.columns:
                        df.at[idx, col] = old.iloc[0][col]
                preserved_rows += 1
                continue
        review = _review_row(audit_dir, row)
        for col, value in review.items():
            df.at[idx, col] = value
        reviewed_rows += 1

    if not args.overwrite and not args.preserve_existing_reviews and output_sheet.exists():
        df = _merge_existing_reviews(df, existing, overwrite=False)

    output_sheet.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_sheet, index=False)
    todo_path = _write_todo(output_sheet, df, run_name) if args.export_review_todo else None
    counts = df["visual_status"].astype(str).str.upper().value_counts().to_dict()
    result = {
        "audit_dir": str(audit_dir),
        "base_sheet": str(base_sheet),
        "output_sheet": str(output_sheet),
        "run_name": run_name,
        "rows_written": int(len(df)),
        "rows_reviewed_by_heuristic": int(reviewed_rows),
        "rows_preserved": int(preserved_rows),
        "review_status": REVIEW_STATUS,
        "visual_status_counts": counts,
        "todo_path": str(todo_path) if todo_path else None,
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
        "no_causal_claim": True,
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill D13B visual slot audit sheets with AI-assisted heuristic reviews."
    )
    parser.add_argument("--audit_dir", required=True, help="Run-specific visual slot audit directory.")
    parser.add_argument("--run_name", required=True, help="D13B run name.")
    parser.add_argument("--output_sheet", required=True, help="Filled review sheet path to write.")
    parser.add_argument(
        "--preserve_existing_reviews",
        action="store_true",
        help="Preserve existing rows whose review_status is non-empty unless --overwrite is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing review rows.")
    parser.add_argument("--limit", type=int, default=None, help="Only review the first N rows for smoke testing.")
    parser.add_argument(
        "--export_review_todo",
        action="store_true",
        help="Write a markdown TODO list for FAIL or high-risk cases next to the output sheet.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
