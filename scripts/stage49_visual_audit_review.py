#!/usr/bin/env python
"""Manual-review workflow for the Stage 4.9 near-miss Hybrid SLIC audit pack.

This utility deliberately does not reopen selector search, retrain anything, or
change the Stage 4.9 deletion report. It only:

1. prepares a human-review sheet + instructions + index for the direct figures;
2. summarizes a manually filled sheet into audit outputs and a final report.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger("stage49_visual_audit_review")

SELECTOR = "hybrid_slic_region__E_grad_50__b0p1__max_mean_mix"
TARGET_RATIO = 0.15
EXPECTED_SAMPLE_COUNT = 105

PREPARE_COLUMNS = [
    "sample_id",
    "split",
    "label",
    "selector",
    "ratio",
    "figure_path",
    "mask_path",
    "review_status",
    "visual_status",
    "facial_evidence_score",
    "region_like_score",
    "hair_glasses_risk",
    "background_border_risk",
    "center_shortcut_risk",
    "main_selected_area",
    "notes",
]

SUMMARY_COLUMNS = [
    "total_samples",
    "reviewed_samples",
    "unreviewed_samples",
    "pass_count",
    "partial_count",
    "fail_count",
    "pass_rate",
    "partial_rate",
    "fail_rate",
    "pass_partial_count",
    "pass_partial_rate",
    "avg_facial_evidence_score",
    "avg_region_like_score",
    "avg_hair_glasses_risk",
    "avg_background_border_risk",
    "avg_center_shortcut_risk",
]

AREA_SUMMARY_COLUMNS = [
    "main_selected_area",
    "count",
    "rate",
    "pass_count",
    "partial_count",
    "fail_count",
    "avg_facial_evidence_score",
    "avg_region_like_score",
    "avg_hair_glasses_risk",
    "avg_background_border_risk",
    "avg_center_shortcut_risk",
]

RISK_CASE_COLUMNS = [
    "sample_id",
    "label",
    "figure_path",
    "mask_path",
    "visual_status",
    "facial_evidence_score",
    "region_like_score",
    "hair_glasses_risk",
    "background_border_risk",
    "center_shortcut_risk",
    "main_selected_area",
    "notes",
]

VISUAL_STATUSES = {"PASS", "PARTIAL", "FAIL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage49_dir",
        default="outputs/stage4_9_confirm_nearmiss_hybrid_slic",
        help="Root directory of the existing Stage 4.9 bundle.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/stage4_9_confirm_nearmiss_hybrid_slic/visual_audit_review",
        help="Directory for review outputs.",
    )
    parser.add_argument("--mode", choices=("prepare", "summarize"), default="prepare")
    parser.add_argument(
        "--review_sheet",
        default=None,
        help="Filled review sheet for summarize mode. If omitted, prefer *_filled.csv, then the base sheet.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {kind} does not exist: {path}")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def relative_link(path: Path, from_dir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(from_dir.resolve()).as_posix()
    except Exception:
        try:
            import os

            return Path(os.path.relpath(path.resolve(), start=from_dir.resolve())).as_posix()
        except Exception:
            return path.as_posix()


def normalize_ratio(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def parse_graph_id(text: str) -> str:
    match = re.search(r"graph[_-](\d+)", text)
    return match.group(1) if match else "UNKNOWN"


def parse_label_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.match(r"class[_-](.+)", part)
        if match:
            return match.group(1)
    return "UNKNOWN"


def read_existing_metadata(stage49_dir: Path) -> pd.DataFrame:
    path = stage49_dir / "stage49_visual_audit_sheet.csv"
    if not path.exists():
        LOGGER.warning("No existing Stage 4.9 audit sheet found at %s; filename parsing only.", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def find_candidate_dirs(stage49_dir: Path, dirname: str) -> List[Path]:
    dirs: List[Path] = []
    direct = stage49_dir / dirname
    if direct.exists():
        dirs.append(direct)
    for path in stage49_dir.rglob(dirname):
        if path.is_dir() and path not in dirs:
            dirs.append(path)
    return dirs


def find_audit_files(stage49_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find figure and mask files belonging to the near-miss selector."""
    ensure_exists(stage49_dir, "Stage 4.9 directory")
    figure_dirs = find_candidate_dirs(stage49_dir, "figures")
    mask_dirs = find_candidate_dirs(stage49_dir, "masks")
    if not figure_dirs:
        LOGGER.warning("No figures/ directory found under %s", stage49_dir)
    if not mask_dirs:
        LOGGER.warning("No masks/ directory found under %s", stage49_dir)

    figures: List[Path] = []
    masks: List[Path] = []
    for root in figure_dirs:
        figures.extend(path for path in root.rglob("*.png") if SELECTOR in path.as_posix())
    for root in mask_dirs:
        masks.extend(path for path in root.rglob("*.png") if SELECTOR in path.as_posix())

    figures = sorted(set(figures))
    masks = sorted(set(masks))
    LOGGER.info("Found %d figure files and %d mask files for selector %s", len(figures), len(masks), SELECTOR)
    if len(figures) != EXPECTED_SAMPLE_COUNT or len(masks) != EXPECTED_SAMPLE_COUNT:
        LOGGER.warning(
            "Expected %d figures and %d masks, found %d figures and %d masks.",
            EXPECTED_SAMPLE_COUNT,
            EXPECTED_SAMPLE_COUNT,
            len(figures),
            len(masks),
        )
    return figures, masks


def metadata_lookup(metadata: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if metadata.empty:
        return {}
    lookup: Dict[str, Dict[str, object]] = {}
    for _, row in metadata.iterrows():
        graph_id = str(row.get("graph_id", "")).strip()
        if graph_id:
            lookup[graph_id] = row.to_dict()
    return lookup


def build_prepare_sheet(stage49_dir: Path, output_dir: Path) -> pd.DataFrame:
    figures, masks = find_audit_files(stage49_dir)
    metadata = read_existing_metadata(stage49_dir)
    meta_by_graph = metadata_lookup(metadata)
    masks_by_graph = {parse_graph_id(path.name): path for path in masks}

    rows: List[Dict[str, object]] = []
    for figure in figures:
        graph_id = parse_graph_id(figure.name)
        meta = meta_by_graph.get(str(int(graph_id)) if graph_id.isdigit() else graph_id, {})
        ratio = normalize_ratio(meta.get("ratio")) if meta else None
        if ratio is not None and not math.isclose(ratio, TARGET_RATIO, rel_tol=0.0, abs_tol=1e-9):
            continue
        label = str(meta.get("class_name", "")).strip() if meta else ""
        sample_id = str(meta.get("graph_id", "")).strip() if meta else ""
        row = {
            "sample_id": sample_id or graph_id or "UNKNOWN",
            "split": str(meta.get("split", "")).strip() if meta else "UNKNOWN",
            "label": label or parse_label_from_path(figure),
            "selector": SELECTOR,
            "ratio": ratio if ratio is not None else TARGET_RATIO,
            "figure_path": repo_relative(figure),
            "mask_path": repo_relative(masks_by_graph.get(graph_id, Path(""))) if graph_id in masks_by_graph else "",
            "review_status": "NEEDS_MANUAL_REVIEW",
            "visual_status": "",
            "facial_evidence_score": "",
            "region_like_score": "",
            "hair_glasses_risk": "",
            "background_border_risk": "",
            "center_shortcut_risk": "",
            "main_selected_area": "",
            "notes": "",
        }
        if not row["split"]:
            row["split"] = "UNKNOWN"
        if not row["label"]:
            row["label"] = "UNKNOWN"
        if not row["sample_id"]:
            row["sample_id"] = "UNKNOWN"
        rows.append(row)

    sheet = pd.DataFrame(rows, columns=PREPARE_COLUMNS)
    if sheet.empty:
        LOGGER.warning("Prepare sheet is empty.")
    else:
        sheet["_sample_id_sort"] = pd.to_numeric(sheet["sample_id"], errors="coerce")
        sheet = (
            sheet.sort_values(by=["label", "_sample_id_sort", "sample_id"], kind="stable")
            .drop(columns=["_sample_id_sort"])
            .reset_index(drop=True)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "stage49_visual_audit_sheet.csv"
    sheet.to_csv(sheet_path, index=False)
    LOGGER.info("Wrote review sheet: %s (%d rows)", sheet_path, len(sheet))
    return sheet


def write_review_instructions(output_dir: Path) -> Path:
    text = """# Stage 4.9 Visual Audit Instructions

Review only the direct Stage 4.9 near-miss selector figures:
`hybrid_slic_region__E_grad_50__b0p1__max_mean_mix @0.15`

Fill `stage49_visual_audit_sheet.csv` manually. This review may document a
promising near-miss, but it does **not** open Stage 5.

## visual_status

- `PASS`: mask chủ yếu nằm trên vùng biểu cảm hợp lý như miệng, mắt, lông mày, vùng mũi-má, nếp nhăn/biên local có liên quan biểu cảm.
- `PARTIAL`: có một phần hợp lý nhưng vẫn lẫn tóc, kính, viền mặt, nền, hoặc vùng không rõ.
- `FAIL`: chủ yếu chọn tóc, kính, nền, viền mặt, vùng trơn, center shortcut hoặc contour không liên quan biểu cảm.

## facial_evidence_score

- `0` = không giống evidence biểu cảm
- `1` = có một phần evidence biểu cảm
- `2` = evidence biểu cảm rõ

## region_like_score

- `0` = pixel-dust / rời rạc / khó thành part
- `1` = có cụm nhưng còn nhiễu
- `2` = cụm vùng rõ, region-like tốt

## hair_glasses_risk

- `0` = không đáng kể
- `1` = có nhưng không dominant
- `2` = dominant hoặc gây nghi ngờ nghiêm trọng

## background_border_risk

- `0` = không đáng kể
- `1` = có nhưng không dominant
- `2` = dominant background/border/face contour

## center_shortcut_risk

- `0` = không đáng kể
- `1` = hơi nghi ngờ
- `2` = rõ center shortcut

## main_selected_area

Chọn một trong:
`mouth`, `eyes`, `eyebrows`, `nose_cheek`, `forehead`, `face_contour`,
`hair`, `glasses`, `background`, `border`, `mixed`, `unclear`.

## Notes

- Không claim motif.
- Không claim semantic part.
- Không claim causal evidence mạnh.
- Nếu hình khó đọc, dùng `PARTIAL` hoặc ghi rõ trong `notes`.
"""
    path = output_dir / "stage49_visual_audit_instructions.md"
    path.write_text(text, encoding="utf-8")
    LOGGER.info("Wrote review instructions: %s", path)
    return path


def write_review_index(sheet: pd.DataFrame, output_dir: Path, stage49_dir: Path) -> Path:
    lines = [
        "# Stage 4.9 Visual Audit Index",
        "",
        f"Selector: `{SELECTOR} @ {TARGET_RATIO}`",
        "",
        "| sample_id | label | figure | mask | review_status |",
        "|---|---|---|---|---|",
    ]
    for _, row in sheet.iterrows():
        figure_abs = Path(str(row["figure_path"]))
        mask_abs = Path(str(row["mask_path"])) if str(row["mask_path"]) else Path("")
        if not figure_abs.is_absolute():
            figure_abs = Path.cwd() / figure_abs
        if str(mask_abs) and not mask_abs.is_absolute():
            mask_abs = Path.cwd() / mask_abs
        figure_rel = relative_link(figure_abs, output_dir)
        mask_rel = relative_link(mask_abs, output_dir) if str(row["mask_path"]) else ""
        figure_link = f"[figure]({figure_rel})" if figure_rel else ""
        mask_link = f"[mask]({mask_rel})" if mask_rel else ""
        lines.append(
            f"| {row['sample_id']} | {row['label']} | {figure_link} | {mask_link} | {row['review_status']} |"
        )
    path = output_dir / "stage49_visual_audit_index.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote review index: %s", path)
    return path


def choose_review_sheet(output_dir: Path, review_sheet: Optional[str]) -> Path:
    if review_sheet:
        return Path(review_sheet)
    filled = output_dir / "stage49_visual_audit_sheet_filled.csv"
    if filled.exists():
        return filled
    return output_dir / "stage49_visual_audit_sheet.csv"


def load_review_sheet(path: Path) -> pd.DataFrame:
    ensure_exists(path, "review sheet")
    sheet = pd.read_csv(path)
    missing = [column for column in PREPARE_COLUMNS if column not in sheet.columns]
    if missing:
        raise ValueError("Review sheet is missing required columns: " + ", ".join(missing))
    invalid_status = sorted(
        {
            str(value).strip().upper()
            for value in sheet["visual_status"].dropna()
            if str(value).strip() and str(value).strip().upper() not in VISUAL_STATUSES
        }
    )
    if invalid_status:
        raise ValueError("Invalid visual_status values: " + ", ".join(invalid_status))
    LOGGER.info("Loaded review sheet: %s (%d rows)", path, len(sheet))
    return sheet


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def summarize_review(sheet: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    frame = sheet.copy()
    frame["visual_status_norm"] = frame["visual_status"].fillna("").astype(str).str.strip().str.upper()
    reviewed = frame[frame["visual_status_norm"].isin(VISUAL_STATUSES)].copy()
    total = len(frame)
    reviewed_count = len(reviewed)
    pass_count = int((reviewed["visual_status_norm"] == "PASS").sum())
    partial_count = int((reviewed["visual_status_norm"] == "PARTIAL").sum())
    fail_count = int((reviewed["visual_status_norm"] == "FAIL").sum())
    pass_partial_count = pass_count + partial_count

    for column in [
        "facial_evidence_score",
        "region_like_score",
        "hair_glasses_risk",
        "background_border_risk",
        "center_shortcut_risk",
    ]:
        reviewed[column] = numeric_series(reviewed, column)
        frame[column] = numeric_series(frame, column)

    summary = pd.DataFrame(
        [
            {
                "total_samples": total,
                "reviewed_samples": reviewed_count,
                "unreviewed_samples": total - reviewed_count,
                "pass_count": pass_count,
                "partial_count": partial_count,
                "fail_count": fail_count,
                "pass_rate": pass_count / reviewed_count if reviewed_count else math.nan,
                "partial_rate": partial_count / reviewed_count if reviewed_count else math.nan,
                "fail_rate": fail_count / reviewed_count if reviewed_count else math.nan,
                "pass_partial_count": pass_partial_count,
                "pass_partial_rate": pass_partial_count / reviewed_count if reviewed_count else math.nan,
                "avg_facial_evidence_score": reviewed["facial_evidence_score"].mean(),
                "avg_region_like_score": reviewed["region_like_score"].mean(),
                "avg_hair_glasses_risk": reviewed["hair_glasses_risk"].mean(),
                "avg_background_border_risk": reviewed["background_border_risk"].mean(),
                "avg_center_shortcut_risk": reviewed["center_shortcut_risk"].mean(),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )

    grouped_rows: List[Dict[str, object]] = []
    area_source = reviewed.copy()
    area_source["main_selected_area"] = area_source["main_selected_area"].fillna("").replace("", "UNKNOWN")
    for area, group in area_source.groupby("main_selected_area", dropna=False):
        count = len(group)
        grouped_rows.append(
            {
                "main_selected_area": area,
                "count": count,
                "rate": count / reviewed_count if reviewed_count else math.nan,
                "pass_count": int((group["visual_status_norm"] == "PASS").sum()),
                "partial_count": int((group["visual_status_norm"] == "PARTIAL").sum()),
                "fail_count": int((group["visual_status_norm"] == "FAIL").sum()),
                "avg_facial_evidence_score": group["facial_evidence_score"].mean(),
                "avg_region_like_score": group["region_like_score"].mean(),
                "avg_hair_glasses_risk": group["hair_glasses_risk"].mean(),
                "avg_background_border_risk": group["background_border_risk"].mean(),
                "avg_center_shortcut_risk": group["center_shortcut_risk"].mean(),
            }
        )
    area_summary = pd.DataFrame(grouped_rows, columns=AREA_SUMMARY_COLUMNS)
    if not area_summary.empty:
        area_summary = area_summary.sort_values(by=["count", "main_selected_area"], ascending=[False, True]).reset_index(drop=True)

    risk_mask = (
        frame["visual_status_norm"].eq("FAIL")
        | frame["hair_glasses_risk"].ge(2)
        | frame["background_border_risk"].ge(2)
        | frame["center_shortcut_risk"].ge(2)
        | frame["facial_evidence_score"].eq(0)
    )
    risk_cases = frame.loc[risk_mask, RISK_CASE_COLUMNS].copy()

    summary_row = summary.iloc[0].to_dict()
    return summary, area_summary, risk_cases, summary_row


def fmt(value: object) -> str:
    try:
        number = float(value)
        if math.isnan(number):
            return "NA"
        return f"{number:.6g}"
    except Exception:
        return str(value)


def visual_gate_pass(summary_row: Dict[str, float]) -> bool:
    return all(
        [
            summary_row["pass_partial_rate"] >= 0.60,
            summary_row["avg_facial_evidence_score"] >= 1.20,
            summary_row["avg_region_like_score"] >= 1.20,
            summary_row["avg_hair_glasses_risk"] < 1.0,
            summary_row["avg_background_border_risk"] < 1.0,
            summary_row["avg_center_shortcut_risk"] < 1.0,
        ]
    )


def final_decision(summary_row: Dict[str, float]) -> str:
    if int(summary_row["reviewed_samples"]) < int(summary_row["total_samples"]):
        return "VISUAL_AUDIT_INCOMPLETE_KEEP_STAGE5_LOCKED"
    if visual_gate_pass(summary_row):
        return "DOCUMENT_AS_PROMISING_NEAR_MISS_BUT_KEEP_STAGE5_LOCKED"
    return "DOCUMENT_AS_VISUALLY_UNRELIABLE_NEAR_MISS_AND_STOP_STAGE5_PATH"


def write_final_report(
    output_dir: Path,
    summary_row: Dict[str, float],
    area_summary: pd.DataFrame,
    risk_cases: pd.DataFrame,
) -> Path:
    decision = final_decision(summary_row)
    area_lines = (
        "\n".join(
            f"- `{row.main_selected_area}`: {int(row.count)} ({fmt(row.rate)})"
            for row in area_summary.itertuples()
        )
        if not area_summary.empty
        else "- Chưa có area distribution vì review chưa hoàn tất."
    )
    top_risks = risk_cases.head(10)
    if top_risks.empty:
        risk_lines = "- Không có risk case theo rule hiện tại."
    else:
        risk_lines = "\n".join(
            f"- sample `{row.sample_id}` ({row.label}): status `{row.visual_status}`, area `{row.main_selected_area}`, notes `{row.notes}`"
            for row in top_risks.itertuples()
        )
    visual_ok = visual_gate_pass(summary_row) if int(summary_row["reviewed_samples"]) == int(summary_row["total_samples"]) else False
    text = f"""# Stage 4.9 Visual Audit Review Report

## 1. Context

- Selector: `{SELECTOR} @0.15`
- deletion_drop = `0.0220192`
- gap_vs_random = `0.0181878`
- components = `3.207`
- long_contour = `0.148351`
- Prior decision before visual audit: `KEEP_STAGE5_LOCKED_BUT_DOCUMENT_NEAR_MISS`

## 2. Manual Visual Audit Summary

- total_samples = `{int(summary_row['total_samples'])}`
- reviewed_samples = `{int(summary_row['reviewed_samples'])}`
- unreviewed_samples = `{int(summary_row['unreviewed_samples'])}`
- pass_count = `{int(summary_row['pass_count'])}`
- partial_count = `{int(summary_row['partial_count'])}`
- fail_count = `{int(summary_row['fail_count'])}`
- pass_partial_rate = `{fmt(summary_row['pass_partial_rate'])}`

## 3. Region and Evidence Quality

- avg_facial_evidence_score = `{fmt(summary_row['avg_facial_evidence_score'])}`
- avg_region_like_score = `{fmt(summary_row['avg_region_like_score'])}`

Area distribution:

{area_lines}

## 4. Risk Analysis

- avg_hair_glasses_risk = `{fmt(summary_row['avg_hair_glasses_risk'])}`
- avg_background_border_risk = `{fmt(summary_row['avg_background_border_risk'])}`
- avg_center_shortcut_risk = `{fmt(summary_row['avg_center_shortcut_risk'])}`
- Full risk table: `stage49_visual_audit_risk_cases.csv`

Top risk cases:

{risk_lines}

## 5. Gate Check

Proposed visual gate:

- pass_partial_rate >= 0.60 -> `{'PASS' if summary_row['pass_partial_rate'] >= 0.60 else 'FAIL'}`
- avg_facial_evidence_score >= 1.2 -> `{'PASS' if summary_row['avg_facial_evidence_score'] >= 1.2 else 'FAIL'}`
- avg_region_like_score >= 1.2 -> `{'PASS' if summary_row['avg_region_like_score'] >= 1.2 else 'FAIL'}`
- avg_hair_glasses_risk < 1.0 -> `{'PASS' if summary_row['avg_hair_glasses_risk'] < 1.0 else 'FAIL'}`
- avg_background_border_risk < 1.0 -> `{'PASS' if summary_row['avg_background_border_risk'] < 1.0 else 'FAIL'}`
- avg_center_shortcut_risk < 1.0 -> `{'PASS' if summary_row['avg_center_shortcut_risk'] < 1.0 else 'FAIL'}`
- overall direct visual audit acceptable -> `{'PASS' if visual_ok else 'FAIL'}`

Stage 5 total gate still also requires:

- deletion_drop >= 0.02
- gap_vs_random >= 0.02
- components < 18
- long_contour < 0.20
- direct visual audit acceptable

Because `gap_vs_random = 0.0181878 < 0.02`, **STAGE 5 REMAINS LOCKED**.

## 6. Final Decision

`{decision}`

## 7. Scope Guardrails

- Không output quyết định OPEN_STAGE5.
- Không thay đổi gate để pass.
- Không claim motif đã được tìm thấy.
- Không claim causal evidence mạnh.
- Chỉ được gọi đây là evidence selector near-miss / region-like evidence candidate nếu visual audit thực sự phù hợp.
"""
    path = output_dir / "stage49_visual_audit_final_report.md"
    path.write_text(text, encoding="utf-8")
    LOGGER.info("Wrote final report: %s", path)
    return path


def write_summary_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    area_summary: pd.DataFrame,
    risk_cases: pd.DataFrame,
    summary_row: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "stage49_visual_audit_summary.csv", index=False)
    area_summary.to_csv(output_dir / "stage49_visual_audit_area_summary.csv", index=False)
    risk_cases.to_csv(output_dir / "stage49_visual_audit_risk_cases.csv", index=False)
    write_final_report(output_dir, summary_row, area_summary, risk_cases)
    LOGGER.info("Wrote summarize outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[Stage4.9Review] %(message)s")
    args = parse_args()
    stage49_dir = Path(args.stage49_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "prepare":
        sheet = build_prepare_sheet(stage49_dir, output_dir)
        write_review_instructions(output_dir)
        write_review_index(sheet, output_dir, stage49_dir)
        LOGGER.info("prepare_complete rows=%d output_dir=%s", len(sheet), output_dir)
        return

    review_sheet = choose_review_sheet(output_dir, args.review_sheet)
    sheet = load_review_sheet(review_sheet)
    summary, area_summary, risk_cases, summary_row = summarize_review(sheet)
    write_summary_outputs(output_dir, summary, area_summary, risk_cases, summary_row)
    LOGGER.info(
        "summarize_complete reviewed=%d/%d final_decision=%s",
        int(summary_row["reviewed_samples"]),
        int(summary_row["total_samples"]),
        final_decision(summary_row),
    )


if __name__ == "__main__":
    main()
