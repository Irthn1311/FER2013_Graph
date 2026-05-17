#!/usr/bin/env python
"""Stage 4.6 Gemini Vision visual audit.

This script reviews existing selector visualization figures with Gemini Vision
and fills the Stage 4.6 visual audit sheet. It never trains models, modifies
graph artifacts, opens Stage 5, or infers visual conclusions from metric hints.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCORE_FIELDS = [
    "selected_eye_eyebrow",
    "selected_mouth_nasolabial",
    "selected_face_muscle_cheek_wrinkle",
    "selected_hair_glasses",
    "selected_border_background",
    "long_contour_dominant",
    "center_collapse",
    "fragmented_pixel_dust",
    "region_like",
    "facial_evidence_like",
]

VALID_PASS = {"PASS", "PARTIAL", "FAIL", "UNREVIEWED"}

BASE_COLUMNS = [
    "audit_id",
    "source_stage",
    "selector_name",
    "variant_name",
    "class_name",
    "graph_id",
    "ratio",
    "figure_path",
    *SCORE_FIELDS,
    "overall_visual_pass",
    "notes",
    "confidence",
    "metric_high_center_hint",
    "metric_high_border_hint",
    "metric_high_long_hint",
    "metric_high_fragmentation_hint",
    "metric_region_like_hint",
]

SUMMARY_SELECTOR_COLUMNS = [
    "selector_name",
    "source_stage",
    "reviewed_count",
    "pass_count",
    "partial_count",
    "fail_count",
    "unreviewed_count",
    "pass_rate",
    "partial_or_pass_rate",
    "avg_eye_eyebrow",
    "avg_mouth_nasolabial",
    "avg_face_muscle_cheek_wrinkle",
    "avg_hair_glasses",
    "avg_border_background",
    "avg_long_contour",
    "avg_center_collapse",
    "avg_fragmentation",
    "avg_region_like",
    "avg_facial_evidence_like",
    "avg_confidence",
    "main_failure_reason",
]

SUMMARY_CLASS_COLUMNS = [
    "class_name",
    "selector_name",
    "reviewed_count",
    "pass_rate",
    "partial_or_pass_rate",
    "avg_facial_evidence_like",
    "avg_hair_glasses",
    "avg_border_background",
    "avg_long_contour",
    "avg_center_collapse",
    "avg_fragmentation",
    "avg_region_like",
]

ERROR_COLUMNS = ["audit_id", "figure_path", "error_type", "message"]


PROMPT = """Bạn là reviewer thị giác cho một thí nghiệm FER-2013 48x48. Ảnh là visualization/overlay/mask/comparison của một selector evidence trên khuôn mặt.

Nhiệm vụ:
Đánh giá mask/overlay có chọn vùng evidence biểu cảm hay shortcut không.

Không được claim motif, semantic part, hay causal evidence.
Chỉ đánh giá visual evidence candidate.

Scoring:
0 = không thấy / rất ít / không đáng kể
1 = có một phần / mixed / vừa phải
2 = rõ ràng / dominant

Fields:

selected_eye_eyebrow:
- 0: không chọn vùng mắt/lông mày
- 1: có chọn một phần
- 2: chọn rõ mắt/mí/lông mày

selected_mouth_nasolabial:
- 0: không chọn miệng/rãnh mũi má
- 1: có một phần
- 2: rõ miệng/môi/khóe/rãnh mũi má

selected_face_muscle_cheek_wrinkle:
- 0: không thấy má/nếp nhăn/cơ mặt
- 1: có một phần
- 2: rõ

selected_hair_glasses:
- 0: không đáng kể
- 1: có một phần
- 2: dominant tóc/kính/gọng kính

selected_border_background:
- 0: không đáng kể
- 1: có một phần
- 2: dominant nền/viền ảnh/ngoài mặt/áo/cổ

long_contour_dominant:
- 0: không
- 1: có một phần
- 2: contour dài chi phối, giống viền mặt/tóc/nền hơn là cấu trúc biểu cảm

center_collapse:
- 0: không
- 1: hơi center-biased
- 2: blob trung tâm rõ, không bám mắt/miệng/lông mày

fragmented_pixel_dust:
- 0: liền vùng/ổn
- 1: hơi rời
- 2: rất rời rạc kiểu bụi pixel

region_like:
- 0: không region-like
- 1: trung bình
- 2: rõ ràng thành vùng/cụm có thể dùng làm region candidate

facial_evidence_like:
- 0: chủ yếu shortcut/nhiễu, không giống evidence biểu cảm
- 1: mixed, có một phần facial evidence nhưng còn risk
- 2: khá rõ là facial evidence candidate

overall_visual_pass:
- PASS nếu facial_evidence_like=2 và các risk chính <=1
- PARTIAL nếu có evidence nhưng mixed/risk
- FAIL nếu chủ yếu shortcut, tóc/kính, border/background, long contour, center blob, pixel dust, hoặc không thấy evidence
- UNREVIEWED nếu ảnh không đọc được

Cẩn thận:
- FER-2013 rất nhỏ; nếu không chắc, dùng PARTIAL hoặc ghi unclear.
- Mắt/miệng thật sự thường ở trung tâm, nên center không luôn xấu.
- Viền miệng/mí mắt có thể là contour hữu ích; chỉ đánh long_contour=2 nếu contour dài kiểu viền mặt/tóc/nền chi phối.
- Không overclaim motif/part.

Trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.

JSON schema:
{
  "selected_eye_eyebrow": 0,
  "selected_mouth_nasolabial": 0,
  "selected_face_muscle_cheek_wrinkle": 0,
  "selected_hair_glasses": 0,
  "selected_border_background": 0,
  "long_contour_dominant": 0,
  "center_collapse": 0,
  "fragmented_pixel_dust": 0,
  "region_like": 0,
  "facial_evidence_like": 0,
  "overall_visual_pass": "UNREVIEWED",
  "notes": "short note",
  "confidence": 0.0
}
"""


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            clean: Dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean_key = key.replace("\ufeff", "").strip().strip('"')
                clean[clean_key] = value
            rows.append(clean)
        return rows


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def ensure_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        normalized = dict(row)
        if "confidence" not in normalized:
            normalized["confidence"] = ""
        if not normalized.get("overall_visual_pass"):
            normalized["overall_visual_pass"] = "UNREVIEWED"
        for field in SCORE_FIELDS:
            normalized.setdefault(field, "")
        normalized.setdefault("notes", "")
        out.append(normalized)
    return out


def is_reviewed(row: Dict[str, str]) -> bool:
    return row.get("overall_visual_pass") in {"PASS", "PARTIAL", "FAIL"}


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def mean_score(rows: List[Dict[str, str]], field: str) -> str:
    values = [safe_float(r.get(field)) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


def load_existing_raw(raw_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not raw_jsonl.exists():
        return out
    with raw_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            audit_id = obj.get("audit_id")
            if audit_id and obj.get("status") == "ok" and isinstance(obj.get("parsed"), dict):
                out[audit_id] = obj["parsed"]
    return out


def apply_existing_raw(rows: List[Dict[str, str]], existing_raw: Dict[str, Dict[str, Any]]) -> int:
    synced = 0
    for idx, row in enumerate(rows):
        audit_id = row.get("audit_id", "")
        if audit_id in existing_raw and not is_reviewed(row):
            rows[idx] = update_row(row, validate_result(existing_raw[audit_id]))
            synced += 1
    return synced


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_checkpoint(
    output_sheet: Path,
    rows: List[Dict[str, str]],
    columns: Iterable[str],
    errors_csv: Path,
    errors: List[Dict[str, str]],
) -> None:
    write_csv(output_sheet, rows, columns)
    write_csv(errors_csv, errors, ERROR_COLUMNS)


def normalize_json_object(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError(f"expected JSON object, got {type(obj).__name__}")


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return normalize_json_object(json.loads(cleaned))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return normalize_json_object(json.loads(match.group(0)))
        raise


def validate_result(obj: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for field in SCORE_FIELDS:
        value = obj.get(field)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if value not in {0, 1, 2}:
            raise ValueError(f"invalid {field}: {value}")
        result[field] = str(value)
    overall = str(obj.get("overall_visual_pass", "UNREVIEWED")).strip().upper()
    if overall not in VALID_PASS:
        raise ValueError(f"invalid overall_visual_pass: {overall}")
    result["overall_visual_pass"] = overall
    notes = str(obj.get("notes", "")).replace("\n", " ").strip()
    result["notes"] = notes[:200]
    conf = safe_float(obj.get("confidence"))
    if conf is None:
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    result["confidence"] = f"{conf:.4f}"
    return result


def infer_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def make_client():
    from google import genai  # type: ignore

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_gemini(client: Any, model: str, image_path: Path) -> Tuple[Dict[str, str], str]:
    from google.genai import types  # type: ignore

    data = image_path.read_bytes()
    mime = infer_mime(image_path)
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            PROMPT,
        ],
        config=config,
    )
    text = getattr(response, "text", "") or ""
    parsed = parse_json_response(text)
    return validate_result(parsed), text


def update_row(row: Dict[str, str], result: Dict[str, str]) -> Dict[str, str]:
    updated = dict(row)
    for key, value in result.items():
        updated[key] = value
    return updated


def mark_unreviewed(row: Dict[str, str], note: str) -> Dict[str, str]:
    updated = dict(row)
    for field in SCORE_FIELDS:
        updated[field] = ""
    updated["overall_visual_pass"] = "UNREVIEWED"
    updated["notes"] = note[:200]
    updated["confidence"] = ""
    return updated


def retry_delay_seconds(message: str, attempt: int, base_sleep: float, quota_sleep: float) -> float:
    retry_match = re.search(r"retry in\s+([0-9.]+)s", message, flags=re.IGNORECASE)
    if retry_match:
        return max(base_sleep, float(retry_match.group(1)) + 1.0)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return max(base_sleep, quota_sleep)
    return min(30.0, (2**attempt) + base_sleep)


def failure_reason(rows: List[Dict[str, str]]) -> str:
    reviewed = [r for r in rows if is_reviewed(r)]
    if not reviewed:
        return "UNREVIEWED"
    avg_facial = safe_float(mean_score(reviewed, "facial_evidence_like"))
    if avg_facial is not None and avg_facial < 1.2:
        return "low_facial_evidence"
    risks = [
        ("hair_glasses", safe_float(mean_score(reviewed, "selected_hair_glasses"))),
        ("border_background", safe_float(mean_score(reviewed, "selected_border_background"))),
        ("long_contour", safe_float(mean_score(reviewed, "long_contour_dominant"))),
        ("center_collapse", safe_float(mean_score(reviewed, "center_collapse"))),
        ("fragmentation", safe_float(mean_score(reviewed, "fragmented_pixel_dust"))),
        ("low_confidence", 1.0 - (safe_float(mean_score(reviewed, "confidence")) or 0.0)),
    ]
    risks = [(name, value) for name, value in risks if value is not None]
    risks.sort(key=lambda item: item[1], reverse=True)
    return risks[0][0] if risks else "none"


def summarize_by_selector(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("selector_name", "UNKNOWN") or "UNKNOWN"].append(row)
    out: List[Dict[str, object]] = []
    for selector, group_rows in sorted(grouped.items()):
        reviewed = [r for r in group_rows if is_reviewed(r)]
        counts = Counter(r.get("overall_visual_pass", "UNREVIEWED") for r in group_rows)
        reviewed_count = len(reviewed)
        pass_count = counts.get("PASS", 0)
        partial_count = counts.get("PARTIAL", 0)
        fail_count = counts.get("FAIL", 0)
        stages = sorted({r.get("source_stage", "UNKNOWN") for r in group_rows})
        out.append(
            {
                "selector_name": selector,
                "source_stage": ";".join(stages),
                "reviewed_count": reviewed_count,
                "pass_count": pass_count,
                "partial_count": partial_count,
                "fail_count": fail_count,
                "unreviewed_count": len(group_rows) - reviewed_count,
                "pass_rate": f"{pass_count / reviewed_count:.6f}" if reviewed_count else "",
                "partial_or_pass_rate": f"{(pass_count + partial_count) / reviewed_count:.6f}" if reviewed_count else "",
                "avg_eye_eyebrow": mean_score(reviewed, "selected_eye_eyebrow"),
                "avg_mouth_nasolabial": mean_score(reviewed, "selected_mouth_nasolabial"),
                "avg_face_muscle_cheek_wrinkle": mean_score(reviewed, "selected_face_muscle_cheek_wrinkle"),
                "avg_hair_glasses": mean_score(reviewed, "selected_hair_glasses"),
                "avg_border_background": mean_score(reviewed, "selected_border_background"),
                "avg_long_contour": mean_score(reviewed, "long_contour_dominant"),
                "avg_center_collapse": mean_score(reviewed, "center_collapse"),
                "avg_fragmentation": mean_score(reviewed, "fragmented_pixel_dust"),
                "avg_region_like": mean_score(reviewed, "region_like"),
                "avg_facial_evidence_like": mean_score(reviewed, "facial_evidence_like"),
                "avg_confidence": mean_score(reviewed, "confidence"),
                "main_failure_reason": failure_reason(group_rows),
            }
        )
    return out


def summarize_by_class(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("class_name", "UNKNOWN") or "UNKNOWN",
                row.get("selector_name", "UNKNOWN") or "UNKNOWN",
            )
        ].append(row)
    out: List[Dict[str, object]] = []
    for (class_name, selector), group_rows in sorted(grouped.items()):
        reviewed = [r for r in group_rows if is_reviewed(r)]
        pass_count = sum(1 for r in reviewed if r.get("overall_visual_pass") == "PASS")
        partial_count = sum(1 for r in reviewed if r.get("overall_visual_pass") == "PARTIAL")
        out.append(
            {
                "class_name": class_name,
                "selector_name": selector,
                "reviewed_count": len(reviewed),
                "pass_rate": f"{pass_count / len(reviewed):.6f}" if reviewed else "",
                "partial_or_pass_rate": f"{(pass_count + partial_count) / len(reviewed):.6f}" if reviewed else "",
                "avg_facial_evidence_like": mean_score(reviewed, "facial_evidence_like"),
                "avg_hair_glasses": mean_score(reviewed, "selected_hair_glasses"),
                "avg_border_background": mean_score(reviewed, "selected_border_background"),
                "avg_long_contour": mean_score(reviewed, "long_contour_dominant"),
                "avg_center_collapse": mean_score(reviewed, "center_collapse"),
                "avg_fragmentation": mean_score(reviewed, "fragmented_pixel_dust"),
                "avg_region_like": mean_score(reviewed, "region_like"),
            }
        )
    return out


def risk_cases(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        note = (row.get("notes") or "").lower()
        risk = (
            (safe_float(row.get("selected_hair_glasses")) or 0) >= 2
            or (safe_float(row.get("selected_border_background")) or 0) >= 2
            or (safe_float(row.get("long_contour_dominant")) or 0) >= 2
            or (safe_float(row.get("center_collapse")) or 0) >= 2
            or (safe_float(row.get("fragmented_pixel_dust")) or 0) >= 2
            or row.get("overall_visual_pass") == "FAIL"
            or (safe_float(row.get("confidence")) is not None and (safe_float(row.get("confidence")) or 0) < 0.45)
            or any(token in note for token in ["unclear", "unreadable", "missing"])
        )
        if risk:
            out.append(row)
    return out


def selector_decisions(summary_rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    out = []
    for row in summary_rows:
        reviewed = int(row.get("reviewed_count") or 0)
        if reviewed <= 0:
            decision = "VISUAL_UNREVIEWED"
            reason = "no reviewed rows"
        else:
            pass_count = int(row.get("pass_count") or 0)
            partial_count = int(row.get("partial_count") or 0)
            fail_count = int(row.get("fail_count") or 0)
            partial_or_pass_rate = (pass_count + partial_count) / reviewed
            fail_rate = fail_count / reviewed
            facial = safe_float(row.get("avg_facial_evidence_like")) or 0
            hair = safe_float(row.get("avg_hair_glasses")) or 0
            border = safe_float(row.get("avg_border_background")) or 0
            long = safe_float(row.get("avg_long_contour")) or 0
            center = safe_float(row.get("avg_center_collapse")) or 0
            frag = safe_float(row.get("avg_fragmentation")) or 0
            pass_ok = (
                reviewed >= 30
                and partial_or_pass_rate >= 0.60
                and fail_rate <= 0.30
                and facial >= 1.2
                and hair <= 0.8
                and border <= 0.8
                and long <= 0.8
                and center <= 0.8
                and frag <= 1.0
            )
            partial_ok = reviewed >= 10 and partial_or_pass_rate >= 0.45 and facial >= 0.9
            if pass_ok:
                decision = "VISUAL_PASS_FOR_DELETION_MEASUREMENT"
            elif partial_ok:
                decision = "VISUAL_PARTIAL_NEEDS_HEURISTIC_REFINEMENT"
            else:
                decision = "VISUAL_FAIL_OR_HOLD"
            reason = (
                f"reviewed={reviewed}, partial_or_pass={partial_or_pass_rate:.3f}, fail_rate={fail_rate:.3f}, "
                f"facial={facial:.3f}, hair={hair:.3f}, border={border:.3f}, long={long:.3f}, "
                f"center={center:.3f}, fragmentation={frag:.3f}"
            )
        out.append({"selector_name": str(row.get("selector_name")), "decision": decision, "reason": reason})
    return out


def write_pass_fail_decision(path: Path, summary_rows: List[Dict[str, object]], status: str) -> str:
    decisions = selector_decisions(summary_rows)
    passed = [d for d in decisions if d["decision"] == "VISUAL_PASS_FOR_DELETION_MEASUREMENT"]
    partial = [d for d in decisions if d["decision"] == "VISUAL_PARTIAL_NEEDS_HEURISTIC_REFINEMENT"]
    if status == "MISSING_GEMINI_API_KEY":
        overall = "MISSING_GEMINI_API_KEY"
    elif passed:
        overall = "VISUAL_AUDIT_PASS_RUN_DELETION_MEASUREMENT"
    elif partial:
        overall = "VISUAL_AUDIT_PARTIAL_REFINE_HEURISTIC"
    else:
        overall = "VISUAL_AUDIT_FAIL_STOP_STAGE5" if decisions else "VISUAL_AUDIT_UNREVIEWED"
    lines = [
        "# Stage 4.6 Visual Audit Pass/Fail Decision Filled",
        "",
        f"Overall decision: `{overall}`.",
        "",
        "No selector opens Stage 5 directly from visual audit. Stage 5 remains `LOCKED`.",
        "",
        "## Selector Decisions",
        "",
    ]
    for d in decisions:
        lines.append(f"- `{d['selector_name']}`: `{d['decision']}` ({d['reason']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall


def top_selector_lines(summary_rows: List[Dict[str, object]], n: int = 12) -> str:
    rows = sorted(
        summary_rows,
        key=lambda r: (
            safe_float(r.get("partial_or_pass_rate")) if safe_float(r.get("partial_or_pass_rate")) is not None else -1,
            safe_float(r.get("avg_facial_evidence_like")) if safe_float(r.get("avg_facial_evidence_like")) is not None else -1,
        ),
        reverse=True,
    )[:n]
    if not rows:
        return "| none | 0 |  |  |  |\n"
    return "\n".join(
        f"| `{r['selector_name']}` | {r['reviewed_count']} | {r['partial_or_pass_rate']} | {r['avg_facial_evidence_like']} | {r['main_failure_reason']} |"
        for r in rows
    ) + "\n"


def top_risk_lines(summary_rows: List[Dict[str, object]], n: int = 10) -> str:
    rows = sorted(
        summary_rows,
        key=lambda r: max(
            safe_float(r.get("avg_hair_glasses")) or 0,
            safe_float(r.get("avg_border_background")) or 0,
            safe_float(r.get("avg_long_contour")) or 0,
            safe_float(r.get("avg_center_collapse")) or 0,
            safe_float(r.get("avg_fragmentation")) or 0,
        ),
        reverse=True,
    )[:n]
    if not rows:
        return "| none |  |  |  |  |  |\n"
    return "\n".join(
        f"| `{r['selector_name']}` | {r['avg_hair_glasses']} | {r['avg_border_background']} | {r['avg_long_contour']} | {r['avg_center_collapse']} | {r['avg_fragmentation']} |"
        for r in rows
    ) + "\n"


def write_report(
    path: Path,
    rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, object]],
    class_rows: List[Dict[str, object]],
    risk_rows: List[Dict[str, str]],
    model: str,
    errors: List[Dict[str, str]],
    decision: str,
    status: str,
) -> None:
    reviewed = [r for r in rows if is_reviewed(r)]
    selectors = sorted({r.get("selector_name", "UNKNOWN") for r in rows})
    class_review_counts = Counter(r.get("class_name", "UNKNOWN") for r in reviewed)
    class_lines = "\n".join(f"- `{k}`: `{v}` reviewed" for k, v in sorted(class_review_counts.items())) or "- none"
    report = f"""# Stage 4.6 Gemini Visual Audit Report

## 1. Executive Summary

- Total rows: `{len(rows)}`.
- Reviewed rows: `{len(reviewed)}`.
- Model used: `{model}`.
- Run status: `{status}`.
- Errors: `{len(errors)}`.
- Stage 5 status: `locked`.
- Overall decision: `{decision}`.

## 2. Method

- Gemini Vision API through Google GenAI SDK.
- API key is read from `GEMINI_API_KEY`; it is never logged.
- Output is validated as JSON with numeric 0/1/2 visual fields, `overall_visual_pass`, notes, and confidence.
- Metric hints are not used as visual conclusions.
- No motif, semantic part, causal, or Q1 claim is made.

## 3. Selector-level Results

| Selector | Reviewed | Partial/pass rate | Facial evidence avg | Main failure reason |
|---|---:|---:|---:|---|
{top_selector_lines(summary_rows)}

## 4. Class-level Results

Reviewed rows by class:

{class_lines}

Class-level CSV is written to `visual_audit_summary_by_class_filled.csv`.

## 5. Risk Cases

- Risk case rows: `{len(risk_rows)}`.
- Risk cases include hair/glasses, border/background, long contour, center collapse, fragmentation, FAIL, low confidence, or unclear/missing notes.

High-risk selector summary:

| Selector | Hair | Border | Long | Center | Fragmentation |
|---|---:|---:|---:|---:|---:|
{top_risk_lines(summary_rows)}

## 6. Recommended Selector For Next Deletion Measurement

Choose at most three only if selector decisions include `VISUAL_PASS_FOR_DELETION_MEASUREMENT`:

- one information-rich selector;
- one region/continuity selector;
- one structure-aware selector.

Do not choose learned selector as main. If no selector passes, perform heuristic refinement or complete more visual review.

## 7. Decision

Decision: `{decision}`.

Stage 5 remains locked. Visual PASS, if any, only permits the next analysis-only fixed-original + local_mean deletion measurement.

## 8. What Not To Claim

- Not motif.
- Not semantic part.
- Not causal.
- Not Q1.
- Stage 5 still locked.
"""
    path.write_text(report, encoding="utf-8")


def write_missing_key_outputs(
    audit_dir: Path,
    rows: List[Dict[str, str]],
    columns: List[str],
    model: str,
    output_sheet: Path,
) -> None:
    write_csv(output_sheet, rows, columns)
    errors = [{"audit_id": "GLOBAL", "figure_path": "", "error_type": "MISSING_GEMINI_API_KEY", "message": "GEMINI_API_KEY is not set"}]
    write_csv(audit_dir / "gemini_audit_errors.csv", errors, ERROR_COLUMNS)
    (audit_dir / "gemini_audit_raw_jsonl.jsonl").touch()
    summary = summarize_by_selector(rows)
    class_summary = summarize_by_class(rows)
    risks = risk_cases(rows)
    write_csv(audit_dir / "visual_audit_summary_by_selector_filled.csv", summary, SUMMARY_SELECTOR_COLUMNS)
    write_csv(audit_dir / "visual_audit_summary_by_class_filled.csv", class_summary, SUMMARY_CLASS_COLUMNS)
    write_csv(audit_dir / "visual_audit_risk_cases_filled.csv", risks, columns)
    decision = write_pass_fail_decision(
        audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md",
        summary,
        "MISSING_GEMINI_API_KEY",
    )
    write_report(
        audit_dir / "stage46_gemini_visual_audit_report.md",
        rows,
        summary,
        class_summary,
        risks,
        model,
        errors,
        decision,
        "MISSING_GEMINI_API_KEY",
    )


def should_process(row: Dict[str, str], args: argparse.Namespace, index: int) -> bool:
    if index < args.start_index:
        return False
    if args.selector_filter and row.get("selector_name") != args.selector_filter and row.get("variant_name") != args.selector_filter:
        return False
    if args.class_filter and row.get("class_name") != args.class_filter:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini Vision audit over Stage 4.6 visual audit sheet.")
    parser.add_argument("--audit_dir", default="outputs/stage4_6_visual_audit")
    parser.add_argument("--sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet.csv")
    parser.add_argument("--manifest", default="outputs/stage4_6_visual_audit/visual_audit_manifest.csv")
    parser.add_argument("--output_sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--selector_filter", default=None)
    parser.add_argument("--class_filter", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--quota_sleep", type=float, default=65.0)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resume", default="true")
    parser.add_argument("--dry_run", default="false")
    parser.add_argument("--sync_raw_only", default="false")
    parser.add_argument("--max_file_mb", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = Path(args.sheet)
    output_sheet = Path(args.output_sheet)
    raw_jsonl = audit_dir / "gemini_audit_raw_jsonl.jsonl"
    errors_csv = audit_dir / "gemini_audit_errors.csv"
    resume = parse_bool(args.resume)
    dry_run = parse_bool(args.dry_run)
    sync_raw_only = parse_bool(args.sync_raw_only)
    checkpoint_every = max(1, args.checkpoint_every)

    source_sheet = output_sheet if resume and output_sheet.exists() else sheet_path
    rows = ensure_columns(read_csv(source_sheet))
    if not rows:
        rows = ensure_columns(read_csv(sheet_path))
    columns = list(dict.fromkeys([*BASE_COLUMNS, *([c for row in rows for c in row.keys()])]))

    existing_raw = load_existing_raw(raw_jsonl) if resume else {}
    synced_from_raw = apply_existing_raw(rows, existing_raw) if existing_raw else 0

    if sync_raw_only:
        errors = read_csv(errors_csv) if errors_csv.exists() else []
        write_checkpoint(output_sheet, rows, columns, errors_csv, errors)
        summary = summarize_by_selector(rows)
        class_summary = summarize_by_class(rows)
        risks = risk_cases(rows)
        write_csv(audit_dir / "visual_audit_summary_by_selector_filled.csv", summary, SUMMARY_SELECTOR_COLUMNS)
        write_csv(audit_dir / "visual_audit_summary_by_class_filled.csv", class_summary, SUMMARY_CLASS_COLUMNS)
        write_csv(audit_dir / "visual_audit_risk_cases_filled.csv", risks, columns)
        decision = write_pass_fail_decision(audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md", summary, "RAW_SYNC_ONLY")
        write_report(audit_dir / "stage46_gemini_visual_audit_report.md", rows, summary, class_summary, risks, args.model, errors, decision, "RAW_SYNC_ONLY")
        print("[Stage4.6Gemini] status=RAW_SYNC_ONLY")
        print(f"[Stage4.6Gemini] synced_from_raw={synced_from_raw}")
        print(f"[Stage4.6Gemini] reviewed_rows={sum(1 for r in rows if is_reviewed(r))}")
        print("[Stage4.6Gemini] stage5=no")
        return

    if dry_run:
        write_csv(output_sheet, rows, columns)
        errors: List[Dict[str, str]] = []
        write_csv(errors_csv, errors, ERROR_COLUMNS)
        summary = summarize_by_selector(rows)
        class_summary = summarize_by_class(rows)
        risks = risk_cases(rows)
        write_csv(audit_dir / "visual_audit_summary_by_selector_filled.csv", summary, SUMMARY_SELECTOR_COLUMNS)
        write_csv(audit_dir / "visual_audit_summary_by_class_filled.csv", class_summary, SUMMARY_CLASS_COLUMNS)
        write_csv(audit_dir / "visual_audit_risk_cases_filled.csv", risks, columns)
        decision = write_pass_fail_decision(audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md", summary, "DRY_RUN")
        write_report(audit_dir / "stage46_gemini_visual_audit_report.md", rows, summary, class_summary, risks, args.model, errors, decision, "DRY_RUN")
        print("[Stage4.6Gemini] dry_run=true")
        print(f"[Stage4.6Gemini] rows={len(rows)}")
        print("[Stage4.6Gemini] stage5=no")
        return

    if not os.environ.get("GEMINI_API_KEY"):
        write_missing_key_outputs(audit_dir, rows, columns, args.model, output_sheet)
        print("[Stage4.6Gemini] status=MISSING_GEMINI_API_KEY")
        print(f"[Stage4.6Gemini] reviewed_rows={sum(1 for r in rows if is_reviewed(r))}")
        print("[Stage4.6Gemini] errors=1")
        print("[Stage4.6Gemini] decision=MISSING_GEMINI_API_KEY")
        print("[Stage4.6Gemini] stage5=no")
        return

    try:
        client = make_client()
    except Exception as exc:
        errors = [{"audit_id": "GLOBAL", "figure_path": "", "error_type": "SDK_IMPORT_OR_CLIENT_ERROR", "message": str(exc)[:500]}]
        write_csv(errors_csv, errors, ERROR_COLUMNS)
        write_csv(output_sheet, rows, columns)
        summary = summarize_by_selector(rows)
        class_summary = summarize_by_class(rows)
        risks = risk_cases(rows)
        decision = write_pass_fail_decision(audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md", summary, "SDK_IMPORT_OR_CLIENT_ERROR")
        write_report(audit_dir / "stage46_gemini_visual_audit_report.md", rows, summary, class_summary, risks, args.model, errors, decision, "SDK_IMPORT_OR_CLIENT_ERROR")
        print("[Stage4.6Gemini] status=SDK_IMPORT_OR_CLIENT_ERROR")
        print("[Stage4.6Gemini] stage5=no")
        return

    processed = 0
    errors: List[Dict[str, str]] = read_csv(errors_csv) if resume and errors_csv.exists() else []
    if resume:
        reviewed_ids = {row.get("audit_id", "") for row in rows if is_reviewed(row)}
        errors = [
            err
            for err in errors
            if err.get("audit_id") != "GLOBAL" and err.get("audit_id", "") not in reviewed_ids
        ]
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    for idx, row in enumerate(rows):
        if args.max_rows is not None and processed >= args.max_rows:
            break
        if not should_process(row, args, idx):
            continue
        if resume and is_reviewed(row):
            continue
        audit_id = row.get("audit_id", f"row_{idx}")
        if resume and audit_id in existing_raw:
            rows[idx] = update_row(row, validate_result(existing_raw[audit_id]))
            processed += 1
            if processed % checkpoint_every == 0:
                write_checkpoint(output_sheet, rows, columns, errors_csv, errors)
            continue

        image_path = Path(row.get("figure_path", ""))
        if not image_path.exists():
            rows[idx] = mark_unreviewed(row, "missing image")
            err = {"audit_id": audit_id, "figure_path": str(image_path), "error_type": "MISSING_IMAGE", "message": "figure_path does not exist"}
            errors.append(err)
            append_jsonl(raw_jsonl, {"audit_id": audit_id, "status": "error", **err})
            processed += 1
            if processed % checkpoint_every == 0:
                write_checkpoint(output_sheet, rows, columns, errors_csv, errors)
            continue
        if image_path.stat().st_size > max_bytes:
            rows[idx] = mark_unreviewed(row, "file_too_large")
            err = {"audit_id": audit_id, "figure_path": str(image_path), "error_type": "FILE_TOO_LARGE", "message": f"file exceeds {args.max_file_mb}MB"}
            errors.append(err)
            append_jsonl(raw_jsonl, {"audit_id": audit_id, "status": "error", **err})
            processed += 1
            if processed % checkpoint_every == 0:
                write_checkpoint(output_sheet, rows, columns, errors_csv, errors)
            continue

        last_error: Optional[str] = None
        for attempt in range(args.max_retries):
            try:
                result, raw_text = call_gemini(client, args.model, image_path)
                rows[idx] = update_row(row, result)
                errors = [err for err in errors if err.get("audit_id") != audit_id]
                append_jsonl(
                    raw_jsonl,
                    {
                        "audit_id": audit_id,
                        "status": "ok",
                        "model": args.model,
                        "figure_path": str(image_path),
                        "parsed": result,
                        "raw_text": raw_text,
                    },
                )
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)[:500]
                if attempt < args.max_retries - 1:
                    time.sleep(retry_delay_seconds(last_error, attempt, args.sleep, args.quota_sleep))
        if last_error:
            rows[idx] = mark_unreviewed(row, "json_parse_or_api_error")
            err = {"audit_id": audit_id, "figure_path": str(image_path), "error_type": "GEMINI_ERROR", "message": last_error}
            errors.append(err)
            append_jsonl(raw_jsonl, {"audit_id": audit_id, "status": "error", **err})
        processed += 1
        if processed % checkpoint_every == 0:
            write_checkpoint(output_sheet, rows, columns, errors_csv, errors)
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_csv(output_sheet, rows, columns)
    write_csv(errors_csv, errors, ERROR_COLUMNS)
    summary = summarize_by_selector(rows)
    class_summary = summarize_by_class(rows)
    risks = risk_cases(rows)
    write_csv(audit_dir / "visual_audit_summary_by_selector_filled.csv", summary, SUMMARY_SELECTOR_COLUMNS)
    write_csv(audit_dir / "visual_audit_summary_by_class_filled.csv", class_summary, SUMMARY_CLASS_COLUMNS)
    write_csv(audit_dir / "visual_audit_risk_cases_filled.csv", risks, columns)
    decision = write_pass_fail_decision(audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md", summary, "OK")
    write_report(audit_dir / "stage46_gemini_visual_audit_report.md", rows, summary, class_summary, risks, args.model, errors, decision, "OK")

    reviewed = [r for r in rows if is_reviewed(r)]
    ranked = sorted(
        summary,
        key=lambda r: (
            safe_float(r.get("partial_or_pass_rate")) if safe_float(r.get("partial_or_pass_rate")) is not None else -1,
            safe_float(r.get("avg_facial_evidence_like")) if safe_float(r.get("avg_facial_evidence_like")) is not None else -1,
        ),
        reverse=True,
    )
    top3 = [r["selector_name"] for r in ranked[:3] if safe_float(r.get("partial_or_pass_rate")) is not None]
    risk_ranked = sorted(
        summary,
        key=lambda r: max(
            safe_float(r.get("avg_hair_glasses")) or 0,
            safe_float(r.get("avg_border_background")) or 0,
            safe_float(r.get("avg_long_contour")) or 0,
            safe_float(r.get("avg_center_collapse")) or 0,
            safe_float(r.get("avg_fragmentation")) or 0,
        ),
        reverse=True,
    )
    risk3 = [r["selector_name"] for r in risk_ranked[:3] if safe_float(r.get("avg_facial_evidence_like")) is not None]

    print(f"[Stage4.6Gemini] model={args.model}")
    print(f"[Stage4.6Gemini] reviewed_rows={len(reviewed)}")
    print(f"[Stage4.6Gemini] errors={len(errors)}")
    print(f"[Stage4.6Gemini] top3_selectors={top3}")
    print(f"[Stage4.6Gemini] top3_risk_selectors={risk3}")
    print(f"[Stage4.6Gemini] decision={decision}")
    print("[Stage4.6Gemini] stage5=no")


if __name__ == "__main__":
    main()
