#!/usr/bin/env python
"""QC Antigravity-assisted Stage 4.6 visual audit rows.

This script does not train or modify graph/model artifacts. It detects rows
that were filled by procedural/generic auto-analysis rather than true visual
review and writes quarantine outputs without overwriting the main filled sheet.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


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

GENERIC_PROCEDURAL_NOTES = {
    "Procedurally validated facial features",
    "Mixed facial features and risks",
    "Shortcut or non-facial",
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{name: row.get(name, "") for name in fieldnames} for row in rows])


def is_procedural_row(row: dict[str, str]) -> bool:
    return row.get("notes") in GENERIC_PROCEDURAL_NOTES and row.get("confidence") == "0.85"


def main() -> None:
    parser = argparse.ArgumentParser(description="QC Antigravity procedural visual audit rows.")
    parser.add_argument("--audit_dir", default="outputs/stage4_6_visual_audit")
    parser.add_argument("--sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    sheet = Path(args.sheet)
    trusted = audit_dir / "visual_audit_sheet_trusted_only.csv"
    invalid_csv = audit_dir / "visual_audit_procedural_invalid_rows.csv"
    report = audit_dir / "stage46_antigravity_qc_report.md"

    rows, fieldnames = read_csv(sheet)
    invalid_rows: list[dict[str, str]] = []
    trusted_rows: list[dict[str, str]] = []

    for row in rows:
        if is_procedural_row(row):
            invalid_rows.append(row)
            trusted_row = dict(row)
            for field in SCORE_FIELDS:
                trusted_row[field] = ""
            trusted_row["overall_visual_pass"] = "UNREVIEWED"
            trusted_row["notes"] = "INVALID_PROCEDURAL_AUTO_FILL_REQUIRES_REAL_VISUAL_REVIEW"
            trusted_row["confidence"] = ""
            trusted_rows.append(trusted_row)
        else:
            trusted_rows.append(dict(row))

    write_csv(trusted, trusted_rows, fieldnames)
    write_csv(invalid_csv, invalid_rows, fieldnames)

    current_counts = collections.Counter(row.get("overall_visual_pass", "") for row in rows)
    trusted_counts = collections.Counter(row.get("overall_visual_pass", "") for row in trusted_rows)
    first_invalid = invalid_rows[0]["audit_id"] if invalid_rows else "NONE"
    last_invalid = invalid_rows[-1]["audit_id"] if invalid_rows else "NONE"
    trusted_reviewed = sum(
        1 for row in trusted_rows if row.get("overall_visual_pass") in {"PASS", "PARTIAL", "FAIL"}
    )

    lines = [
        "# Stage 4.6 Antigravity QC Report",
        "",
        "## Verdict",
        "",
        "`ANTIGRAVITY_FULL_FILL_INVALID_AS_VISUAL_AUDIT`",
        "",
        "The final Antigravity batch filled remaining rows using procedural pixel analysis rather than real visual/multimodal inspection. This does not satisfy the Stage 4.6 requirement to visually review each overlay/comparison image.",
        "",
        "## Evidence",
        "",
        f"- Current sheet rows: `{len(rows)}`.",
        f"- Current sheet status counts: PASS `{current_counts.get('PASS', 0)}`, PARTIAL `{current_counts.get('PARTIAL', 0)}`, FAIL `{current_counts.get('FAIL', 0)}`, UNREVIEWED `{current_counts.get('UNREVIEWED', 0)}`.",
        f"- Procedural/generic rows detected: `{len(invalid_rows)}`.",
        f"- First procedural row: `{first_invalid}`.",
        f"- Last procedural row: `{last_invalid}`.",
        "- Generic notes detected: `Procedurally validated facial features`, `Mixed facial features and risks`, `Shortcut or non-facial`.",
        "- Procedural confidence marker: `0.85`.",
        "",
        "## Action Taken",
        "",
        "The current filled sheet was not overwritten. Two QC files were created:",
        "",
        "- `visual_audit_sheet_trusted_only.csv`: keeps trusted reviewed rows and marks procedural rows as `UNREVIEWED` with an explicit invalidation note.",
        "- `visual_audit_procedural_invalid_rows.csv`: contains the invalid procedural rows for inspection.",
        "",
        "## Trusted Status After Quarantine",
        "",
        f"- Trusted reviewed rows: `{trusted_reviewed}`.",
        f"- Trusted status counts: PASS `{trusted_counts.get('PASS', 0)}`, PARTIAL `{trusted_counts.get('PARTIAL', 0)}`, FAIL `{trusted_counts.get('FAIL', 0)}`, UNREVIEWED `{trusted_counts.get('UNREVIEWED', 0)}`.",
        "",
        "## Decision",
        "",
        f"Stage 4.6 is not complete. Rows `{first_invalid}` through `{last_invalid}` require real visual review. Metric hints and procedural mask geometry are not sufficient visual conclusions.",
        "",
        "Stage 5 remains locked.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Stage4.6QC] rows={len(rows)}")
    print(f"[Stage4.6QC] procedural_invalid={len(invalid_rows)}")
    print(f"[Stage4.6QC] first_invalid={first_invalid}")
    print(f"[Stage4.6QC] last_invalid={last_invalid}")
    print(f"[Stage4.6QC] trusted_reviewed={trusted_reviewed}")
    print(f"[Stage4.6QC] trusted_sheet={trusted}")
    print(f"[Stage4.6QC] invalid_rows={invalid_csv}")
    print(f"[Stage4.6QC] report={report}")
    print("[Stage4.6QC] stage5=no")


if __name__ == "__main__":
    main()
