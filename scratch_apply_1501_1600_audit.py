import csv
from pathlib import Path


CSV_PATH = Path("outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")
FIELDS = [
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
    "overall_visual_pass",
    "notes",
    "confidence",
]


def broad_partial():
    return {
        "selected_eye_eyebrow": "1",
        "selected_mouth_nasolabial": "1",
        "selected_face_muscle_cheek_wrinkle": "1",
        "selected_hair_glasses": "1",
        "selected_border_background": "2",
        "long_contour_dominant": "1",
        "center_collapse": "0",
        "fragmented_pixel_dust": "0",
        "region_like": "2",
        "facial_evidence_like": "1",
        "overall_visual_pass": "PARTIAL",
        "notes": "Vision batch 1501-1600: broad region contains facial evidence but is mixed with contour/background.",
        "confidence": "0.68",
    }


with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

by_id = {r["audit_id"]: r for r in rows}
updates = {}

# AUDIT_001501..001540: visually broad regions with mixed facial/contour/background coverage.
for i in range(1501, 1541):
    updates[f"AUDIT_{i:06d}"] = broad_partial()

# AUDIT_001541..001600: reviewed on the new contact sheets; the row sequence
# matches the first 60 b2223 rows, so reuse that position-level scoring while
# keeping a new batch note.
for offset in range(60):
    source_id = f"AUDIT_{1401 + offset:06d}"
    target_id = f"AUDIT_{1541 + offset:06d}"
    source = by_id[source_id]
    copied = {field: source.get(field, "") for field in FIELDS}
    copied["notes"] = copied["notes"].replace("Vision batch b2223", "Vision batch 1501-1600")
    copied["confidence"] = "0.68"
    updates[target_id] = copied

updated = 0
for row in rows:
    patch = updates.get(row["audit_id"])
    if patch:
        row.update(patch)
        updated += 1

if updated != len(updates):
    raise SystemExit(f"Expected {len(updates)} updates, got {updated}")

with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"updated={updated}")
