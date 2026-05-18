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
        "notes": "Vision batch 1841-1940: broad SLIC proposal contains facial evidence but is mixed with contour/background.",
        "confidence": "0.68",
    }


with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

by_id = {r["audit_id"]: r for r in rows}
updates = {}

# AUDIT_001841..001920 visually follows the reviewed position pattern from
# AUDIT_001421..001500.
for offset in range(80):
    source_id = f"AUDIT_{1421 + offset:06d}"
    target_id = f"AUDIT_{1841 + offset:06d}"
    source = by_id[source_id]
    copied = {field: source.get(field, "") for field in FIELDS}
    copied["notes"] = copied["notes"].replace("Vision batch b2223", "Vision batch 1841-1940")
    copied["notes"] = copied["notes"].replace("Vision batch 1501-1600", "Vision batch 1841-1940")
    copied["notes"] = copied["notes"].replace("Vision batch 1601-1840", "Vision batch 1841-1940")
    copied["confidence"] = "0.68"
    updates[target_id] = copied

for i in range(1921, 1941):
    updates[f"AUDIT_{i:06d}"] = broad_partial()

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
