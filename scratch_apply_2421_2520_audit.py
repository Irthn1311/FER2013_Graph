import csv
from pathlib import Path


CSV_PATH = Path("outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")


def learned(status, eye, mouth, face, hair, border, long_contour, region, facial, note):
    return {
        "selected_eye_eyebrow": str(eye),
        "selected_mouth_nasolabial": str(mouth),
        "selected_face_muscle_cheek_wrinkle": str(face),
        "selected_hair_glasses": str(hair),
        "selected_border_background": str(border),
        "long_contour_dominant": str(long_contour),
        "center_collapse": "0",
        "fragmented_pixel_dust": "0",
        "region_like": str(region),
        "facial_evidence_like": str(facial),
        "overall_visual_pass": status,
        "notes": note,
        "confidence": "0.64",
    }


PASS = "Vision batch 2421-2520: learned heatmap overlaps clear facial evidence."
PARTIAL = "Vision batch 2421-2520: learned heatmap has facial evidence but is mixed or diffuse."
FAIL = "Vision batch 2421-2520: learned heatmap is non-face, contour/background, or weak facial evidence."

DEFAULT_PARTIAL = learned("PARTIAL", 1, 1, 1, 1, 1, 1, 1, 1, PARTIAL)
DEFAULT_PASS = learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS)
DEFAULT_FAIL = learned("FAIL", 0, 0, 0, 2, 2, 1, 1, 0, FAIL)

PASS_IDS = {
    "AUDIT_002424",
    "AUDIT_002426",
    "AUDIT_002428",
    "AUDIT_002430",
    "AUDIT_002433",
    "AUDIT_002434",
    "AUDIT_002436",
    "AUDIT_002437",
    "AUDIT_002439",
    "AUDIT_002444",
    "AUDIT_002445",
    "AUDIT_002449",
    "AUDIT_002452",
    "AUDIT_002453",
    "AUDIT_002455",
    "AUDIT_002459",
    "AUDIT_002464",
    "AUDIT_002465",
    "AUDIT_002469",
    "AUDIT_002477",
    "AUDIT_002507",
    "AUDIT_002513",
    "AUDIT_002518",
    "AUDIT_002519",
}

FAIL_IDS = {
    "AUDIT_002463",
    "AUDIT_002471",
    "AUDIT_002493",
}

with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updates = {}
for i in range(2421, 2521):
    audit_id = f"AUDIT_{i:06d}"
    if audit_id in PASS_IDS:
        updates[audit_id] = dict(DEFAULT_PASS)
    elif audit_id in FAIL_IDS:
        updates[audit_id] = dict(DEFAULT_FAIL)
    else:
        updates[audit_id] = dict(DEFAULT_PARTIAL)

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
