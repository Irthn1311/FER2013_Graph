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


PASS = "Vision batch 2521-2599: learned heatmap overlaps clear facial evidence."
PARTIAL = "Vision batch 2521-2599: learned heatmap has facial evidence but is mixed or diffuse."
FAIL = "Vision batch 2521-2599: learned heatmap is non-face, contour/background, or weak facial evidence."

DEFAULT_PARTIAL = learned("PARTIAL", 1, 1, 1, 1, 1, 1, 1, 1, PARTIAL)
DEFAULT_PASS = learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS)
DEFAULT_FAIL = learned("FAIL", 0, 0, 0, 2, 2, 1, 1, 0, FAIL)

PASS_IDS = {
    "AUDIT_002523",
    "AUDIT_002526",
    "AUDIT_002528",
    "AUDIT_002529",
    "AUDIT_002533",
    "AUDIT_002534",
    "AUDIT_002537",
    "AUDIT_002538",
    "AUDIT_002540",
    "AUDIT_002543",
    "AUDIT_002545",
    "AUDIT_002553",
    "AUDIT_002557",
    "AUDIT_002559",
    "AUDIT_002565",
    "AUDIT_002570",
    "AUDIT_002580",
    "AUDIT_002582",
    "AUDIT_002583",
    "AUDIT_002587",
    "AUDIT_002591",
    "AUDIT_002599",
}

FAIL_IDS = {
    "AUDIT_002544",
    "AUDIT_002547",
    "AUDIT_002556",
    "AUDIT_002566",
    "AUDIT_002594",
}

with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updates = {}
for i in range(2521, 2600):
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
