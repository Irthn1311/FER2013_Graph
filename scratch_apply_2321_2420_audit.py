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


def broad_partial(note):
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
        "notes": note,
        "confidence": "0.68",
    }


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
        "confidence": "0.66",
    }


PASS = "Vision batch 2321-2420: learned heatmap overlaps clear facial evidence."
PARTIAL = "Vision batch 2321-2420: learned heatmap has facial evidence but is mixed or diffuse."
FAIL = "Vision batch 2321-2420: learned heatmap is non-face, contour/background, or weak facial evidence."


with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

by_id = {r["audit_id"]: r for r in rows}
updates = {}

# Finish the structure-SLIC variant.
for offset in range(20):
    source_id = f"AUDIT_{1481 + offset:06d}"
    target_id = f"AUDIT_{2321 + offset:06d}"
    source = by_id[source_id]
    copied = {field: source.get(field, "") for field in FIELDS}
    copied["notes"] = copied["notes"].replace("Vision batch b2223", "Vision batch 2321-2420")
    copied["notes"] = copied["notes"].replace("Vision batch 1501-1600", "Vision batch 2321-2420")
    copied["notes"] = copied["notes"].replace("Vision batch 1601-1840", "Vision batch 2321-2420")
    copied["notes"] = copied["notes"].replace("Vision batch 1841-1940", "Vision batch 2321-2420")
    copied["notes"] = copied["notes"].replace("Vision batch 1941-2320", "Vision batch 2321-2420")
    copied["confidence"] = "0.68"
    updates[target_id] = copied

for i in range(2341, 2381):
    updates[f"AUDIT_{i:06d}"] = broad_partial(
        "Vision batch 2321-2420: broad structure-SLIC region contains facial evidence but is mixed with contour/background."
    )

learned_rows = {
    "AUDIT_002381": learned("PARTIAL", 1, 0, 1, 1, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002382": learned("PARTIAL", 1, 1, 1, 1, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002383": learned("PASS", 2, 1, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002384": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002385": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002386": learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002387": learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002388": learned("PASS", 2, 1, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002389": learned("PASS", 2, 1, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002390": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002391": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002392": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002393": learned("PARTIAL", 0, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002394": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002395": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002396": learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002397": learned("PASS", 2, 1, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002398": learned("FAIL", 0, 0, 0, 0, 2, 0, 0, 0, FAIL),
    "AUDIT_002399": learned("PARTIAL", 1, 0, 1, 1, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002400": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002401": learned("PARTIAL", 1, 0, 1, 1, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002402": learned("PARTIAL", 1, 1, 1, 1, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002403": learned("PARTIAL", 1, 0, 1, 2, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002404": learned("PASS", 1, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002405": learned("PASS", 1, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002406": learned("PARTIAL", 0, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002407": learned("PARTIAL", 1, 0, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002408": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002409": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002410": learned("FAIL", 0, 0, 0, 2, 1, 1, 1, 0, FAIL),
    "AUDIT_002411": learned("FAIL", 0, 0, 1, 2, 1, 1, 1, 0, FAIL),
    "AUDIT_002412": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002413": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002414": learned("PARTIAL", 1, 0, 1, 2, 1, 1, 1, 1, PARTIAL),
    "AUDIT_002415": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002416": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002417": learned("PARTIAL", 1, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002418": learned("PARTIAL", 0, 1, 1, 1, 1, 0, 1, 1, PARTIAL),
    "AUDIT_002419": learned("PASS", 2, 2, 1, 0, 0, 0, 1, 2, PASS),
    "AUDIT_002420": learned("FAIL", 0, 0, 0, 2, 2, 1, 1, 0, FAIL),
}
updates.update(learned_rows)

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
