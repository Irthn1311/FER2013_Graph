import csv
from pathlib import Path


CSV_PATH = Path("outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")


def record(status, eye, mouth, face, hair, border, long_contour, fragment, region, facial, note):
    return {
        "selected_eye_eyebrow": str(eye),
        "selected_mouth_nasolabial": str(mouth),
        "selected_face_muscle_cheek_wrinkle": str(face),
        "selected_hair_glasses": str(hair),
        "selected_border_background": str(border),
        "long_contour_dominant": str(long_contour),
        "center_collapse": "0",
        "fragmented_pixel_dust": str(fragment),
        "region_like": str(region),
        "facial_evidence_like": str(facial),
        "overall_visual_pass": status,
        "notes": note,
        "confidence": "0.70",
    }


PASS = "Vision batch b2223: selected region overlaps clear facial evidence."
PARTIAL = "Vision batch b2223: facial evidence present but mixed with hair/contour/background."
FAIL = "Vision batch b2223: mostly hair/contour/background or weak facial evidence."


AUDIT = {
    # scratch_summary_b2223_0.png
    "AUDIT_001401": record("PARTIAL", 1, 0, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001402": record("PARTIAL", 0, 1, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001403": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001404": record("PARTIAL", 1, 1, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001405": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001406": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001407": record("FAIL", 0, 0, 1, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001408": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001409": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001410": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    # scratch_summary_b2223_1.png
    "AUDIT_001411": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001412": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001413": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001414": record("FAIL", 0, 0, 1, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001415": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001416": record("PASS", 1, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001417": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001418": record("PASS", 1, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001419": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001420": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    # scratch_summary_b2223_2.png
    "AUDIT_001421": record("PARTIAL", 0, 1, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001422": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001423": record("PARTIAL", 0, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001424": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001425": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001426": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001427": record("PARTIAL", 1, 1, 0, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001428": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001429": record("PARTIAL", 0, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001430": record("PARTIAL", 0, 1, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_3.png
    "AUDIT_001431": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001432": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001433": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001434": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001435": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001436": record("FAIL", 0, 0, 1, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001437": record("PASS", 1, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001438": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001439": record("PASS", 1, 1, 2, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001440": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_4.png
    "AUDIT_001441": record("PARTIAL", 1, 0, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001442": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001443": record("PARTIAL", 0, 0, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001444": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001445": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001446": record("PASS", 2, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001447": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001448": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001449": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001450": record("PARTIAL", 0, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_5.png
    "AUDIT_001451": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001452": record("PASS", 2, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001453": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001454": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001455": record("PASS", 1, 2, 2, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001456": record("PARTIAL", 1, 0, 1, 2, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001457": record("PARTIAL", 0, 1, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001458": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001459": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001460": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_6.png
    "AUDIT_001461": record("PARTIAL", 1, 0, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001462": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001463": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001464": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001465": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001466": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001467": record("PARTIAL", 1, 0, 1, 1, 1, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001468": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    "AUDIT_001469": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001470": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_7.png
    "AUDIT_001471": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001472": record("PASS", 1, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001473": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001474": record("PARTIAL", 0, 1, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001475": record("PARTIAL", 0, 1, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001476": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001477": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001478": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001479": record("PARTIAL", 0, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001480": record("FAIL", 0, 0, 0, 1, 2, 1, 0, 1, 0, FAIL),
    # scratch_summary_b2223_8.png
    "AUDIT_001481": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001482": record("PASS", 1, 2, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001483": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001484": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001485": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001486": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001487": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001488": record("PARTIAL", 1, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001489": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001490": record("PARTIAL", 1, 0, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    # scratch_summary_b2223_9.png
    "AUDIT_001491": record("PARTIAL", 0, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
    "AUDIT_001492": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001493": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001494": record("FAIL", 0, 0, 0, 2, 1, 1, 0, 1, 0, FAIL),
    "AUDIT_001495": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001496": record("PARTIAL", 1, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001497": record("PASS", 2, 1, 1, 0, 0, 0, 0, 2, 2, PASS),
    "AUDIT_001498": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001499": record("PARTIAL", 0, 1, 1, 1, 1, 0, 0, 2, 1, PARTIAL),
    "AUDIT_001500": record("PARTIAL", 0, 0, 1, 1, 2, 1, 0, 2, 1, PARTIAL),
}


with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

if not fieldnames or "audit_id" not in fieldnames:
    raise SystemExit("CSV does not contain audit_id after utf-8-sig decoding")

updated = 0
for row in rows:
    audit_id = row["audit_id"]
    if audit_id in AUDIT:
        row.update(AUDIT[audit_id])
        updated += 1

if updated != len(AUDIT):
    raise SystemExit(f"Expected to update {len(AUDIT)} rows, updated {updated}")

with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"updated={updated}")
