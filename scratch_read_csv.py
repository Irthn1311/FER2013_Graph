import csv
import sys

with open('d:/SGU/CNTT/DIP/FER_2013_GRAPH/fer_d5/outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

unreviewed = [r for r in rows if r['overall_visual_pass'] == 'UNREVIEWED']
print(f'Total unreviewed: {len(unreviewed)}')
print('First 50 paths:')
for r in unreviewed[:50]:
    print(f"{r['audit_id']}|{r['figure_path']}")
