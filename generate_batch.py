import csv
import os
from PIL import Image, ImageDraw

with open('outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv', 'r', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

unreviewed = [r for r in rows if r['overall_visual_pass'] == 'UNREVIEWED']
batch = unreviewed[:100]

print(f"Generating for IDs: {batch[0]['audit_id']} to {batch[-1]['audit_id']}")

for i in range(10):
    images_to_stack = []
    for j in range(10):
        idx = i*10 + j
        if idx >= len(batch): break
        r = batch[idx]
        path = r['figure_path']
        if os.path.exists(path):
            img = Image.open(path)
            # Add text with ID
            draw = ImageDraw.Draw(img)
            # try to use a default font, simple text
            draw.text((10, 10), f"{r['audit_id']}", fill="red")
            images_to_stack.append(img)
        else:
            print(f"Missing: {path}")
            # create empty image
            img = Image.new('RGB', (800, 200), color='black')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"MISSING: {r['audit_id']}", fill="red")
            images_to_stack.append(img)
    
    if not images_to_stack: continue
    
    widths, heights = zip(*(img.size for img in images_to_stack))
    total_height = sum(heights)
    max_width = max(widths)
    
    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images_to_stack:
        new_im.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    new_im.save(f'scratch_summary_b2223_{i}.png')

print("Done generating 10 summaries for batch 22-23")
