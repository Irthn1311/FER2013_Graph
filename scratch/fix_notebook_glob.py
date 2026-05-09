import json
from pathlib import Path

nb_path = Path("notebooks/kaggle-end-to-end.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "# Cell 4" in "".join(cell["source"]):
        source = cell["source"]
        for i, line in enumerate(source):
            if "candidates = list(OUTPUT_BASE.glob(\"d10_p5_stage1_supcon*/checkpoints/best.pth\"))" in line:
                source[i] = "    candidates = list(OUTPUT_BASE.glob(\"d10_p5_stage1_supcon*/**/checkpoints/best.pth\"))\n"
            elif "last_candidates = list(OUTPUT_BASE.glob(\"d10_p5_stage1_supcon*/checkpoints/last.pth\"))" in line:
                source[i] = "        last_candidates = list(OUTPUT_BASE.glob(\"d10_p5_stage1_supcon*/**/checkpoints/last.pth\"))\n"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Fixed notebook glob patterns.")
