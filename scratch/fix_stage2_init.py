import json
from pathlib import Path

# Fix the notebook to fallback to last.pth
nb_path = Path("notebooks/kaggle-end-to-end.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "# Cell 4" in "".join(cell["source"]):
        source = cell["source"]
        for i, line in enumerate(source):
            if "ckpt = stage1_result.get(\"best_checkpoint\")" in line:
                source.insert(i+1, "    if ckpt is None:\n        ckpt = stage1_result.get(\"last_checkpoint\")\n")
                break
        
        # update find_latest_stage1_best to also look for last.pth if best.pth not found
        for i, line in enumerate(source):
            if "if ckpt is None:" in line and "raise RuntimeError(" in "".join(source[i:i+3]):
                new_logic = [
                    "    if ckpt is None:\n",
                    "        last_candidates = list(OUTPUT_BASE.glob(\"d10_p5_stage1_supcon*/checkpoints/last.pth\"))\n",
                    "        last_candidates = [p for p in last_candidates if p.exists()]\n",
                    "        if last_candidates:\n",
                    "            last_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)\n",
                    "            ckpt = last_candidates[0]\n",
                    "    if ckpt is None:\n"
                ]
                source[i] = new_logic[0]
                source.insert(i+1, new_logic[1])
                source.insert(i+2, new_logic[2])
                source.insert(i+3, new_logic[3])
                source.insert(i+4, new_logic[4])
                source.insert(i+5, new_logic[5])
                source.insert(i+6, new_logic[6])
                break
        
        # Also fix epochs in smoke_test so it runs validation!
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "# Cell 2" in "".join(cell["source"]):
        source = cell["source"]
        for i, line in enumerate(source):
            if "EPOCHS_OVERRIDE = 2" in line and "smoke_test" in "".join(source[max(0, i-5):i]):
                source[i] = "    EPOCHS_OVERRIDE = 4\n"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

# Now fix the yaml configs to use val_loss for stage 1
import glob

stage1_configs = glob.glob("configs/experiments/d10_p5_stage1_*.yaml")
for path in stage1_configs:
    content = Path(path).read_text()
    if "checkpoint:" not in content:
        content += "\ncheckpoint:\n  save_best_metric: val_loss\n  save_best_mode: min\n"
        Path(path).write_text(content)

print("Fixed notebook fallback and YAML metrics.")
