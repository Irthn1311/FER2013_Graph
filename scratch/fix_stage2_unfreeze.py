import glob
from pathlib import Path
import re

stage2_configs = glob.glob("configs/experiments/d10_p5_stage2_*.yaml")
count = 0
for path_str in stage2_configs:
    path = Path(path_str)
    content = path.read_text()
    
    if "freeze_encoder: true" in content:
        content = content.replace("freeze_encoder: true", "freeze_encoder: false")
        path.write_text(content)
        count += 1

print(f"Unfroze encoder in {count} stage 2 configs.")
