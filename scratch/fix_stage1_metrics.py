import glob
from pathlib import Path
import re

stage1_configs = glob.glob("configs/experiments/d10_p5_stage1_*.yaml")
for path_str in stage1_configs:
    path = Path(path_str)
    content = path.read_text()
    
    # We want to ensure `training: monitor: val_loss` is set
    if "training:" not in content:
        content += "\ntraining:\n  monitor: val_loss\n"
    elif "monitor:" not in content.split("training:")[1]:
        # Inject monitor: val_loss under training section
        content = re.sub(r'(training:\n)', r'\1  monitor: val_loss\n', content)
        
    path.write_text(content)

print(f"Fixed {len(stage1_configs)} stage 1 configs to use val_loss for training monitor.")
