import glob
from pathlib import Path

stage2_configs = glob.glob("configs/experiments/d10_p5_stage2_*.yaml")
for path_str in stage2_configs:
    path = Path(path_str)
    content = path.read_text()
    
    if "early_stopping:" not in content:
        content += "\nearly_stopping:\n  monitor: val_macro_f1\n  mode: max\n"
        path.write_text(content)

print(f"Fixed {len(stage2_configs)} stage 2 configs to use val_macro_f1 for early stopping.")
