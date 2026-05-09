import glob
from pathlib import Path

stage2_configs = glob.glob("configs/experiments/d10_p5_stage2_*.yaml")
for path in stage2_configs:
    content = Path(path).read_text()
    if "checkpoint:" not in content:
        content += "\ncheckpoint:\n  save_best_metric: val_macro_f1\n  save_best_mode: max\n\ntraining:\n  monitor: val_macro_f1\n"
        Path(path).write_text(content)

print(f"Fixed {len(stage2_configs)} stage 2 configs to use val_macro_f1.")
