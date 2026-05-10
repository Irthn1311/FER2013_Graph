"""Update all Stage 2 variant configs for Run 3.
Key changes: freeze_encoder=false, lighter regularization.
"""
import glob
from pathlib import Path

stage2_variants = glob.glob("configs/experiments/d10_p5_stage2_relation_*.yaml")

anti_overfit_block = """model:
  freeze_classifier: false
  freeze_encoder: false
  dropout: 0.3
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
  label_smoothing: 0.05
optimizer:
  weight_decay: 0.001
training:
  epochs: 80
  early_stopping_patience: 25
  monitor: val_macro_f1
  val_frequency: 3
scheduler:
  name: cosine_warmup
  warmup_epochs: 5
  t_max: 80
  min_lr: 0.000001
checkpoint:
  save_best_metric: val_macro_f1
  save_best_mode: max
early_stopping:
  monitor: val_macro_f1
  mode: max
"""

# Focal variant gets focal_gamma added
focal_extra = "  focal_gamma: 1.5\n"

for path_str in sorted(stage2_variants):
    path = Path(path_str)
    content = path.read_text()
    lines = content.strip().split("\n")
    
    # Keep header (inherits, experiment, paths, environments blocks)
    header_lines = []
    for line in lines:
        header_lines.append(line)
        if "resolved_output_root: /kaggle/working" in line:
            break
    
    block = anti_overfit_block
    if "focal" in path.name:
        # Insert focal_gamma after label_smoothing line
        block = block.replace("  label_smoothing: 0.05\n", "  label_smoothing: 0.05\n  focal_gamma: 1.5\n")
    
    new_content = "\n".join(header_lines) + "\n" + block
    path.write_text(new_content)
    print(f"Updated: {path.name}")

print(f"\nTotal: {len(stage2_variants)} configs updated")
