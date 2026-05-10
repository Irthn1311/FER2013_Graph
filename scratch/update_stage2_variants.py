"""Update all Stage 2 variant configs with anti-overfitting settings.
These settings layer ON TOP of what the base d10_p5_stage2_relation.yaml provides via inheritance.
We need to explicitly set them because the variants inherit from their Stage 1 counterpart, not from the Stage 2 base.
"""
import glob
from pathlib import Path

stage2_variants = glob.glob("configs/experiments/d10_p5_stage2_relation_*.yaml")
# Skip focal (already handled manually)
stage2_variants = [p for p in stage2_variants if "focal" not in p]

anti_overfit_block = """model:
  freeze_classifier: false
  freeze_encoder: true
  dropout: 0.4
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
  label_smoothing: 0.1
  focal_gamma: 2.0
  class_weight_power: 0.5
optimizer:
  weight_decay: 0.01
training:
  epochs: 60
  early_stopping_patience: 20
  monitor: val_macro_f1
  val_frequency: 3
scheduler:
  name: cosine_warmup
  warmup_epochs: 5
  t_max: 60
  min_lr: 0.000001
checkpoint:
  save_best_metric: val_macro_f1
  save_best_mode: max
early_stopping:
  monitor: val_macro_f1
  mode: max
"""

for path_str in sorted(stage2_variants):
    path = Path(path_str)
    content = path.read_text()
    lines = content.strip().split("\n")
    
    # Keep header (inherits, experiment, paths, environments blocks)
    header_lines = []
    for line in lines:
        header_lines.append(line)
        # Stop after the kaggle output root line
        if "resolved_output_root: /kaggle/working" in line:
            break
    
    new_content = "\n".join(header_lines) + "\n" + anti_overfit_block
    path.write_text(new_content)
    print(f"Updated: {path.name}")

print(f"\nTotal updated: {len(stage2_variants)} variant configs")
