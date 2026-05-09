import os
from pathlib import Path

config_dir = Path("configs/experiments")

configs = {
    "d10_p5_stage1_supcon_sharp.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_sharp
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_sharp
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_sharp
loss:
  supcon_temperature: 0.05
""",
    "d10_p5_stage2_relation_sharp.yaml": """inherits:
  - d10_p5_stage1_supcon_sharp.yaml
experiment:
  name: d10_p5_stage2_relation_sharp
paths:
  resolved_output_root: output/d10_p5_stage2_relation_sharp
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_sharp
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
""",
    "d10_p5_stage1_supcon_smooth.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_smooth
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_smooth
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_smooth
loss:
  supcon_temperature: 0.2
""",
    "d10_p5_stage2_relation_smooth.yaml": """inherits:
  - d10_p5_stage1_supcon_smooth.yaml
experiment:
  name: d10_p5_stage2_relation_smooth
paths:
  resolved_output_root: output/d10_p5_stage2_relation_smooth
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_smooth
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
""",
    "d10_p5_stage1_supcon_high_lr.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_high_lr
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_high_lr
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_high_lr
optimizer:
  lr: 0.0005
""",
    "d10_p5_stage2_relation_high_lr.yaml": """inherits:
  - d10_p5_stage1_supcon_high_lr.yaml
experiment:
  name: d10_p5_stage2_relation_high_lr
paths:
  resolved_output_root: output/d10_p5_stage2_relation_high_lr
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_high_lr
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
""",
    "d10_p5_stage1_supcon_fast.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_fast
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_fast
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_fast
model:
  hidden_dim: 64
  pixel_gnn_layers: 1
  multi_scale_gnn: false
""",
    "d10_p5_stage2_relation_fast.yaml": """inherits:
  - d10_p5_stage1_supcon_fast.yaml
experiment:
  name: d10_p5_stage2_relation_fast
paths:
  resolved_output_root: output/d10_p5_stage2_relation_fast
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_fast
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
"""
}

for name, content in configs.items():
    p = config_dir / name
    p.write_text(content, encoding='utf-8')
    print(f"Created {name}")
