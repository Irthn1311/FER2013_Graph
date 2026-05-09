import os
from pathlib import Path

config_dir = Path("configs/experiments")

configs = {
    "d10_p5_stage1_supcon_motif6.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_motif6
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_motif6
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_motif6
model:
  num_motifs: 6
""",
    "d10_p5_stage2_relation_motif6.yaml": """inherits:
  - d10_p5_stage1_supcon_motif6.yaml
experiment:
  name: d10_p5_stage2_relation_motif6
paths:
  resolved_output_root: output/d10_p5_stage2_relation_motif6
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_motif6
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
""",
    "d10_p5_stage1_supcon_focal.yaml": """inherits:
  - d10_p5_stage1_supcon.yaml
experiment:
  name: d10_p5_stage1_supcon_focal
paths:
  resolved_output_root: output/d10_p5_stage1_supcon_focal
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage1_supcon_focal
""",
    "d10_p5_stage2_relation_focal.yaml": """inherits:
  - d10_p5_stage1_supcon_focal.yaml
experiment:
  name: d10_p5_stage2_relation_focal
paths:
  resolved_output_root: output/d10_p5_stage2_relation_focal
environments:
  kaggle:
    paths:
      resolved_output_root: /kaggle/working/outputs/d10_p5_stage2_relation_focal
model:
  freeze_classifier: false
  freeze_encoder: true
loss:
  lambda_cls: 1.0
  lambda_aux_ce: 0.3
  lambda_supcon: 0.0
  focal_gamma: 1.0
"""
}

for name, content in configs.items():
    p = config_dir / name
    p.write_text(content, encoding='utf-8')
    print(f"Created {name}")
