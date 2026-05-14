import json
with open('notebooks/kaggle-end-to-end.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_source = nb['cells'][2]['source']
new_source = []
skip = 0
for line in cell_source:
    if skip > 0:
        skip -= 1
        continue
    if 'EXPERIMENT_CONFIGS =' in line:
        new_source.append('EXPERIMENT_CONFIGS = {\n')
        new_source.append('    "d11_exp_h_pseudo_2stage": "configs/experiments/d11_exp_h_pseudo_2stage.yaml",\n')
        new_source.append('    "d11_exp_i_supcon_decay":  "configs/experiments/d11_exp_i_supcon_decay.yaml",\n')
        new_source.append('    "d11_exp_j_high_cap":      "configs/experiments/d11_exp_j_high_cap.yaml",\n')
        new_source.append('    "d11_exp_k_more_slots":    "configs/experiments/d11_exp_k_more_slots.yaml",\n')
        new_source.append('    "d11_exp_l_d10_repro":     "configs/experiments/d11_exp_l_d10_repro.yaml"\n')
        new_source.append('}\n')
        skip = 5 # skip the 5 lines of previous configs
    elif 'EXPERIMENT_ORDER =' in line:
        new_source.append('EXPERIMENT_ORDER = ["d11_exp_h_pseudo_2stage", "d11_exp_i_supcon_decay", "d11_exp_j_high_cap", "d11_exp_k_more_slots", "d11_exp_l_d10_repro"]\n')
    elif 'EXPERIMENT_LABELS =' in line:
        new_source.append('EXPERIMENT_LABELS = {\n')
        new_source.append('    "d11_exp_h_pseudo_2stage": "Exp H: Pseudo-2-Stage (Soft CE Warmup)",\n')
        new_source.append('    "d11_exp_i_supcon_decay":  "Exp I: SupCon Decay",\n')
        new_source.append('    "d11_exp_j_high_cap":      "Exp J: High-Cap Bottleneck 128",\n')
        new_source.append('    "d11_exp_k_more_slots":    "Exp K: 16 Slots",\n')
        new_source.append('    "d11_exp_l_d10_repro":     "Exp L: D10 Reproduction (Hard CE Delay)",\n')
        new_source.append('}\n')
        skip = 5 # skip previous labels
    elif 'EXPERIMENTS_TO_RUN =' in line:
        new_source.append('EXPERIMENTS_TO_RUN = "d11_exp_h_pseudo_2stage"\n')
    else:
        new_source.append(line)

nb['cells'][2]['source'] = new_source
with open('notebooks/kaggle-end-to-end.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
