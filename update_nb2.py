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
        new_source.append('    "d11_exp_d_high_supcon":   "configs/experiments/d11_exp_d_high_supcon.yaml",\n')
        new_source.append('    "d11_exp_e_strong_local":  "configs/experiments/d11_exp_e_strong_local.yaml",\n')
        new_source.append('    "d11_exp_f_high_diversity":"configs/experiments/d11_exp_f_high_diversity.yaml",\n')
        new_source.append('    "d11_exp_g_no_global":     "configs/experiments/d11_exp_g_no_global.yaml"\n')
        new_source.append('}\n')
        skip = 3
    elif 'EXPERIMENT_ORDER =' in line:
        new_source.append('EXPERIMENT_ORDER = ["d11_exp_d_high_supcon", "d11_exp_e_strong_local", "d11_exp_f_high_diversity", "d11_exp_g_no_global"]\n')
    elif 'EXPERIMENTS_TO_RUN =' in line:
        new_source.append('EXPERIMENTS_TO_RUN = "all"\n')
    elif 'if EXPERIMENTS_TO_RUN in' in line:
        skip = 8 # Skip the old if blocks
        new_source.append('if EXPERIMENTS_TO_RUN == "all":\n')
        new_source.append('    EXPERIMENT_ORDER = list(EXPERIMENT_CONFIGS.keys())\n')
        new_source.append('else:\n')
        new_source.append('    EXPERIMENT_ORDER = [EXPERIMENTS_TO_RUN]\n')
    else:
        new_source.append(line)

nb['cells'][2]['source'] = new_source
with open('notebooks/kaggle-end-to-end.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
