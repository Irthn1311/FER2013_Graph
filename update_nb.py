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
        new_source.append('    "d11_exp_b_supcon_only": "configs/experiments/d11_exp_b_supcon_only.yaml",\n')
        new_source.append('    "d11_exp_c_supcon_div":  "configs/experiments/d11_exp_c_supcon_div.yaml"\n')
        new_source.append('}\n')
        skip = 3
    elif 'EXPERIMENT_ORDER =' in line:
        new_source.append('EXPERIMENT_ORDER = ["d11_exp_b_supcon_only", "d11_exp_c_supcon_div"]\n')
    else:
        new_source.append(line)

nb['cells'][2]['source'] = new_source
with open('notebooks/kaggle-end-to-end.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
