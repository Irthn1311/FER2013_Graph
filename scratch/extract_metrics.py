import json
import glob
from pathlib import Path

base_dir = Path(r'D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs')
p3_dirs = glob.glob(str(base_dir / 'd10_p3_*'))

results = []
for d in p3_dirs:
    d = Path(d)
    metrics_files = list(d.rglob('metrics.json'))
    if metrics_files:
        with open(metrics_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            acc = data.get('accuracy', 0)
            f1 = data.get('macro_f1', 0)
            results.append({'name': d.name, 'acc': acc, 'f1': f1})
    else:
        results.append({'name': d.name, 'acc': 0, 'f1': 0})

results.sort(key=lambda x: x['f1'], reverse=True)
print(f"| {'Config':<35} | {'Macro F1':<10} | {'Accuracy':<10} |")
print('|' + '-'*37 + '|' + '-'*12 + '|' + '-'*12 + '|')
for r in results:
    print(f"| {r['name']:<35} | {r['f1']:<10.4f} | {r['acc']:<10.4f} |")
