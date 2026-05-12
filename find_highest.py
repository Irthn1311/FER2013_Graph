import os
import json
from pathlib import Path

root_dir = Path(r'D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs')
for metrics_path in root_dir.glob('*/evaluation/metrics.json'):
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            acc = data.get('accuracy', 0)
            f1 = data.get('macro_f1', 0)
            print(f"{metrics_path.parent.parent.name}: Acc={acc:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
