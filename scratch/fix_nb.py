import json
from pathlib import Path

nb_path = Path('notebooks/kaggle_d5_end_to_end.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = '''# Cell 5: Validate output artifacts + Evaluate on TEST set
import csv
import json
import numpy as np
import sys
import subprocess

print("OUTPUT_DIR:", OUTPUT_DIR, OUTPUT_DIR.exists())

run_dir = OUTPUT_DIR
resolved_path = run_dir / "resolved_config.yaml"

if not resolved_path.exists():
    found = list(OUTPUT_DIR.rglob("resolved_config.yaml"))
    if found:
        # Sort by timestamp (which is the parent folder)
        found.sort(key=lambda p: str(p.parent), reverse=True)
        run_dir = found[0].parent
        resolved_path = found[0]
        print("Found run dir:", run_dir)
    else:
        print("[WARN] Could not find resolved_config.yaml anywhere in", OUTPUT_DIR)

checkpoint_dir = run_dir / "checkpoints"
for name in ["best.pth", "last.pth"]:
    p = checkpoint_dir / name
    print(f"  {name}: {p.exists()}")

history_files = list(run_dir.glob("logs/*.csv"))
if history_files:
    rows = list(csv.DictReader(history_files[0].open("r", encoding="utf-8")))
    epochs_run = max(int(float(r["epoch"])) for r in rows) if rows else 0
    print(f"\\nepochs_run: {epochs_run}")
    best_row = max(rows, key=lambda r: float(r.get("val_macro_f1", 0)))
    print(f"best_epoch: {best_row.get('epoch')}")
    print(f"best_val_macro_f1: {best_row.get('val_macro_f1')}")
    print(f"best_val_accuracy: {best_row.get('val_accuracy')}")
    print(f"\\nLast 5 epochs:")
    for r in rows[-5:]:
        print(f"  ep={r.get('epoch', '?'):>3s} val_f1={r.get('val_macro_f1', '?'):>8s} val_acc={r.get('val_accuracy', '?'):>8s}")

print("\\n[OK] Validation complete")

# ========== TEST SET EVALUATION ==========
print("\\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

ckpt = checkpoint_dir / "best.pth"
if not ckpt.exists():
    ckpt = checkpoint_dir / "last.pth"

if not ckpt.exists():
    print("[SKIP] No checkpoint found, skipping evaluation")
else:
    print(f"Checkpoint: {ckpt}")
    eval_cmd = [
        sys.executable, "-m", "scripts.evaluate_d5a",
        "--config", CONFIG_PATH,
        "--environment", ENVIRONMENT,
        "--checkpoint", str(ckpt),
        "--graph_repo_path", str(GRAPH_REPO_PATH),
        "--output_root", str(run_dir),
        "--device", DEVICE,
        "--no_wandb",
        "--chunk_cache_size", str(CHUNK_CACHE_SIZE),
    ]
    if NUM_WORKERS is not None:
        eval_cmd += ["--num_workers", str(NUM_WORKERS)]
    print("$", " ".join(str(x) for x in eval_cmd))
    subprocess.run(eval_cmd, check=True)
    
    # Print evaluation results summary
    eval_metrics = run_dir / "evaluation" / "metrics.json"
    if eval_metrics.exists():
        m = json.loads(eval_metrics.read_text(encoding="utf-8"))
        print(f"\\n--> Accuracy: {m['accuracy']*100:.2f}%")
        print(f"--> Macro F1: {m['macro_f1']:.4f}")
        print(f"--> Weighted F1: {m['weighted_f1']:.4f}")
        print(f"--> pred_count: {m['pred_count']}")
    
    eval_report = run_dir / "evaluation" / "classification_report.txt"
    if eval_report.exists():
        print(f"\\n--> Report:")
        print(eval_report.read_text(encoding="utf-8"))
    
    print("\\n[OK] Test evaluation complete")
'''

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if isinstance(cell['source'], list) and len(cell['source']) > 0 and 'Cell 5: Validate' in cell['source'][0]:
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
        elif isinstance(cell['source'], str) and 'Cell 5: Validate' in cell['source']:
            cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f)
print('Notebook Cell 5 updated!')
