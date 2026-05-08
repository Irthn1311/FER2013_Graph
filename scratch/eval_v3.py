import os
import subprocess
import json
from pathlib import Path

outputs_dir = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs")
checkpoints = []

# Find all v3 checkpoints
for d in outputs_dir.glob("d10_v3*"):
    if d.is_dir():
        # Find the timestamp folder
        for ts_dir in d.iterdir():
            if ts_dir.is_dir() and (ts_dir / "checkpoints" / "best.pth").exists():
                checkpoints.append(str(ts_dir / "checkpoints" / "best.pth"))

results = {}

for ckpt in checkpoints:
    print("\n" + "="*60)
    print(f"Evaluating {ckpt}")
    
    parts = ckpt.split("\\")
    config_name = parts[-4]
    output_root = str(Path(ckpt).parent.parent)
    
    cmd = [
        "python", "-m", "scripts.evaluate_d5a",
        "--config", f"configs/experiments/{config_name}.yaml",
        "--environment", "local",
        "--checkpoint", ckpt,
        "--output_root", output_root,
        "--device", "cuda:0",
        "--no_wandb"
    ]
    
    subprocess.run(cmd, check=True)
    
    metrics_path = Path(output_root) / "evaluation" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            results[config_name] = {
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "pred_count": m.get("pred_count")
            }
            print(f"--> {config_name}: Acc={m.get('accuracy')*100:.2f}%, F1={m.get('macro_f1'):.4f}")

print("\n" + "="*60)
print("FINAL V3 SUMMARY:")
# Sort by F1 descending
sorted_results = sorted(results.items(), key=lambda item: item[1]['macro_f1'], reverse=True)
for name, m in sorted_results:
    print(f"{name:30s} | F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']*100:.2f}% | pred: {m['pred_count']}")
