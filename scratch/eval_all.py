import os
import subprocess
import json
from pathlib import Path

checkpoints = [
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_3_gnn1_iter5\20260508_112819\checkpoints\best.pth",
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_5_gnn1_dim128\20260508_113246\checkpoints\best.pth",
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_1_iter5\20260508_112450\checkpoints\best.pth",
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_2_iter5_lr3e4\20260508_112654\checkpoints\best.pth",
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_4_iter7_lr3e4\20260508_113717\checkpoints\best.pth"
]

results = {}

for ckpt in checkpoints:
    print("\n" + "="*60)
    print(f"Evaluating {ckpt}")
    
    # Extract config name from path (e.g. d10_v2_3_gnn1_iter5)
    parts = ckpt.split("\\")
    config_name = parts[-4]
    output_root = str(Path(ckpt).parent.parent)
    
    cmd = [
        "conda", "run", "-n", "fer-graph", "python", "-m", "scripts.evaluate_d5a",
        "--config", f"configs/experiments/{config_name}.yaml",
        "--environment", "local",
        "--checkpoint", ckpt,
        "--output_root", output_root,
        "--device", "cuda",
        "--no_wandb"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Read metrics.json
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
print("FINAL SUMMARY:")
for name, m in results.items():
    print(f"{name:30s} | F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']*100:.2f}% | pred: {m['pred_count']}")
