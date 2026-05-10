import json
from pathlib import Path

base_dir = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_run4")
runs = ["d10_p5_stage2_run4_moderate", "d10_p5_stage2_run4_strong"]

for r in runs:
    print(f"=== {r} ===")
    history_file = base_dir / r / "training_history.json"
    metrics_file = base_dir / r / "evaluation" / "metrics.json"
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            hist = json.load(f)
        if hist:
            best = max(hist, key=lambda x: x.get('val_macro_f1', 0))
            print(f"Best val_macro_f1: {best.get('val_macro_f1'):.4f} at epoch {best.get('epoch')}")
            
            # Print gap
            train_f1 = best.get('train_macro_f1', 0)
            val_f1 = best.get('val_macro_f1', 0)
            print(f"Train/Val Gap at best: {train_f1:.4f} vs {val_f1:.4f} (Gap: {train_f1 - val_f1:.4f})")
    else:
        print("No training_history.json found.")
        
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"Test Macro F1: {metrics.get('macro_f1'):.4f}")
        print(f"Test Accuracy: {metrics.get('accuracy') * 100:.2f}%")
    else:
        print("No metrics.json found (Not evaluated yet).")
    print()
