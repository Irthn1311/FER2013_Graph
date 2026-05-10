import json
from pathlib import Path

base = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_run3")

for name in ["d10_p5_joint_outputs", "d10_p5_joint_light_outputs"]:
    d = base / name
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")
    
    hist = d / "training_history.json"
    if hist.exists():
        with open(hist) as f:
            data = json.load(f)
        if data:
            best = max(data, key=lambda x: x.get("val_macro_f1", 0))
            last = data[-1]
            print(f"Epochs: {len(data)}, Last: {int(last['epoch'])}")
            print(f"Best Epoch: {int(best['epoch'])}")
            print(f"Best val_macro_f1: {best['val_macro_f1']:.4f}")
            print(f"Best val_loss: {best['val_loss']:.4f}")
            print(f"Best train_f1: {best.get('train_macro_f1', 0):.4f}")
            print(f"Gap: {best.get('train_macro_f1', 0) - best['val_macro_f1']:.4f}")
            print()
            print("Trajectory:")
            for ep in [1,3,6,9,12,15,20,30,40,50,60,70,80]:
                matches = [x for x in data if int(x["epoch"]) == ep]
                if matches:
                    x = matches[0]
                    print(f"  Ep {int(x['epoch']):3d}: val_f1={x['val_macro_f1']:.4f}  train_f1={x.get('train_macro_f1',0):.4f}  val_loss={x['val_loss']:.4f}")
    
    eval_path = d / "evaluation" / "classification_report.txt"
    if eval_path.exists():
        print(f"\nTest Report:")
        print(eval_path.read_text(encoding="utf-8"))
    
    metrics_path = d / "evaluation" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"TEST Macro F1: {m['macro_f1']:.4f}")
        print(f"TEST Accuracy: {m['accuracy']*100:.2f}%")
