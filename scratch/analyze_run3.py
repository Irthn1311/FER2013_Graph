import json
from pathlib import Path

base = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_run3")

for name in ["d10_p5_stage2_relation", "d10_p5_stage2_relation_minreg"]:
    d = base / name
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")
    
    hist = d / "training_history.json"
    if hist.exists():
        with open(hist) as f:
            data = json.load(f)
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

# Also load Run 1 best for comparison
print(f"\n{'='*60}")
print("=== COMPARISON: Run 1 Best (reference) ===")
print(f"{'='*60}")
r1 = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_standard\d10_p5_stage2_relation_outputs_lan2")
r1_metrics = r1 / "evaluation" / "metrics.json"
if r1_metrics.exists():
    with open(r1_metrics) as f:
        m = json.load(f)
    print(f"TEST Macro F1: {m['macro_f1']:.4f}")
    print(f"TEST Accuracy: {m['accuracy']*100:.2f}%")
r1_hist = r1 / "training_history.json"
if r1_hist.exists():
    with open(r1_hist) as f:
        data = json.load(f)
    best = max(data, key=lambda x: x.get("val_macro_f1", 0))
    print(f"Best val_macro_f1: {best['val_macro_f1']:.4f} (epoch {int(best['epoch'])})")
    print(f"Train F1 at best: {best.get('train_macro_f1', 0):.4f}")
    print(f"Gap: {best.get('train_macro_f1', 0) - best['val_macro_f1']:.4f}")
