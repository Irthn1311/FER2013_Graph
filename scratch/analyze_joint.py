import json

path = r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_p5_joint_outputs\training_history.json"
with open(path) as f:
    data = json.load(f)

best = max(data, key=lambda x: x.get("val_macro_f1", 0))
last = data[-1]

print(f"Total epochs: {len(data)}")
print(f"Last epoch: {int(last['epoch'])}")
print()
print("=== BEST EPOCH ===")
print(f"Epoch: {int(best['epoch'])}")
print(f"val_macro_f1: {best['val_macro_f1']:.4f}")
print(f"val_accuracy: {best.get('val_accuracy', 0):.4f}")
print(f"val_loss: {best['val_loss']:.4f}")
print(f"train_macro_f1: {best.get('train_macro_f1', 0):.4f}")
print(f"train_loss: {best['train_loss']:.4f}")
print()
print("=== TRAJECTORY ===")
for ep in [1, 3, 6, 9, 12, 15, 20, 30, 40, 50, 60, 68]:
    matches = [d for d in data if int(d["epoch"]) == ep]
    if matches:
        d = matches[0]
        vf1 = d["val_macro_f1"]
        vl = d["val_loss"]
        tf1 = d.get("train_macro_f1", 0)
        tl = d["train_loss"]
        print(f"Epoch {int(d['epoch']):3d}: val_f1={vf1:.4f}  val_loss={vl:.4f}  train_f1={tf1:.4f}  train_loss={tl:.4f}")

print()
print("=== PER-CLASS VAL F1 (Best Epoch) ===")
for cls in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]:
    key = f"val_f1_{cls}"
    if key in best:
        print(f"{cls:12s}: {best[key]:.4f}")

print()
print("=== OVERFITTING CHECK ===")
print(f"Train F1: {best.get('train_macro_f1', 0):.4f}")
print(f"Val F1:   {best['val_macro_f1']:.4f}")
gap = best.get("train_macro_f1", 0) - best["val_macro_f1"]
print(f"Gap:      {gap:.4f}")
