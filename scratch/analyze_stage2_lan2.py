import json

path = r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_standard\d10_p5_stage2_relation_outputs_lan2\training_history.json"
with open(path, "r") as f:
    data = json.load(f)

best = max(data, key=lambda x: x.get("val_macro_f1", 0))
last = data[-1]

print(f"Total epochs trained: {len(data)}")
print(f"Last epoch: {last['epoch']}")
print()
print("=== BEST EPOCH ===")
print(f"Epoch: {best['epoch']}")
print(f"val_macro_f1: {best['val_macro_f1']:.4f}")
print(f"val_accuracy: {best['val_accuracy']:.4f}")
print(f"val_loss: {best['val_loss']:.4f}")
print(f"train_loss: {best['train_loss']:.4f}")
print(f"train_macro_f1: {best['train_macro_f1']:.4f}")
print()
print("=== LAST EPOCH ===")
print(f"Epoch: {last['epoch']}")
print(f"val_macro_f1: {last['val_macro_f1']:.4f}")
print(f"val_loss: {last['val_loss']:.4f}")
print()
print("=== TRAJECTORY ===")
milestones = [1, 3, 6, 9, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
for ep_target in milestones:
    matches = [d for d in data if int(d["epoch"]) == ep_target]
    if matches:
        d = matches[0]
        print(f"Epoch {int(d['epoch']):3d}: val_f1={d['val_macro_f1']:.4f}  val_loss={d['val_loss']:.4f}  train_loss={d['train_loss']:.4f}")
