import csv
from pathlib import Path

log_dir = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_v2_2_iter5_lr3e4\20260508_112654")
csvs = list(log_dir.glob("logs/*.csv"))
if not csvs:
    print("No CSV found")
    exit()

rows = list(csv.DictReader(csvs[0].open("r", encoding="utf-8")))
print("Total epochs logged:", len(rows))
print("CSV columns:", list(rows[0].keys())[:15])
print()

# Show last 15 epochs
print("Last 15 epochs:")
for r in rows[-15:]:
    ep = r.get("epoch", "?")
    val_f1 = r.get("val_macro_f1", "")
    val_acc = r.get("val_accuracy", "")
    train_loss = r.get("loss", r.get("train_loss", ""))
    val_loss = r.get("val_loss", "")
    lr = r.get("lr", r.get("learning_rate", ""))
    print(f"  ep={ep:>3s}  val_f1={val_f1:>8s}  val_acc={val_acc:>8s}  loss={train_loss:>8s}  val_loss={val_loss:>8s}  lr={lr}")

# Find best epoch
best = max(rows, key=lambda r: float(r.get("val_macro_f1", 0)))
print()
print("Best epoch:", best.get("epoch"), " val_f1=", best.get("val_macro_f1"), " val_acc=", best.get("val_accuracy"))

# Check if val_f1 was still improving in last 10 epochs
last10 = [float(r.get("val_macro_f1", 0)) for r in rows[-10:]]
print("val_f1 last 10 epochs:", [round(x, 4) for x in last10])
print("Trend: max=%.4f  min=%.4f  last=%.4f" % (max(last10), min(last10), last10[-1]))

# Check LR schedule
lrs = []
for r in rows:
    lr_val = r.get("lr", r.get("learning_rate", ""))
    if lr_val:
        lrs.append((r.get("epoch"), lr_val))
if lrs:
    print()
    print("LR schedule (sampled):")
    for ep, lr in lrs[::10]:
        print(f"  ep={ep}  lr={lr}")
    print(f"  ep={lrs[-1][0]}  lr={lrs[-1][1]}")
