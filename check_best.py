import json
with open(r'D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_standard\d10_p5_stage2_relation_outputs_lan2\training_history.json') as f:
    hist = json.load(f)
    best = max(hist, key=lambda x: x.get('val_macro_f1', 0))
    print(f"Best Epoch: {best['epoch']}, Best F1: {best['val_macro_f1']:.4f}")
