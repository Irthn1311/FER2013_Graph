import json
with open(r'D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_final\d10_final_stage2_finetune\training_history.json') as f:
    hist = json.load(f)
    print(f'Total epochs trained: {len(hist)}')
    if len(hist)>0:
        print(f'Final val_macro_f1: {hist[-1].get("val_macro_f1",0):.4f}')
