import json
import os
from pathlib import Path

dirs = [
    'outputs/d11/d11_exp_c_supcon_div_outputs',
    'outputs/d11/d11_exp_d_high_supcon_outputs',
    'outputs/d11/d11_exp_e_strong_local_outputs',
    'outputs/d11/d11_exp_f_high_diversity_outputs',
    'outputs/d11/d11_exp_g_no_global_outputs'
]

for d in dirs:
    print('='*60)
    print(d)
    history_path = Path(d) / 'training_history.json'
    metrics_path = Path(d) / 'evaluation' / 'metrics.json'
    
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
            last = history[-1]
            print(f"Epochs: {len(history)}")
            print(f"Train Acc: {last.get('train_accuracy', 0):.4f} | Val Acc: {last.get('val_accuracy', 0):.4f}")
            print(f"Train Local Acc: {last.get('train_diag_local_accuracy', 0):.4f} | Val Local Acc: {last.get('val_diag_local_accuracy', 0):.4f}")
            print(f"Val Macro F1: {last.get('val_macro_f1', 0):.4f}")
            print(f"Loss SupCon (val): {last.get('val_loss_supcon', 0):.4f} | Loss Div (val): {last.get('val_loss_div', 0):.4f}")
    else:
        print('No training history found.')
        
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"Test Acc: {metrics.get('accuracy', 0):.4f} | Test Macro F1: {metrics.get('macro_f1', 0):.4f}")
            cr = metrics.get('classification_report', {})
            for cls in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']:
                if cls in cr:
                    print(f"  {cls}: {cr[cls]['f1-score']:.4f}")
    else:
        print('No evaluation metrics found.')
