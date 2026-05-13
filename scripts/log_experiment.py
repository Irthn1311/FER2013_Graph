import json
import csv
import yaml
from pathlib import Path
from datetime import datetime

def log_experiment(output_dir: Path, csv_path: Path = Path('reports/d11_experiments_log.csv'), notes: str = ""):
    output_dir = Path(output_dir)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'Timestamp',
        'Experiment',
        'Epochs',
        'Use Global',
        'Lambda Local',
        'Lambda Spatial',
        'Lambda SupCon',
        'Lambda Div',
        'Div Margin',
        'Val Acc',
        'Val Macro F1',
        'Val Local Acc',
        'Test Acc',
        'Test Macro F1',
        'Notes'
    ]

    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    try:
        with open(output_dir / 'resolved_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: resolved_config.yaml not found in {output_dir}")
        return

    try:
        with open(output_dir / 'training_history.json', 'r', encoding='utf-8') as f:
            history = json.load(f)
            if not history: return
            last_epoch = history[-1]
    except FileNotFoundError:
        print(f"Warning: training_history.json not found in {output_dir}")
        return
        
    try:
        with open(output_dir / 'evaluation' / 'metrics.json', 'r', encoding='utf-8') as f:
            test_metrics = json.load(f)
    except FileNotFoundError:
        test_metrics = {}

    exp_name = config.get('run', {}).get('config_name', output_dir.name)
    epochs = len(history)
    use_global = config.get('model', {}).get('use_global_branch', False)
    
    loss_cfg = config.get('loss', {})
    lambda_local = loss_cfg.get('lambda_local', 0.0)
    lambda_spatial = loss_cfg.get('lambda_spatial', 0.0)
    lambda_supcon = loss_cfg.get('lambda_supcon', 0.0)
    lambda_div = loss_cfg.get('lambda_div', 0.0)
    div_margin = loss_cfg.get('diversity_margin', 'N/A')
    
    val_acc = last_epoch.get('val_accuracy', 0.0) * 100
    val_f1 = last_epoch.get('val_macro_f1', 0.0)
    val_local_acc = last_epoch.get('val_diag_local_accuracy', 0.0) * 100
    
    test_acc = test_metrics.get('accuracy', 0.0) * 100 if test_metrics else 0.0
    test_f1 = test_metrics.get('macro_f1', 0.0) if test_metrics else 0.0
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    row = [
        timestamp,
        exp_name,
        epochs,
        str(use_global),
        lambda_local,
        lambda_spatial,
        lambda_supcon,
        lambda_div,
        div_margin,
        f"{val_acc:.2f}%",
        f"{val_f1:.4f}",
        f"{val_local_acc:.2f}%",
        f"{test_acc:.2f}%" if test_metrics else "N/A",
        f"{test_f1:.4f}" if test_metrics else "N/A",
        notes
    ]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Logged {exp_name} to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default='reports/d11_experiments_log.csv')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()
    log_experiment(Path(args.output_dir), Path(args.csv_path), args.notes)
