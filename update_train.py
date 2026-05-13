with open('scripts/train_d5a.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'from scripts.log_experiment import log_experiment' not in content:
    content = content.replace('import argparse\nimport shutil', 'import argparse\nimport shutil\nfrom scripts.log_experiment import log_experiment')

hook = '''        max_val_batches=training_cfg.get("max_val_batches"),
    )
    print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
    return result'''
new_hook = '''        max_val_batches=training_cfg.get("max_val_batches"),
    )
    print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
    try:
        log_experiment(trainer.output_dir)
    except Exception as e:
        print(f"Failed to log experiment: {e}")
    return result'''

if hook in content:
    content = content.replace(hook, new_hook)

with open('scripts/train_d5a.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated train_d5a.py')
