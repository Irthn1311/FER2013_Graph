import json

with open('notebooks/kaggle-end-to-end.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 2 (Configuration)
nb['cells'][2]['source'] = '''# Cell 2: Configuration - 1-Stage D11 Experiments
from pathlib import Path
import os
import sys
import time
import json
import shutil
import subprocess

import numpy as np
import yaml

# ============================================================
# RUN MODE: "train_full", "train_quick", "smoke_test"
# ============================================================
RUN_MODE = "train_full"

# ============================================================
# EXPERIMENTS
# EXPERIMENTS_TO_RUN: "facs_only", "full_d11a", "both"
# ============================================================
EXPERIMENTS_TO_RUN = "both"

FACS_ONLY_CONFIG = "configs/experiments/d11_facs_only_1stage.yaml"
FULL_D11A_CONFIG = "configs/experiments/d11_full_1stage.yaml"

ENVIRONMENT = "kaggle"
DEVICE = "cuda:0"
FORCE_OVERWRITE = False
ZIP_OUTPUTS = True
RUN_TEST_EVALUATION = True

REPO_ROOT = Path.cwd()

GRAPH_REPO_INPUT = Path("/kaggle/input/datasets/irthn1311/graph-repo/graph_repo")
GRAPH_REPO_WORKING = Path("/kaggle/working/graph_repo")
GRAPH_REPO_PATH = GRAPH_REPO_WORKING
OUTPUT_BASE = Path("/kaggle/working/outputs")

if RUN_MODE == "train_full":
    EPOCHS_OVERRIDE = None
    MAX_TRAIN_BATCHES = None
    MAX_VAL_BATCHES = None
    PROFILE_BATCHES = 3
elif RUN_MODE == "train_quick":
    EPOCHS_OVERRIDE = 12
    MAX_TRAIN_BATCHES = 300
    MAX_VAL_BATCHES = 80
    PROFILE_BATCHES = 3
elif RUN_MODE == "smoke_test":
    EPOCHS_OVERRIDE = 4
    MAX_TRAIN_BATCHES = 8
    MAX_VAL_BATCHES = 4
    PROFILE_BATCHES = 0
else:
    raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}")

BATCH_SIZE = None
NUM_WORKERS = 2
CHUNK_CACHE_SIZE = 8

EXPERIMENT_ORDER = []
EXPERIMENT_CONFIGS = {}

if EXPERIMENTS_TO_RUN in ["facs_only", "both"]:
    EXPERIMENT_ORDER.append("facs_only")
    EXPERIMENT_CONFIGS["facs_only"] = FACS_ONLY_CONFIG

if EXPERIMENTS_TO_RUN in ["full_d11a", "both"]:
    EXPERIMENT_ORDER.append("full_d11a")
    EXPERIMENT_CONFIGS["full_d11a"] = FULL_D11A_CONFIG

EXPERIMENT_LABELS = {
    "facs_only": f"Exp 1: D10 + FACS ({Path(FACS_ONLY_CONFIG).stem})",
    "full_d11a": f"Exp 2: D11A Simple ({Path(FULL_D11A_CONFIG).stem})",
}

def run_cmd(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or Path.cwd()), check=True)

def copy_dir_if_needed(src: Path, dst: Path, force: bool = False):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise RuntimeError(f"Missing source folder: {src}")
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[COPY] Reusing existing folder: {dst}")
            return dst
    print(f"[COPY] {src} -> {dst}")
    shutil.copytree(src, dst)
    return dst

def zip_dir(src_dir: Path, zip_path: Path):
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(src_dir))
    print("ZIP:", zip_path)
    return zip_path

def ensure_config_exists(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise RuntimeError(f"Missing config: {path}")
    return path

print("="*60)
print("D11 1-STAGE EXPERIMENTS")
print("="*60)
print(f"RUN_MODE:           {RUN_MODE}")
print(f"EXPERIMENTS_TO_RUN: {EXPERIMENTS_TO_RUN}")
for exp in EXPERIMENT_ORDER:
    print(f"  {exp}: {EXPERIMENT_CONFIGS[exp]}")
print(f"EPOCHS_OVERRIDE:    {EPOCHS_OVERRIDE}")
print()

for cfg in EXPERIMENT_CONFIGS.values():
    ensure_config_exists(cfg)

copy_dir_if_needed(GRAPH_REPO_INPUT, GRAPH_REPO_WORKING, force=False)
GRAPH_REPO_PATH = GRAPH_REPO_WORKING
print("GRAPH_REPO_PATH:", GRAPH_REPO_PATH)
'''.splitlines(keepends=True)

# Update Cell 4 (Training Loop)
nb['cells'][4]['source'] = '''# Cell 4: Train models
import os
import sys
import time as _time
from pathlib import Path
import subprocess

EXPERIMENT_RESULTS = {}

def prepare_exp_output(exp_name: str) -> Path:
    config_path = EXPERIMENT_CONFIGS[exp_name]
    output_dir = OUTPUT_BASE / Path(config_path).stem
    if FORCE_OVERWRITE and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"[Clean] Removed old output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def build_train_cmd(exp_name: str, output_dir: Path) -> list:
    config_path = EXPERIMENT_CONFIGS[exp_name]
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", str(config_path),
        "--environment", ENVIRONMENT,
        "--graph_repo_path", str(GRAPH_REPO_PATH),
        "--output_root", str(output_dir),
        "--device", DEVICE,
    ]
    if BATCH_SIZE is not None:
        cmd += ["--batch_size", str(BATCH_SIZE)]
    if NUM_WORKERS is not None:
        cmd += ["--num_workers", str(NUM_WORKERS)]
    if EPOCHS_OVERRIDE is not None:
        cmd += ["--epochs", str(EPOCHS_OVERRIDE)]
    if MAX_TRAIN_BATCHES is not None:
        cmd += ["--max_train_batches", str(MAX_TRAIN_BATCHES)]
    if MAX_VAL_BATCHES is not None:
        cmd += ["--max_val_batches", str(MAX_VAL_BATCHES)]
    if PROFILE_BATCHES is not None:
        cmd += ["--profile_batches", str(PROFILE_BATCHES)]
    return cmd

for exp_name in EXPERIMENT_ORDER:
    print("\\n" + "="*60)
    print(f"TRAIN {exp_name.upper()}: {EXPERIMENT_LABELS[exp_name]}")
    print("="*60)
    output_dir = prepare_exp_output(exp_name)
    cmd = build_train_cmd(exp_name, output_dir)
    print("Command:")
    print(" ".join(str(x) for x in cmd))
    print()

    t0 = _time.time()
    run_cmd(cmd)
    elapsed = _time.time() - t0

    checkpoint_dir = output_dir / "checkpoints"
    best_ckpt = checkpoint_dir / "best.pth"
    last_ckpt = checkpoint_dir / "last.pth"
    EXPERIMENT_RESULTS[exp_name] = {
        "label": EXPERIMENT_LABELS[exp_name],
        "config_path": EXPERIMENT_CONFIGS[exp_name],
        "output_dir": output_dir,
        "best_checkpoint": best_ckpt if best_ckpt.exists() else None,
        "last_checkpoint": last_ckpt if last_ckpt.exists() else None,
        "elapsed_minutes": elapsed / 60,
    }
    print(f"\\n[{exp_name}] finished in {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"[{exp_name}] best: {best_ckpt} exists={best_ckpt.exists()}")

try:
    run_dir = Path(EXPERIMENT_RESULTS[EXPERIMENT_ORDER[-1]]["output_dir"])
    print("\\nACTIVE_RUN_DIR:", run_dir)
except Exception:
    pass
'''.splitlines(keepends=True)

# Update Cell 5 (Evaluation)
nb['cells'][5]['source'] = '''# Cell 5: Validate output artifacts + Evaluate on TEST set
import csv
import json
import numpy as np
import sys
import subprocess

def summarize_history(run_dir: Path):
    history_files = list(run_dir.glob("logs/*.csv"))
    if not history_files:
        print("  [WARN] No logs/*.csv found")
        return None
    rows = list(csv.DictReader(history_files[0].open("r", encoding="utf-8")))
    if not rows:
        return None
    epochs_run = max(int(float(r.get("epoch", 0))) for r in rows)
    print(f"  epochs_run: {epochs_run}")
    if "val_macro_f1" in rows[0]:
        best_row = max(rows, key=lambda r: float(r.get("val_macro_f1") or 0.0))
        print(f"  best_val_macro_f1_epoch: {best_row.get('epoch')}")
        print(f"  best_val_macro_f1: {best_row.get('val_macro_f1')}")
    return rows

def validate_exp(exp_name: str, result: dict):
    run_dir = Path(result["output_dir"])
    print("\\n" + "="*60)
    print(f"VALIDATE {exp_name.upper()}: {result['label']}")
    print("="*60)
    print("  output_dir:", run_dir, run_dir.exists())
    checkpoint_dir = run_dir / "checkpoints"
    print(f"  best.pth: {(checkpoint_dir / 'best.pth').exists()}")
    summarize_history(run_dir)

if "EXPERIMENT_RESULTS" not in globals() or not EXPERIMENT_RESULTS:
    raise RuntimeError("No EXPERIMENT_RESULTS found. Run Cell 4 before validation.")

for exp_name, result in EXPERIMENT_RESULTS.items():
    validate_exp(exp_name, result)

# ========== TEST SET EVALUATION ==========
print("\\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

if not RUN_TEST_EVALUATION:
    print("[SKIP] RUN_TEST_EVALUATION=False")
else:
    for exp_name in EXPERIMENT_ORDER:
        if exp_name not in EXPERIMENT_RESULTS:
            continue
        print(f"\\nEvaluating: {exp_name}")
        result = EXPERIMENT_RESULTS[exp_name]
        run_dir = Path(result["output_dir"])
        ckpt = run_dir / "checkpoints" / "best.pth"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoints" / "last.pth"
            
        if not ckpt.exists():
            print("[SKIP] No checkpoint found")
            continue
            
        eval_cmd = [
            sys.executable, "-m", "scripts.evaluate_d5a",
            "--config", str(result["config_path"]),
            "--environment", ENVIRONMENT,
            "--checkpoint", str(ckpt),
            "--graph_repo_path", str(GRAPH_REPO_PATH),
            "--output_root", str(run_dir),
            "--device", DEVICE,
            "--no_wandb",
            "--chunk_cache_size", str(CHUNK_CACHE_SIZE),
        ]
        if NUM_WORKERS is not None:
            eval_cmd += ["--num_workers", str(NUM_WORKERS)]
        subprocess.run(eval_cmd, check=True)

        eval_metrics = run_dir / "evaluation" / "metrics.json"
        if eval_metrics.exists():
            m = json.loads(eval_metrics.read_text(encoding="utf-8"))
            print(f"--> Accuracy: {m['accuracy']*100:.2f}%")
            print(f"--> Macro F1: {m['macro_f1']:.4f}")
'''.splitlines(keepends=True)

# Update Cell 6 (Summary)
nb['cells'][6]['source'] = '''# Cell 6: Summary and zip outputs
import time as _time
import json
from pathlib import Path

summary_items = []
for exp_name, result in EXPERIMENT_RESULTS.items():
    summary_items.append({
        "experiment": exp_name,
        "label": result["label"],
        "config_path": str(result["config_path"]),
        "output_dir": str(result["output_dir"]),
        "elapsed_minutes": result.get("elapsed_minutes"),
    })

run_summary = {
    "run_mode": RUN_MODE,
    "experiments_run": EXPERIMENTS_TO_RUN,
    "results": summary_items,
}

summary_out = Path("/kaggle/working/d11_1stage_summary.json")
summary_out.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
print(json.dumps(run_summary, indent=2))

if ZIP_OUTPUTS:
    for item in summary_items:
        exp_dir = Path(item["output_dir"])
        if not exp_dir.exists():
            continue
        zip_name = f"{exp_dir.name}_outputs"
        zip_path = Path(f"/kaggle/working/{zip_name}.zip")
        if zip_path.exists():
            zip_path = zip_path.with_name(zip_name + "_" + _time.strftime("%Y%m%d_%H%M%S") + ".zip")
        zip_dir(exp_dir, zip_path)
'''.splitlines(keepends=True)

with open('notebooks/kaggle-end-to-end.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
