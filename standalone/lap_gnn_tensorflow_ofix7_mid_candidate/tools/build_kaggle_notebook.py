"""Generate the concise fail-closed TensorFlow seed42 Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "kaggle-end-to-end.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# TensorFlow LAP-GNN OFIX7-mid Seed 42

Runs exactly one fresh TensorFlow/Keras seed-42 experiment after strict bounded validation.

Required Kaggle Inputs:

- FER2013 split CSVs: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`
- verified D16 MediaPipe priors: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`

The preserved PyTorch runner is `notebooks/kaggle-end-to-end-pytorch.ipynb`.
"""
    ),
    markdown("## 1. User Configuration\n"),
    code(
        """from pathlib import Path

REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
REPO_BRANCH = "main"
EXPECTED_COMMIT = None
FER_SPLIT_ROOT = Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
FER_CSV_PATH = FER_SPLIT_ROOT / "train.csv"
PRIOR_ROOT = Path("/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
OUTPUT_ROOT = Path("/kaggle/working/outputs/tensorflow_validation/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42")
TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
DEVICE_POLICY = "gpu"
ALLOW_CPU_TRAINING = False
GRAPH_WORKERS = 2
BATCH_SIZE = 16
RUN_FULL_TRAINING = True
RUN_TESTS = True
ARCHIVE_OUTPUT = True
SEED = 42
RESUME = False
WANDB_ENABLED = False
XLA_ENABLED = False

assert SEED == 42 and BATCH_SIZE == 16
assert RESUME is False and WANDB_ENABLED is False and XLA_ENABLED is False
"""
    ),
    markdown("## 2. Clone and Source Validation\n"),
    code(
        """import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
PROJECT_PATH = WORKING / "FER2013_Graph"

def run_checked(command, cwd=None, env=None, capture=False):
    actual = [str(item) for item in command]
    display = [re.sub(r"(https://x-access-token:)[^@]+@", r"\\1***@", item) for item in actual]
    print("$", " ".join(display))
    result = subprocess.run(
        actual, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode:
        if capture and result.stdout:
            print("\\n".join(result.stdout.splitlines()[-100:]))
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result.stdout if capture else ""

if PROJECT_PATH.exists():
    shutil.rmtree(PROJECT_PATH)
clone_url = REPO_URL
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    token = None
if token and clone_url.startswith("https://github.com/"):
    clone_url = clone_url.replace("https://", f"https://x-access-token:{token}@")
run_checked(["git", "clone", "--branch", REPO_BRANCH, "--single-branch", clone_url, PROJECT_PATH])
os.chdir(PROJECT_PATH)
actual_commit = run_checked(["git", "rev-parse", "HEAD"], cwd=PROJECT_PATH, capture=True).strip()
print("actual_commit:", actual_commit)
if EXPECTED_COMMIT:
    if actual_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Commit mismatch: {actual_commit} != {EXPECTED_COMMIT}")
    dirty = run_checked(["git", "status", "--porcelain"], cwd=PROJECT_PATH, capture=True).strip()
    if dirty:
        raise RuntimeError("Expected-commit run requires a clean checkout")

TF_PACKAGE_PATH = PROJECT_PATH / TF_PACKAGE_RELATIVE
PYTORCH_GOLDEN = PROJECT_PATH / "standalone/lap_gnn_pytorch_ofix7_mid_candidate/validation_assets/golden"
for required in [
    TF_PACKAGE_PATH / "pyproject.toml",
    TF_PACKAGE_PATH / "CHECKSUMS.sha256",
    TF_PACKAGE_PATH / "package_manifest.json",
    TF_PACKAGE_PATH / "validation_assets/golden/model_state.npz",
    PYTORCH_GOLDEN / "model_state.npz",
]:
    if not required.is_file():
        raise FileNotFoundError(required)
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_checksums.py"], cwd=TF_PACKAGE_PATH)
"""
    ),
    markdown("## 3. Environment Inspection\n"),
    code(
        """import importlib.util
import platform
import psutil

print("python:", sys.version)
print("platform:", platform.platform(), platform.machine())
print("cpu logical/physical:", psutil.cpu_count(True), psutil.cpu_count(False))
print("ram GiB:", round(psutil.virtual_memory().total / 2**30, 2))
print("disk free GiB:", round(shutil.disk_usage("/kaggle/working").free / 2**30, 2))
print("thread env:", {key: os.environ.get(key) for key in [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS", "TF_ENABLE_ONEDNN_OPTS",
]})
tf_spec = importlib.util.find_spec("tensorflow")
print("tensorflow installed:", bool(tf_spec))
if tf_spec:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
    print("keras:", tf.keras.__version__)
    print("build:", json.dumps(tf.sysconfig.get_build_info(), default=str, indent=2))
    print("CPU:", tf.config.list_physical_devices("CPU"))
    print("GPU:", tf.config.list_physical_devices("GPU"))
if shutil.which("nvidia-smi"):
    run_checked(["nvidia-smi"])
"""
    ),
    markdown("## 4. Dependency Installation\n"),
    code(
        """TESTED_TF = "2.18.1"
TESTED_KERAS = "3.15.0"
compatible = bool(tf_spec) and tf.__version__ == TESTED_TF and tf.keras.__version__ == TESTED_KERAS
if not compatible:
    run_checked([
        sys.executable, "-m", "pip", "install", "-q", "-r",
        TF_PACKAGE_PATH / "requirements-kaggle.txt",
    ])
    raise RuntimeError(
        "TensorFlow/Keras was replaced with the tested build. Restart the Kaggle session, "
        "then Run All again before training."
    )

missing = [
    requirement for module, requirement in [
        ("yaml", "PyYAML==6.0.2"),
        ("sklearn", "scikit-learn==1.6.1"),
        ("psutil", "psutil==6.1.1"),
    ] if importlib.util.find_spec(module) is None
]
if missing:
    run_checked([sys.executable, "-m", "pip", "install", "-q", *missing])
run_checked([sys.executable, "-m", "pip", "install", "-q", "-e", TF_PACKAGE_PATH, "--no-deps"])
run_checked([sys.executable, "-B", "-m", "lap_gnn_tf.cli.inspect_environment",
             "--output", WORKING / "tensorflow_environment.json"], cwd=TF_PACKAGE_PATH)
"""
    ),
    markdown("## 5. Import Isolation\n"),
    code(
        """import lap_gnn_tf

resolved_package = Path(lap_gnn_tf.__file__).resolve()
print("lap_gnn_tf:", resolved_package)
if TF_PACKAGE_PATH.resolve() not in resolved_package.parents:
    raise RuntimeError("lap_gnn_tf resolved outside the standalone TensorFlow package")
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_no_torch_runtime.py"], cwd=TF_PACKAGE_PATH)
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_no_parent_imports.py"], cwd=TF_PACKAGE_PATH)
if "torch" in sys.modules:
    raise RuntimeError("TensorFlow notebook imported torch")
"""
    ),
    markdown("## 6. Bounded Tests\n"),
    code(
        """if RUN_TESTS:
    parity_env = os.environ.copy()
    parity_env["CUDA_VISIBLE_DEVICES"] = ""
    parity_env["TF_DETERMINISTIC_OPS"] = "1"
    test_output = run_checked(
        [sys.executable, "-B", "-m", "pytest", "-q"],
        cwd=TF_PACKAGE_PATH, env=parity_env, capture=True,
    )
    print("\\n".join(test_output.splitlines()[-30:]))
else:
    raise RuntimeError("RUN_TESTS must remain True for the first TensorFlow seed42 run")
"""
    ),
    markdown("## 7. Dataset and Prior Validation\n"),
    code(
        """import csv

expected_counts = {"train": 28709, "val": 3589, "test": 3589}
for split, expected in expected_counts.items():
    csv_path = FER_SPLIT_ROOT / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = [item.strip().lower() for item in next(reader)]
        count = sum(1 for _ in reader)
    if "emotion" not in header or "pixels" not in header or count != expected:
        raise RuntimeError(f"{split} CSV mismatch: rows={count}, header={header}")
    prior_count = len(list((PRIOR_ROOT / split).glob("*.npz")))
    if prior_count != expected:
        raise RuntimeError(f"{split} prior count mismatch: {prior_count} != {expected}")
schema_path = PRIOR_ROOT / "prior_schema.json"
if not schema_path.is_file() or "d16_mediapipe_pixel_priors_v1" not in schema_path.read_text(encoding="utf-8"):
    raise RuntimeError("Verified prior schema mismatch")

run_checked([
    sys.executable, "-B", "-m", "lap_gnn_tf.cli.validate",
    "--config", TF_PACKAGE_PATH / "configs/fer2013_ofix7_mid_tensorflow_seed42.yaml",
    "--fer-csv", FER_CSV_PATH,
    "--prior-root", PRIOR_ROOT,
], cwd=TF_PACKAGE_PATH)

from lap_gnn_tf.constants import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, EXPECTED_PARAMETER_COUNT
assert len(NODE_FEATURE_NAMES) == 37
assert len(EDGE_FEATURE_NAMES) == 8
assert EXPECTED_PARAMETER_COUNT == 1_061_192
if DEVICE_POLICY == "gpu" and not tf.config.list_physical_devices("GPU") and not ALLOW_CPU_TRAINING:
    raise RuntimeError("GPU required but TensorFlow sees no GPU")
"""
    ),
    markdown("## 8. Golden Parity Preflight\n"),
    code(
        """PARITY_REPORT = WORKING / "tensorflow_golden_parity.json"
parity_env = os.environ.copy()
parity_env["CUDA_VISIBLE_DEVICES"] = ""
parity_env["TF_DETERMINISTIC_OPS"] = "1"
run_checked([
    sys.executable, "-B", "-m", "lap_gnn_tf.cli.compare_golden",
    "--package-root", TF_PACKAGE_PATH,
    "--output", PARITY_REPORT,
], cwd=TF_PACKAGE_PATH, env=parity_env)
parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
if not parity["pass"]:
    raise RuntimeError(f"Golden parity failed: {parity}")
if parity["prediction_agreement"] != 1.0 or parity["max_logit_difference"] > 1e-5:
    raise RuntimeError("TensorFlow forward parity gate failed")
print("READY_FOR_TENSORFLOW_KAGGLE_SEED42")
"""
    ),
    markdown("## 9. Fresh TensorFlow Seed42 Training\n"),
    code(
        """if RUN_FULL_TRAINING:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(f"Fresh output is contaminated: {OUTPUT_ROOT}")
    if RESUME:
        raise RuntimeError("Resume is forbidden for the locked TensorFlow seed42 run")
    train_command = [
        sys.executable, "-B", "-m", "lap_gnn_tf.cli.train",
        "--config", TF_PACKAGE_PATH / "configs/fer2013_ofix7_mid_tensorflow_seed42.yaml",
        "--fer-csv", FER_CSV_PATH,
        "--prior-root", PRIOR_ROOT,
        "--output-root", OUTPUT_ROOT,
        "--device", DEVICE_POLICY,
        "--graph-workers", str(GRAPH_WORKERS),
        "--batch-size", str(BATCH_SIZE),
        "--no-resume",
        "--mixed-precision",
        "--no-xla",
        "--memory-growth",
    ]
    if ALLOW_CPU_TRAINING:
        train_command.append("--allow-cpu-training")
    run_checked(train_command, cwd=TF_PACKAGE_PATH)
else:
    print("RUN_FULL_TRAINING=False: bounded implementation validation only; no epoch was launched.")
"""
    ),
    markdown("## 10. Post-run Validation\n"),
    code(
        """execution_summary = {
    "framework": "tensorflow",
    "seed": SEED,
    "commit": actual_commit,
    "full_training_requested": RUN_FULL_TRAINING,
    "resume": RESUME,
    "golden_parity": parity,
}
if RUN_FULL_TRAINING:
    required = [
        OUTPUT_ROOT / "TRAINING_COMPLETE.json",
        OUTPUT_ROOT / "resolved_config.yaml",
        OUTPUT_ROOT / "provenance.json",
        OUTPUT_ROOT / "history.json",
        OUTPUT_ROOT / "telemetry.json",
        OUTPUT_ROOT / "test_metrics_best_val_macro_f1.json",
        OUTPUT_ROOT / "checkpoints/best.keras",
        OUTPUT_ROOT / "checkpoints/best_val_macro_f1.keras",
        OUTPUT_ROOT / "checkpoints/best_val_accuracy.keras",
        OUTPUT_ROOT / "checkpoints/last.keras",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"TensorFlow run artifacts missing: {missing}")
    completion = json.loads((OUTPUT_ROOT / "TRAINING_COMPLETE.json").read_text())
    if completion.get("resume") or completion.get("test_used_for_selection"):
        raise RuntimeError("Run provenance violates fresh validation-only selection")
    metrics = json.loads((OUTPUT_ROOT / "test_metrics_best_val_macro_f1.json").read_text())
    for key in ["loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "nll", "brier", "ece"]:
        if key not in metrics or not float("-inf") < float(metrics[key]) < float("inf"):
            raise RuntimeError(f"Non-finite or missing metric: {key}")
    loaded = tf.keras.models.load_model(OUTPUT_ROOT / "checkpoints/best_val_macro_f1.keras", compile=False)
    if sum(int(v.shape.num_elements()) for v in loaded.trainable_variables) != 1_061_192:
        raise RuntimeError("Checkpoint architecture drift")
    execution_summary["test_metrics"] = metrics
(WORKING / "tensorflow_notebook_execution_summary.json").write_text(
    json.dumps(execution_summary, indent=2, default=str), encoding="utf-8"
)
print(json.dumps(execution_summary, indent=2, default=str))
"""
    ),
    markdown("## 11. Archive\n"),
    code(
        """ARCHIVE_PATH = WORKING / "ofix7_mid_seed42_tensorflow_outputs.zip"
if ARCHIVE_OUTPUT and RUN_FULL_TRAINING:
    staging = WORKING / "tensorflow_archive_metadata"
    staging.mkdir(exist_ok=True)
    for source in [
        WORKING / "tensorflow_environment.json",
        WORKING / "tensorflow_golden_parity.json",
        WORKING / "tensorflow_notebook_execution_summary.json",
        TF_PACKAGE_PATH / "package_manifest.json",
        TF_PACKAGE_PATH / "CHECKSUMS.sha256",
    ]:
        shutil.copy2(source, staging / source.name)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    run_checked([
        "zip", "-qr", ARCHIVE_PATH,
        OUTPUT_ROOT.relative_to(WORKING),
        staging.relative_to(WORKING),
    ], cwd=WORKING)
    print("archive:", ARCHIVE_PATH, "bytes:", ARCHIVE_PATH.stat().st_size)
else:
    print("Archive skipped because full training was not run or ARCHIVE_OUTPUT=False.")
"""
    ),
]

payload = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(NOTEBOOK)
