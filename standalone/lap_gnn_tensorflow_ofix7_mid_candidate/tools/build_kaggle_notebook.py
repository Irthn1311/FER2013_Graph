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

Runs exactly one fresh TensorFlow/Keras seed-42 experiment after strict bounded
validation. The run persists one model checkpoint only, selected by validation
accuracy. Validation macro-F1 remains a logged diagnostic metric.

Required Kaggle Inputs:

- FER2013 split CSVs: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`
- verified D16 MediaPipe priors: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`
- clean graph cache: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`

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
GRAPH_CACHE_ROOT = Path("/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records")
OUTPUT_ROOT = Path("/kaggle/working/outputs/tensorflow_validation/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42")
TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
TF_PACKAGE_PATH = Path("/kaggle/working/FER2013_Graph") / TF_PACKAGE_RELATIVE
EXPECTED_TENSORFLOW_PAYLOAD_SHA = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
EXPECTED_EXECUTION_CONTRACT_SHA = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
TRAIN_CONFIG = TF_PACKAGE_PATH / "configs/fer2013_ofix7_mid_tensorflow_optimized_seed42.yaml"
FINAL_TEST_CHECKPOINT = "best_val_accuracy"
SELECTED_EXECUTION_STRATEGY = "SELECT_G1_RESTRICTED_GRAPH_OPTIMIZER"
SELECTED_EXECUTION_MODE = "restricted_tf_function"
SELECTED_GRAPPLER_PROFILE = "G1-A"
DEVICE_POLICY = "gpu"
ALLOW_CPU_TRAINING = False
GRAPH_WORKERS = 3
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
TF_DATA_PREFETCH = 4
RUN_FULL_TRAINING = True
RUN_TESTS = True
ARCHIVE_OUTPUT = True
SEED = 42
RESUME = False
WANDB_ENABLED = False
XLA_ENABLED = False

assert SEED == 42 and BATCH_SIZE == 16 and EVAL_BATCH_SIZE >= BATCH_SIZE
assert FINAL_TEST_CHECKPOINT == "best_val_accuracy"
assert RESUME is False and WANDB_ENABLED is False and XLA_ENABLED is False
if GRAPH_CACHE_ROOT is not None:
    GRAPH_CACHE_ROOT = Path(GRAPH_CACHE_ROOT)
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
dirty = run_checked(["git", "status", "--porcelain"], cwd=PROJECT_PATH, capture=True).strip()
print("requested_branch:", REPO_BRANCH)
print("actual_commit:", actual_commit)
print("dirty_tree:", bool(dirty))
if EXPECTED_COMMIT:
    if actual_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Commit mismatch: {actual_commit} != {EXPECTED_COMMIT}")
if dirty:
    raise RuntimeError("TensorFlow validation requires a clean cloned source tree")

TF_PACKAGE_PATH = PROJECT_PATH / TF_PACKAGE_RELATIVE
PYTORCH_GOLDEN = PROJECT_PATH / "standalone/lap_gnn_pytorch_ofix7_mid_candidate/validation_assets/golden"
for required in [
    TF_PACKAGE_PATH / "pyproject.toml",
    TF_PACKAGE_PATH / "CHECKSUMS.sha256",
    TF_PACKAGE_PATH / "package_manifest.json",
    TF_PACKAGE_PATH / "contracts/tensorflow_execution_contract_v2.json",
    TF_PACKAGE_PATH / "validation_assets/execution_mode/frozen_pytorch_two_step_reference.npz",
    TF_PACKAGE_PATH / "validation_assets/golden/model_state.npz",
    PYTORCH_GOLDEN / "model_state.npz",
]:
    if not required.is_file():
        raise FileNotFoundError(required)
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_checksums.py"], cwd=TF_PACKAGE_PATH)
manifest = json.loads((TF_PACKAGE_PATH / "package_manifest.json").read_text(encoding="utf-8"))
if manifest.get("scientific_payload_sha256") != EXPECTED_TENSORFLOW_PAYLOAD_SHA:
    raise RuntimeError("TensorFlow scientific payload SHA mismatch")
if manifest.get("execution_contract_sha256") != EXPECTED_EXECUTION_CONTRACT_SHA:
    raise RuntimeError("TensorFlow execution contract manifest SHA mismatch")
contract_path = TF_PACKAGE_PATH / "contracts/tensorflow_execution_contract_v2.json"
if hashlib.sha256(contract_path.read_bytes()).hexdigest() != EXPECTED_EXECUTION_CONTRACT_SHA:
    raise RuntimeError("TensorFlow execution contract file SHA mismatch")
if manifest.get("readiness_decision") != "READY_FOR_TENSORFLOW_KAGGLE_SEED42":
    raise RuntimeError("TensorFlow package is not registered READY")
print("tensorflow_package_path:", TF_PACKAGE_PATH)
print("package_payload_sha256:", manifest["scientific_payload_sha256"])
print("execution_contract_sha256:", EXPECTED_EXECUTION_CONTRACT_SHA)
print("package_manifest_files:", len(manifest.get("files", [])))
print("package_manifest_sha256:", hashlib.sha256(
    (TF_PACKAGE_PATH / "package_manifest.json").read_bytes()
).hexdigest())
print("package_checksums_sha256:", hashlib.sha256(
    (TF_PACKAGE_PATH / "CHECKSUMS.sha256").read_bytes()
).hexdigest())
"""
    ),
    markdown("## 3. Environment Inspection\n"),
    code(
        """import importlib.metadata
import importlib.util
import platform

print("python:", sys.version)
print("platform:", platform.platform(), platform.machine())
print("cpu logical:", os.cpu_count())
print("disk free GiB:", round(shutil.disk_usage("/kaggle/working").free / 2**30, 2))
print("thread env:", {key: os.environ.get(key) for key in [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS", "TF_ENABLE_ONEDNN_OPTS",
]})
def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

PREINSTALLED_TF = distribution_version("tensorflow")
PREINSTALLED_KERAS = distribution_version("keras")
print("preinstalled tensorflow metadata:", PREINSTALLED_TF)
print("preinstalled keras metadata:", PREINSTALLED_KERAS)
print("tensorflow imported before bootstrap:", "tensorflow" in sys.modules)
print("keras imported before bootstrap:", "keras" in sys.modules)
if "tensorflow" in sys.modules or "keras" in sys.modules:
    raise RuntimeError("TensorFlow/Keras was imported before the Save Version bootstrap")
if shutil.which("nvidia-smi"):
    run_checked(["nvidia-smi"])
"""
    ),
    markdown("## 4. Dependency Installation\n"),
    code(
        """TESTED_TF = "2.18.1"
TESTED_KERAS = "3.15.0"
compatible = PREINSTALLED_TF == TESTED_TF and PREINSTALLED_KERAS == TESTED_KERAS
if not compatible:
    print(
        "Installing the registered TensorFlow/Keras environment before either "
        "framework is imported. This is compatible with Kaggle Save Version."
    )
    run_checked([
        sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "-r",
        TF_PACKAGE_PATH / "requirements-kaggle.txt",
    ])
    importlib.invalidate_caches()

resolved_tf = distribution_version("tensorflow")
resolved_keras = distribution_version("keras")
if resolved_tf != TESTED_TF or resolved_keras != TESTED_KERAS:
    raise RuntimeError(
        f"Registered environment installation failed: tensorflow={resolved_tf}, "
        f"keras={resolved_keras}"
    )

missing = [
    requirement for module, requirement in [
        ("yaml", "PyYAML==6.0.2"),
        ("sklearn", "scikit-learn==1.6.1"),
        ("psutil", "psutil==6.1.1"),
        ("matplotlib", "matplotlib==3.10.0"),
    ] if importlib.util.find_spec(module) is None
]
if missing:
    run_checked([sys.executable, "-m", "pip", "install", "-q", *missing])
run_checked([sys.executable, "-m", "pip", "install", "-q", "-e", TF_PACKAGE_PATH, "--no-deps"])
probe = run_checked([
    sys.executable, "-B", "-c",
    "import json, tensorflow as tf; "
    "print(json.dumps({'tensorflow': tf.__version__, "
    "'keras': tf.keras.__version__, "
    "'gpus': [d.name for d in tf.config.list_physical_devices('GPU')], "
    "'build': tf.sysconfig.get_build_info()}, default=str))",
], cwd=TF_PACKAGE_PATH, capture=True).strip().splitlines()[-1]
probe_payload = json.loads(probe)
if probe_payload["tensorflow"] != TESTED_TF or probe_payload["keras"] != TESTED_KERAS:
    raise RuntimeError(f"Fresh-process TensorFlow/Keras mismatch: {probe_payload}")
print("fresh_process_environment:", json.dumps(probe_payload, indent=2))
run_checked([sys.executable, "-B", "-m", "lap_gnn_tf.cli.inspect_environment",
             "--output", WORKING / "tensorflow_environment.json"], cwd=TF_PACKAGE_PATH)
"""
    ),
    markdown("## 5. Import Isolation\n"),
    code(
        """PACKAGE_SRC = TF_PACKAGE_PATH / "src"
if not PACKAGE_SRC.is_dir():
    raise FileNotFoundError(PACKAGE_SRC)
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))
importlib.invalidate_caches()

import tensorflow as tf
import lap_gnn_tf
from lap_gnn_tf.training.execution import configure_restricted_grappler

resolved_package = Path(lap_gnn_tf.__file__).resolve()
print("lap_gnn_tf:", resolved_package)
print("lap_gnn_tf_source_bootstrap:", PACKAGE_SRC)
if TF_PACKAGE_PATH.resolve() not in resolved_package.parents:
    raise RuntimeError("lap_gnn_tf resolved outside the standalone TensorFlow package")
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_no_torch_runtime.py"], cwd=TF_PACKAGE_PATH)
run_checked([sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_no_parent_imports.py"], cwd=TF_PACKAGE_PATH)
if "torch" in sys.modules:
    raise RuntimeError("TensorFlow notebook imported torch")
if SELECTED_EXECUTION_MODE != "restricted_tf_function" or SELECTED_GRAPPLER_PROFILE != "G1-A":
    raise RuntimeError("Unregistered TensorFlow execution selection")
effective_grappler = configure_restricted_grappler()
for key in ("arithmetic_optimization", "remapping"):
    if effective_grappler.get(key) is not False:
        raise RuntimeError(f"G1-A Grappler option not active: {key}")
print("selected_execution_strategy =", SELECTED_EXECUTION_STRATEGY)
print("optimizer_execution_mode =", SELECTED_EXECUTION_MODE)
print("arithmetic_optimization =", str(effective_grappler["arithmetic_optimization"]).lower())
print("remapping =", str(effective_grappler["remapping"]).lower())
print("execution_contract_sha256 =", EXPECTED_EXECUTION_CONTRACT_SHA)
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
from lap_gnn_tf.config import load_config, validate_locked_config
assert len(NODE_FEATURE_NAMES) == 37
assert len(EDGE_FEATURE_NAMES) == 8
assert EXPECTED_PARAMETER_COUNT == 1_061_192
locked_config = load_config(TF_PACKAGE_PATH / "configs/fer2013_ofix7_mid_tensorflow_seed42.yaml")
validate_locked_config(locked_config)
training_config = locked_config["training"]
if training_config.get("gradient_execution_mode") != "tf_function":
    raise RuntimeError("Gradient execution mode drift")
if training_config.get("optimizer_execution_mode") != "restricted_tf_function":
    raise RuntimeError("Optimizer execution mode drift")
if training_config.get("grappler_arithmetic_optimization") is not False:
    raise RuntimeError("Resolved config arithmetic optimization drift")
if training_config.get("grappler_remapping") is not False:
    raise RuntimeError("Resolved config remapping drift")
if DEVICE_POLICY == "gpu" and not tf.config.list_physical_devices("GPU") and not ALLOW_CPU_TRAINING:
    raise RuntimeError("GPU required but TensorFlow sees no GPU")
if GRAPH_CACHE_ROOT is not None:
    cache_marker = GRAPH_CACHE_ROOT / "CACHE_COMPLETE.json"
    if not cache_marker.is_file():
        raise FileNotFoundError(f"Clean graph cache marker missing: {cache_marker}")
    cache_payload = json.loads(cache_marker.read_text(encoding="utf-8"))
    if cache_payload.get("schema_version") != "tf_clean_graph_cache_v2_records":
        raise RuntimeError("Unsupported clean graph cache schema")
    if cache_payload.get("node_dim") != 37 or cache_payload.get("edge_dim") != 8:
        raise RuntimeError("Clean graph cache feature dimensions do not match 37/8")
    print("clean_graph_cache_root:", GRAPH_CACHE_ROOT)
    print("clean_graph_cache_graph_config_sha256:", cache_payload.get("graph_config_sha256"))
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
EXECUTION_PREFLIGHT = WORKING / "tensorflow_g1_a_two_step_preflight.json"
run_checked([
    sys.executable, "-B", TF_PACKAGE_PATH / "tools/evaluate_g1_grappler.py",
    "--worker",
    "--mode", "G1-A",
    "--repeats", "1",
    "--package-root", TF_PACKAGE_PATH,
    "--reference", TF_PACKAGE_PATH / "validation_assets/execution_mode/frozen_pytorch_two_step_reference.npz",
    "--result", EXECUTION_PREFLIGHT,
], cwd=TF_PACKAGE_PATH, env=parity_env)
execution_preflight = json.loads(EXECUTION_PREFLIGHT.read_text(encoding="utf-8"))
if not execution_preflight.get("pass"):
    raise RuntimeError(f"G1-A two-step optimizer preflight failed: {execution_preflight}")
if execution_preflight.get("configuration") != SELECTED_GRAPPLER_PROFILE:
    raise RuntimeError("Optimizer preflight execution profile mismatch")
if execution_preflight["graph_audit"].get("contains_py_function"):
    raise RuntimeError("Optimizer preflight contains forbidden PyFunc")
for repeat in execution_preflight["repetitions"]:
    for step in repeat["steps"]:
        if step["variable_count"] != 127 or not step["pass"]:
            raise RuntimeError(f"Optimizer preflight state mismatch: {step}")

selected_validation_path = (
    TF_PACKAGE_PATH / "validation_assets/execution_mode/selected_g1_validation.json"
)
selected_validation = json.loads(selected_validation_path.read_text(encoding="utf-8"))
if not selected_validation.get("pass"):
    raise RuntimeError("Registered G1 validation bundle is not PASS")
if not selected_validation.get("checkpoint_continuation", {}).get("pass"):
    raise RuntimeError("Registered checkpoint roundtrip/continuation is not PASS")
if not selected_validation.get("mixed_precision", {}).get("pass"):
    raise RuntimeError("Registered mixed-precision bounded smoke is not PASS")
if RESUME or XLA_ENABLED or WANDB_ENABLED or SEED != 42:
    raise RuntimeError("Fresh TensorFlow seed42 runtime lock drift")
if RUN_FULL_TRAINING and OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise RuntimeError(f"Fresh output is contaminated: {OUTPUT_ROOT}")
print("READY_FOR_TENSORFLOW_KAGGLE_SEED42")
"""
    ),
    markdown("## 9. Fresh TensorFlow Seed42 Training\n"),
    code(
        """if RUN_FULL_TRAINING:
    if RESUME:
        raise RuntimeError("Resume is forbidden for the locked TensorFlow seed42 run")
    train_command = [
        sys.executable, "-B", "-m", "lap_gnn_tf.cli.train",
        "--config", TRAIN_CONFIG,
        "--fer-csv", FER_CSV_PATH,
        "--prior-root", PRIOR_ROOT,
        "--output-root", OUTPUT_ROOT,
        "--device", DEVICE_POLICY,
        "--graph-workers", str(GRAPH_WORKERS),
        "--tf-data-prefetch", str(TF_DATA_PREFETCH),
        "--batch-size", str(BATCH_SIZE),
        "--eval-batch-size", str(EVAL_BATCH_SIZE),
        "--no-resume",
        "--mixed-precision",
        "--no-xla",
        "--memory-growth",
    ]
    if GRAPH_CACHE_ROOT is not None:
        train_command.extend(["--clean-graph-cache-dir", GRAPH_CACHE_ROOT])
    if ALLOW_CPU_TRAINING:
        train_command.append("--allow-cpu-training")
    run_checked(train_command, cwd=TF_PACKAGE_PATH)
else:
    print("RUN_FULL_TRAINING=False: bounded implementation validation only; no epoch was launched.")
"""
    ),
    markdown("## 10. Post-run Validation\n"),
    code(
        """import hashlib
import math
import yaml
from lap_gnn_tf.model import LapGNN

execution_summary = {
    "framework": "tensorflow",
    "seed": SEED,
    "commit": actual_commit,
    "full_training_requested": RUN_FULL_TRAINING,
    "resume": RESUME,
    "tensorflow_payload_sha256": EXPECTED_TENSORFLOW_PAYLOAD_SHA,
    "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA,
    "selected_execution_mode": SELECTED_EXECUTION_MODE,
    "selected_grappler_profile": SELECTED_GRAPPLER_PROFILE,
    "effective_grappler": effective_grappler,
    "golden_parity": parity,
    "optimizer_two_step_preflight": execution_preflight,
    "bounded_mixed_precision": selected_validation["mixed_precision"],
    "bounded_checkpoint_roundtrip": selected_validation["checkpoint_continuation"],
}
if RUN_FULL_TRAINING:
    selected_checkpoint_stem = FINAL_TEST_CHECKPOINT
    selected_checkpoint_name = f"{selected_checkpoint_stem}.keras"
    selected_metrics_name = f"test_metrics_{selected_checkpoint_stem}.json"
    required = [
        OUTPUT_ROOT / "TRAINING_COMPLETE.json",
        OUTPUT_ROOT / "resolved_config.yaml",
        OUTPUT_ROOT / "provenance.json",
        OUTPUT_ROOT / "history.json",
        OUTPUT_ROOT / "telemetry.json",
        OUTPUT_ROOT / selected_metrics_name,
        OUTPUT_ROOT / "checkpoints/best_val_accuracy.keras",
        OUTPUT_ROOT / "checkpoints/best_val_accuracy.metadata.json",
        OUTPUT_ROOT / "checkpoints/best_val_accuracy.weights.h5",
        OUTPUT_ROOT / "train_log.csv",
        OUTPUT_ROOT / "latest_epoch_summary.json",
        OUTPUT_ROOT / "run_summary.json",
        OUTPUT_ROOT / "artifact_manifest.json",
        OUTPUT_ROOT / "training_curves.png",
        OUTPUT_ROOT / "confusion_matrix.csv",
        OUTPUT_ROOT / "confusion_matrix.png",
        OUTPUT_ROOT / "per_class_metrics.csv",
        OUTPUT_ROOT / "predictions.csv",
        OUTPUT_ROOT / "resolved_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"TensorFlow run artifacts missing: {missing}")
    completion = json.loads((OUTPUT_ROOT / "TRAINING_COMPLETE.json").read_text())
    if not completion.get("completed"):
        raise RuntimeError("Training completion marker is not complete")
    if completion.get("resume") or completion.get("test_used_for_selection"):
        raise RuntimeError("Run provenance violates fresh validation-only selection")
    if completion.get("selected_checkpoint") != selected_checkpoint_name:
        raise RuntimeError("Primary checkpoint selection drift")

    resolved_config = yaml.safe_load((OUTPUT_ROOT / "resolved_config.yaml").read_text())
    resolved_json = json.loads((OUTPUT_ROOT / "resolved_config.json").read_text())
    if resolved_config != resolved_json:
        raise RuntimeError("Resolved YAML/JSON configuration mismatch")
    resolved_training = resolved_config["training"]
    if resolved_training.get("final_test_checkpoint") != FINAL_TEST_CHECKPOINT:
        raise RuntimeError("Resolved final test checkpoint mismatch")
    if resolved_training.get("checkpoint_monitor") != "val_accuracy":
        raise RuntimeError("Resolved checkpoint monitor mismatch")
    checkpoint_policy = resolved_training.get("checkpoint_policy", {})
    if (
        checkpoint_policy.get("type") != "single"
        or checkpoint_policy.get("monitor") != "val_accuracy"
    ):
        raise RuntimeError("Resolved configuration is not single-checkpoint accuracy")
    if resolved_training.get("gradient_execution_mode") != "tf_function":
        raise RuntimeError("Resolved gradient execution mode mismatch")
    if resolved_training.get("optimizer_execution_mode") != "restricted_tf_function":
        raise RuntimeError("Resolved optimizer execution mode mismatch")
    if resolved_training.get("grappler_arithmetic_optimization") is not False:
        raise RuntimeError("Resolved arithmetic optimization mismatch")
    if resolved_training.get("grappler_remapping") is not False:
        raise RuntimeError("Resolved remapping mismatch")
    if resolved_config["standalone"].get("resume_enabled") is not False:
        raise RuntimeError("Resolved configuration permits resume")
    if resolved_config["resources"].get("xla") is not False:
        raise RuntimeError("Resolved configuration enables XLA")
    if resolved_config["locked"].get("package_checksum") != EXPECTED_TENSORFLOW_PAYLOAD_SHA:
        raise RuntimeError("Resolved package payload lock mismatch")
    if resolved_config["locked"].get("execution_contract_sha256") != EXPECTED_EXECUTION_CONTRACT_SHA:
        raise RuntimeError("Resolved execution contract lock mismatch")

    from lap_gnn_tf.config import canonical_config_hash
    resolved_config_hash = canonical_config_hash(resolved_config)
    provenance = json.loads((OUTPUT_ROOT / "provenance.json").read_text())
    if provenance.get("config_hash") != resolved_config_hash:
        raise RuntimeError("Provenance config hash mismatch")
    expected_signatures = {
        "config": resolved_config_hash,
        "graph": resolved_config["locked"]["graph_signature"],
        "feature": resolved_config["locked"]["feature_signature"],
        "prior": resolved_config["locked"]["prior_signature"],
        "dataset_split": resolved_config["locked"]["dataset_split_signature"],
    }
    if provenance.get("signatures") != expected_signatures:
        raise RuntimeError("Run provenance signature mismatch")

    history_payload = json.loads((OUTPUT_ROOT / "history.json").read_text())
    history = history_payload.get("epochs", [])
    if not history or len(history) != int(completion.get("epochs", -1)):
        raise RuntimeError("Training history length mismatch")
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise RuntimeError("Training history epoch sequence is not contiguous")
    history_numeric = [
        "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr", "epoch_time_sec",
    ]
    for row in history:
        for key in history_numeric:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"Non-finite history value: epoch={row['epoch']} key={key}")
    with (OUTPUT_ROOT / "train_log.csv").open("r", encoding="utf-8", newline="") as stream:
        train_log_rows = list(csv.DictReader(stream))
    if len(train_log_rows) != len(history):
        raise RuntimeError("train_log.csv and history.json length mismatch")
    artifact_manifest = json.loads(
        (OUTPUT_ROOT / "artifact_manifest.json").read_text()
    )
    artifact_names = {
        item["path"] for item in artifact_manifest.get("artifacts", [])
    }
    expected_artifacts = {
        "history.json", "train_log.csv", "training_curves.png",
        selected_metrics_name, "per_class_metrics.csv",
        "predictions.csv", "confusion_matrix.csv", "confusion_matrix.png",
        "run_summary.json",
    }
    if not expected_artifacts.issubset(artifact_names):
        raise RuntimeError(
            f"Artifact manifest incomplete: {expected_artifacts - artifact_names}"
        )
    for item in artifact_manifest.get("artifacts", []):
        artifact_path = OUTPUT_ROOT / item["path"]
        if not artifact_path.is_file():
            raise RuntimeError(f"Artifact manifest path missing: {artifact_path}")
        actual_bytes = artifact_path.stat().st_size
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_bytes != int(item["bytes"]) or actual_sha256 != item["sha256"]:
            raise RuntimeError(
                f"Artifact manifest hash mismatch: {artifact_path}"
            )

    metrics = json.loads((OUTPUT_ROOT / selected_metrics_name).read_text())
    for key in ["loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "nll", "brier", "ece"]:
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"Non-finite or missing metric: {key}")
    for key in ["per_class_precision", "per_class_recall", "per_class_f1", "support_per_class"]:
        if len(metrics.get(key, [])) != 7:
            raise RuntimeError(f"Classwise metric shape mismatch: {key}")
    if len(metrics.get("confusion_matrix", [])) != 7 or any(
        len(row) != 7 for row in metrics["confusion_matrix"]
    ):
        raise RuntimeError("Confusion matrix shape mismatch")

    checkpoint_files = sorted(
        path.name for path in (OUTPUT_ROOT / "checkpoints").iterdir()
        if path.is_file()
    )
    expected_checkpoint_files = [
        "best_val_accuracy.keras",
        "best_val_accuracy.metadata.json",
        "best_val_accuracy.weights.h5",
    ]
    if checkpoint_files != expected_checkpoint_files:
        raise RuntimeError(
            "Single-checkpoint inventory mismatch: "
            f"actual={checkpoint_files} expected={expected_checkpoint_files}"
        )
    stem = "best_val_accuracy"
    checkpoint_path = OUTPUT_ROOT / "checkpoints" / f"{stem}.keras"
    metadata_path = OUTPUT_ROOT / "checkpoints" / f"{stem}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("config_hash") != resolved_config_hash:
        raise RuntimeError(f"{stem} config hash mismatch")
    if metadata.get("package_checksum") != EXPECTED_TENSORFLOW_PAYLOAD_SHA:
        raise RuntimeError(f"{stem} package payload mismatch")
    if metadata.get("execution_contract_sha256") != EXPECTED_EXECUTION_CONTRACT_SHA:
        raise RuntimeError(f"{stem} execution contract mismatch")
    for key, expected in [
        ("graph_signature", expected_signatures["graph"]),
        ("feature_signature", expected_signatures["feature"]),
        ("prior_signature", expected_signatures["prior"]),
        ("dataset_split_signature", expected_signatures["dataset_split"]),
    ]:
        if metadata.get(key) != expected:
            raise RuntimeError(f"{stem} {key} mismatch")
    loaded = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"LapGNN": LapGNN, "lap_gnn_tf>LapGNN": LapGNN},
        compile=False,
    )
    parameters = sum(int(v.shape.num_elements()) for v in loaded.trainable_variables)
    if parameters != 1_061_192:
        raise RuntimeError(f"{stem} checkpoint architecture drift")
    if not all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in loaded.variables):
        raise RuntimeError(f"{stem} contains non-finite parameters")
    checkpoint_inventory = {
        stem: {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "epoch": int(metadata["epoch"]),
            "validation_metrics": metadata["validation_metrics"],
            "parameter_count": parameters,
            "strict_load": True,
        }
    }

    telemetry = json.loads((OUTPUT_ROOT / "telemetry.json").read_text())
    train_steps = [float(value) for value in telemetry.get("train_step_sec", [])]
    validation_times = [float(value) for value in telemetry.get("validation_sec", [])]
    if not train_steps or not validation_times:
        raise RuntimeError("Training telemetry is incomplete")
    if not all(math.isfinite(value) and value >= 0.0 for value in train_steps + validation_times):
        raise RuntimeError("Training telemetry contains non-finite durations")
    best_accuracy = checkpoint_inventory["best_val_accuracy"]
    selected_checkpoint = checkpoint_inventory[selected_checkpoint_stem]
    total_epochs = len(history)
    max_epochs = int(resolved_training["max_epochs"])
    stop_reason = "early_stopping" if total_epochs < max_epochs else "max_epochs"
    execution_summary.update({
        "config_hash": resolved_config_hash,
        "signatures": expected_signatures,
        "checkpoint_inventory": checkpoint_inventory,
        "selected_checkpoint": selected_checkpoint_name,
        "best_epoch": selected_checkpoint["epoch"],
        "best_accuracy_epoch": best_accuracy["epoch"],
        "total_epochs": total_epochs,
        "stop_reason": stop_reason,
        "validation_metrics_at_selected_checkpoint": selected_checkpoint[
            "validation_metrics"
        ],
        "test_metrics": metrics,
        "classwise_metrics": {
            "precision": metrics["per_class_precision"],
            "recall": metrics["per_class_recall"],
            "f1": metrics["per_class_f1"],
            "support": metrics["support_per_class"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "calibration_metrics": {
            "nll": metrics["nll"], "brier": metrics["brier"], "ece": metrics["ece"],
        },
        "peak_ram_bytes": int(telemetry.get("peak_host_rss_bytes", 0)),
        "peak_gpu_memory_bytes": int(telemetry.get("peak_gpu_memory_bytes", 0)),
        "average_train_step_sec": sum(train_steps) / len(train_steps),
        "validation_duration_sec_total": sum(validation_times),
        "bounded_overflow_and_dynamic_loss_scale_events": selected_validation[
            "mixed_precision"
        ].get("loss_scale_recovery_attempts", []),
        "full_run_overflow_event_note": (
            "The locked trainer does not emit per-step overflow events; bounded G1 mixed-"
            "precision recovery evidence is preserved above without changing trainer math."
        ),
        "test_evaluation_after_checkpoint_selection": True,
        "artifact_manifest": artifact_manifest,
        "training_curves": str(OUTPUT_ROOT / "training_curves.png"),
        "confusion_matrix_png": str(OUTPUT_ROOT / "confusion_matrix.png"),
        "confusion_matrix_csv": str(OUTPUT_ROOT / "confusion_matrix.csv"),
        "per_class_metrics_csv": str(OUTPUT_ROOT / "per_class_metrics.csv"),
        "predictions_csv": str(OUTPUT_ROOT / "predictions.csv"),
    })
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
        WORKING / "tensorflow_g1_a_two_step_preflight.json",
        WORKING / "tensorflow_notebook_execution_summary.json",
        TF_PACKAGE_PATH / "package_manifest.json",
        TF_PACKAGE_PATH / "CHECKSUMS.sha256",
        TF_PACKAGE_PATH / "contracts/tensorflow_execution_contract_v2.json",
        TF_PACKAGE_PATH / "contracts/tensorflow_execution_contract_v2.sha256",
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
with NOTEBOOK.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")
print(NOTEBOOK)
