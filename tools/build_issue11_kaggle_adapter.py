"""Build the preregistered Issue #11 Kaggle T4 execution adapter notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue11-fixed-topology-prior-probe.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown(
            """
            # Issue #11: fixed-topology validation prior-sensitivity adapter

            This is the dedicated **pre-run adapter** for the preregistered Step 6
            experiment. It is intentionally unexecuted in this PR. When approved, it
            runs the reviewed Step 5 tool once on Kaggle GPU T4, using the exact Issue #7
            epoch-31 checkpoint and validation assets only.

            Required Kaggle Inputs and resolved reads:

            - FER split input: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`;
              only `val.csv` is opened.
            - MediaPipe prior input:
              `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
              only `val/*.npz` and shared root schema/name metadata required by the frozen
              loader are used.
            - Clean graph cache input:
              `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; only
              `val/index.json`, its referenced validation shards, and root
              `CACHE_COMPLETE.json` are used. The completion marker is disclosed as
              shared non-sample aggregate metadata and may summarize other splits.
            - Issue #7 artifact input: attach one separate read-only Kaggle Input
              containing `best_val_accuracy.keras`, `best_val_accuracy.metadata.json`,
              and `resolved_config.json`. Its mount name is intentionally not trusted;
              the adapter locates each expected basename outside the three public sample
              inputs and accepts it only by its preregistered SHA-256.

            Internet is required only to clone the exact repository commit and, if the
            Kaggle image is incompatible, install the registered package dependencies.
            All data and checkpoint artifacts are offline Kaggle Inputs.

            Compact outputs:

            - report: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md`
            - archive: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip`

            C2 remains conditional on the official MediaPipe-derived topology. It is not
            a prior-free or MediaPipe-free graph.
            """
        ),
        markdown("## 1. Preregistered constants and immutable paths\n"),
        code(
            """
            from pathlib import Path

            REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
            REPO_BRANCH = "main"
            EXPECTED_COMMIT = "69f4571c5069da9a7f8558ef3c01101635ee904a"
            TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
            PROBE_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_prior_probe.py"

            EXPECTED_PROBE_TOOL_SHA256 = "564eab26b7cf683bd531fec08bf6539a1384d9ef370961b9484335726c7c2351"
            EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
            EXPECTED_EXECUTION_CONTRACT_SHA256 = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
            EXPECTED_CONFIG_HASH = "a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf"
            EXPECTED_GRAPH_SIGNATURE = "1c7597b170fd8604056ab7787fd2880d6e84f3025962fc4b6c8fb3e3faf8e1e8"
            EXPECTED_FEATURE_SIGNATURE = "752538062fa2e40d9615c650c529e9f4117f33a030b74d281b5b21fa573731fc"
            EXPECTED_PRIOR_SIGNATURE = "ea888bab9c003af9b279719025da7c39f90537179411326c2c3119fc8c3f0824"
            EXPECTED_DATASET_SPLIT_SIGNATURE = "fer2013_train28709_val3589_test3589"

            EXPECTED_ARTIFACTS = {
                "checkpoint": {
                    "basename": "best_val_accuracy.keras",
                    "sha256": "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
                },
                "checkpoint_metadata": {
                    "basename": "best_val_accuracy.metadata.json",
                    "sha256": "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
                },
                "resolved_config": {
                    "basename": "resolved_config.json",
                    "sha256": "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
                },
            }

            EXPECTED_CHECKPOINT_EPOCH = 31
            EXPECTED_SEED = 42
            EXPECTED_VALIDATION_SAMPLES = 3589
            C0_REFERENCE = {
                "accuracy": 0.6319308999721371,
                "macro_f1": 0.5938407974340496,
                "loss": 1.1538367092081931,
            }
            C0_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}

            KAGGLE_INPUT_ROOT = Path("/kaggle/input")
            FER_DATASET_MOUNT = Path("/kaggle/input/datasets/doduyquynii/fer13-split")
            PRIOR_DATASET_MOUNT = Path("/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue")
            CACHE_DATASET_MOUNT = Path("/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records")
            FER_SPLIT_ROOT = FER_DATASET_MOUNT / "fer13-split"
            FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"
            PRIOR_ROOT = PRIOR_DATASET_MOUNT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
            PRIOR_VAL_DIR = PRIOR_ROOT / "val"
            GRAPH_CACHE_ROOT = CACHE_DATASET_MOUNT
            GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"
            CACHE_COMPLETE_PATH = GRAPH_CACHE_ROOT / "CACHE_COMPLETE.json"
            PUBLIC_SAMPLE_INPUT_ROOTS = (
                FER_DATASET_MOUNT,
                PRIOR_DATASET_MOUNT,
                CACHE_DATASET_MOUNT,
            )

            WORKING = Path("/kaggle/working")
            PROJECT_PATH = WORKING / "FER2013_Graph"
            TF_PACKAGE_PATH = PROJECT_PATH / TF_PACKAGE_RELATIVE
            PROBE_TOOL_PATH = PROJECT_PATH / PROBE_TOOL_RELATIVE
            RUN_ROOT = WORKING / "tf_step6_fixed_topology_prior_sensitivity"
            PROBE_OUTPUT_ROOT = RUN_ROOT / "probe"
            ADAPTER_METADATA_ROOT = RUN_ROOT / "adapter_metadata"
            REPORT_PATH = WORKING / "tf_step6_fixed_topology_prior_sensitivity.md"
            ARCHIVE_PATH = WORKING / "tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip"

            EVAL_BATCH_SIZE = 32
            GRAPH_WORKERS = 2
            GRAPH_CACHE_SIZE = 64
            TESTED_TENSORFLOW = "2.18.1"
            TESTED_KERAS = "3.15.0"
            """
        ),
        markdown("## 2. Clone exact source and verify the reviewed harness\n"),
        code(
            """
            import hashlib
            import importlib
            import importlib.metadata
            import json
            import os
            import platform
            import re
            import shutil
            import subprocess
            import sys

            def run_checked(command, cwd=None, env=None, capture=False):
                actual = [str(item) for item in command]
                display = [
                    re.sub(r"(https://x-access-token:)[^@]+@", r"\\1***@", item)
                    for item in actual
                ]
                print("$", " ".join(display))
                result = subprocess.run(
                    actual,
                    cwd=cwd,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE if capture else None,
                    stderr=subprocess.STDOUT if capture else None,
                )
                if result.returncode:
                    if capture and result.stdout:
                        print("\\n".join(result.stdout.splitlines()[-100:]))
                    raise RuntimeError(f"Command failed with exit code {result.returncode}")
                return result.stdout if capture else ""

            def sha256(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            if PROJECT_PATH.exists():
                if PROJECT_PATH.parent != WORKING or PROJECT_PATH.name != "FER2013_Graph":
                    raise RuntimeError(f"Unsafe clone cleanup target: {PROJECT_PATH}")
                shutil.rmtree(PROJECT_PATH)

            clone_url = REPO_URL
            try:
                from kaggle_secrets import UserSecretsClient
                github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
            except Exception:
                github_token = None
            if github_token and clone_url.startswith("https://github.com/"):
                clone_url = clone_url.replace(
                    "https://", f"https://x-access-token:{github_token}@"
                )
            run_checked([
                "git", "clone", "--branch", REPO_BRANCH, "--single-branch",
                clone_url, PROJECT_PATH,
            ])
            run_checked(["git", "checkout", "--detach", EXPECTED_COMMIT], cwd=PROJECT_PATH)
            actual_commit = run_checked(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_PATH, capture=True
            ).strip()
            dirty = run_checked(
                ["git", "status", "--porcelain"], cwd=PROJECT_PATH, capture=True
            ).strip()
            if actual_commit != EXPECTED_COMMIT or dirty:
                raise RuntimeError(
                    f"Source lock failed: commit={actual_commit}, dirty={bool(dirty)}"
                )

            for required in (
                TF_PACKAGE_PATH / "pyproject.toml",
                TF_PACKAGE_PATH / "CHECKSUMS.sha256",
                TF_PACKAGE_PATH / "package_manifest.json",
                PROBE_TOOL_PATH,
            ):
                if not required.is_file():
                    raise FileNotFoundError(required)
            run_checked(
                [sys.executable, "-B", TF_PACKAGE_PATH / "tools/verify_checksums.py"],
                cwd=TF_PACKAGE_PATH,
            )
            package_manifest = json.loads(
                (TF_PACKAGE_PATH / "package_manifest.json").read_text(encoding="utf-8")
            )
            if package_manifest.get("scientific_payload_sha256") != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
                raise RuntimeError("Frozen scientific payload drift")
            if package_manifest.get("execution_contract_sha256") != EXPECTED_EXECUTION_CONTRACT_SHA256:
                raise RuntimeError("Execution contract drift")
            if sha256(PROBE_TOOL_PATH) != EXPECTED_PROBE_TOOL_SHA256:
                raise RuntimeError("Reviewed Step 5 probe tool drift")
            print(json.dumps({
                "commit": actual_commit,
                "probe_tool_sha256": sha256(PROBE_TOOL_PATH),
                "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
                "execution_contract_sha256": package_manifest["execution_contract_sha256"],
            }, indent=2))
            """
        ),
        markdown("## 3. Require Kaggle T4 and the registered software environment\n"),
        code(
            """
            def distribution_version(name):
                try:
                    return importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    return None

            if "tensorflow" in sys.modules or "keras" in sys.modules:
                raise RuntimeError("TensorFlow/Keras imported before environment bootstrap")
            gpu_names = [
                line.strip()
                for line in run_checked(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture=True,
                ).splitlines()
                if line.strip()
            ]
            if not gpu_names or not all("T4" in name for name in gpu_names):
                raise RuntimeError(f"Issue #11 requires Kaggle GPU T4, got {gpu_names}")
            run_checked(["nvidia-smi"])

            if (
                distribution_version("tensorflow") != TESTED_TENSORFLOW
                or distribution_version("keras") != TESTED_KERAS
            ):
                run_checked([
                    sys.executable, "-m", "pip", "install", "-q",
                    "--no-warn-conflicts", "-r",
                    TF_PACKAGE_PATH / "requirements-kaggle.txt",
                ])
                importlib.invalidate_caches()
            run_checked([
                sys.executable, "-m", "pip", "install", "-q", "-e",
                TF_PACKAGE_PATH, "--no-deps",
            ])
            environment_text = run_checked([
                sys.executable, "-B", "-c",
                "import json,tensorflow as tf; print(json.dumps({"
                "'tensorflow':tf.__version__,'keras':tf.keras.__version__,"
                "'gpus':[d.name for d in tf.config.list_physical_devices('GPU')],"
                "'build':tf.sysconfig.get_build_info()},default=str))",
            ], cwd=TF_PACKAGE_PATH, capture=True)
            environment_payload = json.loads(environment_text.strip().splitlines()[-1])
            if (
                environment_payload["tensorflow"] != TESTED_TENSORFLOW
                or environment_payload["keras"] != TESTED_KERAS
                or not environment_payload["gpus"]
            ):
                raise RuntimeError(f"Registered software/GPU mismatch: {environment_payload}")
            environment_payload.update({
                "python": sys.version,
                "platform": platform.platform(),
                "nvidia_gpu_names": gpu_names,
            })
            print(json.dumps(environment_payload, indent=2, default=str))
            """
        ),
        markdown("## 4. Locate Issue #7 artifacts by exact SHA-256 only\n"),
        code(
            """
            def is_within(path, root):
                try:
                    Path(path).resolve().relative_to(Path(root).resolve())
                    return True
                except ValueError:
                    return False

            def locate_unique_sha_artifact(spec):
                matches = []
                for current_root, directory_names, file_names in os.walk(
                    KAGGLE_INPUT_ROOT, topdown=True
                ):
                    current_path = Path(current_root)
                    if any(is_within(current_path, root) for root in PUBLIC_SAMPLE_INPUT_ROOTS):
                        directory_names[:] = []
                        continue
                    directory_names[:] = [
                        name
                        for name in directory_names
                        if not any(
                            is_within(current_path / name, root)
                            for root in PUBLIC_SAMPLE_INPUT_ROOTS
                        )
                    ]
                    if spec["basename"] not in file_names:
                        continue
                    candidate = current_path / spec["basename"]
                    if any(is_within(candidate, root) for root in PUBLIC_SAMPLE_INPUT_ROOTS):
                        raise RuntimeError(f"Artifact candidate overlaps a sample input: {candidate}")
                    actual_sha = sha256(candidate)
                    if actual_sha == spec["sha256"]:
                        matches.append(candidate)
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one SHA-matched {spec['basename']}; found {matches}"
                    )
                resolved = matches[0].resolve()
                if not is_within(resolved, KAGGLE_INPUT_ROOT):
                    raise RuntimeError(f"Artifact is not a read-only Kaggle Input: {resolved}")
                return resolved

            located_artifacts = {
                name: locate_unique_sha_artifact(spec)
                for name, spec in EXPECTED_ARTIFACTS.items()
            }
            for name, path in located_artifacts.items():
                if sha256(path) != EXPECTED_ARTIFACTS[name]["sha256"]:
                    raise RuntimeError(f"Artifact hash changed during discovery: {name}")

            checkpoint_metadata = json.loads(
                located_artifacts["checkpoint_metadata"].read_text(encoding="utf-8")
            )
            expected_metadata_identity = {
                "epoch": EXPECTED_CHECKPOINT_EPOCH,
                "seed": EXPECTED_SEED,
                "config_hash": EXPECTED_CONFIG_HASH,
                "package_checksum": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
                "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
                "graph_signature": EXPECTED_GRAPH_SIGNATURE,
                "feature_signature": EXPECTED_FEATURE_SIGNATURE,
                "prior_signature": EXPECTED_PRIOR_SIGNATURE,
                "dataset_split_signature": EXPECTED_DATASET_SPLIT_SIGNATURE,
            }
            metadata_mismatches = {
                key: {"expected": expected, "actual": checkpoint_metadata.get(key)}
                for key, expected in expected_metadata_identity.items()
                if checkpoint_metadata.get(key) != expected
            }
            if metadata_mismatches:
                raise RuntimeError(f"Issue #7 checkpoint metadata drift: {metadata_mismatches}")
            metadata_validation_metrics = checkpoint_metadata.get("validation_metrics") or {}
            reference_metric_mismatches = {
                key: {"expected": expected, "actual": metadata_validation_metrics.get(key)}
                for key, expected in C0_REFERENCE.items()
                if metadata_validation_metrics.get(key) != expected
            }
            if reference_metric_mismatches:
                raise RuntimeError(
                    f"Issue #7 reference validation metrics drift: {reference_metric_mismatches}"
                )
            if located_artifacts["checkpoint"].name != "best_val_accuracy.keras":
                raise RuntimeError("Only the persisted epoch-31 best-validation-accuracy checkpoint is allowed")
            print(json.dumps({
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in located_artifacts.items()
            }, indent=2))
            """
        ),
        markdown("## 5. Validation-only asset and freshness gate\n"),
        code(
            """
            import csv

            if not FER_VAL_CSV.is_file():
                raise FileNotFoundError(FER_VAL_CSV)
            with FER_VAL_CSV.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                fer_header = [item.strip().lower() for item in next(reader)]
                fer_val_rows = sum(1 for _ in reader)
            if (
                "emotion" not in fer_header
                or "pixels" not in fer_header
                or fer_val_rows != EXPECTED_VALIDATION_SAMPLES
            ):
                raise RuntimeError(
                    f"Validation FER mismatch: rows={fer_val_rows}, header={fer_header}"
                )

            prior_val_files = sorted(PRIOR_VAL_DIR.glob("*.npz"))
            if len(prior_val_files) != EXPECTED_VALIDATION_SAMPLES:
                raise RuntimeError(
                    f"Validation prior count mismatch: {len(prior_val_files)}"
                )
            cache_val_index_path = GRAPH_CACHE_VAL_DIR / "index.json"
            cache_val_index = json.loads(cache_val_index_path.read_text(encoding="utf-8"))
            if (
                cache_val_index.get("schema_version") != "tf_clean_graph_cache_v2_records"
                or int(cache_val_index.get("sample_count", -1)) != EXPECTED_VALIDATION_SAMPLES
            ):
                raise RuntimeError("Validation clean-cache index mismatch")
            cache_val_shards = []
            for shard in cache_val_index.get("shards", []):
                shard_path = GRAPH_CACHE_VAL_DIR / shard["path"]
                if not shard_path.is_file():
                    raise FileNotFoundError(shard_path)
                cache_val_shards.append(str(shard_path))
            cache_complete = json.loads(CACHE_COMPLETE_PATH.read_text(encoding="utf-8"))
            if cache_complete.get("schema_version") != "tf_clean_graph_cache_v2_records":
                raise RuntimeError("Shared cache completion marker schema mismatch")

            shared_prior_metadata = {}
            for name in ("prior_schema.json", "part_names.json", "micro_anchor_names.json"):
                path = PRIOR_ROOT / name
                if path.is_file():
                    shared_prior_metadata[name] = {
                        "path": str(path), "sha256": sha256(path)
                    }

            for fresh_path in (RUN_ROOT, REPORT_PATH, ARCHIVE_PATH):
                if fresh_path.exists():
                    raise FileExistsError(f"Registered output must be fresh: {fresh_path}")
            ADAPTER_METADATA_ROOT.mkdir(parents=True, exist_ok=False)
            artifact_hashes_before = {
                name: sha256(path) for name, path in located_artifacts.items()
            }
            validation_asset_evidence = {
                "sample_split": "val",
                "fer_val_csv": {
                    "path": str(FER_VAL_CSV),
                    "sha256": sha256(FER_VAL_CSV),
                    "rows": fer_val_rows,
                },
                "prior_val_dir": str(PRIOR_VAL_DIR),
                "prior_val_files": len(prior_val_files),
                "shared_prior_metadata": shared_prior_metadata,
                "cache_val_index": {
                    "path": str(cache_val_index_path),
                    "sha256": sha256(cache_val_index_path),
                    "samples": int(cache_val_index["sample_count"]),
                    "shards": len(cache_val_shards),
                },
                "shared_cache_completion_marker": {
                    "path": str(CACHE_COMPLETE_PATH),
                    "sha256": sha256(CACHE_COMPLETE_PATH),
                    "scope": "non-sample aggregate metadata required by the frozen loader; may summarize other splits",
                },
                "test_csv_accessed": False,
                "test_prior_records_accessed": False,
                "test_cache_index_accessed": False,
                "test_cache_shards_accessed": False,
                "test_labels_accessed": False,
                "test_predictions_accessed": False,
                "test_metrics_accessed": False,
                "test_inference_run": False,
            }
            pre_run_manifest = {
                "issue": 11,
                "base_commit": actual_commit,
                "probe_tool_sha256": sha256(PROBE_TOOL_PATH),
                "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
                "artifact_hashes": artifact_hashes_before,
                "checkpoint_metadata_identity": expected_metadata_identity,
                "environment": environment_payload,
                "validation_assets": validation_asset_evidence,
                "registered_full_run": True,
                "limit_val_batches": None,
                "training": False,
                "test_access": False,
            }
            (ADAPTER_METADATA_ROOT / "pre_run_manifest.json").write_text(
                json.dumps(pre_run_manifest, indent=2, sort_keys=True, default=str) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps(validation_asset_evidence, indent=2))
            print("READY_FOR_ISSUE11_REGISTERED_PROBE")
            """
        ),
        markdown(
            """
            ## 6. One registered full validation probe

            **Pre-run review gate:** do not execute this cell until the adapter PR is
            approved by the research lead. It invokes the reviewed Step 5 tool exactly
            once and deliberately supplies no bounded-batch argument.
            """
        ),
        code(
            """
            probe_command = [
                sys.executable,
                "-B",
                PROBE_TOOL_PATH,
                "--checkpoint",
                located_artifacts["checkpoint"],
                "--checkpoint-metadata",
                located_artifacts["checkpoint_metadata"],
                "--resolved-config",
                located_artifacts["resolved_config"],
                "--prior-root",
                PRIOR_ROOT,
                "--clean-graph-cache-dir",
                GRAPH_CACHE_ROOT,
                "--output-root",
                PROBE_OUTPUT_ROOT,
                "--eval-batch-size",
                str(EVAL_BATCH_SIZE),
                "--graph-workers",
                str(GRAPH_WORKERS),
                "--graph-cache-size",
                str(GRAPH_CACHE_SIZE),
            ]
            if any(str(argument).startswith("--limit-") for argument in probe_command):
                raise RuntimeError("Registered Issue #11 probe must be an unbounded validation run")
            if probe_command.count(PROBE_TOOL_PATH) != 1:
                raise RuntimeError("Reviewed probe tool must be invoked exactly once")
            run_checked(probe_command, cwd=TF_PACKAGE_PATH)

            artifact_hashes_after = {
                name: sha256(path) for name, path in located_artifacts.items()
            }
            if artifact_hashes_after != artifact_hashes_before:
                raise RuntimeError("Issue #7 checkpoint/config artifacts changed during probe")
            print(json.dumps({
                "artifact_hashes_before": artifact_hashes_before,
                "artifact_hashes_after": artifact_hashes_after,
                "unchanged": True,
            }, indent=2))
            """
        ),
        markdown("## 7. Apply only the preregistered C0 gate and diagnostics\n"),
        code(
            """
            import numpy as np

            CONDITION_C0 = "official"
            CONDITION_C1 = "direct_part_path_zero_fixed_graph"
            CONDITION_C2 = "semantic_prior_zero_fixed_graph"
            CONDITIONS = (CONDITION_C0, CONDITION_C1, CONDITION_C2)

            probe_manifest = json.loads(
                (PROBE_OUTPUT_ROOT / "probe_manifest.json").read_text(encoding="utf-8")
            )
            integrity = json.loads(
                (PROBE_OUTPUT_ROOT / "intervention_integrity.json").read_text(encoding="utf-8")
            )
            condition_metrics = {}
            for condition in CONDITIONS:
                payload = json.loads(
                    (PROBE_OUTPUT_ROOT / f"validation_metrics_{condition}.json").read_text(
                        encoding="utf-8"
                    )
                )
                if payload.get("split") != "val" or payload.get("condition") != condition:
                    raise RuntimeError(f"Condition output identity mismatch: {condition}")
                condition_metrics[condition] = payload["metrics"]

            paired_path = PROBE_OUTPUT_ROOT / "paired_validation_predictions.csv"
            with paired_path.open("r", encoding="utf-8", newline="") as stream:
                paired_rows = list(csv.DictReader(stream))
            sample_count = len(paired_rows)
            if sample_count != EXPECTED_VALIDATION_SAMPLES:
                raise RuntimeError(
                    f"Expected {EXPECTED_VALIDATION_SAMPLES} paired samples, got {sample_count}"
                )
            labels = np.asarray([int(row["label"]) for row in paired_rows], dtype=np.int64)
            predictions = {
                condition: np.asarray(
                    [int(row[f"{condition}_prediction"]) for row in paired_rows],
                    dtype=np.int64,
                )
                for condition in CONDITIONS
            }

            c0_metrics = condition_metrics[CONDITION_C0]
            c0_gate_checks = {
                "sample_count_exact": sample_count == EXPECTED_VALIDATION_SAMPLES,
                "accuracy_within_tolerance": abs(
                    c0_metrics["accuracy"] - C0_REFERENCE["accuracy"]
                ) <= C0_TOLERANCE["accuracy"],
                "macro_f1_within_tolerance": abs(
                    c0_metrics["macro_f1"] - C0_REFERENCE["macro_f1"]
                ) <= C0_TOLERANCE["macro_f1"],
                "loss_within_tolerance": abs(
                    c0_metrics["loss"] - C0_REFERENCE["loss"]
                ) <= C0_TOLERANCE["loss"],
            }
            c0_gate_pass = all(c0_gate_checks.values())

            def paired_transition(intervention):
                c0_correct = predictions[CONDITION_C0] == labels
                intervention_correct = predictions[intervention] == labels
                disagreement = predictions[CONDITION_C0] != predictions[intervention]
                categories = {
                    "c0_correct_to_intervention_incorrect": c0_correct & ~intervention_correct,
                    "c0_incorrect_to_intervention_correct": ~c0_correct & intervention_correct,
                    "unchanged_correct": c0_correct & intervention_correct,
                    "unchanged_incorrect": ~c0_correct & ~intervention_correct,
                }
                return {
                    "prediction_disagreement_count": int(disagreement.sum()),
                    "prediction_disagreement_rate": float(disagreement.mean()),
                    "correctness_transitions": {
                        name: {"count": int(mask.sum()), "rate": float(mask.mean())}
                        for name, mask in categories.items()
                    },
                }

            if not c0_gate_pass:
                diagnostic_label = "INVALID_REFERENCE_REPRODUCTION"
                raw_diagnostics = None
                interpreted_diagnostics = None
                negative_c2_note = None
            else:
                c1 = condition_metrics[CONDITION_C1]
                c2 = condition_metrics[CONDITION_C2]
                raw_diagnostics = {
                    "delta_f1_c1_pp": 100.0 * (c0_metrics["macro_f1"] - c1["macro_f1"]),
                    "delta_f1_c2_pp": 100.0 * (c0_metrics["macro_f1"] - c2["macro_f1"]),
                    "delta_f1_c2_minus_c1_pp": 100.0 * (c1["macro_f1"] - c2["macro_f1"]),
                    "accuracy_change_pp_c0_to_c1": 100.0 * (c1["accuracy"] - c0_metrics["accuracy"]),
                    "accuracy_change_pp_c0_to_c2": 100.0 * (c2["accuracy"] - c0_metrics["accuracy"]),
                    "validation_loss": {
                        condition: condition_metrics[condition]["loss"]
                        for condition in CONDITIONS
                    },
                    "paired": {
                        CONDITION_C1: paired_transition(CONDITION_C1),
                        CONDITION_C2: paired_transition(CONDITION_C2),
                    },
                    "per_class_f1": {
                        condition: condition_metrics[condition]["per_class_f1"]
                        for condition in CONDITIONS
                    },
                    "per_class_c0_minus_c1_f1_pp": (
                        100.0 * (
                            np.asarray(c0_metrics["per_class_f1"])
                            - np.asarray(c1["per_class_f1"])
                        )
                    ).tolist(),
                    "per_class_c0_minus_c2_f1_pp": (
                        100.0 * (
                            np.asarray(c0_metrics["per_class_f1"])
                            - np.asarray(c2["per_class_f1"])
                        )
                    ).tolist(),
                }
                delta_f1_c2 = raw_diagnostics["delta_f1_c2_pp"]
                if delta_f1_c2 >= 10.0:
                    diagnostic_label = "HIGH_EXPLICIT_PRIOR_SENSITIVITY"
                elif delta_f1_c2 >= 5.0:
                    diagnostic_label = "MODERATE_EXPLICIT_PRIOR_SENSITIVITY"
                else:
                    diagnostic_label = "LOW_EXPLICIT_PRIOR_SENSITIVITY"
                negative_c2_note = (
                    "The C2 ablated condition improved macro-F1."
                    if delta_f1_c2 < 0.0
                    else None
                )
                interpreted_diagnostics = raw_diagnostics

            if (
                probe_manifest.get("sample_count") != EXPECTED_VALIDATION_SAMPLES
                or probe_manifest.get("split") != "val"
                or probe_manifest.get("limit_val_batches") is not None
                or not probe_manifest.get("checkpoint", {}).get("unchanged")
                or not probe_manifest.get("checkpoint", {}).get("model_weights_unchanged")
                or not integrity.get("checkpoint_unchanged")
                or not integrity.get("model_weights_unchanged")
                or not integrity.get("paired_original_batch_evaluation")
            ):
                raise RuntimeError("Probe manifest/integrity gate failed")

            final_evidence = {
                "issue": 11,
                "base_commit": actual_commit,
                "probe_tool_sha256": sha256(PROBE_TOOL_PATH),
                "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
                "artifact_hashes_before": artifact_hashes_before,
                "artifact_hashes_after": artifact_hashes_after,
                "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
                "seed": EXPECTED_SEED,
                "environment": environment_payload,
                "sample_count": sample_count,
                "batch_count": probe_manifest["batch_count"],
                "c0_reference": C0_REFERENCE,
                "c0_tolerance": C0_TOLERANCE,
                "c0_gate_checks": c0_gate_checks,
                "c0_gate_pass": c0_gate_pass,
                "condition_metrics": condition_metrics,
                "raw_diagnostics_preserved": raw_diagnostics,
                "interpreted_diagnostics": interpreted_diagnostics,
                "diagnostic_label": diagnostic_label,
                "negative_c2_note": negative_c2_note,
                "intervention_integrity": integrity,
                "topology_scaffold_limitation": (
                    "C2 removes explicit semantic prior/direct-part tensor content "
                    "conditional on the official MediaPipe-derived topology; it is not prior-free."
                ),
                "test_isolation": validation_asset_evidence,
                "training_performed": False,
            }
            (ADAPTER_METADATA_ROOT / "final_evidence.json").write_text(
                json.dumps(final_evidence, indent=2, sort_keys=True, default=str) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "c0_gate_pass": c0_gate_pass,
                "diagnostic_label": diagnostic_label,
                "sample_count": sample_count,
                "batch_count": probe_manifest["batch_count"],
            }, indent=2))
            """
        ),
        markdown("## 8. Write the compact Step 6 report\n"),
        code(
            """
            def metric_line(condition):
                metrics = condition_metrics[condition]
                return (
                    f"- `{condition}`: accuracy `{metrics['accuracy']}`, macro-F1 "
                    f"`{metrics['macro_f1']}`, loss `{metrics['loss']}`."
                )

            reported_metric_lines = [metric_line(CONDITION_C0)]
            if c0_gate_pass:
                reported_metric_lines.extend([
                    metric_line(CONDITION_C1),
                    metric_line(CONDITION_C2),
                ])
            report_lines = [
                "# TensorFlow Step 6 fixed-topology prior sensitivity",
                "",
                "## Provenance",
                "",
                f"- Issue: #11.",
                f"- Base commit: `{actual_commit}`.",
                f"- Probe tool SHA-256: `{sha256(PROBE_TOOL_PATH)}`.",
                f"- Frozen scientific payload SHA-256: `{package_manifest['scientific_payload_sha256']}`.",
                f"- Checkpoint SHA-256: `{artifact_hashes_after['checkpoint']}`.",
                f"- Checkpoint metadata SHA-256: `{artifact_hashes_after['checkpoint_metadata']}`.",
                f"- Resolved config SHA-256: `{artifact_hashes_after['resolved_config']}`.",
                f"- Checkpoint epoch/seed: `{EXPECTED_CHECKPOINT_EPOCH}` / `{EXPECTED_SEED}`.",
                f"- Runtime: Kaggle T4, TensorFlow `{environment_payload['tensorflow']}`, Keras `{environment_payload['keras']}`.",
                "",
                "## C0 reproduction gate",
                "",
                f"- Sample count: `{sample_count}`; required `{EXPECTED_VALIDATION_SAMPLES}`.",
                f"- Gate checks: `{json.dumps(c0_gate_checks, sort_keys=True)}`.",
                f"- Gate result: `{'PASS' if c0_gate_pass else 'FAIL'}`.",
                "",
                "## Validation metrics",
                "",
                *reported_metric_lines,
                "",
                "## Preregistered diagnostics",
                "",
                f"- Diagnostic label: `{diagnostic_label}`.",
            ]
            if c0_gate_pass:
                report_lines.extend([
                    f"- Delta F1 C1: `{raw_diagnostics['delta_f1_c1_pp']}` pp.",
                    f"- Delta F1 C2: `{raw_diagnostics['delta_f1_c2_pp']}` pp.",
                    f"- Incremental C2-minus-C1 effect: `{raw_diagnostics['delta_f1_c2_minus_c1_pp']}` pp.",
                    f"- Accuracy change C0-to-C1: `{raw_diagnostics['accuracy_change_pp_c0_to_c1']}` pp.",
                    f"- Accuracy change C0-to-C2: `{raw_diagnostics['accuracy_change_pp_c0_to_c2']}` pp.",
                    f"- Paired outcomes: `{json.dumps(raw_diagnostics['paired'], sort_keys=True)}`.",
                    f"- Per-class F1: `{json.dumps(raw_diagnostics['per_class_f1'], sort_keys=True)}`.",
                    f"- Per-class C0-minus-C1 F1 deltas: `{raw_diagnostics['per_class_c0_minus_c1_f1_pp']}` pp.",
                    f"- Per-class C0-minus-C2 F1 deltas: `{raw_diagnostics['per_class_c0_minus_c2_f1_pp']}` pp.",
                ])
                if negative_c2_note is not None:
                    report_lines.append(f"- {negative_c2_note}")
            else:
                report_lines.append(
                    "- Scientific interpretation stopped because the C0 reference gate failed; C1/C2 raw output files are preserved without reporting derived outcomes."
                )
            report_lines.extend([
                "",
                "## Integrity and boundaries",
                "",
                f"- Batch count: `{probe_manifest['batch_count']}`; paired same-batch evaluation: `{integrity['paired_original_batch_evaluation']}`.",
                f"- Checkpoint file unchanged: `{integrity['checkpoint_unchanged']}`; model weights unchanged: `{integrity['model_weights_unchanged']}`.",
                "- Test CSV/prior records/cache index/cache shards/labels/predictions/metrics/inference accessed: `false`.",
                "- Shared cache-root `CACHE_COMPLETE.json` accessed: `true`, solely as required non-sample aggregate loader metadata; it may summarize other splits.",
                "- Training, fine-tuning, optimizer steps, raw-prior corruption, graph rebuild, and topology changes: `false`.",
                "- C2 measures explicit semantic-prior/direct-part sensitivity conditional on the official MediaPipe-derived scaffold. It is not MediaPipe removal and is not a prior-free graph.",
                "- The diagnostic label is a preregistered sensitivity heuristic, not causal proof of the Issue #7 generalization gap and not model selection.",
            ])
            REPORT_PATH.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")
            print("report:", REPORT_PATH)
            """
        ),
        markdown("## 9. Archive compact validation-only evidence\n"),
        code(
            """
            import zipfile

            if ARCHIVE_PATH.exists():
                raise FileExistsError(f"Archive path must be fresh: {ARCHIVE_PATH}")
            archive_sources = [RUN_ROOT, REPORT_PATH]
            with zipfile.ZipFile(
                ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for source in archive_sources:
                    if source.is_dir():
                        for path in sorted(source.rglob("*")):
                            if path.is_file():
                                archive.write(path, path.relative_to(WORKING))
                    else:
                        archive.write(source, source.relative_to(WORKING))
            with zipfile.ZipFile(ARCHIVE_PATH) as archive:
                archived_names = archive.namelist()
            forbidden_archive_names = [
                name
                for name in archived_names
                if name.endswith(".keras")
                or Path(name).name in {"train.csv", "test.csv"}
                or "/train/" in f"/{name}"
                or "/test/" in f"/{name}"
                or Path(name).name.startswith("test_metrics_")
            ]
            if forbidden_archive_names:
                raise RuntimeError(
                    f"Compact archive contains forbidden artifacts: {forbidden_archive_names}"
                )
            print("report_path:", REPORT_PATH)
            print("archive_path:", ARCHIVE_PATH)
            print("archive_bytes:", ARCHIVE_PATH.stat().st_size)
            print("archive_sha256:", sha256(ARCHIVE_PATH))
            print("archive_files:", len(archived_names))
            """
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"issue11-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
