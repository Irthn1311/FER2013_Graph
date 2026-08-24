"""Build the unexecuted Issue #15 Gate-A forensic Kaggle adapter."""

from __future__ import annotations

import copy
import importlib.util
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISSUE11_BUILDER = ROOT / "tools" / "build_issue11_kaggle_adapter.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue15-gate-a-forensic.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = _source(text)


def _replace_required(cell: dict, old: str, new: str) -> None:
    source = "".join(cell["source"])
    if old not in source:
        raise RuntimeError(f"Issue #15 forensic template drift; missing {old!r}")
    cell["source"] = source.replace(old, new).splitlines(keepends=True)


def _load_issue11_builder():
    spec = importlib.util.spec_from_file_location(
        "_issue11_forensic_adapter_template", ISSUE11_BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter template: {ISSUE11_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_notebook() -> dict:
    """Return a deterministic notebook whose code cells are all unexecuted."""
    notebook = copy.deepcopy(_load_issue11_builder().build_notebook())
    cells = notebook["cells"]

    _set_source(
        cells[0],
        """
        # Issue #15: technical Gate-A forensic diagnostic

        This is a separate **validation-only technical diagnostic**, not a Step 8
        scientific decomposition run. It is intentionally unexecuted pending
        research-lead review. It runs native D0 twice and reviewed manual D0 twice
        for every original validation batch. It never executes D1-D5.

        Both registered Step 8 attempts remain **PRE-INTERVENTION TECHNICAL HARNESS
        FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE**. The second is the post-hotfix
        attempt. Neither produced a valid D1-D4 scientific outcome, and this notebook
        does not modify or rerun either attempt.

        Required Kaggle Inputs and resolved reads:

        - FER split input: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`;
          only `val.csv` is opened.
        - MediaPipe prior input:
          `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
          only `val/*.npz` and required shared schema/name metadata are used.
        - Clean graph cache input:
          `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; only
          `val/index.json`, its validation shards, and shared root
          `CACHE_COMPLETE.json` are read. Cache completeness is fail-closed and graph
          rebuild is forbidden.
        - Issue #7 artifact input: attach one separate read-only Kaggle Input with
          `best_val_accuracy.keras`, `best_val_accuracy.metadata.json`, and
          `resolved_config.json`. Its mount name is not trusted; each artifact is
          located by unique basename plus exact SHA-256 outside the sample inputs.

        Internet is required only to clone the exact reviewed forensic execution
        commit and, if needed, install the registered dependencies. Dataset, cache,
        checkpoint, metadata, and config assets are offline Kaggle Inputs.

        Compact outputs, produced even when the diagnostic subprocess fails:

        - report: `/kaggle/working/tf_step8_gate_a_forensic.md`
        - archive: `/kaggle/working/tf_step8_gate_a_forensic_kaggle_t4.zip`

        The unchanged Gate-A tolerances are reference markers only. The tool collects
        every validation batch and does not stop on a tolerance exceedance. TensorFlow
        op determinism is deliberately not enabled in this first forensic pass.
        """,
    )
    _set_source(cells[1], "## 1. Exact technical and frozen identities\n")
    _set_source(
        cells[2],
        """
        from pathlib import Path

        REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
        EXPECTED_SCIENTIFIC_BASE_COMMIT = "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
        EXPECTED_HOTFIX_ANCESTOR_COMMIT = "a1b1d279bb9ec388f1d93ad86196e423dc750ad1"
        EXPECTED_EXECUTION_COMMIT = "3cae1f6c78048cd6cd518d87cd0a5429d72f01e1"
        TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
        FORENSIC_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_gate_a_forensic_probe.py"
        STEP7_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
        STEP6_SUPPORT_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_prior_probe.py"

        EXPECTED_FORENSIC_TOOL_SHA256 = "30c00fd6985810533cc09be05f66b64f7da5a794903aef493b9839b461eac7c0"
        EXPECTED_STEP7_TOOL_SHA256 = "fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3"
        EXPECTED_STEP6_SUPPORT_SHA256 = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
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
        GATE_A_REFERENCE_TOLERANCE = {
            "prediction_agreement": 1.0,
            "max_abs_logit_difference": 1e-5,
            "max_abs_probability_difference": 1e-6,
        }
        COMPARISONS = (
            "native_1_vs_native_2",
            "manual_1_vs_manual_2",
            "native_1_vs_manual_1",
            "native_2_vs_manual_2",
        )

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
            FER_DATASET_MOUNT, PRIOR_DATASET_MOUNT, CACHE_DATASET_MOUNT,
        )

        WORKING = Path("/kaggle/working")
        PROJECT_PATH = WORKING / "FER2013_Graph"
        TF_PACKAGE_PATH = PROJECT_PATH / TF_PACKAGE_RELATIVE
        FORENSIC_TOOL_PATH = PROJECT_PATH / FORENSIC_TOOL_RELATIVE
        STEP7_TOOL_PATH = PROJECT_PATH / STEP7_TOOL_RELATIVE
        STEP6_SUPPORT_PATH = PROJECT_PATH / STEP6_SUPPORT_RELATIVE
        RUN_ROOT = WORKING / "tf_step8_gate_a_forensic"
        FORENSIC_OUTPUT_ROOT = RUN_ROOT / "forensic"
        ADAPTER_METADATA_ROOT = RUN_ROOT / "adapter_metadata"
        SUBPROCESS_LOG_PATH = RUN_ROOT / "forensic_subprocess.log"
        REPORT_PATH = WORKING / "tf_step8_gate_a_forensic.md"
        ARCHIVE_PATH = WORKING / "tf_step8_gate_a_forensic_kaggle_t4.zip"

        EVAL_BATCH_SIZE = 32
        GRAPH_WORKERS = 2
        GRAPH_CACHE_SIZE = 64
        TESTED_TENSORFLOW = "2.18.1"
        TESTED_KERAS = "3.15.0"
        """,
    )
    _set_source(cells[3], "## 2. Clone exact forensic source and verify all locks\n")
    _set_source(
        cells[4],
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

        def run_checked(command, cwd=None, capture=False):
            actual = [str(item) for item in command]
            display = [
                re.sub(r"(https://x-access-token:)[^@]+@", r"\\1***@", item)
                for item in actual
            ]
            print("$", " ".join(display))
            result = subprocess.run(
                actual,
                cwd=cwd,
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
        run_checked(["git", "clone", "--no-checkout", clone_url, PROJECT_PATH])
        run_checked(
            ["git", "checkout", "--detach", EXPECTED_EXECUTION_COMMIT],
            cwd=PROJECT_PATH,
        )
        actual_commit = run_checked(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_PATH, capture=True
        ).strip()
        dirty = run_checked(
            ["git", "status", "--porcelain"], cwd=PROJECT_PATH, capture=True
        ).strip()
        if actual_commit != EXPECTED_EXECUTION_COMMIT or dirty:
            raise RuntimeError(
                f"Source lock failed: commit={actual_commit}, dirty={bool(dirty)}"
            )
        for required_ancestor in (
            EXPECTED_SCIENTIFIC_BASE_COMMIT,
            EXPECTED_HOTFIX_ANCESTOR_COMMIT,
        ):
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", required_ancestor, "HEAD"],
                cwd=PROJECT_PATH,
            )
            if ancestry.returncode != 0:
                raise RuntimeError(
                    f"Required execution ancestor missing: {required_ancestor}"
                )

        for required in (
            TF_PACKAGE_PATH / "pyproject.toml",
            TF_PACKAGE_PATH / "CHECKSUMS.sha256",
            TF_PACKAGE_PATH / "package_manifest.json",
            FORENSIC_TOOL_PATH,
            STEP7_TOOL_PATH,
            STEP6_SUPPORT_PATH,
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
        expected_tool_hashes = {
            FORENSIC_TOOL_PATH: EXPECTED_FORENSIC_TOOL_SHA256,
            STEP7_TOOL_PATH: EXPECTED_STEP7_TOOL_SHA256,
            STEP6_SUPPORT_PATH: EXPECTED_STEP6_SUPPORT_SHA256,
        }
        for path, expected in expected_tool_hashes.items():
            if sha256(path) != expected:
                raise RuntimeError(f"Reviewed tool drift: {path}")
        print(json.dumps({
            "scientific_base_commit": EXPECTED_SCIENTIFIC_BASE_COMMIT,
            "required_hotfix_ancestor": EXPECTED_HOTFIX_ANCESTOR_COMMIT,
            "execution_commit": actual_commit,
            "forensic_tool_sha256": sha256(FORENSIC_TOOL_PATH),
            "step7_tool_sha256": sha256(STEP7_TOOL_PATH),
            "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),
            "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
        }, indent=2))
        """,
    )
    _replace_required(cells[6], "Issue #11 requires Kaggle GPU T4", "Issue #15 forensic requires Kaggle GPU T4")

    reference_block = '''metadata_validation_metrics = checkpoint_metadata.get("validation_metrics") or {}
reference_metric_mismatches = {
    key: {"expected": expected, "actual": metadata_validation_metrics.get(key)}
    for key, expected in C0_REFERENCE.items()
    if metadata_validation_metrics.get(key) != expected
}
if reference_metric_mismatches:
    raise RuntimeError(
        f"Issue #7 reference validation metrics drift: {reference_metric_mismatches}"
    )
'''
    _replace_required(cells[8], reference_block, "")
    _set_source(cells[7], "## 4. Locate exact Issue #7 artifacts by SHA-256 only\n")

    _set_source(cells[9], "## 5. Preflight validation-only inputs and fresh outputs\n")
    _replace_required(cells[10], '    "issue": 11,', '    "issue": 15,')
    _replace_required(
        cells[10],
        '    "probe_tool_sha256": sha256(PROBE_TOOL_PATH),',
        '    "forensic_tool_sha256": sha256(FORENSIC_TOOL_PATH),\n    "step7_tool_sha256": sha256(STEP7_TOOL_PATH),\n    "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),',
    )
    _replace_required(
        cells[10],
        '    "registered_full_run": True,',
        '    "full_validation_forensic": True,\n    "scientific_decomposition_run": False,\n    "intervention_conditions_executed": [],\n    "graph_rebuild_allowed": False,',
    )
    _replace_required(
        cells[10],
        "READY_FOR_ISSUE11_REGISTERED_PROBE",
        "READY_FOR_ISSUE15_GATE_A_FORENSIC_REVIEW",
    )

    _set_source(
        cells[11],
        """
        ## 6. One full-validation technical forensic subprocess

        **Pre-run review gate:** do not execute this cell until this draft technical
        PR is approved. It invokes only the forensic tool, without a bounded-batch
        argument. The wrapper archives and verifies partial JSON and the subprocess
        log on any non-zero exit, then completes normally with a scientifically
        fail-closed `TECHNICAL_FORENSIC_FAILURE` status so Kaggle can publish it.
        """,
    )
    _set_source(
        cells[12],
        """
        import zipfile

        def create_forensic_archive():
            if ARCHIVE_PATH.exists():
                ARCHIVE_PATH.unlink()
            with zipfile.ZipFile(
                ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for source in (RUN_ROOT, REPORT_PATH):
                    if source.is_dir():
                        for path in sorted(source.rglob("*")):
                            if path.is_file():
                                archive.write(path, path.relative_to(WORKING))
                    elif source.is_file():
                        archive.write(source, source.relative_to(WORKING))
            with zipfile.ZipFile(ARCHIVE_PATH) as archive:
                names = archive.namelist()
            forbidden = [
                name for name in names
                if name.endswith(".keras")
                or Path(name).name in {"train.csv", "test.csv"}
                or "/train/" in f"/{name}"
                or "/test/" in f"/{name}"
                or Path(name).name.startswith("test_metrics_")
            ]
            if forbidden:
                raise RuntimeError(f"Forensic archive contains forbidden artifacts: {forbidden}")
            return names

        def verify_failure_archive(archived_names):
            archived = set(archived_names)
            required_paths = [
                SUBPROCESS_LOG_PATH,
                ADAPTER_METADATA_ROOT / "wrapper_execution.json",
                REPORT_PATH,
            ]
            conditional_paths = list(
                sorted((FORENSIC_OUTPUT_ROOT / "batches").glob("batch_*.json"))
            )
            conditional_paths.extend([
                FORENSIC_OUTPUT_ROOT / "progress.json",
                FORENSIC_OUTPUT_ROOT / "forensic_manifest.json",
                FORENSIC_OUTPUT_ROOT / "diagnostic_failure.json",
            ])
            for path in required_paths:
                if not path.is_file():
                    raise RuntimeError(f"Missing required failure evidence: {path}")
                member = path.relative_to(WORKING).as_posix()
                if member not in archived:
                    raise RuntimeError(f"Failure archive missing required evidence: {member}")
            for path in conditional_paths:
                if path.is_file() and path.relative_to(WORKING).as_posix() not in archived:
                    raise RuntimeError(f"Failure archive omitted partial evidence: {path}")
            final_evidence_path = ADAPTER_METADATA_ROOT / "final_evidence.json"
            if (
                final_evidence_path.exists()
                or final_evidence_path.relative_to(WORKING).as_posix() in archived
            ):
                raise RuntimeError("Failure path must not fabricate success final_evidence")

        def write_wrapper_report(status, returncode, error_text=None):
            report_lines = [
                "# TensorFlow Step 8 Gate-A technical forensic",
                "",
                "## Status",
                "",
                f"- Wrapper status: `{status}`.",
                f"- Diagnostic subprocess return code: `{returncode}`.",
                "- This is not a scientific decomposition result; D1-D5 were not executed.",
                "- Both registered Step 8 attempts remain pre-intervention technical harness failures with no valid D1-D4 outcome.",
                "",
                "## Provenance",
                "",
                f"- Scientific base: `{EXPECTED_SCIENTIFIC_BASE_COMMIT}`.",
                f"- Required hotfix ancestor: `{EXPECTED_HOTFIX_ANCESTOR_COMMIT}`.",
                f"- Execution commit: `{EXPECTED_EXECUTION_COMMIT}`.",
                f"- Forensic tool SHA-256: `{EXPECTED_FORENSIC_TOOL_SHA256}`.",
                f"- Step-7 tool SHA-256: `{EXPECTED_STEP7_TOOL_SHA256}`.",
                f"- Step-6 support SHA-256: `{EXPECTED_STEP6_SUPPORT_SHA256}`.",
                f"- Frozen scientific payload SHA-256: `{EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}`.",
                "",
                "## Evidence boundary",
                "",
                "- Validation-only native/manual D0 repeatability evidence is under `tf_step8_gate_a_forensic/forensic/`.",
                "- Incremental batch JSON and progress are preserved even if a later batch fails.",
                "- TensorFlow op determinism was not enabled by this tool.",
                "- Scientific interpretation: `null`; scientific decomposition run: `false`.",
                "- Intervention conditions executed: `[]`; D1-D5 were not executed.",
            ]
            if error_text:
                report_lines.extend(["", "## Wrapper error", "", f"- `{error_text}`"])
            REPORT_PATH.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")

        def run_forensic_with_failure_archive(command, cwd):
            wrapper_error = None
            returncode = None
            result = None
            try:
                with SUBPROCESS_LOG_PATH.open("w", encoding="utf-8", newline="\\n") as log:
                    result = subprocess.run(
                        [str(item) for item in command],
                        cwd=cwd,
                        text=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                returncode = int(result.returncode)
                if returncode != 0:
                    wrapper_error = f"Forensic subprocess exited non-zero: {returncode}"
            except BaseException as exc:
                wrapper_error = f"Forensic subprocess wrapper error: {exc}"
            try:
                artifact_hashes_after = {
                    name: sha256(path) for name, path in located_artifacts.items()
                }
                artifacts_unchanged = artifact_hashes_after == artifact_hashes_before
            except BaseException as exc:
                artifact_hashes_after = {}
                artifacts_unchanged = False
                if wrapper_error is None:
                    wrapper_error = f"Post-subprocess artifact verification error: {exc}"
            if not artifacts_unchanged and wrapper_error is None:
                wrapper_error = "Issue #7 checkpoint/config artifacts changed"
            wrapper_payload = {
                "status": "COMPLETE" if wrapper_error is None else "TECHNICAL_FORENSIC_FAILURE",
                "returncode": returncode,
                "error": wrapper_error,
                "artifact_hashes_before": artifact_hashes_before,
                "artifact_hashes_after": artifact_hashes_after,
                "artifacts_unchanged": artifacts_unchanged,
                "scientific_decomposition_run": False,
                "intervention_conditions_executed": [],
                "scientific_interpretation": None,
                "training": False,
                "test_access": False,
            }
            final_evidence_path = ADAPTER_METADATA_ROOT / "final_evidence.json"
            if wrapper_error is not None and final_evidence_path.exists():
                final_evidence_path.unlink()
            (ADAPTER_METADATA_ROOT / "wrapper_execution.json").write_text(
                json.dumps(wrapper_payload, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            write_wrapper_report(wrapper_payload["status"], returncode, wrapper_error)
            archived_names = create_forensic_archive()
            if wrapper_error is not None:
                verify_failure_archive(archived_names)
            return {
                "result": result,
                "wrapper_execution": wrapper_payload,
                "archive_names": archived_names,
            }

        forensic_command = [
            sys.executable,
            "-B",
            FORENSIC_TOOL_PATH,
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
            FORENSIC_OUTPUT_ROOT,
            "--eval-batch-size",
            str(EVAL_BATCH_SIZE),
            "--graph-workers",
            str(GRAPH_WORKERS),
            "--graph-cache-size",
            str(GRAPH_CACHE_SIZE),
        ]
        if any(str(argument).startswith("--limit-") for argument in forensic_command):
            raise RuntimeError("Full-validation forensic must not be bounded")
        if forensic_command.count(FORENSIC_TOOL_PATH) != 1:
            raise RuntimeError("Forensic tool must be invoked exactly once")
        if STEP7_TOOL_PATH in forensic_command or STEP6_SUPPORT_PATH in forensic_command:
            raise RuntimeError("Scientific/support tools must not be invoked directly")
        wrapper_outcome = run_forensic_with_failure_archive(
            forensic_command, TF_PACKAGE_PATH
        )
        forensic_result = wrapper_outcome["result"]
        wrapper_execution = wrapper_outcome["wrapper_execution"]
        initial_archive_names = wrapper_outcome["archive_names"]
        """,
    )

    _set_source(cells[13], "## 7. Verify status-appropriate technical evidence\n")
    _set_source(
        cells[14],
        """
        wrapper_execution_path = ADAPTER_METADATA_ROOT / "wrapper_execution.json"
        wrapper_execution = json.loads(wrapper_execution_path.read_text(encoding="utf-8"))
        wrapper_status = wrapper_execution.get("status")
        final_evidence_path = ADAPTER_METADATA_ROOT / "final_evidence.json"
        if wrapper_status == "COMPLETE":
            forensic_manifest = json.loads(
                (FORENSIC_OUTPUT_ROOT / "forensic_manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (FORENSIC_OUTPUT_ROOT / "progress.json").read_text(encoding="utf-8")
            )
            immutability = json.loads(
                (FORENSIC_OUTPUT_ROOT / "immutability.json").read_text(encoding="utf-8")
            )
            dtype_manifest = json.loads(
                (FORENSIC_OUTPUT_ROOT / "dtype_manifest.json").read_text(encoding="utf-8")
            )
            batch_paths = sorted((FORENSIC_OUTPUT_ROOT / "batches").glob("batch_*.json"))
            if (
                forensic_manifest.get("status") != "COMPLETE"
                or progress.get("status") != "COMPLETE"
                or progress.get("completed_sample_count") != EXPECTED_VALIDATION_SAMPLES
                or len(batch_paths) != progress.get("completed_batch_count")
            ):
                raise RuntimeError("Incomplete full-validation forensic evidence")
            if (
                forensic_manifest.get("intervention_conditions_executed") != []
                or forensic_manifest.get("scientific_decomposition_run") is not False
                or forensic_manifest.get("gate_a_tolerances_are_diagnostic_only") is not True
                or forensic_manifest.get("stop_on_reference_exceedance") is not False
            ):
                raise RuntimeError("Forensic/non-scientific execution boundary drift")
            if forensic_manifest.get("gate_a_reference_tolerances") != GATE_A_REFERENCE_TOLERANCE:
                raise RuntimeError("Gate-A diagnostic reference tolerance drift")
            if not immutability.get("checkpoint_unchanged") or not immutability.get("model_weights_unchanged"):
                raise RuntimeError("Checkpoint/model immutability evidence failed")
            if any(bool(value) for value in forensic_manifest.get("test_access", {}).values()):
                raise RuntimeError("Test isolation evidence failed")
            if any(bool(value) for value in forensic_manifest.get("training_access", {}).values()):
                raise RuntimeError("Training isolation evidence failed")
            required_roles = {
                "outer_lap_gnn", "encoder", "gnn_container", "gnn_layer",
                "part_global_context", "readout", "classifier",
            }
            actual_roles = {item.get("role") for item in dtype_manifest.get("layers", [])}
            if not required_roles.issubset(actual_roles):
                raise RuntimeError(f"Dtype manifest missing roles: {required_roles - actual_roles}")
            artifact_hashes_after = {
                name: sha256(path) for name, path in located_artifacts.items()
            }
            if artifact_hashes_after != artifact_hashes_before:
                raise RuntimeError("Issue #7 artifact identity changed")
            final_evidence = {
                "issue": 15,
                "diagnostic": "gate_a_native_manual_repeatability_forensic",
                "scientific_base_commit": EXPECTED_SCIENTIFIC_BASE_COMMIT,
                "execution_commit": actual_commit,
                "required_hotfix_ancestor": EXPECTED_HOTFIX_ANCESTOR_COMMIT,
                "forensic_tool_sha256": sha256(FORENSIC_TOOL_PATH),
                "step7_tool_sha256": sha256(STEP7_TOOL_PATH),
                "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),
                "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
                "artifact_hashes_before": artifact_hashes_before,
                "artifact_hashes_after": artifact_hashes_after,
                "environment": environment_payload,
                "progress": progress,
                "immutability": immutability,
                "dtype_manifest": dtype_manifest,
                "scientific_decomposition_run": False,
                "intervention_conditions_executed": [],
                "scientific_interpretation": None,
            }
            final_evidence_path.write_text(
                json.dumps(final_evidence, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "status": wrapper_status,
                "batch_count": progress["completed_batch_count"],
                "sample_count": progress["completed_sample_count"],
                "comparisons": progress["comparisons"],
            }, indent=2))
        elif wrapper_status == "TECHNICAL_FORENSIC_FAILURE":
            if (
                wrapper_execution.get("scientific_interpretation") is not None
                or wrapper_execution.get("scientific_decomposition_run") is not False
                or wrapper_execution.get("intervention_conditions_executed") != []
            ):
                raise RuntimeError("Technical failure scientific boundary drift")
            if final_evidence_path.exists():
                raise RuntimeError("Technical failure must not fabricate success final_evidence")
            verify_failure_archive(initial_archive_names)
            print(json.dumps({
                "status": "TECHNICAL_FORENSIC_FAILURE",
                "returncode": wrapper_execution.get("returncode"),
                "error": wrapper_execution.get("error"),
                "archive_path": str(ARCHIVE_PATH),
                "scientific_interpretation": None,
                "scientific_decomposition_run": False,
                "intervention_conditions_executed": [],
            }, indent=2))
        else:
            raise RuntimeError(f"Unknown forensic wrapper status: {wrapper_status}")
        """,
    )

    _set_source(cells[15], "## 8. Write a success report or retain the failure report\n")
    _set_source(
        cells[16],
        """
        if wrapper_status == "COMPLETE":
            report_lines = [
                "# TensorFlow Step 8 Gate-A technical forensic",
                "",
                "## Provenance",
                "",
                f"- Issue: #15 technical diagnostic.",
                f"- Scientific base: `{EXPECTED_SCIENTIFIC_BASE_COMMIT}`.",
                f"- Required hotfix ancestor: `{EXPECTED_HOTFIX_ANCESTOR_COMMIT}`.",
                f"- Execution commit: `{EXPECTED_EXECUTION_COMMIT}`.",
                f"- Forensic tool SHA-256: `{EXPECTED_FORENSIC_TOOL_SHA256}`.",
                f"- Step-7 tool SHA-256: `{EXPECTED_STEP7_TOOL_SHA256}`.",
                f"- Step-6 support SHA-256: `{EXPECTED_STEP6_SUPPORT_SHA256}`.",
                f"- Frozen scientific payload: `{EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}`.",
                "",
                "## Technical measurements",
                "",
                f"- Validation samples/batches: `{progress['completed_sample_count']}` / `{progress['completed_batch_count']}`.",
                f"- Gate-A reference tolerances: `{json.dumps(GATE_A_REFERENCE_TOLERANCE, sort_keys=True)}`.",
                f"- Aggregate envelopes: `{json.dumps(progress['comparisons'], sort_keys=True)}`.",
                "- Per-batch sample-ID hashes, node/edge counts, four comparison records, and boundary dtypes are preserved in the archive.",
                "",
                "## Integrity and boundaries",
                "",
                f"- Checkpoint unchanged: `{immutability['checkpoint_unchanged']}`; model weights unchanged: `{immutability['model_weights_unchanged']}`.",
                "- Source batches are checked after each of the four forwards.",
                "- Validation only; no training, optimizer, test access, graph rebuild, or op-determinism enablement.",
                "- D1-D5 were not executed. This is not a Step 8 scientific outcome.",
                "- Diagnosis between GPU repeatability and remaining native/manual semantics is reserved for research-lead review of the measured envelopes.",
            ]
            REPORT_PATH.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")
        else:
            failure_report = REPORT_PATH.read_text(encoding="utf-8")
            if "TECHNICAL_FORENSIC_FAILURE" not in failure_report:
                raise RuntimeError("Technical failure report status missing")
        """,
    )
    _set_source(cells[17], "## 9. Refresh and verify the compact archive\n")
    _set_source(
        cells[18],
        """
        if wrapper_status == "COMPLETE":
            archived_names = create_forensic_archive()
            required_archive_suffixes = {
                "forensic/forensic_manifest.json",
                "forensic/progress.json",
                "forensic/dtype_manifest.json",
                "forensic/immutability.json",
                "adapter_metadata/pre_run_manifest.json",
                "adapter_metadata/wrapper_execution.json",
                "adapter_metadata/final_evidence.json",
                "forensic_subprocess.log",
            }
            for suffix in required_archive_suffixes:
                if not any(name.endswith(suffix) for name in archived_names):
                    raise RuntimeError(f"Forensic archive missing required evidence: {suffix}")
        else:
            archived_names = initial_archive_names
            verify_failure_archive(archived_names)
        print("report_path:", REPORT_PATH)
        print("archive_path:", ARCHIVE_PATH)
        print("archive_bytes:", ARCHIVE_PATH.stat().st_size)
        print("archive_sha256:", sha256(ARCHIVE_PATH))
        print("archive_files:", len(archived_names))
        """,
    )

    for index, cell in enumerate(cells):
        cell["id"] = f"issue15-forensic-{index:02d}"
    return notebook


def main() -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
