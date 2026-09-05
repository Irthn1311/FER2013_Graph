"""Thin Kaggle-compatible execution wrapper for preregistered Issue #40.

This module owns path/materialization/runtime provenance only.  All P0-P9
calculations, gates, thresholds, and decisions remain in the reviewed Step 13
probe and are invoked exactly once through its public ``main`` entry point.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence
import zipfile


STATUS = "STEP13_EXECUTION_PREPARATION_ONLY"
ISSUE_NUMBER = 40
ADAPTER_VERSION = "1.0.0"
SCIENTIFIC_BASE_COMMIT = "d90cce8c4d23f8f1c2958c76cda4ce9d8cae6608"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBE_RELATIVE_PATH = Path(
    "research/candidates/tf_learned_local_residual_slots/"
    "evaluate_remaining_prior_probe.py"
)
PROBE_PATH = REPOSITORY_ROOT / PROBE_RELATIVE_PATH
FROZEN_PACKAGE_ROOT = (
    REPOSITORY_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
)
FROZEN_PACKAGE_SRC = FROZEN_PACKAGE_ROOT / "src"
CANDIDATE_MODEL_PATH = (
    REPOSITORY_ROOT
    / "research/candidates/tf_learned_local_residual_slots/model.py"
)

EXPECTED_PROBE_SHA256 = (
    "cf68c47d428d0b569828d65028024fcc0713e963419ff5511be91b1377327118"
)
EXPECTED_CANDIDATE_MODEL_SHA256 = (
    "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca"
)
EXPECTED_STEP12E_ARCHIVE_SHA256 = (
    "f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c"
)
EXPECTED_WEIGHTS_SHA256 = (
    "a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74"
)
EXPECTED_METADATA_SHA256 = (
    "a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf"
)
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
)
EXPECTED_Q_SHA256 = (
    "54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)

REGISTERED_CONFIRMATION = "RUN_ISSUE40_VALIDATION_ONLY_P0_P9"
REGISTERED_ENVIRONMENT = {
    "python": "3.12.12",
    "tensorflow": "2.18.1",
    "keras": "3.15.0",
    "cuda": "12.5.1",
    "cudnn_major": "9",
    "gpu_count": 2,
    "gpu_name": "Tesla T4",
    "mixed_precision_policy": "mixed_float16",
    "xla": False,
    "memory_growth": True,
}
REGISTERED_RESOURCES = {
    "eval_batch_size": 32,
    "graph_workers": 2,
    "tf_data_prefetch": 2,
    "graph_cache_size": 64,
}
EXPECTED_FULL_VALIDATION_SAMPLES = 3_589
EXPECTED_CONDITIONS = (
    "official_candidate_manual_forward",
    "node_face_mask_zero_fixed_graph",
    "node_part_soft_channels_zero_fixed_graph",
    "node_distance_map_channels_zero_fixed_graph",
    "node_landmark_missing_flag_zero_fixed_graph",
    "edge_semantic_channels_zero_fixed_graph",
    "context_direct_part_soft_neutralized",
    "readout_direct_part_soft_neutralized",
    "readout_validity_off",
    "all_explicit_semantic_prior_zero_fixed_topology_anchor",
)

ARCHIVE_MEMBERS = {
    "checkpoint": "run/checkpoints/best_val_accuracy.keras",
    "checkpoint_weights": "run/checkpoints/best_val_accuracy.weights.h5",
    "checkpoint_metadata": "run/checkpoints/best_val_accuracy.metadata.json",
    "resolved_config": "run/resolved_config.json",
}
EXPECTED_MEMBER_SHA256 = {
    "checkpoint": EXPECTED_CHECKPOINT_SHA256,
    "checkpoint_weights": EXPECTED_WEIGHTS_SHA256,
    "checkpoint_metadata": EXPECTED_METADATA_SHA256,
    "resolved_config": EXPECTED_RESOLVED_CONFIG_SHA256,
}
EXTRACTED_FILENAMES = {
    key: Path(member).name for key, member in ARCHIVE_MEMBERS.items()
}


class Step13ExecutionAdapterError(RuntimeError):
    """Fail-closed Issue #40 execution-preparation error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


def _run_git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise Step13ExecutionAdapterError(
            f"Git provenance command failed: {arguments}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _literal_string_constants(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    values: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            values[target.id] = value
    return values


def validate_probe_contract_constants(constants: Mapping[str, str]) -> None:
    expected = {
        "EXPECTED_STEP12E_ARCHIVE_SHA256": EXPECTED_STEP12E_ARCHIVE_SHA256,
        "EXPECTED_CHECKPOINT_SHA256": EXPECTED_CHECKPOINT_SHA256,
        "EXPECTED_WEIGHTS_SHA256": EXPECTED_WEIGHTS_SHA256,
        "EXPECTED_METADATA_SHA256": EXPECTED_METADATA_SHA256,
        "EXPECTED_RESOLVED_CONFIG_SHA256": EXPECTED_RESOLVED_CONFIG_SHA256,
        "EXPECTED_Q_SHA256": EXPECTED_Q_SHA256,
        "EXPECTED_SCIENTIFIC_PAYLOAD_SHA256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "EXPECTED_EXECUTION_CONTRACT_SHA256": EXPECTED_EXECUTION_CONTRACT_SHA256,
    }
    drift = {
        name: {"expected": value, "actual": constants.get(name)}
        for name, value in expected.items()
        if constants.get(name) != value
    }
    if drift:
        raise Step13ExecutionAdapterError(f"Reviewed probe contract drift: {drift}")


def validate_reviewed_probe_source(
    probe_path: str | Path, candidate_path: str | Path
) -> dict[str, str]:
    probe = Path(probe_path)
    candidate = Path(candidate_path)
    for path in (probe, candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    probe_sha = sha256_file(probe)
    candidate_sha = sha256_file(candidate)
    if probe_sha != EXPECTED_PROBE_SHA256:
        raise Step13ExecutionAdapterError("Reviewed Step-13 probe SHA drift")
    if candidate_sha != EXPECTED_CANDIDATE_MODEL_SHA256:
        raise Step13ExecutionAdapterError("Reviewed candidate model SHA drift")
    constants = _literal_string_constants(probe.read_text(encoding="utf-8"))
    validate_probe_contract_constants(constants)
    return {"probe_sha256": probe_sha, "candidate_model_sha256": candidate_sha}


def validate_scientific_sources(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    probe_path = root / PROBE_RELATIVE_PATH
    candidate_path = (
        root / "research/candidates/tf_learned_local_residual_slots/model.py"
    )
    frozen_root = root / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
    for path in (probe_path, candidate_path, frozen_root):
        if not path.exists():
            raise FileNotFoundError(path)

    head = _run_git(root, "rev-parse", "HEAD")
    _run_git(root, "merge-base", "--is-ancestor", SCIENTIFIC_BASE_COMMIT, head)
    if _run_git(root, "status", "--porcelain"):
        raise Step13ExecutionAdapterError("Execution checkout is not clean")

    reviewed_sources = validate_reviewed_probe_source(probe_path, candidate_path)

    if str(frozen_root / "src") not in sys.path:
        sys.path.insert(0, str(frozen_root / "src"))
    from lap_gnn_tf.signatures import scientific_payload_checksum

    payload_sha = scientific_payload_checksum(frozen_root)
    if payload_sha != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise Step13ExecutionAdapterError("Frozen scientific payload drift")
    checksum_result = subprocess.run(
        [sys.executable, str(frozen_root / "tools/verify_checksums.py")],
        cwd=frozen_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    checksum_output = checksum_result.stdout.strip()
    if checksum_result.returncode or not checksum_output.startswith("PASS checked="):
        raise Step13ExecutionAdapterError(
            f"Frozen package checksum verification failed: {checksum_output}"
        )
    return {
        "execution_head": head,
        "scientific_base_commit": SCIENTIFIC_BASE_COMMIT,
        "probe_path": str(probe_path),
        **reviewed_sources,
        "scientific_payload_sha256": payload_sha,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
        "worktree_clean": True,
        "frozen_checksum_verification": checksum_output,
    }


def validate_fresh_output_paths(output_root: str | Path, archive_path: str | Path) -> None:
    output = Path(output_root)
    archive = Path(archive_path)
    if output.exists():
        raise Step13ExecutionAdapterError(
            f"Fresh output root already exists; refusing stale run: {output}"
        )
    if archive.exists():
        raise Step13ExecutionAdapterError(
            f"Fresh archive path already exists; refusing overwrite: {archive}"
        )
    try:
        archive.resolve().relative_to(output.resolve())
    except ValueError:
        pass
    else:
        raise Step13ExecutionAdapterError("Archive path must be outside output root")


def materialize_reviewed_archive(
    source: str | Path, destination: str | Path
) -> tuple[Path, dict[str, Any]]:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = sha256_file(source_path)
    if source_path.name.lower().endswith(".b64"):
        try:
            decoded = base64.b64decode(source_path.read_bytes(), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise Step13ExecutionAdapterError("Malformed Step-12E Base64 transport") from exc
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            raise Step13ExecutionAdapterError("Materialized archive already exists")
        destination_path.write_bytes(decoded)
        archive_path = destination_path
        transport = "base64"
    else:
        archive_path = source_path
        transport = "direct_zip"
    archive_sha = sha256_file(archive_path)
    if archive_sha != EXPECTED_STEP12E_ARCHIVE_SHA256:
        if transport == "base64":
            archive_path.unlink(missing_ok=True)
        raise Step13ExecutionAdapterError("Reviewed Step-12E archive SHA drift")
    return archive_path, {
        "source_path": str(source_path.resolve()),
        "source_transport": transport,
        "source_transport_sha256": source_sha,
        "materialized_archive_path": str(archive_path.resolve()),
        "materialized_archive_sha256": archive_sha,
    }


def extract_locked_artifacts(
    archive_path: str | Path, destination: str | Path
) -> dict[str, Path]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=False)
    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for label, member in ARCHIVE_MEMBERS.items():
                try:
                    source = archive.open(member)
                except KeyError as exc:
                    raise Step13ExecutionAdapterError(
                        f"Reviewed archive missing required member: {member}"
                    ) from exc
                target = destination_path / EXTRACTED_FILENAMES[label]
                with source, target.open("wb") as destination_handle:
                    shutil.copyfileobj(source, destination_handle, length=1024 * 1024)
                actual = sha256_file(target)
                if actual != EXPECTED_MEMBER_SHA256[label]:
                    raise Step13ExecutionAdapterError(
                        f"Locked extracted artifact SHA drift: {label}"
                    )
                extracted[label] = target
    except zipfile.BadZipFile as exc:
        raise Step13ExecutionAdapterError("Reviewed Step-12E archive is not a ZIP") from exc
    return extracted


def build_probe_arguments(
    *,
    archive_path: str | Path,
    artifacts: Mapping[str, Path],
    prior_root: str | Path,
    clean_graph_cache_dir: str | Path,
    probe_output_root: str | Path,
) -> list[str]:
    required = set(ARCHIVE_MEMBERS)
    if set(artifacts) != required:
        raise Step13ExecutionAdapterError("Exact extracted artifact inventory required")
    return [
        "--step12e-archive",
        str(Path(archive_path).resolve()),
        "--checkpoint",
        str(Path(artifacts["checkpoint"]).resolve()),
        "--checkpoint-weights",
        str(Path(artifacts["checkpoint_weights"]).resolve()),
        "--checkpoint-metadata",
        str(Path(artifacts["checkpoint_metadata"]).resolve()),
        "--resolved-config",
        str(Path(artifacts["resolved_config"]).resolve()),
        "--prior-root",
        str(Path(prior_root).resolve()),
        "--clean-graph-cache-dir",
        str(Path(clean_graph_cache_dir).resolve()),
        "--output-root",
        str(Path(probe_output_root).resolve()),
        "--eval-batch-size",
        "32",
        "--graph-workers",
        "2",
        "--graph-cache-size",
        "64",
    ]


def reconstructible_command(probe_path: str | Path, arguments: Sequence[str]) -> list[str]:
    return [sys.executable, str(Path(probe_path).resolve()), *map(str, arguments)]


def validate_runtime_identity(observed: Mapping[str, Any]) -> None:
    drift = {
        name: {"expected": expected, "actual": observed.get(name)}
        for name, expected in REGISTERED_ENVIRONMENT.items()
        if observed.get(name) != expected
    }
    if drift:
        raise Step13ExecutionAdapterError(f"Registered runtime drift: {drift}")


def configure_registered_runtime() -> dict[str, Any]:
    import keras
    import tensorflow as tf

    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = list(tf.config.list_physical_devices("GPU"))
    gpu_names = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise Step13ExecutionAdapterError(
                "GPU memory growth must be configured before initialization"
            ) from exc
        details = tf.config.experimental.get_device_details(gpu)
        gpu_names.append(str(details.get("device_name", gpu.name)))
    build = tf.sysconfig.get_build_info()
    observed = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "cuda": str(build.get("cuda_version")),
        "cudnn_major": str(build.get("cudnn_version")).split(".")[0],
        "gpu_count": len(gpus),
        "gpu_name": (
            REGISTERED_ENVIRONMENT["gpu_name"]
            if gpu_names
            and all(REGISTERED_ENVIRONMENT["gpu_name"] in name for name in gpu_names)
            else gpu_names
        ),
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        "xla": bool(tf.config.optimizer.get_jit()),
        "memory_growth": bool(gpus)
        and all(tf.config.experimental.get_memory_growth(gpu) for gpu in gpus),
        "gpu_device_names": gpu_names,
    }
    validate_runtime_identity(observed)
    return observed


def _load_reviewed_probe(probe_path: Path):
    spec = importlib.util.spec_from_file_location("_issue40_reviewed_step13_probe", probe_path)
    if spec is None or spec.loader is None:
        raise Step13ExecutionAdapterError("Unable to load reviewed Step-13 probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_reviewed_probe(probe_path: Path, arguments: Sequence[str]) -> int:
    module = _load_reviewed_probe(probe_path)
    return int(module.main(list(arguments)))


def validate_successful_probe_output(probe_output_root: str | Path) -> dict[str, Any]:
    manifest_path = Path(probe_output_root) / "probe_manifest.json"
    if not manifest_path.is_file():
        raise Step13ExecutionAdapterError("Successful probe lacks probe_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Step13ExecutionAdapterError("Probe manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise Step13ExecutionAdapterError("Probe manifest must be a JSON object")
    gates = manifest.get("registered_gates_and_decision") or {}
    conditions = tuple(manifest.get("condition_order") or ())
    checks = {
        "status": manifest.get("status") == "VALID_REGISTERED_REMAINING_PRIOR_DECOMPOSITION",
        "samples": manifest.get("sample_count") == EXPECTED_FULL_VALIDATION_SAMPLES,
        "conditions": conditions == EXPECTED_CONDITIONS,
        "gate_a": (gates.get("gate_a_native_vs_p0") or {}).get("status") == "PASS",
        "gate_b": (gates.get("gate_b_checkpoint_metrics") or {}).get("status") == "PASS",
        "gate_c": (gates.get("gate_c_checkpoint_identity") or {}).get("status") == "PASS",
        "training": manifest.get("training") is False,
        "optimizer": manifest.get("optimizer_updates") is False,
        "test": manifest.get("test_access") is False,
        "full_run": manifest.get("limit_val_batches") is None,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise Step13ExecutionAdapterError(f"Probe completion evidence invalid: {failures}")
    return manifest


def _is_forbidden_archive_member(name: str) -> bool:
    path = Path(name)
    lowered = [part.lower() for part in path.parts]
    return (
        any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in lowered)
        or path.suffix.lower() in {".keras", ".h5"}
        or path.name.lower() in {"train.csv", "test.csv"}
    )


def publish_compact_archive(output_root: str | Path, archive_path: str | Path) -> list[str]:
    root = Path(output_root)
    destination = Path(archive_path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    members = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "inputs" in path.relative_to(root).parts:
            continue
        name = path.relative_to(root).as_posix()
        if _is_forbidden_archive_member(name):
            raise Step13ExecutionAdapterError(f"Forbidden archive member: {name}")
        members.append((path, name))
    if not members:
        raise Step13ExecutionAdapterError("No compact evidence available to archive")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in members:
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if archive.testzip() is not None or names != [name for _path, name in members]:
                raise Step13ExecutionAdapterError("Compact evidence ZIP verification failed")
        os.replace(temporary, destination)
        return names
    finally:
        temporary.unlink(missing_ok=True)


def run_registered_execution(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm != REGISTERED_CONFIRMATION:
        raise Step13ExecutionAdapterError("Registered execution confirmation mismatch")
    validate_fresh_output_paths(args.output_root, args.archive_path)
    source_identity = validate_scientific_sources(args.repository_root)
    for path, label in (
        (args.prior_root, "validation prior root"),
        (args.clean_graph_cache_dir, "validation graph cache"),
    ):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    inputs_root = output_root / "inputs"
    materialized_archive, archive_identity = materialize_reviewed_archive(
        args.step12e_source, inputs_root / "step12e_reviewed.zip"
    )
    artifacts = extract_locked_artifacts(materialized_archive, inputs_root / "artifacts")
    probe_output = output_root / "probe_output"
    probe_arguments = build_probe_arguments(
        archive_path=materialized_archive,
        artifacts=artifacts,
        prior_root=args.prior_root,
        clean_graph_cache_dir=args.clean_graph_cache_dir,
        probe_output_root=probe_output,
    )
    command = reconstructible_command(
        Path(args.repository_root) / PROBE_RELATIVE_PATH, probe_arguments
    )

    runtime = configure_registered_runtime()
    pre_run = {
        "schema_version": 1,
        "status": STATUS,
        "issue": ISSUE_NUMBER,
        "adapter_version": ADAPTER_VERSION,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(__file__),
        "source_identity": source_identity,
        "archive_identity": archive_identity,
        "extracted_artifact_sha256": {
            label: sha256_file(path) for label, path in artifacts.items()
        },
        "runtime": runtime,
        "resources": {
            **REGISTERED_RESOURCES,
            "tf_data_prefetch_effective": "NOT_APPLICABLE_TO_REVIEWED_ITER_EPOCH",
        },
        "split": "val",
        "expected_validation_samples": EXPECTED_FULL_VALIDATION_SAMPLES,
        "probe_command": command,
        "training": False,
        "optimizer_updates": False,
        "test_access": False,
    }
    atomic_json(output_root / "pre_run_manifest.json", pre_run)

    log_path = output_root / "probe_execution.log"
    status = "TECHNICAL_OR_GATE_FAILURE"
    return_code = 1
    error_text = None
    scientific_result_valid = False
    probe_manifest = None
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                return_code = invoke_reviewed_probe(
                    Path(args.repository_root) / PROBE_RELATIVE_PATH,
                    probe_arguments,
                )
                if return_code != 0:
                    raise Step13ExecutionAdapterError(
                        f"Reviewed probe returned {return_code}"
                    )
                probe_manifest = validate_successful_probe_output(probe_output)
                status = "COMPLETE"
                scientific_result_valid = True
            except BaseException as exc:  # preserve evidence for any probe failure
                error_text = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=log)

    wrapper = {
        "schema_version": 1,
        "status": status,
        "probe_return_code": return_code,
        "error": error_text,
        "scientific_result_valid": scientific_result_valid,
        "scientific_interpretation": (
            probe_manifest.get("scientific_interpretation")
            if scientific_result_valid and probe_manifest
            else None
        ),
        "training": False,
        "optimizer_updates": False,
        "test_access": False,
        "probe_command": command,
    }
    atomic_json(output_root / "wrapper_execution.json", wrapper)
    report = output_root / "execution_report.md"
    report.write_text(
        "# TF Step 13 registered execution\n\n"
        f"Status: `{status}`\n\n"
        f"Scientific result valid: `{str(scientific_result_valid).lower()}`\n\n"
        "Validation only; no training, optimizer update, or test access.\n",
        encoding="utf-8",
        newline="\n",
    )
    members = publish_compact_archive(output_root, args.archive_path)
    wrapper["archive_path"] = str(Path(args.archive_path).resolve())
    wrapper["archive_sha256"] = sha256_file(args.archive_path)
    wrapper["archive_members"] = len(members)
    return wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reviewed Issue #40 validation-only Step-13 probe once."
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--step12e-source", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--clean-graph-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument("--confirm", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_registered_execution(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
