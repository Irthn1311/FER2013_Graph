from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "tools/run_issue40_step13_execution.py"
PROBE_RELATIVE_PATH = Path(
    "research/candidates/tf_learned_local_residual_slots/"
    "evaluate_remaining_prior_probe.py"
)
BASE = "d90cce8c4d23f8f1c2958c76cda4ce9d8cae6608"
PARENT_PROBE_SHA = "cf68c47d428d0b569828d65028024fcc0713e963419ff5511be91b1377327118"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("issue40_step13_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def _probe_constants_source() -> str:
    values = {
        "EXPECTED_STEP12E_ARCHIVE_SHA256": adapter.EXPECTED_STEP12E_ARCHIVE_SHA256,
        "EXPECTED_CHECKPOINT_SHA256": adapter.EXPECTED_CHECKPOINT_SHA256,
        "EXPECTED_WEIGHTS_SHA256": adapter.EXPECTED_WEIGHTS_SHA256,
        "EXPECTED_METADATA_SHA256": adapter.EXPECTED_METADATA_SHA256,
        "EXPECTED_RESOLVED_CONFIG_SHA256": adapter.EXPECTED_RESOLVED_CONFIG_SHA256,
        "EXPECTED_Q_SHA256": adapter.EXPECTED_Q_SHA256,
        "EXPECTED_SCIENTIFIC_PAYLOAD_SHA256": (
            adapter.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
        ),
        "EXPECTED_EXECUTION_CONTRACT_SHA256": (
            adapter.EXPECTED_EXECUTION_CONTRACT_SHA256
        ),
    }
    return "\n".join(f"{name} = {value!r}" for name, value in values.items()) + "\n"


def _archive(tmp_path: Path, *, omit: str | None = None, corrupt: str | None = None):
    archive_path = tmp_path / "step12e.zip"
    payloads = {
        label: f"locked-{label}".encode("ascii")
        for label in adapter.ARCHIVE_MEMBERS
    }
    if corrupt is not None:
        payloads[corrupt] = b"corrupted"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for label, member in adapter.ARCHIVE_MEMBERS.items():
            if label != omit:
                archive.writestr(member, payloads[label])
    return archive_path, payloads


def _patch_archive_hashes(monkeypatch, archive_path: Path, payloads):
    monkeypatch.setattr(
        adapter, "EXPECTED_STEP12E_ARCHIVE_SHA256", adapter.sha256_file(archive_path)
    )
    for label, payload in payloads.items():
        monkeypatch.setitem(
            adapter.EXPECTED_MEMBER_SHA256,
            label,
            hashlib.sha256(payload).hexdigest(),
        )


def _required_cli(output: Path) -> list[str]:
    return [
        "--step12e-source",
        "source.zip.b64",
        "--prior-root",
        "prior",
        "--clean-graph-cache-dir",
        "cache",
        "--output-root",
        str(output),
        "--archive-path",
        str(output.with_suffix(".zip")),
        "--confirm",
        adapter.REGISTERED_CONFIRMATION,
    ]


def _valid_probe_manifest() -> dict:
    return {
        "status": "VALID_REGISTERED_REMAINING_PRIOR_DECOMPOSITION",
        "sample_count": adapter.EXPECTED_FULL_VALIDATION_SAMPLES,
        "condition_order": list(adapter.EXPECTED_CONDITIONS),
        "limit_val_batches": None,
        "registered_gates_and_decision": {
            "gate_a_native_vs_p0": {"status": "PASS"},
            "gate_b_checkpoint_metrics": {"status": "PASS"},
            "gate_c_checkpoint_identity": {"status": "PASS"},
        },
        "training": False,
        "optimizer_updates": False,
        "test_access": False,
        "scientific_interpretation": "SYNTHETIC_VALID_COMPLETION",
    }


def _patch_synthetic_registered_execution(monkeypatch, tmp_path, probe_main):
    source = tmp_path / "source.zip.b64"
    source.write_bytes(b"synthetic-transport")
    prior = tmp_path / "prior"
    cache = tmp_path / "cache"
    prior.mkdir()
    cache.mkdir()
    output = tmp_path / "run"
    archive = tmp_path / "run.zip"

    monkeypatch.setattr(
        adapter,
        "validate_scientific_sources",
        lambda _root: {
            "execution_head": "synthetic",
            "scientific_base_commit": adapter.SCIENTIFIC_BASE_COMMIT,
            "probe_sha256": adapter.EXPECTED_PROBE_SHA256,
            "scientific_payload_sha256": adapter.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        },
    )

    def materialize(_source, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-reviewed-archive")
        return path, {
            "source_path": str(source),
            "source_transport": "base64",
            "source_transport_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "materialized_archive_path": str(path),
            "materialized_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def extract(_archive, destination):
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=False)
        artifacts = {}
        for label, filename in adapter.EXTRACTED_FILENAMES.items():
            path = root / filename
            path.write_bytes(f"synthetic-{label}".encode("ascii"))
            artifacts[label] = path
        return artifacts

    monkeypatch.setattr(adapter, "materialize_reviewed_archive", materialize)
    monkeypatch.setattr(adapter, "extract_locked_artifacts", extract)
    monkeypatch.setattr(
        adapter,
        "configure_registered_runtime",
        lambda: dict(adapter.REGISTERED_ENVIRONMENT),
    )
    monkeypatch.setattr(adapter, "invoke_reviewed_probe", probe_main)
    cli = [
        "--repository-root",
        str(ROOT),
        "--step12e-source",
        str(source),
        "--prior-root",
        str(prior),
        "--clean-graph-cache-dir",
        str(cache),
        "--output-root",
        str(output),
        "--archive-path",
        str(archive),
        "--confirm",
        adapter.REGISTERED_CONFIRMATION,
    ]
    return cli, output, archive


def test_exact_issue40_source_runtime_and_artifact_locks():
    assert adapter.STATUS == "STEP13_EXECUTION_PREPARATION_ONLY"
    assert adapter.SCIENTIFIC_BASE_COMMIT == BASE
    assert adapter.EXPECTED_PROBE_SHA256 == PARENT_PROBE_SHA
    assert adapter.EXPECTED_STEP12E_ARCHIVE_SHA256 == (
        "f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451"
    )
    assert adapter.EXPECTED_CHECKPOINT_SHA256 == (
        "e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c"
    )
    assert adapter.EXPECTED_WEIGHTS_SHA256 == (
        "a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74"
    )
    assert adapter.EXPECTED_METADATA_SHA256 == (
        "a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf"
    )
    assert adapter.EXPECTED_RESOLVED_CONFIG_SHA256 == (
        "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
    )
    assert adapter.EXPECTED_Q_SHA256 == (
        "54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9"
    )
    assert adapter.REGISTERED_RESOURCES == {
        "eval_batch_size": 32,
        "graph_workers": 2,
        "tf_data_prefetch": 2,
        "graph_cache_size": 64,
    }
    assert adapter.REGISTERED_ENVIRONMENT["mixed_precision_policy"] == "mixed_float16"
    assert adapter.REGISTERED_ENVIRONMENT["xla"] is False
    assert adapter.REGISTERED_ENVIRONMENT["memory_growth"] is True
    assert adapter.INVALID_REGISTERED_EXECUTION_EXIT_CODE == 3


def test_wrong_probe_source_identity_fails_closed(monkeypatch, tmp_path):
    probe = tmp_path / "probe.py"
    candidate = tmp_path / "model.py"
    probe.write_text(_probe_constants_source(), encoding="utf-8", newline="\n")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(adapter, "EXPECTED_PROBE_SHA256", adapter.sha256_file(probe))
    monkeypatch.setattr(
        adapter, "EXPECTED_CANDIDATE_MODEL_SHA256", adapter.sha256_file(candidate)
    )
    adapter.validate_reviewed_probe_source(probe, candidate)
    probe.write_text(_probe_constants_source() + "# drift\n", encoding="utf-8")
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="probe SHA drift"):
        adapter.validate_reviewed_probe_source(probe, candidate)


@pytest.mark.parametrize(
    "constant",
    [
        "EXPECTED_CHECKPOINT_SHA256",
        "EXPECTED_RESOLVED_CONFIG_SHA256",
        "EXPECTED_Q_SHA256",
    ],
)
def test_wrong_checkpoint_config_or_q_contract_fails_closed(constant):
    constants = adapter._literal_string_constants(_probe_constants_source())
    constants[constant] = "0" * 64
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="contract drift"):
        adapter.validate_probe_contract_constants(constants)


@pytest.mark.parametrize("corrupt", ["checkpoint", "resolved_config"])
def test_wrong_extracted_checkpoint_or_config_sha_fails_closed(
    monkeypatch, tmp_path, corrupt
):
    archive_path, original = _archive(tmp_path)
    _patch_archive_hashes(monkeypatch, archive_path, original)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for label, member in adapter.ARCHIVE_MEMBERS.items():
            archive.writestr(member, b"corrupted" if label == corrupt else original[label])
    monkeypatch.setattr(
        adapter, "EXPECTED_STEP12E_ARCHIVE_SHA256", adapter.sha256_file(archive_path)
    )
    with pytest.raises(adapter.Step13ExecutionAdapterError, match=corrupt):
        adapter.extract_locked_artifacts(archive_path, tmp_path / "artifacts")


def test_direct_and_base64_archive_materialization_are_exact(monkeypatch, tmp_path):
    archive_path, payloads = _archive(tmp_path)
    _patch_archive_hashes(monkeypatch, archive_path, payloads)
    direct, direct_evidence = adapter.materialize_reviewed_archive(
        archive_path, tmp_path / "unused.zip"
    )
    assert direct == archive_path
    assert direct_evidence["source_transport"] == "direct_zip"

    transport = tmp_path / "step12e.zip.b64"
    transport.write_bytes(base64.b64encode(archive_path.read_bytes()))
    decoded, decoded_evidence = adapter.materialize_reviewed_archive(
        transport, tmp_path / "decoded.zip"
    )
    assert adapter.sha256_file(decoded) == adapter.EXPECTED_STEP12E_ARCHIVE_SHA256
    assert decoded_evidence["source_transport"] == "base64"


def test_wrong_step12e_archive_identity_fails_closed(tmp_path):
    archive_path, _payloads = _archive(tmp_path)
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="archive SHA drift"):
        adapter.materialize_reviewed_archive(
            archive_path, tmp_path / "unused-materialized.zip"
        )


def test_missing_archive_or_required_member_fails_closed(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.materialize_reviewed_archive(
            tmp_path / "missing.zip", tmp_path / "decoded.zip"
        )
    archive_path, payloads = _archive(tmp_path, omit="checkpoint_metadata")
    _patch_archive_hashes(monkeypatch, archive_path, payloads)
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="missing required member"):
        adapter.extract_locked_artifacts(archive_path, tmp_path / "artifacts")


def test_fresh_output_accepted_but_existing_or_nested_archive_fails(tmp_path):
    output = tmp_path / "fresh-run"
    archive = tmp_path / "fresh-run.zip"
    adapter.validate_fresh_output_paths(output, archive)
    assert not output.exists() and not archive.exists()
    output.mkdir()
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="already exists"):
        adapter.validate_fresh_output_paths(output, archive)
    other_output = tmp_path / "other"
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="outside output"):
        adapter.validate_fresh_output_paths(
            other_output, other_output / "scientific.zip"
        )


def test_cli_has_no_split_option_and_rejects_non_validation_attempt(tmp_path):
    parser = adapter.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*_required_cli(tmp_path / "run"), "--split", "test"])
    parsed = parser.parse_args(_required_cli(tmp_path / "run"))
    assert not hasattr(parsed, "split")


def test_probe_command_is_exact_deterministic_and_has_no_limit_or_split(tmp_path):
    artifacts = {
        label: tmp_path / filename
        for label, filename in adapter.EXTRACTED_FILENAMES.items()
    }
    arguments_one = adapter.build_probe_arguments(
        archive_path=tmp_path / "source.zip",
        artifacts=artifacts,
        prior_root=tmp_path / "prior",
        clean_graph_cache_dir=tmp_path / "cache",
        probe_output_root=tmp_path / "probe-output",
    )
    arguments_two = adapter.build_probe_arguments(
        archive_path=tmp_path / "source.zip",
        artifacts=artifacts,
        prior_root=tmp_path / "prior",
        clean_graph_cache_dir=tmp_path / "cache",
        probe_output_root=tmp_path / "probe-output",
    )
    assert arguments_one == arguments_two
    assert arguments_one[-6:] == [
        "--eval-batch-size",
        "32",
        "--graph-workers",
        "2",
        "--graph-cache-size",
        "64",
    ]
    joined = " ".join(arguments_one)
    assert "--limit-val-batches" not in joined
    assert "--split" not in joined


def test_reviewed_probe_is_invoked_exactly_once(monkeypatch, tmp_path):
    calls = []

    class FakeProbe:
        @staticmethod
        def main(arguments):
            calls.append(list(arguments))
            return 0

    monkeypatch.setattr(adapter, "_load_reviewed_probe", lambda path: FakeProbe())
    assert adapter.invoke_reviewed_probe(tmp_path / "probe.py", ["--help"]) == 0
    assert calls == [["--help"]]


def test_runtime_identity_is_exact_and_drift_fails_closed():
    observed = dict(adapter.REGISTERED_ENVIRONMENT)
    adapter.validate_runtime_identity(observed)
    observed["tensorflow"] = "2.19.0"
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="runtime drift"):
        adapter.validate_runtime_identity(observed)


def test_success_evidence_requires_exact_gates_samples_conditions_and_boundaries(tmp_path):
    root = tmp_path / "probe"
    root.mkdir()
    valid = _valid_probe_manifest()
    (root / "probe_manifest.json").write_text(json.dumps(valid), encoding="utf-8")
    assert adapter.validate_successful_probe_output(root)["sample_count"] == 3589
    valid["test_access"] = True
    (root / "probe_manifest.json").write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(adapter.Step13ExecutionAdapterError, match="test"):
        adapter.validate_successful_probe_output(root)


def test_valid_synthetic_completion_returns_zero(monkeypatch, tmp_path):
    def successful_probe(_path, arguments):
        probe_output = Path(arguments[arguments.index("--output-root") + 1])
        probe_output.mkdir(parents=True)
        (probe_output / "probe_manifest.json").write_text(
            json.dumps(_valid_probe_manifest()), encoding="utf-8", newline="\n"
        )
        return 0

    cli, output, archive = _patch_synthetic_registered_execution(
        monkeypatch, tmp_path, successful_probe
    )
    assert adapter.main(cli) == 0
    wrapper = json.loads((output / "wrapper_execution.json").read_text(encoding="utf-8"))
    assert wrapper["status"] == "COMPLETE"
    assert wrapper["scientific_result_valid"] is True
    assert archive.is_file()


def test_gate_invalid_completion_preserves_evidence_and_returns_nonzero(
    monkeypatch, tmp_path
):
    def gate_failure(_path, arguments):
        probe_output = Path(arguments[arguments.index("--output-root") + 1])
        probe_output.mkdir(parents=True)
        manifest = _valid_probe_manifest()
        manifest["registered_gates_and_decision"]["gate_a_native_vs_p0"]["status"] = "FAIL"
        (probe_output / "probe_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8", newline="\n"
        )
        return 0

    cli, output, archive = _patch_synthetic_registered_execution(
        monkeypatch, tmp_path, gate_failure
    )
    assert adapter.main(cli) == adapter.INVALID_REGISTERED_EXECUTION_EXIT_CODE
    wrapper = json.loads((output / "wrapper_execution.json").read_text(encoding="utf-8"))
    assert wrapper["status"] == "TECHNICAL_OR_GATE_FAILURE"
    assert wrapper["scientific_result_valid"] is False
    assert wrapper["scientific_interpretation"] is None
    assert (output / "pre_run_manifest.json").is_file()
    assert (output / "probe_execution.log").is_file()
    assert (output / "execution_report.md").is_file()
    with zipfile.ZipFile(archive) as evidence:
        names = set(evidence.namelist())
    assert {
        "pre_run_manifest.json",
        "probe_execution.log",
        "probe_output/probe_manifest.json",
        "wrapper_execution.json",
        "execution_report.md",
    } <= names


def test_probe_nonzero_preserves_partial_evidence_and_returns_nonzero(
    monkeypatch, tmp_path
):
    def failing_probe(_path, arguments):
        probe_output = Path(arguments[arguments.index("--output-root") + 1])
        probe_output.mkdir(parents=True)
        (probe_output / "partial_probe.json").write_text(
            '{"status":"PARTIAL"}\n', encoding="utf-8", newline="\n"
        )
        return 17

    cli, output, archive = _patch_synthetic_registered_execution(
        monkeypatch, tmp_path, failing_probe
    )
    assert adapter.main(cli) == adapter.INVALID_REGISTERED_EXECUTION_EXIT_CODE
    wrapper = json.loads((output / "wrapper_execution.json").read_text(encoding="utf-8"))
    assert wrapper["status"] == "TECHNICAL_OR_GATE_FAILURE"
    assert wrapper["probe_return_code"] == 17
    assert wrapper["scientific_result_valid"] is False
    with zipfile.ZipFile(archive) as evidence:
        names = set(evidence.namelist())
        log = evidence.read("probe_execution.log").decode("utf-8")
    assert "probe_output/partial_probe.json" in names
    assert "wrapper_execution.json" in names
    assert "Reviewed probe returned 17" in log


def test_keyboard_interrupt_is_not_converted_to_success(monkeypatch, tmp_path):
    def interrupted_probe(_path, _arguments):
        raise KeyboardInterrupt

    cli, _output, _archive = _patch_synthetic_registered_execution(
        monkeypatch, tmp_path, interrupted_probe
    )
    with pytest.raises(KeyboardInterrupt):
        adapter.main(cli)


def test_compact_archive_excludes_inputs_and_model_containers(tmp_path):
    root = tmp_path / "run"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/checkpoint.keras").write_bytes(b"model")
    (root / "probe_output").mkdir()
    (root / "probe_output/probe_manifest.json").write_text("{}", encoding="utf-8")
    (root / "wrapper_execution.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "evidence.zip"
    members = adapter.publish_compact_archive(root, archive)
    assert members == ["probe_output/probe_manifest.json", "wrapper_execution.json"]
    assert archive.is_file()


def test_adapter_fresh_import_does_not_import_pytorch(tmp_path):
    environment = dict(os.environ)
    for name in list(environment):
        if name.upper() == "PYTHONPATH":
            environment.pop(name)
    code = (
        "import importlib.util,sys;"
        f"p=r'{ADAPTER_PATH}';"
        "s=importlib.util.spec_from_file_location('isolated_issue40',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_frozen_package_and_scientific_probe_unchanged_from_exact_base():
    probe_blob = subprocess.check_output(
        ["git", "show", f"{BASE}:{PROBE_RELATIVE_PATH.as_posix()}"], cwd=ROOT
    )
    assert hashlib.sha256(probe_blob).hexdigest() == PARENT_PROBE_SHA
    assert subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            BASE,
            "--",
            PROBE_RELATIVE_PATH.as_posix(),
            "standalone/lap_gnn_tensorflow_ofix7_mid_candidate",
        ],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def test_adapter_source_has_no_training_test_or_pytorch_execution_path():
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported
    assert "--split" not in source
    assert "--limit-val-batches" not in source
    assert "GradientTape" not in source
    assert ".fit(" not in source
