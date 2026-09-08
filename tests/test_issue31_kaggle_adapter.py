from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_issue31_kaggle_adapter.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle-issue31-learned-local-residual-slots-seed42.ipynb"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("issue31_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _all_source():
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes():
    return {label: expected for label, (_path, expected) in builder.SOURCE_LOCKS.items()}


def _history(rows=None):
    if rows is None:
        rows = [
            {
                "epoch": 1,
                "train_macro_f1": 0.60,
                "val_macro_f1": 0.59,
                "val_accuracy": 0.61,
                "val_loss": 1.20,
                "lr": 0.0003,
                "early_stopping_wait": 0,
                "stop_requested": False,
            },
            {
                "epoch": 2,
                "train_macro_f1": 0.64,
                "val_macro_f1": 0.62,
                "val_accuracy": 0.63,
                "val_loss": 1.10,
                "lr": 0.00015,
                "early_stopping_wait": 0,
                "stop_requested": True,
            },
        ]
    return {"epochs": rows}


def _valid_completion(tmp_path: Path):
    output = tmp_path / "run"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    history_path = output / "history.json"
    history_path.write_text(json.dumps(_history()), encoding="utf-8")
    resolved_path = output / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "locked": {
                    "package_checksum": builder.SCIENTIFIC_PAYLOAD_SHA256,
                    "execution_contract_sha256": (
                        builder.BASELINE_EXECUTION_CONTRACT_SHA256
                    ),
                    "parameter_count": 1_061_192,
                },
                "training": {
                    "optimizer_execution_mode": "restricted_tf_function",
                    "gradient_execution_mode": "tf_function",
                    "grappler_profile": "G1-A",
                },
                "resources": {
                    "batch_size": 16,
                    "eval_batch_size": 32,
                    "graph_workers": 2,
                    "tf_data_prefetch": 2,
                    "tf_data_parallel_calls": 1,
                    "graph_cache_size": 64,
                    "mixed_precision": True,
                    "xla": False,
                    "memory_growth": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "telemetry.json").write_text("{}", encoding="utf-8")
    checkpoint = checkpoint_dir / "best_val_accuracy.keras"
    weights = checkpoint_dir / "best_val_accuracy.weights.h5"
    metadata = checkpoint_dir / "best_val_accuracy.metadata.json"
    checkpoint.write_bytes(b"candidate-checkpoint")
    weights.write_bytes(b"candidate-weights")
    metadata.write_text(
        json.dumps(
            {
                "epoch": 2,
                "validation_metrics": {
                    "accuracy": 0.63,
                    "macro_f1": 0.62,
                    "loss": 1.10,
                },
            }
        ),
        encoding="utf-8",
    )
    frozen = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "test_data_constructed": False,
        "test_checkpoint_loaded": False,
        "normal_full_training_completed": False,
        "boundary": "before_resolve_final_checkpoint",
        "trainer_revision_guard_passed": True,
        "intercepted_function_restored": True,
        "trainer_source_sha256": builder.SOURCE_LOCKS["frozen_trainer"][1],
        "scientific_payload_sha256": builder.SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_sha256": builder.SOURCE_LOCKS["seed42_config"][1],
        "seed": 42,
        "final_observed_epoch": 2,
        "history_sha256": _sha256(history_path),
        "resolved_config_sha256": _sha256(resolved_path),
        "bounded_limits": {
            "limit_epochs": None,
            "limit_train_batches": None,
            "limit_val_batches": None,
            "limit_train_eval_batches": None,
        },
    }
    source_hashes = _source_hashes()
    sidecar_keys = {
        "candidate_model",
        "candidate_execution_adapter",
        "candidate_execution_contract",
        "frozen_validation_only_wrapper",
        "frozen_trainer",
        "frozen_execution",
    }
    sidecar_sources = {
        key: value for key, value in source_hashes.items() if key in sidecar_keys
    }
    candidate = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_access": False,
        "original_constructor_restored": True,
        "original_restricted_builder_restored": True,
        "candidate_constructor_injected": True,
        "candidate_restricted_builder_injected": True,
        "candidate_class": builder.EXPECTED_CANDIDATE_CLASS,
        "actual_candidate_parameter_count": builder.EXPECTED_CANDIDATE_PARAMS,
        "candidate_trainable_variable_count": builder.EXPECTED_VARIABLE_COUNT,
        "q_index": builder.EXPECTED_Q_INDEX,
        "q_shape": builder.EXPECTED_Q_SHAPE,
        "q_dtype": builder.EXPECTED_Q_DTYPE,
        "candidate_harness_sha256": builder.SOURCE_LOCKS[
            "candidate_validation_harness"
        ][1],
        "candidate_model_sha256": builder.SOURCE_LOCKS["candidate_model"][1],
        "candidate_execution_adapter_sha256": builder.SOURCE_LOCKS[
            "candidate_execution_adapter"
        ][1],
        "candidate_execution_contract_sha256": builder.SOURCE_LOCKS[
            "candidate_execution_contract"
        ][1],
        "frozen_validation_only_wrapper_sha256": builder.SOURCE_LOCKS[
            "frozen_validation_only_wrapper"
        ][1],
        "frozen_trainer_sha256": builder.SOURCE_LOCKS["frozen_trainer"][1],
        "frozen_execution_sha256": builder.SOURCE_LOCKS["frozen_execution"][1],
        "scientific_payload_sha256": builder.SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_sha256": builder.SOURCE_LOCKS["seed42_config"][1],
        "inherited_baseline_execution_contract_sha256": (
            builder.BASELINE_EXECUTION_CONTRACT_SHA256
        ),
        "source_artifact_sha256_before": sidecar_sources,
        "source_artifact_sha256_after": sidecar_sources,
        "final_observed_epoch": 2,
        "history_sha256": _sha256(history_path),
        "resolved_config_sha256": _sha256(resolved_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_class": builder.EXPECTED_CANDIDATE_CLASS,
        "checkpoint_parameter_count": builder.EXPECTED_CANDIDATE_PARAMS,
        "checkpoint_q_shape": builder.EXPECTED_Q_SHAPE,
        "checkpoint_q_dtype": builder.EXPECTED_Q_DTYPE,
        "learned_q_flat_float32_sha256": "a" * 64,
    }
    (output / builder.FROZEN_MARKER).write_text(json.dumps(frozen), encoding="utf-8")
    candidate["validation_only_marker_sha256"] = _sha256(
        output / builder.FROZEN_MARKER
    )
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    return output, source_hashes, candidate


def _rewrite_history_and_marker_hashes(
    output: Path, candidate: dict, history: dict
) -> None:
    history_path = output / "history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    final_epoch = history["epochs"][-1]["epoch"]
    frozen_path = output / builder.FROZEN_MARKER
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["history_sha256"] = _sha256(history_path)
    frozen["final_observed_epoch"] = final_epoch
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    candidate["history_sha256"] = _sha256(history_path)
    candidate["final_observed_epoch"] = final_epoch
    candidate["validation_only_marker_sha256"] = _sha256(frozen_path)
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )


def test_notebook_is_deterministic_unexecuted_and_compiles(tmp_path):
    assert _notebook() == builder.build_notebook()
    first = json.dumps(builder.build_notebook(), sort_keys=True)
    second = json.dumps(builder.build_notebook(), sort_keys=True)
    assert first == second
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 9
    assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"issue31-{index:02d}"
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"issue31-cell-{index}", "exec")


def test_exact_commit_source_payload_and_execution_locks():
    source = _all_source()
    assert builder.EXECUTION_COMMIT in source
    assert builder.SCIENTIFIC_PAYLOAD_SHA256 in source
    assert builder.BASELINE_EXECUTION_CONTRACT_SHA256 in source
    for relative, digest in builder.SOURCE_LOCKS.values():
        assert relative in source
        assert digest in source
        assert _sha256(ROOT / relative) == digest
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{builder.EXECUTION_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert '["git", "clone", "--no-checkout"' in source
    assert '["git", "checkout", "--detach", EXECUTION_COMMIT]' in source
    assert '["git", "status", "--porcelain"]' in source
    assert builder.verify_source_locks(ROOT) == _source_hashes()


def test_registered_command_is_exact_single_harness_unbounded_and_no_trainer():
    command = builder.registered_command(ROOT, Path("/tmp/issue31-output"))
    harness = str(ROOT / builder.SOURCE_LOCKS["candidate_validation_harness"][0])
    trainer = str(ROOT / builder.SOURCE_LOCKS["frozen_trainer"][0])
    assert command.count(harness) == 1
    assert trainer not in command
    assert not any(value.startswith("--limit-") for value in command)
    assert command[command.index("--device") + 1] == "gpu"
    expected = {
        "--graph-workers": "2",
        "--tf-data-prefetch": "2",
        "--tf-data-parallel-calls": "1",
        "--graph-cache-size": "64",
        "--batch-size": "16",
        "--eval-batch-size": "32",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    for flag in ("--mixed-precision", "--no-xla", "--memory-growth", "--no-resume"):
        assert command.count(flag) == 1
    assert str(builder.FER_TRAIN_CSV) in command
    assert str(builder.PRIOR_ROOT) in command
    assert str(builder.CACHE_ROOT) in command


def test_notebook_locks_candidate_identity_contract_and_registered_environment():
    source = _all_source()
    for token in (
        "LearnedLocalResidualSlotLapGNN",
        "1061576",
        "128",
        "127",
        "[4, 96]",
        "restricted_tf_function",
        "G1-A",
        "mixed_float16_supported",
        "raw_slot_diagnostics_dtype",
        "residual_input_dtype",
        "official_global_cast",
        "3.12.12",
        "2.18.1",
        "3.15.0",
        "EXPECTED_GPU_COUNT = 2",
        "EXPECTED_GPU_TOKEN = 'T4'",
    ):
        assert token in source


def test_only_registered_train_val_inputs_and_shared_aggregate_are_resolved():
    source = _all_source()
    assert builder.FER_ROOT.as_posix() in source
    assert builder.PRIOR_ROOT.as_posix() in source
    assert builder.CACHE_ROOT.as_posix() in source
    assert 'EXPECTED_SPLIT_COUNTS = {\'train\': 28709, \'val\': 3589}' in source
    assert 'CACHE_ROOT / "CACHE_COMPLETE.json"' in source
    for forbidden in (
        'FER_ROOT / "test.csv"',
        'PRIOR_ROOT / "test"',
        'CACHE_ROOT / "test"',
        'split="test"',
        "split='test'",
        "resolve_final_checkpoint(",
    ):
        assert forbidden not in source


def test_primary_comparator_constants_and_exact_decision_boundaries():
    assert builder.BASELINE_BEST_VAL_MACRO_F1 == 0.601166548701511
    assert builder.BASELINE_BEST_VAL_MACRO_EPOCH == 26
    assert builder.BASELINE_TRAIN_MACRO_AT_BEST == 0.7562805286580438
    assert builder.BASELINE_TRAIN_VAL_GAP_PP == 15.511397995653287
    assert builder.BASELINE_BEST_VAL_ACCURACY == 0.6319308999721371
    assert builder.BASELINE_BEST_VAL_ACCURACY_EPOCH == 31
    assert builder.BASELINE_EPOCH31_VAL_MACRO_F1 == 0.5938407974340496
    assert builder.BASELINE_BEST_VAL_LOSS == 1.0625020856350924
    assert builder.BASELINE_BEST_VAL_LOSS_EPOCH == 17
    assert builder.registered_decision(1.0) == "PROMISING_SINGLE_SEED_VALIDATION_GAIN"
    assert builder.registered_decision(0.999999) == "NO_CLEAR_SINGLE_SEED_DIFFERENCE"
    assert builder.registered_decision(-0.999999) == "NO_CLEAR_SINGLE_SEED_DIFFERENCE"
    assert builder.registered_decision(-1.0) == "SINGLE_SEED_VALIDATION_REGRESSION"


def test_metric_derivation_uses_earliest_exact_tie_and_complete_history():
    rows = _history()["epochs"]
    rows.append({**rows[-1], "epoch": 3, "val_macro_f1": 0.62})
    derived = builder.derive_registered_metrics({"epochs": rows})
    assert derived["candidate_best_val_macro_epoch"] == 2
    assert derived["candidate_best_val_macro_f1"] == 0.62
    assert derived["lr_reductions"] == [{"epoch": 2, "lr": 0.00015}]
    assert derived["final_observed_epoch"] == 3
    with pytest.raises(builder.AdapterError, match="sequential"):
        builder.derive_registered_metrics({"epochs": [rows[1]]})


def test_scientific_decision_requires_both_valid_completion_markers(tmp_path):
    output, source_hashes, _candidate = _valid_completion(tmp_path)
    (output / builder.CANDIDATE_MARKER).unlink()
    with pytest.raises(builder.AdapterError, match="candidate completion sidecar"):
        builder.validate_completion(output, source_hashes, source_hashes)


@pytest.mark.parametrize("mode", ["missing", "false"])
def test_frozen_marker_revision_guard_missing_or_false_fails_closed(tmp_path, mode):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    frozen_path = output / builder.FROZEN_MARKER
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if mode == "missing":
        frozen.pop("trainer_revision_guard_passed")
    else:
        frozen["trainer_revision_guard_passed"] = False
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    candidate["validation_only_marker_sha256"] = _sha256(frozen_path)
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(builder.AdapterError, match="trainer_revision_guard_passed"):
        builder.validate_completion(output, source_hashes, source_hashes)


def test_frozen_marker_wrong_seed42_config_sha_fails_closed(tmp_path):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    frozen_path = output / builder.FROZEN_MARKER
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["input_config_sha256"] = "0" * 64
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    candidate["validation_only_marker_sha256"] = _sha256(frozen_path)
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(builder.AdapterError, match="input_config_sha256"):
        builder.validate_completion(output, source_hashes, source_hashes)


def test_candidate_sidecar_wrong_seed42_config_sha_fails_closed(tmp_path):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    candidate["input_config_sha256"] = "f" * 64
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(builder.AdapterError, match="input_config_sha256"):
        builder.validate_completion(output, source_hashes, source_hashes)


def test_frozen_marker_wrong_seed_fails_closed(tmp_path):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    frozen_path = output / builder.FROZEN_MARKER
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["seed"] = 43
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    candidate["validation_only_marker_sha256"] = _sha256(frozen_path)
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(builder.AdapterError, match="seed"):
        builder.validate_completion(output, source_hashes, source_hashes)


@pytest.mark.parametrize("mode", ["malformed", "missing"])
def test_malformed_or_missing_candidate_sidecar_fails_closed(tmp_path, mode):
    output, source_hashes, _candidate = _valid_completion(tmp_path)
    path = output / builder.CANDIDATE_MARKER
    if mode == "malformed":
        path.write_text("{", encoding="utf-8")
    else:
        path.unlink()
    with pytest.raises(builder.AdapterError):
        builder.validate_completion(output, source_hashes, source_hashes)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("checkpoint_class", "WrongModel", "class"),
        ("checkpoint_parameter_count", 1, "parameter"),
        ("checkpoint_q_shape", [4, 95], "Q shape"),
        ("checkpoint_q_dtype", "float16", "Q dtype"),
    ],
)
def test_wrong_checkpoint_class_parameters_or_q_fails_closed(
    tmp_path, field, value, match
):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    candidate[field] = value
    (output / builder.CANDIDATE_MARKER).write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(builder.AdapterError, match=match):
        builder.validate_completion(output, source_hashes, source_hashes)


def test_self_consistent_non_best_checkpoint_metadata_fails_closed(tmp_path):
    output, source_hashes, _candidate = _valid_completion(tmp_path)
    metadata_path = output / "checkpoints" / "best_val_accuracy.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "validation_metrics": {
                    "accuracy": 0.61,
                    "macro_f1": 0.59,
                    "loss": 1.20,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(builder.AdapterError, match="earliest.*global maximum"):
        builder.validate_completion(output, source_hashes, source_hashes)


def test_exact_best_accuracy_tie_requires_earliest_epoch(tmp_path):
    output, source_hashes, candidate = _valid_completion(tmp_path)
    history = _history()
    history["epochs"][0]["val_accuracy"] = 0.63
    _rewrite_history_and_marker_hashes(output, candidate, history)
    with pytest.raises(builder.AdapterError, match="earliest.*global maximum"):
        builder.validate_completion(output, source_hashes, source_hashes)

    metadata_path = output / "checkpoints" / "best_val_accuracy.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "validation_metrics": {
                    "accuracy": 0.63,
                    "macro_f1": 0.59,
                    "loss": 1.20,
                },
            }
        ),
        encoding="utf-8",
    )
    derived = builder.validate_completion(output, source_hashes, source_hashes)
    assert derived["candidate_best_val_accuracy_epoch"] == 1
    assert derived["checkpoint_metadata"]["epoch"] == 1


def test_source_drift_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "locked.py"
    source.write_text("reviewed\n", encoding="utf-8")
    expected = _sha256(source)
    monkeypatch.setattr(builder, "SOURCE_LOCKS", {"one": ("locked.py", expected)})
    source.write_text("drift\n", encoding="utf-8")
    with pytest.raises(builder.AdapterError, match="drift"):
        builder.verify_source_locks(tmp_path)


def test_rolling_archive_is_atomic_and_epoch_progression_refreshes(tmp_path):
    run_root = tmp_path / "bundle"
    adapter = run_root / "adapter"
    output = run_root / "run"
    adapter.mkdir(parents=True)
    output.mkdir()
    builder.atomic_json(adapter / "pre_run_manifest.json", {"issue": 31})
    (adapter / "subprocess.log").write_text("epoch stream\n", encoding="utf-8")
    history_path = output / "history.json"
    history_path.write_text(
        json.dumps(_history(_history()["epochs"][:1])), encoding="utf-8"
    )
    archive = tmp_path / "rolling.zip"
    progress = adapter / "runtime_progress.json"
    monitor = builder.RollingArchiveMonitor(
        history_path=history_path,
        progress_path=progress,
        archive_path=archive,
        run_root=run_root,
        train_output_root=output,
        report_path=tmp_path / "report.md",
        poll_seconds=0.01,
    )
    assert monitor.poll_once() is True
    assert monitor.poll_once() is False
    history_path.write_text(json.dumps(_history()), encoding="utf-8")
    assert monitor.poll_once() is True
    assert json.loads(progress.read_text())["latest_completed_epoch"] == 2
    assert archive.is_file()
    assert not archive.with_name(f".{archive.name}.tmp").exists()
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert "adapter/pre_run_manifest.json" in zipped.namelist()
        assert "run/history.json" in zipped.namelist()


class _FakeProcess:
    def __init__(self, return_code=9):
        self.stdout = iter(["started\n", "technical failure\n"])
        self._return_code = return_code

    def wait(self):
        return self._return_code


def test_synthetic_subprocess_failure_preserves_log_partial_evidence_and_no_final(
    tmp_path,
):
    run_root = tmp_path / "bundle"
    adapter = run_root / "adapter"
    output = run_root / "run"
    adapter.mkdir(parents=True)
    output.mkdir()
    builder.atomic_json(adapter / "pre_run_manifest.json", {"issue": 31})
    (output / "history.json").write_text(json.dumps(_history()), encoding="utf-8")
    archive = tmp_path / "failure.zip"
    monitor = builder.RollingArchiveMonitor(
        history_path=output / "history.json",
        progress_path=adapter / "runtime_progress.json",
        archive_path=archive,
        run_root=run_root,
        train_output_root=output,
        report_path=tmp_path / "report.md",
        poll_seconds=0.01,
    )
    log = adapter / "subprocess.log"
    calls = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeProcess()

    return_code = builder.run_subprocess_once(
        ["candidate-harness", "--no-resume"],
        cwd=tmp_path,
        log_path=log,
        monitor=monitor,
        popen_factory=fake_popen,
    )
    assert return_code == 9
    assert len(calls) == 1
    wrapper = builder.write_failure_outputs(
        wrapper_path=adapter / "wrapper_execution.json",
        evidence_path=adapter / "final_evidence.json",
        failure_report_path=adapter / "technical_or_runtime_failure.md",
        subprocess_return_code=return_code,
        error_text="synthetic failure",
        source_hashes_before=_source_hashes(),
        source_hashes_after=_source_hashes(),
    )
    builder.publish_archive_atomic(archive, run_root, output, tmp_path / "report.md")
    assert wrapper["status"] == "TECHNICAL_OR_RUNTIME_FAILURE"
    assert wrapper["scientific_result_valid"] is False
    assert wrapper["scientific_interpretation"] is None
    assert not (adapter / "final_evidence.json").exists()
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        assert "adapter/subprocess.log" in names
        assert "adapter/wrapper_execution.json" in names
        assert "adapter/technical_or_runtime_failure.md" in names
        assert "run/history.json" in names
        assert "adapter/final_evidence.json" not in names


def test_valid_synthetic_completion_creates_final_evidence_and_report(tmp_path):
    output, source_hashes, _candidate = _valid_completion(tmp_path)
    derived = builder.validate_completion(output, source_hashes, source_hashes)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    report = tmp_path / "tf_step12.md"
    wrapper = builder.write_success_outputs(
        derived=derived,
        wrapper_path=adapter / "wrapper_execution.json",
        evidence_path=adapter / "final_evidence.json",
        report_path=report,
        subprocess_return_code=0,
        source_hashes_before=source_hashes,
        source_hashes_after=source_hashes,
    )
    evidence = json.loads((adapter / "final_evidence.json").read_text())
    assert wrapper["status"] == "COMPLETE"
    assert wrapper["scientific_result_valid"] is True
    assert evidence["scientific_interpretation"] == (
        "PROMISING_SINGLE_SEED_VALIDATION_GAIN"
    )
    assert report.is_file()


def test_hard_censor_partial_history_remains_scientifically_invalid(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "final_evidence.json").write_text("fabricated", encoding="utf-8")
    wrapper = builder.write_failure_outputs(
        wrapper_path=adapter / "wrapper_execution.json",
        evidence_path=adapter / "final_evidence.json",
        failure_report_path=adapter / "technical_or_runtime_failure.md",
        subprocess_return_code=None,
        error_text="runtime-censored before completion markers",
        source_hashes_before=_source_hashes(),
        source_hashes_after=None,
    )
    assert wrapper["status"] == "TECHNICAL_OR_RUNTIME_FAILURE"
    assert wrapper["scientific_result_valid"] is False
    assert wrapper["scientific_interpretation"] is None
    assert not (adapter / "final_evidence.json").exists()


def test_no_automatic_retry_or_direct_trainer_execution_is_implemented():
    source = _all_source()
    assert source.count("popen_factory(") == 1
    assert '"automatic_retry": False' in source
    assert '"candidate_harness_invocations": 1' in source
    assert '"direct_frozen_trainer_invocations": 0' in source
    assert "trainer.run_training(" not in source
    assert "TF_DETERMINISTIC_OPS" not in source
    assert "enable_op_determinism" not in source


def test_notebook_documents_kaggle_inputs_internet_and_exact_zip():
    source = _all_source()
    assert "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split" in source
    assert (
        "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
        in source
    )
    assert "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records" in source
    assert "Internet is required" in source
    assert (
        "/kaggle/working/tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip"
        in source
    )
