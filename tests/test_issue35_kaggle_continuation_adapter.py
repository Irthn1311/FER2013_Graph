from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools/build_issue35_kaggle_continuation_adapter.py"
NOTEBOOK_PATH = ROOT / "notebooks/kaggle-issue35-step12c-checkpoint-continuation.ipynb"
BASE = "0f4fde1d4e6645096711a800509f4db2deedf38f"


def _load_builder():
    spec = importlib.util.spec_from_file_location("issue35_adapter_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_builder()


def _rows(count=33):
    rows = []
    for epoch in range(1, count + 1):
        row = {
            "epoch": epoch,
            "train_macro_f1": 0.50 + epoch / 1000.0,
            "val_macro_f1": 0.54 + epoch / 2000.0,
            "val_accuracy": 0.55 + min(epoch, 30) / 1000.0,
            "val_loss": 1.5 - min(epoch, 30) / 1000.0,
            "lr": 0.00015 if epoch < 33 else 0.000075,
            "early_stopping_wait": max(0, epoch - 30),
            "stop_requested": epoch == count,
        }
        rows.append(row)
    return rows


def _combined(source_rows, count=33):
    rows = json.loads(json.dumps(source_rows[:30]))
    for epoch in range(31, count + 1):
        row = _rows(count)[epoch - 1]
        row["val_macro_f1"] += 0.001
        row["row_origin"] = adapter.CONTINUATION_ROW_ORIGIN
        row["continuation_protocol_id"] = adapter.CONTINUATION_PROTOCOL_ID
        rows.append(row)
    return rows


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_archive(monkeypatch, root: Path):
    source_rows = _rows(32)
    history = json.dumps({"epochs": source_rows}, sort_keys=True).encode()
    checkpoint = b"epoch30-checkpoint"
    archive_path = root / "input/a" / adapter.SOURCE_ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("run/history.json", history)
        archive.writestr("run/checkpoints/best_val_accuracy.keras", checkpoint)
    monkeypatch.setattr(adapter, "SOURCE_ARCHIVE_SHA256", adapter.sha256_file(archive_path))
    monkeypatch.setattr(adapter, "SOURCE_HISTORY_SHA256", hashlib.sha256(history).hexdigest())
    monkeypatch.setattr(adapter, "SOURCE_CHECKPOINT_SHA256", hashlib.sha256(checkpoint).hexdigest())
    return archive_path, source_rows


def _runtime():
    return {
        "python": adapter.EXPECTED_PYTHON,
        "tensorflow": adapter.EXPECTED_TENSORFLOW,
        "keras": adapter.EXPECTED_KERAS,
        "cuda": adapter.EXPECTED_CUDA,
        "cudnn_major": adapter.EXPECTED_CUDNN_MAJOR,
        "gpu_count": 2,
        "gpu_names": ["Tesla T4", "Tesla T4"],
    }


def _valid_fixture(monkeypatch, tmp_path: Path, *, reason="early_stopping", count=33):
    archive_path, source_rows = _source_archive(monkeypatch, tmp_path)
    run_root = tmp_path / "run_root"
    output = run_root / "run"
    adapter_root = run_root / "adapter"
    adapter_root.mkdir(parents=True)
    (adapter_root / "subprocess.log").write_text("synthetic\n", encoding="utf-8")
    rows = _combined(source_rows, count)
    history = {"epochs": rows}
    _write_json(output / "history.json", history)
    _write_json(output / "continuation_pre_run_manifest.json", {
        "schema_version": 1,
        "protocol_id": adapter.CONTINUATION_PROTOCOL_ID,
        "source_archive_sha256": adapter.SOURCE_ARCHIVE_SHA256,
        "source_history_sha256": adapter.SOURCE_HISTORY_SHA256,
        "immutable_scientific_prefix_sha256": adapter._canonical_json_sha256(rows[:30]),
    })
    references = {
        "sample_count": 3589,
        "accuracy": 0.603789356366676,
        "macro_f1": 0.5634445160028113,
        "loss": 1.1265364869505958,
    }
    _write_json(output / "pretrain_validation_gate.json", {
        "status": "PASS", "sample_count": 3589,
        "references": references,
        "tolerances": {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005},
        "observed": {key: references[key] for key in ("accuracy", "macro_f1", "loss")},
        "absolute_differences": {"accuracy": 0.0, "macro_f1": 0.0, "loss": 0.0},
        "optimizer_updates_before_gate": 0,
    })
    _write_json(output / adapter.OVERLAP_SOURCE, {
        "schema_version": 1,
        "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS",
        "descriptive_only": True,
        "excluded_from_combined_scientific_history": True,
        "rows": {"31": source_rows[30], "32": source_rows[31]},
    })
    for epoch in (31, 32):
        _write_json(output / f"resume_overlap_epoch{epoch}.json", {
            "schema_version": 1,
            "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS",
            "epoch": epoch, "descriptive_only": True,
            "affects_training": False, "affects_stopping": False,
            "affects_scheduler": False, "affects_checkpoint_selection": False,
            "affects_primary_endpoint": False, "triggers_retry": False,
            "first_run_row": source_rows[epoch - 1],
            "resumed_row": rows[epoch - 1],
        })
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    best_epoch = 30
    best_row = rows[best_epoch - 1]
    best_payloads = {
        "best_val_accuracy.keras": b"best-model",
        "best_val_accuracy.weights.h5": b"best-weights",
        "best_val_accuracy.metadata.json": json.dumps({
            "epoch": best_epoch,
            "validation_metrics": {
                "accuracy": best_row["val_accuracy"],
                "macro_f1": best_row["val_macro_f1"],
                "loss": best_row["val_loss"],
            },
        }, sort_keys=True).encode() + b"\n",
    }
    for name, payload in best_payloads.items():
        (checkpoints / name).write_bytes(payload)

    generation = output / f"latest_states/epoch_{count:04d}"
    generation.mkdir(parents=True)
    model_path = generation / "state.keras"
    model_path.write_bytes(b"state-model")
    generation_history = generation / "history.json"
    _write_json(generation_history, history)
    snapshot = generation / "best_val_accuracy"
    snapshot.mkdir()
    best_hashes = {}
    for name, payload in best_payloads.items():
        path = snapshot / name
        path.write_bytes(payload)
        relative = f"latest_states/epoch_{count:04d}/best_val_accuracy/{name}"
        best_hashes[relative] = adapter.sha256_file(path)
    fingerprint_hash = "f" * 64
    fingerprint = {
        "sha256": fingerprint_hash,
        "variable_count": 2,
        "variables": [
            {"index": index, "shape": [1], "dtype": "float32",
             "name": f"v{index}", "path": f"v{index}", "value_sha256": str(index) * 64}
            for index in (1, 2)
        ],
    }
    # Stable ordered indices are 0..N-1.
    for index, variable in enumerate(fingerprint["variables"]):
        variable["index"] = index
    metadata_path = generation / "state.metadata.json"
    metadata = {
        "schema_version": 2,
        "continuation_protocol_id": adapter.CONTINUATION_PROTOCOL_ID,
        "completed_epoch": count, "next_epoch": count + 1,
        "generation_relative_path": f"latest_states/epoch_{count:04d}",
        "state_keras_sha256": adapter.sha256_file(model_path),
        "combined_history_sha256": adapter.sha256_file(generation_history),
        "optimizer_state_sha256": fingerprint_hash,
        "optimizer_state_fingerprint": fingerprint,
        "optimizer_variable_count": 2,
        "optimizer_class": "LossScaleOptimizer",
        "model_class": adapter.EXPECTED_CANDIDATE_CLASS,
        "model_parameter_count": adapter.EXPECTED_CANDIDATE_PARAMS,
        "model_trainable_variable_count": adapter.EXPECTED_VARIABLE_COUNT,
        "q_index": adapter.EXPECTED_Q_INDEX,
        "q_shape": adapter.EXPECTED_Q_SHAPE,
        "q_dtype": adapter.EXPECTED_Q_DTYPE,
        "q_flat_float32_sha256": "q" * 64,
        "best_val_accuracy_artifact_sha256": best_hashes,
        "partial_epoch": False, "test_access": False,
        "scientific_result_valid": False, "scientific_interpretation": None,
    }
    _write_json(metadata_path, metadata)
    manifest_path = output / adapter.LATEST_STATE_MANIFEST
    manifest = {
        "schema_version": 1,
        "continuation_protocol_id": adapter.CONTINUATION_PROTOCOL_ID,
        "generation_id": f"epoch_{count:04d}",
        "completed_epoch": count, "next_epoch": count + 1,
        "generation_relative_path": f"latest_states/epoch_{count:04d}",
        "model_relative_path": f"latest_states/epoch_{count:04d}/state.keras",
        "metadata_relative_path": f"latest_states/epoch_{count:04d}/state.metadata.json",
        "history_relative_path": f"latest_states/epoch_{count:04d}/history.json",
        "model_sha256": adapter.sha256_file(model_path),
        "metadata_sha256": adapter.sha256_file(metadata_path),
        "combined_history_sha256": adapter.sha256_file(generation_history),
        "optimizer_state_sha256": fingerprint_hash,
        "partial_epoch": False, "test_access": False,
    }
    _write_json(manifest_path, manifest)
    completion = {
        "continuation_protocol_id": adapter.CONTINUATION_PROTOCOL_ID,
        "source_archive_sha256": adapter.SOURCE_ARCHIVE_SHA256,
        "source_checkpoint_sha256": adapter.SOURCE_CHECKPOINT_SHA256,
        "continuation_completed": True, "completion_reason": reason,
        "scientific_prefix_first_epoch": 1, "scientific_prefix_last_epoch": 30,
        "first_continuation_epoch": 31,
        "original_step12_scientific_result_valid": False,
        "original_step12_scientific_interpretation": None,
        "scientific_result_valid": False, "scientific_interpretation": None,
        "training": True, "optimizer_gradient_updates": True,
        "test_access": False, "test_data_constructed": False,
        "final_test_skipped": True, "final_completed_epoch": count,
        "latest_state_generation": f"latest_states/epoch_{count:04d}",
        "latest_state_model_sha256": manifest["model_sha256"],
        "latest_state_optimizer_sha256": fingerprint_hash,
        "combined_history_sha256": adapter.sha256_file(output / "history.json"),
        "latest_state_manifest_sha256": adapter.sha256_file(manifest_path),
    }
    _write_json(output / adapter.COMPLETION_MARKER, completion)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    pre_manifest = {
        "schema_version": 1, "issue": 35,
        "execution_commit": adapter.EXECUTION_COMMIT,
        "checkout_root": str(checkout),
        "source_archive_path": str(archive_path.resolve()),
        "source_archive_sha256": adapter.SOURCE_ARCHIVE_SHA256,
        "source_locks": {"locked": "a" * 64},
        "scientific_payload_sha256": adapter.SCIENTIFIC_PAYLOAD_SHA256,
        "inherited_execution_contract_sha256": adapter.BASELINE_EXECUTION_CONTRACT_SHA256,
        "continuation_harness_invocations": 1,
        "direct_frozen_trainer_invocations": 0,
        "initial_step12_harness_invocations": 0,
        "chained_latest_state_invocations": 0,
        "automatic_retry": False, "seed": 42,
        "resources": {
            "train_batch_size": 16, "eval_batch_size": 32,
            "graph_workers": 2, "tf_data_prefetch": 2,
            "tf_data_parallel_calls": 1, "graph_cache_size": 64,
            "mixed_precision": True, "xla": False, "memory_growth": True,
            "op_determinism_changed": False, "bounded_limits": None,
        },
        "resume_anchor": {
            "epoch": 30, "checkpoint_sha256": adapter.SOURCE_CHECKPOINT_SHA256,
            "q_sha256": adapter.EXPECTED_RESUME_Q_SHA256,
            "optimizer_iterations": adapter.EXPECTED_RESUME_OPTIMIZER_ITERATIONS,
            "optimizer_variables": adapter.EXPECTED_RESUME_OPTIMIZER_VARIABLES,
            "learning_rate": adapter.EXPECTED_RESUME_LR,
        },
        "runtime": _runtime(), "test_access": False,
    }
    pre_manifest["command"] = adapter.registered_command(checkout, output, archive_path.resolve())
    pre_path = adapter_root / "pre_run_manifest.json"
    _write_json(pre_path, pre_manifest)
    deep = {
        "status": "PASS", "completed_epoch": count, "next_epoch": count + 1,
        "model_class": adapter.EXPECTED_CANDIDATE_CLASS,
        "model_parameter_count": adapter.EXPECTED_CANDIDATE_PARAMS,
        "model_trainable_variable_count": adapter.EXPECTED_VARIABLE_COUNT,
        "q_index": adapter.EXPECTED_Q_INDEX, "q_shape": adapter.EXPECTED_Q_SHAPE,
        "q_dtype": adapter.EXPECTED_Q_DTYPE, "q_sha256": "q" * 64,
        "optimizer_state_sha256": fingerprint_hash, "test_access": False,
    }
    return SimpleNamespace(
        run_root=run_root, output=output, adapter_root=adapter_root,
        archive=archive_path, source_rows=source_rows, rows=rows,
        pre_manifest=pre_path, deep=deep, source_hashes={"locked": "a" * 64},
    )


def _validate(fixture):
    return adapter.validate_completion(
        fixture.output, fixture.pre_manifest,
        fixture.source_hashes, fixture.source_hashes, fixture.deep,
    )


def _replace_best_metadata_and_rehash(fixture, payload):
    root_path = fixture.output / "checkpoints/best_val_accuracy.metadata.json"
    snapshot_path = fixture.output / "latest_states/epoch_0033/best_val_accuracy/best_val_accuracy.metadata.json"
    _write_json(root_path, payload)
    _write_json(snapshot_path, payload)
    state_path = fixture.output / "latest_states/epoch_0033/state.metadata.json"
    state = adapter.json_object(state_path, "state")
    relative = "latest_states/epoch_0033/best_val_accuracy/best_val_accuracy.metadata.json"
    state["best_val_accuracy_artifact_sha256"][relative] = adapter.sha256_file(snapshot_path)
    _write_json(state_path, state)
    manifest_path = fixture.output / adapter.LATEST_STATE_MANIFEST
    manifest = adapter.json_object(manifest_path, "manifest")
    manifest["metadata_sha256"] = adapter.sha256_file(state_path)
    _write_json(manifest_path, manifest)
    completion_path = fixture.output / adapter.COMPLETION_MARKER
    completion = adapter.json_object(completion_path, "completion")
    completion["latest_state_manifest_sha256"] = adapter.sha256_file(manifest_path)
    _write_json(completion_path, completion)


def test_notebook_is_deterministic_unexecuted_and_compiles(tmp_path):
    first = adapter.build_notebook()
    second = adapter.build_notebook()
    assert first == second
    payload = json.dumps(first, indent=1, ensure_ascii=False) + "\n"
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == payload
    assert len(first["cells"]) == 9
    code_cells = [cell for cell in first["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
    output = tmp_path / "adapter.ipynb"
    output.write_text(payload, encoding="utf-8", newline="\n")
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(
        NOTEBOOK_PATH.read_bytes()
    ).hexdigest()


def test_exact_execution_source_and_artifact_locks():
    assert adapter.EXECUTION_COMMIT == BASE
    assert adapter.SOURCE_LOCKS["continuation_harness"][1] == (
        "dba0d749b9a8e05b3cd67dad0749ef4235fc06f2a389b552229c76f691edde40"
    )
    assert adapter.SOURCE_LOCKS["step12c_evidence"][1] == (
        "4e48652c4c75cbdcf985b596e04c5658483825ab1e6900f97d52d1cf7ee7f29f"
    )
    assert adapter.SOURCE_ARCHIVE_SHA256 == (
        "2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357"
    )
    assert adapter.verify_source_locks(ROOT)["continuation_harness"] == (
        adapter.SOURCE_LOCKS["continuation_harness"][1]
    )


def test_registered_command_is_exact_one_first_continuation_call(tmp_path):
    command = adapter.registered_command(tmp_path / "checkout", tmp_path / "out", tmp_path / "source.zip")
    joined = " ".join(command)
    assert joined.count("resume_validation_only.py") == 1
    assert "--source-archive" in command
    assert "--fer-csv" in command and str(adapter.FER_TRAIN_CSV) in command
    assert "--prior-root" in command and str(adapter.PRIOR_ROOT) in command
    assert "--clean-graph-cache-dir" in command and str(adapter.CACHE_ROOT) in command
    assert "--batch-size 16" in joined and "--eval-batch-size 32" in joined
    assert "--graph-workers 2" in joined and "--tf-data-prefetch 2" in joined
    assert "--tf-data-parallel-calls 1" in joined and "--graph-cache-size 64" in joined
    assert "--mixed-precision" in command and "--no-xla" in command
    assert "--memory-growth" in command
    assert not any(value.startswith("--limit-") for value in command)
    assert "train_validation_only.py" not in joined
    assert "continue_from_latest_completed_state" not in joined
    assert "trainer.py" not in joined


def test_source_archive_exact_one_discovery_and_sha(monkeypatch, tmp_path):
    archive, _rows_value = _source_archive(monkeypatch, tmp_path)
    assert adapter.discover_source_archive(tmp_path / "input") == archive.resolve()


def test_source_archive_zero_matches_fails(tmp_path):
    with pytest.raises(adapter.AdapterError, match="exactly one"):
        adapter.discover_source_archive(tmp_path)


def test_source_archive_duplicate_matches_fails(monkeypatch, tmp_path):
    archive, _rows_value = _source_archive(monkeypatch, tmp_path)
    duplicate = tmp_path / "input/b" / adapter.SOURCE_ARCHIVE_NAME
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(archive.read_bytes())
    with pytest.raises(adapter.AdapterError, match="found 2"):
        adapter.discover_source_archive(tmp_path / "input")


def test_source_archive_wrong_sha_fails(monkeypatch, tmp_path):
    archive, _rows_value = _source_archive(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter, "SOURCE_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(adapter.AdapterError, match="SHA drift"):
        adapter.discover_source_archive(tmp_path / "input")


def test_runtime_resource_manifest_and_argv_are_exact(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    manifest = adapter.json_object(fixture.pre_manifest, "manifest")
    adapter._validate_runtime_manifest(manifest, fixture.output)
    assert manifest["runtime"] == _runtime()
    assert manifest["resources"]["op_determinism_changed"] is False
    assert manifest["resources"]["bounded_limits"] is None
    assert manifest["automatic_retry"] is False


@pytest.mark.parametrize("reason", ["early_stopping", "max_epochs"])
def test_valid_synthetic_natural_completion_computes_namespaced_label(monkeypatch, tmp_path, reason):
    fixture = _valid_fixture(monkeypatch, tmp_path, reason=reason)
    derived = _validate(fixture)
    assert derived["completion_reason"] == reason
    assert derived["registered_decision"].startswith("CHECKPOINT_CONTINUATION_")
    assert derived["final_epoch"] == 33
    assert set(derived["overlap_diagnostics"]) == {"31", "32"}


@pytest.mark.parametrize(
    "delta,expected",
    [
        (1.0, "CHECKPOINT_CONTINUATION_PROMISING_SINGLE_SEED_VALIDATION_GAIN"),
        (0.999999, "CHECKPOINT_CONTINUATION_NO_CLEAR_SINGLE_SEED_DIFFERENCE"),
        (-0.999999, "CHECKPOINT_CONTINUATION_NO_CLEAR_SINGLE_SEED_DIFFERENCE"),
        (-1.0, "CHECKPOINT_CONTINUATION_SINGLE_SEED_VALIDATION_REGRESSION"),
    ],
)
def test_exact_registered_threshold_boundaries(delta, expected):
    assert adapter.registered_decision(delta) == expected


def test_pretrain_gate_failure_produces_no_scientific_label(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    gate_path = fixture.output / "pretrain_validation_gate.json"
    gate = adapter.json_object(gate_path, "gate")
    gate["status"] = "FAIL"
    _write_json(gate_path, gate)
    with pytest.raises(adapter.AdapterError, match="Pretrain"):
        _validate(fixture)
    evidence = fixture.adapter_root / "final_evidence.json"
    evidence.write_text("fabricated", encoding="utf-8")
    wrapper = adapter.write_failure_outputs(
        subprocess_return_code=1, error_text="gate failed",
        source_hashes_before=fixture.source_hashes, source_hashes_after=fixture.source_hashes,
        wrapper_path=fixture.adapter_root / "wrapper_execution.json",
        evidence_path=evidence,
        failure_report_path=fixture.adapter_root / "failure.md",
    )
    assert wrapper["scientific_interpretation"] is None
    assert not evidence.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_origin", "SOURCE_FIRST_RUN"),
        ("continuation_protocol_id", "wrong-protocol"),
    ],
)
def test_wrong_continuation_row_provenance_fails(field, value):
    rows = _combined(_rows(32), 33)
    rows[30][field] = value
    with pytest.raises(adapter.AdapterError, match="provenance"):
        adapter._validated_rows({"epochs": rows})


@pytest.mark.parametrize("mode", ["gap", "duplicate"])
def test_history_gap_or_duplicate_fails(mode):
    rows = _combined(_rows(32), 33)
    rows[31]["epoch"] = 34 if mode == "gap" else 31
    with pytest.raises(adapter.AdapterError, match="gap, duplicate"):
        adapter._validated_rows({"epochs": rows})


@pytest.mark.parametrize("epoch", [31, 32])
def test_original_source_epoch31_or_32_contamination_fails(monkeypatch, tmp_path, epoch):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    contaminated = dict(fixture.source_rows[epoch - 1])
    contaminated.update({
        "row_origin": adapter.CONTINUATION_ROW_ORIGIN,
        "continuation_protocol_id": adapter.CONTINUATION_PROTOCOL_ID,
    })
    rows = json.loads(json.dumps(fixture.rows))
    rows[epoch - 1] = contaminated
    with pytest.raises(adapter.AdapterError, match="contaminated"):
        adapter._validate_overlap(fixture.output, rows, fixture.source_rows)


def test_final_epoch_manifest_mismatch_fails(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    completion_path = fixture.output / adapter.COMPLETION_MARKER
    completion = adapter.json_object(completion_path, "completion")
    completion["final_completed_epoch"] = 32
    _write_json(completion_path, completion)
    with pytest.raises(adapter.AdapterError, match="final epoch"):
        _validate(fixture)


def test_tampered_generation_model_hash_fails(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    (fixture.output / "latest_states/epoch_0033/state.keras").write_bytes(b"tampered")
    with pytest.raises(adapter.AdapterError, match="model SHA"):
        adapter.validate_canonical_generation(fixture.output)


def test_tampered_generation_metadata_hash_fails(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    path = fixture.output / "latest_states/epoch_0033/state.metadata.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="metadata SHA"):
        adapter.validate_canonical_generation(fixture.output)


def test_tampered_optimizer_fingerprint_deep_validation_fails(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    fixture.deep["optimizer_state_sha256"] = "0" * 64
    with pytest.raises(adapter.AdapterError, match="optimizer_state_sha256"):
        _validate(fixture)


def test_wrong_root_best_checkpoint_epoch_fails(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    row = fixture.rows[30]
    _replace_best_metadata_and_rehash(fixture, {
        "epoch": 31,
        "validation_metrics": {
            "accuracy": row["val_accuracy"],
            "macro_f1": row["val_macro_f1"],
            "loss": row["val_loss"],
        },
    })
    with pytest.raises(adapter.AdapterError, match="earliest global max"):
        _validate(fixture)


def test_exact_tie_val_accuracy_requires_earliest_epoch(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    derived = _validate(fixture)
    assert derived["root_best_val_accuracy_checkpoint"]["epoch"] == 30
    assert fixture.rows[29]["val_accuracy"] == fixture.rows[30]["val_accuracy"]


def test_rolling_archive_contains_exact_canonical_generation_and_omits_older(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    older = fixture.output / "latest_states/epoch_0032"
    older.mkdir()
    (older / "unreferenced.bin").write_bytes(b"older")
    archive_path = tmp_path / "rolling.zip"
    names = adapter.publish_archive_atomic(
        archive_path, fixture.run_root, fixture.output, tmp_path / "report.md"
    )
    required = {
        "run/latest_state_manifest.json",
        "run/latest_states/epoch_0033/state.keras",
        "run/latest_states/epoch_0033/state.metadata.json",
        "run/latest_states/epoch_0033/history.json",
        "run/latest_states/epoch_0033/best_val_accuracy/best_val_accuracy.keras",
        "run/latest_states/epoch_0033/best_val_accuracy/best_val_accuracy.weights.h5",
        "run/latest_states/epoch_0033/best_val_accuracy/best_val_accuracy.metadata.json",
        "adapter/pre_run_manifest.json",
        "adapter/subprocess.log",
    }
    assert required.issubset(names)
    assert not any("epoch_0032" in name for name in names)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None


def test_monitor_refreshes_only_for_new_committed_manifest(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    calls = []

    def publish(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(adapter, "publish_archive_atomic", publish)
    monitor = adapter.RollingArchiveMonitor(
        manifest_path=fixture.output / adapter.LATEST_STATE_MANIFEST,
        progress_path=fixture.adapter_root / "progress.json",
        archive_path=tmp_path / "rolling.zip",
        run_root=fixture.run_root,
        train_output_root=fixture.output,
        report_path=tmp_path / "report.md",
    )
    assert monitor.poll_once() is True
    assert monitor.last_epoch == 33 and len(calls) == 1
    assert monitor.poll_once() is False
    assert len(calls) == 1


def test_atomic_archive_publication_failure_preserves_previous_zip(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    archive_path = tmp_path / "rolling.zip"
    adapter.publish_archive_atomic(
        archive_path, fixture.run_root, fixture.output, tmp_path / "report.md"
    )
    before = archive_path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("injected publication failure")

    with pytest.raises(OSError, match="injected"):
        adapter.publish_archive_atomic(
            archive_path, fixture.run_root, fixture.output, tmp_path / "report.md",
            replace=fail_replace,
        )
    assert archive_path.read_bytes() == before


class _FakeProcess:
    def __init__(self, return_code):
        self.stdout = iter(["line one\n", "line two\n"])
        self.return_code = return_code

    def wait(self):
        return self.return_code


class _IdleMonitor:
    poll_seconds = 0.01

    def run(self, stop_event):
        stop_event.wait()


def test_subprocess_failure_preserves_existing_rolling_zip_and_log(tmp_path):
    archive = tmp_path / "rolling.zip"
    archive.write_bytes(b"previous-valid-zip")
    log = tmp_path / "subprocess.log"
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeProcess(7)

    code = adapter.run_subprocess_once(
        ["python", "continuation.py"], cwd=tmp_path, log_path=log,
        monitor=_IdleMonitor(), popen_factory=factory,
    )
    assert code == 7 and len(calls) == 1
    assert log.read_text() == "line one\nline two\n"
    assert archive.read_bytes() == b"previous-valid-zip"


def test_failure_archive_contains_status_log_partial_evidence_and_no_final(monkeypatch, tmp_path):
    fixture = _valid_fixture(monkeypatch, tmp_path)
    final = fixture.adapter_root / "final_evidence.json"
    final.write_text("fabricated", encoding="utf-8")
    adapter.write_failure_outputs(
        subprocess_return_code=9, error_text="synthetic failure",
        source_hashes_before=fixture.source_hashes,
        source_hashes_after=fixture.source_hashes,
        wrapper_path=fixture.adapter_root / "wrapper_execution.json",
        evidence_path=final,
        failure_report_path=fixture.adapter_root / "technical_continuation_failure.md",
    )
    archive = tmp_path / "failure.zip"
    names = adapter.publish_archive_atomic(
        archive, fixture.run_root, fixture.output, tmp_path / "missing-report.md"
    )
    assert "adapter/wrapper_execution.json" in names
    assert "adapter/technical_continuation_failure.md" in names
    assert "adapter/subprocess.log" in names
    assert "run/history.json" in names
    assert "adapter/final_evidence.json" not in names
    wrapper = adapter.json_object(fixture.adapter_root / "wrapper_execution.json", "wrapper")
    assert wrapper["scientific_result_valid"] is False
    assert wrapper["scientific_interpretation"] is None


def test_hard_censor_partial_disposition_is_invalid_and_null():
    disposition = adapter.partial_disposition()
    assert disposition == {
        "status": "RUNTIME_HARD_CENSORED",
        "scientific_result_valid": False,
        "scientific_interpretation": None,
        "automatic_retry": False,
        "test_access": False,
    }


def test_no_auto_retry_chained_resume_or_old_harness_invocation_is_implemented():
    source = BUILDER_PATH.read_text(encoding="utf-8")
    run_source = inspect.getsource(adapter.run_registered_adapter)
    tree = ast.parse(run_source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    subprocess_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "run_subprocess_once"
    ]
    assert len(subprocess_calls) == 1
    assert "continue_from_latest_completed_state(" not in source
    assert "enable_op_determinism" not in source
    assert "automatic_retry\": True" not in source
    command_source = inspect.getsource(adapter.registered_command)
    assert "candidate_validation_harness" not in command_source
    assert "continuation_harness" in command_source


def test_input_code_opens_only_registered_train_and_val_csvs():
    notebook = adapter.build_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert '{"train": FER_TRAIN_CSV, "val": FER_VAL_CSV}' in source
    assert "FER_TEST" not in source
    assert 'PRIOR_ROOT / "test"' not in source
    assert 'CACHE_ROOT / "test"' not in source
    assert "write_predictions(" not in source
    assert "confusion_matrix(" not in source


def test_notebook_documents_exact_inputs_internet_and_output_zip():
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in adapter.build_notebook()["cells"] if cell["cell_type"] == "markdown"
    )
    assert adapter.FER_ROOT.as_posix() in markdown
    assert adapter.PRIOR_ROOT.as_posix() in markdown
    assert adapter.CACHE_ROOT.as_posix() in markdown
    assert adapter.SOURCE_ARCHIVE_NAME in markdown
    assert adapter.ARCHIVE_PATH.as_posix() in markdown
    assert "Internet" in markdown and "offline" in markdown


def test_frozen_package_diff_empty_and_git_diff_check_passes():
    frozen = "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
    changed = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", frozen],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert changed.stdout.strip() == ""
    assert subprocess.run(["git", "diff", "--check"], cwd=ROOT).returncode == 0
