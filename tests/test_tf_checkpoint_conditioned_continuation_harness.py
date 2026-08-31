from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
PACKAGE_SRC = PACKAGE_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from lap_gnn_tf.config import load_config  # noqa: E402
from lap_gnn_tf.graph.batch import load_golden_batch  # noqa: E402
from lap_gnn_tf.training.optimizer import build_optimizer  # noqa: E402
from research.candidates.tf_learned_local_residual_slots.candidate_execution import (  # noqa: E402
    build_candidate_restricted_graph_train_step,
)
from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    build_candidate_model,
)
from research.candidates.tf_learned_local_residual_slots import (  # noqa: E402
    resume_validation_only as resume,
)


BASE = "cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4"
GOLDEN = PACKAGE_ROOT / "validation_assets" / "golden" / "graph_batch.npz"
CONFIG = PACKAGE_ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"


@pytest.fixture(autouse=True)
def _restore_tensorflow_global_policy():
    policy = tf.keras.mixed_precision.global_policy().name
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows(count: int = 32) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "val_accuracy": epoch / 100.0,
            "val_macro_f1": epoch / 110.0,
            "val_loss": 2.0 - epoch / 100.0,
            "train_macro_f1": epoch / 90.0,
            "lr": 0.00015,
            "early_stopping_wait": epoch % 3,
        }
        for epoch in range(1, count + 1)
    ]


def _combined_rows(count: int) -> list[dict]:
    rows = _rows(count)
    for row in rows[30:]:
        row["row_origin"] = resume.CONTINUATION_ROW_ORIGIN
        row["continuation_protocol_id"] = resume.PROTOCOL_ID
    return rows


def _archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _synthetic_members() -> dict[str, bytes]:
    history = json.dumps({"epochs": _rows()}).encode()
    config = json.dumps(
        {
            "seed": 42,
            "locked": {
                "package_checksum": resume.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
                "execution_contract_sha256": (
                    resume.EXPECTED_BASELINE_EXECUTION_CONTRACT_SHA256
                ),
            },
        }
    ).encode()
    return {
        resume.SOURCE_CHECKPOINT_MEMBER: b"checkpoint",
        resume.SOURCE_WEIGHTS_MEMBER: b"weights",
        resume.SOURCE_METADATA_MEMBER: b"{}",
        resume.SOURCE_HISTORY_MEMBER: history,
        resume.SOURCE_CONFIG_MEMBER: config,
    }


def _patch_archive_locks(monkeypatch, archive: Path, members: dict[str, bytes]):
    monkeypatch.setattr(resume, "EXPECTED_SOURCE_ARCHIVE_SHA256", resume.sha256_file(archive))
    monkeypatch.setattr(
        resume,
        "SOURCE_MEMBER_SHA256",
        {name: _sha(payload) for name, payload in members.items()},
    )


def test_exact_registered_constants_and_source_locks_are_fail_closed():
    assert resume.IMPLEMENTATION_BASE == BASE
    assert resume.EXPECTED_SOURCE_ARCHIVE_SHA256 == (
        "2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357"
    )
    assert resume.SOURCE_MEMBER_SHA256[resume.SOURCE_CHECKPOINT_MEMBER] == (
        "818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932"
    )
    assert resume.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 == (
        "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
    )
    source_hashes = resume.verify_source_locks()
    assert {
        key: source_hashes[key] for key in resume.EXPECTED_SOURCE_LOCKS
    } == resume.EXPECTED_SOURCE_LOCKS
    assert source_hashes["checkpoint_continuation_harness"] == resume.sha256_file(
        resume.__file__
    )
    assert resume.sha256_file(CONFIG) == resume.EXPECTED_SEED42_CONFIG_SHA256


def test_wrong_source_archive_sha_fails_before_extraction(tmp_path):
    archive = _archive(tmp_path, _synthetic_members())
    with pytest.raises(resume.CheckpointContinuationError, match="archive SHA drift"):
        resume.verify_and_extract_source_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


@pytest.mark.parametrize(
    "member",
    [
        resume.SOURCE_CHECKPOINT_MEMBER,
        resume.SOURCE_HISTORY_MEMBER,
        resume.SOURCE_CONFIG_MEMBER,
    ],
)
def test_wrong_checkpoint_history_or_config_member_hash_fails_closed(
    monkeypatch, tmp_path, member
):
    members = _synthetic_members()
    archive = _archive(tmp_path, members)
    _patch_archive_locks(monkeypatch, archive, members)
    locked = dict(resume.SOURCE_MEMBER_SHA256)
    locked[member] = "0" * 64
    monkeypatch.setattr(resume, "SOURCE_MEMBER_SHA256", locked)
    with pytest.raises(resume.CheckpointContinuationError, match="member SHA drift"):
        resume.verify_and_extract_source_archive(archive, tmp_path / "extract")


def test_verified_archive_extracts_only_registered_members(monkeypatch, tmp_path):
    members = _synthetic_members()
    archive = _archive(tmp_path, {**members, "run/unregistered.txt": b"not extracted"})
    _patch_archive_locks(monkeypatch, archive, members)
    extracted = resume.verify_and_extract_source_archive(archive, tmp_path / "extract")
    assert set(extracted) == set(members)
    assert not (tmp_path / "extract" / "run" / "unregistered.txt").exists()
    assert all(resume.sha256_file(path) == _sha(members[name]) for name, path in extracted.items())


def test_source_history_requires_at_least_32_contiguous_reviewed_rows():
    with pytest.raises(resume.CheckpointContinuationError, match="at least epochs 1..32"):
        resume.split_reviewed_history({"epochs": _rows(31)})
    bad = _rows()
    bad[10]["epoch"] = 99
    with pytest.raises(resume.CheckpointContinuationError, match="not contiguous"):
        resume.split_reviewed_history({"epochs": bad})


def test_scientific_prefix_is_exactly_e1_to_e30_and_overlap_is_excluded():
    source = _rows(34)
    prefix, overlap = resume.split_reviewed_history({"epochs": source})
    assert [row["epoch"] for row in prefix] == list(range(1, 31))
    assert len(prefix) == 30
    assert set(overlap) == {31, 32}
    assert [row["epoch"] for row in overlap.values()] == [31, 32]
    assert all(row["epoch"] not in (31, 32) for row in prefix)
    source[0]["epoch"] = -1
    assert prefix[0]["epoch"] == 1


def test_load_reviewed_source_requires_locked_payload_contract_and_seed(tmp_path):
    members = _synthetic_members()
    paths = {}
    for name, payload in members.items():
        path = tmp_path / Path(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        paths[name] = path
    prefix, overlap, config = resume.load_reviewed_source(paths)
    assert len(prefix) == 30 and set(overlap) == {31, 32} and config["seed"] == 42
    bad = json.loads(members[resume.SOURCE_CONFIG_MEMBER])
    bad["locked"]["execution_contract_sha256"] = "bad"
    paths[resume.SOURCE_CONFIG_MEMBER].write_text(json.dumps(bad))
    with pytest.raises(resume.CheckpointContinuationError, match="execution contract drift"):
        resume.load_reviewed_source(paths)


@pytest.fixture(scope="module")
def candidate_and_batch():
    batch = load_golden_batch(str(GOLDEN))
    model = build_candidate_model(batch)
    return model, batch


def test_checkpoint_candidate_class_params_variables_and_q_contract(candidate_and_batch):
    model, _ = candidate_and_batch
    identity = resume.model_identity(model)
    assert identity["class"] == resume.EXPECTED_MODEL_CLASS
    assert identity["parameter_count"] == 1_061_576
    assert identity["trainable_variable_count"] == 128
    assert identity["q_index"] == 127
    assert identity["q_is_index_127"] is True
    assert identity["q_shape"] == [4, 96]
    assert identity["q_dtype"] == "float32"
    assert len(identity["q_flat_float32_sha256"]) == 64


class _FakeOptimizer:
    def __init__(self, iterations=53_822, count=262, lr=resume.EXPECTED_OPTIMIZER_LR):
        self.iterations = tf.Variable(iterations, dtype=tf.int64)
        self.learning_rate = tf.Variable(lr, dtype=tf.float32)
        self.variables = [object()] * count


def test_optimizer_identity_class_iterations_variable_count_and_lr(monkeypatch, candidate_and_batch):
    model, _ = candidate_and_batch
    optimizer = _FakeOptimizer()
    monkeypatch.setattr(model, "optimizer", optimizer, raising=False)
    monkeypatch.setattr(resume, "EXPECTED_OPTIMIZER_CLASS", "_FakeOptimizer")
    current_q = resume._q_digest(model)
    model_state, optimizer_state = resume.verify_model_optimizer_identity(
        model, expected_q_sha256=current_q
    )
    assert model_state["q_flat_float32_sha256"] == current_q
    assert optimizer_state == {
        "class": "_FakeOptimizer",
        "iterations": 53_822,
        "variable_count": 262,
        "learning_rate": pytest.approx(resume.EXPECTED_OPTIMIZER_LR, abs=1e-12),
    }


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("iterations", 1, "optimizer_iterations"),
        ("variables", [object()] * 261, "optimizer_variable_count"),
        ("learning_rate", 0.1, "optimizer_learning_rate"),
    ],
)
def test_optimizer_identity_drift_fails_closed(
    monkeypatch, candidate_and_batch, attribute, value, match
):
    model, _ = candidate_and_batch
    optimizer = _FakeOptimizer()
    if attribute in {"iterations", "learning_rate"}:
        setattr(optimizer, attribute, tf.Variable(value, dtype=tf.float32))
    else:
        setattr(optimizer, attribute, value)
    monkeypatch.setattr(model, "optimizer", optimizer, raising=False)
    monkeypatch.setattr(resume, "EXPECTED_OPTIMIZER_CLASS", "_FakeOptimizer")
    with pytest.raises(resume.CheckpointContinuationError, match=match):
        resume.verify_model_optimizer_identity(
            model, expected_q_sha256=resume._q_digest(model)
        )


def test_reconstructed_early_scheduler_and_checkpoint_policy_state_is_exact(tmp_path):
    optimizer = _FakeOptimizer()
    config = load_config(CONFIG)
    early, scheduler, policy = resume.reconstruct_control_state(optimizer, config, tmp_path)
    assert early.min_epochs == 30 and early.patience == 15
    assert early.get_state() == resume.EARLY_STATE_POST_E30
    assert scheduler.get_state() == resume.SCHEDULER_STATE_POST_E30
    assert scheduler.current_lr == pytest.approx(resume.EXPECTED_OPTIMIZER_LR, abs=1e-12)
    assert resume.checkpoint_policy_state(policy) == resume.CHECKPOINT_POLICY_POST_E30


def test_registered_resources_are_exact_and_drift_fails_closed():
    controls = resume.ResourceControls()
    resume.verify_resource_controls(controls)
    controls.batch_size = 17
    with pytest.raises(resume.CheckpointContinuationError, match="resource drift"):
        resume.verify_resource_controls(controls)


def test_scheduler_reconstruction_does_not_replay_epoch30(monkeypatch, tmp_path):
    calls = []
    original = resume.TorchCompatibleReduceLROnPlateau.step

    def recording_step(self, metric):
        calls.append(metric)
        return original(self, metric)

    monkeypatch.setattr(resume.TorchCompatibleReduceLROnPlateau, "step", recording_step)
    resume.reconstruct_control_state(_FakeOptimizer(), load_config(CONFIG), tmp_path)
    assert calls == []


def test_original_epoch30_checkpoint_trio_is_copied_atomically(monkeypatch, tmp_path):
    extracted = {}
    hashes = {}
    for member in (
        resume.SOURCE_CHECKPOINT_MEMBER,
        resume.SOURCE_WEIGHTS_MEMBER,
        resume.SOURCE_METADATA_MEMBER,
        resume.SOURCE_CONFIG_MEMBER,
    ):
        path = tmp_path / "extracted" / Path(*member.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(member.encode())
        extracted[member] = path
        hashes[member] = resume.sha256_file(path)
    hashes[resume.SOURCE_HISTORY_MEMBER] = resume.SOURCE_MEMBER_SHA256[
        resume.SOURCE_HISTORY_MEMBER
    ]
    monkeypatch.setattr(resume, "SOURCE_MEMBER_SHA256", hashes)
    output = resume.initialize_output(
        tmp_path / "output", extracted, _rows(30), {31: _rows()[30], 32: _rows()[31]},
        {"source": "a" * 64}, CONFIG,
    )
    for member, name in (
        (resume.SOURCE_CHECKPOINT_MEMBER, "best_val_accuracy.keras"),
        (resume.SOURCE_WEIGHTS_MEMBER, "best_val_accuracy.weights.h5"),
        (resume.SOURCE_METADATA_MEMBER, "best_val_accuracy.metadata.json"),
    ):
        assert resume.sha256_file(output / "checkpoints" / name) == hashes[member]
    assert not list(output.rglob("*.tmp"))
    history = json.loads((output / "history.json").read_text())["epochs"]
    assert [row["epoch"] for row in history] == list(range(1, 31))


def test_pretrain_validation_gate_passes_exact_fixture():
    metrics = {
        **resume.PRETRAIN_VALIDATION_REFERENCE,
        "support_per_class": [500, 500, 500, 500, 500, 500, 589],
    }
    evidence = resume.validate_pretrain_validation_gate(metrics)
    assert evidence["status"] == "PASS"
    assert evidence["optimizer_updates_before_gate"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"support_per_class": [3588]},
        {"accuracy": resume.PRETRAIN_VALIDATION_REFERENCE["accuracy"] + 0.00101},
        {"macro_f1": resume.PRETRAIN_VALIDATION_REFERENCE["macro_f1"] - 0.00101},
        {"loss": resume.PRETRAIN_VALIDATION_REFERENCE["loss"] + 0.00501},
    ],
)
def test_pretrain_validation_gate_fails_closed_without_retry(change):
    metrics = {
        **resume.PRETRAIN_VALIDATION_REFERENCE,
        "support_per_class": [3589],
        **change,
    }
    with pytest.raises(resume.CheckpointContinuationError, match="failed closed"):
        resume.validate_pretrain_validation_gate(metrics)


def test_overlap_rows_are_descriptive_only_and_cannot_drive_controls():
    first = _rows()[30]
    resumed = {**first, "val_accuracy": first["val_accuracy"] + 0.1}
    diagnostic = resume.overlap_diagnostic(resumed, first)
    assert diagnostic["classification"] == "FIRST_RUN_OVERLAP_DIAGNOSTICS"
    assert diagnostic["descriptive_only"] is True
    assert diagnostic["delta_val_accuracy"] == pytest.approx(0.1)
    for key in (
        "affects_training",
        "affects_stopping",
        "affects_scheduler",
        "affects_checkpoint_selection",
        "affects_primary_endpoint",
        "triggers_retry",
    ):
        assert diagnostic[key] is False


class _Dataset:
    def __init__(self, name, events, batches=1):
        self.name = name
        self.events = events
        self.batches = batches

    def __len__(self):
        return self.batches

    def as_dataset(self, epoch, prefetch=None):
        self.events.append((self.name, epoch, prefetch))
        return [SimpleNamespace()] * self.batches


class _Control:
    patience = 15

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.state = dict(resume.EARLY_STATE_POST_E30)

    def update(self, epoch, value):
        self.events.append((self.name, epoch, value))
        self.state["epochs_without_improvement"] += 1
        return False

    def get_state(self):
        return dict(self.state)


class _Scheduler:
    def __init__(self, events):
        self.events = events
        self.state = dict(resume.SCHEDULER_STATE_POST_E30)

    def step(self, value):
        self.events.append(("scheduler", value))
        self.state["last_epoch"] += 1
        return resume.EXPECTED_OPTIMIZER_LR

    def get_state(self):
        return dict(self.state)


class _Policy:
    best_macro = resume.CHECKPOINT_POLICY_POST_E30["best_macro"]
    best_macro_epoch = 30
    best_accuracy = resume.CHECKPOINT_POLICY_POST_E30["best_accuracy"]
    best_accuracy_epoch = 30

    def __init__(self, events):
        self.events = events

    def update_best(self, model, optimizer, epoch, metrics, metadata):
        self.events.append(("checkpoint", epoch, metrics["accuracy"]))
        return {"saved": [], "best_macro_epoch": 30, "best_accuracy_epoch": 30}


def test_bounded_epoch_loop_starts_31_uses_epoch_index_and_preserves_order(
    monkeypatch, tmp_path
):
    events = []
    optimizer = _FakeOptimizer(iterations=53_822)
    model = SimpleNamespace(optimizer=optimizer)
    metrics = {
        "loss": 1.2,
        "accuracy": 0.6,
        "macro_f1": 0.56,
    }

    def evaluate(*args, **kwargs):
        events.append(("evaluate",))
        return dict(metrics)

    def execute(batch):
        del batch
        events.append(("train_step",))
        optimizer.iterations.assign_add(1)
        return tf.constant(1.0)

    def persist(output, history, row):
        events.append(("history_persist", row["epoch"]))
        resume._atomic_json(output / "history.json", {"epochs": history})

    def publish(**kwargs):
        events.append(("latest_state", kwargs["completed_epoch"]))
        return {}

    monkeypatch.setattr(resume, "evaluate_batches", evaluate)
    monkeypatch.setattr(resume, "_persist_history", persist)
    monkeypatch.setattr(resume, "_print_epoch_summary", lambda *args: events.append(("summary",)))
    config = load_config(CONFIG)
    output = tmp_path / "run"
    output.mkdir()
    history = _rows(30)
    prefix_snapshot = json.loads(json.dumps(history))
    result = resume.run_continuation_epoch_loop(
        model=model,
        optimizer=optimizer,
        execute_train_step=execute,
        train_data=_Dataset("train", events),
        val_data=_Dataset("val", events),
        train_eval_data=_Dataset("train_eval", events),
        eval_step=lambda batch: batch,
        controls=resume.ResourceControls(tf_data_prefetch=2),
        config=config,
        execution_state={"optimizer_execution_mode": "restricted_tf_function"},
        telemetry=resume.RuntimeTelemetry(),
        early=_Control("early", events),
        scheduler=_Scheduler(events),
        policy=_Policy(events),
        history=history,
        overlap_rows={31: _rows()[30], 32: _rows()[31]},
        output_root=output,
        source_hashes={"source": "a" * 64},
        start_epoch=resume.FIRST_CONTINUATION_EPOCH,
        max_epoch=32,
        latest_state_publisher=publish,
    )
    assert result[:30] == prefix_snapshot
    assert [row["epoch"] for row in result[-2:]] == [31, 32]
    assert [event[:2] for event in events if event[0] in {"train", "val", "train_eval"}] == [
        ("train", 31), ("val", 31), ("train_eval", 31),
        ("train", 32), ("val", 32), ("train_eval", 32),
    ]
    labels = [event[0] for event in events]
    for start in (0, labels.index("train", 1)):
        segment = labels[start:]
        assert segment.index("train_step") < segment.index("evaluate")
        assert segment.index("early") < segment.index("checkpoint")
        assert segment.index("checkpoint") < segment.index("scheduler")
        assert segment.index("scheduler") < segment.index("history_persist")
        assert segment.index("history_persist") < segment.index("summary")
        assert segment.index("summary") < segment.index("latest_state")
    assert int(optimizer.iterations.numpy()) == 53_824
    assert (output / "resume_overlap_epoch31.json").is_file()
    assert (output / "resume_overlap_epoch32.json").is_file()
    assert not any((output / name).exists() for name in resume.FORBIDDEN_ARTIFACT_NAMES)


@pytest.mark.parametrize(
    "history,start_epoch,match",
    [
        ([], 31, "non-empty"),
        (_rows(30), 30, "at least 31"),
        (_combined_rows(32), 32, "contiguous through start_epoch"),
        (_combined_rows(32), 34, "contiguous through start_epoch"),
        (_rows(31), 32, "non-continuation row"),
        (
            [
                *(_rows(30)),
                {
                    **_rows(32)[31],
                    "row_origin": resume.CONTINUATION_ROW_ORIGIN,
                    "continuation_protocol_id": resume.PROTOCOL_ID,
                },
            ],
            32,
            "contiguous through start_epoch",
        ),
        (
            [
                *_combined_rows(31),
                {**_combined_rows(31)[-1]},
            ],
            33,
            "contiguous through start_epoch",
        ),
    ],
)
def test_invalid_continuation_start_epoch_or_history_fails_closed(
    history, start_epoch, match
):
    with pytest.raises(resume.CheckpointContinuationError, match=match):
        resume.validate_continuation_start(history, start_epoch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_origin", "SOURCE_FIRST_RUN"),
        ("continuation_protocol_id", "wrong-protocol"),
    ],
)
def test_continuation_start_rejects_wrong_row_provenance(field, value):
    history = _combined_rows(31)
    history[30][field] = value
    with pytest.raises(
        resume.CheckpointContinuationError, match="non-continuation row"
    ):
        resume.validate_continuation_start(history, 32)


def test_initial_and_chained_paths_share_loop_with_explicit_start_epochs():
    initial = inspect.getsource(resume.run_checkpoint_conditioned_continuation)
    chained = inspect.getsource(resume.continue_from_latest_completed_state)
    assert "run_continuation_epoch_loop(" in initial
    assert "start_epoch=FIRST_CONTINUATION_EPOCH" in initial
    assert resume.FIRST_CONTINUATION_EPOCH == 31
    assert "run_continuation_epoch_loop(" in chained
    assert "start_epoch=state.next_epoch" in chained
    assert "run_checkpoint_conditioned_continuation(" not in chained
    for forbidden in (
        "initialize_output(",
        "verify_and_extract_source_archive(",
        "load_resume_checkpoint(",
        "validate_pretrain_validation_gate(",
    ):
        assert forbidden not in chained


def test_chained_overlap_rows_are_verified_only_while_applicable(tmp_path):
    payload = {
        "schema_version": 1,
        "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS",
        "descriptive_only": True,
        "excluded_from_combined_scientific_history": True,
        "rows": {"31": _rows()[30], "32": _rows()[31]},
    }
    resume._atomic_json(tmp_path / resume.OVERLAP_SOURCE_NAME, payload)
    overlap = resume._load_applicable_overlap_rows(tmp_path, 32)
    assert set(overlap) == {31, 32}
    assert resume._load_applicable_overlap_rows(tmp_path / "missing", 33) == {}
    payload["descriptive_only"] = False
    resume._atomic_json(tmp_path / resume.OVERLAP_SOURCE_NAME, payload)
    with pytest.raises(resume.CheckpointContinuationError, match="contract drift"):
        resume._load_applicable_overlap_rows(tmp_path, 32)


def test_candidate_g1a_adapter_is_directly_used_by_production_entrypoint():
    source = inspect.getsource(resume.run_checkpoint_conditioned_continuation)
    assert "build_candidate_restricted_graph_train_step(" in source
    assert "GraphBatchGenerator.output_signature()" in source
    tree = ast.parse(source)
    split_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "train" in split_literals and "val" in split_literals
    assert "test" not in split_literals


def test_bounded_candidate_q_gradient_update_and_optimizer_iteration(candidate_and_batch):
    model, batch = candidate_and_batch
    config = load_config(CONFIG)
    optimizer = build_optimizer(config)
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, run_eagerly=False)
    step = build_candidate_restricted_graph_train_step(model, optimizer)
    q_before = model.learned_local_residual_slots.Q.numpy().copy()
    iteration_before = int(optimizer.iterations.numpy())
    losses = []
    for _ in range(12):
        losses.append(step(batch))
        if int(optimizer.iterations.numpy()) > iteration_before:
            break
    assert all(bool(tf.math.is_finite(loss).numpy()) for loss in losses)
    assert int(optimizer.iterations.numpy()) == iteration_before + 1
    assert not np.array_equal(q_before, model.learned_local_residual_slots.Q.numpy())


def _build_roundtrip_candidate():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    batch = load_golden_batch(str(GOLDEN))
    model = build_candidate_model(batch)
    optimizer = build_optimizer(load_config(CONFIG))
    optimizer.build(model.trainable_variables)
    optimizer.learning_rate.assign(resume.EXPECTED_OPTIMIZER_LR)
    model.compile(optimizer=optimizer, run_eagerly=False)
    return model


def _roundtrip_controls(tmp_path, optimizer):
    early, scheduler, policy = resume.reconstruct_control_state(
        optimizer, load_config(CONFIG), tmp_path
    )
    return early, scheduler, policy


def _controls_at_epoch(output_root, optimizer, completed_epoch):
    early, scheduler, policy = _roundtrip_controls(output_root, optimizer)
    for epoch in range(31, completed_epoch + 1):
        value = resume.EARLY_STATE_POST_E30["best"] + epoch / 1000.0
        early.update(epoch, value)
        scheduler.step(value)
    return early, scheduler, policy


def _prepare_generation_output(output_root: Path, completed_epoch: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    resume._atomic_json(output_root / "resolved_config.json", config)
    prefix = _rows(30)
    resume._atomic_json(
        output_root / "continuation_pre_run_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": resume.PROTOCOL_ID,
            "source_archive_sha256": resume.EXPECTED_SOURCE_ARCHIVE_SHA256,
            "source_history_sha256": resume.SOURCE_MEMBER_SHA256[
                resume.SOURCE_HISTORY_MEMBER
            ],
            "immutable_scientific_prefix_sha256": resume.immutable_prefix_sha256(
                prefix
            ),
        },
    )
    resume._atomic_json(
        output_root / "history.json", {"epochs": _combined_rows(completed_epoch)}
    )
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir()
    for index, name in enumerate(resume.GENERATION_BEST_ARTIFACT_NAMES):
        (checkpoint_root / name).write_bytes(f"best-{index}".encode())


def _replace_combined_history(output_root: Path, completed_epoch: int) -> None:
    resume._atomic_json(
        output_root / "history.json", {"epochs": _combined_rows(completed_epoch)}
    )


def _copy_saved_state(source: Path):
    def save_model(_model, destination):
        shutil.copyfile(source, destination)

    return save_model


def test_optimizer_full_state_fingerprint_roundtrip_and_slot_tamper(tmp_path):
    model = _build_roundtrip_candidate()
    before = resume.optimizer_state_fingerprint(model.optimizer)
    state_path = tmp_path / "fingerprint.keras"
    resume.save_model_with_optimizer(model, state_path)
    restored = tf.keras.models.load_model(state_path, compile=True)
    after = resume.optimizer_state_fingerprint(restored.optimizer)
    assert before == after
    assert before["variable_count"] == 262
    assert len(before["variables"]) == 262

    readable_before = resume.optimizer_identity(restored.optimizer)
    slot = next(
        variable
        for variable in restored.optimizer.variables
        if any(
            token in str(getattr(variable, "path", "")).lower()
            for token in ("momentum", "velocity")
        )
    )
    slot.assign_add(tf.ones_like(slot) * tf.cast(0.125, slot.dtype))
    readable_after = resume.optimizer_identity(restored.optimizer)
    tampered = resume.optimizer_state_fingerprint(restored.optimizer)
    assert readable_after == readable_before
    assert tampered["sha256"] != before["sha256"]
    assert tampered["variable_count"] == before["variable_count"]


def test_generation_manifest_roundtrip_restores_fresh_model_optimizer_and_controls(
    tmp_path,
):
    model = _build_roundtrip_candidate()
    _prepare_generation_output(tmp_path, 31)
    early, scheduler, policy = _controls_at_epoch(tmp_path, model.optimizer, 31)
    manifest = resume.publish_latest_completed_state(
        model=model,
        completed_epoch=31,
        scheduler=scheduler,
        early=early,
        policy=policy,
        history_path=tmp_path / "history.json",
        output_root=tmp_path,
        source_hashes={"harness": "a" * 64},
    )
    manifest_path = tmp_path / resume.LATEST_STATE_MANIFEST_NAME
    assert manifest_path.is_file()
    assert manifest["generation_id"] == "epoch_0031"
    assert manifest["partial_epoch"] is False and manifest["test_access"] is False
    generation = tmp_path / manifest["generation_relative_path"]
    assert (generation / resume.GENERATION_MODEL_NAME).is_file()
    assert (generation / resume.GENERATION_METADATA_NAME).is_file()
    assert (generation / resume.GENERATION_HISTORY_NAME).is_file()
    current_best = tmp_path / "checkpoints" / "best_val_accuracy.keras"
    current_best.write_bytes(b"partial-newer-epoch-checkpoint")

    restored = resume.load_latest_completed_state(tmp_path)
    assert restored.model is not model
    assert restored.optimizer is not model.optimizer
    assert restored.early is not early
    assert restored.scheduler is not scheduler
    assert restored.policy is not policy
    assert restored.completed_epoch == 31 and restored.next_epoch == 32
    assert restored.early.get_state() == early.get_state()
    assert restored.scheduler.get_state() == scheduler.get_state()
    assert resume.checkpoint_policy_state(restored.policy) == resume.checkpoint_policy_state(
        policy
    )
    assert resume.optimizer_state_fingerprint(restored.optimizer) == (
        resume.optimizer_state_fingerprint(model.optimizer)
    )
    assert [row["epoch"] for row in restored.history] == list(range(1, 32))
    assert len(restored.best_checkpoint_paths) == 3
    canonical_best = (
        generation
        / resume.GENERATION_BEST_DIRECTORY_NAME
        / "best_val_accuracy.keras"
    )
    assert resume.sha256_file(current_best) == resume.sha256_file(canonical_best)


def test_generation_atomicity_fault_injection_preserves_epoch31_manifest(tmp_path):
    model = _build_roundtrip_candidate()
    cached_state = tmp_path / "cached.keras"
    resume.save_model_with_optimizer(model, cached_state)
    save_cached = _copy_saved_state(cached_state)

    scenarios = (
        "before_generation_publish",
        "after_generation_publish_before_manifest_write",
        "during_temporary_manifest_write",
    )
    for scenario in scenarios:
        output = tmp_path / scenario
        _prepare_generation_output(output, 31)
        early31, scheduler31, policy31 = _controls_at_epoch(output, model.optimizer, 31)
        resume.publish_latest_completed_state(
            model=model,
            completed_epoch=31,
            scheduler=scheduler31,
            early=early31,
            policy=policy31,
            history_path=output / "history.json",
            output_root=output,
            source_hashes={},
            save_model=save_cached,
        )
        manifest_path = output / resume.LATEST_STATE_MANIFEST_NAME
        manifest_before = manifest_path.read_bytes()
        _replace_combined_history(output, 32)
        early32, scheduler32, policy32 = _controls_at_epoch(output, model.optimizer, 32)

        def fault(point):
            if point == scenario:
                raise OSError(f"injected {scenario}")

        def partial_manifest_writer(path, payload):
            del payload
            Path(path).write_bytes(b'{"partial":')
            raise OSError("injected during_temporary_manifest_write")

        with pytest.raises(OSError, match="injected"):
            resume.publish_latest_completed_state(
                model=model,
                completed_epoch=32,
                scheduler=scheduler32,
                early=early32,
                policy=policy32,
                history_path=output / "history.json",
                output_root=output,
                source_hashes={},
                save_model=save_cached,
                fault_injector=(None if scenario.startswith("during") else fault),
                manifest_writer=(
                    partial_manifest_writer
                    if scenario.startswith("during")
                    else resume._write_json_fsync
                ),
            )
        assert manifest_path.read_bytes() == manifest_before
        restored = resume.load_latest_completed_state(output)
        assert restored.completed_epoch == 31 and restored.next_epoch == 32
        assert restored.generation_path.name == "epoch_0031"
        epoch32 = output / resume.LATEST_STATES_DIRECTORY_NAME / "epoch_0032"
        assert epoch32.exists() is (scenario != "before_generation_publish")


def test_successful_manifest_switch_restores_epoch32_and_rejects_mixed_generation(
    tmp_path,
):
    model = _build_roundtrip_candidate()
    _prepare_generation_output(tmp_path, 31)
    early31, scheduler31, policy31 = _controls_at_epoch(tmp_path, model.optimizer, 31)
    resume.publish_latest_completed_state(
        model=model,
        completed_epoch=31,
        scheduler=scheduler31,
        early=early31,
        policy=policy31,
        history_path=tmp_path / "history.json",
        output_root=tmp_path,
        source_hashes={},
    )
    _replace_combined_history(tmp_path, 32)
    early32, scheduler32, policy32 = _controls_at_epoch(tmp_path, model.optimizer, 32)
    resume.publish_latest_completed_state(
        model=model,
        completed_epoch=32,
        scheduler=scheduler32,
        early=early32,
        policy=policy32,
        history_path=tmp_path / "history.json",
        output_root=tmp_path,
        source_hashes={},
    )
    restored = resume.load_latest_completed_state(tmp_path)
    assert restored.completed_epoch == 32 and restored.next_epoch == 33
    assert (tmp_path / resume.LATEST_STATES_DIRECTORY_NAME / "epoch_0031").is_dir()
    assert (tmp_path / resume.LATEST_STATES_DIRECTORY_NAME / "epoch_0032").is_dir()

    manifest_path = tmp_path / resume.LATEST_STATE_MANIFEST_NAME
    mixed = resume._json_object(manifest_path, "manifest")
    mixed["metadata_relative_path"] = (
        f"{resume.LATEST_STATES_DIRECTORY_NAME}/epoch_0031/"
        f"{resume.GENERATION_METADATA_NAME}"
    )
    resume._atomic_json(manifest_path, mixed)
    with pytest.raises(resume.CheckpointContinuationError, match="generation path drift"):
        resume.load_latest_completed_state(tmp_path)


def test_restored_epoch32_executes_same_production_loop_from_epoch33(
    monkeypatch, tmp_path
):
    output = tmp_path / "continued"
    model = _build_roundtrip_candidate()
    _prepare_generation_output(output, 32)
    fer_csv = tmp_path / "fer2013.csv"
    fer_csv.write_text("emotion,pixels,Usage\n", encoding="utf-8")
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    controls = resume.ResourceControls()
    runtime_config = resume._registered_runtime_config(
        load_config(CONFIG),
        fer_csv=fer_csv,
        prior_root=prior_root,
        controls=controls,
    )
    resume._atomic_json(output / "resolved_config.json", runtime_config)
    early, scheduler, policy = _controls_at_epoch(output, model.optimizer, 32)
    early.set_state(
        {
            "best": early.get_state()["best"],
            "epochs_without_improvement": early.patience - 1,
        }
    )
    resume.publish_latest_completed_state(
        model=model,
        completed_epoch=32,
        scheduler=scheduler,
        early=early,
        policy=policy,
        history_path=output / "history.json",
        output_root=output,
        source_hashes={"harness": "a" * 64},
    )

    restored = resume.load_latest_completed_state(output)
    assert restored.completed_epoch == 32
    assert restored.next_epoch == 33
    assert [row["epoch"] for row in restored.history] == list(range(1, 33))

    dataset_events = []

    class BoundedGraphBatchGenerator:
        def __init__(
            self,
            _prior_root,
            split,
            _config,
            _batch_size,
            _seed,
            training,
            _cache_size,
            _telemetry,
            **_kwargs,
        ):
            self.name = "train" if training else ("val" if split == "val" else "train_eval")

        def __len__(self):
            return 1

        def as_dataset(self, epoch, prefetch=None):
            dataset_events.append((self.name, epoch, prefetch))
            return [SimpleNamespace()]

        @staticmethod
        def output_signature():
            return "bounded-signature"

    seed_calls = []
    loader_calls = []
    original_loader = resume.load_latest_completed_state

    def tracking_loader(path):
        loader_calls.append(Path(path))
        return original_loader(path)

    def bounded_train_step(_batch):
        state_model = active_state["model"]
        state_model.optimizer.iterations.assign_add(1)
        return tf.constant(2.0, dtype=tf.float32)

    active_state = {}

    def build_train_step(model_arg, optimizer_arg, *, input_signature):
        assert model_arg.optimizer is optimizer_arg
        assert input_signature == "bounded-signature"
        active_state["model"] = model_arg
        return bounded_train_step

    monkeypatch.setattr(resume, "load_latest_completed_state", tracking_loader)
    monkeypatch.setattr(resume, "GraphBatchGenerator", BoundedGraphBatchGenerator)
    monkeypatch.setattr(
        resume, "build_candidate_restricted_graph_train_step", build_train_step
    )
    monkeypatch.setattr(resume, "build_compiled_evaluation_step", lambda _model: object())
    monkeypatch.setattr(
        resume,
        "evaluate_batches",
        lambda *_args, **_kwargs: {
            "loss": 2.5,
            "accuracy": 0.1,
            "macro_f1": 0.1,
        },
    )
    monkeypatch.setattr(
        resume,
        "validate_execution_config",
        lambda _training: {"optimizer_execution_mode": "restricted_tf_function"},
    )
    monkeypatch.setattr(resume, "seed_everything", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(resume.ResourceControls, "apply", lambda _self: None)
    monkeypatch.setattr(resume, "write_training_curves", lambda *_args: None)
    monkeypatch.setattr(resume, "_print_epoch_summary", lambda *_args: None)

    marker = resume.continue_from_latest_completed_state(
        config_path=CONFIG,
        fer_csv=fer_csv,
        prior_root=prior_root,
        output_root=output,
        controls=controls,
    )

    assert seed_calls == [42]
    assert loader_calls == [output.resolve()]
    assert [event[:2] for event in dataset_events] == [
        ("train", 33),
        ("val", 33),
        ("train_eval", 33),
    ]
    history = json.loads((output / "history.json").read_text())["epochs"]
    assert [row["epoch"] for row in history] == list(range(1, 34))
    assert history[-1]["row_origin"] == resume.CONTINUATION_ROW_ORIGIN
    assert history[-1]["continuation_protocol_id"] == resume.PROTOCOL_ID
    latest = json.loads(
        (output / resume.LATEST_STATE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert latest["completed_epoch"] == 33
    assert latest["next_epoch"] == 34
    assert latest["generation_id"] == "epoch_0033"
    assert (output / resume.LATEST_STATES_DIRECTORY_NAME / "epoch_0033").is_dir()
    assert marker["final_completed_epoch"] == 33
    assert marker["scientific_result_valid"] is False
    assert marker["scientific_interpretation"] is None
    assert marker["test_access"] is False


def test_loader_rejects_path_traversal_and_source_first_run_epoch31(tmp_path):
    with pytest.raises(
        resume.CheckpointContinuationError, match="non-continuation row"
    ):
        resume.validate_combined_history({"epochs": _rows(31)}, 31)
    output = tmp_path.resolve()
    with pytest.raises(resume.CheckpointContinuationError, match="path traversal"):
        resume._resolve_relative_under(output, "../outside.keras", "model")


def test_loader_rejects_rehashed_model_with_tampered_optimizer_slot(tmp_path):
    model = _build_roundtrip_candidate()
    _prepare_generation_output(tmp_path, 31)
    early, scheduler, policy = _controls_at_epoch(tmp_path, model.optimizer, 31)
    manifest = resume.publish_latest_completed_state(
        model=model,
        completed_epoch=31,
        scheduler=scheduler,
        early=early,
        policy=policy,
        history_path=tmp_path / "history.json",
        output_root=tmp_path,
        source_hashes={},
    )
    model_path = tmp_path / manifest["model_relative_path"]
    tampered_model = tf.keras.models.load_model(model_path, compile=True)
    readable_before = resume.optimizer_identity(tampered_model.optimizer)
    slot = next(
        variable
        for variable in tampered_model.optimizer.variables
        if "momentum" in str(getattr(variable, "path", "")).lower()
    )
    slot.assign_add(tf.ones_like(slot) * tf.cast(0.25, slot.dtype))
    assert resume.optimizer_identity(tampered_model.optimizer) == readable_before
    temporary_model = model_path.with_name("tampered.keras")
    resume.save_model_with_optimizer(tampered_model, temporary_model)
    temporary_model.replace(model_path)

    metadata_path = tmp_path / manifest["metadata_relative_path"]
    metadata = resume._json_object(metadata_path, "metadata")
    metadata["state_keras_sha256"] = resume.sha256_file(model_path)
    resume._atomic_json(metadata_path, metadata)
    manifest["model_sha256"] = resume.sha256_file(model_path)
    manifest["metadata_sha256"] = resume.sha256_file(metadata_path)
    resume._atomic_json(tmp_path / resume.LATEST_STATE_MANIFEST_NAME, manifest)
    with pytest.raises(
        resume.CheckpointContinuationError,
        match="full optimizer state fingerprint drift",
    ):
        resume.load_latest_completed_state(tmp_path)


def test_no_test_lifecycle_or_scientific_outcome_is_implemented():
    source = Path(resume.__file__).read_text(encoding="utf-8")
    forbidden_calls = (
        "resolve_final_checkpoint(",
        'GraphBatchGenerator(prior_root, "test"',
        "write_predictions(",
        "write_confusion_matrix(",
        "write_per_class_metrics(",
    )
    assert all(call not in source for call in forbidden_calls)
    assert "scientific_result_valid\": False" in source
    assert "scientific_interpretation\": None" in source
    assert "automatic_retry\": False" in source


def test_continuation_epoch_body_has_registered_structural_order():
    source = inspect.getsource(resume.run_continuation_epoch_loop)
    tokens = [
        "execute_train_step(batch)",
        "val_metrics = evaluate_batches(",
        "train_metrics = evaluate_batches(",
        "stop = early.update(",
        "checkpoint = policy.update_best(",
        "lr = scheduler.step(",
        "_persist_history(",
        "_print_epoch_summary(",
        "latest_state_publisher(",
    ]
    positions = [source.index(token) for token in tokens]
    assert positions == sorted(positions)


def test_frozen_package_is_unchanged_from_exact_base_and_diff_is_clean():
    frozen = PACKAGE_ROOT.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", frozen],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == ""
    assert subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, check=False
    ).returncode == 0
