from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
from lap_gnn_tf.model import LapGNN, build_model  # noqa: E402
from lap_gnn_tf.training import trainer  # noqa: E402
from lap_gnn_tf.training.execution import (  # noqa: E402
    build_compiled_gradient_function,
    build_restricted_graph_train_step,
    validate_execution_config,
)
from lap_gnn_tf.training.optimizer import build_optimizer  # noqa: E402
from research.candidates.tf_learned_local_residual_slots.candidate_execution import (  # noqa: E402
    EXPECTED_CANDIDATE_TRAINABLE_VARIABLE_COUNT,
    build_candidate_restricted_graph_train_step,
)
from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    LearnedLocalResidualSlotLapGNN,
    build_candidate_model,
)
from research.candidates.tf_learned_local_residual_slots import (  # noqa: E402
    train_validation_only as harness,
)


BASE = "572885a0bb650434f5b36bd3be2049524377067b"
GOLDEN = PACKAGE_ROOT / "validation_assets" / "golden"
SEED42_CONFIG = (
    PACKAGE_ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
)
HARNESS_PATH = (
    ROOT
    / "research"
    / "candidates"
    / "tf_learned_local_residual_slots"
    / "train_validation_only.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def golden_batch():
    return load_golden_batch(str(GOLDEN / "graph_batch.npz"))


@pytest.fixture(scope="module")
def ordered_models(golden_batch):
    baseline = build_model(golden_batch)
    candidate = build_candidate_model(golden_batch)
    return baseline, candidate


def test_exact_source_locks_contract_and_frozen_package_diff_are_clean():
    assert harness._verify_source_locks(require_original_bindings=True) == {
        "candidate_model": harness.EXPECTED_CANDIDATE_MODEL_SHA256,
        "candidate_execution_adapter": harness.EXPECTED_CANDIDATE_EXECUTION_SHA256,
        "candidate_execution_contract": (
            harness.EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256
        ),
        "frozen_validation_only_wrapper": harness.EXPECTED_WRAPPER_SHA256,
        "frozen_trainer": harness.EXPECTED_TRAINER_SHA256,
        "frozen_execution": harness.EXPECTED_FROZEN_EXECUTION_SHA256,
    }
    contract = harness._verify_candidate_execution_contract()
    assert contract["contract_id"] == harness.CANDIDATE_EXECUTION_CONTRACT_ID
    assert contract["selected_mode"] == "restricted_tf_function"
    assert contract["selected_grappler_profile"] == "G1-A"
    assert contract["eager_exact"]["status"] == "unsupported_out_of_scope"
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=ROOT, check=False
    ).returncode == 0
    frozen_relative = PACKAGE_ROOT.relative_to(ROOT).as_posix()
    assert subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", frozen_relative],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip() == ""


def test_candidate_variable_order_is_exact_baseline_prefix_then_q(ordered_models):
    baseline, candidate = ordered_models
    baseline_variables = baseline.trainable_variables
    candidate_variables = candidate.trainable_variables
    assert baseline.count_params() == harness.BASELINE_PARAMETER_COUNT
    assert candidate.count_params() == harness.CANDIDATE_PARAMETER_COUNT
    assert len(baseline_variables) == harness.BASELINE_VARIABLE_PREFIX_COUNT
    assert len(candidate_variables) == harness.CANDIDATE_VARIABLE_COUNT
    for index, (expected, actual) in enumerate(
        zip(baseline_variables, candidate_variables[:127])
    ):
        assert actual.name == expected.name, index
        assert getattr(actual, "path", None) == getattr(expected, "path", None), index
        assert tuple(actual.shape) == tuple(expected.shape), index
        assert str(actual.dtype) == str(expected.dtype), index
    q = candidate.learned_local_residual_slots.Q
    assert candidate_variables[127] is q
    assert getattr(q, "path", "").endswith("learned_local_residual_slot_pool/Q")
    assert tuple(q.shape) == (4, 96)
    assert str(q.dtype) == "float32"


class _VariableCountModel(tf.keras.Model):
    def __init__(self, count: int):
        super().__init__(name=f"variable_count_model_{count}")
        self.contract_variables = [
            self.add_weight(name=f"w_{index}", shape=(), initializer="ones")
            for index in range(count)
        ]

    def call(self, batch, training=False):
        del training
        value = tf.add_n(self.contract_variables)
        samples = tf.shape(batch["labels"])[0]
        logits = tf.tile(tf.reshape(tf.stack((value, -value)), (1, 2)), (samples, 1))
        return {"logits": logits}


@pytest.mark.parametrize("count", [127, 129])
def test_candidate_builder_rejects_nonregistered_variable_counts(count):
    model = _VariableCountModel(count)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    step = build_candidate_restricted_graph_train_step(model, optimizer)
    batch = {"labels": tf.constant([0, 1], dtype=tf.int64)}
    with pytest.raises(RuntimeError, match=f"Expected 128 trainable variables, got {count}"):
        step(batch)


def test_candidate_builder_accepts_exactly_128_variables():
    model = _VariableCountModel(128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    step = build_candidate_restricted_graph_train_step(model, optimizer)
    before = int(optimizer.iterations.numpy())
    loss = step({"labels": tf.constant([0, 1], dtype=tf.int64)})
    assert bool(tf.math.is_finite(loss).numpy())
    assert int(optimizer.iterations.numpy()) == before + 1


class _NormalizeBuilderAst(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name in {
            "build_restricted_graph_train_step",
            "build_candidate_restricted_graph_train_step",
        }:
            node.name = "normalized_builder"
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "validate_gradient_contract":
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg != "expected_count"
            ]
        return node


def test_candidate_builder_is_structurally_frozen_except_128_extension():
    frozen_tree = ast.parse(inspect.getsource(build_restricted_graph_train_step))
    candidate_tree = ast.parse(
        inspect.getsource(build_candidate_restricted_graph_train_step)
    )
    normalizer = _NormalizeBuilderAst()
    frozen_tree = ast.fix_missing_locations(normalizer.visit(frozen_tree))
    candidate_tree = ast.fix_missing_locations(normalizer.visit(candidate_tree))
    assert ast.dump(candidate_tree, include_attributes=False) == ast.dump(
        frozen_tree, include_attributes=False
    )
    source = inspect.getsource(build_candidate_restricted_graph_train_step)
    assert "expected_count=EXPECTED_CANDIDATE_TRAINABLE_VARIABLE_COUNT" in source
    assert EXPECTED_CANDIDATE_TRAINABLE_VARIABLE_COUNT == 128


@pytest.fixture(scope="module")
def bounded_selected_step(golden_batch):
    previous_policy = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        config = load_config(SEED42_CONFIG)
        state = validate_execution_config(config["training"])
        candidate = build_candidate_model(golden_batch)
        variable_ids_before = tuple(id(variable) for variable in candidate.trainable_variables)
        gradient_fn = build_compiled_gradient_function(candidate, training=True)
        _, _, gradients, finite = gradient_fn(golden_batch)
        q = candidate.learned_local_residual_slots.Q
        q_index = next(
            index for index, variable in enumerate(candidate.trainable_variables)
            if variable is q
        )
        q_gradient = gradients[q_index]
        q_gradient_norm = float(tf.linalg.global_norm([q_gradient]).numpy())
        optimizer = build_optimizer(config)
        optimizer.build(candidate.trainable_variables)
        optimizer_state_before = tuple(
            (variable.name, tuple(variable.shape), str(variable.dtype))
            for variable in optimizer.variables
        )
        q_before = q.numpy().copy()
        iteration_before = int(optimizer.iterations.numpy())
        step = build_candidate_restricted_graph_train_step(candidate, optimizer)
        loss = step(golden_batch)
        q_after = q.numpy().copy()
        iteration_after = int(optimizer.iterations.numpy())
        optimizer_state_after = tuple(
            (variable.name, tuple(variable.shape), str(variable.dtype))
            for variable in optimizer.variables
        )
        return {
            "state": state,
            "loss_finite": bool(tf.math.is_finite(loss).numpy()),
            "all_gradients_finite": bool(finite.numpy()),
            "q_gradient": q_gradient.numpy(),
            "q_gradient_norm": q_gradient_norm,
            "q_before": q_before,
            "q_after": q_after,
            "q_max_abs_delta": float(np.max(np.abs(q_after - q_before))),
            "iteration_before": iteration_before,
            "iteration_after": iteration_after,
            "model_variable_ids_before": variable_ids_before,
            "model_variable_ids_after": tuple(
                id(variable) for variable in candidate.trainable_variables
            ),
            "optimizer_state_before": optimizer_state_before,
            "optimizer_state_after": optimizer_state_after,
        }
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


def test_selected_g1a_step_has_finite_nonzero_q_gradient_and_one_update(
    bounded_selected_step,
):
    result = bounded_selected_step
    assert result["state"]["optimizer_execution_mode"] == "restricted_tf_function"
    assert result["state"]["gradient_execution_mode"] == "tf_function"
    assert result["state"]["grappler_profile"] == "G1-A"
    assert result["loss_finite"] is True
    assert result["all_gradients_finite"] is True
    assert result["q_gradient"].dtype == np.float32
    assert np.all(np.isfinite(result["q_gradient"]))
    assert np.any(result["q_gradient"] != 0)
    assert result["q_gradient_norm"] > 0
    assert result["q_max_abs_delta"] > 0
    assert result["iteration_after"] == result["iteration_before"] + 1
    assert result["model_variable_ids_after"] == result["model_variable_ids_before"]
    assert result["optimizer_state_after"] == result["optimizer_state_before"]


def test_candidate_checkpoint_roundtrip_preserves_exact_class_parameters_and_q(
    tmp_path, golden_batch
):
    candidate = build_candidate_model(golden_batch)
    q_values = tf.reshape(tf.linspace(-0.04, 0.04, 4 * 96), (4, 96))
    candidate.learned_local_residual_slots.Q.assign(q_values)
    expected_q = candidate.learned_local_residual_slots.Q.numpy().copy()
    checkpoint = tmp_path / "checkpoints" / "best_val_accuracy.keras"
    checkpoint.parent.mkdir(parents=True)
    candidate.save(checkpoint)
    provenance = harness._checkpoint_provenance(tmp_path)
    restored = tf.keras.models.load_model(checkpoint, compile=False)
    assert type(restored) is LearnedLocalResidualSlotLapGNN
    assert restored.count_params() == 1_061_576
    assert tuple(restored.learned_local_residual_slots.Q.shape) == (4, 96)
    assert str(restored.learned_local_residual_slots.Q.dtype) == "float32"
    np.testing.assert_array_equal(
        restored.learned_local_residual_slots.Q.numpy(), expected_q
    )
    assert provenance["checkpoint_class"] == "LearnedLocalResidualSlotLapGNN"
    assert provenance["checkpoint_parameter_count"] == 1_061_576
    assert provenance["checkpoint_q_shape"] == [4, 96]
    assert provenance["checkpoint_q_dtype"] == "float32"


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(tmp_path / "config.yaml"),
        fer_csv=str(tmp_path / "fer.csv"),
        prior_root=str(tmp_path / "priors"),
        output_root=str(tmp_path / "output"),
        no_resume=True,
        limit_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_train_eval_batches=1,
    )


def _fake_wrapper(run_validation_only):
    return SimpleNamespace(
        run_validation_only=run_validation_only,
        _resource_controls=lambda _args: SimpleNamespace(),
    )


def test_production_harness_injects_only_two_bindings_and_restores_success(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    original_constructor = trainer.LapGNN
    original_builder = trainer.build_restricted_graph_train_step
    observed = {}

    monkeypatch.setattr(
        harness,
        "_verify_source_locks",
        lambda **_kwargs: {"locked": "same"},
    )
    monkeypatch.setattr(
        harness,
        "_verify_run_config",
        lambda _path: (Path(args.config).resolve(), "config-sha"),
    )
    monkeypatch.setattr(
        harness,
        "_write_candidate_sidecar",
        lambda **kwargs: kwargs,
    )

    def fake_run(*_positional, **_keyword):
        observed["candidate"] = trainer.LapGNN()
        observed["builder"] = trainer.build_restricted_graph_train_step
        return {"valid": True}

    result = harness._run_candidate_validation_only(_fake_wrapper(fake_run), args)
    assert type(observed["candidate"]) is LearnedLocalResidualSlotLapGNN
    assert observed["builder"] is build_candidate_restricted_graph_train_step
    assert trainer.LapGNN is original_constructor is LapGNN
    assert trainer.build_restricted_graph_train_step is original_builder
    assert result["candidate_constructor_injected"] is True
    assert result["candidate_builder_injected"] is True
    assert result["original_constructor_restored"] is True
    assert result["original_builder_restored"] is True


@pytest.mark.parametrize(
    "error",
    [RuntimeError("synthetic training failure"), ValueError("synthetic wrapper failure")],
)
def test_both_bindings_restore_after_training_or_wrapper_exception(
    tmp_path, monkeypatch, error
):
    args = _args(tmp_path)
    original_constructor = trainer.LapGNN
    original_builder = trainer.build_restricted_graph_train_step
    monkeypatch.setattr(
        harness,
        "_verify_source_locks",
        lambda **_kwargs: {"locked": "same"},
    )
    monkeypatch.setattr(
        harness,
        "_verify_run_config",
        lambda _path: (Path(args.config).resolve(), "config-sha"),
    )

    def fail(*_args, **_kwargs):
        assert type(trainer.LapGNN()) is LearnedLocalResidualSlotLapGNN
        assert trainer.build_restricted_graph_train_step is (
            build_candidate_restricted_graph_train_step
        )
        raise error

    with pytest.raises(type(error), match=str(error)):
        harness._run_candidate_validation_only(_fake_wrapper(fail), args)
    assert trainer.LapGNN is original_constructor
    assert trainer.build_restricted_graph_train_step is original_builder
    assert not Path(args.output_root, harness.CANDIDATE_MARKER_NAME).exists()


def _valid_sidecar_fixture(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("locked: {}\n", encoding="utf-8")
    history = {"epochs": [{"epoch": 1}]}
    resolved = {
        "locked": {
            "package_checksum": harness.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
            "execution_contract_sha256": (
                harness.INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
            ),
            "parameter_count": harness.BASELINE_PARAMETER_COUNT,
        }
    }
    history_path = output / "history.json"
    resolved_path = output / "resolved_config.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    marker = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "test_data_constructed": False,
        "test_checkpoint_loaded": False,
        "normal_full_training_completed": False,
        "boundary": "before_resolve_final_checkpoint",
        "trainer_revision_guard_passed": True,
        "intercepted_function_restored": True,
        "trainer_source_sha256": harness.EXPECTED_TRAINER_SHA256,
        "scientific_payload_sha256": harness.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_path": str(config.resolve()),
        "input_config_sha256": _sha256(config),
        "history_sha256": _sha256(history_path),
        "resolved_config_sha256": _sha256(resolved_path),
        "final_observed_epoch": 1,
    }
    (output / harness.FROZEN_MARKER_NAME).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    kwargs = {
        "output_root": output,
        "returned_marker": marker,
        "input_config_path": config.resolve(),
        "input_config_sha256": _sha256(config),
        "source_hashes_before": {"all": "same"},
        "source_hashes_after": {"all": "same"},
        "candidate_constructor_injected": True,
        "candidate_builder_injected": True,
        "original_constructor_restored": True,
        "original_builder_restored": True,
    }
    return output, kwargs


def _fake_checkpoint_provenance():
    return {
        "checkpoint_path": "checkpoints/best_val_accuracy.keras",
        "checkpoint_sha256": "a" * 64,
        "learned_q_flat_float32_sha256": "b" * 64,
        "checkpoint_class": "LearnedLocalResidualSlotLapGNN",
        "checkpoint_parameter_count": 1_061_576,
        "checkpoint_q_shape": [4, 96],
        "checkpoint_q_dtype": "float32",
    }


def test_candidate_sidecar_is_written_only_after_valid_frozen_marker(
    tmp_path, monkeypatch
):
    output, kwargs = _valid_sidecar_fixture(tmp_path)
    monkeypatch.setattr(
        harness, "_checkpoint_provenance", lambda _output: _fake_checkpoint_provenance()
    )
    sidecar = harness._write_candidate_sidecar(**kwargs)
    assert (output / harness.CANDIDATE_MARKER_NAME).is_file()
    assert sidecar["candidate_execution_contract_sha256"] == (
        harness.EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256
    )
    assert sidecar["inherited_baseline_execution_contract_sha256"] == (
        harness.INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
    )
    assert sidecar["baseline_config_parameter_lock"] == 1_061_192
    assert sidecar["actual_candidate_parameter_count"] == 1_061_576
    assert sidecar["test_access"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "missing_marker",
        "malformed_marker",
        "post_test_contamination",
        "missing_checkpoint",
        "wrong_checkpoint_class",
        "constructor_not_restored",
        "builder_not_restored",
        "source_drift",
    ],
)
def test_candidate_sidecar_fail_closed_variants(tmp_path, monkeypatch, failure):
    output, kwargs = _valid_sidecar_fixture(tmp_path)
    marker_path = output / harness.FROZEN_MARKER_NAME
    if failure == "missing_marker":
        marker_path.unlink()
    elif failure == "malformed_marker":
        marker_path.write_text("{", encoding="utf-8")
    elif failure == "post_test_contamination":
        (output / "TRAINING_COMPLETE.json").write_text("{}", encoding="utf-8")
    elif failure == "constructor_not_restored":
        kwargs["original_constructor_restored"] = False
    elif failure == "builder_not_restored":
        kwargs["original_builder_restored"] = False
    elif failure == "source_drift":
        kwargs["source_hashes_after"] = {"all": "drift"}

    if failure not in {"missing_checkpoint", "wrong_checkpoint_class"}:
        monkeypatch.setattr(
            harness,
            "_checkpoint_provenance",
            lambda _output: _fake_checkpoint_provenance(),
        )
    elif failure == "wrong_checkpoint_class":
        checkpoint = output / harness.CANDIDATE_CHECKPOINT_RELATIVE
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"not-a-real-checkpoint")
        monkeypatch.setattr(
            tf.keras.models,
            "load_model",
            lambda *_args, **_kwargs: LapGNN(),
        )

    with pytest.raises(harness.CandidateValidationOnlyError):
        harness._write_candidate_sidecar(**kwargs)
    assert not (output / harness.CANDIDATE_MARKER_NAME).exists()


@pytest.mark.parametrize("drift", ["parameter_count", "q_shape", "q_dtype"])
def test_checkpoint_identity_drift_fails_closed(tmp_path, monkeypatch, drift):
    checkpoint = tmp_path / harness.CANDIDATE_CHECKPOINT_RELATIVE
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")

    class FakeQ:
        shape = (4, 95) if drift == "q_shape" else (4, 96)
        dtype = "float16" if drift == "q_dtype" else "float32"

        @staticmethod
        def numpy():
            return np.zeros(FakeQ.shape, dtype=np.float32)

    class FakeCandidate:
        learned_local_residual_slots = SimpleNamespace(Q=FakeQ())

        @staticmethod
        def count_params():
            return 1_061_575 if drift == "parameter_count" else 1_061_576

    monkeypatch.setattr(harness, "LearnedLocalResidualSlotLapGNN", FakeCandidate)
    monkeypatch.setattr(
        tf.keras.models, "load_model", lambda *_args, **_kwargs: FakeCandidate()
    )
    with pytest.raises(harness.CandidateValidationOnlyError):
        harness._checkpoint_provenance(tmp_path)


def test_source_hash_drift_fails_before_any_injection(monkeypatch):
    original_constructor = trainer.LapGNN
    original_builder = trainer.build_restricted_graph_train_step
    actual = harness._source_hashes()
    actual["candidate_model"] = "0" * 64
    monkeypatch.setattr(harness, "_source_hashes", lambda: actual)
    with pytest.raises(harness.CandidateValidationOnlyError, match="source SHA drift"):
        harness._verify_source_locks(require_original_bindings=True)
    assert trainer.LapGNN is original_constructor
    assert trainer.build_restricted_graph_train_step is original_builder


def test_provenance_exception_leaves_both_original_bindings_restored(
    tmp_path, monkeypatch
):
    original_constructor = trainer.LapGNN
    original_builder = trainer.build_restricted_graph_train_step
    monkeypatch.setattr(
        harness,
        "_verify_source_locks",
        lambda **_kwargs: (_ for _ in ()).throw(
            harness.CandidateValidationOnlyError("synthetic provenance failure")
        ),
    )
    wrapper_called = False

    def must_not_run(*_args, **_kwargs):
        nonlocal wrapper_called
        wrapper_called = True

    with pytest.raises(
        harness.CandidateValidationOnlyError, match="synthetic provenance failure"
    ):
        harness._run_candidate_validation_only(
            _fake_wrapper(must_not_run), _args(tmp_path)
        )
    assert wrapper_called is False
    assert trainer.LapGNN is original_constructor
    assert trainer.build_restricted_graph_train_step is original_builder


def test_production_source_has_only_authorized_patch_boundary_and_no_test_lifecycle():
    source = HARNESS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "trainer"
                ):
                    trainer_assignments.append(target.attr)
    assert set(trainer_assignments) == {"LapGNN", "build_restricted_graph_train_step"}
    assert "wrapper.run_validation_only(" in source
    for forbidden in (
        "trainer.run_training(",
        "GraphBatchGenerator(",
        "resolve_final_checkpoint =",
        "build_optimizer =",
        "evaluate_batches =",
        "CheckpointPolicy =",
        'split="test"',
        "--model",
        "--architecture",
        "TRAINING_COMPLETE.json\").write",
    ):
        assert forbidden not in source


def test_frozen_wrapper_entry_and_config_keep_inherited_baseline_locks():
    wrapper = harness._load_frozen_wrapper()
    assert callable(wrapper.run_validation_only)
    assert wrapper.trainer is trainer
    _, _ = harness._verify_run_config(SEED42_CONFIG)
    config = load_config(SEED42_CONFIG)
    assert config["locked"]["parameter_count"] == 1_061_192
    assert config["locked"]["execution_contract_sha256"] == (
        harness.INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
    )
    assert config["training"]["optimizer_execution_mode"] == "restricted_tf_function"
    assert config["training"]["gradient_execution_mode"] == "tf_function"
    assert config["training"]["grappler_profile"] == "G1-A"
