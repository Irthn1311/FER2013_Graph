from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PACKAGE_ROOT / "tools" / "train_validation_only.py"
SPEC = importlib.util.spec_from_file_location("train_validation_only", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_name": "synthetic_validation_only",
                "seed": 42,
                "locked": {
                    "package_checksum": wrapper.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_boundary_artifacts(
    output_root: Path, *, missing: str | None = None, forbidden: str | None = None
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "history.json": {
            "epochs": [
                {
                    "epoch": 1,
                    "val_loss": 1.2,
                    "val_accuracy": 0.4,
                    "val_macro_f1": 0.35,
                },
                {
                    "epoch": 2,
                    "val_loss": 1.0,
                    "val_accuracy": 0.5,
                    "val_macro_f1": 0.45,
                },
            ]
        },
        "resolved_config.json": {
            "run_name": "synthetic_validation_only",
            "seed": 42,
            "locked": {
                "package_checksum": wrapper.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
            },
        },
        "telemetry.json": {"elapsed_sec": 1.0},
    }
    for name, payload in payloads.items():
        if name != missing:
            (output_root / name).write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
    if forbidden is not None:
        (output_root / forbidden).write_text("{}", encoding="utf-8")


def _run(tmp_path: Path, monkeypatch, fake_run, *, output_name="output"):
    config = _input_config(tmp_path)
    output = tmp_path / output_name
    monkeypatch.setattr(wrapper.trainer, "run_training", fake_run)
    marker = wrapper.run_validation_only(
        config,
        tmp_path / "fer2013.csv",
        tmp_path / "priors",
        output,
        SimpleNamespace(),
        no_resume=True,
        limit_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_train_eval_batches=1,
    )
    return config, output, marker


def test_wrapper_refuses_unexpected_trainer_revision(monkeypatch):
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINER_SHA256", "0" * 64)

    with pytest.raises(
        wrapper.ValidationOnlyExecutionError,
        match="Re-review trainer.py",
    ):
        wrapper.verify_trainer_revision()


def test_success_stops_before_test_load_and_generator_and_restores_resolver(
    tmp_path, monkeypatch
):
    original_resolver = wrapper.trainer.resolve_final_checkpoint
    calls = {"load_model": 0, "test_generator": 0}

    def forbidden_load_model(*_args, **_kwargs):
        calls["load_model"] += 1
        raise AssertionError("final-test model load must not execute")

    def forbidden_graph_generator(*_args, **_kwargs):
        calls["test_generator"] += 1
        raise AssertionError("test GraphBatchGenerator must not execute")

    monkeypatch.setattr(wrapper.trainer.tf.keras.models, "load_model", forbidden_load_model)
    monkeypatch.setattr(wrapper.trainer, "GraphBatchGenerator", forbidden_graph_generator)

    def fake_run(_config, _fer, prior_root, output_root, _controls, **kwargs):
        assert kwargs == {
            "no_resume": True,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "limit_train_eval_batches": 1,
            "limit_epochs": 2,
        }
        _write_boundary_artifacts(Path(output_root))
        wrapper.trainer.resolve_final_checkpoint({}, object())
        wrapper.trainer.tf.keras.models.load_model("forbidden.keras")
        wrapper.trainer.GraphBatchGenerator(prior_root, "test")

    config, output, marker = _run(tmp_path, monkeypatch, fake_run)

    assert calls == {"load_model": 0, "test_generator": 0}
    assert wrapper.trainer.resolve_final_checkpoint is original_resolver
    assert marker["boundary"] == "before_resolve_final_checkpoint"
    assert marker["training_validation_completed"] is True
    assert marker["final_test_skipped"] is True
    assert marker["test_accessed"] is False
    assert marker["test_data_constructed"] is False
    assert marker["test_checkpoint_loaded"] is False
    assert marker["normal_full_training_completed"] is False
    assert marker["final_observed_epoch"] == 2
    assert marker["scientific_payload_sha256"] == (
        wrapper.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    )
    assert marker["input_config_sha256"] == _sha256(config)
    assert marker["history_sha256"] == _sha256(output / "history.json")
    assert marker["resolved_config_sha256"] == _sha256(
        output / "resolved_config.json"
    )
    assert marker["bounded_limits"] == {
        "limit_epochs": 2,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "limit_train_eval_batches": 1,
    }
    saved = json.loads(
        (output / wrapper.MARKER_NAME).read_text(encoding="utf-8")
    )
    assert saved == marker
    assert not (output / "TRAINING_COMPLETE.json").exists()


def test_normal_training_exception_propagates_and_resolver_is_restored(
    tmp_path, monkeypatch
):
    original_resolver = wrapper.trainer.resolve_final_checkpoint

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("synthetic training failure")

    config = _input_config(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper.trainer, "run_training", fail_training)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors", output,
            SimpleNamespace(), no_resume=True,
        )

    assert wrapper.trainer.resolve_final_checkpoint is original_resolver
    assert not (output / wrapper.MARKER_NAME).exists()


def test_return_without_expected_boundary_never_writes_marker(tmp_path, monkeypatch):
    original_resolver = wrapper.trainer.resolve_final_checkpoint

    def returns_normally(_config, _fer, _priors, output_root, _controls, **_kwargs):
        _write_boundary_artifacts(Path(output_root))
        return {"unexpected": "normal return"}

    config = _input_config(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper.trainer, "run_training", returns_normally)

    with pytest.raises(
        wrapper.ValidationOnlyExecutionError,
        match="returned without reaching",
    ):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors", output,
            SimpleNamespace(), no_resume=True,
        )

    assert wrapper.trainer.resolve_final_checkpoint is original_resolver
    assert not (output / wrapper.MARKER_NAME).exists()


@pytest.mark.parametrize(
    "missing", ["history.json", "resolved_config.json", "telemetry.json"]
)
def test_required_boundary_artifacts_must_exist_before_marker(
    tmp_path, monkeypatch, missing
):
    original_resolver = wrapper.trainer.resolve_final_checkpoint

    def incomplete_run(_config, _fer, _priors, output_root, _controls, **_kwargs):
        _write_boundary_artifacts(Path(output_root), missing=missing)
        wrapper.trainer.resolve_final_checkpoint({}, object())

    config = _input_config(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper.trainer, "run_training", incomplete_run)

    with pytest.raises(
        wrapper.ValidationOnlyExecutionError,
        match="without required artifacts",
    ):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors", output,
            SimpleNamespace(), no_resume=True,
        )

    assert wrapper.trainer.resolve_final_checkpoint is original_resolver
    assert not (output / wrapper.MARKER_NAME).exists()


@pytest.mark.parametrize("forbidden", ["TRAINING_COMPLETE.json", "test_metrics_x.json"])
def test_post_test_artifacts_fail_closed_without_marker(
    tmp_path, monkeypatch, forbidden
):
    def contaminated_run(_config, _fer, _priors, output_root, _controls, **_kwargs):
        _write_boundary_artifacts(Path(output_root), forbidden=forbidden)
        wrapper.trainer.resolve_final_checkpoint({}, object())

    config = _input_config(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper.trainer, "run_training", contaminated_run)

    with pytest.raises(
        wrapper.ValidationOnlyExecutionError,
        match="Post-test artifacts exist",
    ):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors", output,
            SimpleNamespace(), no_resume=True,
        )

    assert not (output / wrapper.MARKER_NAME).exists()


def test_nonempty_output_and_external_inputs_remain_unchanged(tmp_path, monkeypatch):
    config = _input_config(tmp_path)
    fer_csv = tmp_path / "fer2013.csv"
    fer_csv.write_text("Usage,pixels,emotion\n", encoding="utf-8")
    prior_root = tmp_path / "priors"
    prior_root.mkdir()
    prior_file = prior_root / "prior.npz"
    prior_file.write_bytes(b"prior")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (config, fer_csv, prior_file, sentinel)
    }
    called = False

    def must_not_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(wrapper.trainer, "run_training", must_not_run)
    with pytest.raises(FileExistsError, match="absent or empty"):
        wrapper.run_validation_only(
            config, fer_csv, prior_root, output, SimpleNamespace(), no_resume=True
        )

    assert called is False
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (config, fer_csv, prior_file, sentinel)
    } == before


def test_input_and_resolved_payload_drift_fail_closed(tmp_path, monkeypatch):
    config = _input_config(tmp_path)
    config.write_text(
        yaml.safe_dump({"locked": {"package_checksum": "wrong"}}),
        encoding="utf-8",
    )
    called = False

    def must_not_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(wrapper.trainer, "run_training", must_not_run)
    with pytest.raises(
        wrapper.ValidationOnlyExecutionError, match="Config scientific payload"
    ):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors",
            tmp_path / "output", SimpleNamespace(), no_resume=True,
        )
    assert called is False

    valid_config = _input_config(tmp_path)

    def drifted_run(_config, _fer, _priors, output_root, _controls, **_kwargs):
        _write_boundary_artifacts(Path(output_root))
        resolved = Path(output_root) / "resolved_config.json"
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        payload["locked"]["package_checksum"] = "wrong"
        resolved.write_text(json.dumps(payload), encoding="utf-8")
        wrapper.trainer.resolve_final_checkpoint({}, object())

    monkeypatch.setattr(wrapper.trainer, "run_training", drifted_run)
    output = tmp_path / "drifted-output"
    with pytest.raises(
        wrapper.ValidationOnlyExecutionError,
        match="changed during execution",
    ):
        wrapper.run_validation_only(
            valid_config, tmp_path / "fer.csv", tmp_path / "priors",
            output, SimpleNamespace(), no_resume=True,
        )
    assert not (output / wrapper.MARKER_NAME).exists()


def test_private_nonmatching_sentinel_is_not_misclassified(tmp_path, monkeypatch):
    original_resolver = wrapper.trainer.resolve_final_checkpoint

    def wrong_sentinel(*_args, **_kwargs):
        raise wrapper._ValidationOnlyBoundaryReached()

    config = _input_config(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper.trainer, "run_training", wrong_sentinel)

    with pytest.raises(wrapper._ValidationOnlyBoundaryReached):
        wrapper.run_validation_only(
            config, tmp_path / "fer.csv", tmp_path / "priors", output,
            SimpleNamespace(), no_resume=True,
        )

    assert wrapper.trainer.resolve_final_checkpoint is original_resolver
    assert not (output / wrapper.MARKER_NAME).exists()


def test_normal_cli_and_scientific_payload_are_unchanged():
    train_cli = PACKAGE_ROOT / "src" / "lap_gnn_tf" / "cli" / "train.py"
    assert _sha256(train_cli) == (
        "9373e43d17ba2ef838a35ab1a10744a5c8fc71e4575365ef5aca4296fba61355"
    )
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'lap-gnn-tf-train = "lap_gnn_tf.cli.train:main"' in pyproject

    finalizer_path = PACKAGE_ROOT / "tools" / "finalize_package.py"
    spec = importlib.util.spec_from_file_location(
        "validation_only_finalizer", finalizer_path
    )
    assert spec is not None and spec.loader is not None
    finalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finalizer)
    assert finalizer.scientific_checksum() == (
        wrapper.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    )


def test_cli_has_bounded_train_val_limits_but_no_test_limit():
    destinations = {action.dest for action in wrapper.build_parser()._actions}
    assert {
        "config",
        "fer_csv",
        "prior_root",
        "output_root",
        "limit_epochs",
        "limit_train_batches",
        "limit_val_batches",
        "limit_train_eval_batches",
    } <= destinations
    assert "limit_test_batches" not in destinations

    source = TOOL_PATH.read_text(encoding="utf-8")
    assert "GraphBatchGenerator(" not in source
    assert "for epoch" not in source
