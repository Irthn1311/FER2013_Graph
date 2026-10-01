from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mpg_fer_o1.protocol import (
    BASELINE_CONFIG_SHA256,
    BASELINE_ID,
    CHECKPOINT_SELECTION,
    FROZEN_SCIENTIFIC_SOURCE_SHA256,
    HISTORICAL_METRIC_DATASET_ROLE,
    HISTORICAL_PUBLIC_RESULTS_NAME,
    HISTORICAL_PUBLIC_RESULTS_SHA256,
    O1_CONFIG_ORDER,
    O1_REGISTRY,
    SCREEN_STOP_EPOCH,
    aggregate_o1,
    frozen_baseline_config,
    o1_source_tree_hash,
    raw_guardrail,
    registry_document,
    resolve_o1_config,
    right_censor_decision,
    screen_stop_identity,
    scientific_diff,
    validate_baseline_reference,
    validate_historical_public_results_artifact,
    validate_mounted_input_firewall,
    verify_single_delta,
    write_hpo_manifest,
    write_json,
    write_resolved_config,
    write_run_checksums,
)
from mpg_fer_v2_3.checkpoint import config_hash, sha256_file
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_v2_3.train import WarmupCosineScheduler, is_better_checkpoint


def _input() -> torch.Tensor:
    return torch.rand(1, 1, 48, 48, generator=torch.Generator().manual_seed(103))


def _baseline_reference() -> dict:
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "usable_for_o1_aggregation": True,
        "config_id": BASELINE_ID,
        "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
        "seed": 42,
        "learning_rate": 3.0e-4,
        "lr_decay_end_epoch": 85,
        "metric_dataset_role": HISTORICAL_METRIC_DATASET_ROLE,
        "public_results_provenance": {
            "dataset_role": HISTORICAL_METRIC_DATASET_ROLE,
            "artifact_name": HISTORICAL_PUBLIC_RESULTS_NAME,
            "artifact_sha256": HISTORICAL_PUBLIC_RESULTS_SHA256,
            "identity_basis": "exact_reviewed_public_results_name_and_sha256",
        },
        "checkpoint_provenance": {"run_id": "historical"},
        "checkpoint_sha256": "a" * 64,
        "history_provenance": {"run_id": "historical"},
        "history_sha256": "b" * 64,
        "selected_epoch": 57,
        "public_metrics": {
            "raw": {"loss": 1.1, "accuracy": 0.67, "macro_f1": 0.65},
            "tta": {"loss": 1.0, "accuracy": 0.69, "macro_f1": 0.67},
        },
        "private_test_artifacts_read": False,
    }


def test_o1_t1_baseline_model_source_and_config_parity() -> None:
    baseline = MPGConfig()
    o1_control = frozen_baseline_config()
    assert asdict(o1_control) == asdict(baseline)
    assert config_hash(o1_control) == BASELINE_CONFIG_SHA256
    torch.manual_seed(42)
    expected = MPGFER(baseline).eval()
    torch.manual_seed(42)
    actual = MPGFER(o1_control).eval()
    actual.load_state_dict(expected.state_dict(), strict=True)
    with torch.no_grad():
        expected_logits, expected_outputs = expected(_input())
        actual_logits, actual_outputs = actual(_input())
    assert torch.equal(expected_logits, actual_logits)
    for key in ("final_logits", "pixel_logits", "motif_logits", "supcon_embeddings"):
        assert torch.equal(expected_outputs[key], actual_outputs[key])


def test_o1_t2_registry_is_exact_and_has_external_control() -> None:
    expected = [
        (f"O1_{index:02d}", lr, horizon)
        for index, (lr, horizon) in enumerate(
            [
                *((lr, 65) for lr in (1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4)),
                *((lr, 75) for lr in (1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4)),
                *((lr, 85) for lr in (1.5e-4, 2e-4, 2.5e-4, 4e-4)),
            ],
            start=1,
        )
    ]
    assert len(O1_REGISTRY) == 14
    assert [(key, O1_REGISTRY[key].learning_rate, O1_REGISTRY[key].lr_decay_end_epoch) for key in O1_CONFIG_ORDER] == expected
    assert len({(spec.learning_rate, spec.lr_decay_end_epoch) for spec in O1_REGISTRY.values()}) == 14
    document = registry_document()
    assert document["historical_control"]["config_id"] == BASELINE_ID
    assert document["historical_control_is_new_job"] is False


@pytest.mark.parametrize("config_id", O1_CONFIG_ORDER)
def test_o1_t3_only_registered_lr_and_horizon_can_differ(config_id) -> None:
    baseline = frozen_baseline_config()
    resolved = resolve_o1_config(config_id)
    differences = scientific_diff(baseline, resolved)
    assert set(differences) <= {"learning_rate", "lr_decay_end_epoch"}
    assert verify_single_delta(config_id, resolved)["declared_delta_fields"] == [
        "learning_rate",
        "lr_decay_end_epoch",
    ]
    assert asdict(resolved)["motif_topk_schedule"] == (8, 16, 16, 16, 24)
    assert asdict(resolved)["motif_residual_scale_schedule"] == (
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
    )


@pytest.mark.parametrize("horizon", [65, 75, 85])
def test_o1_t4_actual_scheduler_warmup_decay_and_floor(horizon) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = WarmupCosineScheduler(optimizer, 3e-4, 5, horizon, 120, 1e-6)
    assert [scheduler.step(epoch) for epoch in range(1, 6)] == pytest.approx(
        [6e-5, 1.2e-4, 1.8e-4, 2.4e-4, 3e-4]
    )
    representatives = {epoch: scheduler.step(epoch) for epoch in (30, 42, 45, 49, 57, 65)}
    assert all(math.isfinite(value) for value in representatives.values())
    assert representatives[30] > representatives[42] > representatives[57]
    assert scheduler.step(horizon) == pytest.approx(1e-6)
    assert scheduler.step(120) == pytest.approx(1e-6)
    assert scheduler.state_dict()["max_epochs"] == 120


def test_o1_t5_operational_stop_is_outside_scientific_and_resume_identity() -> None:
    config = resolve_o1_config("O1_01")
    before = config_hash(config)
    identity = screen_stop_identity(config)
    assert identity["screen_stop_epoch"] == 65
    assert identity["scientific_max_epochs"] == 120
    assert config.max_epochs == 120
    assert config_hash(config) == before == identity["scientific_config_sha256"]
    assert "screen_stop_epoch" not in asdict(config)
    assert identity["scheduler_identity"]["max_epochs"] == 120


def test_o1_t6_comparator_exact_lexicographic_order() -> None:
    incumbent = {"accuracy": 0.7, "macro_f1": 0.6, "loss": 1.0}
    assert is_better_checkpoint({"accuracy": 0.71, "macro_f1": 0.0, "loss": 9.0}, incumbent)
    assert is_better_checkpoint({"accuracy": 0.7, "macro_f1": 0.61, "loss": 9.0}, incumbent)
    assert is_better_checkpoint({"accuracy": 0.7, "macro_f1": 0.6, "loss": 0.9}, incumbent)
    assert not is_better_checkpoint({"accuracy": 0.7, "macro_f1": 0.6, "loss": 1.1}, incumbent)


def test_o1_t8_firewall_allows_train_public_without_opening(tmp_path) -> None:
    root = tmp_path / "input"
    data = root / "train-public"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("never opened", encoding="utf-8")
    (data / "val.csv").write_text("never opened", encoding="utf-8")
    with patch.object(Path, "open", side_effect=AssertionError("file opened")):
        result = validate_mounted_input_firewall(root)
    assert result["forbidden_private_markers_found"] is False


@pytest.mark.parametrize(
    "forbidden",
    ["test.csv", "PrivateTest/data.bin", "private_test/data.bin", "private-test/data.bin"],
)
def test_o1_t8_firewall_rejects_coexisting_marker_without_open(tmp_path, forbidden) -> None:
    root = tmp_path / "input"
    data = root / "train-public"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("never opened", encoding="utf-8")
    (data / "val.csv").write_text("never opened", encoding="utf-8")
    private = root / forbidden
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("must not open", encoding="utf-8")
    with patch.object(Path, "open", side_effect=AssertionError("private file opened")):
        with pytest.raises(RuntimeError, match="PRIVATE_FIREWALL"):
            validate_mounted_input_firewall(root)


def test_o1_t9_finite_provenance_outputs_and_nan_rejection(tmp_path) -> None:
    config = resolve_o1_config("O1_01", output_dir=str(tmp_path), run_id="run")
    resolved = write_resolved_config(tmp_path / "resolved_config.json", config)
    manifest = write_hpo_manifest(
        tmp_path,
        config_id="O1_01",
        config=config,
        source_sha256="c" * 64,
        notebook_sha256="d" * 64,
    )
    resolved_doc = json.loads(resolved.read_text(encoding="utf-8"))
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_doc["config_id"] == "O1_01"
    assert manifest_doc["PRIVATE_EVALUATED"] is False
    assert manifest_doc["screen_stop_epoch"] == 65
    assert manifest_doc["scientific_max_epochs"] == 120
    assert resolved_doc["scientific_config_sha256"] == config_hash(config)
    with pytest.raises(ValueError, match="NaN and Infinity"):
        write_json(tmp_path / "bad.json", {"value": float("nan")})


def _trajectory(final_better: bool = False):
    rows = []
    for epoch in range(1, 66):
        accuracy = 0.7 if epoch == 50 else 0.5
        if epoch == 64:
            accuracy = 0.55
        if epoch == 65:
            accuracy = 0.56 if final_better else 0.54
        rows.append({"epoch": epoch, "accuracy": accuracy, "macro_f1": 0.5, "loss": 1.0})
    return rows


def test_o1_t10_right_censor_exact_rules() -> None:
    assert right_censor_decision(selected_epoch=50, comparator_trajectory=_trajectory())["right_censored"] is False
    assert right_censor_decision(selected_epoch=61, comparator_trajectory=_trajectory())["status"] == "RIGHT_CENSORED_AT_SCREEN_LIMIT"
    improving = right_censor_decision(selected_epoch=50, comparator_trajectory=_trajectory(True))
    assert improving["right_censored"] is True
    assert improving["comparator_still_improving_at_epoch_65"] is True


def test_raw_guardrail_warns_below_minus_half_percentage_point() -> None:
    baseline = {"accuracy": 0.7, "macro_f1": 0.7}
    assert raw_guardrail({"accuracy": 0.6949, "macro_f1": 0.7}, baseline)["warning"] is True
    assert raw_guardrail({"accuracy": 0.695, "macro_f1": 0.695}, baseline)["warning"] is False


def test_baseline_reference_fails_closed_unless_complete_and_verified() -> None:
    assert validate_baseline_reference(_baseline_reference())["selected_epoch"] == 57
    invalid = _baseline_reference()
    invalid["status"] = "UNVERIFIED_FAIL_CLOSED"
    with pytest.raises(RuntimeError, match="BASELINE_REFERENCE_REFUSED"):
        validate_baseline_reference(invalid)


def test_baseline_public_identity_refuses_same_run_nonreviewed_metric_artifact(
    tmp_path,
) -> None:
    candidate = tmp_path / HISTORICAL_PUBLIC_RESULTS_NAME
    write_json(
        candidate,
        {
            "identity": {
                "run_id": "same-reviewed-run",
                "source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
                "config_sha256": BASELINE_CONFIG_SHA256,
                "best_checkpoint_sha256": "a" * 64,
            },
            "dataset_role": "PrivateTest",
        },
    )
    with pytest.raises(RuntimeError, match="exact reviewed PublicTest artifact"):
        validate_historical_public_results_artifact(candidate)


def _authorized_aggregation_design(tmp_path, baseline_path):
    registry_path = tmp_path / "O1_REGISTRY.json"
    write_json(registry_path, registry_document())
    notebook_path = tmp_path / "notebooks" / "MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
    notebook_path.parent.mkdir()
    notebook_path.write_text("synthetic reviewed notebook\n", encoding="utf-8")
    design_path = tmp_path / "O1_HPO_DESIGN_LOCK.json"
    design = {
        "wave1_execution_authorized": True,
        "private_test_permitted": False,
        "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
        "o1_source_tree_sha256": o1_source_tree_hash(),
        "new_config_ids": list(O1_CONFIG_ORDER),
        "historical_control_id": BASELINE_ID,
        "screen_stop_epoch": 65,
        "scientific_max_epochs": 120,
        "registry_path": registry_path.name,
        "registry_sha256": sha256_file(registry_path),
        "notebook_path": notebook_path.relative_to(tmp_path).as_posix(),
        "notebook_sha256": sha256_file(notebook_path),
        "baseline_reference_sha256": sha256_file(baseline_path),
    }
    write_json(design_path, design)
    return design_path, design


def _completed_synthetic_run(
    tmp_path, config_id, index, design, *, exact_resumed=False
):
    spec = O1_REGISTRY[config_id]
    root = tmp_path / config_id
    root.mkdir()
    config = resolve_o1_config(config_id, run_id=f"run-{config_id}")
    write_resolved_config(root / "resolved_config.json", config)
    write_hpo_manifest(
        root,
        config_id=config_id,
        config=config,
        source_sha256=design["o1_source_tree_sha256"],
        notebook_sha256=design["notebook_sha256"],
    )
    (root / "best_val_acc.pt").write_bytes(f"checkpoint-{config_id}".encode())
    checkpoint_sha = sha256_file(root / "best_val_acc.pt")
    score = 0.60 + index / 1000
    write_json(
        root / "selected_public_metrics.json",
        {
            "config_id": config_id,
            "selected_epoch": 50,
            "public": {
                "raw": {"loss": 1.1, "accuracy": score, "macro_f1": score},
                "tta": {"loss": 1.0, "accuracy": score, "macro_f1": score},
            },
            "right_censor": {
                "right_censored": config_id == "O1_02",
                "status": "synthetic",
            },
            "checkpoint_sha256": checkpoint_sha,
            "PRIVATE_EVALUATED": False,
        },
    )
    write_json(root / "history.json", {"synthetic": True})
    (root / "history.csv").write_text("epoch\n65\n", encoding="utf-8")
    write_json(
        root / "execution_manifest.json",
        {
            "config_id": config_id,
            "source_sha256": design["o1_source_tree_sha256"],
            "scientific_config_sha256": config_hash(config),
            "status": "SCREENING_COMPLETED",
            "screen_stop_epoch": 65,
            "scientific_max_epochs": 120,
            "best_checkpoint_sha256": checkpoint_sha,
            "resumed_from": (
                f"/reviewed/resume/{config_id}/resume_latest.pt"
                if exact_resumed
                else None
            ),
            "PRIVATE_EVALUATED": False,
        },
    )
    write_json(root / "segment_manifest.json", {"status": "SCREENING_COMPLETED"})
    (root / "resume_latest.pt").write_bytes(b"synthetic exact-resume checkpoint")
    write_run_checksums(root)
    return root


def test_o1_t11_aggregator_fixed_order_surface_and_promotion_separation(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    write_json(baseline_path, _baseline_reference())
    design_path, design = _authorized_aggregation_design(tmp_path, baseline_path)
    runs = [
        _completed_synthetic_run(
            tmp_path,
            config_id,
            index,
            design,
            exact_resumed=config_id == "O1_02",
        )
        for index, config_id in enumerate(reversed(O1_CONFIG_ORDER))
    ]
    output = tmp_path / "aggregate"
    result = aggregate_o1(runs, baseline_path, design_path, output)
    assert result["canonical_order"] == [*O1_CONFIG_ORDER, BASELINE_ID]
    assert result["right_censored_not_safely_eliminable"] == ["O1_02"]
    assert result["completed_exact_resume_config_ids"] == ["O1_02"]
    registry_lines = (output / "o1_registry.csv").read_text(encoding="utf-8").splitlines()
    assert registry_lines[1].startswith("O1_01,")
    surface = (output / "o1_response_surface.csv").read_text(encoding="utf-8")
    assert len(surface.splitlines()) == 16
    assert "0.0003,85,O1_C0_BASELINE,external_historical_control" in surface
    promotion = (output / "o1_promotion_report.md").read_text(encoding="utf-8")
    assert "Provisional top 3 novel configs" in promotion
    assert "Completed exact-resume jobs accepted: O1_02" in promotion
    assert "Incomplete/partial run artifacts: refused" in promotion
    assert (output / "o1_checksums.sha256").is_file()

    tampered_baseline = tmp_path / "tampered-baseline.json"
    tampered_reference = _baseline_reference()
    tampered_reference["selected_epoch"] = 56
    write_json(tampered_baseline, tampered_reference)
    with pytest.raises(RuntimeError, match="reviewed artifact hash mismatch"):
        aggregate_o1(
            runs,
            tampered_baseline,
            design_path,
            tmp_path / "baseline-refused",
        )

    corrupted = runs[0]
    manifest_path = corrupted / "hpo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"] = "e" * 64
    write_json(manifest_path, manifest)
    write_run_checksums(corrupted)
    with pytest.raises(RuntimeError, match="reviewed design identity"):
        aggregate_o1(runs, baseline_path, design_path, tmp_path / "source-refused")

    manifest["source_sha256"] = design["o1_source_tree_sha256"]
    resolved_path = corrupted / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["scientific_config"]["learning_rate"] = 9.0e-4
    encoded = json.dumps(
        resolved["scientific_config"], sort_keys=True, separators=(",", ":")
    )
    drifted_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    resolved["scientific_config_sha256"] = drifted_hash
    manifest["scientific_config_sha256"] = drifted_hash
    execution_path = corrupted / "execution_manifest.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["scientific_config_sha256"] = drifted_hash
    write_json(resolved_path, resolved)
    write_json(manifest_path, manifest)
    write_json(execution_path, execution)
    write_run_checksums(corrupted)
    with pytest.raises(RuntimeError, match="resolved config identity mismatch"):
        aggregate_o1(runs, baseline_path, design_path, tmp_path / "config-refused")


def test_o1_t11_aggregator_refuses_partial_run(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    write_json(baseline_path, _baseline_reference())
    design_path, _ = _authorized_aggregation_design(tmp_path, baseline_path)
    partial = tmp_path / "O1_01"
    partial.mkdir()
    with pytest.raises(RuntimeError, match="refuses partial run"):
        aggregate_o1([partial], baseline_path, design_path, tmp_path / "aggregate")
