from pathlib import Path
import pytest
from d16.scripts.validate_ofix7_mid_limit_audit import dataset_embargo_unit, tiny_optimizer_trajectories
from d16.scripts.run_ofix7_mid_limit_variant import test_artifacts as scan_test_artifacts, verify_selection_lock, verify_registration
from d16.scripts import prepare_ofix7_mid_limit_audit as prep

SCRATCH=Path("outputs/d16_analysis/ofix7_mid_limit_audit_test_scratch")

def test_test_embargo_skips_test_dataset():
    result=dataset_embargo_unit()
    assert result["passed"] and result["calls"]==["train","val"]

def test_deferral_and_provenance_are_rng_neutral_for_two_steps():
    result=tiny_optimizer_trajectories()
    assert result["steps_per_trajectory"]==2 and result["all_rng_neutral"]
    assert result["radam_initial_match"] and result["radam_first_logits_match"]
    assert result["radam_first_gradients_match"] and result["radam_final_expected_divergence"]

def test_test_artifact_scanner():
    path=SCRATCH/"artifacts";path.mkdir(parents=True,exist_ok=True)
    (path/"test_metrics.csv").write_text("x\n",encoding="utf-8")
    assert "test_metrics.csv" in scan_test_artifacts(path)

def test_heldout_requires_selection_lock():
    registration,sha=verify_registration()
    missing=SCRATCH/"missing/development_variant_selection_lock.json"
    with pytest.raises(RuntimeError): verify_selection_lock(missing,sha,"S1",777,registration)

def test_heldout_rejects_wrong_variant():
    registration,sha=verify_registration()
    path=SCRATCH/"wrong/development_variant_selection_lock.json";path.parent.mkdir(parents=True,exist_ok=True)
    payload={"registration_sha256":sha,"decision":"SELECT_S1_FOR_HELDOUT","selected_variant":"S1",
             "heldout_config_sha256":{"777":registration["config_file_sha256"][prep.relative(prep.config_path("S1",777))]}}
    prep.write_json(path,payload);path.with_suffix(".sha256").write_text(prep.sha256_file(path)+"\n",encoding="utf-8")
    with pytest.raises(RuntimeError): verify_selection_lock(path,sha,"O1",777,registration)

