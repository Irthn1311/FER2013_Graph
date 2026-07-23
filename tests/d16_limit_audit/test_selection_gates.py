from pathlib import Path
import pytest
from d16.scripts.analyze_ofix7_mid_limit_audit import choose_development_winner,evaluate_development_gate,evaluate_heldout_gate,verify_lock
from d16.scripts import prepare_ofix7_mid_limit_audit as prep

def aggregate(acc=1.1,macro=.6,gap=1.0,min_seed=-.5,positive=2,worst_class=-2.0,sd=.4):
    return {"mean_validation_accuracy_gain_pp":acc,"mean_validation_macro_f1_gain_pp":macro,
            "mean_gap_increase_pp":gap,"min_per_seed_validation_accuracy_gain_pp":min_seed,
            "positive_validation_accuracy_seed_count":positive,"worst_mean_per_class_f1_gain_pp":worst_class,
            "validation_accuracy_sample_sd_pp":sd,"mean_train_validation_macro_f1_gap_pp":15.0}

def test_development_gate_and_tie_break_are_validation_only():
    a=aggregate();g=evaluate_development_gate(a,prep.DEVELOPMENT_GATE)
    assert all(g.values())
    b=aggregate(acc=1.0)
    assert choose_development_winner({"S1":a,"O1":b},{"S1":True,"O1":True})=="S1"
    assert "test" not in " ".join(a.keys()).lower()

def test_no_development_variant_passes():
    assert choose_development_winner({"S1":aggregate(),"O1":aggregate()},{"S1":False,"O1":False}) is None

def test_heldout_confirmation_gate():
    a=aggregate(acc=.8,macro=.3,gap=1.0,sd=.7)
    held=[{"validation_accuracy_gain_pp":.2},{"validation_accuracy_gain_pp":.1}]
    assert all(evaluate_heldout_gate(a,held,.4,prep.HELDOUT_GATE).values())

def test_test_reveal_requires_final_lock():
    missing=Path("outputs/d16_analysis/ofix7_mid_limit_audit_test_scratch/missing_final/final_variant_promotion_lock.json")
    with pytest.raises(RuntimeError): verify_lock(missing)

