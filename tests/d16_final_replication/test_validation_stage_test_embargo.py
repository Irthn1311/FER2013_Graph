from pathlib import Path
import pytest
from d16.scripts.analyze_ofix7_mid_5seed import assert_validation_artifact


def test_validation_stage_test_embargo():
    with pytest.raises(RuntimeError): assert_validation_artifact(Path("test_predictions.csv"))
    assert_validation_artifact(Path("best_val_accuracy_predictions.csv"))
