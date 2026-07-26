from _adamw_closure_evidence import OUTPUT_DIR


def test_adamw_closure_evidence_is_packaged():
    assert "outputs" not in OUTPUT_DIR.parts
    assert OUTPUT_DIR.parent.name == "validation_assets"
    for name in [
        "07_tensorflow_primitive_candidates.csv",
        "checkpoint_continuation.json",
        "fresh_live_adamw_comparison.json",
        "production_optimizer_offline_closure.json",
        "repeated_determinism.json",
    ]:
        assert (OUTPUT_DIR / name).is_file()
