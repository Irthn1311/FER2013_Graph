from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_m0_entries_lock_science_and_forbid_private_test():
    names = ("m0-substrate-entry.py", "m0-fit-entry.py", "m0-aggregate-entry.py")
    for name in names:
        source = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert 'SCIENTIFIC_SHA = "8910e9757210674a1504837c5b8bc66e2da9208d"' in source
        assert "PrivateTest" not in source and "test.csv" not in source
    fit = (ROOT / "notebooks/m0-fit-entry.py").read_text(encoding="utf-8")
    assert 'CANONICAL_FIT_WRAPPER_SHA = "0b0306327ec3b7066f99334b9dcd344d95f9b49b"' in fit
    assert 'result["public_labels"]' not in fit
    assert "fit_l(" in fit and "train_seed(" in fit
    substrate = (ROOT / "notebooks/m0-substrate-entry.py").read_text(encoding="utf-8")
    assert "build_substrate(" in substrate and "m0_scientific_sha=SCIENTIFIC_SHA" in substrate and '"public_labels_read": False' in substrate


def test_aggregate_is_only_entry_that_reads_public_labels():
    aggregate = (ROOT / "notebooks/m0-aggregate-entry.py").read_text(encoding="utf-8")
    assert 'result["public_labels"]' in aggregate
    assert "independent_recompute(" in aggregate
    assert 'CANONICAL_FIT_WRAPPER_SHA = "0b0306327ec3b7066f99334b9dcd344d95f9b49b"' in aggregate
