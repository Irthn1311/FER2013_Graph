from pathlib import Path

from d16.scripts.prepare_ofix7_mid_final_replication import (
    CONFIG_DIR,
    PORTABLE_LOCK_HASH_PATH,
    PORTABLE_LOCK_PATH,
    PORTABLE_REGISTRATION_HASH_PATH,
    PORTABLE_REGISTRATION_PATH,
    REGISTERED_SEEDS,
    normalized_text_sha256,
    validate_replication_config,
)


def test_replication_config_parity():
    for seed in REGISTERED_SEEDS:
        result = validate_replication_config(CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml")
        assert result["valid"]
        assert not result["unauthorized"]

def test_portable_hashes_are_newline_stable():
    for artifact, sidecar in (
        (PORTABLE_LOCK_PATH, PORTABLE_LOCK_HASH_PATH),
        (PORTABLE_REGISTRATION_PATH, PORTABLE_REGISTRATION_HASH_PATH),
    ):
        expected = sidecar.read_text(encoding="utf-8").strip()
        assert normalized_text_sha256(artifact) == expected
        linux_text = artifact.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert normalized_text_sha256_from_value(linux_text) == expected


def normalized_text_sha256_from_value(value: str) -> str:
    import hashlib
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_all_five_configs_pass_portable_validation(monkeypatch):
    import d16.scripts.run_ofix7_mid_replication as runner

    monkeypatch.setattr(runner, "REGISTRATION_PATH", Path("missing-registration.json"))
    monkeypatch.setattr(runner, "REGISTRATION_HASH_PATH", Path("missing-registration.sha256"))
    monkeypatch.setattr(runner, "LOCK_PATH", Path("missing-lock.json"))
    for seed in REGISTERED_SEEDS:
        cfg, registration, registration_path = runner.validate_registered_config(
            (CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml").resolve()
        )
        assert int(cfg["seed"]) == seed
        assert registration_path == PORTABLE_REGISTRATION_PATH
        assert registration["locked_candidate_id"] == "d16r_a5b_ofix7_prior_drop_mid_seed42"
