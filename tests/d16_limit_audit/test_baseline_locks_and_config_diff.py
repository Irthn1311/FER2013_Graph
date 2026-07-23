from d16.scripts import prepare_ofix7_mid_limit_audit as prep

def test_baseline_locks_and_hashes():
    result=prep.verify_baseline_locks()
    assert result["checkpoint_policy_lock_sha"]==prep.POLICY_SHA
    assert result["baseline_replication_lock_sha"]==prep.BASELINE_SHA
    assert result["baseline_policy_val_macro_f1"]
    assert result["baseline_status_strong_replication"]

def test_exact_ten_registered_configs_and_authorized_diffs_only():
    paths=[prep.config_path(v,s) for v in prep.VARIANTS for s in prep.ALL_SEEDS]
    assert len(paths)==10 and all(path.exists() for path in paths)
    assert len(list(prep.CONFIG_DIR.glob("*.yaml")))==10
    for variant in prep.VARIANTS:
        for seed in prep.ALL_SEEDS:
            base=prep.load_yaml(prep.baseline_run(seed)/"resolved_config.yaml")
            cfg=prep.load_yaml(prep.config_path(variant,seed))
            rows=prep.semantic_diff(base,cfg,variant,seed)
            assert rows and not [r for r in rows if r["authorization_status"]!="AUTHORIZED"]

def test_seed_registration_exact():
    assert prep.DEVELOPMENT_SEEDS==[42,1009,1337]
    assert prep.HELDOUT_SEEDS==[777,3407]
    assert prep.ALL_SEEDS==[42,1009,1337,777,3407]

def test_portable_text_hash_is_line_ending_invariant(tmp_path):
    crlf=tmp_path/"crlf.json"
    lf=tmp_path/"lf.json"
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    assert prep.sha256_file(crlf)!=prep.sha256_file(lf)
    assert prep.normalized_text_sha256(crlf)==prep.normalized_text_sha256(lf)

