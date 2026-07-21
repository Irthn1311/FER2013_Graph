from __future__ import annotations

import pytest

from d16.scripts.prepare_ofix7_mid_final_replication import CONFIG_DIR, REGISTERED_SEEDS, REGISTRATION_PATH, prepare


@pytest.fixture(scope="session", autouse=True)
def prepared_replication():
    configs = [CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml" for seed in REGISTERED_SEEDS]
    if not REGISTRATION_PATH.exists() or not all(path.exists() for path in configs):
        return prepare()
    return {"status": "ALREADY_PREPARED"}
