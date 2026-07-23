from __future__ import annotations
import pytest
from d16.scripts import prepare_ofix7_mid_limit_audit as prep

@pytest.fixture(scope="session", autouse=True)
def prepared_limit_audit():
    paths=[prep.config_path(v,s) for v in prep.VARIANTS for s in prep.ALL_SEEDS]
    if not prep.REGISTRATION_PATH.exists() or not all(path.exists() for path in paths):
        return prep.prepare()
    return {"status":"ALREADY_PREPARED"}

