"""Package provenance records."""

from lap_gnn.constants import (
    BASELINE_LOCK_SHA256,
    CHECKPOINT_POLICY_LOCK_SHA256,
    PACKAGE_NAME,
    PACKAGE_STATUS,
    PARENT_COMMIT,
    SIGNATURES,
)


def package_provenance() -> dict:
    return {
        "package_name": PACKAGE_NAME,
        "status": PACKAGE_STATUS,
        "parent_commit": PARENT_COMMIT,
        "baseline_lock_sha256": BASELINE_LOCK_SHA256,
        "checkpoint_policy_lock_sha256": CHECKPOINT_POLICY_LOCK_SHA256,
        "signatures": dict(SIGNATURES),
    }
