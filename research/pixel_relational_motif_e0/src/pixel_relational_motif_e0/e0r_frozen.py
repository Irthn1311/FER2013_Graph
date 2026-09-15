from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diag_gmm import DiagonalGaussianMixture
from .e01_occurrence_runner import FrozenDescriptorTransform
from .e02_runner import (
    E01_V533_DICTIONARY_SHA256,
    K,
    TRAIN_SHA256,
    load_actual_dictionary,
    sha256_file,
)


CONTROL_ARTIFACT_SHA256 = {
    42: "ace39fc9f8bac39523080002acab547c4c65cf6015fda8b09101003238d5c1e6",
    43: "4795b0380ea25134878f80202cbc9ab737d1096091728f25af87819590bb4ea9",
    44: "c2d881b9806ee927c3a446a3f79fc7ea15faf1cd2442f39a16283a948af94a07",
    45: "2b03c9da78a970cdf52f86b8211761e1eb7d2436a780f19a223038de70c7bef9",
    46: "0ddea5e79dbe512e3af91f5d2778082832b6a12fe7f63192ce9ecc7a15e8d415",
}
MATCHED_POOL_SHA256 = "4745930e66f1581734dd50e329734a630ff347aca4da9cf778c97fb982e32d24"
E02_SUMMARY_SHA256 = "edd7c0b01f3696b941f34043a3758d2e5e9af9da595b994bfd257b8a61789a1e"
E02_RESULTS_SHA256 = "cd41e6793ef3451ee1d11b1c32e0178026c45532e6386fe2800164d34e6b4cad"


@dataclass(frozen=True)
class FrozenCondition:
    name: str
    control_seed: int | None
    transform: FrozenDescriptorTransform
    model: DiagonalGaussianMixture
    artifact_sha256: str


def load_frozen_control(path: str | Path, *, expected_seed: int) -> FrozenCondition:
    """Load a v535 control by assigning saved state; no estimator is refit."""
    if expected_seed not in CONTROL_ARTIFACT_SHA256:
        raise ValueError("unregistered E0.R control seed")
    p = Path(path)
    digest = sha256_file(p)
    expected = CONTROL_ARTIFACT_SHA256[expected_seed]
    if digest != expected:
        raise ValueError(f"control seed {expected_seed} SHA256 mismatch: {digest}")
    with np.load(p, allow_pickle=False) as z:
        seed = int(np.asarray(z["control_seed"]).reshape(()))
        selected_k = int(np.asarray(z["selected_k"]).reshape(()))
        train_sha = str(np.asarray(z["train_sha256"]).reshape(()).item())
        dictionary_sha = str(np.asarray(z["source_dictionary_sha256"]).reshape(()).item())
        pool_sha = str(np.asarray(z["pool_identity_sha256"]).reshape(()).item())
        floor = np.asarray(z["variance_floor"], dtype=np.float64)
        weights = np.asarray(z["gmm_weights"], dtype=np.float64)
        means = np.asarray(z["gmm_means"], dtype=np.float64)
        variances = np.asarray(z["gmm_variances"], dtype=np.float64)
        transform = FrozenDescriptorTransform(
            pca_mean=np.asarray(z["pca_mean"], dtype=np.float64),
            pca_components=np.asarray(z["pca_components"], dtype=np.float64),
            log_sigma_mean=float(np.asarray(z["log_sigma_mean"]).reshape(())),
            log_sigma_std=float(np.asarray(z["log_sigma_std"]).reshape(())),
        )
    if seed != expected_seed or selected_k != K:
        raise ValueError("control seed-to-artifact or K=128 contract mismatch")
    if train_sha != TRAIN_SHA256 or dictionary_sha != E01_V533_DICTIONARY_SHA256:
        raise ValueError("control Train/dictionary provenance mismatch")
    if pool_sha != MATCHED_POOL_SHA256:
        raise ValueError("control matched-pool provenance mismatch")
    if weights.shape != (K,) or means.shape != (K, 12) or variances.shape != (K, 12):
        raise ValueError("control GMM state shape mismatch")
    model = DiagonalGaussianMixture(K, floor)
    model.weights_ = weights
    model.means_ = means
    model.variances_ = variances
    return FrozenCondition(f"control_{seed}", seed, transform, model, digest)


def load_frozen_conditions(
    actual_dictionary: str | Path,
    control_paths: dict[int, str | Path],
) -> list[FrozenCondition]:
    if set(control_paths) != set(CONTROL_ARTIFACT_SHA256):
        raise ValueError("exact control seed mapping 42..46 is required")
    actual = load_actual_dictionary(actual_dictionary)
    conditions = [
        FrozenCondition(
            "actual",
            None,
            actual.artifact.transform,
            actual.artifact.model,
            actual.sha256,
        )
    ]
    conditions.extend(
        load_frozen_control(control_paths[seed], expected_seed=seed)
        for seed in sorted(CONTROL_ARTIFACT_SHA256)
    )
    if any(condition.model.n_components != K for condition in conditions):
        raise ValueError("all six frozen conditions require K=128")
    return conditions
