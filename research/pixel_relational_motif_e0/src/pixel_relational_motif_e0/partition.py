from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ImagePartition:
    fit: np.ndarray
    heldout: np.ndarray
    anchor: np.ndarray

    def assert_valid(self, n_images: int) -> None:
        parts = [np.asarray(self.fit), np.asarray(self.heldout), np.asarray(self.anchor)]
        all_idx = np.concatenate(parts)
        if len(np.unique(all_idx)) != len(all_idx):
            raise AssertionError("image-level partitions overlap")
        if set(all_idx.tolist()) != set(range(n_images)):
            raise AssertionError("image-level partitions do not cover exactly all images")


def make_image_partition(
    n_images: int,
    *,
    fit_fraction: float = 0.70,
    heldout_fraction: float = 0.15,
    seed: int = 42,
) -> ImagePartition:
    """Deterministic label-free image-level partition for E0 dictionary work."""
    if n_images < 3:
        raise ValueError("need at least three images")
    if not (0 < fit_fraction < 1 and 0 < heldout_fraction < 1):
        raise ValueError("fractions must lie in (0,1)")
    if fit_fraction + heldout_fraction >= 1:
        raise ValueError("fit + heldout fractions must leave a non-empty anchor split")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_images)
    n_fit = int(np.floor(n_images * fit_fraction))
    n_held = int(np.floor(n_images * heldout_fraction))
    part = ImagePartition(
        fit=np.sort(idx[:n_fit]),
        heldout=np.sort(idx[n_fit : n_fit + n_held]),
        anchor=np.sort(idx[n_fit + n_held :]),
    )
    part.assert_valid(n_images)
    return part


def assert_e0_role(role: str) -> str:
    """Fail closed on PrivateTest/test roles for E0 runners."""
    normalized = role.strip().lower().replace("-", "_")
    if normalized in {"test", "private_test", "privatetest", "final_test"}:
        raise ValueError("E0 must not read FER2013 PrivateTest/final-test data")
    if normalized not in {"train", "validation", "public_test", "publictest", "dev"}:
        raise ValueError(f"unsupported E0 data role: {role!r}")
    return normalized
