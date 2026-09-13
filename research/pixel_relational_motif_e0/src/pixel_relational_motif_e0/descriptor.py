from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

IMAGE_SIZE = 48
WINDOW = 5
RADIUS = 2
VALID_SIDE = IMAGE_SIZE - 2 * RADIUS
VALID_LOCATIONS = VALID_SIDE * VALID_SIDE
EPS_GRAY = 1.0 / 255.0


def _as_image(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.shape == (IMAGE_SIZE, IMAGE_SIZE, 1):
        x = x[..., 0]
    if x.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected 48x48 grayscale image, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("image contains non-finite values")
    return x


def valid_center_coordinates() -> np.ndarray:
    """Return valid center coordinates as integer (y, x), row-major."""
    yy, xx = np.meshgrid(
        np.arange(RADIUS, IMAGE_SIZE - RADIUS, dtype=np.int16),
        np.arange(RADIUS, IMAGE_SIZE - RADIUS, dtype=np.int16),
        indexing="ij",
    )
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def extract_raw_relations(
    image: np.ndarray,
    *,
    eps: float = EPS_GRAY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the locked valid-window E0 descriptor inputs.

    Returns
    -------
    S : (1936, 24)
        Ordered 5x5 center-relative differences normalized by
        sqrt(local_variance + eps^2). Offsets are row-major with the center
        omitted.
    log_sigma : (1936,)
        log(local_std + eps), retained explicitly as relation strength.
    coords : (1936, 2)
        Integer center (y, x) coordinates in the original 48x48 image.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = _as_image(image)
    windows = sliding_window_view(x, (WINDOW, WINDOW))
    flat = windows.reshape(VALID_LOCATIONS, WINDOW * WINDOW)
    center = flat[:, WINDOW * WINDOW // 2]
    sigma = flat.std(axis=1, ddof=0)
    mask = np.ones(WINDOW * WINDOW, dtype=bool)
    mask[WINDOW * WINDOW // 2] = False
    diffs = flat[:, mask] - center[:, None]
    denom = np.sqrt(np.square(sigma) + float(eps) ** 2)
    s = diffs / denom[:, None]
    log_sigma = np.log(sigma + float(eps))
    if s.shape != (VALID_LOCATIONS, 24):
        raise AssertionError(f"descriptor shape invariant violated: {s.shape}")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(log_sigma))):
        raise FloatingPointError("descriptor produced non-finite values")
    return s, log_sigma, valid_center_coordinates()


@dataclass
class DescriptorTransform:
    """Frozen E0 PCA plus explicit z-scored log-contrast coordinate."""

    n_components: int = 11
    random_state: int = 42
    pca: PCA | None = None
    log_sigma_mean: float | None = None
    log_sigma_std: float | None = None

    def fit(self, s: np.ndarray, log_sigma: np.ndarray) -> "DescriptorTransform":
        s = np.asarray(s, dtype=np.float64)
        l = np.asarray(log_sigma, dtype=np.float64).reshape(-1)
        if s.ndim != 2 or s.shape[1] != 24 or len(l) != len(s):
            raise ValueError("fit expects S=(n,24) and matching log_sigma=(n,)")
        self.pca = PCA(
            n_components=self.n_components,
            whiten=False,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca.fit(s)
        self.log_sigma_mean = float(l.mean())
        std = float(l.std(ddof=0))
        self.log_sigma_std = std if std > 1e-12 else 1.0
        return self

    @property
    def fitted(self) -> bool:
        return (
            self.pca is not None
            and self.log_sigma_mean is not None
            and self.log_sigma_std is not None
        )

    def transform(self, s: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("DescriptorTransform must be fit before transform")
        s = np.asarray(s, dtype=np.float64)
        l = np.asarray(log_sigma, dtype=np.float64).reshape(-1)
        if s.ndim != 2 or s.shape[1] != 24 or len(l) != len(s):
            raise ValueError("transform expects S=(n,24) and matching log_sigma")
        p = self.pca.transform(s)
        z = (l - float(self.log_sigma_mean)) / float(self.log_sigma_std)
        out = np.concatenate([p, z[:, None]], axis=1)
        if out.shape[1] != self.n_components + 1:
            raise AssertionError("transformed descriptor dimension invariant violated")
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("transformed descriptor contains non-finite values")
        return out

    def transform_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, l, coords = extract_raw_relations(image)
        return self.transform(s, l), coords
