from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeSpec:
    c: float = 1.0
    max_iter: int = 2000
    tol: float = 1e-4
    seed: int = 42


def fit_probe(x, y: np.ndarray, *, spec: ProbeSpec = ProbeSpec()):
    """Fixed E0 probe: L2 multinomial logistic regression only."""
    scaler = StandardScaler(with_mean=False)
    clf = LogisticRegression(penalty="l2", C=spec.c, solver="saga", max_iter=spec.max_iter, tol=spec.tol, random_state=spec.seed)
    pipe = make_pipeline(scaler, clf)
    pipe.fit(x, np.asarray(y))
    return pipe


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    return float(accuracy_score(y, p)), float(f1_score(y, p, average="macro", zero_division=0))


def paired_bootstrap_vs_control_mean(y_true: np.ndarray, pred_actual: np.ndarray, pred_controls: np.ndarray, *, n_replicates: int = 2000, seed: int = 42) -> dict[str, np.ndarray | float]:
    """Paired bootstrap actual minus mean matched-control metric."""
    y = np.asarray(y_true)
    a = np.asarray(pred_actual)
    c = np.asarray(pred_controls)
    if c.ndim != 2 or a.shape != y.shape or c.shape[1] != len(y):
        raise ValueError("prediction shapes do not match labels")
    rng = np.random.default_rng(seed)
    da = np.empty(n_replicates)
    df = np.empty(n_replicates)
    n = len(y)
    for b in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        acc_a, f1_a = metrics(y[idx], a[idx])
        control_metrics = [metrics(y[idx], row[idx]) for row in c]
        da[b] = acc_a - float(np.mean([m[0] for m in control_metrics]))
        df[b] = f1_a - float(np.mean([m[1] for m in control_metrics]))
    return {"delta_accuracy": da, "delta_macro_f1": df, "accuracy_ci95": np.percentile(da, [2.5, 97.5]), "macro_f1_ci95": np.percentile(df, [2.5, 97.5])}
