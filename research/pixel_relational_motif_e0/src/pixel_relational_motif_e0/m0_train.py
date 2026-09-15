"""Fixed training and per-fit artifact contracts for preregistered PGM M0."""

from __future__ import annotations

import json
import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import sklearn
import torch
from torch import nn

from .e02_runner import sha256_array, sha256_file
from .m0_config import (
    EPOCHS,
    EXPERIMENT_ID,
    G_CONFIG,
    G_CONFIG_SHA256,
    ISSUE_NUMBER,
    L_CONFIG,
    L_CONFIG_SHA256,
    M_CONFIG,
    M_CONFIG_SHA256,
    SEEDS,
)
from .m0_models import MinimalRelationGraphModel, SparseAggregateMLP, parameter_count
from .m0_substrate import LoadedM0Substrate


def environment_record() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def seed_everything(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError("unregistered M0 seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def scipy_batch_to_torch(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    batch = matrix[rows].tocoo()
    indices = torch.from_numpy(np.vstack([batch.row, batch.col]).astype(np.int64))
    values = torch.from_numpy(batch.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, batch.shape, dtype=torch.float32, device=device).coalesce()


def graph_batch(graph: dict[str, np.ndarray], rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sizes = graph["node_offsets"][rows + 1] - graph["node_offsets"][rows]
    width = int(sizes.max(initial=0))
    width = max(width, 1)
    components = np.zeros((len(rows), width), dtype=np.int64)
    relations = np.full((len(rows), width, width), 255, dtype=np.uint8)
    mask = np.zeros((len(rows), width), dtype=bool)
    for slot, row in enumerate(rows):
        lo, hi = int(graph["node_offsets"][row]), int(graph["node_offsets"][row + 1])
        n = hi - lo
        if n:
            components[slot, :n] = graph["components"][lo:hi]
            rlo, rhi = int(graph["relation_offsets"][row]), int(graph["relation_offsets"][row + 1])
            relations[slot, :n, :n] = graph["relations"][rlo:rhi].reshape(n, n)
            mask[slot, :n] = True
    return (
        torch.from_numpy(components).to(device),
        torch.from_numpy(relations).to(device),
        torch.from_numpy(mask).to(device),
    )


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    for key, value in payload.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite prediction field {key}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("expected scalar text")
    return str(array.item())


def validate_fit_artifact(path: str | Path, substrate: LoadedM0Substrate, *, family: str, seed: int | None, scientific_sha: str, wrapper_sha: str) -> dict:
    path = Path(path)
    config_hash = {"L": L_CONFIG_SHA256, "M": M_CONFIG_SHA256, "G": G_CONFIG_SHA256}[family]
    with np.load(path, allow_pickle=False) as artifact:
        expected = {
            "experiment": EXPERIMENT_ID, "scientific_sha": scientific_sha, "wrapper_sha": wrapper_sha,
            "model_family": family, "substrate_sha256": substrate.manifest["substrate_sha256"],
            "train_ids_sha256": substrate.manifest["train_ids_sha256"],
            "public_ids_sha256": substrate.manifest["public_ids_sha256"], "model_config_sha256": config_hash,
        }
        for key, value in expected.items():
            if _text(artifact[key]) != value:
                raise ValueError(f"completed artifact {key} mismatch")
        if int(artifact["issue"]) != ISSUE_NUMBER or int(artifact["seed"]) != (-1 if seed is None else seed):
            raise ValueError("completed artifact issue/seed mismatch")
        if bool(artifact["public_metrics_present"]):
            raise ValueError("completed artifact contains Public metrics")
        prediction = np.asarray(artifact["public_predictions"])
        if prediction.dtype != np.int8 or prediction.shape != (len(substrate.public_ids),) or np.any((prediction < 0) | (prediction >= 7)):
            raise ValueError("completed artifact prediction mismatch")
        if family != "L":
            if int(artifact["final_epoch"]) != EPOCHS or len(artifact["training_loss"]) != EPOCHS:
                raise ValueError("completed artifact epoch mismatch")
            if int(artifact["parameter_count"]) != int((M_CONFIG if family == "M" else G_CONFIG)["expected_parameter_count"]):
                raise ValueError("completed artifact parameter count mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "status": "SKIPPED_VALID_COMPLETE"}


def fit_l(substrate: LoadedM0Substrate, output_dir: str | Path, *, scientific_sha: str, wrapper_sha: str, account: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "m0_L_prediction.npz"
    if path.is_file():
        return validate_fit_artifact(path, substrate, family="L", seed=None, scientific_sha=scientific_sha, wrapper_sha=wrapper_sha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, tol=1e-6
        )
        model.fit(substrate.train_x, substrate.train_labels)
    prediction = model.predict(substrate.public_x).astype(np.int8)
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    diagnostic = {
        "n_iter": np.asarray(model.n_iter_).astype(int).tolist(),
        "converged": not convergence_warnings and bool(np.all(np.asarray(model.n_iter_) < 2000)),
        "warnings": [str(item.message) for item in caught],
        "train_objective_available": False,
    }
    _atomic_npz(path, {
        "experiment": np.asarray(EXPERIMENT_ID), "issue": np.asarray(ISSUE_NUMBER, np.int32),
        "scientific_sha": np.asarray(scientific_sha), "wrapper_sha": np.asarray(wrapper_sha),
        "model_family": np.asarray("L"), "seed": np.asarray(-1, np.int32),
        "substrate_sha256": np.asarray(substrate.manifest["substrate_sha256"]),
        "train_ids_sha256": np.asarray(substrate.manifest["train_ids_sha256"]),
        "public_ids_sha256": np.asarray(substrate.manifest["public_ids_sha256"]),
        "model_config_sha256": np.asarray(L_CONFIG_SHA256),
        "public_predictions": prediction, "n_iter": np.asarray(model.n_iter_, np.int64),
        "converged": np.asarray(diagnostic["converged"]),
        "warnings_json": np.asarray(json.dumps(diagnostic["warnings"], allow_nan=False)),
        "environment_json": np.asarray(json.dumps(environment_record(), sort_keys=True, allow_nan=False)),
        "account": np.asarray(account), "public_metrics_present": np.asarray(False),
    })
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "diagnostic": diagnostic}


def _train_one(
    substrate: LoadedM0Substrate,
    family: str,
    seed: int,
    output_dir: Path,
    scientific_sha: str,
    wrapper_sha: str,
    account: str,
    device: torch.device,
) -> dict:
    if family not in {"M", "G"} or seed not in SEEDS:
        raise ValueError("invalid M0 family/seed")
    seed_everything(seed)
    config = M_CONFIG if family == "M" else G_CONFIG
    config_hash = M_CONFIG_SHA256 if family == "M" else G_CONFIG_SHA256
    model: nn.Module = SparseAggregateMLP() if family == "M" else MinimalRelationGraphModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    checkpoint = output_dir / f"m0_{family}_seed_{seed}_epoch_checkpoint.pt"
    losses: list[float] = []
    start_epoch = 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["family"] != family or state["seed"] != seed or state["scientific_sha"] != scientific_sha or state["substrate_sha256"] != substrate.manifest["substrate_sha256"]:
            raise ValueError("epoch checkpoint provenance mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng(state["rng"])
        losses = list(state["losses"])
        start_epoch = int(state["epoch"])
    batch_size = int(config["batch_size"])
    for epoch in range(start_epoch, EPOCHS):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(substrate.train_labels), generator=generator).numpy()
        model.train()
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            label = torch.from_numpy(substrate.train_labels[rows].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            if family == "M":
                logits = model(scipy_batch_to_torch(substrate.train_x, rows, device))
            else:
                logits = model(*graph_batch(substrate.train_graph, rows, device))
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows)
            total_rows += len(rows)
        losses.append(total_loss / total_rows)
        _atomic_torch_save(checkpoint, {
            "experiment": EXPERIMENT_ID, "issue": ISSUE_NUMBER, "scientific_sha": scientific_sha,
            "wrapper_sha": wrapper_sha, "substrate_sha256": substrate.manifest["substrate_sha256"],
            "family": family, "seed": seed, "epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(),
            "losses": losses, "config_sha256": config_hash,
        })
        print(f"[PGM-M0] {family}{seed} epoch={epoch + 1}/{EPOCHS} loss={losses[-1]:.9f}", flush=True)

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(substrate.public_ids), batch_size):
            rows = np.arange(start, min(start + batch_size, len(substrate.public_ids)))
            if family == "M":
                logits = model(scipy_batch_to_torch(substrate.public_x, rows, device))
            else:
                logits = model(*graph_batch(substrate.public_graph, rows, device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int8))
    prediction = np.concatenate(predictions)
    rng_meta = {"seed": seed, "python": seed, "numpy": seed, "torch": seed, "cuda": seed, "dataloader_epoch_key": "seed*1000+epoch"}
    path = output_dir / f"m0_{family}_seed_{seed}.npz"
    _atomic_npz(path, {
        "experiment": np.asarray(EXPERIMENT_ID), "issue": np.asarray(ISSUE_NUMBER, np.int32),
        "scientific_sha": np.asarray(scientific_sha), "wrapper_sha": np.asarray(wrapper_sha),
        "model_family": np.asarray(family), "seed": np.asarray(seed, np.int32),
        "substrate_sha256": np.asarray(substrate.manifest["substrate_sha256"]),
        "train_ids_sha256": np.asarray(substrate.manifest["train_ids_sha256"]),
        "public_ids_sha256": np.asarray(substrate.manifest["public_ids_sha256"]),
        "model_config_sha256": np.asarray(config_hash), "final_epoch": np.asarray(EPOCHS, np.int32),
        "public_predictions": prediction, "training_loss": np.asarray(losses, np.float64),
        "parameter_count": np.asarray(parameter_count(model), np.int64),
        "environment_json": np.asarray(json.dumps(environment_record(), sort_keys=True, allow_nan=False)),
        "rng_json": np.asarray(json.dumps(rng_meta, sort_keys=True, allow_nan=False)),
        "account": np.asarray(account), "public_metrics_present": np.asarray(False),
    })
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "epochs": EPOCHS}


def train_seed(substrate: LoadedM0Substrate, family: str, seed: int, output_dir: str | Path, *, scientific_sha: str, wrapper_sha: str, account: str, device: str | None = None) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"m0_{family}_seed_{seed}.npz"
    if final.is_file():
        return validate_fit_artifact(final, substrate, family=family, seed=seed, scientific_sha=scientific_sha, wrapper_sha=wrapper_sha)
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return _train_one(substrate, family, seed, output, scientific_sha, wrapper_sha, account, target)
