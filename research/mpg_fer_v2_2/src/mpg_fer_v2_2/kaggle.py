"""Fail-closed Kaggle data and explicit MPG-FER v2.2 resume resolution."""

from __future__ import annotations

import os
from pathlib import Path
import json

from .checkpoint import find_latest_valid_snapshot, sha256_file

from .data import validate_split_paths


DATASET_SLUG = "doduyquynii/fer13-split"
ATTACHED_ROOTS = (
    Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split"),
    Path("/kaggle/input/fer13-split/fer13-split"),
    Path("/kaggle/input/fer13-split"),
)


def _complete_split_triplet(root: Path) -> tuple[Path, Path, Path] | None:
    paths = (root / "train.csv", root / "val.csv", root / "test.csv")
    return paths if all(path.is_file() for path in paths) else None


def resolve_kaggle_splits(
    download_root: str | Path = "/kaggle/working/fer13-split-download",
) -> tuple[Path, Path, Path]:
    """Use an attached dataset or download it with Kaggle Secrets.

    Credentials are read only from ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``
    secrets. The key is never printed or written to ``kaggle.json``.
    """
    for root in ATTACHED_ROOTS:
        triplet = _complete_split_triplet(root)
        if triplet is not None:
            return validate_split_paths(*triplet)  # type: ignore[return-value]

    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        username = secrets.get_secret("KAGGLE_USERNAME")
        key = secrets.get_secret("KAGGLE_KEY")
    except Exception as exc:
        raise RuntimeError(
            "FER13 split is not attached. Add Kaggle Secrets KAGGLE_USERNAME "
            "and KAGGLE_KEY to download doduyquynii/fer13-split."
        ) from exc
    if not username or not key:
        raise RuntimeError(
            "FER13 split is not attached and Kaggle credentials are incomplete"
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    target = Path(download_root)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_SLUG, path=target, unzip=True, quiet=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download Kaggle dataset {DATASET_SLUG}") from exc

    candidates = [target, target / "fer13-split"]
    candidates.extend(path.parent for path in target.rglob("train.csv"))
    for root in candidates:
        triplet = _complete_split_triplet(root)
        if triplet is not None:
            return validate_split_paths(*triplet)  # type: ignore[return-value]
    raise RuntimeError(
        f"Downloaded {DATASET_SLUG}, but train.csv/val.csv/test.csv were not found together"
    )


def resolve_resume_artifact(
    mode: str = "auto",
    explicit_path: str | Path | None = None,
    input_root: str | Path = "/kaggle/input",
) -> tuple[Path | None, str | None]:
    """Resolve only an explicit or clearly named v2.2 resume attachment.

    Auto-discovery is deliberately restricted to datasets whose directory name
    starts with ``mpg-fer-v2-2-resume``. It cannot discover v1/v2/v2.1
    checkpoints or a generic ``best_val_acc.pt``.
    """
    normalized = mode.lower()
    if normalized not in {"auto", "fresh", "required"}:
        raise ValueError("RESUME_MODE must be auto, fresh, or required")
    if normalized == "fresh":
        return None, None
    if explicit_path is not None:
        candidates = [Path(explicit_path)]
    else:
        root = Path(input_root)
        candidates = sorted(
            path
            for path in root.rglob("resume_latest.pt")
            if any(
                part.startswith("mpg-fer-v2-2-resume")
                for part in path.relative_to(root).parts[:-1]
            )
        ) if root.exists() else []
    if not candidates:
        if normalized == "required":
            raise FileNotFoundError("No explicit MPG-FER v2.2 resume artifact attached")
        return None, None
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one v2.2 resume artifact, found {len(candidates)}"
        )
    checkpoint = candidates[0]
    if checkpoint.name != "resume_latest.pt" or not checkpoint.is_file():
        raise RuntimeError(f"Invalid v2.2 resume checkpoint path: {checkpoint}")
    metadata_path = checkpoint.with_name("resume_latest.json")
    if not metadata_path.is_file():
        raise RuntimeError("Resume metadata resume_latest.json is required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("checkpoint_sha256")
    actual = sha256_file(checkpoint)
    if not expected or actual != expected:
        fallback = find_latest_valid_snapshot(
            checkpoint.parent,
            expected_identity=metadata,
        )
        detail = (
            "none"
            if fallback is None
            else f"{fallback[0]} (sha256={fallback[1]})"
        )
        raise RuntimeError(
            "Attached resume checkpoint SHA-256 mismatch; refusing automatic "
            f"fallback. Most recent valid immutable snapshot: {detail}"
        )
    return checkpoint, actual
