"""Portable SHA-256 helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path, exclude: set[str] | None = None) -> str:
    root = Path(root)
    excluded = exclude or {"CHECKSUMS.sha256"}
    digest = hashlib.sha256()
    files = [item for item in root.rglob("*") if item.is_file()]
    for path in sorted(
        files, key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or "__pycache__" in path.parts:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def scientific_payload_checksum(package_root: str | Path) -> str:
    package_root = Path(package_root)
    digest = hashlib.sha256()
    roots = ["src/lap_gnn_tf", "contracts", "validation_assets"]
    files = []
    for relative_root in roots:
        root = package_root / relative_root
        files.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(
        files, key=lambda item: item.relative_to(package_root).as_posix()
    ):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
