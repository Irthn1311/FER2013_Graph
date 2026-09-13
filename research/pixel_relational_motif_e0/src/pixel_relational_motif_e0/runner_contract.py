from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .partition import assert_e0_role


@dataclass(frozen=True)
class E0DataPaths:
    train_csv: Path
    public_test_csv: Path

    def validate_roles(self) -> None:
        assert_e0_role("train")
        assert_e0_role("public_test")


def bind_split_path(role: str, path: str | Path) -> Path:
    """Role-gated path binding; intentionally has no PrivateTest role."""
    assert_e0_role(role)
    return Path(path)
