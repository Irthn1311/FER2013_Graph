"""FER-2013 CSV helpers for D16."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_split_csv(data_dir: str | Path, split: str) -> Path:
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.csv"
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing FER split CSV: {path}")


def read_fer_split(data_dir: str | Path, split: str, max_samples: int | None = None) -> pd.DataFrame:
    path = resolve_split_csv(data_dir, split)
    df = pd.read_csv(path, usecols=["emotion", "pixels"])
    if max_samples is not None:
        df = df.iloc[: int(max_samples)].copy()
    df["sample_index"] = df.index.astype(int)
    return df


def available_splits(data_dir: str | Path, requested: Iterable[str]) -> list[str]:
    data_dir = Path(data_dir)
    out = []
    for split in requested:
        if (data_dir / f"{split}.csv").exists():
            out.append(str(split))
    return out
