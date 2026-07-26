"""FER2013 CSV validation without bundling private data."""

from __future__ import annotations

import csv
from pathlib import Path


def inspect_fer_csv(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FER CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        normalized = [item.strip().lower() for item in header]
        if "emotion" not in normalized or "pixels" not in normalized:
            raise ValueError(f"FER CSV must contain emotion and pixels columns: {header}")
        rows = sum(1 for _ in reader)
    return {"path": str(path.resolve()), "rows": rows, "columns": header}
