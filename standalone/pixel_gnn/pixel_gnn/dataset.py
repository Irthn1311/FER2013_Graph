"""Dataset loader for FER2013 pixel graphs without landmark priors."""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from lap_gnn_tf.constants import SPLIT_COUNTS
from pixel_gnn.grid import StaticGridTopology

SPLIT_FILENAMES = {
    "train": ["train.csv"],
    "val": ["val.csv", "validation.csv"],
    "test": ["test.csv"],
}


def resolve_split_source(fer_csv: str | Path, split: str) -> tuple[Path, bool]:
    source = Path(fer_csv)
    explicit_directory = source.is_dir()
    if explicit_directory:
        source = source / "train.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
    has_usage = "usage" in [x.lower().strip() for x in header]
    known_names = {name for names in SPLIT_FILENAMES.values() for name in names}
    has_sibling_split = source.name.lower() in known_names and any(
        (source.parent / name).is_file()
        for name in known_names
        if name != source.name
    )
    presplit = explicit_directory or has_sibling_split or not has_usage
    if presplit:
        candidates = [
            source.parent / name
            for name in SPLIT_FILENAMES[split]
            if (source.parent / name).is_file()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Need exactly one existing {split} CSV in {source.parent}; got {candidates}"
            )
        return candidates[0], False
    return source, True


def extract_node_features_single(image_48: np.ndarray, coords_norm: np.ndarray) -> np.ndarray:
    img = np.asarray(image_48, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    gy, gx = np.gradient(img)
    intensity = img.reshape(-1, 1)
    gx_flat = gx.reshape(-1, 1)
    gy_flat = gy.reshape(-1, 1)

    return np.concatenate([intensity, coords_norm, gx_flat, gy_flat], axis=1).astype(np.float32)


class FERPixelDataset:
    def __init__(self, fer_csv: str | Path, split: str):
        if split not in SPLIT_COUNTS:
            raise ValueError(f"Unknown split: {split}")
        source, filter_usage = resolve_split_source(fer_csv, split)
        started = time.perf_counter()
        selection = "Usage" if filter_usage else "pre-split file"
        print(f"[DATA] {split}: loading {source} ({selection})", flush=True)

        self.images, self.labels = [], []
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = {x.lower().strip(): x for x in reader.fieldnames or []}
            if not {"emotion", "pixels"}.issubset(columns):
                raise ValueError("FER CSV requires emotion and pixels columns")
            expected_usage = {
                "train": "training",
                "val": "publictest",
                "test": "privatetest",
            }[split]
            for row in reader:
                if filter_usage and row[columns["usage"]].strip().lower() != expected_usage:
                    continue
                values = np.fromstring(row[columns["pixels"]], dtype=np.float32, sep=" ")
                label = int(row[columns["emotion"]])
                if values.size != 2304 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
                    raise ValueError(f"Invalid pixel row {len(self.images)} in {source}")
                if not 0 <= label < 7:
                    raise ValueError(f"Invalid FER label: {label}")
                self.images.append(values.astype(np.uint8).reshape(48, 48))
                self.labels.append(label)
                if len(self.images) % 5000 == 0:
                    print(
                        f"[DATA] {split}: parsed {len(self.images)} rows in {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

        if len(self.images) != SPLIT_COUNTS[split]:
            raise ValueError(
                f"Frozen {split} split requires {SPLIT_COUNTS[split]} rows; found {len(self.images)} "
                f"in {source}. Check train.csv / val.csv / test.csv integrity."
            )

        self.split = split
        self.coords_norm = StaticGridTopology.get_instance().normalized_coords.numpy()
        self.provenance = {
            "split": split,
            "path": str(source.resolve()),
            "rows": len(self.images),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        print(f"[DATA] {split}: ready, {len(self.images)} rows in {time.perf_counter() - started:.1f}s", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        img = self.images[index]
        nodes = extract_node_features_single(img, self.coords_norm)
        return {
            "node_features": nodes,
            "label": int(self.labels[index]),
            "sample_id": int(index),
            "image_48": img.astype(np.float32) / 255.0,
        }
