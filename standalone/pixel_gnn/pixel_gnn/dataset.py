"""Dataset loader for FER2013 pixel graphs without landmark priors."""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from pixel_gnn.utils import SPLIT_COUNTS
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


def extract_node_features_single(image_48: np.ndarray, coords_norm: np.ndarray, node_dim: int = 5) -> np.ndarray:
    img = np.asarray(image_48, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    gy, gx = np.gradient(img)
    intensity = img.reshape(-1, 1)
    gx_flat = gx.reshape(-1, 1)
    gy_flat = gy.reshape(-1, 1)

    feats = [intensity, coords_norm, gx_flat, gy_flat]
    if node_dim == 7:
        grad_mag = np.sqrt(gx**2 + gy**2).reshape(-1, 1)
        gyy, _ = np.gradient(gy)
        _, gxx = np.gradient(gx)
        laplacian = (gxx + gyy).reshape(-1, 1)
        feats.extend([grad_mag, laplacian])

    return np.concatenate(feats, axis=1).astype(np.float32)


class FERPixelDataset:
    def __init__(self, fer_csv: str | Path, split: str, node_dim: int = 5):
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
        self.node_dim = int(node_dim)
        self.coords_norm = StaticGridTopology.get_instance().normalized_coords.numpy()
        self.provenance = {
            "split": split,
            "path": str(source.resolve()),
            "rows": len(self.images),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

        # Vectorized node features precomputation (<0.5s for 28k images)
        t_pre = time.perf_counter()
        imgs_arr = np.stack(self.images, axis=0).astype(np.float32) / 255.0  # [N, 48, 48]
        gy, gx = np.gradient(imgs_arr, axis=(1, 2))                          # [N, 48, 48]
        n_samples = len(self.images)
        intensity = imgs_arr.reshape(n_samples, 2304, 1)
        coords_exp = np.broadcast_to(self.coords_norm[None, :, :], (n_samples, 2304, 2))
        gx_flat = gx.reshape(n_samples, 2304, 1)
        gy_flat = gy.reshape(n_samples, 2304, 1)

        feats = [intensity, coords_exp, gx_flat, gy_flat]
        if self.node_dim >= 7:
            grad_mag = np.sqrt(gx**2 + gy**2).reshape(n_samples, 2304, 1)
            gyy, _ = np.gradient(gy, axis=(1, 2))
            _, gxx = np.gradient(gx, axis=(1, 2))
            laplacian = (gxx + gyy).reshape(n_samples, 2304, 1)
            feats.extend([grad_mag, laplacian])

        if self.node_dim == 9:
            # Pure pixel 8-neighbor local variance and LBP code (No CNNs)
            grid = StaticGridTopology.get_instance()
            neighbors_idx = grid.neighbors_idx.numpy()      # [2304, 8]
            neighbor_valid = grid.neighbor_valid.numpy()    # [2304, 8]

            # Gather neighbor intensities: [N, 2304, 8]
            intensity_2d = intensity.squeeze(-1)            # [N, 2304]
            nbr_intensity = intensity_2d[:, neighbors_idx]   # [N, 2304, 8]
            diff = nbr_intensity - intensity_2d[:, :, None] # [N, 2304, 8]

            valid_mask = neighbor_valid[None, :, :].astype(np.float32)
            # Local variance
            var_flat = np.sum((diff**2) * valid_mask, axis=-1, keepdims=True) / (np.sum(valid_mask, axis=-1, keepdims=True) + 1e-6)

            # Local Binary Pattern (8-bit code normalized to [0, 1])
            powers = (2 ** np.arange(8, dtype=np.float32))[None, None, :]
            lbp_bits = (diff >= 0).astype(np.float32) * valid_mask
            lbp_flat = np.sum(lbp_bits * powers, axis=-1, keepdims=True) / 255.0

            feats.extend([var_flat, lbp_flat])

        self.all_node_features = np.concatenate(feats, axis=-1).astype(np.float32)  # [N, 2304, node_dim]
        self.all_labels = np.array(self.labels, dtype=np.int64)
        self.all_sample_ids = np.arange(n_samples, dtype=np.int64)
        self.all_images = imgs_arr

        print(
            f"[DATA] {split}: ready, {len(self.images)} rows precomputed in "
            f"{time.perf_counter() - started:.1f}s (node_dim={self.node_dim}, vectorization: {time.perf_counter() - t_pre:.2f}s)",
            flush=True,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        return {
            "node_features": self.all_node_features[index],
            "label": int(self.all_labels[index]),
            "sample_id": int(self.all_sample_ids[index]),
            "image_48": self.all_images[index],
        }
