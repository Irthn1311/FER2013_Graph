"""Dataset loading, streaming validation, and paired deterministic preprocessing for Pure-GNN scientific runs."""

import csv
import hashlib
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from pure_gnn_v31.scientific.augmentation import augment_image_stateless
from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    validate_split_row_counts,
    DataGovernanceError,
)


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Computes exact raw binary SHA256 of a file using chunked streaming."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_and_hash_fer_csv(
    csv_path: Union[str, Path],
    expected_role: str,
    expected_rows: Optional[int] = None,
) -> Dict[str, Union[str, int]]:
    """Performs streaming validation and raw binary SHA256 computation of a FER2013 CSV."""
    p = validate_dataset_path(csv_path, expected_role=expected_role)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    # Compute TRUE raw binary whole-file SHA256
    true_sha256 = compute_file_sha256(p)
    row_count = 0

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {p}")

        header = [col.strip().lower() for col in raw_header]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"Invalid FER CSV header in {p}: {header}. Expected 'emotion' and 'pixels'.")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        for row in reader:
            if not row:
                continue

            if len(row) <= max(emo_idx, pix_idx):
                raise ValueError(f"Malformed row {row_count} in {p}: incomplete columns.")

            # Validate emotion
            try:
                emotion = int(row[emo_idx])
            except ValueError:
                raise ValueError(f"Malformed emotion label at row {row_count} in {p}: '{row[emo_idx]}'")
            if not (0 <= emotion <= 6):
                raise ValueError(f"Emotion label out of range [0, 6] at row {row_count} in {p}: {emotion}")

            # Validate pixels
            raw_pixels = row[pix_idx].strip().split()
            if len(raw_pixels) != 2304:
                raise ValueError(
                    f"Pixel count mismatch at row {row_count} in {p}: got {len(raw_pixels)}, expected 2304."
                )

            for pix_idx_in_row, p_str in enumerate(raw_pixels):
                try:
                    val = float(p_str)
                except ValueError:
                    raise ValueError(f"Non-numeric pixel value at row {row_count}, index {pix_idx_in_row}: '{p_str}'")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Non-finite pixel value at row {row_count}, index {pix_idx_in_row}: {val}")
                if not (0.0 <= val <= 255.0):
                    raise ValueError(
                        f"Pixel value out of range [0, 255] at row {row_count}, index {pix_idx_in_row}: {val}"
                    )

            row_count += 1

    if expected_rows is not None and row_count != expected_rows:
        raise DataGovernanceError(
            f"Row count mismatch for {expected_role} in {p}: observed {row_count}, expected {expected_rows}."
        )

    return {
        "file_path": str(p),
        "role": expected_role,
        "row_count": row_count,
        "sha256": true_sha256,
    }


def load_fer_csv_split(
    csv_path: Union[str, Path],
    role: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Production scientific loader for official FER2013 CSV splits.

    ALWAYS enforces exact row count (Train=28,709, Val=3,589).
    NO bypass argument exists in the production API.

    Returns:
        images: (N, 48, 48, 1) float32 in [0, 1]
        labels: (N,) int32
        source_row_indices: (N,) int32 (0-indexed position from original CSV)
        sha256: exact raw binary file SHA256
    """
    validate_dataset_path(csv_path, expected_role=role)
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    # Compute true raw binary whole-file SHA256
    true_sha256 = compute_file_sha256(p)

    images: List[np.ndarray] = []
    labels: List[int] = []
    indices: List[int] = []

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {p}")

        header = [col.strip().lower() for col in raw_header]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"Invalid FER CSV header: {header}")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        count = 0
        for row in reader:
            if not row:
                continue

            emotion = int(row[emo_idx])
            if not (0 <= emotion <= 6):
                raise ValueError(f"Emotion out of range: {emotion}")

            raw_pixels = row[pix_idx].split()
            if len(raw_pixels) != 2304:
                raise ValueError(f"Expected 2304 pixels, got {len(raw_pixels)} at row {count}")

            pixel_vals = []
            for px in raw_pixels:
                v = float(px)
                if math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 255.0):
                    raise ValueError(f"Invalid pixel: {v}")
                pixel_vals.append(v)

            img = np.array(pixel_vals, dtype=np.float32).reshape(48, 48, 1) / 255.0
            images.append(img)
            labels.append(emotion)
            indices.append(count)
            count += 1

    expected = 28709 if role.lower() == "train" else 3589
    validate_split_row_counts(role, count, expected)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        true_sha256,
    )


def _load_synthetic_fer_csv_for_testing(
    csv_path: Union[str, Path],
    role: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Internal test fixture loader that does not enforce official full-dataset row counts."""
    validate_dataset_path(csv_path, expected_role=role)
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    true_sha256 = compute_file_sha256(p)
    images: List[np.ndarray] = []
    labels: List[int] = []
    indices: List[int] = []

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {p}")

        header = [col.strip().lower() for col in raw_header]
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        count = 0
        for row in reader:
            if not row:
                continue
            emotion = int(row[emo_idx])
            pixel_vals = [float(px) for px in row[pix_idx].split()]
            img = np.array(pixel_vals, dtype=np.float32).reshape(48, 48, 1) / 255.0
            images.append(img)
            labels.append(emotion)
            indices.append(count)
            count += 1

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        true_sha256,
    )


def create_paired_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
    indices: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset yielding identical sample ordering across paired conditions."""
    if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Explicit positive integer batch_size is required, got: {batch_size}")
    if seed is None or not isinstance(seed, int):
        raise ValueError(f"Explicit integer seed is required, got: {seed}")
    if shuffle is None or not isinstance(shuffle, bool):
        raise ValueError(f"Explicit boolean shuffle is required, got: {shuffle}")

    if indices is None:
        indices = np.arange(len(images), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((images, labels, indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_epoch_paired_training_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    base_seed: int,
    epoch: int,
    augment: bool,
) -> tf.data.Dataset:
    """Creates an epoch-specific training dataset yielding exact paired order and augmentation.

    Guarantees:
    - Base seed and epoch deterministically control shuffle order.
    - If augment=True, applies stateless Gen2/Gen3 image augmentation conditioned only
      on base_seed, epoch, and post-shuffle order index.
    - Different conditions (G0, G0.5, G1) passed the same base_seed and epoch produce
      bit-for-bit identical batches.
    """
    if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Explicit positive integer batch_size is required, got: {batch_size}")
    if base_seed is None or not isinstance(base_seed, int):
        raise ValueError(f"Explicit integer base_seed is required, got: {base_seed}")
    if epoch is None or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Explicit non-negative integer epoch is required, got: {epoch}")

    # Derive deterministic epoch shuffle seed
    epoch_shuffle_seed = (base_seed * 10007 + epoch * 997) % (2**31 - 1)

    ds = tf.data.Dataset.from_tensor_slices((images, labels, indices))
    # Reshuffle deterministically per epoch
    ds = ds.shuffle(buffer_size=len(images), seed=epoch_shuffle_seed, reshuffle_each_iteration=False)

    # Attach post-shuffle enumeration index (0..N-1) for deterministic stateless augmentation
    ds = ds.enumerate()

    def _map_fn(post_shuffle_idx, item):
        img, lbl, src_idx = item
        if augment:
            def _py_aug(im, idx):
                return augment_image_stateless(im, base_seed=base_seed, epoch=epoch, item_index=idx)
            img = tf.py_function(_py_aug, [img, post_shuffle_idx], tf.float32)
            img.set_shape([48, 48, 1])
        return img, lbl, src_idx

    ds = ds.map(_map_fn, num_parallel_calls=None)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
