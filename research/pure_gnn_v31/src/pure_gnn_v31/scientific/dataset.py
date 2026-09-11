"""Dataset loading, streaming validation, and paired deterministic preprocessing for Pure-GNN scientific runs."""

import csv
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    validate_split_row_counts,
    DataGovernanceError,
)


def validate_and_hash_fer_csv(
    csv_path: Union[str, Path],
    expected_role: str,
    expected_rows: Optional[int] = None,
) -> Dict[str, Union[str, int]]:
    """Performs streaming validation and SHA256 computation of a FER2013 CSV without loading entire file into RAM."""
    p = validate_dataset_path(csv_path, expected_role=expected_role)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    hasher = hashlib.sha256()
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

        # Update hash with header line
        hasher.update((",".join(raw_header) + "\n").encode("utf-8"))

        for row in reader:
            if not row:
                continue
            hasher.update((",".join(row) + "\n").encode("utf-8"))

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
        "sha256": hasher.hexdigest(),
    }


def load_fer_csv_split(
    csv_path: Union[str, Path],
    role: str,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Loads a FER2013 split (train or validation only) with strict governance assertions.

    Returns:
        images: (N, 48, 48, 1) float32 in [0, 1]
        labels: (N,) int32
        source_row_indices: (N,) int32 (0-indexed position from original CSV)
        sha256: file SHA256
    """
    validate_dataset_path(csv_path, expected_role=role)
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    hasher = hashlib.sha256()
    images: List[np.ndarray] = []
    labels: List[int] = []
    indices: List[int] = []

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        hasher.update((",".join(raw_header) + "\n").encode("utf-8"))

        header = [col.strip().lower() for col in raw_header]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"Invalid FER CSV header: {header}")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        count = 0
        for row in reader:
            if not row:
                continue
            hasher.update((",".join(row) + "\n").encode("utf-8"))

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
            if max_rows is not None and count >= max_rows:
                break

    # If full split loaded, validate exact row count
    if max_rows is None:
        expected = 28709 if role.lower() == "train" else 3589
        validate_split_row_counts(role, count, expected)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        hasher.hexdigest(),
    )


def create_paired_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    indices: Optional[np.ndarray] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset yielding identical sample ordering across paired conditions.

    NO default arguments for batch_size or seed!
    """
    if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Explicit positive integer batch_size is required, got: {batch_size}")
    if seed is None or not isinstance(seed, int):
        raise ValueError(f"Explicit integer seed is required, got: {seed}")

    if indices is None:
        indices = np.arange(len(images), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((images, labels, indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
