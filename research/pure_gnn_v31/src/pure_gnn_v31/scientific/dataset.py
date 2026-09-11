"""Dataset loading and paired deterministic preprocessing for Pure-GNN scientific runs."""

import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    validate_split_row_counts,
)


def load_fer_csv_split(
    csv_path: Union[str, Path],
    role: str,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a FER2013 split (train or validation only) with strict governance assertions."""
    validate_dataset_path(csv_path, expected_role=role)
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    images: List[np.ndarray] = []
    labels: List[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [col.strip().lower() for col in next(reader)]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"Invalid FER CSV header: {header}")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        count = 0
        for row in reader:
            if not row:
                continue
            emotion = int(row[emo_idx])
            pixel_vals = [float(p) for p in row[pix_idx].split()]
            if len(pixel_vals) != 2304:
                raise ValueError(f"Expected 2304 pixels, got {len(pixel_vals)} at row {count}")
            img = np.array(pixel_vals, dtype=np.float32).reshape(48, 48, 1) / 255.0
            images.append(img)
            labels.append(emotion)
            count += 1
            if max_rows is not None and count >= max_rows:
                break

    # If full split loaded, validate exact row count
    if max_rows is None:
        expected = 28709 if role.lower() == "train" else 3589
        validate_split_row_counts(role, count, expected)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def create_paired_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    seed: int = 42,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset yielding identical ordering across paired conditions."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
