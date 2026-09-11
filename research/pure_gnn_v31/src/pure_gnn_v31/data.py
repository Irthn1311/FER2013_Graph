"""Data loading and governance utilities for Pure-GNN v3.1."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf


EXPECTED_OFFICIAL_TRAIN_ROWS = 28709


def assert_not_test_access(file_path: Union[str, Path]) -> None:
    """Enforces data governance contract: official test split must NEVER be accessed."""
    p_str = str(file_path).lower().replace("\\", "/")
    # Reject paths that point to test data
    if "test.csv" in p_str or "/test/" in p_str or "official_test" in p_str:
        raise PermissionError(
            f"DATA GOVERNANCE VIOLATION: Access to test data is strictly prohibited! "
            f"Attempted path: {file_path}"
        )


def load_fer2013_train_csv(
    csv_path: Union[str, Path],
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads images and labels from the official FER2013 train.csv.
    
    Images are normalized to [0, 1] float32 array of shape (N, 48, 48, 1).
    Labels are int32 array of shape (N,).
    """
    assert_not_test_access(csv_path)
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    images = []
    labels = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [col.strip().lower() for col in next(reader)]
        if "emotion" not in header or "pixels" not in header:
            raise ValueError(f"Invalid FER CSV header: {header}")
        emo_idx = header.index("emotion")
        pix_idx = header.index("pixels")

        row_count = 0
        for row in reader:
            if not row:
                continue
            emotion = int(row[emo_idx])
            pixel_vals = [float(p) for p in row[pix_idx].split()]
            if len(pixel_vals) != 2304:
                raise ValueError(f"Expected 2304 pixels, got {len(pixel_vals)} at row {row_count}")
            img = np.array(pixel_vals, dtype=np.float32).reshape(48, 48, 1) / 255.0
            images.append(img)
            labels.append(emotion)
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def create_research_split_manifest(
    train_csv_path: Union[str, Path],
    seed: int = 42,
    dev_ratio: float = 0.15,
    output_manifest_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """Derives ResearchTrain and ResearchDev indices strictly from official train.csv.
    
    Returns a manifest dictionary containing SHA256 of train CSV, seed, indices,
    and class distribution.
    """
    assert_not_test_access(train_csv_path)
    train_csv_path = Path(train_csv_path)

    # Compute source train CSV SHA256
    file_bytes = train_csv_path.read_bytes()
    source_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # Parse labels for stratified split
    labels = []
    with train_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [col.strip().lower() for col in next(reader)]
        emo_idx = header.index("emotion")
        for row in reader:
            if row:
                labels.append(int(row[emo_idx]))

    num_samples = len(labels)
    if num_samples != EXPECTED_OFFICIAL_TRAIN_ROWS:
        pass  # allow subset during tests, but record actual count

    labels_arr = np.array(labels, dtype=np.int32)
    rng = np.random.RandomState(seed)

    train_indices = []
    dev_indices = []

    # Stratify by class
    unique_classes = np.unique(labels_arr)
    class_counts_train = {}
    class_counts_dev = {}

    for cls in unique_classes:
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_dev = int(round(len(cls_idx) * dev_ratio))
        dev_idx = cls_idx[:n_dev]
        trn_idx = cls_idx[n_dev:]

        dev_indices.extend(dev_idx.tolist())
        train_indices.extend(trn_idx.tolist())
        class_counts_train[int(cls)] = len(trn_idx)
        class_counts_dev[int(cls)] = len(dev_idx)

    train_indices.sort()
    dev_indices.sort()

    manifest = {
        "source_train_csv": str(train_csv_path),
        "source_train_csv_sha256": source_sha256,
        "split_seed": seed,
        "stratification_policy": "stratified_by_emotion",
        "total_source_rows": num_samples,
        "research_train_count": len(train_indices),
        "research_dev_count": len(dev_indices),
        "class_counts_research_train": class_counts_train,
        "class_counts_research_dev": class_counts_dev,
        "research_train_indices": train_indices,
        "research_dev_indices": dev_indices,
    }

    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest_sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    manifest["manifest_sha256"] = manifest_sha256

    if output_manifest_path is not None:
        out_path = Path(output_manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
