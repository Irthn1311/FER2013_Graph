"""Explicit train/validation data and detector-output-cache adapter for WS-HPG."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import tensorflow as tf

from research.candidates.tf_ws_hpg_v1_weak_support.support import (
    FER_PIXEL_COORDINATE_SYSTEM,
    support_from_landmarks,
)


IMAGE_SIZE = 48
PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 7
TRAIN_SAMPLES = 28_709
VALIDATION_SAMPLES = 3_589
ALLOWED_CACHE_FIELDS = frozenset(
    {"sample_index", "label", "image_48", "detected", "landmark_xy_48"}
)
FORBIDDEN_CACHE_FIELDS = frozenset(
    {
        "face_mask",
        "part_soft_masks",
        "micro_anchor_maps",
        "distance_maps",
        "valid_part_mask",
        "valid_anchor_mask",
        "quality_score",
    }
)
ALLOWED_SPLITS = frozenset({"train", "val", "validation"})
_FORBIDDEN_DIRECTORIES = frozenset({"test", "testing", "test_split", "test-split"})
_TEST_BASENAME = re.compile(r"^test(?:\.csv|[_-].+\.csv)?$", re.IGNORECASE)


class WSHPGDataError(ValueError):
    """Fail-closed data/cache contract violation."""


def reject_test_path(path: str | Path) -> Path:
    """Lexically reject test-like paths before any open, read, hash, or listing."""

    source = Path(path).expanduser().absolute()
    if _TEST_BASENAME.fullmatch(source.name):
        raise WSHPGDataError(f"Test path is forbidden: {source}")
    components = {part.casefold() for part in source.parts}
    if components & _FORBIDDEN_DIRECTORIES:
        raise WSHPGDataError(f"Test directory is forbidden: {source}")
    return source


def validate_split(split: str) -> str:
    value = str(split).casefold()
    if value not in ALLOWED_SPLITS:
        raise WSHPGDataError(f"Only explicit train/validation cache splits are allowed: {split}")
    return value


def load_fer_csv(path: str | Path, expected_samples: int | None = None):
    source = reject_test_path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    images, labels = [], []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"emotion", "pixels"} <= set(reader.fieldnames):
            raise WSHPGDataError("CSV requires emotion and pixels columns")
        for row_index, row in enumerate(reader):
            label = int(row["emotion"])
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
            if pixels.size != PIXEL_COUNT or not 0 <= label < NUM_CLASSES:
                raise WSHPGDataError(f"Invalid FER row {row_index}")
            images.append(pixels.reshape(48, 48, 1))
            labels.append(label)
    if expected_samples is not None and len(labels) != expected_samples:
        raise WSHPGDataError(f"Observed {len(labels)} samples; expected {expected_samples}")
    return np.asarray(images, np.float32), np.asarray(labels, np.int32)


@dataclass(frozen=True)
class CachedSupport:
    support: np.ndarray
    detected: bool
    coverage: float


def _read_allowed(record: Mapping, field: str):
    if field not in ALLOWED_CACHE_FIELDS:
        raise WSHPGDataError(f"Cache field is not allowlisted: {field}")
    return record[field]


def support_from_cache_record(
    record: Mapping,
    *,
    expected_sample_index: int,
    expected_label: int,
    expected_clean_image: np.ndarray,
) -> CachedSupport:
    """Read only the five allowlisted arrays and immediately discard landmarks."""

    sample_index = int(np.asarray(_read_allowed(record, "sample_index")).item())
    label = int(np.asarray(_read_allowed(record, "label")).item())
    image = np.asarray(_read_allowed(record, "image_48"), np.float32)
    detected = bool(np.asarray(_read_allowed(record, "detected")).item())
    expected_image = np.asarray(expected_clean_image, np.float32).reshape(48, 48)
    if sample_index != int(expected_sample_index):
        raise WSHPGDataError("Cache sample_index alignment mismatch")
    if label != int(expected_label):
        raise WSHPGDataError("Cache label alignment mismatch")
    if image.shape != (48, 48) or not np.array_equal(image, expected_image):
        raise WSHPGDataError("Cache clean-image identity mismatch")
    landmarks = None
    if detected:
        try:
            landmarks = np.asarray(_read_allowed(record, "landmark_xy_48"), np.float32)
        except KeyError:
            detected = False
    if (
        landmarks is None
        or landmarks.ndim != 2
        or landmarks.shape[1:] != (2,)
        or landmarks.shape[0] == 0
        or not np.all(np.isfinite(landmarks))
        or np.any(landmarks < 0.0)
        or np.any(landmarks > 47.0)
    ):
        detected = False
        field = np.ones((48, 48, 1), np.float32)
    else:
        field = support_from_landmarks(
            landmarks, coordinate_system=FER_PIXEL_COORDINATE_SYSTEM
        )
        if np.all(field == 1.0):
            detected = False
    return CachedSupport(field, detected, float(np.mean(field)))


def load_cached_support(
    prior_root: str | Path,
    split: str,
    sample_index: int,
    label: int,
    clean_image: np.ndarray,
) -> CachedSupport:
    root = reject_test_path(prior_root)
    split = validate_split(split)
    path = root / split / f"{int(sample_index):06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as record:
        return support_from_cache_record(
            record,
            expected_sample_index=sample_index,
            expected_label=label,
            expected_clean_image=clean_image,
        )


def load_support_split(prior_root, split, images, labels):
    """Use exact indexed cache paths; never glob or discover another split."""

    supports, detected, coverages = [], [], []
    for index, (image, label) in enumerate(zip(images, labels)):
        cached = load_cached_support(prior_root, split, index, int(label), image)
        supports.append(cached.support)
        detected.append(cached.detected)
        coverages.append(cached.coverage)
    return (
        np.asarray(supports, np.float32),
        np.asarray(detected, bool),
        np.asarray(coverages, np.float32),
    )


def _training_records(images, supports, labels, *, seed=42):
    """Match accepted CF/RA ordering: shuffle first, then enumerate positions."""

    original_sample_indices = tf.range(tf.shape(images)[0])
    records = tf.data.Dataset.from_tensor_slices(
        (original_sample_indices, images, supports, labels)
    )
    return records.shuffle(
        int(tf.shape(images)[0]), seed=seed, reshuffle_each_iteration=True
    ).enumerate()


def build_dataset(images, supports, labels: Iterable[int], *, training: bool, batch_size=64):
    from .augmentation import augment_example

    if training:
        dataset = _training_records(images, supports, labels)

        def map_train(augmentation_index, record):
            original_sample_index, image, field, label = record
            del original_sample_index  # Identity already served cache/support alignment.
            inputs, _, _ = augment_example(image, field, augmentation_index)
            return inputs, label

        dataset = dataset.map(
            map_train,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        indices = tf.range(tf.shape(images)[0])
        dataset = tf.data.Dataset.from_tensor_slices(
            (indices, images, supports, labels)
        )
        dataset = dataset.map(
            lambda index, image, field, label: (
                {"images": tf.cast(image, tf.float32) / 255.0, "support": tf.cast(field, tf.float32)},
                label,
            ),
            deterministic=True,
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)
