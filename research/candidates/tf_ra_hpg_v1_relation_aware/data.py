"""Train/validation data pipeline for the isolated CF-HPG v1.0 candidate."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import tensorflow as tf


IMAGE_SIZE = 48
PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 7
REGISTERED_SEED = 42
GEOMETRIC_FILL_MODE = "REFLECT"
ERASE_FILL_NORMALIZED = 0.0


class CFHPGDataError(ValueError):
    """Raised when an explicitly supplied train/validation CSV is invalid."""


_FORBIDDEN_SPLIT_DIRECTORIES = frozenset(
    {"test", "testing", "test_split", "test-split"}
)
_FORBIDDEN_TEST_BASENAME = re.compile(r"^test(?:\.csv|[_-].+\.csv)$", re.IGNORECASE)


def validate_allowed_csv_path(path: str | Path) -> Path:
    """Reject explicit final-split paths lexically before any file access."""

    source = Path(path).expanduser().resolve()
    if _FORBIDDEN_TEST_BASENAME.fullmatch(source.name):
        raise CFHPGDataError(f"Final-split CSV path is forbidden: {source}")
    directory_components = {part.casefold() for part in source.parts[:-1]}
    if directory_components & _FORBIDDEN_SPLIT_DIRECTORIES:
        raise CFHPGDataError(f"Final-split directory is forbidden: {source}")
    return source


def load_fer_csv(path: str | Path, expected_samples: int | None = None):
    """Load one explicitly supplied FER CSV without split discovery."""

    source = validate_allowed_csv_path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    images: list[np.ndarray] = []
    labels: list[int] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"emotion", "pixels"} <= set(
            reader.fieldnames
        ):
            raise CFHPGDataError("CSV requires emotion and pixels columns")
        for row_index, row in enumerate(reader):
            label = int(row["emotion"])
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
            if pixels.size != PIXEL_COUNT:
                raise CFHPGDataError(
                    f"Row {row_index} has {pixels.size} pixels; expected {PIXEL_COUNT}"
                )
            if not 0 <= label < NUM_CLASSES:
                raise CFHPGDataError(f"Row {row_index} has invalid class {label}")
            images.append(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 1))
            labels.append(label)
    if expected_samples is not None and len(labels) != expected_samples:
        raise CFHPGDataError(
            f"Observed {len(labels)} samples; expected {expected_samples}"
        )
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _seed(sample_index: tf.Tensor, salt: int, seed: int) -> tf.Tensor:
    base = tf.stack([tf.cast(seed, tf.int32), tf.cast(sample_index, tf.int32)])
    return tf.random.experimental.stateless_fold_in(base, salt)


def _geometric_transform(
    image: tf.Tensor, sample_index: tf.Tensor, seed: int
) -> tf.Tensor:
    angle = tf.random.stateless_uniform(
        [], _seed(sample_index, 1, seed), minval=-10.0, maxval=10.0
    )
    angle = angle * (tf.constant(3.141592653589793, tf.float32) / 180.0)
    translation = tf.random.stateless_uniform(
        [2], _seed(sample_index, 2, seed), minval=-4.0, maxval=4.0
    )
    cosine = tf.cos(angle)
    sine = tf.sin(angle)
    center = tf.constant((IMAGE_SIZE - 1) / 2.0, tf.float32)
    dx, dy = translation[0], translation[1]
    offset_x = center - cosine * center - sine * center - cosine * dx - sine * dy
    offset_y = center + sine * center - cosine * center + sine * dx - cosine * dy
    transform = tf.stack(
        [cosine, sine, offset_x, -sine, cosine, offset_y, 0.0, 0.0]
    )[None, :]
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, :, :, :],
        transforms=transform,
        output_shape=tf.constant([IMAGE_SIZE, IMAGE_SIZE], tf.int32),
        interpolation="BILINEAR",
        fill_mode=GEOMETRIC_FILL_MODE,
        fill_value=0.0,
    )
    return transformed[0]


def _random_erasing(
    image: tf.Tensor, sample_index: tf.Tensor, seed: int
) -> tf.Tensor:
    apply = tf.random.stateless_uniform([], _seed(sample_index, 5, seed)) < 0.25

    def erase():
        area_fraction = tf.random.stateless_uniform(
            [], _seed(sample_index, 6, seed), minval=0.02, maxval=0.10
        )
        aspect = tf.random.stateless_uniform(
            [], _seed(sample_index, 7, seed), minval=0.5, maxval=2.0
        )
        area = area_fraction * float(IMAGE_SIZE * IMAGE_SIZE)
        height = tf.clip_by_value(
            tf.cast(tf.round(tf.sqrt(area / aspect)), tf.int32), 1, IMAGE_SIZE
        )
        width = tf.clip_by_value(
            tf.cast(tf.round(tf.sqrt(area * aspect)), tf.int32), 1, IMAGE_SIZE
        )
        top = tf.random.stateless_uniform(
            [],
            _seed(sample_index, 8, seed),
            minval=0,
            maxval=IMAGE_SIZE - height + 1,
            dtype=tf.int32,
        )
        left = tf.random.stateless_uniform(
            [],
            _seed(sample_index, 9, seed),
            minval=0,
            maxval=IMAGE_SIZE - width + 1,
            dtype=tf.int32,
        )
        rows = tf.range(IMAGE_SIZE)[:, None]
        columns = tf.range(IMAGE_SIZE)[None, :]
        mask = tf.logical_and(
            tf.logical_and(rows >= top, rows < top + height),
            tf.logical_and(columns >= left, columns < left + width),
        )
        return tf.where(mask[:, :, None], ERASE_FILL_NORMALIZED, image)

    return tf.cond(apply, erase, lambda: image)


def augment_train_image(
    raw_image: tf.Tensor, sample_index: tf.Tensor, seed: int = REGISTERED_SEED
) -> tf.Tensor:
    """Apply the exact stateless v1.0 augmentation sequence before patchification."""

    raw_image = tf.cast(tf.convert_to_tensor(raw_image), tf.float32)
    tf.debugging.assert_equal(tf.shape(raw_image), [IMAGE_SIZE, IMAGE_SIZE, 1])
    image = raw_image / 127.5 - 1.0
    flip = tf.random.stateless_uniform([], _seed(sample_index, 0, seed)) < 0.5
    image = tf.cond(flip, lambda: tf.reverse(image, axis=[1]), lambda: image)
    image = _geometric_transform(image, sample_index, seed)
    contrast = tf.random.stateless_uniform(
        [], _seed(sample_index, 3, seed), minval=0.85, maxval=1.15
    )
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * contrast + mean
    brightness = tf.random.stateless_uniform(
        [], _seed(sample_index, 4, seed), minval=-0.10, maxval=0.10
    )
    image = tf.clip_by_value(image + brightness, -1.0, 1.0)
    image = _random_erasing(image, sample_index, seed)
    return (image + 1.0) * 127.5


def build_dataset(
    images: np.ndarray | tf.Tensor,
    labels: Iterable[int] | tf.Tensor,
    *,
    training: bool,
    batch_size: int = 64,
    seed: int = REGISTERED_SEED,
) -> tf.data.Dataset:
    """Build a deterministic validation path or seeded train-only augmented path."""

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        sample_count = int(tf.shape(images)[0])
        dataset = dataset.shuffle(
            sample_count, seed=seed, reshuffle_each_iteration=True
        ).enumerate()
        dataset = dataset.map(
            lambda index, item: (augment_train_image(item[0], index, seed), item[1]),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_clean_evaluation_dataset(
    images: np.ndarray | tf.Tensor,
    labels: Iterable[int] | tf.Tensor,
    *,
    batch_size: int = 64,
) -> tf.data.Dataset:
    return build_dataset(images, labels, training=False, batch_size=batch_size)
