"""Exact materialization of the accepted WS-HPG shuffle/augmentation stream."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import tensorflow as tf

from research.candidates.tf_ws_hpg_v1_training import augmentation


SEED = 42


class AcceptedOrderError(RuntimeError):
    """Fail-closed accepted-order contract error."""


def materialize_raw_shuffle_iterations(
    sample_count: int, iteration_count: int, seed: int = SEED
) -> np.ndarray:
    """Materialize consecutive iterations of the accepted TensorFlow shuffle."""

    if sample_count < 1 or iteration_count < 1 or seed != SEED:
        raise AcceptedOrderError("Accepted shuffle requires positive sizes and seed 42")
    source = tf.data.Dataset.from_tensor_slices(
        tf.range(sample_count, dtype=tf.int64)
    )
    shuffled = source.shuffle(
        sample_count, seed=seed, reshuffle_each_iteration=True
    )
    # ``repeat`` creates the same successive reshuffled traversals while one
    # iterator avoids hundreds of expensive iterator start/stop boundaries at
    # production size. Focused regression compares this result directly with
    # repeated traversal of the accepted dataset object.
    traversals = shuffled.repeat(iteration_count).batch(
        sample_count, drop_remainder=True
    )
    return np.asarray(list(traversals.as_numpy_iterator()), dtype=np.int64)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AcceptedShufflePlan:
    """The exact permutations emitted by the accepted tf.data shuffle.

    The accepted pipeline shuffles original sample identities with
    ``Dataset.shuffle(seed=42, reshuffle_each_iteration=True)`` and enumerates
    only after shuffling.  We materialize that very TensorFlow stream once and
    replay its permutations; no replacement RNG is introduced.
    """

    orders: np.ndarray
    seed: int = SEED

    @classmethod
    def materialize(cls, sample_count: int, max_epochs: int, seed: int = SEED):
        # Keras 3.15's finite tf.data EpochIterator creates a replacement
        # iterator at the end of an epoch and then another at the beginning of
        # the next epoch when steps_per_epoch is None. The accepted one-fit
        # Issue #70 lifecycle therefore consumes shuffle iterations 1,3,5,...
        # as training epochs. Preserve that observed runtime contract exactly.
        raw = materialize_raw_shuffle_iterations(
            sample_count, 2 * max_epochs, seed
        )
        plan = cls(np.asarray(raw[0::2][:max_epochs]), seed)
        plan.verify(sample_count=sample_count, max_epochs=max_epochs)
        return plan

    @classmethod
    def from_orders(cls, orders, seed: int = SEED):
        array = np.asarray(orders, dtype=np.int64)
        if array.ndim != 2:
            raise AcceptedOrderError("Persisted accepted shuffle plan must be rank two")
        plan = cls(array, seed)
        plan.verify(sample_count=array.shape[1], max_epochs=array.shape[0])
        return plan

    def verify(self, *, sample_count: int, max_epochs: int) -> None:
        if self.seed != SEED or self.orders.shape != (max_epochs, sample_count):
            raise AcceptedOrderError("Accepted shuffle plan shape/seed drift")
        expected = np.arange(sample_count, dtype=np.int64)
        for row in self.orders:
            if not np.array_equal(np.sort(row), expected):
                raise AcceptedOrderError("Accepted shuffle row is not a sample permutation")

    @property
    def sha256(self) -> str:
        return _array_sha256(np.asarray(self.orders, dtype=np.int64))

    def order(self, epoch_one_based: int) -> np.ndarray:
        epoch = int(epoch_one_based)
        if epoch < 1 or epoch > len(self.orders):
            raise AcceptedOrderError(f"Epoch outside materialized plan: {epoch}")
        return np.asarray(self.orders[epoch - 1], dtype=np.int64)


_AUGMENTATION_KEYS = (
    "flip",
    "rotation_degrees",
    "translation_x",
    "translation_y",
    "contrast",
    "brightness",
    "erase",
    "erase_area_fraction",
    "erase_aspect",
)


@tf.function(reduce_retracing=True)
def _augmentation_parameter_tensor(sample_count):
    """Graph-execute the exact accepted per-enumeration stateless function."""

    def one(index):
        values = augmentation.sample_parameters(index)
        return tf.stack(
            [tf.cast(values[key], tf.float32) for key in _AUGMENTATION_KEYS]
        )

    return tf.map_fn(
        one, tf.range(sample_count), fn_output_signature=tf.float32
    )


def augmentation_parameter_matrix(sample_count: int) -> np.ndarray:
    """Return the accepted post-shuffle enumeration parameter sequence."""

    return np.asarray(
        _augmentation_parameter_tensor(tf.constant(sample_count, tf.int32)).numpy(),
        dtype=np.float32,
    )


def next_epoch_stream(
    plan: AcceptedShufflePlan,
    completed_epoch: int,
    *,
    augmentation_parameters: np.ndarray | None = None,
) -> dict:
    next_epoch = int(completed_epoch) + 1
    order = plan.order(next_epoch)
    indices = np.arange(len(order), dtype=np.int64)
    parameters = (
        augmentation_parameter_matrix(len(order))
        if augmentation_parameters is None
        else np.asarray(augmentation_parameters, dtype=np.float32)
    )
    if parameters.shape != (len(order), len(_AUGMENTATION_KEYS)):
        raise AcceptedOrderError("Accepted augmentation parameter shape drift")
    return {
        "next_epoch": next_epoch,
        "original_sample_order": order,
        "enumeration_indices": indices,
        "augmentation_parameters": parameters,
        "order_sha256": _array_sha256(order),
        "enumeration_sha256": _array_sha256(indices),
        "augmentation_parameters_sha256": _array_sha256(parameters),
    }


def build_epoch_training_dataset(
    images,
    supports,
    labels,
    *,
    plan: AcceptedShufflePlan,
    epoch_one_based: int,
    batch_size: int = 64,
):
    """Build one epoch from an exact accepted order, then enumerate/augment."""

    order = tf.convert_to_tensor(plan.order(epoch_one_based), dtype=tf.int64)
    images = tf.convert_to_tensor(images)
    supports = tf.convert_to_tensor(supports)
    labels = tf.convert_to_tensor(labels)
    dataset = tf.data.Dataset.from_tensor_slices(order).enumerate()

    def map_record(augmentation_index, original_sample_index):
        inputs, _, _ = augmentation.augment_example(
            tf.gather(images, original_sample_index),
            tf.gather(supports, original_sample_index),
            augmentation_index,
        )
        return inputs, tf.gather(labels, original_sample_index)

    dataset = dataset.map(
        map_record, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_segment_training_dataset(
    images,
    supports,
    labels,
    *,
    plan: AcceptedShufflePlan,
    start_epoch: int,
    end_epoch: int,
    batch_size: int = 64,
):
    """Build exact epoch streams for one uninterrupted ``model.fit`` segment.

    ``steps_per_epoch`` is supplied by the caller. Keras therefore retains one
    iterator across fit epochs, consuming one materialized permutation per
    epoch without creating extra fit/train-function boundaries.
    """

    start, end = int(start_epoch), int(end_epoch)
    if start < 1 or end < start or end > len(plan.orders):
        raise AcceptedOrderError("Invalid materialized segment epoch range")
    epoch_orders = tf.convert_to_tensor(plan.orders[start - 1 : end], tf.int64)
    images = tf.convert_to_tensor(images)
    supports = tf.convert_to_tensor(supports)
    labels = tf.convert_to_tensor(labels)

    def epoch_dataset(order):
        records = tf.data.Dataset.from_tensor_slices(order).enumerate()

        def map_record(augmentation_index, original_sample_index):
            inputs, _, _ = augmentation.augment_example(
                tf.gather(images, original_sample_index),
                tf.gather(supports, original_sample_index),
                augmentation_index,
            )
            return inputs, tf.gather(labels, original_sample_index)

        return records.map(
            map_record, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        ).batch(batch_size)

    dataset = tf.data.Dataset.from_tensor_slices(epoch_orders).flat_map(epoch_dataset)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
