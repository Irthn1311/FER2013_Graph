"""Issue #67 training/data/scientific-gate preparation regressions."""

from __future__ import annotations

import builtins
import hashlib
import inspect
from pathlib import Path
import shutil

import numpy as np
import pytest
import tensorflow as tf

from research.candidates.tf_ws_hpg_v1_training import augmentation, data
from research.candidates.tf_ws_hpg_v1_training import train_validation_only as training
from research.candidates.tf_ws_hpg_v1_weak_support.model import build_ws_hpg_v1_weak_support
from research.candidates.tf_ws_hpg_v1_weak_support.support import (
    FER_PIXEL_COORDINATE_SYSTEM,
    support_from_landmarks,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = "157c8a87f84e0aa53b4432393b99a3a560c0974e"
WS = ROOT / "research/candidates/tf_ws_hpg_v1_weak_support"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepted_sources_and_exact_model_identity_are_locked():
    assert sha(WS / "model.py") == training.EXPECTED_MODEL_SHA256 == "177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3"
    assert sha(WS / "support.py") == training.EXPECTED_SUPPORT_SHA256 == "b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee"
    candidate = build_ws_hpg_v1_weak_support()
    assert training.validate_model_identity(candidate) == {
        "parameters": 707_213, "trainable_variables": 118, "keras_variables": 138
    }
    assert training.verify_accepted_source_hashes() == {
        "model.py": training.EXPECTED_MODEL_SHA256,
        "support.py": training.EXPECTED_SUPPORT_SHA256,
    }


@pytest.mark.parametrize("drifted_name", ["model.py", "support.py"])
def test_runtime_source_hash_guard_rejects_same_shape_source_drift(
    tmp_path, drifted_name
):
    model_path = tmp_path / "model.py"
    support_path = tmp_path / "support.py"
    shutil.copy2(WS / "model.py", model_path)
    shutil.copy2(WS / "support.py", support_path)
    target = model_path if drifted_name == "model.py" else support_path
    target.write_bytes(target.read_bytes() + b"\n# same-shape source drift\n")
    with pytest.raises(training.TrainingPreparationError, match="source identity drift"):
        training.verify_accepted_source_hashes(model_path, support_path)


def test_cli_source_guard_runs_before_any_fer_or_prior_io(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training,
        "verify_accepted_source_hashes",
        lambda: (_ for _ in ()).throw(
            training.TrainingPreparationError("source identity drift")
        ),
    )
    monkeypatch.setattr(
        training,
        "load_fer_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FER CSV was opened")
        ),
    )
    with pytest.raises(training.TrainingPreparationError, match="source identity drift"):
        training.main(
            [
                "--train-csv", str(tmp_path / "train.csv"),
                "--val-csv", str(tmp_path / "val.csv"),
                "--prior-root", str(tmp_path / "priors"),
                "--output-root", str(tmp_path / "output"),
            ]
        )


def test_direction_one_red_lines_and_post_pool_support_boundary_remain():
    source = (WS / "model.py").read_text(encoding="utf-8").lower()
    layer_types = {type(layer).__name__ for layer in build_ws_hpg_v1_weak_support()._flatten_layers()}
    assert not {"Conv2D", "DepthwiseConv2D", "SeparableConv2D", "MultiHeadAttention"} & layer_types
    assert all(token not in source for token in ("roialign", "semantic region", "anatomical", "eye", "mouth", "brow"))
    suffix = inspect.getsource(type(build_ws_hpg_v1_weak_support()).call).split("nodes = self.pool_1(nodes)", 1)[1]
    assert "fine_support" not in suffix and "pixel_support" not in suffix


@pytest.mark.parametrize("value", ["test.csv", "/data/test/train.csv", "/data/testing/priors", "/data/test_split/cache", "/data/test-split/cache"])
def test_test_paths_fail_lexically_before_access(monkeypatch, value):
    monkeypatch.setattr(Path, "is_file", lambda self: (_ for _ in ()).throw(AssertionError("accessed")))
    with pytest.raises(data.WSHPGDataError, match="Test"):
        data.reject_test_path(value)


def test_no_test_discovery_and_only_explicit_cache_path():
    source = inspect.getsource(data.load_support_split) + inspect.getsource(data.load_cached_support)
    assert ".glob(" not in source and ".rglob(" not in source and "os.walk" not in source
    assert data.validate_split("train") == "train" and data.validate_split("val") == "val"
    with pytest.raises(data.WSHPGDataError):
        data.validate_split("test")


class TrackingRecord(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def cache_values(landmarks=None, detected=True):
    image = np.arange(48 * 48, dtype=np.float32).reshape(48, 48)
    return image, {
        "sample_index": np.asarray(3), "label": np.asarray(2), "image_48": image,
        "detected": np.asarray(detected),
        "landmark_xy_48": np.asarray(landmarks if landmarks is not None else [[10, 12], [37, 36]], np.float32),
        "face_mask": np.zeros((48, 48)), "part_soft_masks": np.zeros((5, 48, 48)),
        "quality_score": np.asarray(0.99),
    }


def test_cache_reads_allowlist_only_and_forbidden_semantics_are_untouched():
    image, values = cache_values()
    record = TrackingRecord(values)
    cached = data.support_from_cache_record(record, expected_sample_index=3, expected_label=2, expected_clean_image=image)
    assert set(record.reads) <= data.ALLOWED_CACHE_FIELDS
    assert not set(record.reads) & data.FORBIDDEN_CACHE_FIELDS
    assert cached.support.min() < 1.0


def test_cached_detector_output_is_bit_identical_to_direct_helper():
    landmarks = np.asarray([[10, 12], [37, 36], [22, 28]], np.float32)
    image, values = cache_values(landmarks)
    cached = data.support_from_cache_record(values, expected_sample_index=3, expected_label=2, expected_clean_image=image)
    direct = support_from_landmarks(landmarks, coordinate_system=FER_PIXEL_COORDINATE_SYSTEM)
    np.testing.assert_array_equal(cached.support, direct)


@pytest.mark.parametrize("field,value", [("sample_index", 4), ("label", 1)])
def test_cache_sample_and_label_mismatch_fail_closed(field, value):
    image, values = cache_values()
    values[field] = np.asarray(value)
    with pytest.raises(data.WSHPGDataError, match="alignment"):
        data.support_from_cache_record(values, expected_sample_index=3, expected_label=2, expected_clean_image=image)


def test_cache_clean_image_mismatch_fails_closed():
    image, values = cache_values()
    with pytest.raises(data.WSHPGDataError, match="identity"):
        data.support_from_cache_record(values, expected_sample_index=3, expected_label=2, expected_clean_image=image + 1)


@pytest.mark.parametrize("landmarks,detected", [([], False), ([[20, 10], [20, 35]], True), ([[-1, 10], [20, 35]], True)])
def test_invalid_or_missing_detector_output_is_all_ones(landmarks, detected):
    image, values = cache_values(landmarks, detected)
    cached = data.support_from_cache_record(values, expected_sample_index=3, expected_label=2, expected_clean_image=image)
    np.testing.assert_array_equal(cached.support, np.ones((48, 48, 1), np.float32))
    assert not cached.detected


def test_stateless_augmentation_and_exact_geometry_coupling():
    image = tf.reshape(tf.range(48 * 48, dtype=tf.float32) % 255, [48, 48, 1])
    field = tf.reshape(tf.linspace(0.0, 1.0, 48 * 48), [48, 48, 1])
    first = augmentation.augment_example(image, field, tf.constant(7), return_debug=True)
    second = augmentation.augment_example(image, field, tf.constant(7), return_debug=True)
    np.testing.assert_array_equal(first[0]["images"], second[0]["images"])
    np.testing.assert_array_equal(first[0]["support"], second[0]["support"])
    np.testing.assert_array_equal(first[1]["image_transform"], first[1]["support_transform"])
    for key in first[2]:
        np.testing.assert_array_equal(first[2][key], second[2][key])


def _augmentation_epochs(epoch_count=2):
    sample_count = 16
    images = np.stack(
        [np.full((48, 48, 1), index, np.float32) for index in range(sample_count)]
    )
    supports = np.stack(
        [np.full((48, 48, 1), index / sample_count, np.float32) for index in range(sample_count)]
    )
    labels = np.arange(sample_count, dtype=np.int32)
    records = data._training_records(images, supports, labels)
    epochs = []
    for _ in range(epoch_count):
        observed = []
        for augmentation_index, record in records:
            original_index, image, support, label = record
            parameters = augmentation.sample_parameters(augmentation_index)
            observed.append(
                (
                    int(original_index),
                    int(label),
                    float(image[0, 0, 0]),
                    float(support[0, 0, 0]),
                    tuple(
                        np.asarray(parameters[key]).tobytes()
                        for key in sorted(parameters)
                    ),
                )
            )
        epochs.append(observed)
    return epochs


def test_complete_seed42_replay_has_identical_order_and_augmentation_sequence():
    assert _augmentation_epochs() == _augmentation_epochs()


def test_successive_epochs_reassign_augmentation_while_preserving_sample_identity():
    first, second = _augmentation_epochs()
    assert [row[0] for row in first] != [row[0] for row in second]
    first_by_sample = {row[0]: row[-1] for row in first}
    second_by_sample = {row[0]: row[-1] for row in second}
    assert any(first_by_sample[index] != second_by_sample[index] for index in first_by_sample)
    for epoch in (first, second):
        for original_index, label, image_identity, support_identity, _ in epoch:
            assert label == original_index
            assert image_identity == float(original_index)
            assert support_identity == pytest.approx(original_index / 16.0)


def test_augmentation_has_no_mediapipe_rerun_or_support_dropout():
    source = inspect.getsource(augmentation) + inspect.getsource(data.build_dataset)
    assert "mediapipe" not in source.casefold()
    assert "support_dropout" not in source


def test_photometric_and_erasing_do_not_modify_support(monkeypatch):
    image = tf.ones([48, 48, 1]) * 127.5
    field = tf.reshape(tf.linspace(0.0, 1.0, 48 * 48), [48, 48, 1])
    base = augmentation.sample_parameters(tf.constant(11))
    no_photo = dict(base, contrast=tf.constant(1.0), brightness=tf.constant(0.0), erase=tf.constant(False))
    photo_erase = dict(base, contrast=tf.constant(1.15), brightness=tf.constant(0.1), erase=tf.constant(True))
    monkeypatch.setattr(augmentation, "sample_parameters", lambda index: no_photo)
    reference, _, _ = augmentation.augment_example(image, field, tf.constant(11), return_debug=True)
    monkeypatch.setattr(augmentation, "sample_parameters", lambda index: photo_erase)
    changed, _, _ = augmentation.augment_example(image, field, tf.constant(11), return_debug=True)
    np.testing.assert_array_equal(reference["support"], changed["support"])
    assert not np.array_equal(reference["images"], changed["images"])


def test_detector_failure_all_ones_survives_geometry():
    inputs, _, _ = augmentation.augment_example(tf.zeros([48, 48, 1]), tf.ones([48, 48, 1]), tf.constant(19), return_debug=True)
    np.testing.assert_array_equal(inputs["support"], np.ones((48, 48, 1), np.float32))


def test_exact_training_configuration_optimizer_schedule_and_losses():
    assert training.TRAINING_CONFIG == {
        "seed": 42, "optimizer": "AdamW", "learning_rate": 3e-4,
        "weight_decay": 5e-4, "global_clipnorm": 1.0, "batch_size": 64,
        "max_epochs": 100, "warmup_epochs": 5, "cosine_final_learning_rate": 1e-6,
        "training_label_smoothing": 0.05, "validation_every_epochs": 1,
        "checkpoint": "earliest_strict_max_val_accuracy", "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 15, "early_stopping_min_delta": 0.0,
        "mixed_precision": False, "xla": False, "mirrored_strategy": False,
        "support_dropout": False,
    }
    schedule = training.WarmupCosine(10)
    assert float(schedule(0)) == 0.0
    assert float(schedule(50)) == pytest.approx(3e-4)
    assert float(schedule(1000)) == pytest.approx(1e-6)
    optimizer = training.build_optimizer(10)
    assert isinstance(optimizer, tf.keras.optimizers.AdamW)
    assert float(optimizer.weight_decay) == pytest.approx(5e-4)
    assert float(optimizer.global_clipnorm) == pytest.approx(1.0)
    labels, logits = tf.constant([0]), tf.zeros([1, 7])
    assert np.isfinite(training.training_loss(labels, logits)).all()
    assert np.isfinite(training.hard_evaluation_loss(labels, logits)).all()


def test_earliest_strict_maximum_policy():
    assert training.earliest_strict_max_epoch([0.5, 0.6, 0.6, 0.59]) == 1
    assert training.earliest_strict_max_epoch([0.5, 0.5, 0.7, 0.7]) == 2
    with pytest.raises(training.TrainingPreparationError):
        training.earliest_strict_max_epoch([])


@pytest.mark.parametrize(("metrics", "expected"), [
    (dict(validation_accuracy=.6500, validation_macro_f1=.6200, clean_train_macro_f1=.7200, support_dependency_accuracy_pp=3.0, support_dependency_macro_pp=3.0), "WS_HPG_V1_STRETCH_REPLACEMENT_CANDIDATE"),
    (dict(validation_accuracy=.6369308999721371, validation_macro_f1=.6038407974340496, clean_train_macro_f1=.7538407974340496, support_dependency_accuracy_pp=5.0, support_dependency_macro_pp=5.0), "WS_HPG_V1_REPLACEMENT_CANDIDATE"),
    (dict(validation_accuracy=.6269308999721371, validation_macro_f1=.5888407974340496, clean_train_macro_f1=.7407655611897143, support_dependency_accuracy_pp=9.0, support_dependency_macro_pp=9.0), "WS_HPG_V1_COMPETITIVE_GENERALIZATION_SIGNAL_NOT_REPLACE"),
    (dict(validation_accuracy=.59, validation_macro_f1=.54, clean_train_macro_f1=.70, support_dependency_accuracy_pp=9.0, support_dependency_macro_pp=9.0), "WS_HPG_V1_BEATS_CF_ONLY_NOT_REPLACE"),
    (dict(validation_accuracy=.5806631373641683, validation_macro_f1=.54, clean_train_macro_f1=.70, support_dependency_accuracy_pp=0.0, support_dependency_macro_pp=0.0), "WS_HPG_V1_UNDERFIT_OR_REGRESSION"),
    (dict(validation_accuracy=float("nan"), validation_macro_f1=.54, clean_train_macro_f1=.70, support_dependency_accuracy_pp=0.0, support_dependency_macro_pp=0.0), "WS_HPG_V1_INCONCLUSIVE"),
])
def test_exact_classifier_boundaries_and_precedence(metrics, expected):
    assert training.classify_outcome(**metrics) == expected


def test_capacity_limited_applies_only_after_higher_rules():
    stretch = dict(validation_accuracy=.66, validation_macro_f1=.63, clean_train_macro_f1=.70, support_dependency_accuracy_pp=1.0, support_dependency_macro_pp=1.0)
    assert training.classify_outcome(**stretch, capacity_limited=True) == "WS_HPG_V1_STRETCH_REPLACEMENT_CANDIDATE"
    weak = dict(validation_accuracy=.59, validation_macro_f1=.54, clean_train_macro_f1=.70, support_dependency_accuracy_pp=9.0, support_dependency_macro_pp=9.0)
    assert training.classify_outcome(**weak, capacity_limited=True) == "WS_HPG_V1_BEATS_CF_ONLY_NOT_REPLACE"


def test_lifecycle_reload_and_evaluation_inventory(monkeypatch, tmp_path):
    calls = []
    class FakeModel:
        variables = []
        def compile(self, **kwargs): calls.append(("compile", kwargs))
        def fit(self, train, **kwargs):
            calls.append(("fit", train, kwargs))
            checkpoint = kwargs["callbacks"][0]
            checkpoint.selected_epoch = 2
            return type("History", (), {"history": {"loss": [1.0, .9, .8]}})()
    original, selected = FakeModel(), FakeModel()
    monkeypatch.setattr(training, "build_ws_hpg_v1_weak_support", lambda: original)
    monkeypatch.setattr(training, "validate_model_identity", lambda candidate: training.EXPECTED_IDENTITY)
    monkeypatch.setattr(training, "build_optimizer", lambda steps: "optimizer")
    def load(path, *, compile):
        calls.append(("load", path, compile))
        assert compile is False
        return selected
    monkeypatch.setattr(tf.keras.models, "load_model", load)
    metrics = {
        "clean": {"sample_count": 2, "accuracy": .8, "macro_f1": .7, "loss": .5},
        "normal": {"sample_count": 2, "accuracy": .64, "macro_f1": .61, "loss": .7},
        "ones": {"sample_count": 2, "accuracy": .63, "macro_f1": .60, "loss": .8},
    }
    def evaluate(candidate, dataset, *, support_override):
        calls.append(("evaluate", candidate, dataset, support_override))
        return metrics[dataset if dataset == "clean" else support_override]
    monkeypatch.setattr(training, "evaluate", evaluate)
    result = training.run_registered_lifecycle("train", "validation", "clean", tmp_path / "run", {"validation": {"coverage_mean": .8}})
    evaluations = [call for call in calls if call[0] == "evaluate"]
    assert [(call[2], call[3]) for call in evaluations] == [("clean", "normal"), ("validation", "normal"), ("validation", "ones")]
    assert all(call[1] is selected for call in evaluations)
    assert result["support_diagnostics"]["validation"]["coverage_mean"] == .8
    assert result["test_access"] is False


def test_tf_function_gradients_optimizer_and_keras_round_trip(tmp_path):
    candidate = build_ws_hpg_v1_weak_support()
    inputs = {"images": tf.zeros([1, 48, 48, 1]), "support": tf.ones([1, 48, 48, 1])}
    @tf.function
    def gradients():
        with tf.GradientTape() as tape:
            logits = candidate(inputs, training=True)
            loss = tf.reduce_mean(training.training_loss(tf.constant([1]), logits))
        return loss, tape.gradient(loss, candidate.trainable_variables)
    loss, grads = gradients()
    assert np.isfinite(loss) and all(g is not None and np.isfinite(g).all() for g in grads)
    before = [v.numpy().copy() for v in candidate.trainable_variables]
    tf.keras.optimizers.SGD(1e-4).apply_gradients(zip(grads, candidate.trainable_variables))
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(before, candidate.trainable_variables))
    path = tmp_path / "ws.keras"
    expected = candidate(inputs, training=False)
    candidate.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_array_equal(restored(inputs, training=False), expected)


def test_benchmark_contract_has_all_representative_batches_and_no_training_authorization():
    benchmark = (WS / "synthetic_benchmark.py").read_text(encoding="utf-8")
    assert "batch_sizes=(8, 16, 32, 64)" in benchmark
    package_source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "research/candidates/tf_ws_hpg_v1_training").glob("*.py"))
    assert "test.csv" not in package_source and "import torch" not in package_source
    assert training.STATUS == "WS_HPG_V1_TRAINING_PREPARATION_ONLY"
