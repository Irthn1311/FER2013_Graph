from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mpg_fer_v2_1.data import EpochRandomSampler, FER2013Dataset


class IndexedDataset(Dataset):
    """Pickle-safe wrapper exposing sample identity for worker-order audits."""

    def __init__(self, base: FER2013Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, label, index

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)


def _write_train_csv(path: Path, samples: int = 24) -> None:
    rng = np.random.default_rng(123)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels"])
        for index in range(samples):
            pixels = rng.integers(0, 256, size=2304, dtype=np.uint8)
            pixels[0] = index
            writer.writerow([index % 7, " ".join(map(str, pixels.tolist()))])


def _collect_epochs(
    dataset: IndexedDataset,
    sampler: EpochRandomSampler,
    generator: torch.Generator,
    epochs: range,
    num_workers: int,
) -> dict[int, list[tuple[int, str]]]:
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=False,
        generator=generator,
    )
    result: dict[int, list[tuple[int, str]]] = {}
    for epoch in epochs:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        records = []
        for images, _labels, indices in loader:
            for image, index in zip(images, indices):
                records.append(
                    (
                        int(index),
                        hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                    )
                )
        result[epoch] = records
    return result


def _objects(csv_path: Path, seed: int = 42):
    base = FER2013Dataset(csv_path, split="train", augment=True, seed=seed)
    dataset = IndexedDataset(base)
    sampler = EpochRandomSampler(dataset, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return dataset, sampler, generator


def test_multiworker_sample_order_and_augmentation_match_after_resume(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    _write_train_csv(csv_path)

    continuous = _objects(csv_path)
    continuous_records = _collect_epochs(*continuous, range(1, 5), num_workers=2)

    interrupted = _objects(csv_path)
    resumed_records = _collect_epochs(*interrupted, range(1, 3), num_workers=2)
    generator_state = interrupted[2].get_state()
    sampler_state = interrupted[1].state_dict()
    augmentation_state = interrupted[0].base.state_dict()

    resumed = _objects(csv_path)
    resumed[2].set_state(generator_state)
    resumed[1].load_state_dict(sampler_state)
    resumed[0].base.load_state_dict(augmentation_state)
    resumed_records.update(_collect_epochs(*resumed, range(3, 5), num_workers=2))

    assert resumed_records == continuous_records
    for epoch, records in continuous_records.items():
        assert len(records) == 24
        assert sorted(index for index, _ in records) == list(range(24))


def test_sample_local_augmentation_ignores_global_rng_history(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    _write_train_csv(csv_path, samples=3)
    dataset = FER2013Dataset(csv_path, split="train", augment=True, seed=9)
    dataset.set_epoch(7)
    first = dataset[1][0]
    torch.rand(1000)
    np.random.default_rng(55).random(1000)
    second = dataset[1][0]
    assert torch.equal(first, second)
