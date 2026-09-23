"""FER2013 loading with split isolation and resume-stable augmentation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms.functional as TF


EXPECTED_TRAIN_ROWS = 28709
EXPECTED_PUBLIC_ROWS = 3589
EXPECTED_PRIVATE_ROWS = 3589
ROLE_TO_BASENAME = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}
ROLE_TO_EXPECTED_ROWS = {
    "train": EXPECTED_TRAIN_ROWS,
    "val": EXPECTED_PUBLIC_ROWS,
    "test": EXPECTED_PRIVATE_ROWS,
}


def augmentation_seed(base_seed: int, epoch: int, sample_index: int) -> int:
    """Stable sample-local seed, independent of worker process RNG history."""
    return int(base_seed) + int(epoch) * 1_000_003 + int(sample_index) * 97_409


class EpochRandomSampler(Sampler[int]):
    """Derive each epoch's permutation solely from base seed and epoch."""

    def __init__(self, data_source: Dataset, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 2_000_033
        )
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict:
        return {"scheme": "base_seed_epoch_v1", "seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        if state.get("scheme") != "base_seed_epoch_v1" or int(state["seed"]) != self.seed:
            raise RuntimeError("Training sampler state is incompatible")
        self.epoch = int(state["epoch"])


def validate_split_path(csv_path: str | Path, role: str) -> Path:
    """Validate a split by explicit role and basename, never by row count alone."""
    normalized_role = role.lower()
    if normalized_role not in ROLE_TO_BASENAME:
        raise ValueError(f"Unknown FER2013 split role: {role!r}")
    path = Path(csv_path)
    expected = ROLE_TO_BASENAME[normalized_role]
    if path.name.lower() != expected:
        raise ValueError(
            f"Split role {normalized_role!r} requires basename {expected!r}; "
            f"refusing suspicious path {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"FER2013 {normalized_role} CSV not found: {path}")
    return path


def validate_split_paths(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Fail closed on role swaps, aliases, and duplicate split paths."""
    train_path = validate_split_path(train_csv, "train")
    val_path = validate_split_path(val_csv, "val")
    test_path = validate_split_path(test_csv, "test") if test_csv is not None else None
    resolved = [p.resolve() for p in (train_path, val_path, test_path) if p is not None]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Train, PublicTest, and PrivateTest paths must be distinct")
    return train_path, val_path, test_path


def inspect_split_file(
    csv_path: str | Path,
    role: str,
    validate_content: bool,
) -> dict:
    """Validate split metadata and, when authorized, every labeled image row."""
    path = validate_split_path(csv_path, role)
    normalized_role = role.lower()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [column.strip().lower() for column in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"Empty FER2013 CSV: {path}") from exc
        if "pixels" not in header or "emotion" not in header:
            raise ValueError(f"CSV requires emotion and pixels columns: {path}")
        pixel_index = header.index("pixels")
        emotion_index = header.index("emotion")
        for row in reader:
            if not row:
                continue
            rows += 1
            if validate_content:
                pixels = np.fromstring(row[pixel_index], sep=" ", dtype=np.int16)
                if len(pixels) != 2304:
                    raise ValueError(
                        f"Malformed pixel row with {len(pixels)} pixels in {path}"
                    )
                if np.any((pixels < 0) | (pixels > 255)):
                    raise ValueError(f"Pixel outside [0, 255] in {path}")
                label = int(row[emotion_index])
                if not 0 <= label <= 6:
                    raise ValueError(f"Invalid label {label} in {path}")
    expected_rows = ROLE_TO_EXPECTED_ROWS[normalized_role]
    if rows != expected_rows:
        raise ValueError(
            f"{normalized_role} requires exactly {expected_rows} rows, got {rows}"
        )
    return {
        "role": normalized_role,
        "path": str(path.resolve()),
        "rows": rows,
        "expected_rows": expected_rows,
        "content_validated": validate_content,
    }


def validate_dataset_gate(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> dict[str, dict]:
    """Phase-A gate: inspect Train/Public content, Private metadata only."""
    train_path, val_path, test_path = validate_split_paths(
        train_csv, val_csv, test_csv
    )
    assert test_path is not None
    return {
        "train": inspect_split_file(train_path, "train", validate_content=True),
        "public": inspect_split_file(val_path, "val", validate_content=True),
        # PrivateTest content stays locked until the frozen-checkpoint boundary.
        "private": inspect_split_file(test_path, "test", validate_content=False),
    }


class FER2013Dataset(Dataset):
    """FER2013 CSV Dataset returning normalized [1, 48, 48] images and labels."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        aliases = {"public": "val", "private": "test"}
        self.split = aliases.get(split.lower(), split.lower())
        if self.split not in ROLE_TO_BASENAME:
            raise ValueError(f"Unknown FER2013 split role: {split!r}")
        self.augment = augment and (self.split == "train")
        self.seed = int(seed)
        self.epoch = 0

        p = validate_split_path(csv_path, self.split)

        images_list = []
        labels_list = []

        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = [c.strip().lower() for c in next(reader)]
            if "pixels" not in header:
                raise ValueError(f"CSV missing 'pixels' column: {p}")
            pix_idx = header.index("pixels")
            if "emotion" not in header:
                raise ValueError(f"CSV missing 'emotion' column: {p}")
            emo_idx = header.index("emotion")

            for row in reader:
                if not row:
                    continue
                pix = np.fromstring(row[pix_idx], sep=" ", dtype=np.int16)
                if len(pix) != 2304:
                    raise ValueError(f"Malformed pixel row with {len(pix)} pixels in {p}")
                if np.any((pix < 0) | (pix > 255)):
                    raise ValueError(f"Pixel outside [0, 255] in {p}")
                images_list.append(pix.astype(np.uint8).reshape(48, 48))

                label = int(row[emo_idx])
                if not 0 <= label <= 6:
                    raise ValueError(f"Invalid label {label} in {p}")
                labels_list.append(label)

        if not images_list:
            raise ValueError(f"FER2013 CSV contains no data rows: {p}")
        self.images = np.stack(images_list, axis=0)  # [N, 48, 48] uint8
        self.labels = np.array(labels_list, dtype=np.int64)

        if self.split == "train" and len(self.images) != EXPECTED_TRAIN_ROWS:
            warnings_msg = f"Expected {EXPECTED_TRAIN_ROWS} rows for train, got {len(self.images)}"
            print(f"[Warning] {warnings_msg}")
        elif self.split in ("val", "public") and len(self.images) != EXPECTED_PUBLIC_ROWS:
            print(f"[Warning] Expected {EXPECTED_PUBLIC_ROWS} rows for val, got {len(self.images)}")
        elif self.split in ("test", "private") and len(self.images) != EXPECTED_PRIVATE_ROWS:
            print(f"[Warning] Expected {EXPECTED_PRIVATE_ROWS} rows for test, got {len(self.images)}")

    def __len__(self) -> int:
        return len(self.images)

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic, sample-local augmentation for an epoch."""
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        return {
            "scheme": "sample_local_seed_v1",
            "seed": self.seed,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("scheme") != "sample_local_seed_v1" or int(state["seed"]) != self.seed:
            raise RuntimeError("Training augmentation state is incompatible")
        self.epoch = int(state["epoch"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_np = self.images[idx]
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0  # [1, 48, 48] in [0, 1]
        label = int(self.labels[idx])

        if self.augment:
            generator = torch.Generator().manual_seed(
                augmentation_seed(self.seed, self.epoch, int(idx))
            )
            # Conservative FER-safe augmentations:
            # 1. Random horizontal flip (p = 0.5)
            if torch.rand(1, generator=generator).item() > 0.5:
                img_tensor = TF.hflip(img_tensor)

            # 2. Random translation (crop/shift by up to 2 pixels)
            if torch.rand(1, generator=generator).item() > 0.5:
                shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
                shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
                img_tensor = TF.affine(
                    img_tensor,
                    angle=0.0,
                    translate=[shift_x, shift_y],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    fill=0.0,
                )

            # 3. Very mild intensity scaling: [0.95, 1.05]
            if torch.rand(1, generator=generator).item() > 0.5:
                scale = 0.95 + 0.10 * torch.rand(1, generator=generator).item()
                img_tensor = torch.clamp(img_tensor * scale, 0.0, 1.0)

        return img_tensor, label


def create_training_dataloaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    generator: torch.Generator | None = None,
) -> dict[str, DataLoader]:
    """Create only Train and PublicTest loaders; PrivateTest stays unopened."""
    train_path, val_path, _ = validate_split_paths(train_csv, val_csv)
    train_ds = FER2013Dataset(train_path, split="train", augment=True, seed=seed)
    val_ds = FER2013Dataset(val_path, split="val", augment=False)
    if generator is None:
        generator = torch.Generator().manual_seed(seed)
    train_sampler = EpochRandomSampler(train_ds, seed=seed)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=generator,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    return loaders


def create_private_dataloader(
    test_csv: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
) -> DataLoader:
    """Construct the PrivateTest loader only after checkpoint freeze."""
    test_path = validate_split_path(test_csv, "test")
    test_ds = FER2013Dataset(test_path, split="test", augment=False)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
