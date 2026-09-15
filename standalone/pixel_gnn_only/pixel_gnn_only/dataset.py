"""Read raw FER2013 CSVs directly; never import or load a prior."""

import csv
import hashlib
from pathlib import Path

import numpy as np

from lap_gnn_tf.constants import SPLIT_COUNTS
from pixel_gnn_only.graph import build_pixel_graph


class PixelDataset:
    def __init__(self, fer_csv, split):
        if split not in SPLIT_COUNTS:
            raise ValueError(f"Unknown split: {split}")
        source = Path(fer_csv)
        if source.is_dir():
            source = source / "train.csv"
        with source.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream))
        has_usage = "usage" in [x.lower().strip() for x in header]
        if not has_usage:
            names = {"train": ["train.csv"], "val": ["val.csv", "validation.csv"], "test": ["test.csv"]}[split]
            candidates = [source.parent / name for name in names if (source.parent / name).is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Need exactly one existing {split} CSV in {source.parent}; got {candidates}")
            source = candidates[0]
        self.images, self.labels = [], []
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = {x.lower().strip(): x for x in reader.fieldnames or []}
            if not {"emotion", "pixels"}.issubset(columns):
                raise ValueError("FER CSV requires emotion and pixels columns")
            expected_usage = {"train": "training", "val": "publictest", "test": "privatetest"}[split]
            for row in reader:
                if has_usage and row[columns["usage"]].strip().lower() != expected_usage:
                    continue
                values = np.fromstring(row[columns["pixels"]], dtype=np.float32, sep=" ")
                label = int(row[columns["emotion"]])
                if values.size != 2304 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
                    raise ValueError(f"Invalid pixel row {len(self.images)} in {source}")
                if not np.equal(values, np.round(values)).all():
                    raise ValueError("FER CSV pixels must be integer intensities; refusing silent quantization")
                if not 0 <= label < 7:
                    raise ValueError(f"Invalid FER label: {label}")
                self.images.append(values.astype(np.uint8).reshape(48, 48))
                self.labels.append(label)
        if len(self.images) != SPLIT_COUNTS[split]:
            raise ValueError(f"Frozen {split} split requires {SPLIT_COUNTS[split]} rows; found {len(self.images)}")
        self.split = split
        self.epoch = 0
        self.provenance = {"split": split, "path": str(source.resolve()), "rows": len(self.images),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "selection": expected_usage if has_usage else "pre-split CSV; unchanged row order"}

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def current_corruption_probability(self):
        # The reference corruption modifies priors only, never image pixels.
        # No prior exists in this ablation, so no corruption operation is run.
        return 0.0

    def __getitem__(self, index):
        return build_pixel_graph(self.images[index].astype(np.float32) / 255.0,
                                 self.labels[index], index)
