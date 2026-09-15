"""Synthetic CSV regression checks; run on Kaggle, never real test evidence."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"))
sys.path.insert(0, str(REPO_ROOT / "standalone/pixel_gnn_only"))

from pixel_gnn_only.dataset import PixelDataset, resolve_split_source


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        counts = patch("pixel_gnn_only.dataset.SPLIT_COUNTS", {"train": 1, "val": 1, "test": 1})
        counts.start()
        self.addCleanup(counts.stop)

    def write_csv(self, name, rows, usage=True):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["emotion", "pixels"] + (["Usage"] if usage else []))
            for label, intensity, group in rows:
                writer.writerow([label, " ".join([str(intensity)] * 2304)] + ([group] if usage else []))
        return path

    def test_presplit_usage_column_does_not_redirect_validation_to_train(self):
        train = self.write_csv("train.csv", [(0, 10, "Training")])
        val = self.write_csv("val.csv", [(3, 20, "PublicTest")])
        dataset = PixelDataset(train, "val")
        self.assertEqual(dataset.labels, [3])
        self.assertEqual(int(dataset.images[0][0, 0]), 20)
        self.assertEqual(dataset.provenance["path"], str(val.resolve()))

    def test_validation_csv_alias_without_usage_keeps_row_order(self):
        train = self.write_csv("train.csv", [(0, 10, "")], usage=False)
        self.write_csv("validation.csv", [(4, 30, "")], usage=False)
        self.assertEqual(PixelDataset(train, "val").labels, [4])

    def test_original_single_csv_filters_official_usage(self):
        source = self.write_csv("fer2013.csv", [(0, 10, "Training"), (3, 20, "PublicTest"), (6, 30, "PrivateTest")])
        self.assertEqual(PixelDataset(source, "train").labels, [0])
        self.assertEqual(PixelDataset(source, "val").labels, [3])

    def test_combined_train_csv_without_siblings_still_uses_usage(self):
        source = self.write_csv("train.csv", [(0, 10, "Training"), (3, 20, "PublicTest")])
        self.assertEqual(PixelDataset(source, "val").labels, [3])

    def test_two_validation_files_fail_closed(self):
        train = self.write_csv("train.csv", [(0, 10, "Training")])
        self.write_csv("val.csv", [(3, 20, "PublicTest")])
        self.write_csv("validation.csv", [(3, 20, "PublicTest")])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_split_source(train, "val")

    def test_missing_validation_does_not_substitute_train(self):
        self.write_csv("train.csv", [(0, 10, "Training")])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            PixelDataset(self.root, "val")

    def test_wrong_count_is_still_rejected(self):
        train = self.write_csv("train.csv", [(0, 10, "Training")])
        self.write_csv("val.csv", [])
        with self.assertRaisesRegex(ValueError, "requires 1 rows; found 0"):
            PixelDataset(train, "val")


if __name__ == "__main__":
    unittest.main()
