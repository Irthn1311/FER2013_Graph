import csv
from pathlib import Path

import numpy as np

from pixel_relational_motif_e0.e01b_diagnostic import (
    concentration_metrics,
    load_public_images_label_blind,
    support_conditioned_neff_null,
)


def test_concentration_metrics_separate_recurrence_from_dominance():
    counts = np.array(
        [
            [10, 10, 10, 10, 0],
            [37, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    m = concentration_metrics(counts)
    assert m["support_images"].tolist() == [4, 4, 0]
    assert np.isclose(m["support_rate"][0], 0.8)
    assert np.isclose(m["neff"][0], 4.0)
    assert np.isclose(m["neff_over_support"][0], 1.0)
    assert m["neff"][1] < 2.0
    assert m["max_image_share"][1] > 0.9
    assert m["neff"][2] == 0.0


def test_support_conditioned_null_preserves_effective_support_boundaries():
    # T=S means every supporting image must have exactly one occurrence.
    null = support_conditioned_neff_null(
        total_occurrences=7,
        support_images=7,
        n_replicates=50,
        seed=3,
    )
    assert np.allclose(null, 7.0)

    # A concentrated observed allocation should sit below a null that conditions
    # on the same support cardinality rather than all images in the dataset.
    null2 = support_conditioned_neff_null(
        total_occurrences=100,
        support_images=4,
        n_replicates=500,
        seed=4,
    )
    observed = concentration_metrics(np.array([[97, 1, 1, 1]], dtype=np.int32))["neff"][0]
    assert observed < np.percentile(null2, 5)
    assert np.all((null2 >= 1.0) & (null2 <= 4.0 + 1e-12))


def test_public_loader_ignores_label_values(tmp_path: Path):
    path = tmp_path / "val.csv"
    pixels = " ".join(["0"] * (48 * 48))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "pixels"])
        # Deliberately non-FER label strings: loader must never parse/use them.
        w.writerow(["DO_NOT_USE", pixels])
        w.writerow(["ALSO_IGNORE", pixels])
    data = load_public_images_label_blind(path, expected_rows=2)
    assert data.images_uint8.shape == (2, 48, 48)
    assert data.images_uint8.dtype == np.uint8
    assert len(data.sha256) == 64
