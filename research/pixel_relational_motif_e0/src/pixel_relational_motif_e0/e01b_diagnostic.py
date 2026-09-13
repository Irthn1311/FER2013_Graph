from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from .descriptor import extract_raw_relations, VALID_LOCATIONS
from .e01_occurrence_runner import load_dictionary_artifact
from .e01_runner import load_official_train_images

PUBLIC_ROWS = 3_589
E01_V533_SCIENTIFIC_SHA = "5fda000413c4dcfe810917f07e37bcafb1c8d394"
E01_V533_DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
E01_V533_SELECTED_K = 128
E01B_NULL_REPLICATES = 2_000
E01B_SEED = 42


@dataclass(frozen=True)
class ImageOnlyData:
    images_uint8: np.ndarray
    sha256: str


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _reject_private_path(path: Path) -> None:
    text = str(path).lower().replace("-", "_").replace(" ", "_")
    compact = text.replace("_", "")
    if (
        "privatetest" in compact
        or "private_test" in text
        or "final_test" in text
        or path.name.lower() == "test.csv"
    ):
        raise ValueError("E0.1b must not read FER2013 PrivateTest/final-test data")


def load_public_images_label_blind(csv_path: str | Path, *, expected_rows: int = PUBLIC_ROWS) -> ImageOnlyData:
    """Load PublicTest/validation pixels while deliberately ignoring emotion labels."""
    path = Path(csv_path)
    _reject_private_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    images = np.empty((expected_rows, 48 * 48), dtype=np.uint8)
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("FER CSV is empty") from exc
        header = [c.strip().lower() for c in raw_header]
        if "pixels" not in header:
            raise ValueError("FER CSV requires pixels column")
        pix_idx = header.index("pixels")
        for row in reader:
            if not row:
                continue
            if row_count >= expected_rows:
                raise ValueError(f"row count exceeds expected {expected_rows}")
            pixels = np.fromstring(row[pix_idx], sep=" ", dtype=np.int16)
            if pixels.size != 48 * 48:
                raise ValueError(f"expected 2304 pixels at row {row_count}, got {pixels.size}")
            if np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(f"pixel outside [0,255] at row {row_count}")
            images[row_count] = pixels.astype(np.uint8, copy=False)
            row_count += 1
    if row_count != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, observed {row_count}")
    return ImageOnlyData(images.reshape(expected_rows, 48, 48), digest)


def locked_candidate_components(dictionary_npz: str | Path) -> np.ndarray:
    """Map the original BH-stable medoid motifs into canonical component IDs.

    E0.1b is diagnostic only: it intentionally does not apply the registered
    non-degeneracy gate that produced the v533 negative result.
    """
    p = Path(dictionary_npz)
    digest = _sha256(p)
    if digest != E01_V533_DICTIONARY_SHA256:
        raise ValueError(
            f"E0.1b is locked to v533 dictionary SHA256 {E01_V533_DICTIONARY_SHA256}, got {digest}"
        )
    with np.load(p, allow_pickle=False) as z:
        stable = np.asarray(z["medoid_stable_mask"], dtype=bool).reshape(-1)
        mapping = np.asarray(z["medoid_to_canonical"], dtype=np.int64).reshape(-1)
        selected_k = int(np.asarray(z["selected_k"]).reshape(()))
    if selected_k != E01_V533_SELECTED_K:
        raise ValueError(f"v533 dictionary must have K={E01_V533_SELECTED_K}, got {selected_k}")
    if stable.shape != (selected_k,) or mapping.shape != (selected_k,):
        raise ValueError("v533 stability/mapping shape mismatch")
    if sorted(mapping.tolist()) != list(range(selected_k)):
        raise ValueError("medoid_to_canonical must be a permutation")
    return np.asarray(sorted(mapping[np.flatnonzero(stable)].tolist()), dtype=np.int64)


def count_canonical_assignments(
    images_uint8: np.ndarray,
    image_ids: np.ndarray,
    dictionary_npz: str | Path,
    *,
    log_every: int = 500,
) -> np.ndarray:
    artifact = load_dictionary_artifact(dictionary_npz)
    ids = np.asarray(image_ids, dtype=np.int64).reshape(-1)
    counts = np.zeros((artifact.selected_k, len(ids)), dtype=np.int32)
    for pos, image_id in enumerate(ids):
        image = np.asarray(images_uint8[int(image_id)], dtype=np.float64) / 255.0
        s, l, _ = extract_raw_relations(image)
        r = artifact.transform.transform(s, l)
        labels = artifact.model.predict(r)
        counts[:, pos] = np.bincount(labels, minlength=artifact.selected_k).astype(np.int32)
        if log_every and ((pos + 1) % log_every == 0 or pos + 1 == len(ids)):
            print(f"[PGM-E0.1b] assignments {pos + 1}/{len(ids)} images", flush=True)
    if not np.all(counts.sum(axis=0) == VALID_LOCATIONS):
        raise AssertionError("each image must contribute exactly 1936 hard assignments")
    return counts


def concentration_metrics(counts_by_component_image: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(counts_by_component_image, dtype=np.float64)
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("counts must have shape (K,n_images) and be non-negative")
    k, n_images = counts.shape
    total = counts.sum(axis=1)
    support = (counts > 0).sum(axis=1).astype(np.int64)
    support_rate = support / float(n_images)
    neff = np.zeros(k, dtype=np.float64)
    neff_over_support = np.zeros(k, dtype=np.float64)
    max_share = np.zeros(k, dtype=np.float64)
    top1_share = np.zeros(k, dtype=np.float64)
    top5_share = np.zeros(k, dtype=np.float64)
    prevalence = total / float(n_images * VALID_LOCATIONS)
    for comp in range(k):
        t = total[comp]
        if t <= 0:
            continue
        positive = counts[comp, counts[comp] > 0]
        q = positive / t
        neff[comp] = 1.0 / np.square(q).sum()
        neff_over_support[comp] = neff[comp] / len(positive)
        ordered = np.sort(positive)[::-1]
        max_share[comp] = ordered[0] / t
        n1 = max(1, int(math.ceil(0.01 * len(positive))))
        n5 = max(1, int(math.ceil(0.05 * len(positive))))
        top1_share[comp] = ordered[:n1].sum() / t
        top5_share[comp] = ordered[:n5].sum() / t
    return {
        "total_occurrences": total,
        "support_images": support,
        "support_rate": support_rate,
        "prevalence": prevalence,
        "neff": neff,
        "neff_over_support": neff_over_support,
        "max_image_share": max_share,
        "top1pct_support_share": top1_share,
        "top5pct_support_share": top5_share,
    }


def support_conditioned_neff_null(
    *,
    total_occurrences: int,
    support_images: int,
    n_replicates: int = E01B_NULL_REPLICATES,
    seed: int = E01B_SEED,
    batch_size: int = 128,
) -> np.ndarray:
    """Null for concentration conditional on observed support size.

    Each of the S observed supporting images receives one occurrence first;
    the remaining T-S occurrences are allocated uniformly over those S images.
    This preserves total occurrence count and positive support cardinality by
    construction, unlike the registered v533 null that spread mass over every
    anchor image.
    """
    t = int(total_occurrences)
    s = int(support_images)
    if t < 0 or s < 0 or s > t:
        raise ValueError("need 0 <= support_images <= total_occurrences")
    if n_replicates <= 0 or batch_size <= 0:
        raise ValueError("n_replicates and batch_size must be positive")
    if s == 0:
        return np.zeros(n_replicates, dtype=np.float64)
    if s == 1:
        return np.ones(n_replicates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    p = np.full(s, 1.0 / s, dtype=np.float64)
    out = np.empty(n_replicates, dtype=np.float64)
    remaining = t - s
    pos = 0
    while pos < n_replicates:
        b = min(batch_size, n_replicates - pos)
        draws = rng.multinomial(remaining, p, size=b).astype(np.float64)
        draws += 1.0
        q2 = np.square(draws).sum(axis=1) / float(t * t)
        out[pos : pos + b] = 1.0 / q2
        pos += b
    return out


def conditional_nondominance(
    counts_by_component_image: np.ndarray,
    *,
    n_replicates: int = E01B_NULL_REPLICATES,
    seed: int = E01B_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metrics = concentration_metrics(counts_by_component_image)
    total = metrics["total_occurrences"].astype(np.int64)
    support = metrics["support_images"].astype(np.int64)
    observed = metrics["neff"]
    p05 = np.empty(len(total), dtype=np.float64)
    passed = np.zeros(len(total), dtype=bool)
    lower_p = np.empty(len(total), dtype=np.float64)
    for comp in range(len(total)):
        cseed = int(np.random.SeedSequence([seed, 701, comp]).generate_state(1)[0])
        null = support_conditioned_neff_null(
            total_occurrences=int(total[comp]),
            support_images=int(support[comp]),
            n_replicates=n_replicates,
            seed=cseed,
        )
        p05[comp] = float(np.percentile(null, 5.0))
        passed[comp] = bool(observed[comp] > p05[comp])
        lower_p[comp] = float((1 + np.sum(null <= observed[comp])) / (len(null) + 1))
    return passed, p05, lower_p


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return a JSON-safe Spearman statistic, or None when undefined."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    statistic = float(spearmanr(a, b).statistic)
    return statistic if np.isfinite(statistic) else None


def _metric_rows(components: np.ndarray, metrics: dict[str, np.ndarray], p05: np.ndarray, passed: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for comp in np.asarray(components, dtype=np.int64):
        rows.append(
            {
                "component": int(comp),
                "total_occurrences": int(metrics["total_occurrences"][comp]),
                "support_images": int(metrics["support_images"][comp]),
                "support_rate": float(metrics["support_rate"][comp]),
                "prevalence": float(metrics["prevalence"][comp]),
                "neff": float(metrics["neff"][comp]),
                "neff_over_support": float(metrics["neff_over_support"][comp]),
                "conditional_null_neff_p05": float(p05[comp]),
                "conditional_nondominance_pass": bool(passed[comp]),
                "max_image_share": float(metrics["max_image_share"][comp]),
                "top1pct_support_share": float(metrics["top1pct_support_share"][comp]),
                "top5pct_support_share": float(metrics["top5pct_support_share"][comp]),
            }
        )
    return rows


def run_e01b(
    train_csv: str | Path,
    public_csv: str | Path,
    dictionary_npz: str | Path,
    output_dir: str | Path,
    *,
    n_null: int = E01B_NULL_REPLICATES,
    seed: int = E01B_SEED,
) -> dict:
    """Post-hoc E0.1b diagnostic using the exact v533 dictionary.

    This does not overwrite the registered E0.1 negative verdict. PublicTest
    emotion labels are ignored; only pixels are used to assess out-of-sample
    recurrence/concentration of the already-discovered canonical patterns.
    """
    if seed != E01B_SEED or n_null != E01B_NULL_REPLICATES:
        raise ValueError("E0.1b runner is locked to seed=42 and 2000 conditional-null replicates")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dictionary_path = Path(dictionary_npz)
    components = locked_candidate_components(dictionary_path)
    artifact = load_dictionary_artifact(dictionary_path)

    with np.load(dictionary_path, allow_pickle=False) as z:
        anchor_ids = np.asarray(z["anchor_ids"], dtype=np.int64).reshape(-1)
        medoid_jaccard = np.asarray(z["medoid_jaccard"], dtype=np.float64).reshape(-1)
        medoid_q = np.asarray(z["medoid_q_values"], dtype=np.float64).reshape(-1)
        mapping = np.asarray(z["medoid_to_canonical"], dtype=np.int64).reshape(-1)

    train = load_official_train_images(train_csv)
    if train.sha256 != artifact.train_sha256:
        raise ValueError("Train SHA256 does not match v533 dictionary provenance")
    public = load_public_images_label_blind(public_csv, expected_rows=PUBLIC_ROWS)

    print("[PGM-E0.1b] counting canonical hard assignments on Train_anchor", flush=True)
    anchor_counts = count_canonical_assignments(train.images_uint8, anchor_ids, dictionary_path)
    print("[PGM-E0.1b] counting canonical hard assignments on PublicTest pixels (labels ignored)", flush=True)
    public_counts = count_canonical_assignments(public.images_uint8, np.arange(PUBLIC_ROWS), dictionary_path)

    anchor_metrics = concentration_metrics(anchor_counts)
    public_metrics = concentration_metrics(public_counts)
    anchor_pass, anchor_p05, _ = conditional_nondominance(anchor_counts, n_replicates=n_null, seed=seed)
    public_pass, public_p05, _ = conditional_nondominance(public_counts, n_replicates=n_null, seed=seed + 1)

    ca_to_medoid = np.empty(artifact.selected_k, dtype=np.int64)
    ca_to_medoid[mapping] = np.arange(artifact.selected_k, dtype=np.int64)
    candidate_jaccard = medoid_jaccard[ca_to_medoid[components]]
    candidate_q = medoid_q[ca_to_medoid[components]]

    prev_train = anchor_metrics["prevalence"][components]
    prev_public = public_metrics["prevalence"][components]
    support_train = anchor_metrics["support_rate"][components]
    support_public = public_metrics["support_rate"][components]
    spearman_prev = finite_spearman(prev_train, prev_public)
    spearman_support = finite_spearman(support_train, support_public)

    stability_effect = {
        "candidate_count": int(len(components)),
        "jaccard_min": float(np.min(candidate_jaccard)) if len(candidate_jaccard) else None,
        "jaccard_median": float(np.median(candidate_jaccard)) if len(candidate_jaccard) else None,
        "jaccard_max": float(np.max(candidate_jaccard)) if len(candidate_jaccard) else None,
        "count_ge_0_1": int(np.sum(candidate_jaccard >= 0.1)),
        "count_ge_0_2": int(np.sum(candidate_jaccard >= 0.2)),
        "count_ge_0_3": int(np.sum(candidate_jaccard >= 0.3)),
        "count_ge_0_5": int(np.sum(candidate_jaccard >= 0.5)),
        "all_original_bh_q_lt_0_05": bool(np.all(candidate_q < 0.05)),
    }

    summary = {
        "experiment": "PGM_E0.1b_recurrence_nondominance_diagnostic",
        "status": "POST_HOC_DIAGNOSTIC_ONLY",
        "registered_e01_v533_verdict": "E0.1 NEGATIVE — MOTIF EXISTENCE NOT SUPPORTED",
        "registered_verdict_overwritten": False,
        "e01_v533_scientific_sha": E01_V533_SCIENTIFIC_SHA,
        "e01_v533_dictionary_sha256": E01_V533_DICTIONARY_SHA256,
        "selected_k": artifact.selected_k,
        "candidate_components": components.tolist(),
        "candidate_component_count": int(len(components)),
        "public_test_read": True,
        "public_test_labels_used": False,
        "private_test_read": False,
        "train_anchor_images": int(len(anchor_ids)),
        "public_images": PUBLIC_ROWS,
        "train_sha256": train.sha256,
        "public_sha256": public.sha256,
        "stability_effect_size": stability_effect,
        "anchor_conditional_nondominance_pass_count": int(anchor_pass[components].sum()),
        "public_conditional_nondominance_pass_count": int(public_pass[components].sum()),
        "cross_set": {
            "spearman_prevalence": spearman_prev,
            "spearman_support_rate": spearman_support,
            "median_support_rate_train_anchor": float(np.median(support_train)),
            "median_support_rate_public": float(np.median(support_public)),
            "median_prevalence_train_anchor": float(np.median(prev_train)),
            "median_prevalence_public": float(np.median(prev_public)),
        },
        "train_anchor_components": _metric_rows(components, anchor_metrics, anchor_p05, anchor_pass),
        "public_components": _metric_rows(components, public_metrics, public_p05, public_pass),
        "interpretation_boundary": (
            "E0.1b diagnoses whether the registered non-degeneracy null was measuring ubiquity rather than few-image domination. "
            "It is post-hoc and cannot convert the registered v533 E0.1 negative into a confirmatory positive."
        ),
    }

    np.savez_compressed(
        out / "e01b_counts.npz",
        candidate_components=components.astype(np.int32),
        anchor_ids=anchor_ids.astype(np.int32),
        anchor_counts=anchor_counts,
        public_counts=public_counts,
        anchor_conditional_p05=anchor_p05,
        public_conditional_p05=public_p05,
        anchor_conditional_pass=anchor_pass,
        public_conditional_pass=public_pass,
    )
    (out / "e01b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    spearman_text = "undefined" if spearman_prev is None else f"{spearman_prev:.4f}"
    print(
        f"[PGM-E0.1b] candidates={len(components)} anchor_conditional_pass={int(anchor_pass[components].sum())} "
        f"public_conditional_pass={int(public_pass[components].sum())} spearman_prevalence={spearman_text}",
        flush=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PGM E0.1b post-hoc recurrence/non-dominance diagnostic")
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--public-csv", required=True, type=Path)
    p.add_argument("--dictionary-npz", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_e01b(args.train_csv, args.public_csv, args.dictionary_npz, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
