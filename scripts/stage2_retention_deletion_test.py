"""Stage 2 retention/deletion information test for Stage 1 selectors.

This script does not mutate graph artifacts, model code, losses, or configs. It
recreates Stage 1 masks deterministically, builds masked image tensors, trains
the same lightweight classifier for each selector/ratio/mode, and writes a
report focused on whether selected pixels retain emotion information.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.labels import EMOTION_NAMES
from stage1_pixel_region_selection import (
    DEFAULT_SELECTORS,
    EDGE_FEATURES,
    NODE_FEATURES,
    center_scores,
    edge_to_node_mean,
    ensure_dir,
    feature_index,
    fmt,
    random_mask,
    resolve_graph_repo,
    safe_name,
    save_png,
    selector_masks,
    to_u8,
    try_import_slic,
)

DEFAULT_RATIOS = (0.05, 0.10, 0.20, 0.40)
MODES = ("only_selected", "delete_selected")
METRIC_FIELDS = (
    "selector",
    "retention_ratio",
    "mode",
    "fill_mode",
    "classifier",
    "num_train",
    "num_eval",
    "original_accuracy",
    "original_macro_f1",
    "original_weighted_f1",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "only_selected_accuracy",
    "only_selected_macro_f1",
    "only_selected_weighted_f1",
    "delete_selected_accuracy",
    "delete_selected_macro_f1",
    "delete_selected_weighted_f1",
    "per_class_f1_Angry",
    "per_class_f1_Disgust",
    "per_class_f1_Fear",
    "per_class_f1_Happy",
    "per_class_f1_Sad",
    "per_class_f1_Surprise",
    "per_class_f1_Neutral",
)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        fields = ["note"]
        rows = [{"note": "NO_ROWS"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def class_name(label: int) -> str:
    label = int(label)
    return EMOTION_NAMES[label] if 0 <= label < len(EMOTION_NAMES) else f"label_{label}"


def ratio_name(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100)):02d}"


def iter_limited(reader: GraphRepositoryReader, split: str, max_samples: Optional[int]) -> Iterable[Any]:
    seen = 0
    for sample in reader.iter_split(split):
        yield sample
        seen += 1
        if max_samples is not None and seen >= int(max_samples):
            break


def make_comparison_grid(original: np.ndarray, mask: np.ndarray, only_selected: np.ndarray, delete_selected: np.ndarray) -> np.ndarray:
    original_u8 = to_u8(original)
    mask_u8 = mask.astype(np.uint8) * 255
    only_u8 = to_u8(only_selected)
    delete_u8 = to_u8(delete_selected)
    parts = [original_u8, mask_u8, only_u8, delete_u8]
    sep = np.full((original_u8.shape[0], 4), 255, dtype=np.uint8)
    grid = parts[0]
    for part in parts[1:]:
        grid = np.concatenate([grid, sep, part], axis=1)
    return grid


def apply_mask(intensity: np.ndarray, mask: np.ndarray, mode: str, fill_mode: str) -> np.ndarray:
    image = np.asarray(intensity, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    fill = 0.0 if fill_mode == "zero" else float(image.mean())
    out = image.copy()
    if mode == "only_selected":
        out[~mask] = fill
    elif mode == "delete_selected":
        out[mask] = fill
    elif mode == "original":
        pass
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return out.astype(np.float32)


def save_visuals(
    output_dir: Path,
    selector: str,
    ratio: float,
    split: str,
    graph_id: int,
    label: int,
    intensity: np.ndarray,
    mask: np.ndarray,
    fill_mode: str,
) -> None:
    class_dir = f"class_{int(label)}_{safe_name(class_name(label))}"
    base = ensure_dir(output_dir / "figures" / selector / ratio_name(ratio) / class_dir)
    stem = f"{split}_graph_{int(graph_id)}"
    only_selected = apply_mask(intensity, mask, "only_selected", fill_mode).reshape(48, 48)
    delete_selected = apply_mask(intensity, mask, "delete_selected", fill_mode).reshape(48, 48)
    save_png(base / f"{stem}_comparison.png", make_comparison_grid(intensity.reshape(48, 48), mask.reshape(48, 48), only_selected, delete_selected))


def save_masked_npz(
    output_dir: Path,
    selector: str,
    ratio: float,
    mode: str,
    fill_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    graph_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    graph_eval: np.ndarray,
) -> None:
    base = ensure_dir(output_dir / "masked_data" / selector / ratio_name(ratio))
    np.savez_compressed(
        base / f"{mode}_{fill_mode}.npz",
        x_train=(np.clip(x_train, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        y_train=y_train.astype(np.int64),
        graph_id_train=graph_train.astype(np.int64),
        x_eval=(np.clip(x_eval, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        y_eval=y_eval.astype(np.int64),
        graph_id_eval=graph_eval.astype(np.int64),
    )


def load_split_records(
    reader: GraphRepositoryReader,
    resolver: GraphResolver,
    split: str,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    height = int(resolver.height)
    width = int(resolver.width)
    num_nodes = height * width
    for sample in iter_limited(reader, split, max_samples):
        resolved = resolver.resolve(sample)
        node_names = list(resolved.node_feature_names or NODE_FEATURES)
        edge_names = list(resolved.edge_feature_names or EDGE_FEATURES)
        node = resolved.node_features.detach().cpu().numpy()
        intensity = node[:, feature_index(node_names, "intensity", 0)].astype(np.float32)
        x_norm = node[:, feature_index(node_names, "x_norm", 1)].astype(np.float32)
        y_norm = node[:, feature_index(node_names, "y_norm", 2)].astype(np.float32)
        grad_mag = node[:, feature_index(node_names, "grad_mag", 5)].astype(np.float32)
        local_contrast = node[:, feature_index(node_names, "local_contrast", 6)].astype(np.float32)
        delta_idx = feature_index(edge_names, "delta_intensity", 3)
        delta_edge_node = edge_to_node_mean(resolved.edge_index, resolved.edge_attr[:, delta_idx], num_nodes)
        records.append(
            {
                "split": split,
                "graph_id": int(resolved.graph_id),
                "label": int(resolved.label),
                "intensity": intensity,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "grad_mag": grad_mag,
                "local_contrast": local_contrast,
                "delta_edge_node": delta_edge_node,
            }
        )
    return records


def build_mask(
    record: Dict[str, Any],
    selector: str,
    ratio: float,
    seed: int,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
) -> Optional[np.ndarray]:
    generated = list(
        selector_masks(
            [selector],
            ratio,
            int(record["graph_id"]),
            seed,
            48,
            48,
            record["intensity"],
            record["x_norm"],
            record["y_norm"],
            record["grad_mag"],
            record["local_contrast"],
            record["delta_edge_node"],
            slic_fn,
            slic_segments,
            slic_compactness,
        )
    )
    if not generated:
        return None
    return generated[0][1].astype(bool)


def build_dataset(
    records: Sequence[Dict[str, Any]],
    selector: str,
    ratio: float,
    mode: str,
    fill_mode: str,
    seed: int,
    slic_fn: Any,
    slic_segments: int,
    slic_compactness: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    graph_ids: List[int] = []
    for record in records:
        if mode == "original":
            masked = record["intensity"].astype(np.float32)
        else:
            mask = build_mask(record, selector, ratio, seed, slic_fn, slic_segments, slic_compactness)
            if mask is None:
                continue
            masked = apply_mask(record["intensity"], mask, mode, fill_mode)
        x_rows.append(masked)
        y_rows.append(int(record["label"]))
        graph_ids.append(int(record["graph_id"]))
    if not x_rows:
        return np.zeros((0, 2304), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.stack(x_rows, axis=0).astype(np.float32), np.asarray(y_rows, dtype=np.int64), np.asarray(graph_ids, dtype=np.int64)


def train_eval_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
        ),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_eval)
    per_class = f1_score(y_eval, pred, average=None, labels=list(range(len(EMOTION_NAMES))), zero_division=0)
    metrics: Dict[str, Any] = {
        "classifier": "StandardScaler+SGDClassifier(log_loss)",
        "num_train": int(len(y_train)),
        "num_eval": int(len(y_eval)),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_eval, pred, average="weighted", zero_division=0)),
    }
    for idx, name in enumerate(EMOTION_NAMES):
        metrics[f"per_class_f1_{name}"] = float(per_class[idx])
    cm = confusion_matrix(y_eval, pred, labels=list(range(len(EMOTION_NAMES))))
    return metrics, pred.astype(np.int64), cm.astype(np.int64)


def save_confusion(path: Path, cm: np.ndarray) -> None:
    rows = []
    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            rows.append(
                {
                    "true_label": true_label,
                    "true_class": class_name(true_label),
                    "pred_label": pred_label,
                    "pred_class": class_name(pred_label),
                    "count": int(cm[true_label, pred_label]),
                }
            )
    write_csv(path, rows)


def make_original_dataset(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([r["intensity"] for r in records], axis=0).astype(np.float32),
        np.asarray([int(r["label"]) for r in records], dtype=np.int64),
        np.asarray([int(r["graph_id"]) for r in records], dtype=np.int64),
    )


def compute_gaps(retention_rows: Sequence[Dict[str, Any]], deletion_rows: Sequence[Dict[str, Any]], original_macro_f1: float) -> List[Dict[str, Any]]:
    retention_lookup = {(r["selector"], float(r["retention_ratio"])): r for r in retention_rows}
    deletion_lookup = {(r["selector"], float(r["retention_ratio"])): r for r in deletion_rows}
    rows: List[Dict[str, Any]] = []
    for key, row in sorted(retention_lookup.items()):
        selector, ratio = key
        random_row = retention_lookup.get(("random_pixel", ratio))
        center_row = retention_lookup.get(("center_prior", ratio))
        deletion = deletion_lookup.get(key)
        random_deletion = deletion_lookup.get(("random_pixel", ratio))
        only_f1 = float(row["macro_f1"])
        delete_f1 = float(deletion["macro_f1"]) if deletion else float("nan")
        random_only = float(random_row["macro_f1"]) if random_row else float("nan")
        center_only = float(center_row["macro_f1"]) if center_row else float("nan")
        random_delete = float(random_deletion["macro_f1"]) if random_deletion else float("nan")
        deletion_drop = original_macro_f1 - delete_f1
        random_drop = original_macro_f1 - random_delete
        rows.append(
            {
                "selector": selector,
                "retention_ratio": ratio,
                "only_selected_macro_f1": only_f1,
                "random_only_selected_macro_f1": random_only,
                "center_only_selected_macro_f1": center_only,
                "selected_vs_random_macro_f1_gap": only_f1 - random_only if math.isfinite(random_only) else "nan",
                "selected_vs_center_macro_f1_gap": only_f1 - center_only if math.isfinite(center_only) else "nan",
                "delete_selected_macro_f1": delete_f1,
                "original_macro_f1": original_macro_f1,
                "deletion_drop": deletion_drop,
                "random_deletion_drop": random_drop,
                "deletion_drop_vs_random": deletion_drop - random_drop if math.isfinite(random_drop) else "nan",
            }
        )
    return rows


def per_class_rows(retention_rows: Sequence[Dict[str, Any]], deletion_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deletion_lookup = {(r["selector"], float(r["retention_ratio"])): r for r in deletion_rows}
    rows: List[Dict[str, Any]] = []
    for row in retention_rows:
        deletion = deletion_lookup.get((row["selector"], float(row["retention_ratio"])), {})
        for name in EMOTION_NAMES:
            rows.append(
                {
                    "selector": row["selector"],
                    "retention_ratio": row["retention_ratio"],
                    "class_name": name,
                    "only_selected_f1": row.get(f"per_class_f1_{name}", 0.0),
                    "delete_selected_f1": deletion.get(f"per_class_f1_{name}", 0.0),
                }
            )
    return rows


def best_by_ratio(rows: Sequence[Dict[str, Any]], ratios: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for ratio in ratios:
        candidates = [r for r in rows if abs(float(r["retention_ratio"]) - float(ratio)) < 1e-9]
        if candidates:
            out[float(ratio)] = max(candidates, key=lambda r: float(r["macro_f1"]))
    return out


def markdown_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Selector | Ratio | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['selector']} | {fmt(row['retention_ratio'])} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['weighted_f1'])} |"
        )
    if len(lines) == 2:
        lines.append("| NO_ROWS | | | | |")
    return lines


def write_report(
    output_dir: Path,
    retention_rows: Sequence[Dict[str, Any]],
    deletion_rows: Sequence[Dict[str, Any]],
    gap_rows: Sequence[Dict[str, Any]],
    per_class: Sequence[Dict[str, Any]],
    original_metrics: Dict[str, Any],
    selectors: Sequence[str],
    ratios: Sequence[float],
    max_train: Optional[int],
    max_eval: Optional[int],
    checkpoint: Optional[str],
    checkpoint_status: str,
) -> str:
    best_retention = best_by_ratio(retention_rows, ratios)
    best_delete_drop = {}
    for ratio in ratios:
        rows = [r for r in gap_rows if abs(float(r["retention_ratio"]) - float(ratio)) < 1e-9]
        if rows:
            best_delete_drop[float(ratio)] = max(rows, key=lambda r: float(r["deletion_drop"]) if math.isfinite(float(r["deletion_drop"])) else -999)
    positive_gaps = [r for r in gap_rows if r["selector"] not in {"random_pixel", "center_prior"} and float(r["selected_vs_random_macro_f1_gap"]) > 0.01]
    verdict = "PARTIAL" if retention_rows else "FAIL"
    if positive_gaps and max_train is None and max_eval is None:
        verdict = "PASS"

    lines: List[str] = [
        "# Stage 2 Retention / Deletion Test Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Stage 2 verdict: **{verdict}**.",
        f"- Lightweight classifier: `StandardScaler+SGDClassifier(log_loss)` on flattened 48x48 masked images.",
        f"- Train/eval sample cap: `{max_train if max_train is not None else 'FULL'}` / `{max_eval if max_eval is not None else 'FULL'}`.",
        f"- Original control: accuracy={fmt(original_metrics.get('accuracy'))}, macro_f1={fmt(original_metrics.get('macro_f1'))}, weighted_f1={fmt(original_metrics.get('weighted_f1'))}.",
    ]
    for ratio in (0.05, 0.10, 0.20):
        row = best_retention.get(ratio)
        if row:
            lines.append(f"- Best selector at {int(ratio * 100)}% only-selected: `{row['selector']}` macro_f1={fmt(row['macro_f1'])}.")
    if positive_gaps:
        best_gap = max(positive_gaps, key=lambda r: float(r["selected_vs_random_macro_f1_gap"]))
        lines.append(
            f"- Có selector hơn random rõ nhất: `{best_gap['selector']}` @{fmt(best_gap['retention_ratio'])}, gap={fmt(best_gap['selected_vs_random_macro_f1_gap'])}."
        )
    else:
        lines.append("- Chưa có selector feature-driven vượt random một cách rộng rãi với gap lớn trong probe này.")
    slic_rows = [r for r in retention_rows if r["selector"] == "slic_region_proposal"]
    if slic_rows:
        best_slic = max(slic_rows, key=lambda r: float(r["macro_f1"]))
        lines.append(f"- SLIC chạy được; best only-selected macro_f1={fmt(best_slic['macro_f1'])} @{fmt(best_slic['retention_ratio'])}.")
    lines.append(f"- Checkpoint inference: `{checkpoint_status}`.")
    lines.extend(
        [
            "",
            "## 2. Only-selected Results",
            "",
            *markdown_table(retention_rows),
            "",
            "Only-selected đo lượng thông tin còn giữ được khi chỉ nhìn vùng selected. Selector tốt phải vượt random cùng ratio, không chỉ vượt center.",
            "",
            "## 3. Delete-selected Results",
            "",
            *markdown_table(deletion_rows),
            "",
            "Delete-selected đo thông tin bị mất khi che vùng selected. Nếu che một selector làm macro F1 giảm mạnh hơn random, vùng đó có khả năng chứa evidence hữu ích.",
            "",
            "## 4. Selector Efficiency",
            "",
            "| Selector | Ratio | Only F1 | Gap vs Random | Gap vs Center | Deletion Drop | Drop vs Random |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gap_rows:
        lines.append(
            f"| {row['selector']} | {fmt(row['retention_ratio'])} | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row['selected_vs_random_macro_f1_gap'])} | {fmt(row['selected_vs_center_macro_f1_gap'])} | "
            f"{fmt(row['deletion_drop'])} | {fmt(row['deletion_drop_vs_random'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Per-class Analysis",
            "",
        ]
    )
    for ratio in (0.10,):
        for name in EMOTION_NAMES:
            candidates = [r for r in per_class if r["class_name"] == name and abs(float(r["retention_ratio"]) - ratio) < 1e-9]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: float(r["only_selected_f1"]))
            lines.append(f"- `{name}` @10% best only-selected: `{best['selector']}` F1={fmt(best['only_selected_f1'])}.")
    lines.extend(
        [
            "",
            "Per-class conclusions remain probe-level only. A selector can look good on one class and still be shortcut-heavy in overlay or coordinate stats.",
            "",
            "## 6. Shortcut Analysis",
            "",
        ]
    )
    random_best = best_retention.get(0.10)
    center_rows = [r for r in retention_rows if r["selector"] == "center_prior"]
    if center_rows:
        best_center = max(center_rows, key=lambda r: float(r["macro_f1"]))
        lines.append(f"- Center prior best macro_f1={fmt(best_center['macro_f1'])}; nếu quá gần feature selectors, positional shortcut còn mạnh.")
    if random_best:
        lines.append(f"- Best @10% only-selected selector is `{random_best['selector']}`; compare every claim against random/center controls.")
    if slic_rows:
        lines.append("- SLIC is worth keeping when it approaches top-k F1 while reducing fragmentation; if its F1 lags too much, use it as regularized baseline rather than winner.")
    lines.extend(
        [
            "- Gradient/delta selectors still need Stage 1 border/component stats beside this F1 report; retention alone cannot distinguish mouth/eye evidence from hair/background edges.",
            "",
            "## 7. Decision for Stage 3",
            "",
            f"- Decision: **{verdict}**.",
            "- If PASS/PARTIAL, learned selector v0 should start as a controlled hybrid: feature top-k objective plus SLIC/region continuity regularization, with random and center controls kept.",
            "- If selector-vs-random gaps are small or unstable, return to Stage 1/0.5 for mask smoothing, SLIC tuning, and actual checkpoint deletion tests.",
            "- Do not train D12/D13 selector modules from this alone; treat this as a lightweight information probe.",
            "",
            "## Output Index",
            "",
            "- `retention_metrics.csv`",
            "- `deletion_metrics.csv`",
            "- `selector_vs_random_gap.csv`",
            "- `per_class_retention_metrics.csv`",
            "- `confusion_matrices/`",
            "- `masked_data/`",
            "- `figures/`",
        ]
    )
    (output_dir / "stage2_retention_deletion_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 retention/deletion test for Stage 1 selector masks.")
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage1_dir", default="outputs/stage1_pixel_region_selection")
    parser.add_argument("--output_dir", default="outputs/stage2_retention_deletion_test")
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS), choices=list(DEFAULT_SELECTORS))
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--figure_samples_per_class", type=int, default=3)
    parser.add_argument("--save_masked_data", action="store_true", default=True)
    parser.add_argument("--no_save_masked_data", action="store_false", dest="save_masked_data")
    parser.add_argument("--fill_mode", choices=["mean", "zero"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slic_segments", type=int, default=96)
    parser.add_argument("--slic_compactness", type=float, default=0.2)
    parser.add_argument("--checkpoint", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    slic_fn, slic_error = try_import_slic()
    selectors = list(args.selectors)
    if "slic_region_proposal" in selectors and slic_fn is None:
        print(f"[Stage2] slic_region_proposal skipped: {slic_error}")
        selectors = [s for s in selectors if s != "slic_region_proposal"]

    print("[Stage2] loading records")
    train_records = load_split_records(reader, resolver, args.train_split, args.max_train_samples)
    eval_records = load_split_records(reader, resolver, args.eval_split, args.max_eval_samples)
    x_train_orig, y_train, graph_train = make_original_dataset(train_records)
    x_eval_orig, y_eval, graph_eval = make_original_dataset(eval_records)

    print("[Stage2] training original control")
    original_metrics, _, original_cm = train_eval_classifier(x_train_orig, y_train, x_eval_orig, y_eval, args.seed)
    original_row = dict(original_metrics)
    original_row.update({"selector": "original", "retention_ratio": 1.0, "mode": "original", "fill_mode": args.fill_mode})
    save_confusion(output_dir / "confusion_matrices" / "original.csv", original_cm)
    if args.save_masked_data:
        save_masked_npz(output_dir, "original", 1.0, "original", args.fill_mode, x_train_orig, y_train, graph_train, x_eval_orig, y_eval, graph_eval)

    retention_rows: List[Dict[str, Any]] = []
    deletion_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, float, int], int] = {}

    for selector in selectors:
        for ratio in [float(r) for r in args.retention_ratios]:
            for mode in MODES:
                print(f"[Stage2] {selector} ratio={ratio} mode={mode}")
                x_train, y_train_variant, graph_train_variant = build_dataset(
                    train_records,
                    selector,
                    ratio,
                    mode,
                    args.fill_mode,
                    args.seed,
                    slic_fn,
                    args.slic_segments,
                    args.slic_compactness,
                )
                x_eval, y_eval_variant, graph_eval_variant = build_dataset(
                    eval_records,
                    selector,
                    ratio,
                    mode,
                    args.fill_mode,
                    args.seed,
                    slic_fn,
                    args.slic_segments,
                    args.slic_compactness,
                )
                metrics, _, cm = train_eval_classifier(x_train, y_train_variant, x_eval, y_eval_variant, args.seed)
                row = dict(metrics)
                row.update(
                    {
                        "selector": selector,
                        "retention_ratio": ratio,
                        "mode": mode,
                        "fill_mode": args.fill_mode,
                        "original_accuracy": original_metrics["accuracy"],
                        "original_macro_f1": original_metrics["macro_f1"],
                        "original_weighted_f1": original_metrics["weighted_f1"],
                    }
                )
                if mode == "only_selected":
                    row.update(
                        {
                            "only_selected_accuracy": row["accuracy"],
                            "only_selected_macro_f1": row["macro_f1"],
                            "only_selected_weighted_f1": row["weighted_f1"],
                        }
                    )
                else:
                    row.update(
                        {
                            "delete_selected_accuracy": row["accuracy"],
                            "delete_selected_macro_f1": row["macro_f1"],
                            "delete_selected_weighted_f1": row["weighted_f1"],
                        }
                    )
                cm_path = output_dir / "confusion_matrices" / selector / ratio_name(ratio) / f"{mode}.csv"
                save_confusion(cm_path, cm)
                if mode == "only_selected":
                    retention_rows.append(row)
                else:
                    deletion_rows.append(row)
                if args.save_masked_data:
                    save_masked_npz(output_dir, selector, ratio, mode, args.fill_mode, x_train, y_train_variant, graph_train_variant, x_eval, y_eval_variant, graph_eval_variant)

            for record in eval_records:
                key = (selector, ratio, int(record["label"]))
                if visual_counts.get(key, 0) >= int(args.figure_samples_per_class):
                    continue
                mask = build_mask(record, selector, ratio, args.seed, slic_fn, args.slic_segments, args.slic_compactness)
                if mask is None:
                    continue
                save_visuals(output_dir, selector, ratio, record["split"], int(record["graph_id"]), int(record["label"]), record["intensity"], mask, args.fill_mode)
                visual_counts[key] = visual_counts.get(key, 0) + 1

    gap_rows = compute_gaps(retention_rows, deletion_rows, float(original_metrics["macro_f1"]))
    per_class = per_class_rows(retention_rows, deletion_rows)
    write_csv(output_dir / "retention_metrics.csv", retention_rows, METRIC_FIELDS)
    write_csv(output_dir / "deletion_metrics.csv", deletion_rows, METRIC_FIELDS)
    write_csv(output_dir / "selector_vs_random_gap.csv", gap_rows)
    write_csv(output_dir / "per_class_retention_metrics.csv", per_class)
    checkpoint_status = "SKIPPED: no --checkpoint provided"
    if args.checkpoint:
        checkpoint_status = "SKIPPED: checkpoint inference not implemented in this lightweight script; no model artifacts were modified"
    verdict = write_report(
        output_dir,
        retention_rows,
        deletion_rows,
        gap_rows,
        per_class,
        original_row,
        selectors,
        [float(r) for r in args.retention_ratios],
        args.max_train_samples,
        args.max_eval_samples,
        args.checkpoint,
        checkpoint_status,
    )
    best = best_by_ratio(retention_rows, [0.05, 0.10, 0.20])
    print(f"[Stage2] output_dir={output_dir}")
    print(f"[Stage2] verdict={verdict}")
    for ratio, row in best.items():
        print(f"[Stage2] best_{int(ratio * 100)}={row['selector']} macro_f1={fmt(row['macro_f1'])}")
    slic_rows = [r for r in retention_rows if r["selector"] == "slic_region_proposal"]
    print(f"[Stage2] slic={'OK' if slic_rows else 'SKIPPED'}")


if __name__ == "__main__":
    main()
