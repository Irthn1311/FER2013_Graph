"""Stage 4 refinement run for learned evidence selector v0.

This is still Stage 4 only: no motif bank, no SupCon, no part grouping, no
D12/D13 training. It reruns a small set of learned selector configurations and
adds fixed-probe deletion protocols plus multiple fill modes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import defaultdict
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
from stage1_pixel_region_selection import ensure_dir, fmt, resolve_graph_repo, safe_name, try_import_slic
from stage2_retention_deletion_test import class_name, make_original_dataset, ratio_name, save_confusion
from stage4_learned_evidence_selector import (
    CONTROL_SELECTORS,
    ExperimentConfig,
    aggregate_stats,
    build_control_mask,
    build_parser as build_stage4_parser,
    input_feature_names,
    mask_structure_stats,
    predict_score,
    score_correlations,
    train_one_experiment,
)
from stage36_structure_aware_diagnostics import load_split_records


REFINEMENT_CONFIGS = (
    ("pixel_mlp", "no_xy_basic", False, "pixel_mlp__no_xy_basic__main_hybrid"),
    ("tiny_conv", "no_xy_basic", False, "tiny_conv__no_xy_basic__main_hybrid"),
    ("tiny_conv", "structure_augmented_no_xy", True, "tiny_conv__structure_augmented_no_xy__main_hybrid__regularized"),
)
DEFAULT_EVAL_RATIOS = (0.10, 0.20)
DEFAULT_TARGET_RATIOS = (0.10, 0.20)
DEFAULT_FILL_MODES = ("mean", "zero", "local_mean")
REFINEMENT_CONTROLS = (
    "random_pixel",
    "center_prior",
    "gradient_topk",
    "delta_edge_topk",
    "main_hybrid_teacher",
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


def local_mean_fill(image: np.ndarray, size: int = 7) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32).reshape(48, 48)
    try:
        from scipy.ndimage import uniform_filter

        out = uniform_filter(img, size=int(size), mode="nearest")
    except Exception:
        pad = int(size) // 2
        padded = np.pad(img, pad, mode="edge")
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(size):
            for x in range(size):
                out += padded[y : y + 48, x : x + 48]
        out /= float(size * size)
    return out.reshape(-1).astype(np.float32)


def apply_mask_refined(intensity: np.ndarray, mask: np.ndarray, mode: str, fill_mode: str) -> np.ndarray:
    image = np.asarray(intensity, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if fill_mode == "zero":
        fill = np.zeros_like(image, dtype=np.float32)
    elif fill_mode == "local_mean":
        fill = local_mean_fill(image)
    else:
        fill = np.full_like(image, float(image.mean()), dtype=np.float32)
    out = image.copy()
    if mode == "only_selected":
        out[~mask] = fill[~mask]
    elif mode == "delete_selected":
        out[mask] = fill[mask]
    elif mode == "original":
        pass
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return out.astype(np.float32)


def train_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int, max_iter: int) -> Any:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=int(max_iter),
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        ),
    )
    clf.fit(x_train, y_train)
    return clf


def eval_probe(clf: Any, x_eval: np.ndarray, y_eval: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    pred = clf.predict(x_eval)
    per_class = f1_score(y_eval, pred, average=None, labels=list(range(len(EMOTION_NAMES))), zero_division=0)
    row: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_eval, pred, average="weighted", zero_division=0)),
    }
    for idx, name in enumerate(EMOTION_NAMES):
        row[f"per_class_f1_{name}"] = float(per_class[idx])
    cm = confusion_matrix(y_eval, pred, labels=list(range(len(EMOTION_NAMES))))
    return row, cm.astype(np.int64)


def source_mask_score(
    source_name: str,
    source_type: str,
    model: Optional[torch.nn.Module],
    input_variant: str,
    record: Dict[str, Any],
    ratio: float,
    args: argparse.Namespace,
    slic_fn: Any,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if source_type == "learned":
        assert model is not None
        score = predict_score(model, record, input_variant, device)
        mask = np.zeros_like(score, dtype=bool)
        k = max(1, int(round(score.size * float(ratio))))
        idx = np.argpartition(score, score.size - k)[score.size - k :]
        mask[idx] = True
        return mask, score, 0
    return build_control_mask(
        record,
        source_name,
        ratio,
        slic_fn,
        args.slic_segments,
        args.slic_compactness,
        args.smooth_sigma,
        args.seed,
    )


def build_source_arrays(
    source_name: str,
    source_type: str,
    model: Optional[torch.nn.Module],
    input_variant: str,
    records: Sequence[Dict[str, Any]],
    ratio: float,
    fill_mode: str,
    args: argparse.Namespace,
    slic_fn: Any,
    device: torch.device,
    collect_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    only_rows: List[np.ndarray] = []
    delete_rows: List[np.ndarray] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    for record in records:
        mask, score, region_count = source_mask_score(source_name, source_type, model, input_variant, record, ratio, args, slic_fn, device)
        only_rows.append(apply_mask_refined(record["intensity"], mask, "only_selected", fill_mode))
        delete_rows.append(apply_mask_refined(record["intensity"], mask, "delete_selected", fill_mode))
        if collect_stats:
            _, comp, coord = mask_structure_stats(record, source_type, source_name, float(ratio), mask, region_count, input_variant)
            corr = score_correlations(record, score)
            comp.update(corr)
            coord.update(corr)
            component_rows.append(comp)
            coordinate_rows.append(coord)
            structure_rows.append(comp)
    return (
        np.stack(only_rows, axis=0).astype(np.float32),
        np.stack(delete_rows, axis=0).astype(np.float32),
        component_rows,
        coordinate_rows,
        structure_rows,
    )


def build_refinement_experiments(target_ratios: Sequence[float]) -> List[ExperimentConfig]:
    exps: List[ExperimentConfig] = []
    for target_ratio in target_ratios:
        for arch, input_variant, regularized, prefix in REFINEMENT_CONFIGS:
            suffix = f"r{str(float(target_ratio)).replace('.', 'p')}"
            exps.append(
                ExperimentConfig(
                    name=f"{prefix}__{suffix}",
                    selector_arch=arch,
                    input_variant=input_variant,
                    teacher="main_hybrid",
                    target_ratio=float(target_ratio),
                    use_structure_regularizers=bool(regularized),
                )
            )
    return exps


def compare_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {
        (r["protocol"], r["fill_mode"], r["source_name"], float(r["retention_ratio"]), r["eval_mode"]): r
        for r in metric_rows
    }
    out: List[Dict[str, Any]] = []
    for row in metric_rows:
        if row["eval_mode"] != "only_selected":
            continue
        key_base = (row["protocol"], row["fill_mode"], float(row["retention_ratio"]))
        random_row = lookup.get((key_base[0], key_base[1], "random_pixel", key_base[2], "only_selected"), {})
        center_row = lookup.get((key_base[0], key_base[1], "center_prior", key_base[2], "only_selected"), {})
        hybrid_row = lookup.get((key_base[0], key_base[1], "main_hybrid_teacher", key_base[2], "only_selected"), {})
        delete_row = lookup.get((key_base[0], key_base[1], row["source_name"], key_base[2], "delete_selected"), {})
        original_row = lookup.get((key_base[0], key_base[1], "original", key_base[2], "original"), {})
        if row["protocol"] == "fixed_only_selected_probe":
            drop_base = float(row["macro_f1"])
        else:
            drop_base = float(original_row.get("macro_f1", "nan"))
        delete_f1 = float(delete_row.get("macro_f1", "nan"))
        out.append(
            {
                "protocol": row["protocol"],
                "fill_mode": row["fill_mode"],
                "source_name": row["source_name"],
                "source_type": row["source_type"],
                "retention_ratio": row["retention_ratio"],
                "only_selected_macro_f1": row["macro_f1"],
                "delete_selected_macro_f1": delete_f1,
                "deletion_drop": drop_base - delete_f1 if math.isfinite(drop_base) and math.isfinite(delete_f1) else "nan",
                "gap_vs_random": float(row["macro_f1"]) - float(random_row.get("macro_f1", "nan")),
                "gap_vs_center": float(row["macro_f1"]) - float(center_row.get("macro_f1", "nan")),
                "gap_vs_main_hybrid_teacher": float(row["macro_f1"]) - float(hybrid_row.get("macro_f1", "nan")),
                "border_ratio": row.get("mean_border_ratio", ""),
                "center_ratio": row.get("mean_center_ratio", ""),
                "components": row.get("mean_connected_components", ""),
                "long_contour_ratio": row.get("mean_selected_long_contour_ratio", ""),
                "smooth_region_ratio": row.get("mean_selected_smooth_region_ratio", ""),
                "short_structure_ratio": row.get("mean_selected_short_structure_ratio", ""),
            }
        )
    return out


def per_class_rows(metric_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in metric_rows:
        if row["eval_mode"] not in {"only_selected", "delete_selected", "original"}:
            continue
        for name in EMOTION_NAMES:
            rows.append(
                {
                    "protocol": row["protocol"],
                    "fill_mode": row["fill_mode"],
                    "source_name": row["source_name"],
                    "source_type": row["source_type"],
                    "retention_ratio": row["retention_ratio"],
                    "eval_mode": row["eval_mode"],
                    "class_name": name,
                    "f1": row.get(f"per_class_f1_{name}", 0.0),
                }
            )
    return rows


def best_learned(compare: Sequence[Dict[str, Any]], protocol: str, fill_mode: str, ratio: float) -> Optional[Dict[str, Any]]:
    candidates = [
        r
        for r in compare
        if r["source_type"] == "learned"
        and r["protocol"] == protocol
        and r["fill_mode"] == fill_mode
        and abs(float(r["retention_ratio"]) - float(ratio)) < 1e-9
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["only_selected_macro_f1"]))


def write_report(output_dir: Path, compare: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    best10 = best_learned(compare, "fixed_original_probe", "mean", 0.10)
    best20 = best_learned(compare, "fixed_original_probe", "mean", 0.20)
    positive_drop = [
        r
        for r in compare
        if r["source_type"] == "learned"
        and r["protocol"] == "fixed_original_probe"
        and r["fill_mode"] == "mean"
        and float(r["retention_ratio"]) in {0.10, 0.20}
        and float(r.get("deletion_drop", -999)) >= 0
    ]
    ready = False
    if best10 and best20:
        ready = (
            float(best10["gap_vs_random"]) > 0.03
            and float(best10["gap_vs_center"]) > 0.03
            and float(best20["gap_vs_random"]) > 0.03
            and float(best20["gap_vs_center"]) > 0.03
            and float(best10["deletion_drop"]) >= 0
            and float(best20["deletion_drop"]) >= 0
            and float(best10["center_ratio"]) < 0.35
            and float(best20["center_ratio"]) < 0.35
        )
    verdict = "Stage 4_READY_FOR_STAGE_5" if ready else "Stage 4_STILL_NOT_READY"
    lines: List[str] = [
        "# Stage 4 Refinement Run Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Verdict: **{verdict}**.",
        "- Scope: Stage 4 refinement only; no motif, no SupCon, no part grouping.",
        f"- Protocols: `fixed_original_probe`, `fixed_only_selected_probe`.",
        f"- Fill modes: `{', '.join(args.fill_modes)}`.",
        f"- Trained target ratios: `{', '.join(str(r) for r in args.target_train_ratios)}`.",
    ]
    for ratio, row in ((0.10, best10), (0.20, best20)):
        if row:
            lines.append(
                f"- Best learned fixed-original mean @{int(ratio * 100)}%: `{row['source_name']}` "
                f"F1={fmt(row['only_selected_macro_f1'])}, deletion_drop={fmt(row['deletion_drop'])}, "
                f"gap_random={fmt(row['gap_vs_random'])}, gap_center={fmt(row['gap_vs_center'])}."
            )
    lines.extend(
        [
            "",
            "## 2. Deletion Protocol Interpretation",
            "",
            "Deletion yếu được xem là protocol issue nếu fixed-original probe tạo drop dương trong khi old retrain-delete từng âm. Nếu fixed-original vẫn âm hoặc rất nhỏ ở best rows, selector chưa có causal evidence đủ mạnh.",
            "",
            "| Protocol | Fill | Ratio | Best learned | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap hybrid | Border | Center | Components | Long | Smooth | Short |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in ("fixed_original_probe", "fixed_only_selected_probe"):
        for fill in args.fill_modes:
            for ratio in args.retention_ratios:
                row = best_learned(compare, protocol, fill, float(ratio))
                if not row:
                    continue
                lines.append(
                    f"| {protocol} | {fill} | {fmt(ratio)} | {row['source_name']} | {fmt(row['only_selected_macro_f1'])} | "
                    f"{fmt(row['delete_selected_macro_f1'])} | {fmt(row['deletion_drop'])} | {fmt(row['gap_vs_random'])} | "
                    f"{fmt(row['gap_vs_center'])} | {fmt(row['gap_vs_main_hybrid_teacher'])} | {fmt(row['border_ratio'])} | "
                    f"{fmt(row['center_ratio'])} | {fmt(row['components'])} | {fmt(row['long_contour_ratio'])} | "
                    f"{fmt(row['smooth_region_ratio'])} | {fmt(row['short_structure_ratio'])} |"
                )
    lines.extend(
        [
            "",
            "## 3. Best Configuration After Refinement",
            "",
        ]
    )
    if best20:
        lines.append(f"- Main candidate @20%: `{best20['source_name']}`.")
        lines.append(f"- Evidence: macro F1 `{fmt(best20['only_selected_macro_f1'])}`, gap random `{fmt(best20['gap_vs_random'])}`, gap center `{fmt(best20['gap_vs_center'])}`, deletion drop `{fmt(best20['deletion_drop'])}`.")
    if best10:
        lines.append(f"- Sparse diagnostic @10%: `{best10['source_name']}` with macro F1 `{fmt(best10['only_selected_macro_f1'])}` and deletion drop `{fmt(best10['deletion_drop'])}`.")
    lines.extend(
        [
            "",
            "## 4. Target Ratio 0.20 vs 0.10",
            "",
            "- Target ratio 0.20 is preferred only if it improves @20 retention without making @10 collapse and without negative fixed-original deletion drop.",
            "- Compare rows in `stage4_refinement_vs_controls.csv` by source suffix `r0p1` versus `r0p2`.",
            "",
            "## 5. no_xy Direction",
            "",
            "- `no_xy` remains the main direction if it matches or beats `with_xy` from the previous run while keeping center ratio low. This refinement run does not retrain `with_xy`; it focuses on the selected three no-xy/structure configs from the memo.",
            "",
            "## 6. Structure Regularization",
            "",
            "- Structure regularization is useful only if it reduces border/long-contour/component count without hurting retention or deletion. If it only improves teacher loss, keep it as auxiliary, not main.",
            "",
            "## 7. Stage 4 Readiness",
            "",
        ]
    )
    if ready:
        lines.append("Stage 4 is READY by the numeric criteria in this run. Manual visual audit is still required before any part grouping.")
    else:
        lines.append("Stage 4 is **STILL_NOT_READY**. Do not proceed to Stage 5. Continue refinement of deletion protocol/teacher ratio/structure penalties.")
    lines.extend(
        [
            "",
            "## 8. What Not To Do",
            "",
            "- Không motif.",
            "- Không SupCon.",
            "- Không part grouping.",
            "- Không full D12/D13.",
            "- Không claim semantic motif hoặc causal evidence nếu deletion chưa ổn định.",
            "",
        ]
    )
    (output_dir / "stage4_refinement_run_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--output_dir", default="outputs/stage4_learned_evidence_selector")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--probe_train_cap", type=int, default=2000)
    parser.add_argument("--probe_eval_cap", type=int, default=1000)
    parser.add_argument("--target_train_ratios", nargs="+", type=float, default=list(DEFAULT_TARGET_RATIOS))
    parser.add_argument("--retention_ratios", nargs="+", type=float, default=list(DEFAULT_EVAL_RATIOS))
    parser.add_argument("--fill_modes", nargs="+", default=list(DEFAULT_FILL_MODES))
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    parser.add_argument("--lambda_teacher", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.02)
    parser.add_argument("--lambda_region", type=float, default=0.02)
    parser.add_argument("--lambda_long", type=float, default=0.02)
    parser.add_argument("--lambda_smooth_region", type=float, default=0.01)
    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--lambda_border", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    return parser


def stage4_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = build_stage4_parser().parse_args([])
    for key, value in vars(args).items():
        setattr(ns, key, value)
    ns.teacher = "main_hybrid"
    ns.fill_mode = "mean"
    ns.selector_arch = "tiny_conv"
    ns.input_variant = "no_xy_basic"
    ns.target_ratio = 0.20
    ns.experiment_suite = "single"
    ns.use_structure_regularizers = True
    return ns


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4Refine] SLIC unavailable: {slic_error}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"[Stage4Refine] device={device}")
    print("[Stage4Refine] loading records")
    train_records_full = load_split_records(reader, resolver, "train", int(args.max_train_samples))
    val_records_full = load_split_records(reader, resolver, "val", int(args.max_val_samples))
    probe_train_records = train_records_full[: min(len(train_records_full), int(args.probe_train_cap))]
    probe_eval_records = val_records_full[: min(len(val_records_full), int(args.probe_eval_cap))]
    x_train_original, y_train, _ = make_original_dataset(probe_train_records)
    x_eval_original, y_eval, _ = make_original_dataset(probe_eval_records)

    s4_args = stage4_namespace(args)
    experiments = build_refinement_experiments([float(r) for r in args.target_train_ratios])
    learned_models: Dict[str, Tuple[torch.nn.Module, str]] = {}
    train_rows: List[Dict[str, Any]] = []
    print(f"[Stage4Refine] training experiments={len(experiments)}")
    for exp in experiments:
        print(f"[Stage4Refine] train {exp.name}")
        s4_args.target_ratio = exp.target_ratio
        model, logs, meta = train_one_experiment(exp, train_records_full, val_records_full, s4_args, slic_fn, output_dir, device)
        learned_models[exp.name] = (model, exp.input_variant)
        for row in logs:
            row.update(meta)
            row["refinement_run"] = 1
        train_rows.extend(logs)

    sources: List[Tuple[str, str, Optional[torch.nn.Module], str]] = []
    for control in REFINEMENT_CONTROLS:
        sources.append((control, "control", None, ""))
    for exp_name, (model, input_variant) in learned_models.items():
        sources.append((exp_name, "learned", model, input_variant))

    metric_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    print("[Stage4Refine] fixed-probe evaluation")
    for fill_mode in args.fill_modes:
        original_probe = train_probe(x_train_original, y_train, args.seed, args.classifier_max_iter)
        original_eval_metrics, original_cm = eval_probe(original_probe, x_eval_original, y_eval)
        for ratio in [float(r) for r in args.retention_ratios]:
            original_row = {
                "protocol": "fixed_original_probe",
                "fill_mode": fill_mode,
                "source_name": "original",
                "source_type": "control",
                "retention_ratio": ratio,
                "eval_mode": "original",
                **original_eval_metrics,
            }
            metric_rows.append(original_row)
            save_confusion(output_dir / "confusion_matrices_refinement" / "fixed_original_probe" / fill_mode / "original" / f"{ratio_name(ratio)}.csv", original_cm)

        for source_name, source_type, model, input_variant in sources:
            for ratio in [float(r) for r in args.retention_ratios]:
                x_train_only, x_train_delete, _, _, _ = build_source_arrays(
                    source_name,
                    source_type,
                    model,
                    input_variant,
                    probe_train_records,
                    ratio,
                    fill_mode,
                    s4_args,
                    slic_fn,
                    device,
                    collect_stats=False,
                )
                x_eval_only, x_eval_delete, comps, coords, structs = build_source_arrays(
                    source_name,
                    source_type,
                    model,
                    input_variant,
                    probe_eval_records,
                    ratio,
                    fill_mode,
                    s4_args,
                    slic_fn,
                    device,
                    collect_stats=True,
                )
                agg = aggregate_stats(comps)
                component_rows.extend([{**r, "fill_mode": fill_mode} for r in comps])
                coordinate_rows.extend([{**r, "fill_mode": fill_mode} for r in coords])
                structure_rows.extend([{**r, "fill_mode": fill_mode} for r in structs])

                for eval_mode, x_eval in (("only_selected", x_eval_only), ("delete_selected", x_eval_delete)):
                    metrics, cm = eval_probe(original_probe, x_eval, y_eval)
                    metric_rows.append(
                        {
                            "protocol": "fixed_original_probe",
                            "fill_mode": fill_mode,
                            "source_name": source_name,
                            "source_type": source_type,
                            "retention_ratio": ratio,
                            "eval_mode": eval_mode,
                            **metrics,
                            **agg,
                        }
                    )
                    save_confusion(output_dir / "confusion_matrices_refinement" / "fixed_original_probe" / fill_mode / safe_name(source_name) / f"{ratio_name(ratio)}_{eval_mode}.csv", cm)

                only_probe = train_probe(x_train_only, y_train, args.seed, args.classifier_max_iter)
                for eval_mode, x_eval in (("only_selected", x_eval_only), ("delete_selected", x_eval_delete)):
                    metrics, cm = eval_probe(only_probe, x_eval, y_eval)
                    metric_rows.append(
                        {
                            "protocol": "fixed_only_selected_probe",
                            "fill_mode": fill_mode,
                            "source_name": source_name,
                            "source_type": source_type,
                            "retention_ratio": ratio,
                            "eval_mode": eval_mode,
                            **metrics,
                            **agg,
                        }
                    )
                    save_confusion(output_dir / "confusion_matrices_refinement" / "fixed_only_selected_probe" / fill_mode / safe_name(source_name) / f"{ratio_name(ratio)}_{eval_mode}.csv", cm)

    compare = compare_rows(metric_rows)
    per_class = per_class_rows(metric_rows)
    write_csv(output_dir / "stage4_refinement_train_log.csv", train_rows)
    write_csv(output_dir / "stage4_refinement_probe_metrics.csv", metric_rows)
    write_csv(output_dir / "stage4_refinement_vs_controls.csv", compare)
    write_csv(output_dir / "stage4_refinement_component_stats.csv", component_rows)
    write_csv(output_dir / "stage4_refinement_coordinate_stats.csv", coordinate_rows)
    write_csv(output_dir / "stage4_refinement_structure_stats.csv", structure_rows)
    write_csv(output_dir / "stage4_refinement_per_class_f1.csv", per_class)
    verdict = write_report(output_dir, compare, train_rows, args)
    print(f"[Stage4Refine] output_dir={output_dir}")
    print(f"[Stage4Refine] verdict={verdict}")
    for protocol in ("fixed_original_probe", "fixed_only_selected_probe"):
        for ratio in [0.10, 0.20]:
            row = best_learned(compare, protocol, "mean", ratio)
            if row:
                print(
                    f"[Stage4Refine] {protocol}_best_{int(ratio*100)}={row['source_name']} "
                    f"f1={fmt(row['only_selected_macro_f1'])} drop={fmt(row['deletion_drop'])}"
                )
    print("[Stage4Refine] next_step=Stage4 refinement only; do not proceed to Stage5 unless READY")


if __name__ == "__main__":
    main()
