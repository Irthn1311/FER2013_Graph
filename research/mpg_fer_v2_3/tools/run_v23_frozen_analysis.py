"""Run the registered post-training analysis for the frozen MPG-FER v2.3 model.

This tool is deliberately outside the training/source-lock path.  It refuses to
open PrivateTest until the official execution manifest records completed
training and a one-shot post-freeze Private evaluation.  Representations use
raw single-view FP32 inference and the A6 Train-fit-only probe protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V23_ROOT = PROJECT_ROOT / "research" / "mpg_fer_v2_3"
V23_SRC = V23_ROOT / "src"
A6_ROOT = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6"
V21_RUN = (
    PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs"
    / "segment_02" / "mpg_fer_v2_1_run"
)
V22_RUN = (
    PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs"
    / "segment_02" / "mpg_fer_v2_2_run"
)

EXPECTED_SOURCE_SHA256 = (
    "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
)
EXPECTED_CONFIG_SHA256 = (
    "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2"
)
EXPECTED_V22_CHECKPOINT_SHA256 = (
    "a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4"
)
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
STAGES = {
    "PRE": "R1",
    "L1": "R2",
    "L2": "R3",
    "L5": "R6",
    "Motif Readout": "R7",
    "Fusion": "R8",
}
CONFUSION_DIRECTIONS = [
    ("Fear", "Sad"),
    ("Fear", "Neutral"),
    ("Fear", "Angry"),
    ("Sad", "Neutral"),
    ("Neutral", "Sad"),
    ("Angry", "Sad"),
]

sys.path.insert(0, str(V23_SRC))
from mpg_fer_v2_3.checkpoint import config_hash  # noqa: E402
from mpg_fer_v2_3.config import MPGConfig  # noqa: E402
from mpg_fer_v2_3.data import FER2013Dataset, validate_split_path  # noqa: E402
from mpg_fer_v2_3.model import MPGFER  # noqa: E402


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_frozen_run(run_dir: Path) -> dict:
    required = [
        "execution_manifest.json",
        "segment_manifest.json",
        "best_val_acc.json",
        "best_val_acc.pt",
        "resume_latest.json",
        "resume_latest.pt",
        "final_selection_manifest.json",
        "public_metrics.json",
        "private_metrics.json",
        "history.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    require(not missing, f"Final run is missing required artifacts: {missing}")

    execution = read_json(run_dir / "execution_manifest.json")
    segment = read_json(run_dir / "segment_manifest.json")
    best = read_json(run_dir / "best_val_acc.json")
    resume = read_json(run_dir / "resume_latest.json")
    selection = read_json(run_dir / "final_selection_manifest.json")
    require(isinstance(execution, dict), "Invalid execution manifest")
    require(execution.get("status") == "TRAINING_COMPLETED", "Training is not complete")
    require(execution.get("source_hash") == EXPECTED_SOURCE_SHA256, "Execution source mismatch")
    require(execution.get("PRIVATE_EVALUATED") is True, "Private evaluation is not recorded")
    require(
        execution.get("private_evaluated_only_after_freeze") is True,
        "Private post-freeze firewall evidence is absent",
    )
    require(segment.get("status") == "TRAINING_COMPLETED", "Final segment is incomplete")

    checkpoint_sha = sha256_file(run_dir / "best_val_acc.pt")
    resume_sha = sha256_file(run_dir / "resume_latest.pt")
    run_id = execution.get("run_id")
    require(run_id and best.get("run_id") == run_id, "Best-checkpoint run_id mismatch")
    require(segment.get("run_id") == run_id, "Segment run_id mismatch")
    require(selection.get("run_id") == run_id, "Selection-manifest run_id mismatch")
    require(segment.get("source_hash") == EXPECTED_SOURCE_SHA256, "Segment source mismatch")
    require(segment.get("config_hash") == EXPECTED_CONFIG_SHA256, "Segment config mismatch")
    require(best.get("source_hash") == EXPECTED_SOURCE_SHA256, "Source hash mismatch")
    require(best.get("config_hash") == EXPECTED_CONFIG_SHA256, "Config hash mismatch")
    require(config_hash(MPGConfig()) == EXPECTED_CONFIG_SHA256, "Local config identity mismatch")
    require(best.get("checkpoint_sha256") == checkpoint_sha, "Best-checkpoint SHA mismatch")
    require(execution.get("best_checkpoint_sha256") == checkpoint_sha, "Execution SHA mismatch")
    require(selection.get("checkpoint_sha256") == checkpoint_sha, "Selection SHA mismatch")
    require(resume.get("checkpoint_sha256") == resume_sha, "Final resume SHA mismatch")
    require(resume.get("source_hash") == EXPECTED_SOURCE_SHA256, "Resume source mismatch")
    require(resume.get("config_hash") == EXPECTED_CONFIG_SHA256, "Resume config mismatch")
    require(best.get("best_epoch") == selection.get("best_epoch"), "Best epoch mismatch")
    require(best.get("weights_type") == "EMA", "Best checkpoint is not EMA")
    require(selection.get("weights_type") == "EMA", "Selection weights are not EMA")
    require(
        selection.get("private_evaluated_only_after_freeze") is True,
        "Selection manifest lacks the Private firewall statement",
    )

    return {
        "run_id": run_id,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "best_epoch": int(best["best_epoch"]),
        "best_checkpoint_sha256": checkpoint_sha,
        "final_resume_sha256": resume_sha,
        "completed_epoch": int(resume["epoch"]),
        "private_evaluated_only_after_freeze": True,
    }


def register_hooks(model: MPGFER, capture: dict[str, torch.Tensor]):
    handles = []

    def hook(key: str):
        def capture_hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            capture[key] = value.detach()
        return capture_hook

    def pre_hook(key: str):
        def capture_pre_hook(_module, inputs):
            capture[key] = inputs[0].detach()
        return capture_pre_hook

    handles.append(model.motif_composer.register_forward_hook(hook("R1")))
    handles.append(model.motif_gnn[0].register_forward_hook(hook("R2")))
    handles.append(model.motif_gnn[1].register_forward_hook(hook("R3")))
    handles.append(model.motif_gnn[4].register_forward_hook(hook("R6")))
    handles.append(model.motif_readout_proj.register_forward_hook(hook("R7")))
    handles.append(model.classifier.register_forward_pre_hook(pre_hook("R8")))
    handles.append(model.classifier.register_forward_hook(hook("R10")))
    return handles


def extract_features(
    model: MPGFER,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    integrity: dict,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    capture: dict[str, torch.Tensor] = {}
    handles = register_hooks(model, capture)

    # A hook must be observational: verify that registering it changes no logits.
    probe = torch.linspace(0.0, 1.0, 2 * 48 * 48, device=device).reshape(2, 1, 48, 48)
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        baseline, _ = model(probe)
    handles = register_hooks(model, capture)
    with torch.no_grad():
        observed, _ = model(probe)
    max_hook_diff = float((baseline - observed).abs().max().cpu())
    require(max_hook_diff == 0.0, f"Hook validation failed: max diff {max_hook_diff}")

    save: dict[str, np.ndarray] = {
        "hook_max_logit_abs_diff": np.asarray([max_hook_diff], dtype=np.float64),
        "run_id": np.asarray([integrity["run_id"]]),
        "source_sha256": np.asarray([integrity["source_sha256"]]),
        "config_sha256": np.asarray([integrity["config_sha256"]]),
        "best_checkpoint_sha256": np.asarray(
            [integrity["best_checkpoint_sha256"]]
        ),
    }
    split_specs = [
        ("train", PROJECT_ROOT / "data" / "train.csv", "train"),
        ("public", PROJECT_ROOT / "data" / "val.csv", "val"),
        # Reached only after verify_frozen_run proves the post-freeze boundary.
        ("private", PROJECT_ROOT / "data" / "test.csv", "test"),
    ]
    for split_name, csv_path, role in split_specs:
        started = time.time()
        dataset = FER2013Dataset(
            validate_split_path(csv_path, role), split=role, augment=False
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        chunks = {stage: [] for stage in ["R1", "R2", "R3", "R6", "R7", "R8", "R10"]}
        with torch.no_grad():
            for batch_index, (images, _targets) in enumerate(loader, start=1):
                capture.clear()
                model(images.to(device, non_blocking=True))
                for stage in chunks:
                    value = capture[stage]
                    if stage in {"R1", "R2", "R3", "R6"}:
                        value = value.mean(dim=1)
                    chunks[stage].append(value.float().cpu().numpy())
                if batch_index % 100 == 0 or batch_index == len(loader):
                    print(
                        f"{split_name}: {batch_index}/{len(loader)} batches "
                        f"({time.time() - started:.1f}s)",
                        flush=True,
                    )
        save[f"{split_name}_targets"] = dataset.labels.astype(np.int64)
        for stage, values in chunks.items():
            save[f"{split_name}_{stage}"] = np.concatenate(values).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save)
    for handle in handles:
        handle.remove()


def evaluate_probe(
    stage_name: str,
    stage_key: str,
    features: np.lib.npyio.NpzFile,
) -> dict:
    x_train = features[f"train_{stage_key}"]
    y_train = features["train_targets"]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    started = time.time()
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-6,
        random_state=42,
    )
    classifier.fit(x_train_scaled, y_train)

    result = {
        "stage": stage_name,
        "a6_stage": stage_key,
        "dimensions": int(x_train.shape[1]),
        "fit_duration_sec": time.time() - started,
        "iterations": int(classifier.n_iter_.max()),
        "converged": bool(classifier.n_iter_.max() < classifier.max_iter),
    }
    for split in ["train", "public", "private"]:
        x = features[f"{split}_{stage_key}"]
        y = features[f"{split}_targets"]
        predictions = classifier.predict(scaler.transform(x))
        result[f"{split}_accuracy"] = float(accuracy_score(y, predictions))
        result[f"{split}_macro_f1"] = float(
            f1_score(y, predictions, average="macro", zero_division=0)
        )
    result["train_public_gap"] = (
        result["train_accuracy"] - result["public_accuracy"]
    )
    result["train_private_gap"] = (
        result["train_accuracy"] - result["private_accuracy"]
    )
    return result


def expected_calibration_error(logits: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    total = len(targets)
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if selected.any():
            ece += selected.sum() / total * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(ece)


def raw_fp32_metrics(features: np.lib.npyio.NpzFile, split: str) -> dict:
    logits = features[f"{split}_R10"].astype(np.float64)
    targets = features[f"{split}_targets"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    nll = float((-shifted[np.arange(len(targets)), targets] + log_normalizer).mean())
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": float((predictions == targets).mean()),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "nll": nll,
        "ece_15_equal_width_bins": expected_calibration_error(logits, targets),
    }


def official_summary(metrics: dict) -> dict:
    return {
        view: {
            key: metrics[view][key]
            for key in ["loss", "accuracy", "macro_f1"]
        }
        for view in ["raw", "tta"]
    }


def classwise_payload(v22: dict, v23: dict) -> dict:
    payload = {}
    for split in ["public", "private"]:
        payload[split] = {}
        for view in ["raw", "tta"]:
            old, new = v22[split][view], v23[split][view]
            rows = []
            for index, name in enumerate(CLASS_NAMES):
                rows.append({
                    "class": name,
                    "v22_f1": old["per_class_f1"][index],
                    "v23_f1": new["per_class_f1"][index],
                    "delta_f1": new["per_class_f1"][index] - old["per_class_f1"][index],
                    "v22_support": old["support"][index],
                    "v23_support": new["support"][index],
                })
            directions = []
            for source, target in CONFUSION_DIRECTIONS:
                i, j = CLASS_NAMES.index(source), CLASS_NAMES.index(target)
                directions.append({
                    "direction": f"{source} -> {target}",
                    "v22_count": old["confusion_matrix"][i][j],
                    "v23_count": new["confusion_matrix"][i][j],
                    "delta_count": new["confusion_matrix"][i][j] - old["confusion_matrix"][i][j],
                })
            payload[split][view] = {"per_class_f1": rows, "requested_confusions": directions}
    return payload


def resolvable_counts(features: np.lib.npyio.NpzFile, a6: dict[str, np.lib.npyio.NpzFile]) -> dict:
    result = {}
    for split in ["public", "private"]:
        targets = features[f"{split}_targets"]
        require(np.array_equal(targets, a6[split]["targets"]), f"A6 {split} target order mismatch")
        v22_predictions = a6[split]["v22_R10"].argmax(axis=1)
        v23_predictions = features[f"{split}_R10"].argmax(axis=1)
        v22_correct = v22_predictions == targets
        v23_correct = v23_predictions == targets
        result[split] = {
            "v23_correct_v22_wrong": int((v23_correct & ~v22_correct).sum()),
            "v22_correct_v23_wrong": int((v22_correct & ~v23_correct).sum()),
            "both_correct": int((v22_correct & v23_correct).sum()),
            "both_wrong": int((~v22_correct & ~v23_correct).sum()),
            "net_resolved": int(v23_correct.sum() - v22_correct.sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=V23_ROOT / "official_runs" / "analysis_cache" / "v23_frozen_features.npz",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    require(args.batch_size > 0, "batch size must be positive")
    integrity = verify_frozen_run(run_dir)
    print(f"Frozen run verified: {integrity['run_id']}", flush=True)

    checkpoint = torch.load(run_dir / "best_val_acc.pt", map_location="cpu", weights_only=False)
    require(checkpoint.get("source_hash") == EXPECTED_SOURCE_SHA256, "Checkpoint source mismatch")
    require(int(checkpoint.get("epoch", -1)) == integrity["best_epoch"], "Checkpoint epoch mismatch")
    require(checkpoint.get("weights_type") == "EMA", "Checkpoint weights are not EMA")
    config = MPGConfig()
    model = MPGFER(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 2_304_528, "Parameter count mismatch")
    require(
        list(config.motif_residual_scale_schedule) == [0.5, 0.5, 1.0, 1.0, 1.0],
        "Residual-scale schedule mismatch",
    )
    require(list(config.motif_topk_schedule) == [8, 16, 16, 16, 24], "Top-K mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Analysis device: {device}; batch size: {args.batch_size}", flush=True)

    if not args.feature_cache.is_file():
        extract_features(model, device, args.batch_size, args.feature_cache, integrity)
    else:
        print(f"Using existing feature cache: {args.feature_cache}", flush=True)
    features = np.load(args.feature_cache)
    require(float(features["hook_max_logit_abs_diff"][0]) == 0.0, "Cached hook validation failed")
    for key in ["run_id", "source_sha256", "config_sha256", "best_checkpoint_sha256"]:
        require(str(features[key][0]) == integrity[key], f"Feature-cache {key} mismatch")

    a6 = {
        "train": np.load(A6_ROOT / "a6_train_features.npz"),
        "public": np.load(A6_ROOT / "a6_public_features.npz"),
        "private": np.load(A6_ROOT / "a6_private_features.npz"),
    }
    for split in ["train", "public", "private"]:
        require(
            np.array_equal(features[f"{split}_targets"], a6[split]["targets"]),
            f"A6 {split} target order mismatch",
        )

    a6_metrics = read_json(A6_ROOT / "a6_stage_probe_metrics.json")
    require(isinstance(a6_metrics, dict) and "v22" in a6_metrics, "A6 v2.2 probes missing")
    probes = {}
    for stage_name, stage_key in STAGES.items():
        print(f"Fitting fixed probe: {stage_name} ({stage_key})", flush=True)
        probes[stage_name] = evaluate_probe(stage_name, stage_key, features)
    probe_rows = []
    for stage_name, stage_key in STAGES.items():
        old, new = a6_metrics["v22"][stage_key], probes[stage_name]
        probe_rows.append({
            "stage": stage_name,
            "a6_stage": stage_key,
            "v22_train": old["train_accuracy"],
            "v22_public": old["public_accuracy"],
            "v22_train_public_gap": old["train_public_gap"],
            "v22_private": old["private_accuracy"],
            "v22_train_private_gap": old["train_private_gap"],
            "v23_train": new["train_accuracy"],
            "v23_public": new["public_accuracy"],
            "v23_train_public_gap": new["train_public_gap"],
            "v23_private": new["private_accuracy"],
            "v23_train_private_gap": new["train_private_gap"],
            "delta_public": new["public_accuracy"] - old["public_accuracy"],
            "delta_public_gap": new["train_public_gap"] - old["train_public_gap"],
            "delta_private": new["private_accuracy"] - old["private_accuracy"],
            "delta_private_gap": new["train_private_gap"] - old["train_private_gap"],
        })

    v21 = {
        "public": read_json(V21_RUN / "public_metrics.json"),
        "private": read_json(V21_RUN / "private_metrics.json"),
    }
    v22 = {
        "public": read_json(V22_RUN / "public_metrics.json"),
        "private": read_json(V22_RUN / "private_metrics.json"),
    }
    require(sha256_file(V22_RUN / "best_val_acc.pt") == EXPECTED_V22_CHECKPOINT_SHA256, "v2.2 checkpoint identity mismatch")
    v23 = {
        "public": read_json(run_dir / "public_metrics.json"),
        "private": read_json(run_dir / "private_metrics.json"),
    }
    resolvable = resolvable_counts(features, a6)
    canonical_fp32 = {
        split: raw_fp32_metrics(features, split)
        for split in ["train", "public", "private"]
    }

    history = read_json(run_dir / "history.json")
    require(isinstance(history, list) and history, "Training history is empty")
    best_history = next(
        (row for row in history if int(row["epoch"]) == integrity["best_epoch"]),
        None,
    )
    require(best_history is not None, "Best epoch is absent from history")
    training_summary = {
        "identity": integrity,
        "epoch_count": len(history),
        "first_epoch": int(history[0]["epoch"]),
        "last_epoch": int(history[-1]["epoch"]),
        "best_epoch": integrity["best_epoch"],
        "best_epoch_train_accuracy": best_history.get("train_accuracy"),
        "best_epoch_public_raw": best_history.get("val_raw"),
        "best_epoch_public_tta": best_history.get("val_tta"),
        "final_epoch_train_accuracy": history[-1].get("train_accuracy"),
        "trajectory": history,
    }
    best_metadata = dict(read_json(run_dir / "best_val_acc.json"))
    best_metadata["verified_integrity"] = integrity
    public_results = {
        "metric_provenance": "official Kaggle T4 frozen EMA checkpoint",
        "identity": integrity,
        **v23["public"],
        "canonical_local_fp32_raw_diagnostic": canonical_fp32["public"],
    }
    private_results = {
        "metric_provenance": "official Kaggle T4 one-shot post-freeze evaluation",
        "identity": integrity,
        **v23["private"],
        "canonical_local_fp32_raw_diagnostic": canonical_fp32["private"],
    }
    probe_output = {
        "protocol": {
            "inference": "raw single-view FP32",
            "fit_split": "Train only",
            "scaler": "StandardScaler fit on Train only",
            "classifier": "multinomial LogisticRegression",
            "C": 1.0,
            "penalty": "L2",
            "solver": "lbfgs",
            "max_iter": 5000,
            "tol": 1e-6,
            "hyperparameter_search": False,
        },
        "identity": integrity,
        "v23": probes,
        "v22_v23_depth_specialization": probe_rows,
    }
    comparison = {
        "official_kaggle_t4_metrics": {
            version: {
                split: official_summary(values[split])
                for split in ["public", "private"]
            }
            for version, values in [("v2.1", v21), ("v2.2", v22), ("v2.3", v23)]
        },
        "v23_minus_v22": {
            split: {
                view: {
                    metric: v23[split][view][metric] - v22[split][view][metric]
                    for metric in ["accuracy", "macro_f1", "loss"]
                }
                for view in ["raw", "tta"]
            }
            for split in ["public", "private"]
        },
        "probe_depth_specialization": probe_rows,
        "canonical_local_fp32_raw_diagnostics": {
            "label": "separate from official Kaggle mixed-precision metrics",
            "v23": canonical_fp32,
        },
        "v22_v23_model_resolvable_raw_fp32": resolvable,
    }

    write_json(V23_ROOT / "v23_training_history.json", training_summary)
    write_json(V23_ROOT / "v23_best_checkpoint_metadata.json", best_metadata)
    write_json(V23_ROOT / "v23_public_results.json", public_results)
    write_json(V23_ROOT / "v23_private_results.json", private_results)
    write_json(V23_ROOT / "v23_probe_analysis.json", probe_output)
    write_json(V23_ROOT / "v23_classwise_metrics.json", classwise_payload(v22, v23))
    write_json(V23_ROOT / "v23_comparison_v21_v22_v23.json", comparison)
    print("V23_FROZEN_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
