"""Generate the Issue #99 canonical multi-seed Kaggle notebook.

The frozen scientific package and the reviewed seed-42 notebook remain
unchanged.  This generator derives one operational template from that notebook
and adds only seed injection, provenance, frozen FP32 evaluation, and paper
artifact packaging.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR = PROJECT_ROOT / "tools" / "sync_notebook.py"
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb"
)


def _load_base_generator():
    spec = importlib.util.spec_from_file_location("v23_seed42_notebook", BASE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base notebook generator: {BASE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one replacement target, found {count}: {old!r}")
    return source.replace(old, new)


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def build_notebook() -> dict:
    base = _load_base_generator()
    notebook = base.build_notebook()
    notebook["cells"][0]["source"] = (
        "# MPG-FER v2.3 — canonical official multi-seed Kaggle T4 run\n\n"
        "Generated for GitHub Issue #99 from the reviewed Issue #97 seed-42 "
        "notebook. The embedded scientific package is byte-identical; only the "
        "numeric seed and operational run metadata vary. Raw single-view is "
        "primary, flip-TTA is secondary, and PrivateTest remains locked until "
        "training completion and checkpoint freeze.\n"
    ).splitlines(True)

    notebook["cells"][1]["source"] = '''# STAGING TOOL EDITS ONLY THESE EXECUTION VALUES.
SEED = 42
ACCOUNT_ALIAS = "UNASSIGNED"
RESUME_MODE = "fresh"       # "fresh", "auto", or "required"
RESUME_PATH = None           # optional explicit /kaggle/input/.../resume_latest.pt
SEGMENT_NUMBER = 1
OUTPUT_DIR = f"/kaggle/working/outputs/mpg_v2_3_seed_{SEED}"
KERNEL_REF = "owner/mpg-fer-v2-3-seed-42-seg-01"
GIT_BRANCH = "research/mpg-fer-v2-3-early-depth-generalization"
GIT_COMMIT_SHA = "SET_BY_STAGING"
ARCHITECTURE_COMMIT_SHA = "08faea291ef425c10cc11ab0bc880e6cef302e97"
DIRTY_WORKTREE = False
EXPECTED_SOURCE_SHA = "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
EXPECTED_CONFIG_SHA = "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2"
EXPECTED_PARAMETERS = 2_304_528
'''.splitlines(True)

    environment_cell = "".join(notebook["cells"][3]["source"])
    environment_cell = _replace_once(
        environment_cell,
        "output_dir = Path(OUTPUT_DIR)\noutput_dir.mkdir(parents=True, exist_ok=True)\n",
        "run_started_at = datetime.now(timezone.utc).isoformat()\n"
        "run_started_monotonic = time.monotonic()\n"
        "output_dir = Path(OUTPUT_DIR)\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n",
    )
    environment_cell = _replace_once(
        environment_cell,
        "cfg = MPGConfig(segment_number=SEGMENT_NUMBER, output_dir=str(output_dir), resume_path=RESUME_PATH)\n"
        "validate_official_batch_contract(cfg)\n",
        "cfg = MPGConfig(seed=SEED, segment_number=SEGMENT_NUMBER, "
        "output_dir=str(output_dir), resume_path=RESUME_PATH)\n"
        "validate_official_batch_contract(cfg)\n"
        "resolved_config_sha = config_hash(cfg)\n"
        "if resolved_config_sha != EXPECTED_CONFIG_SHA:\n"
        "    raise RuntimeError(\n"
        "        f\"CONFIG_HASH_MISMATCH: expected {EXPECTED_CONFIG_SHA}, \"\n"
        "        f\"got {resolved_config_sha}\"\n"
        "    )\n",
    )
    environment_cell = _replace_once(
        environment_cell,
        "cuda_available = torch.cuda.is_available()\n",
        "set_seed(cfg.seed)\n"
        "cuda_available = torch.cuda.is_available()\n",
    )
    environment_cell = _replace_once(
        environment_cell,
        '    "torch_version": torch.__version__,\n}\n',
        '    "torch_version": torch.__version__,\n'
        '    "seed": cfg.seed,\n'
        '    "python_version": sys.version,\n'
        '    "cudnn_deterministic": torch.backends.cudnn.deterministic,\n'
        '    "cudnn_benchmark": torch.backends.cudnn.benchmark,\n'
        '    "account_alias": ACCOUNT_ALIAS,\n'
        '    "kernel_ref": KERNEL_REF,\n'
        '}\n'
        'atomic_json(output_dir / "environment.json", environment)\n',
    )
    environment_cell = _replace_once(
        environment_cell,
        '    "account": ACCOUNT, "kernel_ref": KERNEL_REF, "git_commit": GIT_COMMIT_SHA,\n'
        '    "segment_number": SEGMENT_NUMBER, "source_sha256": reviewed_source_sha,\n',
        '    "account_alias": ACCOUNT_ALIAS, "kernel_ref": KERNEL_REF,\n'
        '    "git_commit": GIT_COMMIT_SHA, "architecture_commit": ARCHITECTURE_COMMIT_SHA,\n'
        '    "dirty_worktree": DIRTY_WORKTREE, "seed": cfg.seed,\n'
        '    "segment_number": SEGMENT_NUMBER, "source_sha256": reviewed_source_sha,\n'
        '    "config_sha256": resolved_config_sha,\n',
    )
    notebook["cells"][3]["source"] = environment_cell.splitlines(True)

    final_cell = "".join(notebook["cells"][6]["source"])
    final_cell = _replace_once(
        final_cell,
        'archive = shutil.make_archive("/kaggle/working/mpg_fer_v2_3_artifacts", "zip", root_dir=output_dir)\n'
        'print(f"Artifacts ZIP: {archive}")\n',
        'print("Base finalization complete; Issue #99 packaging follows.")\n',
    )
    notebook["cells"][6]["source"] = final_cell.splitlines(True)

    notebook["cells"].append(_code(r'''# Issue #99 provenance, canonical FP32 evaluation, and paper artifact package.
import csv
import platform
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def split_metric_document(metrics, split_name, view, precision):
    return {
        "split": split_name,
        "view": view,
        "evaluation_precision": precision,
        "weights_type": "EMA",
        **metrics[view],
    }

def canonical_fp32_evaluate(csv_path, role, split_name):
    dataset = FER2013Dataset(csv_path, split=role, augment=False)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    checkpoint = torch.load(
        output_dir / "best_val_acc.pt", map_location=device, weights_only=False
    )
    model = MPGFER(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    raw_logits_all, tta_logits_all, targets_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            raw_logits, _ = model(images)
            flip_logits, _ = model(TF.hflip(images))
            raw_logits_all.append(raw_logits.float().cpu())
            tta_logits_all.append((0.5 * (raw_logits + flip_logits)).float().cpu())
            targets_all.append(targets.long().cpu())
    raw_logits = torch.cat(raw_logits_all).numpy().astype(np.float64)
    tta_logits = torch.cat(tta_logits_all).numpy().astype(np.float64)
    targets = torch.cat(targets_all).numpy().astype(np.int64)

    def metrics_for(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1)
        nll = float(
            (-shifted[np.arange(len(targets)), targets]
             + np.log(np.exp(shifted).sum(axis=1))).mean()
        )
        return {
            "loss": nll,
            "accuracy": float(np.mean(predictions == targets)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "per_class_f1": [float(x) for x in f1_score(
                targets, predictions, labels=np.arange(7), average=None, zero_division=0
            )],
            "precision": [float(x) for x in precision_score(
                targets, predictions, labels=np.arange(7), average=None, zero_division=0
            )],
            "recall": [float(x) for x in recall_score(
                targets, predictions, labels=np.arange(7), average=None, zero_division=0
            )],
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=np.arange(7)
            ).tolist(),
            "support": [int(np.sum(targets == label)) for label in range(7)],
            "predictions": predictions,
            "probabilities": probabilities,
        }

    raw = metrics_for(raw_logits)
    tta = metrics_for(tta_logits)
    predictions = [
        {
            "row_index": index,
            "target": int(targets[index]),
            "prediction": int(raw["predictions"][index]),
            "confidence": float(raw["probabilities"][index].max()),
        }
        for index in range(len(targets))
    ]
    for result in (raw, tta):
        result.pop("predictions")
        result.pop("probabilities")
    return {"split": split_name, "raw": raw, "tta": tta}, predictions

evaluation_dir = output_dir / "evaluation"
logs_dir = output_dir / "logs"
hashes_dir = output_dir / "hashes"
for directory in (evaluation_dir, logs_dir, hashes_dir):
    directory.mkdir(parents=True, exist_ok=True)

atomic_json(output_dir / "config_resolved.json", asdict(cfg))
environment.update({
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "cudnn_version": torch.backends.cudnn.version(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
})
atomic_json(output_dir / "environment.json", environment)
(output_dir / "command.txt").write_text(
    f"Kaggle kernel={KERNEL_REF}\nseed={SEED}\nsegment={SEGMENT_NUMBER}\n"
    f"resume_mode={RESUME_MODE}\naccount_alias={ACCOUNT_ALIAS}\n",
    encoding="utf-8",
)

dataset_files = {
    "train.csv": {"role": "Train", "rows": dataset_gate["train"]["rows"],
                  "sha256": sha256_file(train_csv)},
    "val.csv": {"role": "PublicTest", "rows": dataset_gate["public"]["rows"],
                "sha256": sha256_file(val_csv)},
    "test.csv": {"role": "PrivateTest", "rows": dataset_gate["private"]["rows"],
                 "sha256": sha256_file(test_csv)},
}
dataset_payload = json.dumps(dataset_files, sort_keys=True, separators=(",", ":"))
dataset_manifest_sha = hashlib.sha256(dataset_payload.encode("utf-8")).hexdigest()
atomic_json(output_dir / "dataset_manifest.json", {
    "dataset": "doduyquynii/fer13-split",
    "files": dataset_files,
    "dataset_manifest_sha256": dataset_manifest_sha,
    "private_content_used_for_selection": False,
})

if (output_dir / "history.csv").is_file():
    shutil.copy2(output_dir / "history.csv", logs_dir / "history.csv")
if (output_dir / "history.json").is_file():
    history_rows = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    with (logs_dir / "train.log").open("w", encoding="utf-8") as handle:
        for row in history_rows:
            handle.write(json.dumps({
                "epoch": row["epoch"], "lr": row["lr"],
                "train_loss": row["train_loss"],
                "train_accuracy": row["train_accuracy"],
                "public_raw_accuracy": row["val_raw"]["accuracy"],
                "public_tta_accuracy": row["val_tta"]["accuracy"],
                "early_stop_patience": row["early_stop_patience"],
            }) + "\n")

execution = json.loads((output_dir / "execution_manifest.json").read_text(encoding="utf-8"))
official_public = None
official_private = None
canonical_public = None
canonical_private = None

if execution["status"] == "TRAINING_COMPLETED":
    if not execution.get("PRIVATE_EVALUATED") or not execution.get(
        "private_evaluated_only_after_freeze"
    ):
        raise RuntimeError("FP32 evaluation refused: Private freeze firewall is absent")
    selection = json.loads(
        (output_dir / "final_selection_manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(output_dir / "best_val_acc.pt") != selection["checkpoint_sha256"]:
        raise RuntimeError("FP32 evaluation refused: frozen checkpoint SHA mismatch")
    official_public = json.loads((output_dir / "public_metrics.json").read_text(encoding="utf-8"))
    official_private = json.loads((output_dir / "private_metrics.json").read_text(encoding="utf-8"))
    for split_name, metrics in (("PublicTest", official_public), ("PrivateTest", official_private)):
        prefix = "public" if split_name == "PublicTest" else "private"
        for view in ("raw", "tta"):
            atomic_json(
                evaluation_dir / f"{prefix}_{view}.json",
                split_metric_document(metrics, split_name, view, "official_runtime_amp"),
            )

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    canonical_public, _ = canonical_fp32_evaluate(val_csv, "val", "PublicTest")
    canonical_private, private_predictions = canonical_fp32_evaluate(
        test_csv, "test", "PrivateTest"
    )
    torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    for split_name, metrics in (("public", canonical_public), ("private", canonical_private)):
        for view in ("raw", "tta"):
            atomic_json(
                evaluation_dir / f"{split_name}_{view}_canonical_fp32.json",
                split_metric_document(
                    metrics, metrics["split"], view, "canonical_fp32_no_autocast_tf32_disabled"
                ),
            )
    write_csv(
        evaluation_dir / "private_raw_predictions.csv",
        ["row_index", "target", "prediction", "confidence"],
        private_predictions,
    )
    write_csv(
        evaluation_dir / "confusion_matrix_raw.csv",
        ["true_class", *CLASS_NAMES],
        [
            {"true_class": CLASS_NAMES[index], **{
                CLASS_NAMES[column]: canonical_private["raw"]["confusion_matrix"][index][column]
                for column in range(7)
            }}
            for index in range(7)
        ],
    )
    write_csv(
        evaluation_dir / "per_class_raw.csv",
        ["class", "precision", "recall", "f1", "support"],
        [
            {
                "class": CLASS_NAMES[index],
                "precision": canonical_private["raw"]["precision"][index],
                "recall": canonical_private["raw"]["recall"][index],
                "f1": canonical_private["raw"]["per_class_f1"][index],
                "support": canonical_private["raw"]["support"][index],
            }
            for index in range(7)
        ],
    )

run_ended_at = datetime.now(timezone.utc).isoformat()
run_manifest = {
    "method": "MPG-FER", "version": "2.3", "seed": cfg.seed,
    "git_branch": GIT_BRANCH, "git_commit": GIT_COMMIT_SHA,
    "architecture_commit": ARCHITECTURE_COMMIT_SHA,
    "dirty_worktree": DIRTY_WORKTREE,
    "source_sha256": reviewed_source_sha,
    "config_file": "config_resolved.json", "config_sha256": resolved_config_sha,
    "dataset_manifest_sha256": dataset_manifest_sha,
    "account_alias": ACCOUNT_ALIAS, "kaggle_kernel_slug": KERNEL_REF,
    "kaggle_run_url": f"https://www.kaggle.com/code/{KERNEL_REF}",
    "segment_number": SEGMENT_NUMBER,
    "gpu": environment["gpu_name"], "pytorch_version": environment["torch_version"],
    "cuda_version": environment["cuda_version"],
    "start_time": run_started_at, "end_time": run_ended_at,
    "runtime_seconds": time.monotonic() - run_started_monotonic,
    "selected_epoch": execution.get("best_epoch"),
    "checkpoint_selection_metric": "EMA Public flip-TTA accuracy",
    "checkpoint_sha256": execution.get("best_checkpoint_sha256"),
    "public_raw_accuracy": None if official_public is None else official_public["raw"]["accuracy"],
    "public_raw_macro_f1": None if official_public is None else official_public["raw"]["macro_f1"],
    "private_raw_accuracy": None if official_private is None else official_private["raw"]["accuracy"],
    "private_raw_macro_f1": None if official_private is None else official_private["raw"]["macro_f1"],
    "public_tta_accuracy": None if official_public is None else official_public["tta"]["accuracy"],
    "public_tta_macro_f1": None if official_public is None else official_public["tta"]["macro_f1"],
    "private_tta_accuracy": None if official_private is None else official_private["tta"]["accuracy"],
    "private_tta_macro_f1": None if official_private is None else official_private["tta"]["macro_f1"],
    "canonical_fp32": None if canonical_public is None else {
        "public_raw": canonical_public["raw"], "public_tta": canonical_public["tta"],
        "private_raw": canonical_private["raw"], "private_tta": canonical_private["tta"],
    },
    "run_status": execution["status"],
    "private_evaluated_only_after_freeze": bool(
        execution.get("private_evaluated_only_after_freeze", False)
    ),
    "artifact_mapping": {
        "checkpoints/best_val_acc.pt": "best_val_acc.pt",
        "checkpoints/checkpoint_metadata.json": "best_val_acc.json",
        "logs/history.csv": "logs/history.csv",
        "official_runtime_metrics": "evaluation/{public,private}_{raw,tta}.json",
        "canonical_fp32_metrics": "evaluation/*_canonical_fp32.json",
    },
}
atomic_json(output_dir / "RUN_MANIFEST.json", run_manifest)

hash_rows = []
for path in sorted(output_dir.rglob("*")):
    if path.is_file() and path != hashes_dir / "SHA256SUMS.txt":
        hash_rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
(hashes_dir / "SHA256SUMS.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")

archive_base = f"/kaggle/working/mpg_v2_3_seed_{SEED}_artifacts"
archive = shutil.make_archive(archive_base, "zip", root_dir=output_dir)
print(json.dumps({
    "status": execution["status"], "seed": SEED, "segment": SEGMENT_NUMBER,
    "run_id": execution["run_id"], "archive": archive,
    "private_evaluated_only_after_freeze": run_manifest[
        "private_evaluated_only_after_freeze"
    ],
}, indent=2))
'''))

    notebook["metadata"]["mpg_fer_v2_3_issue"] = 99
    notebook["metadata"]["mpg_fer_v2_3_base_issue"] = 97
    notebook["metadata"]["mpg_fer_notebook_generator"] = (
        "tools/sync_multiseed_notebook.py"
    )
    notebook["metadata"]["mpg_fer_multiseed_contract"] = {
        "seeds": [0, 1, 42, 43, 123, 3047],
        "primary_metric": "raw_single_view",
        "secondary_metric": "horizontal_flip_tta",
        "scientific_source_changed": False,
    }
    return notebook


if __name__ == "__main__":
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {NOTEBOOK_PATH}")
