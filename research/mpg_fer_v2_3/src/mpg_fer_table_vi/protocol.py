"""Fail-closed protocol utilities for the MPG-FER Table VI ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import ABLATION_REGISTRY, TABLE_VI_ORDER, AblationMode
from mpg_fer_v2_3.checkpoint import sha256_file
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import validate_split_path


BASE_COMMIT = "232e7a9f09251e7c3353684d34351356bd2b023b"
ISSUE_URL = "https://github.com/Irthn1311/FER2013_Graph/issues/101"
TABLE_INFERENCE = "raw_single_view_fp32"
CHECKPOINT_SELECTION = (
    "EMA Public horizontal-flip TTA accuracy, then higher Macro-F1, "
    "then lower CE loss"
)
REQUIRED_RUN_ARTIFACTS = (
    "config.json",
    "ablation_manifest.json",
    "history.json",
    "history.csv",
    "best_val_acc.pt",
    "execution_manifest.json",
    "canonical_public_metrics.json",
    "segment_manifest.json",
    "resume_latest.pt",
    "checksums.sha256",
)


def ablation_source_tree_hash(source_root: str | Path | None = None) -> str:
    """Hash the frozen v2.3 package plus the additive Table VI package."""
    root = (
        Path(source_root)
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    digest = hashlib.sha256()
    for package_name in ("mpg_fer_v2_3", "mpg_fer_table_vi"):
        package = root / package_name
        for path in sorted(package.glob("*.py")):
            relative = f"{package_name}/{path.name}"
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass
class AblationConfig(MPGConfig):
    """The frozen v2.3 config plus resume-identifying ablation fields."""

    ablation_mode: str = AblationMode.FULL.value
    final_recipe_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ablation_mode = AblationMode(self.ablation_mode).value
        if self.seed != 42:
            raise ValueError("Table VI ablation seed is frozen at 42")


def _contains_private_marker(path: Path) -> bool:
    markers = {"test.csv", "privatetest", "private_test", "private-test"}
    return any(part.lower() in markers for part in path.parts)


def validate_ablation_data_paths(
    train_csv: str | Path, public_csv: str | Path
) -> tuple[Path, Path]:
    """Accept only explicit Train and PublicTest roles; PrivateTest is impossible."""
    candidates = (Path(train_csv), Path(public_csv))
    if any(_contains_private_marker(path) for path in candidates):
        raise RuntimeError("PRIVATE_FIREWALL: PrivateTest/test.csv is forbidden")
    train = validate_split_path(candidates[0], "train")
    public = validate_split_path(candidates[1], "val")
    if train.resolve() == public.resolve():
        raise RuntimeError("PRIVATE_FIREWALL: ambiguous or duplicated split role")
    return train, public


def validate_public_role(
    dataset_role: str, source_path: str | Path | None = None
) -> None:
    if dataset_role != "PublicTest":
        raise RuntimeError(
            "PRIVATE_FIREWALL: canonical evaluator requires explicit PublicTest role"
        )
    if source_path is not None:
        path = Path(source_path)
        if _contains_private_marker(path) or path.name.lower() != "val.csv":
            raise RuntimeError(
                "PRIVATE_FIREWALL: canonical evaluator accepts only val.csv as PublicTest"
            )


def _finite_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return [_finite_json(item) for item in value.detach().cpu().tolist()]
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Infinity are forbidden in scientific JSON")
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_finite_json(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _recipe_fields(config: AblationConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler": "linear_warmup_cosine_then_floor",
        "warmup_epochs": config.warmup_epochs,
        "lr_decay_end_epoch": config.lr_decay_end_epoch,
        "min_learning_rate": config.min_learning_rate,
        "max_epochs": config.max_epochs,
        "min_epochs": config.min_epochs,
        "ema_decay": config.ema_decay,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "checkpoint_selection": CHECKPOINT_SELECTION,
    }


def validate_final_recipe_lock(
    recipe_path: str | Path,
    design_lock_path: str | Path,
    config: AblationConfig,
) -> dict[str, Any]:
    """Require the later reviewed recipe and its SHA-bound design authorization."""
    recipe_file = Path(recipe_path)
    design_file = Path(design_lock_path)
    if not recipe_file.is_file():
        raise RuntimeError("SCIENTIFIC_TRAINING_REFUSED: FINAL_RECIPE_LOCK.json absent")
    if not design_file.is_file():
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_REFUSED: ABLATION_DESIGN_LOCK.json absent"
        )
    recipe_sha = sha256_file(recipe_file)
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    design = json.loads(design_file.read_text(encoding="utf-8"))
    bound_sha = design.get("final_recipe_lock_sha256")
    if not bound_sha or bound_sha != recipe_sha:
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_REFUSED: final recipe SHA is absent or mismatched"
        )
    if config.final_recipe_lock_sha256 != recipe_sha:
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_REFUSED: config does not bind the final recipe SHA"
        )
    if design.get("scientific_training_authorized") is not True:
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_REFUSED: independent authorization is not frozen"
        )
    if (
        recipe.get("method") != "MPG-FER"
        or recipe.get("private_test_permitted") is not False
    ):
        raise RuntimeError("SCIENTIFIC_TRAINING_REFUSED: recipe provenance mismatch")
    actual = recipe.get("training_recipe")
    expected = _recipe_fields(config)
    if actual != expected:
        raise RuntimeError(
            "SCIENTIFIC_TRAINING_REFUSED: recipe values do not match resolved config"
        )
    return {"path": str(recipe_file.resolve()), "sha256": recipe_sha, "payload": recipe}


def config_from_final_recipe(
    recipe_path: str | Path,
    *,
    mode: AblationMode | str,
    run_id: str,
    output_dir: str | Path,
    resume_path: str | Path | None = None,
    segment_number: int = 1,
) -> AblationConfig:
    """Resolve optimization fields only from the separately frozen recipe."""
    recipe_file = Path(recipe_path)
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    values = recipe.get("training_recipe")
    if not isinstance(values, dict):
        raise RuntimeError("FINAL_RECIPE_LOCK.json lacks training_recipe")
    field_map = {
        "seed": "seed",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_epochs": "warmup_epochs",
        "lr_decay_end_epoch": "lr_decay_end_epoch",
        "min_learning_rate": "min_learning_rate",
        "max_epochs": "max_epochs",
        "min_epochs": "min_epochs",
        "ema_decay": "ema_decay",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
    }
    missing = [name for name in field_map if name not in values]
    if missing:
        raise RuntimeError(f"FINAL_RECIPE_LOCK.json lacks fields: {missing}")
    kwargs = {target: values[source] for source, target in field_map.items()}
    return AblationConfig(
        **kwargs,
        ablation_mode=AblationMode(mode).value,
        final_recipe_lock_sha256=sha256_file(recipe_file),
        run_id=run_id,
        output_dir=str(output_dir),
        resume_path=None if resume_path is None else str(resume_path),
        segment_number=segment_number,
    )


@torch.no_grad()
def evaluate_canonical_public_fp32(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    *,
    dataset_role: str,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Canonical raw, single-view, FP32 PublicTest evaluator for Table VI."""
    validate_public_role(dataset_role, source_path)
    model.eval()
    resolved = torch.device(device)
    predictions: list[int] = []
    targets_all: list[int] = []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for images, targets in dataloader:
            images = images.to(resolved, dtype=torch.float32)
            targets = targets.to(resolved)
            with torch.amp.autocast("cuda", enabled=False):
                logits, _ = model(images)
                loss = criterion(logits.float(), targets)
            count = len(targets)
            loss_total += float(loss) * count
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets_all.extend(targets.cpu().tolist())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn
    if not targets_all:
        raise ValueError("Cannot evaluate an empty PublicTest dataloader")
    truth = np.asarray(targets_all, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    return {
        "dataset_role": "PublicTest",
        "inference": TABLE_INFERENCE,
        "autocast_enabled": False,
        "tf32_enabled": False,
        "single_view": True,
        "metrics": {
            "loss": loss_total / len(truth),
            "accuracy": float(np.mean(truth == predicted)),
            "macro_f1": float(
                f1_score(truth, predicted, average="macro", zero_division=0)
            ),
        },
        "sample_count": int(len(truth)),
    }


def write_ablation_manifest(
    output_dir: str | Path,
    config: AblationConfig,
    *,
    source_sha256: str,
    recipe_sha256: str,
) -> Path:
    spec = ABLATION_REGISTRY[AblationMode(config.ablation_mode)]
    return write_json(
        Path(output_dir) / "ablation_manifest.json",
        {
            "schema_version": 1,
            "issue": ISSUE_URL,
            "method": "MPG-FER",
            "seed": config.seed,
            "ablation_mode": spec.internal_id,
            "paper_name": spec.paper_name,
            "scientific_question": spec.scientific_question,
            "exact_intervention": spec.exact_intervention,
            "active_modules": spec.active_modules,
            "applicable_losses": spec.applicable_losses,
            "applicable_diagnostics": spec.applicable_diagnostics,
            "expected_tensor_invariants": spec.expected_tensor_invariants,
            "ablation_split": "PublicTest",
            "table_inference": TABLE_INFERENCE,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "private_test_permitted": False,
            "source_sha256": source_sha256,
            "final_recipe_lock_sha256": recipe_sha256,
        },
    )


def write_checksums(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    missing = [
        name for name in REQUIRED_RUN_ARTIFACTS[:-1] if not (root / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Cannot finalize checksums; missing artifacts: {missing}")
    lines = [
        f"{sha256_file(root / name)}  {name}" for name in REQUIRED_RUN_ARTIFACTS[:-1]
    ]
    target = root / "checksums.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def aggregate_table_vi(
    run_directories: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Aggregate canonical metrics in preregistered semantic order only."""
    by_mode: dict[AblationMode, dict[str, Any]] = {}
    for raw_root in run_directories:
        root = Path(raw_root)
        manifest = json.loads(
            (root / "ablation_manifest.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (root / "canonical_public_metrics.json").read_text(encoding="utf-8")
        )
        mode = AblationMode(manifest["ablation_mode"])
        if mode in by_mode:
            raise RuntimeError(f"Duplicate Table VI mode: {mode.value}")
        if (
            manifest.get("seed") != 42
            or manifest.get("private_test_permitted") is not False
        ):
            raise RuntimeError(f"Invalid scientific manifest for {mode.value}")
        if (
            metrics.get("inference") != TABLE_INFERENCE
            or metrics.get("dataset_role") != "PublicTest"
        ):
            raise RuntimeError(f"Non-canonical metrics for {mode.value}")
        by_mode[mode] = metrics["metrics"]
    missing = [mode.value for mode in TABLE_VI_ORDER if mode not in by_mode]
    if missing:
        raise RuntimeError(f"Incomplete Table VI inputs: {missing}")
    rows = []
    for mode in TABLE_VI_ORDER:
        metrics = by_mode[mode]
        rows.append(
            {
                "Configuration": ABLATION_REGISTRY[mode].paper_name,
                "Acc. (%)": 100.0 * float(metrics["accuracy"]),
                "Macro-F1 (%)": 100.0 * float(metrics["macro_f1"]),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "seed": 42,
        "inference": TABLE_INFERENCE,
        "row_order": [mode.value for mode in TABLE_VI_ORDER],
        "rows": rows,
    }
    write_json(output / "table_vi.json", document)
    with (output / "table_vi.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Configuration", "Acc. (%)", "Macro-F1 (%)"]
        )
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| Configuration | Acc. (%) | Macro-F1 (%) |",
        "|---|---:|---:|",
        *[
            f"| {row['Configuration']} | {row['Acc. (%)']:.2f} | {row['Macro-F1 (%)']:.2f} |"
            for row in rows
        ],
    ]
    (output / "table_vi.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return document


def resolved_config_document(config: AblationConfig) -> dict[str, Any]:
    return asdict(config)
