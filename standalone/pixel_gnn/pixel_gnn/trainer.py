"""Universal Training Pipeline for Pixel GNN Models."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from pixel_gnn.utils import (
    EarlyStopping,
    ReduceLROnPlateau,
    WarmupCosineDecay,
    build_optimizer,
    build_scheduler,
    load_config,
    seed_everything,
)

from pixel_gnn.batching import PixelBatchGenerator
from pixel_gnn.dataset import FERPixelDataset
from pixel_gnn.evaluator import evaluate_model
from pixel_gnn.losses import compute_total_loss
from pixel_gnn.models import build_model


def run_training(
    config_path: str | Path,
    fer_csv: str | Path,
    output_root: str | Path,
    limit_epochs: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
):
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    batch_size = int(config.get("training", {}).get("batch_size", 32))
    eval_batch_size = int(config.get("training", {}).get("eval_batch_size", batch_size))
    max_epochs = limit_epochs if limit_epochs is not None else int(config.get("training", {}).get("max_epochs", 90))

    model_name = config.get("model", {}).get("name", "pixel_neighbor_motif")
    loss_cfg = config.get("loss", {})
    lambda_diversity = float(loss_cfg.get("lambda_motif_diversity", 0.0))
    lambda_spatial_coherence = float(loss_cfg.get("lambda_spatial_coherence", 0.0))

    # Augmentation config (only active for train split)
    aug_cfg = config.get("augmentation", {})
    augment_train = bool(aug_cfg.get("enabled", False))
    flip_prob = float(aug_cfg.get("horizontal_flip_prob", 0.5))
    brightness_delta = float(aug_cfg.get("brightness_delta", 0.08))
    contrast_range = tuple(aug_cfg.get("contrast_range", [0.9, 1.1]))

    print("=" * 80)
    print(f"[INIT] Pixel GNN Universal Trainer | Model: {model_name}")
    print(f"       batch_size={batch_size} | max_epochs={max_epochs} | seed={seed}")
    print(f"       output_dir={output_dir}")
    if augment_train:
        print(f"       augmentation=ENABLED (flip_p={flip_prob}, brightness=±{brightness_delta}, contrast={contrast_range})")
    else:
        print(f"       augmentation=DISABLED")
    print("=" * 80, flush=True)

    # Initialize Datasets
    train_dataset = FERPixelDataset(fer_csv, "train")
    val_dataset = FERPixelDataset(fer_csv, "val")

    train_gen = PixelBatchGenerator(
        fer_csv,
        "train",
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        dataset=train_dataset,
        augment=augment_train,
        flip_prob=flip_prob,
        brightness_delta=brightness_delta,
        contrast_range=contrast_range,
    )
    val_gen = PixelBatchGenerator(
        fer_csv,
        "val",
        batch_size=eval_batch_size,
        seed=seed,
        shuffle=False,
        dataset=val_dataset,
        augment=False,
    )

    # Strategy setup: auto-detect 2 GPUs on Kaggle or multi-GPU environments
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        print(f"[DEVICE] Detected {len(gpus)} GPUs: {[g.name for g in gpus]}")
        print(f"[DEVICE] Activating MirroredStrategy with {num_replicas} replicas (both GPUs will be utilized)!", flush=True)
    else:
        strategy = tf.distribute.get_strategy()
        num_replicas = 1
        device_type = "1 GPU" if len(gpus) == 1 else "CPU"
        print(f"[DEVICE] Single device ({device_type}).", flush=True)

    # Instantiate model and optimizer inside strategy scope
    with strategy.scope():
        model = build_model(config)
        # Trigger build with one batch
        sample_batch = next(iter(train_gen.as_dataset(0, limit_batches=1)))
        _ = model(sample_batch, training=False)
        optimizer = build_optimizer(config)
        optimizer.build(model.trainable_variables)

    param_count = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"[MODEL] Build complete: {model.name}. Trainable parameters: {param_count:,}", flush=True)

    scheduler = build_scheduler(config, optimizer, max_epochs=max_epochs)
    sched_cfg = config.get("training", {}).get("scheduler", {})
    sched_type = sched_cfg.get("type", "plateau")
    print(f"[SCHEDULER] Initialized scheduler: {sched_type} (max_epochs={max_epochs})", flush=True)

    early_cfg = config.get("training", {}).get("early_stopping", {})
    early_stopping = EarlyStopping(
        min_epochs=early_cfg.get("min_epochs_before_stop", 30),
        patience=early_cfg.get("patience", 20),
    )
    print(f"[EARLY STOPPING] Configured: min_epochs={early_stopping.min_epochs}, patience={early_stopping.patience}", flush=True)

    def step_fn(batch):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss, metrics = compute_total_loss(
                batch["labels"],
                out,
                lambda_diversity=lambda_diversity,
                lambda_spatial_coherence=lambda_spatial_coherence,
            )
            scaled_loss = loss / float(num_replicas)
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        preds = tf.argmax(out["logits"], axis=-1, output_type=batch["labels"].dtype)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, batch["labels"]), tf.float32))
        batch_size_f = tf.cast(tf.shape(batch["labels"])[0], tf.float32)
        return loss, metrics["ce_loss"], correct, batch_size_f

    if num_replicas > 1:
        @tf.function
        def dist_train_step(dist_batch):
            per_replica_loss, per_replica_ce, per_replica_correct, per_replica_count = strategy.run(step_fn, args=(dist_batch,))
            total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None) / float(num_replicas)
            total_ce = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_ce, axis=None) / float(num_replicas)
            total_correct = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_correct, axis=None)
            total_count = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_count, axis=None)
            return total_loss, total_ce, total_correct, total_count
    else:
        @tf.function
        def dist_train_step(batch):
            return step_fn(batch)

    history = []
    best_val_acc = 0.0
    best_val_epoch = 0
    history_csv = output_dir / "training_history.csv"

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        scheduler.on_epoch_start(epoch)
        raw_train_ds = train_gen.as_dataset(epoch, limit_batches=limit_train_batches)
        if num_replicas > 1:
            train_ds = strategy.experimental_distribute_dataset(raw_train_ds)
        else:
            train_ds = raw_train_ds

        step_losses, step_ce = [], []
        train_correct, train_samples = 0.0, 0.0
        for batch in train_ds:
            loss_val, ce_val, correct_val, count_val = dist_train_step(batch)
            step_losses.append(float(loss_val.numpy()))
            step_ce.append(float(ce_val.numpy()))
            train_correct += float(correct_val.numpy())
            train_samples += float(count_val.numpy())

        train_loss = float(np.mean(step_losses))
        train_acc = float(train_correct / max(train_samples, 1.0))

        # Validation
        val_ds = val_gen.as_dataset(epoch, limit_batches=limit_val_batches)
        val_metrics = evaluate_model(
            model,
            val_ds,
            lambda_diversity=lambda_diversity,
            lambda_spatial_coherence=lambda_spatial_coherence,
            include_diagnostics=(epoch % 5 == 0 or epoch == max_epochs),
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        scheduler.on_epoch_end(epoch, val_loss)
        current_lr = float(optimizer.learning_rate.numpy()) if hasattr(optimizer.learning_rate, "numpy") else float(optimizer.learning_rate)

        saved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            model.save_weights(str(output_dir / "best_val_accuracy.weights.h5"))
            saved = True

        total_time = time.perf_counter() - t0

        print(
            f"[EPOCH {epoch:03d}/{max_epochs:03d}] "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | "
            f"val_macro_f1={val_macro_f1*100:.2f}% | "
            f"lr={current_lr:.6f} | time={total_time:.1f}s"
            f"{' [BEST SAVED]' if saved else ''}",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "epoch_time_sec": total_time,
            "best_accuracy": best_val_acc,
            "best_epoch": best_val_epoch,
        }
        history.append(epoch_record)

        with history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_record.keys()))
            writer.writeheader()
            writer.writerows(history)

        if early_stopping.step(val_loss, epoch):
            print(f"[EARLY STOPPING] Triggered at epoch {epoch} (patience={early_stopping.patience})", flush=True)
            break

    print("=" * 80)
    print(f"[COMPLETED] Best Validation Accuracy: {best_val_acc*100:.2f}% at epoch {best_val_epoch}")
    print("=" * 80, flush=True)

    # Load best checkpoint weights for final test evaluation
    best_weights_path = output_dir / "best_val_accuracy.weights.h5"
    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
        print(f"[CHECKPOINT] Loaded best weights from {best_weights_path} for final test evaluation.", flush=True)

    # Final Test Set Evaluation (3,589 images)
    print("\n" + "=" * 80)
    print("[TEST] Running final evaluation on TEST set...")
    print("=" * 80, flush=True)
    test_dataset = FERPixelDataset(fer_csv, "test")
    test_gen = PixelBatchGenerator(fer_csv, "test", batch_size=eval_batch_size, seed=seed, shuffle=False, dataset=test_dataset)
    test_ds = test_gen.as_dataset(0)

    test_metrics = evaluate_model(
        model,
        test_ds,
        lambda_diversity=lambda_diversity,
        lambda_spatial_coherence=lambda_spatial_coherence,
        include_diagnostics=True,
    )
    test_acc = test_metrics["accuracy"]
    test_macro_f1 = test_metrics["macro_f1"]
    test_loss = test_metrics["loss"]

    print("=" * 80)
    print(
        f"[TEST RESULT] Final Test Accuracy: {test_acc*100:.2f}% | "
        f"Test Macro F1: {test_macro_f1*100:.2f}% | "
        f"Test Loss: {test_loss:.4f}"
    )
    print("=" * 80, flush=True)

    cm = test_metrics.get("confusion_matrix")
    if cm is not None and hasattr(cm, "tolist"):
        cm = cm.tolist()
    per_class = test_metrics.get("per_class_f1")
    if per_class is not None and hasattr(per_class, "tolist"):
        per_class = per_class.tolist()

    test_metrics_to_save = {
        "model_name": model_name,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(best_val_acc),
        "best_val_epoch": int(best_val_epoch),
        "epochs_trained": len(history),
        "per_class_f1": per_class,
        "confusion_matrix": cm,
    }

    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics_to_save, indent=2), encoding="utf-8")

    if test_metrics.get("motif_diagnostics", {}).get("motif_enabled"):
        final_diag = test_metrics["motif_diagnostics"]
        diag_path = output_dir / "motif_diagnostics.json"
        serializable_diag = {
            "usage": final_diag["usage"].tolist() if hasattr(final_diag["usage"], "tolist") else list(final_diag["usage"]),
            "active_motifs": final_diag.get("active_motifs"),
            "motif_usage_min": final_diag.get("motif_usage_min"),
            "motif_usage_max": final_diag.get("motif_usage_max"),
            "motif_usage_std": final_diag.get("motif_usage_std"),
            "assignment_entropy": final_diag.get("assignment_entropy"),
            "top_nodes_per_motif": final_diag["top_nodes_per_motif"],
        }
        diag_path.write_text(json.dumps(serializable_diag, indent=2), encoding="utf-8")
        print(
            f"[MOTIF] Diagnostics: active={final_diag.get('active_motifs')} | "
            f"usage_min={final_diag.get('motif_usage_min'):.4f} | "
            f"usage_max={final_diag.get('motif_usage_max'):.4f} | "
            f"entropy={final_diag.get('assignment_entropy'):.3f}",
            flush=True,
        )
        print(f"[MOTIF] Diagnostics saved to {diag_path}", flush=True)

    print(f"[SUCCESS] All artifacts and test metrics saved to {output_dir}")
