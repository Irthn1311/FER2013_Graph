#!/usr/bin/env python3
"""Runner script for LAP-GNN training on Kaggle / Local.

Usage:
    python run_train.py
    python run_train.py --config standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_kaggle_2gpu.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Force line buffering for stdout/stderr so logs appear immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Add standalone package to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent
STANDALONE_SRC = REPO_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate" / "src"
if STANDALONE_SRC.exists() and str(STANDALONE_SRC) not in sys.path:
    sys.path.insert(0, str(STANDALONE_SRC))

DEFAULT_CONFIG = (
    REPO_ROOT
    / "standalone"
    / "lap_gnn_tensorflow_ofix7_mid_candidate"
    / "configs"
    / "fer2013_ofix7_mid_tensorflow_kaggle_2gpu.yaml"
)


def find_first_match(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        if "**" in pattern:
            prefix, _, suffix = pattern.partition("**")
            prefix = prefix.rstrip("/\\")
            suffix = suffix.lstrip("/\\")
            base = Path(prefix)
            if base.exists():
                for level in [
                    f"{prefix}/{suffix}",
                    f"{prefix}/*/{suffix}",
                    f"{prefix}/*/*/{suffix}",
                    f"{prefix}/*/*/*/{suffix}",
                ]:
                    matches = sorted(glob.glob(level))
                    if matches:
                        return Path(matches[0]).resolve()
        else:
            matches = sorted(glob.glob(pattern))
            if matches:
                return Path(matches[0]).resolve()
    return None


def resolve_path(cli_val: str | None, cfg_val: str | None, auto_patterns: list[str], name: str) -> Path:
    print(f"[RESOLVE] Locating {name}...", flush=True)
    if cli_val:
        p = Path(cli_val).resolve()
        if p.exists():
            print(f"[-] Using CLI path for {name}: {p}", flush=True)
            return p
        print(f"[WARN] CLI path for {name} does not exist: {p}", flush=True)

    if cfg_val:
        p = Path(cfg_val)
        if p.exists():
            print(f"[-] Found {name} from config: {p.resolve()}", flush=True)
            return p.resolve()

        # Handle common Kaggle mistake: /kaggle/input/datasets/<username>/<slug>/... -> /kaggle/input/<slug>/...
        path_str = str(cfg_val).replace("\\", "/")
        if "/kaggle/input/datasets/" in path_str:
            parts = [part for part in path_str.split("/") if part]
            try:
                ds_idx = parts.index("datasets")
                if len(parts) > ds_idx + 2:
                    normalized_parts = parts[:ds_idx] + parts[ds_idx + 2:]
                    normalized = Path("/" + "/".join(normalized_parts))
                    if normalized.exists():
                        print(f"[AUTO-DETECT] Normalized Kaggle path for {name}: {normalized.resolve()}", flush=True)
                        return normalized.resolve()
            except ValueError:
                pass

    found = find_first_match(auto_patterns)
    if found and found.exists():
        print(f"[AUTO-DETECT] Found {name}: {found}", flush=True)
        return found.resolve()

    if cfg_val:
        return Path(cfg_val)
    if cli_val:
        return Path(cli_val)
    raise FileNotFoundError(f"Cannot resolve path for {name}. Checked auto-patterns: {auto_patterns}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAP-GNN training with YAML config on Kaggle or local GPU.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument("--fer-csv", default=None, help="Path to FER2013 train.csv")
    parser.add_argument("--prior-root", default=None, help="Directory containing MediaPipe pixel priors")
    parser.add_argument("--cache-root", default=None, help="Directory containing clean graph cache")
    parser.add_argument("--output-root", default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--device", default="gpu", help="Device to use ('gpu' or 'cpu')")
    parser.add_argument("--allow-cpu-training", action="store_true", help="Allow training on CPU if GPU missing")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and paths without running training")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from lap_gnn_tf.config import load_config
    from lap_gnn_tf.resources import ResourceControls

    config = load_config(config_path)
    paths_cfg = config.get("paths", {})

    print("=" * 70)
    print(f"LAP-GNN Training Runner | Run Name: {config.get('run_name', 'unnamed')}")
    print(f"Config: {config_path}")
    print("=" * 70)

    # 1. Resolve paths
    fer_csv = resolve_path(
        args.fer_csv,
        paths_cfg.get("fer_csv"),
        [
            "/kaggle/input/**/train.csv",
            "/kaggle/input/**/fer2013.csv",
            "data/**/train.csv",
            "../data/**/train.csv",
        ],
        "FER CSV",
    )

    prior_root = resolve_path(
        args.prior_root,
        paths_cfg.get("prior_root"),
        [
            "/kaggle/input/**/d16_mediapipe_pixel_priors_best_retry_rescue",
            "/kaggle/input/**/d16_mediapipe_pixel_priors*",
            "priors/**",
        ],
        "MediaPipe Prior Root",
    )

    cache_root = None
    try:
        cache_root = resolve_path(
            args.cache_root,
            paths_cfg.get("graph_cache_dir"),
            [
                "/kaggle/input/**/ofix7-mid-seed42-records*",
                "/kaggle/input/**/graph_cache*",
            ],
            "Graph Cache",
        )
    except FileNotFoundError:
        print("[INFO] Clean graph cache not found; will build graphs on-the-fly.")

    output_root = args.output_root or paths_cfg.get("output_root")
    if not output_root:
        if Path("/kaggle/working").exists():
            output_root = f"/kaggle/working/outputs/{config.get('run_name', 'lap_gnn_run')}_{int(time.time())}"
        else:
            output_root = f"./outputs/{config.get('run_name', 'lap_gnn_run')}_{int(time.time())}"
    output_dir = Path(output_root).resolve()

    # If directory exists and is not empty, add timestamp suffix
    if output_dir.exists() and any(output_dir.iterdir()):
        output_dir = Path(f"{output_dir}_{int(time.time())}")

    print(f"[-] Resolved train.csv:   {fer_csv}")
    print(f"[-] Resolved prior_root:  {prior_root}")
    print(f"[-] Resolved cache_root:  {cache_root}")
    print(f"[-] Output directory:     {output_dir}")

    # 2. Check GPUs
    import tensorflow as tf

    physical_gpus = tf.config.list_physical_devices("GPU")
    print(f"[-] Physical GPUs detected: {len(physical_gpus)}")
    for i, gpu in enumerate(physical_gpus):
        print(f"    GPU {i}: {gpu.name}")

    if not physical_gpus and not args.allow_cpu_training and args.device.lower().startswith("gpu"):
        raise RuntimeError("No GPU detected! Set --allow-cpu-training to test on CPU, or select GPU on Kaggle.")

    # Apply memory growth to all GPUs
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[WARN] Memory growth error: {e}")

    # 3. Setup ResourceControls
    res_cfg = config.get("resources", {})
    data_cfg = config.get("data", {})
    batch_size = int(res_cfg.get("batch_size") or data_cfg.get("batch_size", 32))
    eval_batch_size = int(res_cfg.get("eval_batch_size", 64))

    controls = ResourceControls(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        graph_workers=int(res_cfg.get("graph_workers", 4)),
        tf_data_prefetch=int(res_cfg.get("tf_data_prefetch", 4)),
        graph_cache_size=int(res_cfg.get("graph_cache_size", 64)),
        clean_graph_cache_dir=str(cache_root) if cache_root else None,
        memory_growth=bool(res_cfg.get("memory_growth", True)),
        mixed_precision=bool(res_cfg.get("mixed_precision", True)),
        xla=bool(res_cfg.get("xla", False)),
        device=args.device,
    )

    print(f"[-] Global batch size:    {controls.batch_size} (distributed across available GPUs)")
    print(f"[-] Eval batch size:      {controls.eval_batch_size}")
    print(f"[-] Mixed precision:      {controls.mixed_precision}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Preflight checks passed! Exiting without training.")
        return

    from lap_gnn_tf.training.trainer import run_training

    print("[START] Launching LAP-GNN training pipeline...")
    result = run_training(
        config_path=str(config_path),
        fer_csv=str(fer_csv),
        prior_root=str(prior_root),
        output_root=str(output_dir),
        controls=controls,
        no_resume=True,
    )
    print("=" * 70)
    print(f"[SUCCESS] Training finished! Artifacts saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
