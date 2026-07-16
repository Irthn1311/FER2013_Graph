"""Plan or execute common post-training evaluation for OFIX18 paired C0/C2 seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "d18/scripts/evaluate_ofix18_factorial.py"
TRAINING_SEEDS = (7, 21, 42, 84, 123)
TOPOLOGY_SEEDS = (11, 23, 37, 53, 71)
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"


def run_name(cell: str, seed: int) -> str:
    part = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    return f"d18_ofix18_{part}_seed{seed}"


def run_dir(cell: str, seed: int, new_root: Path) -> Path:
    name = run_name(cell, seed)
    if seed == 42:
        return ROOT / "outputs/d18_runs/ofix18" / name
    return new_root / name


def command(
    run: Path,
    checkpoint: str,
    output: Path,
    prior: Path,
    cache: Path,
    device: str,
    modes: str,
    topology_seed: int,
    manifest: Path | None,
) -> list[str]:
    result = [
        sys.executable,
        "-B",
        str(EVALUATOR),
        "--run_dir",
        str(run),
        "--prior_dir",
        str(prior),
        "--graph_cache_dir",
        str(cache),
        "--checkpoint",
        checkpoint,
        "--output_dir",
        str(output),
        "--device",
        device,
        "--batch_size",
        "16",
        "--seed",
        str(topology_seed),
        "--counterfactual_modes",
        modes,
        "--skip_edge_ablations",
    ]
    if manifest is not None:
        result += ["--sample_manifest", str(manifest)]
    return result


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Evaluation failed with exit code {code}: {log_path}")


def verify_locked(output: Path) -> None:
    payload = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if int(payload["sample_count"]) != 715:
        raise RuntimeError(f"Locked sample count mismatch: {output}")
    if payload["sample_index_sha256"] != LOCKED_SAMPLE_SHA256:
        raise RuntimeError(f"Locked sample hash mismatch: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_run_root", default="outputs/d18_runs/ofix18_multiseed")
    parser.add_argument(
        "--prior_dir",
        default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue",
    )
    parser.add_argument(
        "--graph_cache_dir",
        default="outputs/d18_graph_cache/ofix17_structure_reg/base6_shared",
    )
    parser.add_argument(
        "--locked_manifest",
        default="outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_evaluation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    new_root = Path(args.new_run_root)
    prior = Path(args.prior_dir)
    cache = Path(args.graph_cache_dir)
    manifest = Path(args.locked_manifest)
    output_root = Path(args.output_dir)
    jobs: list[dict[str, Any]] = []
    missing_runs: list[str] = []

    if not EVALUATOR.exists():
        raise FileNotFoundError(EVALUATOR)
    if args.execute:
        for required in (prior, cache, manifest):
            if not required.exists():
                raise FileNotFoundError(required)

    for cell in ("C0", "C2"):
        for seed in TRAINING_SEEDS:
            source = run_dir(cell, seed, new_root)
            missing = [
                str(path)
                for path in (
                    source / "resolved_config.yaml",
                    source / "checkpoints/best.pt",
                    source / "checkpoints/last.pt",
                )
                if not path.exists()
            ]
            if missing:
                missing_runs.append(f"{cell}/seed{seed}: {missing}")
            for checkpoint in ("best", "last"):
                base = output_root / run_name(cell, seed) / checkpoint
                specs = [
                    (
                        "full_official",
                        None,
                        "official",
                        42,
                    ),
                    (
                        "locked_core",
                        manifest,
                        "official,remove_structure,shuffle_structure",
                        42,
                    ),
                ]
                specs.extend(
                    (
                        f"locked_topology_seed{topology_seed}",
                        manifest,
                        "permute_structure_destinations,degree_matched_random_structure",
                        topology_seed,
                    )
                    for topology_seed in TOPOLOGY_SEEDS
                )
                for label, sample_manifest, modes, topology_seed in specs:
                    destination = base / label
                    cmd = command(
                        source,
                        checkpoint,
                        destination,
                        prior,
                        cache,
                        args.device,
                        modes,
                        topology_seed,
                        sample_manifest,
                    )
                    jobs.append(
                        {
                            "cell": cell,
                            "training_seed": seed,
                            "checkpoint": checkpoint,
                            "evaluation": label,
                            "topology_seed": topology_seed,
                            "run_dir": str(source),
                            "output_dir": str(destination),
                            "command": cmd,
                        }
                    )

    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "READY" if not missing_runs else "WAITING_FOR_RUNS",
        "execute": bool(args.execute),
        "training_seeds": list(TRAINING_SEEDS),
        "topology_seeds": list(TOPOLOGY_SEEDS),
        "locked_sample_sha256": LOCKED_SAMPLE_SHA256,
        "job_count": len(jobs),
        "missing_runs": missing_runs,
        "jobs": jobs,
    }
    (output_root / "evaluation_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.execute:
        print(json.dumps({
            "status": plan["status"],
            "job_count": len(jobs),
            "missing_run_count": len(missing_runs),
            "plan": str(output_root / "evaluation_plan.json"),
            "note": "Dry-run only. Add --execute after all checkpoints exist.",
        }, indent=2))
        return
    if missing_runs:
        raise RuntimeError("Required run artifacts are missing:\n" + "\n".join(missing_runs))

    for index, job in enumerate(jobs, start=1):
        destination = Path(job["output_dir"])
        complete = destination / "AUDIT_COMPLETE.json"
        if complete.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip complete: {destination}")
            if job["evaluation"].startswith("locked_"):
                verify_locked(destination)
            continue
        if destination.exists() and any(destination.iterdir()) and not args.overwrite:
            raise RuntimeError(f"Partial evaluation output exists: {destination}")
        print(f"[{index}/{len(jobs)}] {job['cell']} seed{job['training_seed']} {job['checkpoint']} {job['evaluation']}")
        run_logged(job["command"], destination / "evaluation_console.log")
        if job["evaluation"].startswith("locked_"):
            verify_locked(destination)

    completion = {
        "status": "COMPLETE",
        "job_count": len(jobs),
        "locked_sample_sha256": LOCKED_SAMPLE_SHA256,
    }
    (output_root / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
