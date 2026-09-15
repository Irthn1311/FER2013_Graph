"""Kaggle map-job entry point for fixed PGM M0 L/M/G fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
SCIENTIFIC_SHA = "8910e9757210674a1504837c5b8bc66e2da9208d"
SEEDS = (42, 43, 44, 45, 46)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_and_test() -> tuple[Path, str]:
    wrapper = Path(__file__).resolve().parents[1]
    wrapper_sha = subprocess.check_output(["git", "-C", str(wrapper), "rev-parse", "HEAD"], text=True).strip()
    science = Path("/kaggle/working/FER2013_Graph_M0_SCIENCE")
    if science.exists():
        shutil.rmtree(science)
    subprocess.run(["git", "clone", REPO_URL, str(science)], check=True)
    subprocess.run(["git", "-C", str(science), "checkout", "--detach", SCIENTIFIC_SHA], check=True)
    if subprocess.check_output(["git", "-C", str(science), "rev-parse", "HEAD"], text=True).strip() != SCIENTIFIC_SHA:
        raise RuntimeError("M0 scientific source-lock mismatch")
    paths = [f"research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_{name}.py" for name in ("config", "substrate", "models", "train", "aggregate")]
    changed = subprocess.check_output(["git", "-C", str(wrapper), "diff", "--name-only", SCIENTIFIC_SHA, wrapper_sha, "--", *paths], text=True).strip()
    if changed:
        raise RuntimeError(f"scientific M0 files changed after lock: {changed}")
    package = science / "research/pixel_relational_motif_e0"
    env = os.environ.copy(); env["PYTHONPATH"] = str(package / "src")
    test = subprocess.run([sys.executable, "-m", "pytest", str(package / "tests"), "-q"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError("pre-data M0 tests failed")
    sys.path.insert(0, str(package / "src"))
    return science, wrapper_sha


def _substrate_root() -> Path:
    candidates = (
        Path("/kaggle/input/pgm-m0-frozen-substrate"),
        Path("/kaggle/input/datasets/irthn1311/pgm-m0-frozen-substrate"),
        Path("/kaggle/input/datasets/trngthngcnhi/pgm-m0-frozen-substrate"),
        Path("/kaggle/input/datasets/thanhhhgng/pgm-m0-frozen-substrate"),
    )
    matches = [root for root in candidates if (root / "m0_substrate_manifest.json").is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"need exactly one frozen M0 substrate mount: {matches}")
    return matches[0]


def _update_manifest(output: Path, job: str, fits: list[dict], wrapper_sha: str, account: str) -> None:
    payload = {
        "experiment": "PGM_M0_INCREMENTAL_GRAPH_STRUCTURED_PROCESSING", "issue": 82,
        "scientific_sha": SCIENTIFIC_SHA, "execution_wrapper_sha": wrapper_sha,
        "job": job, "account": account, "fits": fits, "public_labels_read": False,
        "public_metrics_calculated": False, "private_test_read": False,
    }
    (output / f"m0_{job}_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("L", "M", "G"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--account", required=True)
    args = parser.parse_args()
    if args.family == "L" and args.seeds:
        raise ValueError("L does not accept seeds")
    if args.family != "L" and (not args.seeds or any(seed not in SEEDS for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("M/G seeds must be distinct registered seeds")
    _science, wrapper_sha = _lock_and_test()
    from pixel_relational_motif_e0.m0_substrate import load_substrate
    from pixel_relational_motif_e0.m0_train import environment_record, fit_l, train_seed

    substrate = load_substrate(_substrate_root())
    output = Path(f"/kaggle/working/outputs/pgm_m0_{args.family.lower()}_{'_'.join(map(str,args.seeds)) or 'fixed'}")
    output.mkdir(parents=True, exist_ok=True)
    job = f"{args.family}_{'_'.join(map(str,args.seeds)) or 'fixed'}"
    fits: list[dict] = []
    _update_manifest(output, job, fits, wrapper_sha, args.account)
    if args.family == "L":
        fits.append(fit_l(substrate, output, scientific_sha=SCIENTIFIC_SHA, wrapper_sha=wrapper_sha, account=args.account))
        _update_manifest(output, job, fits, wrapper_sha, args.account)
    else:
        for seed in args.seeds:
            fits.append(train_seed(substrate, args.family, seed, output, scientific_sha=SCIENTIFIC_SHA, wrapper_sha=wrapper_sha, account=args.account))
            _update_manifest(output, job, fits, wrapper_sha, args.account)
            print(f"[PGM-M0] checkpointed {args.family}{seed} sha256={fits[-1]['sha256']}", flush=True)
    (output / "environment.json").write_text(json.dumps(environment_record(), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (output / "source_lock_provenance.json").write_text(json.dumps({
        "issue": 82, "scientific_sha": SCIENTIFIC_SHA, "execution_wrapper_sha": wrapper_sha,
        "account": args.account, "family": args.family, "seeds": args.seeds,
        "public_labels_read": False, "private_test_read": False,
    }, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"job": job, "fits": fits}, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
