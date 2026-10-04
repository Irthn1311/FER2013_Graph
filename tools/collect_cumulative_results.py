"""Collect results from completed Kaggle jobs and generate final summary tables."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.model import (
    CUMULATIVE_ABLATION_ORDER,
    CUMULATIVE_REGISTRY,
    CumulativeAblationMode,
)
from mpg_fer_cumulative_ablation7.protocol import (
    CANONICAL_DATASET_HASHES,
    sha256_file,
)

ANALYSIS_DIR = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("A0", "irthn1311", "mpg-fer-cumabl7-opus-a0", None),
    ("A1", "maiduyen311", "mpg-fer-cumabl7-opus-a1", r"D:\Downloads\kaggle (4).json"),
    ("A2", "nuyntai", "mpg-fer-cumabl7-opus-a2", r"D:\Downloads\kaggle (5).json"),
    ("A3", "quangdangnguyen30", "mpg-fer-cumabl7-opus-a3", r"D:\Downloads\kaggle (6).json"),
    ("A4", "nadkli2704", "mpg-fer-cumabl7-opus-a4", r"D:\Downloads\kaggle (7).json"),
    ("A5", "irthn1311", "mpg-fer-cumabl7-opus-a5", None),
]


def poll_all_jobs() -> dict[str, str]:
    statuses = {}
    for mode, username, slug, cred_file in JOBS:
        if cred_file:
            with open(cred_file, "r") as fp:
                creds = json.load(fp)
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]
        else:
            os.environ.pop("KAGGLE_USERNAME", None)
            os.environ.pop("KAGGLE_KEY", None)

        api = KaggleApi()
        api.authenticate()
        kernel_ref = f"{username}/{slug}"
        try:
            st = api.kernels_status(kernel_ref)
            statuses[mode] = str(st.status)
        except Exception as e:
            statuses[mode] = f"ERROR: {e}"
    return statuses


def download_and_extract_job(mode: str, username: str, slug: str, cred_file: str | None) -> dict | None:
    if cred_file:
        with open(cred_file, "r") as fp:
            creds = json.load(fp)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
    else:
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)

    api = KaggleApi()
    api.authenticate()
    kernel_ref = f"{username}/{slug}"

    dest_dir = ANALYSIS_DIR / "runs" / mode
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading outputs for [{mode}] from {kernel_ref}...")
    api.kernels_output(kernel_ref, path=str(dest_dir))

    # Look for zip or json outputs
    zip_files = list(dest_dir.glob("*.zip"))
    if zip_files:
        for z in zip_files:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest_dir)

    result_json = dest_dir / f"{mode}_RESULT.json"
    if not result_json.is_file():
        # Look for canonical_private_metrics.json
        priv_json = dest_dir / "canonical_private_metrics.json"
        pub_json = dest_dir / "canonical_public_metrics.json"
        exec_json = dest_dir / "execution_manifest.json"
        if priv_json.is_file() and pub_json.is_file():
            priv = json.loads(priv_json.read_text(encoding="utf-8"))
            pub = json.loads(pub_json.read_text(encoding="utf-8"))
            ex = json.loads(exec_json.read_text(encoding="utf-8")) if exec_json.is_file() else {}
            spec = CUMULATIVE_REGISTRY[CumulativeAblationMode(mode)]
            res = {
                "schema_version": 1,
                "configuration": mode,
                "display_name": spec.paper_name,
                "vietnamese_name": spec.vietnamese_name,
                "status": "TRAINING_COMPLETED",
                "seed": 42,
                "checkpoint_sha256": priv.get("checkpoint_sha256", ex.get("checkpoint_sha256")),
                "selected_epoch": priv.get("selected_epoch", ex.get("best_epoch")),
                "weights_type": "EMA",
                "dataset_hashes": CANONICAL_DATASET_HASHES,
                "public_metrics": {
                    "raw": pub["views"]["raw"] if "views" in pub else pub["raw"],
                    "tta": pub["views"]["horizontal_flip_tta"] if "views" in pub else pub["tta"],
                },
                "private_metrics": {
                    "raw": priv["views"]["raw"] if "views" in priv else priv["raw"],
                    "tta": priv["views"]["horizontal_flip_tta"] if "views" in priv else priv["tta"],
                },
                "kaggle_job_ref": kernel_ref,
            }
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    if result_json.is_file():
        target_res = ANALYSIS_DIR / f"{mode}_RESULT.json"
        shutil.copy2(result_json, target_res)
        return json.loads(result_json.read_text(encoding="utf-8"))
    return None


def generate_summary_tables() -> None:
    results = {}
    for mode in CUMULATIVE_ABLATION_ORDER:
        m = mode.value
        res_file = ANALYSIS_DIR / f"{m}_RESULT.json"
        if not res_file.is_file():
            print(f"Waiting for {m}_RESULT.json...")
            return
        results[m] = json.loads(res_file.read_text(encoding="utf-8"))

    rows = []
    prev_tta_acc = None
    for i, mode in enumerate(CUMULATIVE_ABLATION_ORDER):
        m = mode.value
        res = results[m]
        spec = CUMULATIVE_REGISTRY[mode]
        priv = res["private_metrics"]

        raw_acc = priv["raw"]["accuracy"] * 100.0
        raw_f1 = priv["raw"]["macro_f1"] * 100.0
        tta_acc = priv["tta"]["accuracy"] * 100.0
        tta_f1 = priv["tta"]["macro_f1"] * 100.0

        if i == 0:
            gain = None
        else:
            gain = tta_acc - prev_tta_acc
        prev_tta_acc = tta_acc

        rows.append({
            "configuration": m,
            "display_name": spec.paper_name,
            "vietnamese_name": spec.vietnamese_name,
            "raw_accuracy": raw_acc,
            "raw_macro_f1": raw_f1,
            "tta_accuracy": tta_acc,
            "tta_macro_f1": tta_f1,
            "gain": gain,
            "checkpoint_sha256": res.get("checkpoint_sha256"),
            "selected_epoch": res.get("selected_epoch"),
            "status": res.get("status"),
        })

    summary_data = {
        "schema_version": 1,
        "method": "MPG-FER",
        "nested_ladder": "A0 ⊂ A1 ⊂ A2 ⊂ A3 ⊂ A4 ⊂ A5 ⊂ A6",
        "rows": rows,
    }

    # 1. Summary JSON
    summary_json_path = ANALYSIS_DIR / "CUMULATIVE_ABLATION7_SUMMARY.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Wrote", summary_json_path)

    # 2. Summary CSV
    summary_csv_path = ANALYSIS_DIR / "CUMULATIVE_ABLATION7_SUMMARY.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "display_name", "raw_accuracy", "raw_macro_f1",
            "tta_accuracy", "tta_macro_f1", "gain", "selected_epoch", "checkpoint_sha256"
        ])
        for r in rows:
            g_str = f"{r['gain']:+.2f}" if r["gain"] is not None else "—"
            writer.writerow([
                r["configuration"], r["display_name"], f"{r['raw_accuracy']:.2f}",
                f"{r['raw_macro_f1']:.2f}", f"{r['tta_accuracy']:.2f}",
                f"{r['tta_macro_f1']:.2f}", g_str, r["selected_epoch"], r["checkpoint_sha256"]
            ])
    print("Wrote", summary_csv_path)

    # 3. Paper-facing Markdown Table (English + Vietnamese)
    md_lines = [
        "# Cumulative Ablation Study on FER2013 (MPG-FER)",
        "",
        "### English (Paper-Facing Table)",
        "",
        "| Configuration | Acc. (%) | Gain |",
        "|---|---:|---:|",
    ]
    for r in rows:
        g_str = f"{r['gain']:+.2f}" if r["gain"] is not None else "—"
        md_lines.append(f"| {r['display_name']} | {r['tta_accuracy']:.2f} | {g_str} |")

    md_lines.extend([
        "",
        "### Vietnamese (Bảng Tiếng Việt)",
        "",
        "| Cấu hình | Acc. (%) | Gain |",
        "|---|---:|---:|",
    ])
    for r in rows:
        g_str = f"{r['gain']:+.2f}" if r["gain"] is not None else "—"
        md_lines.append(f"| {r['vietnamese_name']} | {r['tta_accuracy']:.2f} | {g_str} |")

    md_lines.extend([
        "",
        "### Full Evaluation Metrics (Raw & TTA)",
        "",
        "| Config | Display Name | Raw Acc. (%) | Raw F1 (%) | TTA Acc. (%) | TTA F1 (%) | Gain | Selected Epoch | Checkpoint SHA256 (first 10) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in rows:
        g_str = f"{r['gain']:+.2f}" if r["gain"] is not None else "—"
        sha_short = str(r['checkpoint_sha256'])[:10] if r['checkpoint_sha256'] else "—"
        md_lines.append(
            f"| {r['configuration']} | {r['display_name']} | {r['raw_accuracy']:.2f} | {r['raw_macro_f1']:.2f} | {r['tta_accuracy']:.2f} | {r['tta_macro_f1']:.2f} | {g_str} | {r['selected_epoch']} | `{sha_short}` |"
        )

    table_md_path = ANALYSIS_DIR / "CUMULATIVE_ABLATION7_TABLE.md"
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Wrote", table_md_path)

    # 4. Paper-facing LaTeX Table
    tex_lines = [
        "% Cumulative Ablation Table for MPG-FER",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Cumulative ablation of design components on FER2013 under test-time augmentation (TTA).}",
        "\\label{tab:cumulative_ablation}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Acc. (\\%)} & \\textbf{Gain} \\\\",
        "\\midrule",
    ]
    for r in rows:
        g_str = f"{r['gain']:+.2f}" if r["gain"] is not None else "---"
        tex_lines.append(f"{r['display_name']} & {r['tta_accuracy']:.2f} & {g_str} \\\\")
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_tex_path = ANALYSIS_DIR / "CUMULATIVE_ABLATION7_TABLE.tex"
    with open(table_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print("Wrote", table_tex_path)

    # 5. Checksums for all artifacts in ANALYSIS_DIR
    chk_lines = []
    for fpath in sorted(ANALYSIS_DIR.glob("*")):
        if fpath.is_file() and fpath.name != "checksums.sha256":
            chk_lines.append(f"{sha256_file(fpath)}  {fpath.name}")
    checksums_path = ANALYSIS_DIR / "checksums.sha256"
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chk_lines) + "\n")
    print("Wrote", checksums_path)


if __name__ == "__main__":
    statuses = poll_all_jobs()
    for m, st in statuses.items():
        print(f"[{m}] {st}")
    generate_summary_tables()
