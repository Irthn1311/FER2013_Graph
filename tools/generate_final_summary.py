"""Generate final summary tables (CSV, JSON, Markdown, LaTeX) and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"

MODES = [
    ("A0", "Pixel Graph baseline", "Pixel Graph baseline"),
    ("A1", "+ Spatial Motif Composer", "+ Spatial Motif Composer"),
    ("A2", "+ Motif Graph", "+ Motif Graph"),
    ("A3", "+ Geometry Bias", "+ Geometry Bias"),
    ("A4", "+ Multi-scale Composition", "+ Multi-scale Composition"),
    ("A5", "+ Dynamic Top-K Relations", "+ Dynamic Top-K Relations"),
    ("A6", "Full Model", "Full Model"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    results = {}
    for m, disp_name, vn_name in MODES:
        res_file = ANALYSIS_DIR / f"{m}_RESULT.json"
        if not res_file.is_file():
            raise FileNotFoundError(f"Missing {res_file}")
        results[m] = json.loads(res_file.read_text(encoding="utf-8"))

    rows = []
    prev_tta_acc = None
    for i, (m, disp_name, vn_name) in enumerate(MODES):
        res = results[m]
        priv = res["private_metrics"]

        # Note: accuracy in percentage units
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
            "display_name": disp_name,
            "vietnamese_name": vn_name,
            "raw_accuracy": raw_acc,
            "raw_macro_f1": raw_f1,
            "tta_accuracy": tta_acc,
            "tta_macro_f1": tta_f1,
            "gain": gain,
            "selected_epoch": res.get("selected_epoch"),
            "checkpoint_sha256": res.get("checkpoint_sha256"),
            "weights_type": res.get("weights_type", "EMA"),
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
    main()
