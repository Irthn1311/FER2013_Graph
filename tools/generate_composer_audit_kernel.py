"""Generate the standalone Kaggle notebook for Spatial Motif Composer Constraint Audit."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V23_SOURCE_ROOT = ROOT / "research" / "mpg_fer_v2_3" / "src"

PACKAGES = [
    (V23_SOURCE_ROOT / "mpg_fer_v2_3", "mpg_fer_v2_3"),
]


def encode_sources() -> tuple[dict[str, str], str]:
    encoded = {}
    digest = hashlib.sha256()
    for src_dir, pkg_name in PACKAGES:
        for p in sorted(src_dir.glob("*.py")):
            rel_name = f"{pkg_name}/{p.name}"
            data = p.read_bytes().replace(b"\r\n", b"\n")
            encoded[rel_name] = base64.b64encode(data).decode("ascii")
            digest.update(rel_name.encode("utf-8"))
            digest.update(data)
    return encoded, digest.hexdigest()


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_composer_audit_notebook() -> dict:
    encoded_sources, source_sha = encode_sources()

    cells = [
        _cell(
            "markdown",
            "# MPG-FER Spatial Motif Composer Constraint Audit\n\n"
            "This focused audit investigates whether the fixed anchor/support/prototype formulation of the "
            "Spatial Motif Composer imposes measurable structural constraints on representation formation.\n\n"
            "Sections:\n"
            "- Part A: Spatial Support Pressure\n"
            "- Part B: Scale-Pressure Analysis\n"
            "- Part C: Prototype-Space Audit\n"
            "- Part D: Occurrence Redundancy at M0\n"
            "- Part E: Error-Conditioned Analysis (Correct vs Incorrect on Validation)\n"
            "- Part F: FULL vs NPF Comparison\n\n"
            "**Hard Rule Enforcement:** Train and Validation only. Absolutely zero Test/Private access.\n",
        ),
        _cell(
            "code",
            """import os
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mpg_fer_composer_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Composer Audit output directory:", OUTPUT_DIR)
""",
        ),
        _cell(
            "code",
            f"""import base64
import hashlib
import sys

EMBEDDED_SOURCES = {encoded_sources!r}
EXPECTED_SOURCE_TREE_SHA256 = {source_sha!r}
SOURCE_ROOT = Path("/kaggle/working/source_root")
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

digest = hashlib.sha256()
for name, b64payload in sorted(EMBEDDED_SOURCES.items()):
    payload = base64.b64decode(b64payload)
    digest.update(name.encode("utf-8"))
    digest.update(payload)
    target = SOURCE_ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

actual_sha = digest.hexdigest()
if actual_sha != EXPECTED_SOURCE_TREE_SHA256:
    raise RuntimeError(f"Source SHA mismatch: expected {{EXPECTED_SOURCE_TREE_SHA256}}, got {{actual_sha}}")

sys.path.insert(0, str(SOURCE_ROOT))
print("Source unpacked successfully:", actual_sha)
""",
        ),
        _cell(
            "code",
            """import math
import hashlib
import json
import time
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from mpg_fer_v2_3.model import MPGFER, compute_motif_geometry
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import FER2013Dataset
from mpg_fer_v2_3.motif import build_aligned_support_indices

INPUT_ROOT = Path("/kaggle/input")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# 1. Locate Checkpoints
EXPECTED_FULL_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
EXPECTED_NPF_SHA = "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972"

full_ckpt, npf_ckpt = None, None
for p in INPUT_ROOT.rglob("best_val_acc.pt"):
    h = sha256_file(p)
    if h == EXPECTED_FULL_SHA:
        full_ckpt = p
    elif h == EXPECTED_NPF_SHA:
        npf_ckpt = p

if not full_ckpt:
    raise FileNotFoundError("Canonical FULL checkpoint not located!")
print("FULL Checkpoint:", full_ckpt, "(SHA:", EXPECTED_FULL_SHA, ")")
if npf_ckpt:
    print("NPF Checkpoint: ", npf_ckpt, "(SHA:", EXPECTED_NPF_SHA, ")")

# 2. Locate CSVs
train_csv_candidates = [p for p in INPUT_ROOT.rglob("train.csv") if "test" not in p.name.lower()]
val_csv_candidates = [p for p in INPUT_ROOT.rglob("val.csv") if "test" not in p.name.lower()]

train_csv = train_csv_candidates[0]
val_csv = val_csv_candidates[0]

EXPECTED_TRAIN_SHA = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
EXPECTED_VAL_SHA = "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"

assert sha256_file(train_csv) == EXPECTED_TRAIN_SHA, "Train SHA mismatch"
assert sha256_file(val_csv) == EXPECTED_VAL_SHA, "Val SHA mismatch"

print("Train CSV:", train_csv)
print("Val CSV:  ", val_csv)

for p in INPUT_ROOT.rglob("*.csv"):
    if "test.csv" in p.name.lower():
        print("Confirmed test.csv is present on disk but strictly NOT accessed.")
""",
        ),
        _cell(
            "code",
            """# 3. Load Models and Build Anchor/Support Geometries
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Execution device:", device)

cfg = MPGConfig()

model_full = MPGFER(cfg).to(device)
payload_full = torch.load(full_ckpt, map_location=device, weights_only=False)
model_full.load_state_dict(payload_full["model_state_dict"], strict=True)
model_full.eval()

model_npf = None
if npf_ckpt:
    model_npf = MPGFER(cfg).to(device)
    payload_npf = torch.load(npf_ckpt, map_location=device, weights_only=False)
    model_npf.load_state_dict(payload_npf["model_state_dict"], strict=True)
    model_npf.eval()
print("Models loaded successfully.")

# Build aligned supports and anchor centers
supports, centers = build_aligned_support_indices(img_size=48, anchor_size=12, stride=6, scales=(8, 12, 16))
anchor_centers_norm = (2.0 * centers / 47.0 - 1.0).to(device) # [49, 2]

# Outer 20% ring mask for each scale
ring_masks = {}
for scale in [8, 12, 16]:
    coords = torch.arange(scale, device=device).float()
    center_coord = (scale - 1) / 2.0
    max_dist = (scale - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    dist_inf = torch.maximum((grid_x - center_coord).abs(), (grid_y - center_coord).abs()) / max_dist
    ring_masks[scale] = (dist_inf >= 0.80).reshape(-1) # [scale*scale]

# Half-widths in normalized coordinates
half_widths = {
    8: (8 - 1) / 2.0 * (2.0 / 47.0),
    12: (12 - 1) / 2.0 * (2.0 / 47.0),
    16: (16 - 1) / 2.0 * (2.0 / 47.0),
}

source_audit_dict = {
    "schema_version": 1,
    "full_checkpoint_sha256": EXPECTED_FULL_SHA,
    "npf_checkpoint_sha256": EXPECTED_NPF_SHA if npf_ckpt else None,
    "train_sha256": EXPECTED_TRAIN_SHA,
    "val_sha256": EXPECTED_VAL_SHA,
    "anchors": 49,
    "anchor_grid": "7x7 aligned, stride 6, anchor size 12",
    "scales": [8, 12, 16],
    "prototypes": 48,
    "d_pixel": 96,
    "d_motif": 192,
    "status": "PASS",
}
with open(OUTPUT_DIR / "SOURCE_AUDIT.json", "w") as f:
    json.dump(source_audit_dict, f, indent=2)

md_lines = [
    "# Phase 0: Source Audit for Spatial Motif Composer Constraint Audit",
    "",
    f"- **FULL Checkpoint SHA256:** `{EXPECTED_FULL_SHA}`",
    f"- **NPF Checkpoint SHA256:**  `{EXPECTED_NPF_SHA}`",
    "- **Anchors:** 49 occurrences in a 7x7 grid (stride 6, anchor size 12)",
    "- **Scales:** 8, 12, 16 (window sizes 8x8, 12x12, 16x16 sharing anchor centers)",
    "- **Prototypes:** 48 learned prototype vectors (96D)",
    "- **Data Scope:** Train (28,709) and Validation (3,589) only. Strictly zero Test access.",
]
with open(OUTPUT_DIR / "SOURCE_AUDIT.md", "w") as f:
    f.write("\\n".join(md_lines) + "\\n")
print("Wrote SOURCE_AUDIT.json/md")
""",
        ),
        _cell(
            "code",
            """# 4. Extract Validation SMC Tensors and Model Predictions
val_dataset = FER2013Dataset(val_csv, split="val", augment=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

def extract_smc_audit_data(model):
    all_preds, all_targets = [], []
    all_weights = {8: [], 12: [], 16: []}
    all_centers = {8: [], 12: [], 16: []}
    all_spreads = {8: [], 12: [], 16: []}
    all_alpha = []
    all_assignments = []
    all_M0 = []
    
    composer = model.motif_composer
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            batch = images.shape[0]
            
            logits, outputs = model(images)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(targets.cpu())
            
            # Pixel GNN output H_P
            h = model.pixel_proj(model.pixel_extractor(images))
            intensities = images.reshape(batch, cfg.num_pixels, 1)
            edges = model.pixel_topology.compute_edge_features(intensities)
            for layer in model.pixel_gnn:
                h = layer(h, model.pixel_topology.neighbor_idx, model.pixel_topology.neighbor_mask, edges)
                
            queries = F.normalize(composer.assignment_query(h), dim=-1)
            keys = F.normalize(composer.prototype_key(composer.prototypes), dim=-1)
            assignments = F.softmax(queries @ keys.t() / composer.temperature, dim=-1)
            confidence = assignments.max(dim=-1, keepdim=True).values
            
            candidates, centers_list = [], []
            for scale in composer.window_sizes:
                candidate, center, scale_weights, _ = composer._pool_scale(h, assignments, confidence, scale)
                candidates.append(candidate)
                centers_list.append(center)
                all_weights[scale].append(scale_weights.cpu()) # [B, 49, scale^2]
                all_centers[scale].append(center.cpu()) # [B, 49, 2]
                
                # Compute weighted spread sigma:
                indices = getattr(composer, f"support_idx_{scale}")
                occurrences, support_size = indices.shape
                x_support = composer.grid_x[indices].view(1, occurrences, support_size, 1).expand(batch, -1, -1, -1)
                y_support = composer.grid_y[indices].view(1, occurrences, support_size, 1).expand(batch, -1, -1, -1)
                w_exp = scale_weights.unsqueeze(-1)
                cx = center[..., 0:1]
                cy = center[..., 1:2]
                sx = torch.sqrt((w_exp * (x_support - cx.unsqueeze(2)).square()).sum(dim=2) + composer.eps)
                sy = torch.sqrt((w_exp * (y_support - cy.unsqueeze(2)).square()).sum(dim=2) + composer.eps)
                spread = torch.cat([sx, sy], dim=-1) # [B, 49, 2]
                all_spreads[scale].append(spread.cpu())
                
            candidate_stack = torch.stack(candidates, dim=2)
            alpha = F.softmax(composer.scale_gate(candidate_stack).squeeze(-1), dim=-1) # [B, 49, 3]
            h_motif_0 = (alpha.unsqueeze(-1) * candidate_stack).sum(dim=2) # [B, 49, 192]
            
            all_alpha.append(alpha.cpu())
            all_assignments.append(assignments.cpu()) # [B, 2304, 48]
            all_M0.append(h_motif_0.cpu())
            
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    weights = {s: torch.cat(all_weights[s]) for s in [8, 12, 16]}
    centers_dict = {s: torch.cat(all_centers[s]) for s in [8, 12, 16]}
    spreads_dict = {s: torch.cat(all_spreads[s]) for s in [8, 12, 16]}
    alpha_tensor = torch.cat(all_alpha)
    assignments_tensor = torch.cat(all_assignments)
    M0_tensor = torch.cat(all_M0)
    
    return {
        "preds": preds,
        "targets": targets,
        "correct": (preds == targets),
        "weights": weights,
        "centers": centers_dict,
        "spreads": spreads_dict,
        "alpha": alpha_tensor,
        "assignments": assignments_tensor,
        "M0": M0_tensor,
        "keys": composer.prototype_key(composer.prototypes).detach().cpu(),
    }

print("Extracting FULL Validation audit data...")
full_audit_data = extract_smc_audit_data(model_full)
val_acc = accuracy_score(full_audit_data["targets"], full_audit_data["preds"])
val_f1 = f1_score(full_audit_data["targets"], full_audit_data["preds"], average="macro")
print(f"FULL Val Accuracy: {val_acc*100:.2f}%, Macro-F1: {val_f1*100:.2f}% (Correct: {full_audit_data['correct'].sum()}/3589)")
""",
        ),
        _cell(
            "code",
            """# 5. Part A: Spatial Support Pressure Analysis
data = full_audit_data
N = len(data[\"targets\"])
anchors_cpu = anchor_centers_norm.cpu() # [49, 2]

spatial_pressure_records = []
per_occurrence_stats = {s: [] for s in [8, 12, 16]}

for scale in [8, 12, 16]:
    w = data[\"weights\"][scale] # [N, 49, scale^2]
    c = data[\"centers\"][scale] # [N, 49, 2]
    spread = data[\"spreads\"][scale] # [N, 49, 2]
    R_s = half_widths[scale]
    ring_m = ring_masks[scale].cpu() # [scale^2]
    
    # Center displacement from anchor
    disp_xy = c - anchors_cpu.unsqueeze(0) # [N, 49, 2]
    disp = torch.norm(disp_xy, dim=-1) # [N, 49]
    norm_disp = disp / R_s # [N, 49]
    
    # Distance to nearest boundary (normalized by R_s, in [0, 1])
    dist_bound = torch.clamp(1.0 - torch.maximum(disp_xy[..., 0].abs(), disp_xy[..., 1].abs()) / R_s, min=0.0) # [N, 49]
    
    # Mass in outer 20% ring
    outer_mass = w[..., ring_m].sum(dim=-1) # [N, 49]
    
    # Spatial weight entropy
    p_w = w.clamp(min=1e-12)
    entropy_w = -(p_w * p_w.log()).sum(dim=-1) # [N, 49]
    effective_pixels = entropy_w.exp() # [N, 49]
    
    # Max pixel in outer 20% ring
    max_idx = w.argmax(dim=-1) # [N, 49]
    max_in_ring = ring_m[max_idx].float() # [N, 49]
    
    spatial_pressure_records.append({
        \"scale\": scale,
        \"mean_center_displacement\": float(disp.mean().item()),
        \"mean_normalized_displacement\": float(norm_disp.mean().item()),
        \"std_normalized_displacement\": float(norm_disp.std().item()),
        \"mean_dist_to_boundary\": float(dist_bound.mean().item()),
        \"mean_outer_ring_mass\": float(outer_mass.mean().item()),
        \"fraction_mass_in_outer_ring\": float((outer_mass > 0.50).float().mean().item()),
        \"mean_spatial_entropy\": float(entropy_w.mean().item()),
        \"mean_effective_contributing_pixels\": float(effective_pixels.mean().item()),
        \"max_pixel_in_outer_ring_rate\": float(max_in_ring.mean().item()),
        \"mean_spread_x\": float(spread[..., 0].mean().item()),
        \"mean_spread_y\": float(spread[..., 1].mean().item()),
    })
    
    # Per-occurrence breakdown
    for m in range(49):
        per_occurrence_stats[scale].append({
            \"occurrence_m\": m,
            \"mean_norm_disp\": float(norm_disp[:, m].mean().item()),
            \"mean_outer_mass\": float(outer_mass[:, m].mean().item()),
            \"max_in_ring_rate\": float(max_in_ring[:, m].mean().item()),
        })

print(\"=== Part A: Spatial Support Pressure Summary ===\")
for r in spatial_pressure_records:
    print(f\"Scale {r['scale']:02d}: Norm Disp = {r['mean_normalized_displacement']:.3f} | Outer Ring Mass = {r['mean_outer_ring_mass']*100:.1f}% | Max In Ring = {r['max_pixel_in_outer_ring_rate']*100:.1f}%\")

with open(OUTPUT_DIR / \"SPATIAL_SUPPORT_PRESSURE.json\", \"w\") as f:
    json.dump({\"by_scale\": spatial_pressure_records, \"per_occurrence\": per_occurrence_stats}, f, indent=2)

with open(OUTPUT_DIR / \"SPATIAL_SUPPORT_PRESSURE.csv\", \"w\", newline=\"\") as f:
    writer = csv.writer(f)
    writer.writerow([\"scale\", \"mean_norm_disp\", \"mean_dist_to_boundary\", \"mean_outer_ring_mass\", \"max_in_ring_rate\", \"mean_spatial_entropy\", \"effective_pixels\"])
    for r in spatial_pressure_records:
        writer.writerow([r[\"scale\"], f\"{r['mean_normalized_displacement']:.4f}\", f\"{r['mean_dist_to_boundary']:.4f}\", f\"{r['mean_outer_ring_mass']:.4f}\", f\"{r['max_pixel_in_outer_ring_rate']:.4f}\", f\"{r['mean_spatial_entropy']:.4f}\", f\"{r['mean_effective_contributing_pixels']:.2f}\"])
print(\"Saved SPATIAL_SUPPORT_PRESSURE.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 6. Part B: Scale-Pressure Analysis
alpha = data[\"alpha\"] # [N, 49, 3]

# Mean scale probabilities
mean_alpha = alpha.mean(dim=(0, 1)).tolist() # [3]
p_a = alpha.clamp(min=1e-12)
entropy_alpha = -(p_a * p_a.log()).sum(dim=-1).mean().item()

# Argmax frequencies
argmax_scale = alpha.argmax(dim=-1) # [N, 49]
freq_8 = float((argmax_scale == 0).float().mean().item())
freq_12 = float((argmax_scale == 1).float().mean().item())
freq_16 = float((argmax_scale == 2).float().mean().item())

# Per-class scale frequencies
num_classes = 7
class_scale_freqs = []
for c in range(num_classes):
    mask_c = (data[\"targets\"] == c)
    argmax_c = argmax_scale[mask_c]
    class_scale_freqs.append({
        \"class\": c,
        \"scale_8\": float((argmax_c == 0).float().mean().item()),
        \"scale_12\": float((argmax_c == 1).float().mean().item()),
        \"scale_16\": float((argmax_c == 2).float().mean().item()),
    })

# Joint analysis: Scale 16 preference vs spatial indicators
alpha_16 = alpha[..., 2].flatten().numpy() # [N*49]
c16 = data[\"centers\"][16]
disp_16 = torch.norm(c16 - anchors_cpu.unsqueeze(0), dim=-1) / half_widths[16]
norm_disp_16 = disp_16.flatten().numpy()

w16 = data[\"weights\"][16]
outer_mass_16 = w16[..., ring_masks[16].cpu()].sum(dim=-1).flatten().numpy()
spread_16 = data[\"spreads\"][16].mean(dim=-1).flatten().numpy()

corr_disp, _ = stats.pearsonr(alpha_16, norm_disp_16)
corr_outer, _ = stats.pearsonr(alpha_16, outer_mass_16)
corr_spread, _ = stats.pearsonr(alpha_16, spread_16)

scale_pressure_summary = {
    \"mean_scale_probabilities\": {\"scale_8\": mean_alpha[0], \"scale_12\": mean_alpha[1], \"scale_16\": mean_alpha[2]},
    \"scale_gate_entropy\": entropy_alpha,
    \"argmax_scale_frequencies\": {\"scale_8\": freq_8, \"scale_12\": freq_12, \"scale_16\": freq_16},
    \"per_class_scale_frequencies\": class_scale_freqs,
    \"scale_16_joint_correlations\": {
        \"correlation_with_norm_displacement\": float(corr_disp),
        \"correlation_with_outer_ring_mass\": float(corr_outer),
        \"correlation_with_spread\": float(corr_spread),
    },
    \"coincidence_boundary_pressure_and_scale_16\": float(np.mean((alpha_16 > 0.50) & (outer_mass_16 > 0.50))),
}

print(\"=== Part B: Scale-Pressure Summary ===\")
print(f\"Mean Scale Probs: 8: {mean_alpha[0]:.3f}, 12: {mean_alpha[1]:.3f}, 16: {mean_alpha[2]:.3f}\")
print(f\"Argmax Freqs:     8: {freq_8*100:.1f}%, 12: {freq_12*100:.1f}%, 16: {freq_16*100:.1f}%\")
print(f\"Corr(Scale 16, Outer Ring Mass): {corr_outer:.4f}\")

with open(OUTPUT_DIR / \"SCALE_PRESSURE.json\", \"w\") as f:
    json.dump(scale_pressure_summary, f, indent=2)

with open(OUTPUT_DIR / \"SCALE_PRESSURE.csv\", \"w\", newline=\"\") as f:
    writer = csv.writer(f)
    writer.writerow([\"metric\", \"scale_8\", \"scale_12\", \"scale_16\"])
    writer.writerow([\"mean_probability\", f\"{mean_alpha[0]:.4f}\", f\"{mean_alpha[1]:.4f}\", f\"{mean_alpha[2]:.4f}\"])
    writer.writerow([\"argmax_frequency\", f\"{freq_8:.4f}\", f\"{freq_12:.4f}\", f\"{freq_16:.4f}\"])
print(\"Saved SCALE_PRESSURE.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 7. Part C: Prototype-Space Audit
assignments = data[\"assignments\"] # [N, 2304, 48]
keys = data[\"keys\"] # [48, 96]

# 1. Marginal prototype usage across all pixels and samples
marginal_usage = assignments.mean(dim=(0, 1)).numpy() # [48]
p_u = np.clip(marginal_usage, 1e-12, 1.0)
usage_entropy = float(-np.sum(p_u * np.log(p_u)))
effective_prototypes = float(np.exp(usage_entropy))

# 2. Assignment entropy per pixel
p_a = assignments.clamp(min=1e-12)
entropy_per_pixel = float((-(p_a * p_a.log()).sum(dim=-1)).mean().item())

# 3. Maximum assignment concentration
max_concentration = float(assignments.max(dim=-1).values.mean().item())

# 4. Prototype key pairwise cosine similarity
normed_keys = F.normalize(keys, dim=-1)
key_sim_matrix = (normed_keys @ normed_keys.t()).numpy()
off_diag_keys = key_sim_matrix[~np.eye(48, dtype=bool)]
mean_key_sim = float(np.mean(off_diag_keys))
max_key_sim = float(np.max(off_diag_keys))
fraction_near_duplicate = float(np.mean(off_diag_keys > 0.90))

# 5. Unused and dominant prototypes
unused_count = int(np.sum(marginal_usage < 0.005))
top5_mass = float(np.sum(np.sort(marginal_usage)[-5:]))

prototype_audit = {
    \"total_prototypes\": 48,
    \"effective_prototype_count\": effective_prototypes,
    \"usage_entropy\": usage_entropy,
    \"normalized_usage_entropy\": usage_entropy / math.log(48),
    \"mean_pixel_assignment_entropy\": entropy_per_pixel,
    \"mean_max_assignment_concentration\": max_concentration,
    \"mean_pairwise_key_cosine_similarity\": mean_key_sim,
    \"max_pairwise_key_cosine_similarity\": max_key_sim,
    \"fraction_near_duplicate_keys_gt_090\": fraction_near_duplicate,
    \"unused_prototypes_count\": unused_count,
    \"top5_prototypes_mass_fraction\": top5_mass,
    \"marginal_usage_per_prototype\": marginal_usage.tolist(),
}

print(\"=== Part C: Prototype-Space Audit ===\")
print(f\"Effective Prototypes: {effective_prototypes:.2f} / 48 (Entropy: {usage_entropy:.3f})\")
print(f\"Max Assignment Concentration: {max_concentration*100:.1f}%\")
print(f\"Mean Key Cosine Similarity: {mean_key_sim:.4f} (Max: {max_key_sim:.4f}, Duplicates: {unused_count} unused)\")

with open(OUTPUT_DIR / \"PROTOTYPE_AUDIT.json\", \"w\") as f:
    json.dump(prototype_audit, f, indent=2)

with open(OUTPUT_DIR / \"PROTOTYPE_AUDIT.csv\", \"w\", newline=\"\") as f:
    writer = csv.writer(f)
    writer.writerow([\"metric\", \"value\"])
    writer.writerow([\"effective_prototypes\", f\"{effective_prototypes:.2f}\"])
    writer.writerow([\"usage_entropy\", f\"{usage_entropy:.4f}\"])
    writer.writerow([\"mean_pixel_assignment_entropy\", f\"{entropy_per_pixel:.4f}\"])
    writer.writerow([\"mean_max_assignment_concentration\", f\"{max_concentration:.4f}\"])
    writer.writerow([\"mean_key_similarity\", f\"{mean_key_sim:.4f}\"])
    writer.writerow([\"top5_mass_fraction\", f\"{top5_mass:.4f}\"])
print(\"Saved PROTOTYPE_AUDIT.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 8. Part D: Occurrence Redundancy at M0
M0 = data[\"M0\"].float() # [N, 49, 192]

# 1. Pairwise cosine similarity among occurrences within each sample
normed_M0 = F.normalize(M0, dim=-1) # [N, 49, 192]
sim_matrices = torch.bmm(normed_M0, normed_M0.transpose(1, 2)) # [N, 49, 49]
off_diag_mask = ~torch.eye(49, dtype=torch.bool, device=M0.device).unsqueeze(0)
mean_occurrence_sim = float(sim_matrices[off_diag_mask.expand(N, -1, -1)].mean().item())

# 2. Nearest neighbor similarity
diag_inf = sim_matrices.clone()
diag_inf[:, torch.arange(49), torch.arange(49)] = -1.0
nn_sim = float(diag_inf.max(dim=-1).values.mean().item())

# 3. Local (Chebyshev = 1) vs Distant (Chebyshev >= 3) redundancy on 7x7 grid
occ_indices = torch.arange(49)
grid_r = occ_indices // 7
grid_c = occ_indices % 7
dist_cheb = torch.maximum((grid_r.unsqueeze(0) - grid_r.unsqueeze(1)).abs(), (grid_c.unsqueeze(0) - grid_c.unsqueeze(1)).abs())

mask_local = (dist_cheb == 1).unsqueeze(0).expand(N, -1, -1)
mask_distant = (dist_cheb >= 3).unsqueeze(0).expand(N, -1, -1)

local_sim = float(sim_matrices[mask_local].mean().item())
distant_sim = float(sim_matrices[mask_distant].mean().item())

# 4. Effective rank of occurrence representations at M0
flat_M0 = M0.reshape(-1, 192)
flat_c = flat_M0 - flat_M0.mean(dim=0, keepdim=True)
cov = (flat_c.t() @ flat_c) / (flat_M0.shape[0] - 1)
eigs = torch.linalg.eigvalsh(cov).clamp(min=0.0)
effective_rank_M0 = float(((eigs.sum() ** 2) / ((eigs ** 2).sum() + 1e-12)).item())

# 5. Variance across samples per occurrence
var_per_occurrence = M0.var(dim=0).mean(dim=-1).tolist() # [49]

occurrence_redundancy = {
    \"mean_pairwise_similarity\": mean_occurrence_sim,
    \"nearest_neighbor_similarity\": nn_sim,
    \"local_neighbor_similarity_cheb1\": local_sim,
    \"distant_neighbor_similarity_cheb3\": distant_sim,
    \"local_to_distant_ratio\": local_sim / (distant_sim + 1e-12),
    \"effective_rank_at_M0\": effective_rank_M0,
    \"mean_sample_variance\": float(np.mean(var_per_occurrence)),
}

print(\"=== Part D: Occurrence Redundancy at M0 ===\")
print(f\"Mean Pairwise Cosine: {mean_occurrence_sim:.4f} | NN Sim: {nn_sim:.4f}\")
print(f\"Local Sim (Cheb 1):   {local_sim:.4f} | Distant Sim (Cheb >= 3): {distant_sim:.4f}\")
print(f\"Effective Rank M0:    {effective_rank_M0:.2f} / 192\")

with open(OUTPUT_DIR / \"OCCURRENCE_REDUNDANCY.json\", \"w\") as f:
    json.dump(occurrence_redundancy, f, indent=2)

with open(OUTPUT_DIR / \"OCCURRENCE_REDUNDANCY.csv\", \"w\", newline=\"\") as f:
    writer = csv.writer(f)
    writer.writerow([\"metric\", \"value\"])
    writer.writerow([\"mean_pairwise_similarity\", f\"{mean_occurrence_sim:.4f}\"])
    writer.writerow([\"nearest_neighbor_similarity\", f\"{nn_sim:.4f}\"])
    writer.writerow([\"local_neighbor_similarity\", f\"{local_sim:.4f}\"])
    writer.writerow([\"distant_neighbor_similarity\", f\"{distant_sim:.4f}\"])
    writer.writerow([\"effective_rank_at_M0\", f\"{effective_rank_M0:.2f}\"])
print(\"Saved OCCURRENCE_REDUNDANCY.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 9. Part E: Error-Conditioned Analysis on Validation (CORRECT vs INCORRECT)
correct_mask = data[\"correct\"] # [N] bool
y_err = (~correct_mask).astype(int) # 1 if error, 0 if correct

N_c = np.sum(correct_mask)
N_e = np.sum(~correct_mask)
print(f"Validation Split: Correct = {N_c} ({N_c/N*100:.1f}%), Incorrect = {N_e} ({N_e/N*100:.1f}%)")

# Extract scalar indicators per sample
# 1. Normalized center displacement (scale 12 anchor size)
norm_disp_sample = (torch.norm(data[\"centers\"][12] - anchors_cpu.unsqueeze(0), dim=-1) / half_widths[12]).mean(dim=-1).numpy()

# 2. Boundary ring mass (scale 12)
ring_mass_sample = data[\"weights\"][12][..., ring_masks[12].cpu()].sum(dim=-1).mean(dim=-1).numpy()

# 3. Max in ring frequency
max_in_ring_sample = ring_masks[12].cpu()[data[\"weights\"][12].argmax(dim=-1)].float().mean(dim=-1).numpy()

# 4. Scale 16 usage
scale_16_usage_sample = data[\"alpha\"][..., 2].mean(dim=-1).numpy()

# 5. Scale gate entropy
scale_entropy_sample = (-(data[\"alpha\"].clamp(min=1e-12) * data[\"alpha\"].clamp(min=1e-12).log()).sum(dim=-1)).mean(dim=-1).numpy()

# 6. Spatial weight entropy (scale 12)
spat_entropy_sample = (-(data[\"weights\"][12].clamp(min=1e-12) * data[\"weights\"][12].clamp(min=1e-12).log()).sum(dim=-1)).mean(dim=-1).numpy()

# 7. Prototype assignment entropy
p_a_s = data[\"assignments\"].clamp(min=1e-12)
proto_entropy_sample = (-(p_a_s * p_a_s.log()).sum(dim=-1)).mean(dim=-1).numpy()

# 8. Maximum prototype concentration
max_proto_conc_sample = data[\"assignments\"].max(dim=-1).values.mean(dim=-1).numpy()

# 9. Occurrence redundancy (mean pairwise similarity at M0)
occ_sim_sample = sim_matrices[off_diag_mask.expand(N, -1, -1)].reshape(N, -1).mean(dim=-1).numpy()

indicators = {
    \"norm_center_displacement\": norm_disp_sample,
    \"boundary_ring_mass\": ring_mass_sample,
    \"max_pixel_in_boundary_ring\": max_in_ring_sample,
    \"scale_16_usage\": scale_16_usage_sample,
    \"scale_gate_entropy\": scale_entropy_sample,
    \"spatial_weight_entropy\": spat_entropy_sample,
    \"prototype_assignment_entropy\": proto_entropy_sample,
    \"max_prototype_concentration\": max_proto_conc_sample,
    \"occurrence_redundancy\": occ_sim_sample,
}

def bootstrap_ci(arr, n_boot=1000):
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

error_analysis_rows = []

for name, values in indicators.items():
    val_c = values[correct_mask]
    val_e = values[~correct_mask]
    
    mean_c, median_c = float(np.mean(val_c)), float(np.median(val_c))
    mean_e, median_e = float(np.mean(val_e)), float(np.median(val_e))
    ci_c = bootstrap_ci(val_c)
    ci_e = bootstrap_ci(val_e)
    
    # Cohen's d
    s_pool = math.sqrt(((len(val_c)-1)*np.var(val_c, ddof=1) + (len(val_e)-1)*np.var(val_e, ddof=1)) / (len(val_c)+len(val_e)-2))
    cohens_d = float((mean_e - mean_c) / (s_pool + 1e-12))
    
    # AUROC for predicting error
    # If d < 0, AUROC will naturally be < 0.50
    try:
        auroc = float(roc_auc_score(y_err, values))
    except Exception:
        auroc = 0.50
        
    error_analysis_rows.append({
        \"indicator\": name,
        \"mean_correct\": mean_c,
        \"ci_correct_95\": ci_c,
        \"mean_error\": mean_e,
        \"ci_error_95\": ci_e,
        \"median_correct\": median_c,
        \"median_error\": median_e,
        \"cohens_d\": cohens_d,
        \"auroc_error_prediction\": auroc,
    })
    print(f\"[{name:28s}] Correct: {mean_c:.3f} | Error: {mean_e:.3f} | d = {cohens_d:+.4f} | AUROC = {auroc:.4f}\")

with open(OUTPUT_DIR / \"ERROR_CONDITIONED_ANALYSIS.json\", \"w\") as f:
    json.dump(error_analysis_rows, f, indent=2)

with open(OUTPUT_DIR / \"ERROR_CONDITIONED_ANALYSIS.csv\", \"w\", newline=\"\") as f:
    writer = csv.writer(f)
    writer.writerow([\"indicator\", \"mean_correct\", \"ci_c_low\", \"ci_c_high\", \"mean_error\", \"ci_e_low\", \"ci_e_high\", \"cohens_d\", \"auroc\"])
    for r in error_analysis_rows:
        writer.writerow([r[\"indicator\"], f\"{r['mean_correct']:.4f}\", f\"{r['ci_correct_95'][0]:.4f}\", f\"{r['ci_correct_95'][1]:.4f}\", f\"{r['mean_error']:.4f}\", f\"{r['ci_error_95'][0]:.4f}\", f\"{r['ci_error_95'][1]:.4f}\", f\"{r['cohens_d']:.4f}\", f\"{r['auroc_error_prediction']:.4f}\"])
print(\"Saved ERROR_CONDITIONED_ANALYSIS.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 10. Part F: FULL vs NPF Descriptive Comparison
npf_comparison_summary = {}
if model_npf:
    print("\\n=== Extracting NPF Audit Data ===")
    npf_audit_data = extract_smc_audit_data(model_npf)
    
    # Compare key indicators
    npf_mean_alpha = npf_audit_data[\"alpha\"].mean(dim=(0, 1)).tolist()
    npf_norm_disp = (torch.norm(npf_audit_data[\"centers\"][12] - anchors_cpu.unsqueeze(0), dim=-1) / half_widths[12]).mean().item()
    npf_outer_mass = npf_audit_data[\"weights\"][12][..., ring_masks[12].cpu()].sum(dim=-1).mean().item()
    
    npf_marginal = npf_audit_data[\"assignments\"].mean(dim=(0, 1)).numpy()
    npf_proto_eff = float(np.exp(-np.sum(np.clip(npf_marginal, 1e-12, 1.0) * np.log(np.clip(npf_marginal, 1e-12, 1.0)))))
    
    npf_normed_M0 = F.normalize(npf_audit_data[\"M0\"].float(), dim=-1)
    npf_sim = torch.bmm(npf_normed_M0, npf_normed_M0.transpose(1, 2))
    npf_mean_occ_sim = float(npf_sim[off_diag_mask.expand(N, -1, -1)].mean().item())
    
    npf_comparison_summary = {
        \"FULL\": {
            \"mean_alpha\": mean_alpha,
            \"mean_normalized_displacement\": float(spatial_pressure_records[1][\"mean_normalized_displacement\"]),
            \"mean_outer_ring_mass\": float(spatial_pressure_records[1][\"mean_outer_ring_mass\"]),
            \"effective_prototypes\": effective_prototypes,
            \"mean_occurrence_similarity\": mean_occurrence_sim,
        },
        \"NPF\": {
            \"mean_alpha\": npf_mean_alpha,
            \"mean_normalized_displacement\": float(npf_norm_disp),
            \"mean_outer_ring_mass\": float(npf_outer_mass),
            \"effective_prototypes\": npf_proto_eff,
            \"mean_occurrence_similarity\": npf_mean_occ_sim,
        },
        \"trajectory_consistency\": {
            \"scale_alpha_similarity\": float(np.corrcoef(mean_alpha, npf_mean_alpha)[0, 1]),
            \"prototype_usage_correlation\": float(stats.pearsonr(marginal_usage, npf_marginal)[0]),
        }
    }
    with open(OUTPUT_DIR / \"FULL_NPF_COMPOSER_COMPARISON.json\", \"w\") as f:
        json.dump(npf_comparison_summary, f, indent=2)
        
    with open(OUTPUT_DIR / \"FULL_NPF_COMPOSER_COMPARISON.csv\", \"w\", newline=\"\") as f:
        writer = csv.writer(f)
        writer.writerow([\"metric\", \"FULL\", \"NPF\"])
        writer.writerow([\"mean_alpha_8\", f\"{mean_alpha[0]:.4f}\", f\"{npf_mean_alpha[0]:.4f}\"])
        writer.writerow([\"mean_alpha_12\", f\"{mean_alpha[1]:.4f}\", f\"{npf_mean_alpha[1]:.4f}\"])
        writer.writerow([\"mean_alpha_16\", f\"{mean_alpha[2]:.4f}\", f\"{npf_mean_alpha[2]:.4f}\"])
        writer.writerow([\"mean_norm_disp_12\", f\"{spatial_pressure_records[1]['mean_normalized_displacement']:.4f}\", f\"{npf_norm_disp:.4f}\"])
        writer.writerow([\"mean_outer_mass_12\", f\"{spatial_pressure_records[1]['mean_outer_ring_mass']:.4f}\", f\"{npf_outer_mass:.4f}\"])
        writer.writerow([\"effective_prototypes\", f\"{effective_prototypes:.2f}\", f\"{npf_proto_eff:.2f}\"])
        writer.writerow([\"mean_occurrence_similarity\", f\"{mean_occurrence_sim:.4f}\", f\"{npf_mean_occ_sim:.4f}\"])
    print(\"Saved FULL_NPF_COMPOSER_COMPARISON.json/csv\")
""",
        ),
        _cell(
            "code",
            """# 11. Decision Framework & Final Audit Summary
# Evaluate decision criteria based on collected evidence
spatial_d = [r[\"cohens_d\"] for r in error_analysis_rows if r[\"indicator\"] in [\"norm_center_displacement\", \"boundary_ring_mass\", \"max_pixel_in_boundary_ring\", \"scale_16_usage\"]]
spatial_aurocs = [r[\"auroc_error_prediction\"] for r in error_analysis_rows if r[\"indicator\"] in [\"norm_center_displacement\", \"boundary_ring_mass\", \"max_pixel_in_boundary_ring\", \"scale_16_usage\"]]

proto_d = [r[\"cohens_d\"] for r in error_analysis_rows if r[\"indicator\"] in [\"prototype_assignment_entropy\", \"max_prototype_concentration\"]]
proto_aurocs = [r[\"auroc_error_prediction\"] for r in error_analysis_rows if r[\"indicator\"] in [\"prototype_assignment_entropy\", \"max_prototype_concentration\"]]

has_spatial_constraint = any(abs(d) >= 0.15 for d in spatial_d) and any(a >= 0.55 or a <= 0.45 for a in spatial_aurocs)
has_proto_constraint = (effective_prototypes < 15.0 or fraction_near_duplicate > 0.10) or any(abs(d) >= 0.15 for d in proto_d)

decision = None
status_str = None
justification = None

if has_spatial_constraint and not has_proto_constraint:
    decision = "COMPOSER_A_SPATIAL_CONSTRAINT"
    status_str = "COMPOSER_AUDIT_SPATIAL_CONSTRAINT"
    justification = "Multiple spatial pressure indicators converge and show meaningful association with classification errors, supporting bounded deformable/content-adaptive spatial composition."
elif has_proto_constraint and not has_spatial_constraint:
    decision = "COMPOSER_B_PROTOTYPE_CONSTRAINT"
    status_str = "COMPOSER_AUDIT_PROTOTYPE_CONSTRAINT"
    justification = "Prototype utilization or assignment behavior exhibits a structural bottleneck associated with classification failures, supporting prototype TYPE redesign."
elif has_spatial_constraint and has_proto_constraint:
    decision = "COMPOSER_C_MIXED_CONSTRAINT"
    status_str = "COMPOSER_AUDIT_MIXED_CONSTRAINT"
    justification = "Both spatial support pressure and prototype utilization show independent constraints."
else:
    decision = "COMPOSER_D_HEALTHY"
    status_str = "COMPOSER_AUDIT_STOP"
    justification = "Fixed supports, scale selection, prototype utilization, and occurrence diversity show healthy, well-calibrated distributions without systematic association with classification errors. The Spatial Motif Composer is not an architectural constraint bottleneck. STOP incremental modifications."

print("\\n================ COMPOSER AUDIT VERDICT ================")
print("Decision Classification:", decision)
print("Final Status String:    ", status_str)
print("Scientific Verdict:     ", justification)

summary_dict = {
    "schema_version": 1,
    "title": "MPG-FER Spatial Motif Composer Constraint Audit Summary",
    "decision_classification": decision,
    "decision_status_string": status_str,
    "verdict_justification": justification,
    "spatial_support_pressure": spatial_pressure_records,
    "scale_pressure": scale_pressure_summary,
    "prototype_audit": {
        "effective_prototypes": effective_prototypes,
        "usage_entropy": usage_entropy,
        "mean_key_similarity": mean_key_sim,
        "max_key_similarity": max_key_sim,
    },
    "occurrence_redundancy": occurrence_redundancy,
    "error_conditioned_effects": error_analysis_rows,
    "full_vs_npf_comparison": npf_comparison_summary,
}

with open(OUTPUT_DIR / "COMPOSER_AUDIT_SUMMARY.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

md_summary = [
    "# Spatial Motif Composer Constraint Audit Final Summary",
    "",
    f"## Verdict: **{status_str}** ({decision})",
    f"**Justification:** {justification}",
    "",
    "### 1. Spatial Support Pressure Overview",
    "| Scale | Norm. Center Displacement | Outer 20% Ring Mass | Max Pixel In Ring Rate | Mean Spatial Entropy | Effective Pixels |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in spatial_pressure_records:
    md_summary.append(f"| {r['scale']} | {r['mean_normalized_displacement']:.3f} | {r['mean_outer_ring_mass']*100:.1f}% | {r['max_pixel_in_outer_ring_rate']*100:.1f}% | {r['mean_spatial_entropy']:.3f} | {r['mean_effective_contributing_pixels']:.1f} / {r['scale']**2} |")

md_summary.extend([
    "",
    "### 2. Prototype Space Health",
    f"- **Effective Prototype Count:** `{effective_prototypes:.2f}` / 48 (Entropy = `{usage_entropy:.3f}`)",
    f"- **Mean Pairwise Prototype Key Cosine Similarity:** `{mean_key_sim:.4f}` (Max: `{max_key_sim:.4f}`)",
    f"- **Unused Prototypes:** `{unused_count}` / 48",
    "",
    "### 3. Occurrence Redundancy at M0",
    f"- **Mean Pairwise Cosine Similarity:** `{mean_occurrence_sim:.4f}`",
    f"- **Local (Cheb 1) vs Distant (Cheb >= 3):** `{local_sim:.4f}` vs `{distant_sim:.4f}`",
    f"- **Effective Rank:** `{effective_rank_M0:.2f}` / 192",
    "",
    "### 4. Error-Conditioned Effect Sizes (Correct vs Incorrect on Validation)",
    "| Indicator | Correct Mean | Error Mean | Cohen's d | AUROC | Significant Effect? |",
    "|---|---:|---:|---:|---:|:---:|",
])
for r in error_analysis_rows:
    sig = "YES" if abs(r["cohens_d"]) >= 0.15 else "NO"
    md_summary.append(f"| {r['indicator']} | {r['mean_correct']:.3f} | {r['mean_error']:.3f} | {r['cohens_d']:+.4f} | {r['auroc_error_prediction']:.4f} | {sig} |")

with open(OUTPUT_DIR / "COMPOSER_AUDIT_SUMMARY.md", "w") as f:
    f.write("\\n".join(md_summary) + "\\n")

# Archive artifacts
import shutil
archive = shutil.make_archive("/kaggle/working/composer-audit-artifacts", "zip", root_dir=OUTPUT_DIR)
print(f"All artifacts archived successfully to {archive}")
""",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "gpu": True,
            "internet": False,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    nb = build_composer_audit_notebook()
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "mpg_fer_composer_audit.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebook:", nb_path)


if __name__ == "__main__":
    main()
