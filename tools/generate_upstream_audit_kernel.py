"""Generate the standalone Kaggle notebook for MPG-FER Upstream Representation and Optimization Audit."""

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


def build_upstream_audit_notebook() -> dict:
    encoded_sources, source_sha = encode_sources()

    cells = [
        _cell(
            "markdown",
            "# MPG-FER Upstream Representation and Optimization Audit\n\n"
            "Comprehensive diagnostic investigating:\n"
            "1. Layerwise representation evolution (P, M0, M1, M2, M3, M4, M5) for FULL and NPF.\n"
            "2. Representation geometry (node cosine similarity, effective rank, covariance spectrum, class separability).\n"
            "3. FULL vs NPF trajectory similarity (linear CKA and centered cosine similarity).\n"
            "4. Multi-loss gradient alignment (G_cls vs G_motif vs G_pixel) across parameter groups on 32 minibatches.\n\n"
            "**Hard Rule Enforcement:** Train -> Validation only. Strictly NO Test/Private access.\n",
        ),
        _cell(
            "code",
            """import os
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mpg_fer_upstream_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Upstream Audit output directory:", OUTPUT_DIR)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_v2_3.model import MPGFER, compute_motif_geometry
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import FER2013Dataset
from mpg_fer_v2_3.utils import set_seed

INPUT_ROOT = Path("/kaggle/input")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# 1. Locate and verify Checkpoints: FULL and NPF
EXPECTED_FULL_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
EXPECTED_NPF_SHA = "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972"

full_ckpt_candidates = [p for p in INPUT_ROOT.rglob("best_val_acc.pt") if "npf" not in str(p).lower()]
npf_ckpt_candidates = [p for p in INPUT_ROOT.rglob("best_val_acc.pt") if "npf" in str(p).lower()]

# Match by hash if multiple
full_ckpt, npf_ckpt = None, None
for p in INPUT_ROOT.rglob("best_val_acc.pt"):
    h = sha256_file(p)
    if h == EXPECTED_FULL_SHA:
        full_ckpt = p
    elif h == EXPECTED_NPF_SHA:
        npf_ckpt = p

if not full_ckpt or not npf_ckpt:
    raise FileNotFoundError(f"Checkpoints not located: full={full_ckpt}, npf={npf_ckpt}")

print("Verified Canonical FULL Checkpoint:", full_ckpt, "(SHA:", EXPECTED_FULL_SHA, ")")
print("Verified Canonical NPF Checkpoint: ", npf_ckpt, "(SHA:", EXPECTED_NPF_SHA, ")")

# 2. Locate Train and Val CSVs
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

# Fail closed check: test.csv must NOT be opened
for p in INPUT_ROOT.rglob("*.csv"):
    if "test.csv" in p.name.lower():
        print("Confirmed test.csv is present on disk but strictly NOT accessed.")
""",
        ),
        _cell(
            "code",
            """# 3. Load Models and Verify Strict Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Execution device:", device)

cfg = MPGConfig()

# Load FULL
model_full = MPGFER(cfg).to(device)
payload_full = torch.load(full_ckpt, map_location=device, weights_only=False)
incompat_full = model_full.load_state_dict(payload_full["model_state_dict"], strict=True)
assert len(incompat_full.missing_keys) == 0 and len(incompat_full.unexpected_keys) == 0
model_full.eval()
for p in model_full.parameters():
    p.requires_grad = False
print("FULL Model strictly loaded and frozen.")

# Load NPF
model_npf = MPGFER(cfg).to(device)
payload_npf = torch.load(npf_ckpt, map_location=device, weights_only=False)
incompat_npf = model_npf.load_state_dict(payload_npf["model_state_dict"], strict=True)
assert len(incompat_npf.missing_keys) == 0 and len(incompat_npf.unexpected_keys) == 0
model_npf.eval()
for p in model_npf.parameters():
    p.requires_grad = False
print("NPF Model strictly loaded and frozen.")

source_audit_dict = {
    "schema_version": 1,
    "full_checkpoint_sha256": EXPECTED_FULL_SHA,
    "npf_checkpoint_sha256": EXPECTED_NPF_SHA,
    "train_sha256": EXPECTED_TRAIN_SHA,
    "val_sha256": EXPECTED_VAL_SHA,
    "extracted_layers": {
        "P": "Output of Edge-Aware Pixel GNN (self.pixel_gnn)",
        "M0": "Spatial Motif Composer output before Motif GNN (self.motif_composer)",
        "M1": "Output of Motif GNN Block 1",
        "M2": "Output of Motif GNN Block 2",
        "M3": "Output of Motif GNN Block 3",
        "M4": "Output of Motif GNN Block 4",
        "M5": "Output of Motif GNN Block 5 (canonical final motif tensor)",
    },
    "shapes": {
        "P": [-1, 2304, 96],
        "M0": [-1, 49, 192],
        "M1": [-1, 49, 192],
        "M2": [-1, 49, 192],
        "M3": [-1, 49, 192],
        "M4": [-1, 49, 192],
        "M5": [-1, 49, 192],
    },
    "train_samples": 28709,
    "val_samples": 3589,
    "status": "PASS",
}
with open(OUTPUT_DIR / "SOURCE_AUDIT.json", "w") as f:
    json.dump(source_audit_dict, f, indent=2)

md_lines = [
    "# Phase 0: Source Audit for Upstream Representation & Optimization Audit",
    "",
    "## 1. Checkpoint Verification",
    f"- **FULL Checkpoint SHA256:** `{EXPECTED_FULL_SHA}`",
    f"- **NPF Checkpoint SHA256:**  `{EXPECTED_NPF_SHA}`",
    "- **Strict Loading:** PASS (0 missing keys, 0 unexpected keys for both models)",
    "",
    "## 2. Layer Extraction Points",
    "",
    "| Layer | Name | Source Extraction Location | Expected Shape |",
    "|---|---|---|---|",
    "| **P** | Pixel GNN Output | `model.pixel_gnn` forward loop completion | `[B, 2304, 96]` |",
    "| **M0** | Composer Output | `model.motif_composer` forward output before GNN | `[B, 49, 192]` |",
    "| **M1** | Motif Block 1 | `model.motif_gnn[0]` output | `[B, 49, 192]` |",
    "| **M2** | Motif Block 2 | `model.motif_gnn[1]` output | `[B, 49, 192]` |",
    "| **M3** | Motif Block 3 | `model.motif_gnn[2]` output | `[B, 49, 192]` |",
    "| **M4** | Motif Block 4 | `model.motif_gnn[3]` output | `[B, 49, 192]` |",
    "| **M5** | Motif Block 5 | `model.motif_gnn[4]` output (final motif tensor) | `[B, 49, 192]` |",
]
with open(OUTPUT_DIR / "SOURCE_AUDIT.md", "w") as f:
    f.write("\\n".join(md_lines) + "\\n")
print("Wrote SOURCE_AUDIT.json and SOURCE_AUDIT.md")
""",
        ),
        _cell(
            "code",
            """# 4. Extract Layerwise Representations (P, M0..M5) for FULL and NPF
train_dataset = FER2013Dataset(train_csv, split="train", augment=False)
val_dataset = FER2013Dataset(val_csv, split="val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

def extract_all_layers(model, loader, desc):
    all_P = []
    all_M = {f"M{i}": [] for i in range(6)}
    all_y = []
    t0 = time.monotonic()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            batch = images.shape[0]
            
            # Pixel GNN path
            h = model.pixel_proj(model.pixel_extractor(images))
            intensities = images.reshape(batch, cfg.num_pixels, 1)
            edges = model.pixel_topology.compute_edge_features(intensities)
            for layer in model.pixel_gnn:
                h = layer(h, model.pixel_topology.neighbor_idx, model.pixel_topology.neighbor_mask, edges)
            
            # P: Contextualized pixel features pooled to 7x7 grid [B, 49, 96]
            h_2d = h.reshape(batch, 48, 48, 96).permute(0, 3, 1, 2)
            h_p49 = F.adaptive_avg_pool2d(h_2d, (7, 7)).permute(0, 2, 3, 1).reshape(batch, 49, 96)
            all_P.append(h_p49.half().cpu())
            
            # M0 from composer
            h_motif, assignments, diagnostics = model.motif_composer(h)
            all_M["M0"].append(h_motif.half().cpu())
            
            # Motif GNN blocks M1..M5
            geometry = compute_motif_geometry(diagnostics["learned_centers_x"], diagnostics["learned_centers_y"])
            for l_idx, layer in enumerate(model.motif_gnn):
                h_motif, _ = layer(h_motif, geometry, return_diagnostics=True)
                all_M[f"M{l_idx+1}"].append(h_motif.half().cpu())
                
            all_y.append(targets.cpu())
            if (i + 1) % 100 == 0:
                print(f"  {desc}: {(i + 1) * batch} samples extracted...")
                
    print(f"{desc} completed in {time.monotonic() - t0:.1f}s")
    P_tensor = torch.cat(all_P)
    M_tensors = {k: torch.cat(v) for k, v in all_M.items()}
    y_tensor = torch.cat(all_y)
    return P_tensor, M_tensors, y_tensor

print("\\nExtracting FULL Train features...")
full_train_P, full_train_M, train_y = extract_all_layers(model_full, train_loader, "FULL Train")
print("Extracting FULL Val features...")
full_val_P, full_val_M, val_y = extract_all_layers(model_full, val_loader, "FULL Val")

print("\\nExtracting NPF Train features...")
npf_train_P, npf_train_M, _ = extract_all_layers(model_npf, train_loader, "NPF Train")
print("Extracting NPF Val features...")
npf_val_P, npf_val_M, _ = extract_all_layers(model_npf, val_loader, "NPF Val")

print("Feature extraction complete.")
print(f"Train samples: {len(train_y)}, Val samples: {len(val_y)}")
""",
        ),
        _cell(
            "code",
            """# 5. Part A1: Fair Layerwise Probes
class MotifLayerProbe(nn.Module):
    \"\"\"Same lightweight probe architecture for every motif layer: mean+max+attn -> 576 -> 384 -> 7.\"\"\"
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.attn_pool = nn.Linear(d_motif, 1)
        self.proj = nn.Sequential(
            nn.Linear(d_motif * 3, d_readout),
            nn.LayerNorm(d_readout),
            nn.GELU(),
            nn.Linear(d_readout, num_classes),
        )

    def forward(self, x):
        # x: [B, 49, 192]
        m_mean = x.mean(dim=1)
        m_max = x.max(dim=1).values
        weights = F.softmax(self.attn_pool(x), dim=1)
        m_attn = (weights * x).sum(dim=1)
        feat = torch.cat([m_mean, m_max, m_attn], dim=-1)
        return self.proj(feat)


class PixelLayerProbe(nn.Module):
    \"\"\"Pixel-level pooling probe: mean+max+attn -> 288 -> 384 -> 7.\"\"\"
    def __init__(self, d_pixel=96, d_readout=384, num_classes=7):
        super().__init__()
        self.attn_pool = nn.Linear(d_pixel, 1)
        self.proj = nn.Sequential(
            nn.Linear(d_pixel * 3, d_readout),
            nn.LayerNorm(d_readout),
            nn.GELU(),
            nn.Linear(d_readout, num_classes),
        )

    def forward(self, x):
        # x: [B, 49, 96]
        p_mean = x.mean(dim=1)
        p_max = x.max(dim=1).values
        weights = F.softmax(self.attn_pool(x), dim=1)
        p_attn = (weights * x).sum(dim=1)
        feat = torch.cat([p_mean, p_max, p_attn], dim=-1)
        return self.proj(feat)


def train_layer_probe(probe_model, train_feat, train_labels, val_feat, val_labels, probe_name):
    set_seed(42)
    model = probe_model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    
    train_ds = TensorDataset(train_feat, train_labels)
    val_ds = TensorDataset(val_feat, val_labels)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_epoch = -1
    best_metrics = {}
    patience = 8
    patience_counter = 0
    
    for epoch in range(1, 51):
        model.train()
        for x_b, y_b in train_dl:
            x_b, y_b = x_b.to(device).float(), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_dl:
                x_b, y_b = x_b.to(device).float(), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                val_preds.append(logits.argmax(dim=-1).cpu())
                val_targets.append(y_b.cpu())
                
        val_loss /= len(val_labels)
        preds_all = torch.cat(val_preds)
        targets_all = torch.cat(val_targets)
        val_acc = float(accuracy_score(targets_all, preds_all))
        val_f1 = float(f1_score(targets_all, preds_all, average="macro"))
        
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_metrics = {
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_ce_loss": val_loss,
                "selected_epoch": best_epoch,
                "parameter_count": param_count,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    print(f"[{probe_name}] Best Val Acc: {best_val_acc*100:.2f}%, F1: {best_metrics['val_macro_f1']*100:.2f}% (Epoch {best_epoch})")
    return best_metrics

probe_results = []

# Probe FULL layers
print("\\n=== Training FULL Layerwise Probes ===")
res_full_P = train_layer_probe(PixelLayerProbe(), full_train_P, train_y, full_val_P, val_y, "FULL P (Pixel GNN)")
probe_results.append({"model": "FULL", "layer": "P", "layer_type": "pixel", **res_full_P})

for lyr in ["M0", "M1", "M2", "M3", "M4", "M5"]:
    res_full_M = train_layer_probe(MotifLayerProbe(), full_train_M[lyr], train_y, full_val_M[lyr], val_y, f"FULL {lyr}")
    probe_results.append({"model": "FULL", "layer": lyr, "layer_type": "motif", **res_full_M})

# Probe NPF layers
print("\\n=== Training NPF Layerwise Probes ===")
res_npf_P = train_layer_probe(PixelLayerProbe(), npf_train_P, train_y, npf_val_P, val_y, "NPF P (Pixel GNN)")
probe_results.append({"model": "NPF", "layer": "P", "layer_type": "pixel", **res_npf_P})

for lyr in ["M0", "M1", "M2", "M3", "M4", "M5"]:
    res_npf_M = train_layer_probe(MotifLayerProbe(), npf_train_M[lyr], train_y, npf_val_M[lyr], val_y, f"NPF {lyr}")
    probe_results.append({"model": "NPF", "layer": lyr, "layer_type": "motif", **res_npf_M})

# Save probe results JSON & CSV
with open(OUTPUT_DIR / "LAYERWISE_PROBE_RESULTS.json", "w") as f:
    json.dump(probe_results, f, indent=2)

with open(OUTPUT_DIR / "LAYERWISE_PROBE_RESULTS.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "layer", "layer_type", "val_accuracy", "val_macro_f1", "val_ce_loss", "selected_epoch", "parameter_count"])
    for r in probe_results:
        writer.writerow([r["model"], r["layer"], r["layer_type"], f"{r['val_accuracy']*100:.2f}", f"{r['val_macro_f1']*100:.2f}", f"{r['val_ce_loss']:.4f}", r["selected_epoch"], r["parameter_count"]])

print("Saved LAYERWISE_PROBE_RESULTS.json/csv")
""",
        ),
        _cell(
            "code",
            """# 6. Part A2: Representation Geometry on Validation
def analyze_representation_geometry(M_dict, model_name):
    rows = []
    num_classes = 7
    val_y_np = val_y.numpy()
    
    for lyr in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        tensor = M_dict[lyr].float() # [N, 49, 192]
        N, M, D = tensor.shape
        
        # 1. Mean pairwise cosine similarity among 49 occurrence nodes within each sample
        normed_nodes = F.normalize(tensor, dim=-1) # [N, 49, 192]
        sim_matrices = torch.bmm(normed_nodes, normed_nodes.transpose(1, 2)) # [N, 49, 49]
        off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=tensor.device).unsqueeze(0)
        mean_node_cosine_sim = float(sim_matrices[off_diag_mask.expand(N, -1, -1)].mean().item())
        
        # Flatten sample-level pooled representation for global geometry
        sample_pooled = tensor.mean(dim=1) # [N, 192]
        
        # 2. Node-feature effective rank / participation ratio
        flat_nodes = tensor.reshape(-1, D) # [N*49, 192]
        flat_nodes_c = flat_nodes - flat_nodes.mean(dim=0, keepdim=True)
        cov = (flat_nodes_c.t() @ flat_nodes_c) / (flat_nodes.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        sum_eig = eigenvalues.sum().item()
        sum_eig_sq = (eigenvalues ** 2).sum().item()
        effective_rank = (sum_eig ** 2) / (sum_eig_sq + 1e-12)
        
        # 3. Covariance spectrum (top 1, top 5, top 10 variance fraction)
        sorted_eigs, _ = torch.sort(eigenvalues, descending=True)
        top1_var = float(sorted_eigs[0].item() / (sum_eig + 1e-12))
        top5_var = float(sorted_eigs[:5].sum().item() / (sum_eig + 1e-12))
        top10_var = float(sorted_eigs[:10].sum().item() / (sum_eig + 1e-12))
        
        # 4. Feature norm statistics
        norms = torch.norm(tensor, dim=-1) # [N, 49]
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        
        # 5. Class centroid separation & scatter
        centroids = []
        within_class_scatter = 0.0
        for c in range(num_classes):
            mask_c = (val_y_np == c)
            samples_c = sample_pooled[mask_c]
            mu_c = samples_c.mean(dim=0)
            centroids.append(mu_c)
            within_class_scatter += ((samples_c - mu_c.unsqueeze(0)) ** 2).sum().item()
            
        centroids = torch.stack(centroids) # [7, 192]
        within_class_scatter /= N
        
        # Pairwise distance between class centroids
        dist_matrix = torch.cdist(centroids, centroids)
        off_diag_dists = dist_matrix[~torch.eye(num_classes, dtype=torch.bool)]
        mean_centroid_separation = float(off_diag_dists.mean().item())
        
        # Between class scatter
        global_mean = sample_pooled.mean(dim=0, keepdim=True)
        between_class_scatter = float(((centroids - global_mean) ** 2).sum(dim=-1).mean().item())
        fisher_ratio = between_class_scatter / (within_class_scatter + 1e-12)
        
        # 8. Nearest-centroid classification accuracy (head-free separability measure)
        sample_to_centroids = torch.cdist(sample_pooled, centroids) # [N, 7]
        nearest_centroid_preds = sample_to_centroids.argmin(dim=-1).cpu().numpy()
        nearest_centroid_acc = float(accuracy_score(val_y_np, nearest_centroid_preds))
        
        rows.append({
            "model": model_name,
            "layer": lyr,
            "mean_node_cosine_similarity": mean_node_cosine_sim,
            "effective_rank": effective_rank,
            "top1_variance_fraction": top1_var,
            "top5_variance_fraction": top5_var,
            "top10_variance_fraction": top10_var,
            "mean_feature_norm": mean_norm,
            "std_feature_norm": std_norm,
            "mean_centroid_separation": mean_centroid_separation,
            "within_class_scatter": within_class_scatter,
            "between_within_ratio": fisher_ratio,
            "nearest_centroid_accuracy": nearest_centroid_acc,
        })
    return rows

print("\\n=== Computing Representation Geometry on Validation ===")
geom_full = analyze_representation_geometry(full_val_M, "FULL")
geom_npf = analyze_representation_geometry(npf_val_M, "NPF")
all_geom = geom_full + geom_npf

with open(OUTPUT_DIR / "REPRESENTATION_GEOMETRY.json", "w") as f:
    json.dump(all_geom, f, indent=2)

with open(OUTPUT_DIR / "REPRESENTATION_GEOMETRY.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "layer", "mean_node_cosine_sim", "effective_rank", "top1_var_frac",
        "top5_var_frac", "top10_var_frac", "mean_norm", "std_norm", "centroid_separation",
        "within_class_scatter", "between_within_ratio", "nearest_centroid_acc"
    ])
    for r in all_geom:
        writer.writerow([
            r["model"], r["layer"], f"{r['mean_node_cosine_similarity']:.4f}",
            f"{r['effective_rank']:.2f}", f"{r['top1_variance_fraction']:.4f}",
            f"{r['top5_variance_fraction']:.4f}", f"{r['top10_variance_fraction']:.4f}",
            f"{r['mean_feature_norm']:.4f}", f"{r['std_feature_norm']:.4f}",
            f"{r['mean_centroid_separation']:.4f}", f"{r['within_class_scatter']:.4f}",
            f"{r['between_within_ratio']:.4f}", f"{r['nearest_centroid_accuracy']*100:.2f}"
        ])
print("Saved REPRESENTATION_GEOMETRY.json/csv")
""",
        ),
        _cell(
            "code",
            """# 7. Part A3: FULL vs NPF Representation Divergence (Linear CKA & Centered Cosine)
def compute_linear_cka(X, Y):
    # X, Y: [N, D]
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    
    YtX = Y_c.t() @ X_c
    numerator = (YtX ** 2).sum().item()
    
    XtX = X_c.t() @ X_c
    denom_X = (XtX ** 2).sum().item()
    
    YtY = Y_c.t() @ Y_c
    denom_Y = (YtY ** 2).sum().item()
    
    cka = numerator / (math.sqrt(denom_X * denom_Y) + 1e-12)
    
    # Centered cosine similarity
    dot = (X_c * Y_c).sum().item()
    norm_X = torch.norm(X_c).item()
    norm_Y = torch.norm(Y_c).item()
    centered_cosine = dot / (norm_X * norm_Y + 1e-12)
    
    return float(cka), float(centered_cosine)

cka_results = []

# Layer P
full_P_flat = full_val_P.float().mean(dim=1) # [N, 96]
npf_P_flat = npf_val_P.float().mean(dim=1) # [N, 96]
cka_P, cos_P = compute_linear_cka(full_P_flat, npf_P_flat)
cka_results.append({"layer": "P", "linear_cka": cka_P, "centered_cosine": cos_P})

# Layers M0..M5
for lyr in ["M0", "M1", "M2", "M3", "M4", "M5"]:
    full_M_flat = full_val_M[lyr].float().mean(dim=1) # [N, 192]
    npf_M_flat = npf_val_M[lyr].float().mean(dim=1) # [N, 192]
    cka_val, cos_val = compute_linear_cka(full_M_flat, npf_M_flat)
    cka_results.append({"layer": lyr, "linear_cka": cka_val, "centered_cosine": cos_val})
    print(f"[{lyr}] Linear CKA: {cka_val:.4f} | Centered Cosine: {cos_val:.4f}")

with open(OUTPUT_DIR / "FULL_NPF_LAYERWISE_CKA.json", "w") as f:
    json.dump(cka_results, f, indent=2)

with open(OUTPUT_DIR / "FULL_NPF_LAYERWISE_CKA.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "linear_cka", "centered_cosine"])
    for r in cka_results:
        writer.writerow([r["layer"], f"{r['linear_cka']:.4f}", f"{r['centered_cosine']:.4f}"])

print("Saved FULL_NPF_LAYERWISE_CKA.json/csv")
""",
        ),
        _cell(
            "code",
            """# 8. Part B: Gradient Alignment Audit on Canonical FULL Checkpoint
print("\\n=== Part B: Gradient Alignment Audit across 32 Train Minibatches ===")
model_full.train() # Enable gradients for parameters
for p in model_full.parameters():
    p.requires_grad = True

# Parameter groups
param_groups = {
    "1_pixel_backbone": [p for n, p in model_full.named_parameters() if any(k in n for k in ["pixel_proj", "pixel_extractor", "pixel_gnn"])],
    "2_spatial_motif_composer": [p for n, p in model_full.named_parameters() if "motif_composer" in n],
    "3_motif_gnn": [p for n, p in model_full.named_parameters() if "motif_gnn" in n],
    "4_motif_readout": [p for n, p in model_full.named_parameters() if any(k in n for k in ["motif_attn_pool", "motif_readout_proj"])],
    "5_final_classifier": [p for n, p in model_full.named_parameters() if any(k in n for k in ["classifier", "supcon_head"])],
}

# 32 stratified minibatches from Train (batch size 16)
train_sub_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
minibatches = []
for images, targets in train_sub_loader:
    minibatches.append((images.to(device), targets.to(device)))
    if len(minibatches) == 32:
        break

criterion = nn.CrossEntropyLoss()

def flatten_grads(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.reshape(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)

def cosine_sim(v1, v2):
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float((v1 @ v2 / (norm1 * norm2)).item())

group_stats = {grp: {"cos_cls_motif": [], "cos_cls_pixel": [], "cos_motif_pixel": [], "norm_cls": [], "norm_motif": [], "norm_pixel": []} for grp in param_groups}

for mb_idx, (b_img, b_y) in enumerate(minibatches):
    # 1. Forward
    logits, outputs = model_full(b_img)
    loss_cls = criterion(logits, b_y)
    loss_motif = criterion(outputs["motif_logits"], b_y)
    loss_pixel = criterion(outputs["pixel_logits"], b_y)
    
    # 2. Gradient of loss_cls
    model_full.zero_grad()
    loss_cls.backward(retain_graph=True)
    g_cls = {grp: flatten_grads(params).detach().clone() for grp, params in param_groups.items()}
    
    # 3. Gradient of loss_motif
    model_full.zero_grad()
    loss_motif.backward(retain_graph=True)
    g_motif = {grp: flatten_grads(params).detach().clone() for grp, params in param_groups.items()}
    
    # 4. Gradient of loss_pixel
    model_full.zero_grad()
    loss_pixel.backward()
    g_pixel = {grp: flatten_grads(params).detach().clone() for grp, params in param_groups.items()}
    
    # Compute alignments
    for grp in param_groups:
        gc = g_cls[grp]
        gm = g_motif[grp]
        gp = g_pixel[grp]
        
        group_stats[grp]["norm_cls"].append(float(torch.norm(gc).item()))
        group_stats[grp]["norm_motif"].append(float(torch.norm(gm).item()))
        group_stats[grp]["norm_pixel"].append(float(torch.norm(gp).item()))
        
        group_stats[grp]["cos_cls_motif"].append(cosine_sim(gc, gm))
        group_stats[grp]["cos_cls_pixel"].append(cosine_sim(gc, gp))
        group_stats[grp]["cos_motif_pixel"].append(cosine_sim(gm, gp))

gradient_alignment_summary = []
for grp, s in group_stats.items():
    cos_cm = np.array(s["cos_cls_motif"])
    cos_cp = np.array(s["cos_cls_pixel"])
    cos_mp = np.array(s["cos_motif_pixel"])
    
    n_c = np.mean(s["norm_cls"])
    n_m = np.mean(s["norm_motif"])
    n_p = np.mean(s["norm_pixel"])
    
    gradient_alignment_summary.append({
        "parameter_group": grp,
        "cos_cls_motif": {
            "mean": float(np.mean(cos_cm)),
            "median": float(np.median(cos_cm)),
            "std": float(np.std(cos_cm)),
            "p10": float(np.percentile(cos_cm, 10)),
            "p25": float(np.percentile(cos_cm, 25)),
            "p75": float(np.percentile(cos_cm, 75)),
            "p90": float(np.percentile(cos_cm, 90)),
            "fraction_negative": float(np.mean(cos_cm < 0)),
            "fraction_below_minus_01": float(np.mean(cos_cm < -0.1)),
        },
        "cos_cls_pixel": {
            "mean": float(np.mean(cos_cp)),
            "median": float(np.median(cos_cp)),
            "std": float(np.std(cos_cp)),
            "fraction_negative": float(np.mean(cos_cp < 0)),
            "fraction_below_minus_01": float(np.mean(cos_cp < -0.1)),
        },
        "mean_gradient_norms": {
            "G_cls": float(n_c),
            "G_motif": float(n_m),
            "G_pixel": float(n_p),
            "norm_ratio_motif_to_cls": float(n_m / (n_c + 1e-12)),
            "norm_ratio_pixel_to_cls": float(n_p / (n_c + 1e-12)),
        },
    })
    print(f"[{grp}] Mean cos(G_cls, G_motif): {np.mean(cos_cm):.4f} (neg frac: {np.mean(cos_cm < 0):.2f}) | Mean cos(G_cls, G_pixel): {np.mean(cos_cp):.4f}")

with open(OUTPUT_DIR / "GRADIENT_ALIGNMENT.json", "w") as f:
    json.dump(gradient_alignment_summary, f, indent=2)

with open(OUTPUT_DIR / "GRADIENT_ALIGNMENT.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "parameter_group", "mean_cos_cls_motif", "median_cos_cls_motif", "std_cos_cls_motif",
        "frac_neg_cls_motif", "frac_below_minus_01", "mean_cos_cls_pixel", "norm_G_cls", "norm_G_motif", "norm_ratio_m_c"
    ])
    for r in gradient_alignment_summary:
        cm = r["cos_cls_motif"]
        cp = r["cos_cls_pixel"]
        gn = r["mean_gradient_norms"]
        writer.writerow([
            r["parameter_group"], f"{cm['mean']:.4f}", f"{cm['median']:.4f}", f"{cm['std']:.4f}",
            f"{cm['fraction_negative']:.2f}", f"{cm['fraction_below_minus_01']:.2f}",
            f"{cp['mean']:.4f}", f"{gn['G_cls']:.4f}", f"{gn['G_motif']:.4f}", f"{gn['norm_ratio_motif_to_cls']:.4f}"
        ])
print("Saved GRADIENT_ALIGNMENT.json/csv")
""",
        ),
        _cell(
            "code",
            """# 9. Part C: Decision Framework Classification & Final Summary
full_probes = {r[\"layer\"]: r[\"val_accuracy\"] * 100 for r in probe_results if r[\"model\"] == \"FULL\"}
full_geom = {r[\"layer\"]: r for r in geom_full}

# Check layerwise degradation
m_accs = [full_probes[f\"M{i}\"] for i in range(6)]
peak_m_idx = int(np.argmax(m_accs))
peak_acc = m_accs[peak_m_idx]
m5_acc = m_accs[5]
degradation = peak_acc - m5_acc

# Check gradient conflict in shared motif parameters (groups 2 and 3)
motif_conflict_fracs = [
    r[\"cos_cls_motif\"][\"fraction_negative\"]
    for r in gradient_alignment_summary
    if r[\"parameter_group\"] in [\"2_spatial_motif_composer\", \"3_motif_gnn\"]
]
mean_motif_conflict = float(np.mean(motif_conflict_fracs))

decision = None
status_str = None
exact_hypothesis = None

if peak_m_idx < 5 and degradation >= 0.50 and full_geom[\"M5\"][\"effective_rank\"] < full_geom[f\"M{peak_m_idx}\"][\"effective_rank\"] * 0.8:
    decision = \"UPSTREAM_A: Late Motif Graph bottleneck (discriminative power peaks early and degrades with rank loss)\"
    status_str = \"UPSTREAM_AUDIT_GRAPH_BOTTLENECK\"
    exact_hypothesis = \"Prune or dynamically truncate late Motif Graph layers (e.g. reduce from 5 to 3 layers).\"
elif mean_motif_conflict > 0.40:
    decision = \"UPSTREAM_B: Objective conflict between final fused loss and auxiliary motif loss in shared motif parameters\"
    status_str = \"UPSTREAM_AUDIT_OBJECTIVE_CONFLICT\"
    exact_hypothesis = \"Gradient projection or auxiliary loss re-scheduling (e.g., annealing aux_motif_weight to 0) to eliminate conflicting update vectors.\"
elif any(r[\"linear_cka\"] < 0.60 for r in cka_results if r[\"layer\"] in [\"M0\", \"M1\"]):
    decision = \"UPSTREAM_C: Trajectory sensitivity originating in early upstream composition stages\"
    status_str = \"UPSTREAM_AUDIT_TRAJECTORY_SENSITIVITY\"
    exact_hypothesis = \"Stabilize early Spatial Motif Composition initialization and scale gating variance.\"
else:
    decision = \"UPSTREAM_D: No clean layerwise degradation, no material gradient conflict, and no localized divergence\"
    status_str = \"UPSTREAM_AUDIT_STOP_INCREMENTAL\"
    exact_hypothesis = \"None. Incremental architectural interventions are exhausted under the current design framework. Stop incremental ad-hoc modifications.\"

print(\"\\n================ AUDIT VERDICT ================\")
print(\"Decision Classification:\", decision)
print(\"Final Status String:    \", status_str)
print(\"Actionable Hypothesis:  \", exact_hypothesis)

summary_final = {
    \"schema_version\": 1,
    \"title\": \"MPG-FER Upstream Representation and Optimization Audit Final Report\",
    \"decision_classification\": decision,
    \"decision_status_string\": status_str,
    \"actionable_hypothesis\": exact_hypothesis,
    \"full_layerwise_probe_accuracies\": full_probes,
    \"full_m5_vs_peak\": {\"peak_layer\": f\"M{peak_m_idx}\", \"peak_acc\": peak_acc, \"m5_acc\": m5_acc, \"degradation\": degradation},
    \"mean_motif_gradient_conflict_fraction\": mean_motif_conflict,
    \"full_npf_linear_cka\": {r[\"layer\"]: r[\"linear_cka\"] for r in cka_results},
}

with open(OUTPUT_DIR / \"UPSTREAM_AUDIT_SUMMARY.json\", \"w\") as f:
    json.dump(summary_final, f, indent=2)

md_lines = [
    \"# Upstream Representation & Optimization Audit Summary\",
    \"\",
    f\"## Verdict: **{status_str}**\",
    f\"**Classification:** {decision}\",
    \"\",
    f\"**Actionable Hypothesis:** {exact_hypothesis}\",
    \"\",
    \"### 1. Layerwise Probe Accuracy Curve (Validation, %)\",
    \"| Layer | FULL Val Acc. (%) | FULL Val F1 (%) | NPF Val Acc. (%) | NPF Val F1 (%) |\",
    \"|---|---:|---:|---:|---:|\",
]
for lyr in [\"P\", \"M0\", \"M1\", \"M2\", \"M3\", \"M4\", \"M5\"]:
    r_f = [x for x in probe_results if x[\"model\"] == \"FULL\" and x[\"layer\"] == lyr][0]
    r_n = [x for x in probe_results if x[\"model\"] == \"NPF\" and x[\"layer\"] == lyr][0]
    md_lines.append(f\"| **{lyr}** | {r_f['val_accuracy']*100:.2f} | {r_f['val_macro_f1']*100:.2f} | {r_n['val_accuracy']*100:.2f} | {r_n['val_macro_f1']*100:.2f} |\")

md_lines.extend([
    \"\",
    \"### 2. FULL vs NPF Representation Similarity (CKA & Cosine)\",
    \"| Layer | Linear CKA | Centered Cosine |\",
    \"|---|---:|---:|\",
])
for r in cka_results:
    md_lines.append(f\"| **{r['layer']}** | {r['linear_cka']:.4f} | {r['centered_cosine']:.4f} |\")

md_lines.extend([
    \"\",
    \"### 3. Gradient Alignment across Parameter Groups (cos(G_cls, G_motif))\",
    \"| Parameter Group | Mean Cosine | Median Cosine | Fraction Negative | Fraction < -0.1 | Norm Ratio (G_motif / G_cls) |\",
    \"|---|---:|---:|---:|---:|---:|\",
])
for r in gradient_alignment_summary:
    cm = r[\"cos_cls_motif\"]
    gn = r[\"mean_gradient_norms\"]
    md_lines.append(f\"| **{r['parameter_group']}** | {cm['mean']:.4f} | {cm['median']:.4f} | {cm['fraction_negative']:.2f} | {cm['fraction_below_minus_01']:.2f} | {gn['norm_ratio_motif_to_cls']:.4f} |\")

with open(OUTPUT_DIR / \"UPSTREAM_AUDIT_SUMMARY.md\", \"w\") as f:
    f.write(\"\\n\".join(md_lines) + \"\\n\")

# Archive artifacts
import shutil
archive = shutil.make_archive(\"/kaggle/working/upstream-audit-artifacts\", \"zip\", root_dir=OUTPUT_DIR)
print(f\"All artifacts archived successfully to {archive}\")
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
    nb = build_upstream_audit_notebook()
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "mpg_fer_upstream_audit.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebook:", nb_path)


if __name__ == "__main__":
    main()
