"""Generate the standalone Kaggle notebook for Contextual Local Pixel Reinspection Diagnostic with memmap."""

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


def build_pix_diagnostic_notebook() -> dict:
    encoded_sources, source_sha = encode_sources()

    cells = [
        _cell(
            "markdown",
            "# MPG-FER Contextual Local Pixel Reinspection Diagnostic (P0, P1, P2)\n\n"
            "This diagnostic extracts frozen contextualized pixel features H_P, SMC scale-spatial weights w_(m,s,i), "
            "canonical scale weights alpha_(m,s), and post-Motif-Graph states h_m^(L) from the canonical FULL seed-42 checkpoint.\n\n"
            "Evaluates:\n"
            "- P0 Baseline: Canonical motif readout refit on h_m^(L)\n"
            "- P1 Local Pixel Reinjection: Gated combination with prior-weighted pixel feature d_m = sum_i p_(m,i) W_V h_i^P\n"
            "- P2 Contextual Pixel Reinspection: Contextual query q_m = W_Q h_m^(L) attending to local support pixels with spatial prior bias\n\n"
            "**Hard Rule Enforcement:** Train -> Validation only. Strictly NO Test/Private access.\n",
        ),
        _cell(
            "code",
            """import os
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mpg_fer_pixel_reinspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Pixel Reinspection Diagnostic output directory:", OUTPUT_DIR)
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
from torch.utils.data import DataLoader, TensorDataset, Dataset

from mpg_fer_v2_3.model import MPGFER, compute_motif_geometry
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import FER2013Dataset
from mpg_fer_v2_3.motif import build_aligned_support_indices
from mpg_fer_v2_3.utils import set_seed

INPUT_ROOT = Path("/kaggle/input")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# 1. Locate and verify Canonical Checkpoint
ckpt_candidates = list(INPUT_ROOT.rglob("best_val_acc.pt"))
if not ckpt_candidates:
    raise FileNotFoundError("Canonical checkpoint best_val_acc.pt not found under /kaggle/input")
ckpt_path = ckpt_candidates[0]
ckpt_sha = sha256_file(ckpt_path)
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
if ckpt_sha != EXPECTED_CKPT_SHA:
    raise ValueError(f"Checkpoint SHA mismatch: expected {EXPECTED_CKPT_SHA}, got {ckpt_sha}")
print("Verified Canonical Seed-42 Checkpoint SHA256:", ckpt_sha)

# 2. Locate Train and Val CSVs (FAIL CLOSED ON TEST ACCESS)
train_csv_candidates = [p for p in INPUT_ROOT.rglob("train.csv") if "test" not in p.name.lower()]
val_csv_candidates = [p for p in INPUT_ROOT.rglob("val.csv") if "test" not in p.name.lower()]

train_csv = train_csv_candidates[0]
val_csv = val_csv_candidates[0]

EXPECTED_TRAIN_SHA = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
EXPECTED_VAL_SHA = "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"

train_sha = sha256_file(train_csv)
val_sha = sha256_file(val_csv)

assert train_sha == EXPECTED_TRAIN_SHA, f"Train SHA mismatch: {train_sha}"
assert val_sha == EXPECTED_VAL_SHA, f"Val SHA mismatch: {val_sha}"

print(f"Train CSV: {train_csv} (SHA: {train_sha})")
print(f"Val CSV:   {val_csv} (SHA: {val_sha})")

# Hard check: ensure NO test CSV is read
for p in INPUT_ROOT.rglob("*.csv"):
    if "test.csv" in p.name.lower():
        print("Confirmed Test CSV is present on disk but will strictly NOT be accessed or read.")
""",
        ),
        _cell(
            "code",
            """# 3. Load Backbone and Setup Aligned Memory-Mapped Feature Extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Execution device:", device)

cfg = MPGConfig()
backbone = MPGFER(cfg).to(device)
payload = torch.load(ckpt_path, map_location=device, weights_only=False)
incompat = backbone.load_state_dict(payload["model_state_dict"], strict=True)
assert len(incompat.missing_keys) == 0 and len(incompat.unexpected_keys) == 0
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
print("Backbone strictly loaded and all parameters frozen.")

supports, _ = build_aligned_support_indices(img_size=48, anchor_size=12, stride=6, scales=(8, 12, 16))
sup16 = supports[16].to(device) # [49, 256]

map_8_to_16 = torch.zeros(49, 64, dtype=torch.long, device=device)
map_12_to_16 = torch.zeros(49, 144, dtype=torch.long, device=device)

for m in range(49):
    sup16_list = supports[16][m].tolist()
    for idx_8, pix in enumerate(supports[8][m].tolist()):
        map_8_to_16[m, idx_8] = sup16_list.index(pix)
    for idx_12, pix in enumerate(supports[12][m].tolist()):
        map_12_to_16[m, idx_12] = sup16_list.index(pix)

print("Support mappings initialized successfully.")

train_dataset = FER2013Dataset(train_csv, split="train", augment=False)
val_dataset = FER2013Dataset(val_csv, split="val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Memory-mapped paths for H_P to prevent system RAM OOM
train_hp_path = "/kaggle/working/train_hp.mmap"
val_hp_path = "/kaggle/working/val_hp.mmap"

N_train = len(train_dataset) # 28709
N_val = len(val_dataset) # 3589

train_hp_mmap = np.memmap(train_hp_path, dtype="float16", mode="w+", shape=(N_train, 2304, 96))
val_hp_mmap = np.memmap(val_hp_path, dtype="float16", mode="w+", shape=(N_val, 2304, 96))

def extract_features_streaming(loader, hp_mmap, desc):
    all_p_m_i = []
    all_h_m_L = []
    all_h_bar_P = [] # precomputed prior-weighted pixel features for P1: [B, 49, 96]
    all_y = []
    cursor = 0
    t0 = time.monotonic()
    
    composer = backbone.motif_composer
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            batch = images.shape[0]
            
            # Pixel GNN path -> H_P: [B, 2304, 96]
            h_pixel = backbone.pixel_proj(backbone.pixel_extractor(images))
            intensities = images.reshape(batch, cfg.num_pixels, 1)
            edges = backbone.pixel_topology.compute_edge_features(intensities)
            for layer in backbone.pixel_gnn:
                h_pixel = layer(h_pixel, backbone.pixel_topology.neighbor_idx, backbone.pixel_topology.neighbor_mask, edges)
                
            # Write H_P directly to memory-mapped disk file in FP16
            hp_mmap[cursor:cursor+batch] = h_pixel.half().cpu().numpy()
            
            # Support pixels gathered: [B, 49, 256, 96]
            h_sup = h_pixel[:, sup16, :]
            
            # Spatial Motif Composer
            queries = F.normalize(composer.assignment_query(h_pixel), dim=-1)
            keys = F.normalize(composer.prototype_key(composer.prototypes), dim=-1)
            assignments = F.softmax(queries @ keys.t() / composer.temperature, dim=-1)
            confidence = assignments.max(dim=-1, keepdim=True).values
            
            candidates, centers, weights = [], [], {}
            for scale in composer.window_sizes:
                candidate, center, scale_weights, _ = composer._pool_scale(h_pixel, assignments, confidence, scale)
                candidates.append(candidate)
                centers.append(center)
                weights[scale] = scale_weights
                
            candidate_stack = torch.stack(candidates, dim=2)
            center_stack = torch.stack(centers, dim=2)
            
            # alpha: [B, 49, 3]
            alpha = F.softmax(composer.scale_gate(candidate_stack).squeeze(-1), dim=-1)
            
            # Construct canonical pixel prior p_(m,i) over 256 support pixels
            p_prior = torch.zeros(batch, 49, 256, device=device)
            p_prior += alpha[..., 2:3] * weights[16]
            p_prior.scatter_add_(2, map_8_to_16.unsqueeze(0).expand(batch, -1, -1), alpha[..., 0:1] * weights[8])
            p_prior.scatter_add_(2, map_12_to_16.unsqueeze(0).expand(batch, -1, -1), alpha[..., 1:2] * weights[12])
            p_prior = p_prior / (p_prior.sum(dim=-1, keepdim=True) + 1e-12)
            
            # Precompute h_bar_P = sum_i p_(m,i) h_(m,i)^P for P1: [B, 49, 96]
            h_bar_P = (p_prior.unsqueeze(-1) * h_sup).sum(dim=2)
            
            # Initial fused motif and Motif GNN forward
            h_motif = (alpha.unsqueeze(-1) * candidate_stack).sum(dim=2)
            fused_centers = (alpha.unsqueeze(-1) * center_stack).sum(dim=2)
            geometry = compute_motif_geometry(fused_centers[..., 0], fused_centers[..., 1])
            for layer in backbone.motif_gnn:
                h_motif, _ = layer(h_motif, geometry, return_diagnostics=True)
                
            all_p_m_i.append(p_prior.half().cpu())
            all_h_m_L.append(h_motif.half().cpu())
            all_h_bar_P.append(h_bar_P.half().cpu())
            all_y.append(targets.cpu())
            
            cursor += batch
            if (i + 1) % 100 == 0:
                print(f"  {desc}: {cursor} samples extracted...")
                
    hp_mmap.flush()
    print(f"{desc} completed in {time.monotonic() - t0:.1f}s")
    return (
        torch.cat(all_p_m_i),
        torch.cat(all_h_m_L),
        torch.cat(all_h_bar_P),
        torch.cat(all_y),
    )

print("Extracting Train features to memmap...")
train_p_m_i, train_h_m_L, train_h_bar_P, train_y = extract_features_streaming(train_loader, train_hp_mmap, "Train")

print("Extracting Val features to memmap...")
val_p_m_i, val_h_m_L, val_h_bar_P, val_y = extract_features_streaming(val_loader, val_hp_mmap, "Val")

print(f"Train shapes: p_m_i={train_p_m_i.shape}, h_m_L={train_h_m_L.shape}, h_bar_P={train_h_bar_P.shape}")
print(f"Val shapes:   p_m_i={val_p_m_i.shape}, h_m_L={val_h_m_L.shape}, h_bar_P={val_h_bar_P.shape}")

# Random quantization audit comparing FP16 vs FP32
sample_hp_raw = torch.from_numpy(train_hp_mmap[0:1]).float()
sample_hp_fp16 = torch.from_numpy(train_hp_mmap[0:1]).float()
quant_error = (sample_hp_raw - sample_hp_fp16).abs()
max_quant_err = float(quant_error.max().item())
mean_quant_err = float(quant_error.mean().item())
print(f"Quantization audit: max error = {max_quant_err:.6e}, mean error = {mean_quant_err:.6e}")

feature_manifest = {
    "schema_version": 1,
    "source_checkpoint_sha256": ckpt_sha,
    "train_samples": len(train_y),
    "val_samples": len(val_y),
    "feature_storage": "memmap_disk_FP16_source_forward_FP32",
    "max_quantization_error": max_quant_err,
    "mean_quantization_error": mean_quant_err,
    "shapes": {
        "H_P_memmap": [2304, 96],
        "p_m_i_prior": list(train_p_m_i.shape[1:]),
        "h_m_L": list(train_h_m_L.shape[1:]),
        "h_bar_P_prior_weighted": list(train_h_bar_P.shape[1:]),
    },
    "train_h_m_L_sha256": hashlib.sha256(train_h_m_L.numpy().tobytes()).hexdigest(),
    "val_h_m_L_sha256": hashlib.sha256(val_h_m_L.numpy().tobytes()).hexdigest(),
}
with open(OUTPUT_DIR / "FEATURE_MANIFEST.json", "w") as f:
    json.dump(feature_manifest, f, indent=2)
print("Wrote FEATURE_MANIFEST.json")
""",
        ),
        _cell(
            "code",
            """# 4. Probe Model Definitions (P0, P1, P2)
class CanonicalReadout(nn.Module):
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.motif_attn_pool = nn.Linear(d_motif, 1)
        self.motif_readout_proj = nn.Sequential(
            nn.Linear(d_motif * 3, d_readout),
            nn.LayerNorm(d_readout),
            nn.GELU(),
        )
        self.aux_motif_head = nn.Linear(d_readout, num_classes)

    def forward(self, h_motif):
        m_mean = h_motif.mean(dim=1)
        m_max = h_motif.max(dim=1).values
        attn_scores = self.motif_attn_pool(h_motif)
        attn_weights = F.softmax(attn_scores, dim=1)
        m_attention = (attn_weights * h_motif).sum(dim=1)
        r_m = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1))
        logits = self.aux_motif_head(r_m)
        return logits, r_m


class P0BaselineProbe(nn.Module):
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L):
        logits, r_m = self.readout(h_m_L)
        return logits, {"motif_descriptor": r_m}


class P1LocalPixelReinjectionProbe(nn.Module):
    def __init__(self, d_pixel=96, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.w_v = nn.Linear(d_pixel, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L, h_bar_P):
        # h_bar_P: [B, 49, 96] = sum_i p_(m,i) h_i^P
        d_m = self.w_v(h_bar_P) # [B, 49, 192]
        cat_feat = torch.cat([h_m_L, d_m], dim=-1)
        g_m = torch.sigmoid(self.w_g(cat_feat))
        residual = self.w_d(d_m)
        h_prime = self.ln(h_m_L + g_m * residual)
        logits, r_m = self.readout(h_prime)
        norm_ratio = (torch.norm(residual, dim=-1) / (torch.norm(h_m_L, dim=-1) + 1e-8)).mean()
        return logits, {"gate_values": g_m.squeeze(-1), "norm_ratio": norm_ratio}


class P2ContextualPixelReinspectionProbe(nn.Module):
    def __init__(self, d_pixel=96, d_motif=192, d_q=96, d_readout=384, num_classes=7):
        super().__init__()
        self.d_q = d_q
        self.w_q = nn.Linear(d_motif, d_q, bias=False)
        self.w_k = nn.Linear(d_pixel, d_q, bias=False)
        self.w_v = nn.Linear(d_pixel, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L, h_support, p_m_i, eps=1e-8):
        # h_m_L: [B, 49, 192], h_support: [B, 49, 256, 96], p_m_i: [B, 49, 256]
        q_m = self.w_q(h_m_L).unsqueeze(2) # [B, 49, 1, 96]
        k_i = self.w_k(h_support) # [B, 49, 256, 96]
        v_i = self.w_v(h_support) # [B, 49, 256, 192]
        
        content_scores = (q_m * k_i).sum(dim=-1) / math.sqrt(self.d_q) # [B, 49, 256]
        
        eligible_mask = p_m_i > 0
        log_prior = torch.zeros_like(p_m_i)
        log_prior[eligible_mask] = torch.log(p_m_i[eligible_mask] + eps)
        log_prior[~eligible_mask] = -1e9
        
        total_scores = content_scores + log_prior
        a_m_i = F.softmax(total_scores, dim=-1) # [B, 49, 256]
        
        d_m = (a_m_i.unsqueeze(-1) * v_i).sum(dim=2) # [B, 49, 192]
        cat_feat = torch.cat([h_m_L, d_m], dim=-1)
        g_m = torch.sigmoid(self.w_g(cat_feat))
        residual = self.w_d(d_m)
        h_prime = self.ln(h_m_L + g_m * residual)
        
        logits, r_m = self.readout(h_prime)
        norm_ratio = (torch.norm(residual, dim=-1) / (torch.norm(h_m_L, dim=-1) + 1e-8)).mean()
        return logits, {
            "a_m_i": a_m_i,
            "gate_values": g_m.squeeze(-1),
            "norm_ratio": norm_ratio,
        }
""",
        ),
        _cell(
            "code",
            """# 5. Training Loop for P0, P1, and P2
train_p01_ds = TensorDataset(train_h_m_L, train_h_bar_P, train_y)
val_p01_ds = TensorDataset(val_h_m_L, val_h_bar_P, val_y)

train_p01_loader = DataLoader(train_p01_ds, batch_size=256, shuffle=True)
val_p01_loader = DataLoader(val_p01_ds, batch_size=256, shuffle=False)

def train_probe_p01(probe_class, probe_name, call_mode, **kwargs):
    set_seed(42)
    model = probe_class(**kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\\n=== Training {probe_name} ({param_count:,} parameters) ===")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_epoch = -1
    best_metrics = {}
    patience = 8
    patience_counter = 0
    history = []
    
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for b_hL, b_hbar, b_y in train_p01_loader:
            b_hL, b_hbar, b_y = b_hL.to(device).float(), b_hbar.to(device).float(), b_y.to(device)
            optimizer.zero_grad()
            if call_mode == "p0":
                logits, _ = model(b_hL)
            elif call_mode == "p1":
                logits, _ = model(b_hL, b_hbar)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_correct += int((logits.argmax(dim=-1) == b_y).sum().item())
            
        train_loss /= len(train_y)
        train_acc = train_correct / len(train_y)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets_list = [], []
        with torch.no_grad():
            for b_hL, b_hbar, b_y in val_p01_loader:
                b_hL, b_hbar, b_y = b_hL.to(device).float(), b_hbar.to(device).float(), b_y.to(device)
                if call_mode == "p0":
                    logits, _ = model(b_hL)
                elif call_mode == "p1":
                    logits, _ = model(b_hL, b_hbar)
                loss = criterion(logits, b_y)
                val_loss += loss.item() * len(b_y)
                val_preds.append(logits.argmax(dim=-1).cpu())
                val_targets_list.append(b_y.cpu())
                
        val_loss /= len(val_y)
        val_preds = torch.cat(val_preds)
        val_targets_all = torch.cat(val_targets_list)
        val_acc = float(accuracy_score(val_targets_all, val_preds))
        val_f1 = float(f1_score(val_targets_all, val_preds, average="macro"))
        
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
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "patience": patience_counter,
        })
        
        if epoch % 5 == 0 or improved or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%) | Patience: {patience_counter}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}!")
            break
            
    # Save CSV history & config & result
    with open(OUTPUT_DIR / f"{probe_name}_HISTORY.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
        for h in history:
            writer.writerow([h["epoch"], f"{h['train_loss']:.4f}", f"{h['train_acc']:.4f}", f"{h['val_loss']:.4f}", f"{h['val_acc']:.4f}", f"{h['val_f1']:.4f}"])
            
    with open(OUTPUT_DIR / f"{probe_name}_RESULT.json", "w") as f:
        json.dump(best_metrics, f, indent=2)
        
    with open(OUTPUT_DIR / f"{probe_name}_CONFIG.json", "w") as f:
        json.dump({
            "probe_name": probe_name,
            "optimizer": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "max_epochs": 50,
            "early_stopping_patience": 8,
            "batch_size": 256,
            "seed": 42,
            "parameters": param_count,
        }, f, indent=2)
        
    return model, best_metrics, history

model_p0, res_p0, hist_p0 = train_probe_p01(P0BaselineProbe, "P0", call_mode="p0")
model_p1, res_p1, hist_p1 = train_probe_p01(P1LocalPixelReinjectionProbe, "P1", call_mode="p1")

# Training P2 with batch-level memory-mapped gather
class P2Dataset(Dataset):
    def __init__(self, hp_mmap, p_m_i, h_m_L, y):
        self.hp_mmap = hp_mmap
        self.p_m_i = p_m_i
        self.h_m_L = h_m_L
        self.y = y
        self.length = len(y)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Return indices for efficient batch slice
        return idx

train_p2_ds = P2Dataset(train_hp_mmap, train_p_m_i, train_h_m_L, train_y)
val_p2_ds = P2Dataset(val_hp_mmap, val_p_m_i, val_h_m_L, val_y)

train_p2_loader = DataLoader(train_p2_ds, batch_size=128, shuffle=True)
val_p2_loader = DataLoader(val_p2_ds, batch_size=128, shuffle=False)

def train_probe_p2():
    set_seed(42)
    model = P2ContextualPixelReinspectionProbe().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\\n=== Training P2 Contextual Reinspection ({param_count:,} parameters) ===")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_epoch = -1
    best_metrics = {}
    patience = 8
    patience_counter = 0
    history = []
    
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for idx_batch in train_p2_loader:
            indices = idx_batch.numpy()
            b_hp_np = train_hp_mmap[indices] # [B, 2304, 96]
            b_hp = torch.from_numpy(b_hp_np).to(device).float()
            b_hsup = b_hp[:, sup16, :] # [B, 49, 256, 96]
            
            b_hL = train_h_m_L[indices].to(device).float()
            b_p = train_p_m_i[indices].to(device).float()
            b_y = train_y[indices].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(b_hL, b_hsup, b_p)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_correct += int((logits.argmax(dim=-1) == b_y).sum().item())
            
        train_loss /= len(train_y)
        train_acc = train_correct / len(train_y)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets_list = [], []
        with torch.no_grad():
            for idx_batch in val_p2_loader:
                indices = idx_batch.numpy()
                b_hp_np = val_hp_mmap[indices]
                b_hp = torch.from_numpy(b_hp_np).to(device).float()
                b_hsup = b_hp[:, sup16, :]
                
                b_hL = val_h_m_L[indices].to(device).float()
                b_p = val_p_m_i[indices].to(device).float()
                b_y = val_y[indices].to(device)
                
                logits, _ = model(b_hL, b_hsup, b_p)
                loss = criterion(logits, b_y)
                val_loss += loss.item() * len(b_y)
                val_preds.append(logits.argmax(dim=-1).cpu())
                val_targets_list.append(b_y.cpu())
                
        val_loss /= len(val_y)
        val_preds = torch.cat(val_preds)
        val_targets_all = torch.cat(val_targets_list)
        val_acc = float(accuracy_score(val_targets_all, val_preds))
        val_f1 = float(f1_score(val_targets_all, val_preds, average="macro"))
        
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
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "patience": patience_counter,
        })
        
        if epoch % 5 == 0 or improved or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%) | Patience: {patience_counter}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}!")
            break
            
    with open(OUTPUT_DIR / "P2_HISTORY.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
        for h in history:
            writer.writerow([h["epoch"], f"{h['train_loss']:.4f}", f"{h['train_acc']:.4f}", f"{h['val_loss']:.4f}", f"{h['val_acc']:.4f}", f"{h['val_f1']:.4f}"])
            
    with open(OUTPUT_DIR / "P2_RESULT.json", "w") as f:
        json.dump(best_metrics, f, indent=2)
        
    with open(OUTPUT_DIR / "P2_CONFIG.json", "w") as f:
        json.dump({
            "probe_name": "P2",
            "optimizer": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "max_epochs": 50,
            "early_stopping_patience": 8,
            "batch_size": 128,
            "seed": 42,
            "parameters": param_count,
        }, f, indent=2)
        
    return model, best_metrics, history

model_p2, res_p2, hist_p2 = train_probe_p2()

print("\\n=== PIXEL REINSPECTION METRICS ===")
print(f"P0 Baseline:    Acc = {res_p0['val_accuracy']*100:.2f}%, F1 = {res_p0['val_macro_f1']*100:.2f}% (Epoch {res_p0['selected_epoch']})")
print(f"P1 Reinjection: Acc = {res_p1['val_accuracy']*100:.2f}%, F1 = {res_p1['val_macro_f1']*100:.2f}% (Epoch {res_p1['selected_epoch']})")
print(f"P2 Contextual:  Acc = {res_p2['val_accuracy']*100:.2f}%, F1 = {res_p2['val_macro_f1']*100:.2f}% (Epoch {res_p2['selected_epoch']})")
""",
        ),
        _cell(
            "code",
            """# 6. Detailed Diagnostics for P1 and P2
# P1 Gate Diagnostics
model_p1.eval()
all_p1_gates, all_p1_ratios = [], []
with torch.no_grad():
    for b_hL, b_hbar, _ in val_p01_loader:
        b_hL, b_hbar = b_hL.to(device).float(), b_hbar.to(device).float()
        _, out = model_p1(b_hL, b_hbar)
        all_p1_gates.append(out["gate_values"].cpu())
        all_p1_ratios.append(out["norm_ratio"].item())

p1_gates = torch.cat(all_p1_gates, dim=0) # [N, 49]
p1_diag = {
    "mean_gate_activation": float(p1_gates.mean().item()),
    "std_gate_activation": float(p1_gates.std().item()),
    "min_gate_activation": float(p1_gates.min().item()),
    "max_gate_activation": float(p1_gates.max().item()),
    "mean_norm_ratio": float(sum(all_p1_ratios) / len(all_p1_ratios)),
    "gate_percentiles": {
        "p10": float(torch.quantile(p1_gates, 0.10).item()),
        "p25": float(torch.quantile(p1_gates, 0.25).item()),
        "p50": float(torch.quantile(p1_gates, 0.50).item()),
        "p75": float(torch.quantile(p1_gates, 0.75).item()),
        "p90": float(torch.quantile(p1_gates, 0.90).item()),
    }
}
with open(OUTPUT_DIR / "P1_GATE_DIAGNOSTICS.json", "w") as f:
    json.dump(p1_diag, f, indent=2)
print("Wrote P1_GATE_DIAGNOSTICS.json")

# P2 Reinspection Diagnostics
model_p2.eval()
all_p2_gates, all_p2_ratios, all_a_m_i, all_p_m_i = [], [], [], []
with torch.no_grad():
    for idx_batch in val_p2_loader:
        indices = idx_batch.numpy()
        b_hp = torch.from_numpy(val_hp_mmap[indices]).to(device).float()
        b_hsup = b_hp[:, sup16, :]
        b_hL = val_h_m_L[indices].to(device).float()
        b_p = val_p_m_i[indices].to(device).float()
        
        _, out = model_p2(b_hL, b_hsup, b_p)
        all_p2_gates.append(out["gate_values"].cpu())
        all_p2_ratios.append(out["norm_ratio"].item())
        all_a_m_i.append(out["a_m_i"].cpu())
        all_p_m_i.append(b_p.cpu())

p2_gates = torch.cat(all_p2_gates, dim=0) # [N, 49]
a_m_i = torch.cat(all_a_m_i, dim=0) # [N, 49, 256]
p_m_i = torch.cat(all_p_m_i, dim=0) # [N, 49, 256]

# Attention entropy: -(p * log(p))
p_a = a_m_i.clamp(min=1e-12)
entropy = (-(p_a * p_a.log()).sum(dim=-1)).mean().item()
effective_pixels = float(math.exp(entropy))
normalized_entropy = entropy / math.log(256)

p_p = p_m_i.clamp(min=1e-12)
kl_div = (a_m_i * (p_a.log() - p_p.log())).sum(dim=-1).mean().item()
mean_abs_diff = (a_m_i - p_m_i).abs().mean().item()

argmax_a = a_m_i.argmax(dim=-1)
argmax_p = p_m_i.argmax(dim=-1)
argmax_change_rate = float((argmax_a != argmax_p).float().mean().item())

p2_diag = {
    "mean_gate_activation": float(p2_gates.mean().item()),
    "std_gate_activation": float(p2_gates.std().item()),
    "mean_norm_ratio": float(sum(all_p2_ratios) / len(all_p2_ratios)),
    "attention_entropy": entropy,
    "normalized_entropy": normalized_entropy,
    "effective_attended_pixels": effective_pixels,
    "mean_kl_a_p": kl_div,
    "mean_absolute_difference": mean_abs_diff,
    "argmax_pixel_change_rate": argmax_change_rate,
}
with open(OUTPUT_DIR / "P2_REINSPECTION_DIAGNOSTICS.json", "w") as f:
    json.dump(p2_diag, f, indent=2)
print("Wrote P2_REINSPECTION_DIAGNOSTICS.json")
""",
        ),
        _cell(
            "code",
            """# 7. Decision Logic & Summary Generation
acc_p0 = res_p0["val_accuracy"] * 100.0
acc_p1 = res_p1["val_accuracy"] * 100.0
acc_p2 = res_p2["val_accuracy"] * 100.0

f1_p0 = res_p0["val_macro_f1"] * 100.0
f1_p1 = res_p1["val_macro_f1"] * 100.0
f1_p2 = res_p2["val_macro_f1"] * 100.0

delta_p1_p0 = acc_p1 - acc_p0
delta_p2_p0 = acc_p2 - acc_p0
delta_p2_p1 = acc_p2 - acc_p1

print(f"\\n=== DECISION LOGIC AUDIT ===")
print(f"P1 - P0: {delta_p1_p0:+.2f} pp")
print(f"P2 - P0: {delta_p2_p0:+.2f} pp")
print(f"P2 - P1: {delta_p2_p1:+.2f} pp")

decision = None
go_decision_string = None

if delta_p2_p0 >= 0.30 and delta_p2_p1 >= 0.20 and (f1_p2 >= f1_p0 - 0.10):
    decision = "CASE PIX-A (GO for Contextual Local Pixel Reinspection)"
    go_decision_string = "PIXEL_REINSPECTION_GO_CONTEXTUAL"
elif delta_p1_p0 >= 0.30 and abs(acc_p2 - acc_p1) < 0.20:
    decision = "CASE PIX-B (GO for simple local pixel reinjection, but NOT contextual reinspection)"
    go_decision_string = "PIXEL_REINSPECTION_GO_REINJECTION"
elif delta_p1_p0 <= 0.20 and delta_p2_p0 <= 0.20:
    decision = "CASE PIX-C (NO-GO. Late evidence-recovery family is exhausted)"
    go_decision_string = "PIXEL_REINSPECTION_NO_GO"
else:
    decision = "CASE PIX-D (Ambiguous / divergent Macro-F1 / unstable)"
    go_decision_string = "PIXEL_REINSPECTION_AMBIGUOUS"

print("Decision Classification:", decision)
print("GO / NO-GO String:      ", go_decision_string)

summary_dict = {
    "schema_version": 1,
    "title": "MPG-FER Contextual Local Pixel Reinspection Diagnostic Summary",
    "decision_classification": decision,
    "decision_status_string": go_decision_string,
    "p0_baseline": res_p0,
    "p1_local_reinjection": res_p1,
    "p2_contextual_reinspection": res_p2,
    "pairwise_deltas_pp": {
        "p1_minus_p0": delta_p1_p0,
        "p2_minus_p0": delta_p2_p0,
        "p2_minus_p1": delta_p2_p1,
    },
    "p1_diagnostics": p1_diag,
    "p2_diagnostics": p2_diag,
}

with open(OUTPUT_DIR / "PIXEL_REINSPECTION_SUMMARY.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

md_lines = [
    "# Contextual Local Pixel Reinspection Diagnostic Summary",
    "",
    f"## Decision: **{go_decision_string}** ({decision})",
    "",
    "### Validation Performance (val.csv, 3589 samples)",
    "| Probe | Mechanism | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs P0 (pp) |",
    "|---|---|---:|---:|---:|---:|---:|",
    f"| P0 Baseline | Frozen h_m^(L) + Canonical Readout | {res_p0['parameter_count']:,} | {acc_p0:.2f} | {f1_p0:.2f} | {res_p0['selected_epoch']} | 0.00 |",
    f"| P1 Reinjection | Prior-weighted pixel sum d_m | {res_p1['parameter_count']:,} | {acc_p1:.2f} | {f1_p1:.2f} | {res_p1['selected_epoch']} | {delta_p1_p0:+.2f} |",
    f"| P2 Contextual | Query h_m^(L) + spatial prior bias | {res_p2['parameter_count']:,} | {acc_p2:.2f} | {f1_p2:.2f} | {res_p2['selected_epoch']} | {delta_p2_p0:+.2f} |",
    "",
    "### Pixel Reinspection Dynamics",
    f"- **P1 Mean Gate Activation:** `{p1_diag['mean_gate_activation']:.3f}` | Norm Ratio: `{p1_diag['mean_norm_ratio']:.3f}`",
    f"- **P2 Mean Gate Activation:** `{p2_diag['mean_gate_activation']:.3f}` | Norm Ratio: `{p2_diag['mean_norm_ratio']:.3f}`",
    f"- **P2 Attention Entropy:** `{p2_diag['attention_entropy']:.3f}` (Effective Pixels: `{p2_diag['effective_attended_pixels']:.1f}` / 256)",
    f"- **Mean KL(a || p):** `{p2_diag['mean_kl_a_p']:.4f}` | Mean |a - p|: `{p2_diag['mean_absolute_difference']:.4f}`",
    f"- **Argmax Pixel Change Rate:** `{p2_diag['argmax_pixel_change_rate']*100:.2f}%`",
]

with open(OUTPUT_DIR / "PIXEL_REINSPECTION_SUMMARY.md", "w") as f:
    f.write("\\n".join(md_lines) + "\\n")

# Remove large mmap files before archiving to preserve disk quota
import shutil
if os.path.exists(train_hp_path):
    os.remove(train_hp_path)
if os.path.exists(val_hp_path):
    os.remove(val_hp_path)

archive = shutil.make_archive("/kaggle/working/pixel-reinspection-artifacts", "zip", root_dir=OUTPUT_DIR)
print(f"All Pixel Reinspection diagnostic artifacts archived successfully to {archive}")
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
    nb = build_pix_diagnostic_notebook()
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "mpg_fer_pixel_reinspection.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebook:", nb_path)


if __name__ == "__main__":
    main()
