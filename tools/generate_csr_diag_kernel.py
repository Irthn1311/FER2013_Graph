"""Generate the standalone Kaggle notebook for Contextual Scale Recomposition (CSR) Diagnostic."""

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


def build_csr_diagnostic_notebook() -> dict:
    encoded_sources, source_sha = encode_sources()

    cells = [
        _cell(
            "markdown",
            "# MPG-FER Contextual Scale Recomposition (CSR) Diagnostic\n\n"
            "This diagnostic extracts frozen scale-specific occurrence candidates z_(m,s), canonical scale weights alpha_(m,s), "
            "initial fused occurrence states h_m^(0), and post-Motif-Graph states h_m^(L) from the canonical FULL seed-42 checkpoint.\n\n"
            "Evaluates:\n"
            "- C0 Baseline: Canonical motif readout on frozen h_m^(L)\n"
            "- C1 Early-State Skip: Gated skip connection from h_m^(0)\n"
            "- C2 Contextual Scale Recomposition: Occurrence-wise query from h_m^(L) attending to original scale candidates z_(m,s)\n\n"
            "**Hard Rule Enforcement:** Train -> Validation only. Strictly NO Test/Private access.\n",
        ),
        _cell(
            "code",
            """import os
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mpg_fer_contextual_recomposition")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("CSR Diagnostic output directory:", OUTPUT_DIR)
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
            """# 3. Load Backbone and Prepare Feature Extraction
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

train_dataset = FER2013Dataset(train_csv, split="train", augment=False)
val_dataset = FER2013Dataset(val_csv, split="val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

def extract_csr_features(loader, desc):
    all_z_ms = []
    all_alpha_ms = []
    all_h_m_0 = []
    all_h_m_L = []
    all_y = []
    t0 = time.monotonic()
    
    composer = backbone.motif_composer
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            batch = images.shape[0]
            
            # Pixel GNN path
            h_pixel = backbone.pixel_proj(backbone.pixel_extractor(images))
            intensities = images.reshape(batch, cfg.num_pixels, 1)
            edges = backbone.pixel_topology.compute_edge_features(intensities)
            for layer in backbone.pixel_gnn:
                h_pixel = layer(h_pixel, backbone.pixel_topology.neighbor_idx, backbone.pixel_topology.neighbor_mask, edges)
                
            # Spatial Motif Composer forward decomposition
            queries = F.normalize(composer.assignment_query(h_pixel), dim=-1)
            keys = F.normalize(composer.prototype_key(composer.prototypes), dim=-1)
            assignments = F.softmax(queries @ keys.t() / composer.temperature, dim=-1)
            confidence = assignments.max(dim=-1, keepdim=True).values
            
            candidates, centers = [], []
            for scale in composer.window_sizes:
                candidate, center, _, _ = composer._pool_scale(h_pixel, assignments, confidence, scale)
                candidates.append(candidate)
                centers.append(center)
            
            # z_(m,s): [B, 49, 3, 192]
            z_m_s = torch.stack(candidates, dim=2)
            center_stack = torch.stack(centers, dim=2)
            
            # alpha_(m,s): [B, 49, 3]
            alpha_m_s = F.softmax(composer.scale_gate(z_m_s).squeeze(-1), dim=-1)
            
            # h_m^(0): [B, 49, 192]
            h_m_0 = (alpha_m_s.unsqueeze(-1) * z_m_s).sum(dim=2)
            fused_centers = (alpha_m_s.unsqueeze(-1) * center_stack).sum(dim=2)
            
            # Motif GNN reasoning -> h_m^(L): [B, 49, 192]
            geometry = compute_motif_geometry(fused_centers[..., 0], fused_centers[..., 1])
            h_m_L = h_m_0
            for layer in backbone.motif_gnn:
                h_m_L, _ = layer(h_m_L, geometry, return_diagnostics=True)
                
            all_z_ms.append(z_m_s.cpu())
            all_alpha_ms.append(alpha_m_s.cpu())
            all_h_m_0.append(h_m_0.cpu())
            all_h_m_L.append(h_m_L.cpu())
            all_y.append(targets.cpu())
            
            if (i + 1) % 100 == 0:
                print(f"  {desc}: {(i + 1) * batch} samples extracted...")
                
    print(f"{desc} completed in {time.monotonic() - t0:.1f}s")
    return (
        torch.cat(all_z_ms),
        torch.cat(all_alpha_ms),
        torch.cat(all_h_m_0),
        torch.cat(all_h_m_L),
        torch.cat(all_y),
    )

print("Extracting Train features...")
train_z_ms, train_alpha_ms, train_h_m_0, train_h_m_L, train_y = extract_csr_features(train_loader, "Train")

print("Extracting Val features...")
val_z_ms, val_alpha_ms, val_h_m_0, val_h_m_L, val_y = extract_csr_features(val_loader, "Val")

print(f"Extracted Train tensors: z_ms={train_z_ms.shape}, alpha={train_alpha_ms.shape}, h0={train_h_m_0.shape}, hL={train_h_m_L.shape}")
print(f"Extracted Val tensors:   z_ms={val_z_ms.shape}, alpha={val_alpha_ms.shape}, h0={val_h_m_0.shape}, hL={val_h_m_L.shape}")

feature_manifest = {
    "schema_version": 1,
    "source_checkpoint_sha256": ckpt_sha,
    "train_samples": len(train_y),
    "val_samples": len(val_y),
    "shapes": {
        "z_m_s": list(train_z_ms.shape[1:]),
        "alpha_m_s": list(train_alpha_ms.shape[1:]),
        "h_m_0": list(train_h_m_0.shape[1:]),
        "h_m_L": list(train_h_m_L.shape[1:]),
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
            """# 4. Probe Model Definitions (C0, C1, C2)
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
        attn_weights = F.softmax(self.motif_attn_pool(h_motif), dim=1)
        m_attention = (attn_weights * h_motif).sum(dim=1)
        r_m = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1))
        logits = self.aux_motif_head(r_m)
        return logits, r_m


class C0BaselineProbe(nn.Module):
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L):
        logits, r_m = self.readout(h_m_L)
        return logits, {"motif_descriptor": r_m}


class C1EarlySkipProbe(nn.Module):
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.w_skip = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L, h_m_0):
        cat_features = torch.cat([h_m_L, h_m_0], dim=-1)
        g_m = torch.sigmoid(self.w_g(cat_features))
        skip_feat = self.w_skip(h_m_0)
        h_final = self.ln(h_m_L + g_m * skip_feat)
        logits, r_m = self.readout(h_final)
        return logits, {"gate_values": g_m.squeeze(-1)}


class C2ContextualScaleRecompositionProbe(nn.Module):
    def __init__(self, d_motif=192, num_scales=3, d_readout=384, num_classes=7):
        super().__init__()
        self.d_motif = d_motif
        self.w_q = nn.Linear(d_motif, d_motif, bias=False)
        self.w_k = nn.Linear(d_motif, d_motif, bias=False)
        self.w_v = nn.Linear(d_motif, d_motif, bias=False)
        self.w_d = nn.Linear(d_motif, d_motif, bias=False)
        self.w_g = nn.Linear(d_motif * 2, 1)
        self.ln = nn.LayerNorm(d_motif)
        self.readout = CanonicalReadout(d_motif, d_readout, num_classes)

    def forward(self, h_m_L, z_m_s):
        batch, occurrences, scales, dim = z_m_s.shape
        q_m = self.w_q(h_m_L).unsqueeze(2) # [B, 49, 1, 192]
        k_ms = self.w_k(z_m_s) # [B, 49, 3, 192]
        v_ms = self.w_v(z_m_s) # [B, 49, 3, 192]
        
        scores = (q_m * k_ms).sum(dim=-1, keepdim=True) / math.sqrt(dim)
        beta = F.softmax(scores.squeeze(-1), dim=-1) # [B, 49, 3]
        
        d_m = (beta.unsqueeze(-1) * v_ms).sum(dim=2) # [B, 49, 192]
        cat_features = torch.cat([h_m_L, d_m], dim=-1)
        g_m = torch.sigmoid(self.w_g(cat_features))
        h_prime = self.ln(h_m_L + g_m * self.w_d(d_m))
        
        logits, r_m = self.readout(h_prime)
        return logits, {"beta": beta, "gate_values": g_m.squeeze(-1)}
""",
        ),
        _cell(
            "code",
            """# 5. Training Loop for C0, C1, and C2
train_ds = TensorDataset(train_h_m_L, train_h_m_0, train_z_ms, train_alpha_ms, train_y)
val_ds = TensorDataset(val_h_m_L, val_h_m_0, val_z_ms, val_alpha_ms, val_y)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

def train_csr_probe(probe_class, probe_name, call_mode, **kwargs):
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
        
        for batch_item in train_loader:
            b_hL, b_h0, b_z, b_alpha, b_y = [item.to(device) for item in batch_item]
            optimizer.zero_grad()
            if call_mode == "c0":
                logits, _ = model(b_hL)
            elif call_mode == "c1":
                logits, _ = model(b_hL, b_h0)
            elif call_mode == "c2":
                logits, _ = model(b_hL, b_z)
            else:
                raise ValueError(call_mode)
                
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
            for batch_item in val_loader:
                b_hL, b_h0, b_z, b_alpha, b_y = [item.to(device) for item in batch_item]
                if call_mode == "c0":
                    logits, _ = model(b_hL)
                elif call_mode == "c1":
                    logits, _ = model(b_hL, b_h0)
                elif call_mode == "c2":
                    logits, _ = model(b_hL, b_z)
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
            
    # Save CSV history
    csv_path = OUTPUT_DIR / f"{probe_name}_HISTORY.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])
        for h in history:
            writer.writerow([h["epoch"], f"{h['train_loss']:.4f}", f"{h['train_acc']:.4f}", f"{h['val_loss']:.4f}", f"{h['val_acc']:.4f}", f"{h['val_f1']:.4f}"])
            
    # Save result JSON
    res_path = OUTPUT_DIR / f"{probe_name}_RESULT.json"
    with open(res_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
        
    # Save config
    cfg_path = OUTPUT_DIR / f"{probe_name}_CONFIG.json"
    with open(cfg_path, "w") as f:
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

model_c0, res_c0, hist_c0 = train_csr_probe(C0BaselineProbe, "C0", call_mode="c0")
model_c1, res_c1, hist_c1 = train_csr_probe(C1EarlySkipProbe, "C1", call_mode="c1")
model_c2, res_c2, hist_c2 = train_csr_probe(C2ContextualScaleRecompositionProbe, "C2", call_mode="c2")

print("\\n=== CSR EXPERIMENT METRICS ===")
print(f"C0 Baseline: Acc = {res_c0['val_accuracy']*100:.2f}%, F1 = {res_c0['val_macro_f1']*100:.2f}% (Epoch {res_c0['selected_epoch']})")
print(f"C1 Skip:     Acc = {res_c1['val_accuracy']*100:.2f}%, F1 = {res_c1['val_macro_f1']*100:.2f}% (Epoch {res_c1['selected_epoch']})")
print(f"C2 CSR:      Acc = {res_c2['val_accuracy']*100:.2f}%, F1 = {res_c2['val_macro_f1']*100:.2f}% (Epoch {res_c2['selected_epoch']})")
""",
        ),
        _cell(
            "code",
            """# 6. Diagnostics: C1 Gate & C2 Recomposition / Alpha vs Beta
# C1 Gate Diagnostics
model_c1.eval()
all_c1_gates = []
with torch.no_grad():
    for batch_item in val_loader:
        b_hL, b_h0, _, _, _ = [item.to(device) for item in batch_item]
        _, out = model_c1(b_hL, b_h0)
        all_c1_gates.append(out["gate_values"].cpu())
c1_gates = torch.cat(all_c1_gates, dim=0) # [N, 49]

c1_gate_diag = {
    "mean_gate_activation": float(c1_gates.mean().item()),
    "std_gate_activation": float(c1_gates.std().item()),
    "min_gate_activation": float(c1_gates.min().item()),
    "max_gate_activation": float(c1_gates.max().item()),
    "gate_percentiles": {
        "p10": float(torch.quantile(c1_gates, 0.10).item()),
        "p25": float(torch.quantile(c1_gates, 0.25).item()),
        "p50": float(torch.quantile(c1_gates, 0.50).item()),
        "p75": float(torch.quantile(c1_gates, 0.75).item()),
        "p90": float(torch.quantile(c1_gates, 0.90).item()),
    }
}
with open(OUTPUT_DIR / "C1_GATE_DIAGNOSTICS.json", "w") as f:
    json.dump(c1_gate_diag, f, indent=2)
print("Wrote C1_GATE_DIAGNOSTICS.json:", c1_gate_diag)

# C2 Recomposition Diagnostics
model_c2.eval()
all_c2_gates, all_beta, all_alpha = [], [], []
with torch.no_grad():
    for batch_item in val_loader:
        b_hL, _, b_z, b_alpha, _ = [item.to(device) for item in batch_item]
        _, out = model_c2(b_hL, b_z)
        all_c2_gates.append(out["gate_values"].cpu())
        all_beta.append(out["beta"].cpu())
        all_alpha.append(b_alpha.cpu())

c2_gates = torch.cat(all_c2_gates, dim=0) # [N, 49]
beta = torch.cat(all_beta, dim=0) # [N, 49, 3]
alpha = torch.cat(all_alpha, dim=0) # [N, 49, 3]

# 1. Gate activation
c2_mean_gate = float(c2_gates.mean().item())

# 2. Beta entropy: -sum(beta * log(beta))
p_beta = beta.clamp(min=1e-12)
beta_entropy = (-(p_beta * p_beta.log()).sum(dim=-1)).mean().item()
effective_scales = float(math.exp(beta_entropy))

# 3. KL divergence: KL(beta || alpha) = sum(beta * log(beta / alpha))
p_alpha = alpha.clamp(min=1e-12)
kl_div = (beta * (p_beta.log() - p_alpha.log())).sum(dim=-1).mean().item()

# 4. Mean absolute difference: |beta - alpha|
mean_abs_diff = (beta - alpha).abs().mean().item()

# 5. Argmax scale change rate
argmax_beta = beta.argmax(dim=-1) # [N, 49]
argmax_alpha = alpha.argmax(dim=-1) # [N, 49]
scale_changes = (argmax_beta != argmax_alpha).float()
argmax_change_rate = float(scale_changes.mean().item())

# 6. Per-scale selection frequencies
alpha_scale_counts = [float((argmax_alpha == s).float().mean().item()) for s in range(3)]
beta_scale_counts = [float((argmax_beta == s).float().mean().item()) for s in range(3)]

# 7. Per-occurrence scale-change frequency
per_occurrence_change_freq = scale_changes.mean(dim=0).tolist()

c2_diag = {
    "mean_gate_activation": c2_mean_gate,
    "beta_entropy": beta_entropy,
    "effective_number_of_scales": effective_scales,
    "mean_kl_beta_alpha": kl_div,
    "mean_absolute_difference": mean_abs_diff,
    "argmax_scale_change_rate": argmax_change_rate,
    "canonical_alpha_scale_distribution": {
        "scale_8": alpha_scale_counts[0],
        "scale_12": alpha_scale_counts[1],
        "scale_16": alpha_scale_counts[2],
    },
    "recomposed_beta_scale_distribution": {
        "scale_8": beta_scale_counts[0],
        "scale_12": beta_scale_counts[1],
        "scale_16": beta_scale_counts[2],
    },
    "per_occurrence_change_frequency": per_occurrence_change_freq,
}
with open(OUTPUT_DIR / "C2_RECOMPOSITION_DIAGNOSTICS.json", "w") as f:
    json.dump(c2_diag, f, indent=2)
print("Wrote C2_RECOMPOSITION_DIAGNOSTICS.json")
""",
        ),
        _cell(
            "code",
            """# 7. Decision Logic & Summary Generation
acc_c0 = res_c0["val_accuracy"] * 100.0
acc_c1 = res_c1["val_accuracy"] * 100.0
acc_c2 = res_c2["val_accuracy"] * 100.0

f1_c0 = res_c0["val_macro_f1"] * 100.0
f1_c1 = res_c1["val_macro_f1"] * 100.0
f1_c2 = res_c2["val_macro_f1"] * 100.0

delta_c1_c0 = acc_c1 - acc_c0
delta_c2_c0 = acc_c2 - acc_c0
delta_c2_c1 = acc_c2 - acc_c1

print(f"\\n=== DECISION LOGIC AUDIT ===")
print(f"C1 - C0: {delta_c1_c0:+.2f} pp")
print(f"C2 - C0: {delta_c2_c0:+.2f} pp")
print(f"C2 - C1: {delta_c2_c1:+.2f} pp")

decision = None
go_decision_string = None

if delta_c2_c0 >= 0.30 and delta_c2_c1 >= 0.20 and (f1_c2 >= f1_c0 - 0.10):
    decision = "CASE CSR-A (GO for Contextual Scale Recomposition)"
    go_decision_string = "CSR_DIAGNOSTIC_GO_RECOMPOSITION"
elif delta_c1_c0 >= 0.30 and abs(acc_c2 - acc_c1) < 0.20:
    decision = "CASE CSR-B (GO for multi-level skip/reuse, but NOT specifically CSR)"
    go_decision_string = "CSR_DIAGNOSTIC_GO_SKIP"
elif delta_c1_c0 <= 0.20 and delta_c2_c0 <= 0.20:
    decision = "CASE CSR-C (NO-GO for scale-level recomposition; move below SMC to pixel reinspection)"
    go_decision_string = "CSR_DIAGNOSTIC_NO_GO"
else:
    decision = "CASE CSR-D (Ambiguous / divergent Macro-F1 / unstable)"
    go_decision_string = "CSR_DIAGNOSTIC_AMBIGUOUS"

print("Decision Classification:", decision)
print("GO / NO-GO String:      ", go_decision_string)

summary_dict = {
    "schema_version": 1,
    "title": "MPG-FER Contextual Scale Recomposition (CSR) Diagnostic Summary",
    "decision_classification": decision,
    "decision_status_string": go_decision_string,
    "c0_baseline": res_c0,
    "c1_early_skip": res_c1,
    "c2_csr": res_c2,
    "pairwise_deltas_pp": {
        "c1_minus_c0": delta_c1_c0,
        "c2_minus_c0": delta_c2_c0,
        "c2_minus_c1": delta_c2_c1,
    },
    "c1_diagnostics": c1_gate_diag,
    "c2_diagnostics": c2_diag,
}

with open(OUTPUT_DIR / "CSR_DIAGNOSTIC_SUMMARY.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

md_lines = [
    "# Contextual Scale Recomposition (CSR) Diagnostic Summary",
    "",
    f"## Decision: **{go_decision_string}** ({decision})",
    "",
    "### Validation Performance (val.csv, 3589 samples)",
    "| Probe | Mechanism | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs C0 (pp) |",
    "|---|---|---:|---:|---:|---:|---:|",
    f"| C0 Baseline | Frozen h_m^(L) + Canonical Readout | {res_c0['parameter_count']:,} | {acc_c0:.2f} | {f1_c0:.2f} | {res_c0['selected_epoch']} | 0.00 |",
    f"| C1 Early Skip | Gated skip from h_m^(0) | {res_c1['parameter_count']:,} | {acc_c1:.2f} | {f1_c1:.2f} | {res_c1['selected_epoch']} | {delta_c1_c0:+.2f} |",
    f"| C2 CSR | Query h_m^(L) attending to z_(m,s) | {res_c2['parameter_count']:,} | {acc_c2:.2f} | {f1_c2:.2f} | {res_c2['selected_epoch']} | {delta_c2_c0:+.2f} |",
    "",
    "### Recomposition Dynamics (Alpha vs Beta)",
    f"- **Argmax Scale Change Rate:** `{c2_diag['argmax_scale_change_rate']*100:.2f}%` of occurrences altered their primary scale choice",
    f"- **Mean KL(Beta || Alpha):** `{c2_diag['mean_kl_beta_alpha']:.4f}`",
    f"- **Mean |Beta - Alpha|:** `{c2_diag['mean_absolute_difference']:.4f}`",
    f"- **Beta Entropy:** `{c2_diag['beta_entropy']:.3f}` (Effective scales: `{c2_diag['effective_number_of_scales']:.2f}` / 3)",
    f"- **Mean CSR Gate Activation:** `{c2_diag['mean_gate_activation']:.3f}`",
    f"- **Mean C1 Gate Activation:** `{c1_gate_diag['mean_gate_activation']:.3f}`",
]

with open(OUTPUT_DIR / "CSR_DIAGNOSTIC_SUMMARY.md", "w") as f:
    f.write("\\n".join(md_lines) + "\\n")

# Archive all artifacts
import shutil
archive = shutil.make_archive("/kaggle/working/csr-diagnostic-artifacts", "zip", root_dir=OUTPUT_DIR)
print(f"All CSR diagnostic artifacts archived successfully to {archive}")
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
    nb = build_csr_diagnostic_notebook()
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "mpg_fer_csr_diagnostic.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebook:", nb_path)


if __name__ == "__main__":
    main()
