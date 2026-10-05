"""Generate the standalone Kaggle notebook for MPG-FER Readout Diagnostic."""

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


def build_diagnostic_notebook() -> dict:
    encoded_sources, source_sha = encode_sources()

    cells = [
        _cell(
            "markdown",
            "# MPG-FER Motif Readout Diagnostic (R0, R1, R2)\n\n"
            "This diagnostic extracts frozen motif node representations H_M from canonical FULL seed-42 checkpoint "
            "and evaluates whether readout expressivity is a bottleneck (R0 Canonical vs R0 Refit vs R1 Multi-Slot vs R2 Class-Conditioned).\n\n"
            "**Hard Rule Enforcement:** Train -> Validation only. Strictly NO Test/Private access.\n",
        ),
        _cell(
            "code",
            """import os
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/mpg_fer_readout_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Diagnostic output directory:", OUTPUT_DIR)
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

from mpg_fer_v2_3.model import MPGFER
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
            """# 3. Load Backbone and Verify Validation Parity
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

val_dataset = FER2013Dataset(val_csv, split="val", augment=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

r0_canon_raw_preds, r0_canon_tta_preds, val_targets = [], [], []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        _, out_orig = backbone(images)
        _, out_flip = backbone(TF.hflip(images))
        
        m_logits_orig = out_orig["motif_logits"]
        m_logits_flip = out_flip["motif_logits"]
        m_logits_tta = (m_logits_orig + m_logits_flip) / 2.0
        
        r0_canon_raw_preds.append(m_logits_orig.argmax(dim=-1).cpu())
        r0_canon_tta_preds.append(m_logits_tta.argmax(dim=-1).cpu())
        val_targets.append(targets.cpu())

val_targets = torch.cat(val_targets)
r0_canon_raw = torch.cat(r0_canon_raw_preds)
r0_canon_tta = torch.cat(r0_canon_tta_preds)

r0_canon_metrics = {
    "raw_accuracy": float(accuracy_score(val_targets, r0_canon_raw)),
    "raw_macro_f1": float(f1_score(val_targets, r0_canon_raw, average="macro")),
    "tta_accuracy": float(accuracy_score(val_targets, r0_canon_tta)),
    "tta_macro_f1": float(f1_score(val_targets, r0_canon_tta, average="macro")),
    "checkpoint_sha256": ckpt_sha,
}
print("R0 CANONICAL Validation Metrics:")
print("  Raw Acc:", f"{r0_canon_metrics['raw_accuracy']*100:.2f}%")
print("  TTA Acc:", f"{r0_canon_metrics['tta_accuracy']*100:.2f}%")

with open(OUTPUT_DIR / "R0_CANONICAL_RESULT.json", "w") as f:
    json.dump(r0_canon_metrics, f, indent=2)
""",
        ),
        _cell(
            "code",
            """# 4. Extract H_M Features (Train and Val)
print("Extracting frozen H_M features [B, 49, 192]...")

train_dataset = FER2013Dataset(train_csv, split="train", augment=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)

def extract_features(loader, desc):
    all_hm = []
    all_y = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            # Forward through pixel projection, topology, pixel gnn, composer, motif gnn
            batch = images.shape[0]
            h = backbone.pixel_proj(backbone.pixel_extractor(images))
            intensities = images.reshape(batch, cfg.num_pixels, 1)
            edges = backbone.pixel_topology.compute_edge_features(intensities)
            for layer in backbone.pixel_gnn:
                h = layer(h, backbone.pixel_topology.neighbor_idx, backbone.pixel_topology.neighbor_mask, edges)
            
            h_motif, assignments, diagnostics = backbone.motif_composer(h)
            geometry = compute_motif_geometry(diagnostics["learned_centers_x"], diagnostics["learned_centers_y"])
            for l_idx, layer in enumerate(backbone.motif_gnn):
                h_motif, _ = layer(h_motif, geometry, return_diagnostics=True)
            
            # h_motif is now the final motif node tensor [B, 49, 192]
            all_hm.append(h_motif.cpu())
            all_y.append(targets.cpu())
            if (i + 1) % 100 == 0:
                print(f"  {desc}: {(i + 1) * batch} samples extracted...")
    print(f"{desc} completed in {time.monotonic() - t0:.1f}s")
    return torch.cat(all_hm), torch.cat(all_y)

# Extract
from mpg_fer_v2_3.model import compute_motif_geometry

train_hm, train_y = extract_features(train_loader, "Train")
val_hm, val_y = extract_features(val_loader, "Val")

print(f"Extracted Train H_M: {train_hm.shape}, Train y: {train_y.shape}")
print(f"Extracted Val H_M:   {val_hm.shape}, Val y:   {val_y.shape}")

feature_manifest = {
    "schema_version": 1,
    "source_checkpoint_sha256": ckpt_sha,
    "train_samples": len(train_y),
    "val_samples": len(val_y),
    "h_m_shape": [49, 192],
    "train_h_m_sha256": hashlib.sha256(train_hm.numpy().tobytes()).hexdigest(),
    "val_h_m_sha256": hashlib.sha256(val_hm.numpy().tobytes()).hexdigest(),
}
with open(OUTPUT_DIR / "FEATURE_MANIFEST.json", "w") as f:
    json.dump(feature_manifest, f, indent=2)
print("Wrote FEATURE_MANIFEST.json")
""",
        ),
        _cell(
            "code",
            """# 5. Probe Implementations
class R0RefitProbe(nn.Module):
    def __init__(self, d_motif=192, d_readout=384, num_classes=7):
        super().__init__()
        self.d_motif = d_motif
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
        return logits, {"attention_weights": attn_weights.squeeze(-1)}


class R1MultiSlotProbe(nn.Module):
    def __init__(self, d_motif=192, num_slots=7, d_slot_proj=64, d_hidden=256, num_classes=7):
        super().__init__()
        self.num_slots = num_slots
        self.d_motif = d_motif
        self.slots = nn.Parameter(torch.randn(num_slots, d_motif) / math.sqrt(d_motif))
        self.k_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.v_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.slot_proj = nn.Sequential(
            nn.Linear(d_motif, d_slot_proj),
            nn.LayerNorm(d_slot_proj),
            nn.GELU(),
        )
        flat_dim = num_slots * d_slot_proj
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, h_motif):
        batch = h_motif.shape[0]
        k = self.k_proj(h_motif)
        v = self.v_proj(h_motif)
        q = self.slots.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_motif)
        attn = F.softmax(scores, dim=-1) # [B, 7, 49]
        slots_e = torch.matmul(attn, v)
        slots_projected = self.slot_proj(slots_e)
        flat = slots_projected.reshape(batch, -1)
        logits = self.classifier(flat)
        return logits, {"attention_weights": attn, "slot_queries": self.slots}


class R2ClassConditionedProbe(nn.Module):
    def __init__(self, d_motif=192, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.d_motif = d_motif
        self.class_queries = nn.Parameter(torch.randn(num_classes, d_motif) / math.sqrt(d_motif))
        self.k_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.v_proj = nn.Linear(d_motif, d_motif, bias=False)
        self.w_c = nn.Parameter(torch.randn(num_classes, d_motif) / math.sqrt(d_motif))
        self.b_c = nn.Parameter(torch.zeros(num_classes))

    def forward(self, h_motif):
        batch = h_motif.shape[0]
        k = self.k_proj(h_motif)
        v = self.v_proj(h_motif)
        q = self.class_queries.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_motif)
        alpha = F.softmax(scores, dim=-1) # [B, 7, 49]
        r = torch.matmul(alpha, v)
        logits = (r * self.w_c.unsqueeze(0)).sum(dim=-1) + self.b_c.unsqueeze(0)
        return logits, {"attention_weights": alpha, "class_queries": self.class_queries}
""",
        ),
        _cell(
            "code",
            """# 6. Training Pipeline for Probes
train_feat_ds = TensorDataset(train_hm, train_y)
val_feat_ds = TensorDataset(val_hm, val_y)

train_feat_loader = DataLoader(train_feat_ds, batch_size=256, shuffle=True)
val_feat_loader = DataLoader(val_feat_ds, batch_size=256, shuffle=False)

def train_probe(probe_class, probe_name, **probe_kwargs):
    set_seed(42)
    model = probe_class(**probe_kwargs).to(device)
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
        for x_b, y_b in train_feat_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            train_correct += int((logits.argmax(dim=-1) == y_b).sum().item())
        
        train_loss /= len(train_hm)
        train_acc = train_correct / len(train_hm)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets_list = [], []
        last_out_dict = {}
        with torch.no_grad():
            for x_b, y_b in val_feat_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits, out_dict = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                val_preds.append(logits.argmax(dim=-1).cpu())
                val_targets_list.append(y_b.cpu())
                last_out_dict = out_dict
        
        val_loss /= len(val_hm)
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
""",
        ),
        _cell(
            "code",
            """# 7. Train R0_REFIT, R1, and R2
model_r0, res_r0, hist_r0 = train_probe(R0RefitProbe, "R0_REFIT")
model_r1, res_r1, hist_r1 = train_probe(R1MultiSlotProbe, "R1")
model_r2, res_r2, hist_r2 = train_probe(R2ClassConditionedProbe, "R2")

print("\\n=== PROBE RESULTS SUMMARY ===")
print(f"R0 Canonical: Acc = {r0_canon_metrics['tta_accuracy']*100:.2f}% (TTA), {r0_canon_metrics['raw_accuracy']*100:.2f}% (Raw)")
print(f"R0 Refit:     Acc = {res_r0['val_accuracy']*100:.2f}%, F1 = {res_r0['val_macro_f1']*100:.2f}% (Epoch {res_r0['selected_epoch']})")
print(f"R1 MultiSlot: Acc = {res_r1['val_accuracy']*100:.2f}%, F1 = {res_r1['val_macro_f1']*100:.2f}% (Epoch {res_r1['selected_epoch']})")
print(f"R2 ClassCond: Acc = {res_r2['val_accuracy']*100:.2f}%, F1 = {res_r2['val_macro_f1']*100:.2f}% (Epoch {res_r2['selected_epoch']})")
""",
        ),
        _cell(
            "code",
            """# 8. Attention Diagnostics and Collapse Checks
def compute_attention_diagnostics_r1(model, val_loader):
    model.eval()
    all_attn = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            _, out = model(x_b)
            all_attn.append(out["attention_weights"].cpu())
    attn = torch.cat(all_attn, dim=0) # [N, 7, 49]
    
    # Entropy per slot: H_s = -sum(p log p)
    p = attn.clamp(min=1e-12)
    entropy_per_slot = (-(p * p.log()).sum(dim=-1)).mean(dim=0).tolist()
    effective_nodes_per_slot = [float(math.exp(h)) for h in entropy_per_slot]
    mean_entropy = float(sum(entropy_per_slot) / len(entropy_per_slot))
    
    # Query cosine similarity
    queries = F.normalize(model.slots, dim=-1) # [7, 192]
    sim_matrix = (queries @ queries.t()).tolist()
    
    # Off-diagonal similarity
    off_diag = []
    for i in range(7):
        for j in range(7):
            if i != j:
                off_diag.append(sim_matrix[i][j])
    mean_off_diag_sim = float(sum(off_diag) / len(off_diag))
    
    return {
        "entropy_per_slot": entropy_per_slot,
        "mean_entropy": mean_entropy,
        "effective_nodes_per_slot": effective_nodes_per_slot,
        "mean_effective_nodes": float(sum(effective_nodes_per_slot) / len(effective_nodes_per_slot)),
        "query_cosine_similarity_matrix": sim_matrix,
        "mean_off_diagonal_query_similarity": mean_off_diag_sim,
        "collapsed_slots": mean_off_diag_sim > 0.95,
    }

def compute_attention_diagnostics_r2(model, val_loader):
    model.eval()
    all_attn = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            _, out = model(x_b)
            all_attn.append(out["attention_weights"].cpu())
    attn = torch.cat(all_attn, dim=0) # [N, 7, 49]
    
    p = attn.clamp(min=1e-12)
    entropy_per_class = (-(p * p.log()).sum(dim=-1)).mean(dim=0).tolist()
    effective_nodes_per_class = [float(math.exp(h)) for h in entropy_per_class]
    mean_entropy = float(sum(entropy_per_class) / len(entropy_per_class))
    
    queries = F.normalize(model.class_queries, dim=-1) # [7, 192]
    sim_matrix = (queries @ queries.t()).tolist()
    
    off_diag = []
    for i in range(7):
        for j in range(7):
            if i != j:
                off_diag.append(sim_matrix[i][j])
    mean_off_diag_sim = float(sum(off_diag) / len(off_diag))
    
    # Top-5 attended occurrence indices per class
    mean_attn_per_class = attn.mean(dim=0) # [7, 49]
    top5_per_class = [mean_attn_per_class[c].topk(5).indices.tolist() for c in range(7)]
    
    return {
        "entropy_per_class": entropy_per_class,
        "mean_entropy": mean_entropy,
        "effective_nodes_per_class": effective_nodes_per_class,
        "mean_effective_nodes": float(sum(effective_nodes_per_class) / len(effective_nodes_per_class)),
        "query_cosine_similarity_matrix": sim_matrix,
        "mean_off_diagonal_query_similarity": mean_off_diag_sim,
        "top5_attended_nodes_per_class": top5_per_class,
        "collapsed_classes": mean_off_diag_sim > 0.95,
    }

diag_r1 = compute_attention_diagnostics_r1(model_r1, val_feat_loader)
diag_r2 = compute_attention_diagnostics_r2(model_r2, val_feat_loader)

with open(OUTPUT_DIR / "R1_ATTENTION_DIAGNOSTICS.json", "w") as f:
    json.dump(diag_r1, f, indent=2)

with open(OUTPUT_DIR / "R2_ATTENTION_DIAGNOSTICS.json", "w") as f:
    json.dump(diag_r2, f, indent=2)

print("Attention diagnostics computed and saved.")
""",
        ),
        _cell(
            "code",
            """# 9. Decision Logic & Summary Generation
acc_r0 = res_r0["val_accuracy"] * 100.0
acc_r1 = res_r1["val_accuracy"] * 100.0
acc_r2 = res_r2["val_accuracy"] * 100.0

delta_r1_r0 = acc_r1 - acc_r0
delta_r2_r0 = acc_r2 - acc_r0
delta_r2_r1 = acc_r2 - acc_r1

print(f"\\n=== DECISION LOGIC AUDIT ===")
print(f"R1 - R0: {delta_r1_r0:+.2f} pp")
print(f"R2 - R0: {delta_r2_r0:+.2f} pp")
print(f"R2 - R1: {delta_r2_r1:+.2f} pp")

decision = None
go_decision_string = None
if delta_r2_r1 >= 0.30 and delta_r2_r0 > 0.0:
    decision = "CASE B (Class-conditioned evidence selection specifically supported)"
    go_decision_string = "READOUT_DIAGNOSTIC_GO_CLASS_CONDITIONED"
elif delta_r1_r0 >= 0.30 and abs(acc_r2 - acc_r1) < 0.30:
    decision = "CASE A (Single-vector compression is bottleneck, multi-slot supported)"
    go_decision_string = "READOUT_DIAGNOSTIC_GO_MULTISLOT"
elif delta_r1_r0 <= 0.20 and delta_r2_r0 <= 0.20:
    decision = "CASE C (STOP readout-expansion direction; gain depends on end-to-end co-adaptation)"
    go_decision_string = "READOUT_DIAGNOSTIC_NO_GO"
else:
    decision = "CASE D (Ambiguous gain / Macro-F1 divergence)"
    go_decision_string = "READOUT_DIAGNOSTIC_AMBIGUOUS"

print("Decision Classification:", decision)
print("GO / NO-GO String:      ", go_decision_string)

summary_dict = {
    "schema_version": 1,
    "title": "MPG-FER Motif Readout Diagnostic Summary",
    "decision_classification": decision,
    "decision_status_string": go_decision_string,
    "r0_canonical": r0_canon_metrics,
    "r0_refit": res_r0,
    "r1_multi_slot": res_r1,
    "r2_class_conditioned": res_r2,
    "pairwise_deltas_pp": {
        "r1_minus_r0": delta_r1_r0,
        "r2_minus_r0": delta_r2_r0,
        "r2_minus_r1": delta_r2_r1,
    },
    "diagnostics_summary": {
        "r1_mean_entropy": diag_r1["mean_entropy"],
        "r1_mean_effective_nodes": diag_r1["mean_effective_nodes"],
        "r1_mean_off_diag_similarity": diag_r1["mean_off_diagonal_query_similarity"],
        "r2_mean_entropy": diag_r2["mean_entropy"],
        "r2_mean_effective_nodes": diag_r2["mean_effective_nodes"],
        "r2_mean_off_diag_similarity": diag_r2["mean_off_diagonal_query_similarity"],
    },
}

with open(OUTPUT_DIR / "READOUT_DIAGNOSTIC_SUMMARY.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

md_lines = [
    "# MPG-FER Motif Readout Diagnostic Summary",
    "",
    f"## Decision: **{go_decision_string}** ({decision})",
    "",
    "### Validation Performance (val.csv, 3589 samples)",
    "| Probe | Architecture | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs R0 Refit (pp) |",
    "|---|---|---:|---:|---:|---:|---:|",
    f"| R0 Canonical | Mean + Max + AttnPool -> Proj | 225,224 | {r0_canon_metrics['raw_accuracy']*100:.2f} (TTA: {r0_canon_metrics['tta_accuracy']*100:.2f}) | {r0_canon_metrics['raw_macro_f1']*100:.2f} | 57 | — |",
    f"| R0 Refit | Mean + Max + AttnPool -> Proj | {res_r0['parameter_count']:,} | {acc_r0:.2f} | {res_r0['val_macro_f1']*100:.2f} | {res_r0['selected_epoch']} | 0.00 |",
    f"| R1 MultiSlot | S=7 Class-Agnostic Slots | {res_r1['parameter_count']:,} | {acc_r1:.2f} | {res_r1['val_macro_f1']*100:.2f} | {res_r1['selected_epoch']} | {delta_r1_r0:+.2f} |",
    f"| R2 ClassCond | C=7 Class-Conditioned Queries | {res_r2['parameter_count']:,} | {acc_r2:.2f} | {res_r2['val_macro_f1']*100:.2f} | {res_r2['selected_epoch']} | {delta_r2_r0:+.2f} |",
    "",
    "### Attention Collapse Diagnostics",
    f"- **R1 Multi-Slot:** Mean Entropy = `{diag_r1['mean_entropy']:.3f}` | Effective Nodes = `{diag_r1['mean_effective_nodes']:.2f}` / 49 | Query Cosine Sim = `{diag_r1['mean_off_diagonal_query_similarity']:.3f}`",
    f"- **R2 Class-Conditioned:** Mean Entropy = `{diag_r2['mean_entropy']:.3f}` | Effective Nodes = `{diag_r2['mean_effective_nodes']:.2f}` / 49 | Query Cosine Sim = `{diag_r2['mean_off_diagonal_query_similarity']:.3f}`",
]

with open(OUTPUT_DIR / "READOUT_DIAGNOSTIC_SUMMARY.md", "w") as f:
    f.write("\\n".join(md_lines) + "\\n")

# 10. Archive all artifacts
import shutil
archive = shutil.make_archive("/kaggle/working/readout-diagnostic-artifacts", "zip", root_dir=OUTPUT_DIR)
print(f"All diagnostic artifacts archived successfully to {archive}")
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
    nb = build_diagnostic_notebook()
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "mpg_fer_readout_diagnostic.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebook:", nb_path)


if __name__ == "__main__":
    main()
