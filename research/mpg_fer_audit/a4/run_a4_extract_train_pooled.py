"""Extract Train pooled features for v2.1 and v2.2 (raw and flip-TTA).

Saves:
- research/mpg_fer_audit/a4/train_pooled_features.npz
Containing:
- v21_orig_r0, v21_orig_r7, v21_orig_r8, v21_orig_r9, v21_orig_r10
- v21_flip_r0, v21_flip_r7, v21_flip_r8, v21_flip_r9, v21_flip_r10
- v22_orig_r0, v22_orig_r7, v22_orig_r8, v22_orig_r9, v22_orig_r10
- v22_flip_r0, v22_flip_r7, v22_flip_r8, v22_flip_r9, v22_flip_r10
- targets
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"

V21_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"
sys.path.insert(0, str(V21_SRC))
sys.path.insert(0, str(V22_SRC))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_1.model import MPGFER as MPGFERV21

from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
from mpg_fer_v2_2.model import MPGFER as MPGFERV22

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting Train pooled features on {device}...", flush=True)

    # Note: augment=False so that exact fixed representations of the 28,709 train images are extracted
    train_path = validate_split_path(PROJECT_ROOT / "data" / "train.csv", "train")
    train_ds = FER2013Dataset(train_path, split="train", augment=False)
    print(f"Loaded Train dataset with {len(train_ds)} samples.", flush=True)

    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Models
    m21 = MPGFERV21(MPGConfigV21()).to(device).eval()
    m22 = MPGFERV22(MPGConfigV22()).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    m21.load_state_dict(c21, strict=True)
    m22.load_state_dict(c22, strict=True)

    # Hooks
    cap21 = {}
    cap22 = {}
    
    def make_hook(d, k):
        return lambda m, inp, out: d.update({k: (out[0] if isinstance(out, tuple) else out).detach()})
    def make_pre_hook(d, k):
        return lambda m, inp: d.update({k: inp[0].detach()})

    m21.pixel_readout_proj.register_forward_hook(make_hook(cap21, "R0"))
    m21.motif_readout_proj.register_forward_hook(make_hook(cap21, "R7"))
    m21.classifier.register_forward_pre_hook(make_pre_hook(cap21, "R8"))
    m21.classifier[2].register_forward_hook(make_hook(cap21, "R9"))
    m21.classifier.register_forward_hook(make_hook(cap21, "R10"))

    m22.pixel_readout_proj.register_forward_hook(make_hook(cap22, "R0"))
    m22.motif_readout_proj.register_forward_hook(make_hook(cap22, "R7"))
    m22.classifier.register_forward_pre_hook(make_pre_hook(cap22, "R8"))
    m22.classifier[2].register_forward_hook(make_hook(cap22, "R9"))
    m22.classifier.register_forward_hook(make_hook(cap22, "R10"))

    p21_orig = {k: [] for k in ["R0", "R7", "R8", "R9", "R10"]}
    p21_flip = {k: [] for k in ["R0", "R7", "R8", "R9", "R10"]}
    p22_orig = {k: [] for k in ["R0", "R7", "R8", "R9", "R10"]}
    p22_flip = {k: [] for k in ["R0", "R7", "R8", "R9", "R10"]}

    with torch.no_grad():
        for batch_i, (x, y) in enumerate(loader):
            if (batch_i + 1) % 50 == 0 or (batch_i + 1) == len(loader):
                print(f"  Batch {batch_i + 1}/{len(loader)} in {time.time() - start_time:.1f}s", flush=True)
            x = x.to(device)
            x_flip = TF.hflip(x)

            # v2.1 orig
            cap21.clear()
            with torch.amp.autocast("cuda"):
                _ = m21(x)
            for k in p21_orig: p21_orig[k].append(cap21[k].cpu())

            # v2.1 flip
            cap21.clear()
            with torch.amp.autocast("cuda"):
                _ = m21(x_flip)
            for k in p21_flip: p21_flip[k].append(cap21[k].cpu())

            # v2.2 orig
            cap22.clear()
            with torch.amp.autocast("cuda"):
                _ = m22(x)
            for k in p22_orig: p22_orig[k].append(cap22[k].cpu())

            # v2.2 flip
            cap22.clear()
            with torch.amp.autocast("cuda"):
                _ = m22(x_flip)
            for k in p22_flip: p22_flip[k].append(cap22[k].cpu())

    out_path = AUDIT_DIR / "train_pooled_features.npz"
    np.savez_compressed(
        out_path,
        v21_orig_r0=torch.cat(p21_orig["R0"]).numpy().astype(np.float32),
        v21_orig_r7=torch.cat(p21_orig["R7"]).numpy().astype(np.float32),
        v21_orig_r8=torch.cat(p21_orig["R8"]).numpy().astype(np.float32),
        v21_orig_r9=torch.cat(p21_orig["R9"]).numpy().astype(np.float32),
        v21_orig_r10=torch.cat(p21_orig["R10"]).numpy().astype(np.float32),
        v21_flip_r0=torch.cat(p21_flip["R0"]).numpy().astype(np.float32),
        v21_flip_r7=torch.cat(p21_flip["R7"]).numpy().astype(np.float32),
        v21_flip_r8=torch.cat(p21_flip["R8"]).numpy().astype(np.float32),
        v21_flip_r9=torch.cat(p21_flip["R9"]).numpy().astype(np.float32),
        v21_flip_r10=torch.cat(p21_flip["R10"]).numpy().astype(np.float32),
        v22_orig_r0=torch.cat(p22_orig["R0"]).numpy().astype(np.float32),
        v22_orig_r7=torch.cat(p22_orig["R7"]).numpy().astype(np.float32),
        v22_orig_r8=torch.cat(p22_orig["R8"]).numpy().astype(np.float32),
        v22_orig_r9=torch.cat(p22_orig["R9"]).numpy().astype(np.float32),
        v22_orig_r10=torch.cat(p22_orig["R10"]).numpy().astype(np.float32),
        v22_flip_r0=torch.cat(p22_flip["R0"]).numpy().astype(np.float32),
        v22_flip_r7=torch.cat(p22_flip["R7"]).numpy().astype(np.float32),
        v22_flip_r8=torch.cat(p22_flip["R8"]).numpy().astype(np.float32),
        v22_flip_r9=torch.cat(p22_flip["R9"]).numpy().astype(np.float32),
        v22_flip_r10=torch.cat(p22_flip["R10"]).numpy().astype(np.float32),
        targets=train_ds.labels.astype(np.int64),
    )
    print(f"Saved {out_path} in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
