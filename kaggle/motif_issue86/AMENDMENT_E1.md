# Issue #86 Execution Amendment E1 Governance

## Objective & Classification

Execution Amendment E1 is a **wrapper and transport fix only**. It resolves the pre-submission Kaggle execution blocker without altering any scientific parameter, code, or preregistered threshold.

Classification: **`TECHNICAL WRAPPER AMENDMENT — SCIENTIFIC SOURCE IMMUTABLE`**.

---

## Provenance & Locks

- **Preregistration**: `6e6ba806cc9ff0bc2b8a5534807d55f7feb00d73`
- **Scientific implementation**: `16a84b2b36f0d3584afd0a547487c4373dc22128`
- **Historical blocked wrapper**: `728473a9928c2982ba5ff8b22adf2a9034be94ac`
  - Notebook: `notebooks/motif-qualification-train-kaggle.ipynb`
  - SHA256: `b2791ad54d9b69a1740fce19c8fa380be5dab5c9f22d05eb7585d544e4313e66`
  - Status: `BLOCKED PRE-EXECUTION — ENV ROUTING UNSUPPORTED` (preserved byte-identical)
- **Input artifact hashes**:
  - Official Train CSV: `deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8`
  - v533 dictionary: `68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b`
  - Accepted CRS Train model: `77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a`

---

## Problem Solved

The historical notebook selected its execution target (`BUILD_SUBSTRATE`, `RUN_REPLICATE`, `FINALIZE_TRAIN`) via process environment variables (`MOTIF_EXECUTION_MODE`, `MOTIF_STABILITY_ARM`, `MOTIF_STABILITY_REPLICATE_ID`), defaulting to `REVIEW_ONLY`.

The supported Kaggle `kernels push` CLI/API interface provides no runtime environment variable injection contract, and Kaggle Secrets are not exposed as process environment variables.

---

## E1 Architecture

1. **Deterministic Generator**: `kaggle/motif_issue86/generate_execution_units.py` generates exactly 42 execution scripts and their SHA256-verified manifest with Black-standard formatting.
2. **Machine-Readable Manifest**: `kaggle/motif_issue86/execution_units_manifest.json` maps all 42 unit IDs to their filename, mode, arm, replicate ID, scientific source SHA, and script SHA256.
3. **42 Standalone Scripts**:
   - 1 Substrate: `kaggle/motif_issue86/generated/motif_build_substrate.py`
   - 20 Ordered M replicates: `kaggle/motif_issue86/generated/motif_M00.py` .. `motif_M19.py`
   - 20 Destroyed C replicates: `kaggle/motif_issue86/generated/motif_C00.py` .. `motif_C19.py`
   - 1 Finalizer: `kaggle/motif_issue86/generated/motif_finalize_train.py`

---

## Execution Invariants

Every unit script:
- Pins `SCIENTIFIC_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"`.
- Checks out scientific source detached (`git checkout --detach SCIENTIFIC_SHA`) and asserts clean HEAD.
- Executes full `pytest` before verifying or accessing any FER input.
- Verifies input hashes before launching runner command.
- Hardcodes its unit identity as Python literals without environment variable or secret routing.
- Invokes exactly one registered `pixel_relational_motif_e0.motif_train_runner` CLI command.
- Emits an SHA256 and byte-size inventory of all output files.

---

## Six-Slot Kaggle Assignment (Execution Orchestration Only)

- **Slot 0 — nuyntai**: `M00 M06 M12 M18 C04 C10 C16`
- **Slot 1 — nuyntai**: `M01 M07 M13 M19 C05 C11 C17`
- **Slot 2 — thanhhgng**: `M02 M08 M14 C00 C06 C12 C18`
- **Slot 3 — thanhhgng**: `M03 M09 M15 C01 C07 C13 C19`
- **Slot 4 — trngthngcnhi**: `M04 M10 M16 C02 C08 C14`
- **Slot 5 — trngthngcnhi**: `M05 M11 M17 C03 C09 C15`

Each job executes exactly one unit script (`kernel_type="script"`, `code_file="kaggle/motif_issue86/generated/motif_<unit>.py"`).
Multiple replicates are never combined into a single script.
