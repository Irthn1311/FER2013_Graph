# LAP-GNN PyTorch OFIX7-mid Candidate

Self-contained extraction of the locked five-seed OFIX7-mid baseline. Normal
runtime imports only `lap_gnn` and installed third-party packages. Parent
repository access is limited to explicit tools under `tools/`.

## Install

```powershell
python -m pip install -e .
```

## Validate

```powershell
python -m lap_gnn.cli.validate `
  --config configs/fer2013_ofix7_mid_seed42.yaml `
  --fer-csv <FER_CSV> `
  --prior-root <PRIOR_ROOT> `
  --device cpu `
  --num-workers 0
```

## Train

```powershell
python -m lap_gnn.cli.train `
  --config configs/fer2013_ofix7_mid_seed42.yaml `
  --fer-csv <FER_CSV> `
  --prior-root <PRIOR_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --device cuda:0 `
  --num-workers 2 `
  --no-resume
```

Resume is deliberately unsupported. Existing non-empty output directories are
refused. No data, prior, cache or checkpoint is bundled.
