# Frozen FULL internal-complementarity diagnostic

This directory implements an inference-only diagnostic over the frozen MPG-FER
v2.3 FULL EMA checkpoint and the frozen Table VI `NO_PIXEL_FUSION` EMA
checkpoint. It does not train, call `backward`, create an optimizer, update a
checkpoint, modify scientific source, or change paper claims.

`run_audit.py` extracts the already-returned fused, pixel auxiliary, and motif
auxiliary logits plus the two readouts from the unmodified model forward pass.
It measures head complementarity, descriptive frozen-logit fusion, checkpoint
weight distance, and the preregistered weight interpolation grid under both
forward semantics. The seven classwise fusion coefficients are explicitly an
offline diagnostic and are never installed into the model.

Runtime outputs are intentionally written outside Git by default:

`D:\KaggleStaging\mpg-fer-ablation7-20261002\analysis\full_internal_complementarity`

Run with the project environment:

```powershell
& 'C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe' run_audit.py
& 'C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe' validate_audit.py
```

The validator reopens the immutable checkpoints, checks all recorded hashes,
recomputes metrics and fusion results from the exported arrays, validates the
cross-fit partitions, independently recomputes the global weight distance, and
checks the complete interpolation grid.

The validated run from 2026-10-03 is indexed by
`RUNTIME_EVIDENCE.json`. Large logits/readouts and PrivateTest-derived files
remain in the external artifact directory and are not committed to Git.
