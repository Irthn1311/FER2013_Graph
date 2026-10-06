# Canonical MPG-FER v2.3 package

This directory contains the unchanged scientific source, tests, locked notebook generators/templates, operational staging tools, config and compact final evidence. Start with the [root overview](../../README.md) and [reproducibility instructions](../../docs/REPRODUCIBILITY.md).

Source SHA256: `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87`.
Seed-42 scientific config SHA256: `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2`.
Selected FULL checkpoint SHA256: `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e`.

`src/mpg_fer_v2_3/` and the existing tests/tools/templates retain their original bytes. The historical `run_v23_frozen_analysis.py` helper is archive-only because it requires v2.1/v2.2/A6 runtime assets. `aggregate_multiseed_results.py` is retained as an external-archive replay tool: its historical `official_runs/` inputs are intentionally absent from Git. Use the new read-only compact-package verifier for a clone-only integrity/statistics check.

```sh
python research/mpg_fer_v2_3/tools/verify_canonical_results.py
```

`v23_provenance.json` adds explicit operational/scientific/result scopes while retaining every historical field and the original unsupported Issue97 verdict. No training behavior changes.
