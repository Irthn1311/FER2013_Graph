# Research history and closed scope

This page indexes the preserved eight-month history. It is not a new experiment plan.

| Stage | Preserved evidence and decision |
|---|---|
| D5 and D16–D19, TensorFlow/standalone parity | Old main at [`archive/pre-final-main-2026-10-06`](https://github.com/Irthn1311/FER2013_Graph/tree/archive/pre-final-main-2026-10-06), exact commit `48e5141fdff25ae6b67f2014407c701ec3166594`; no old commits were rewritten. |
| MPG-FER v1/v2/v2.1/v2.2 and A0–A7/A2R/A6-R2 audits | [`archive/mpg-fer-local-evidence-2026-10-06`](https://github.com/Irthn1311/FER2013_Graph/tree/archive/mpg-fer-local-evidence-2026-10-06), commit `e5b4ea293ad1af437204d8d58c47f4fcf099d2eb`; full compact local conclusions, provenance, CSVs, producer source and an exact manifest. Historical source is reachable through retained research branches and the closure archive parent. |
| v2.3 early-depth mechanism | [Issue97](https://github.com/Irthn1311/FER2013_Graph/issues/97), architecture commit `08faea291ef425c10cc11ab0bc880e6cef302e97`, reviewed v2.2 base `0a258fd43cc4d8f45afa54ea1328f068c52cbee0`; completed training, mechanism `V23_NOT_SUPPORTED`. |
| Six-seed replication | [Issue99](https://github.com/Irthn1311/FER2013_Graph/issues/99); canonical source/config and all six compact results retained in main. |
| Accepted paper P0–P5 ladder and FIXED_POOL discrepancy | [final ablation](../analysis/final_ablation/README.md); source-locked summaries from ablation-paper-audit commit `c035345b41ef16c15bbafca92cc75d9de3b28b38`, preserved in the closure archive. |
| Readout diagnostic | `READOUT_DIAGNOSTIC_NO_GO`, commit `2b5d5c4939e26dfd73ea24cccc5f9b1d9e092704`. |
| Contextual Scale Recomposition | `CSR_DIAGNOSTIC_NO_GO`, commit `a21cb3ec34287c2275ed442eff0cc06ae48b9bfc`. |
| Pixel Reinspection | `PIXEL_REINSPECTION_NO_GO`, commit `da0969274489ce707f9bf982a6489531058339d0`. |
| Upstream audit | `UPSTREAM_AUDIT_STOP_INCREMENTAL`, commit `89004cb49a8337b21d9aa331d62a1a0721dbd253`. |
| Composer audit | `COMPOSER_AUDIT_STOP`, commit `44daeefe3a1816220b2bb7149295ab85b77d7504`; closes incremental changes under the registered framework. |

The complete committed terminal research chain remains at [`archive/mpg-fer-research-closure-2026-10-06`](https://github.com/Irthn1311/FER2013_Graph/tree/archive/mpg-fer-research-closure-2026-10-06), fixed at `44daeefe3a1816220b2bb7149295ab85b77d7504`. Its ancestry includes all seven final diagnostic/ablation heads. Full A0–A7 compact audit files are archived rather than copied into canonical main.

The local-evidence archive's root [`archive_manifest.json`](https://github.com/Irthn1311/FER2013_Graph/blob/archive/mpg-fer-local-evidence-2026-10-06/archive_manifest.json) lists every payload path, size and SHA256. The manifest excludes its own bytes to avoid a circular hash; its Git blob identifies it. Raw historical tensors/ZIPs and FER2013/checkpoint assets were not silently presumed uploaded by any archive ref.

Five reviewed historical files are archive-only: the pre-final one-shot notebook, the v2.3 frozen-analysis helper and the three stale handoff scripts. They were preserved without modernization. Retained archives are evidence snapshots, not an assertion that every historical runtime input is distributed in Git.

The single canonical transformation supersedes the integration intent of stacked PRs #94/#96/#98/#100/#102/#104. Those PRs are not merged sequentially or closed by this task. Issues and remote branches remain. Historical branch/archive refs must not be deleted until a separate post-review decision.
