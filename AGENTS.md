# Repository rules

MPG-FER v2.3 is the frozen canonical PyTorch model, under the explicit Issue97 framework exception. The incremental research cycle is closed. This repository does not authorize a new architecture, experiment, tuning run or test-driven model selection.

GitHub is the source of truth. Read the assigned Issue and verify the base/source/config/checkpoint identities before implementation. Scientific interpretation and new experiment selection remain lead/reviewer responsibilities.

Preserve scientific source bytes, architecture, features, graph topology, prototype/occurrence semantics, losses, optimizer/scheduler, batch/accumulation, EMA, early stopping, checkpoint comparator, augmentation, seeds, splits, exact-resume state and TTA unless an explicitly reviewed Issue authorizes a change. Never tune or select checkpoints with PrivateTest. Validation-only work must fail closed and must not read test evidence or invoke final evaluation.

Preserve original provenance and frozen evidence. Report missing evidence as UNKNOWN. Implementation tests establish correctness, not scientific support or causal validity. Checkpoints/data/credentials are external artifacts and must not enter ordinary Git. New diagnostics should be additive and leave training behavior unchanged.

Use Python 3.11 or 3.12 and the documented package environment. Run scoped tests, checksum/source/config gates and notebook structural checks; record actual commands and limitations in the PR. Do not run full training without explicit authorization. Merge approval belongs to the reviewer.

Historical TensorFlow and D5/D16–D19 rules/source remain in the immutable pre-final-main archive. This canonical consolidation does not change those historical implementations. Full archives and terminal decisions are indexed in docs/RESEARCH_HISTORY.md.
