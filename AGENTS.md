# AGENTS.md

## Purpose

This repository is a research project for graph-based Facial Expression Recognition on FER2013.
For all new work, GitHub is the source of truth. Task-specific requirements belong in GitHub Issues; this file contains durable project rules only.

## Roles

- ChatGPT Web acts as research/technical lead, architect, specification author, and reviewer.
- Codex acts as implementation engineer.
- Codex must read this file and the assigned GitHub Issue before changing code.
- Scientific interpretation and next-experiment selection are not implementation tasks. Codex should report evidence/artifacts, not invent a new research direction.

## Current TensorFlow Direction

All new research implementation must use TensorFlow/Keras unless a GitHub Issue explicitly states otherwise.

PyTorch D16-D19 code and results are historical/reference evidence only. Do not add new PyTorch model work, do not port a new idea to PyTorch first, and do not create a parallel PyTorch research branch.

The current frozen TensorFlow reference is:

`standalone/lap_gnn_tensorflow_ofix7_mid_candidate/`

It is the self-contained TensorFlow/Keras parity candidate for the locked OFIX7-mid model. Treat it as a frozen scientific/parity reference. Do not silently alter its architecture, feature schema, graph construction, training semantics, checkpoint behavior, or parity contracts.

## Research Workflow

For scientific work, follow this sequence:

1. Research question.
2. Observed facts.
3. Unknowns and competing hypotheses.
4. Smallest discriminating diagnostic/experiment.
5. Preregister the experiment in a GitHub Issue.
6. Implement only the Issue scope.
7. Produce required artifacts and tests.
8. Open a PR.
9. ChatGPT reviews the PR against the Issue.
10. Run/analyze the experiment.
11. Record a decision: KEEP, REVERT, or DEFER, plus "What did we learn?"

Do not collapse research, architecture decisions, implementation, and scientific interpretation into one unbounded task.

## Experimental Discipline

- One objective -> one experiment -> one decision.
- Change the minimum number of variables needed to answer the registered question.
- Keep all non-target variables frozen unless the Issue explicitly authorizes a change.
- Never tune or select models using the test set.
- Validation may be used for registered diagnostics/model selection according to the Issue.
- Test evaluation is final reporting only unless the Issue explicitly documents a different already-established protocol.
- Never overwrite a frozen baseline, historical artifact, or reference checkpoint.
- Preserve provenance: base commit, resolved config, seed, relevant package/version information, metrics, and generated artifacts.
- Do not infer scientific validity from "tests pass". Unit/integration tests establish implementation correctness only.
- If required evidence is missing, report UNKNOWN rather than guessing.
- Historical absence of an output is not evidence that an experiment failed.

## Forbidden Silent Changes

Unless the assigned Issue explicitly authorizes them, Codex must not change:

- model architecture;
- GNN layer count/hidden dimensions;
- graph topology or node-selection policy;
- node/edge feature schema;
- MediaPipe prior semantics or generation;
- loss/objective;
- optimizer or scheduler semantics;
- early-stopping semantics;
- checkpoint/model-selection metric;
- train/validation/test split;
- seed protocol;
- augmentation/corruption policy;
- public CLI behavior;
- major dependencies;
- unrelated modules.

If an out-of-scope problem is discovered, document it in the PR and stop short of fixing it unless the Issue requires the fix for correctness.

## Frozen TensorFlow Reference Contract

The TensorFlow OFIX7-mid candidate currently represents the locked reference with 37 node channels, 8 edge channels, hidden size 96, three mean-aggregation edge-context layers, five semantic anchors, 20-token micro-motif readout, 7 classes, and 1,061,192 trainable parameters.

Changes intended only for diagnostics should preferably be additive/read-only and live outside the frozen model/training path unless the Issue explicitly requires touching it.

A diagnostic must not change training behavior merely to make the diagnostic easier to implement.

## Data and Test-Set Isolation

For any diagnostic marked validation-only:

- do not read test predictions, test metrics, test labels, or test result artifacts;
- do not call final test evaluation as a side effect;
- fail closed if the requested input cannot be identified as allowed evidence;
- clearly record which artifacts were read.

The test set must not influence hypothesis selection, thresholds, architecture choice, checkpoint selection, or follow-up experiment design.

## TensorFlow Package Environment

Primary package:

`standalone/lap_gnn_tensorflow_ofix7_mid_candidate/`

Supported Python range from the package metadata is Python >=3.10,<3.13.

Typical validation commands from that package include:

```bash
python -m lap_gnn_tf.cli.inspect_environment
python -m lap_gnn_tf.cli.compare_golden --package-root .
python -m lap_gnn_tf.cli.validate --config configs/fer2013_ofix7_mid_tensorflow_seed42.yaml --golden
pytest -q
```

Run commands from the package directory or from an environment where the package is installed/importable.

Do not run expensive full training unless the Issue explicitly requires it.

## Implementation Requirements

Before coding:

1. Read the assigned Issue completely.
2. Record/verify the requested base commit or base branch.
3. Inspect the relevant implementation and tests.
4. Identify the smallest change set satisfying the Issue.

During coding:

- keep changes scoped;
- preserve backward compatibility unless explicitly waived;
- add/update tests for new behavior;
- prefer deterministic, machine-readable artifacts for research diagnostics;
- make failure modes explicit;
- do not hide missing data by substituting fabricated/default scientific values.

Before opening the PR:

- run the Issue-required tests;
- run relevant existing tests where feasible;
- document commands actually run and their results;
- list files changed;
- list generated artifacts expected from a real run;
- state any limitation or unverified assumption.

## Pull Request Requirements

Every implementation PR should reference the Issue and include:

- objective;
- implementation summary;
- exact scope;
- tests/commands run and outcomes;
- compatibility/regression notes;
- scientific-scope boundary (what the change does NOT prove);
- any required runtime artifact not available in CI/local environment.

Do not claim a research hypothesis is confirmed merely because the implementation passes tests.

## Definition of Done

An implementation task is done only when:

- the Issue acceptance criteria are satisfied;
- implementation is within scope;
- new/relevant tests pass;
- existing relevant tests show no known regression;
- required documentation/artifact schemas are present;
- no forbidden silent change occurred;
- the PR contains enough provenance for ChatGPT to review without relying on chat history.

Merge approval is a reviewer decision, not an implementation-agent decision.
