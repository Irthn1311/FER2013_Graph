# TF Research Step 9: local residual-slot harness implementation evidence

Status: `HARNESS_IMPLEMENTATION_ONLY`

Issue: #21  
Exact implementation base: `cd6a6b751d52729f7330adad58d94fbe7d1a7ac4`

## Locked identities

- frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- frozen execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`
- reviewed Step-7 tool: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`
- reviewed Step-6 support tool: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`

The new harness verifies the reviewed Step-7 and Step-6 file identities before
loading them. The package finalizer reproduced both frozen contract identities
without changing `src/lap_gnn_tf`, `contracts`, or `validation_assets`.

## Implemented fixed pathway

The external validation-only harness obtains the official post-`part_pool`
state from the reviewed Step-7 D0 manual forward. It evaluates exactly this
fixed order on the same immutable source batch and model:

1. `S0 official_manual_forward`
2. `S1 mouth_local_residual_zero`
3. `S2 eye_local_residual_zero`
4. `S3 brow_local_residual_zero`
5. `S4 nose_cheek_local_residual_zero`
6. `S5 all_local_residuals_zero_anchor`

S1-S4 replace only the named pooled local residual embedding with zeros
immediately before readout. S5 replaces all four local pooled residuals and is
golden-regressed against reviewed Step-7 D3. Global pooling, validity flags,
`readout.part_soft`, context output, message-passing state, graph tensors,
coordinates, labels, and sample IDs remain official.

There is no CLI condition, slot, pair, split, training, optimizer, gradient,
test, raw-prior mutation, or graph-rebuild selector. The checkpoint loader uses
one model with `compile=False`, and both checkpoint bytes and model weights are
hashed before and after evaluation.

## Registered gates and decisions

- future Gate A: prediction agreement `== 1.0`, logit difference `<= 1e-5`,
  probability difference `<= 3e-6`
- future Gate B S0 reference: accuracy `0.63137364168292`, macro-F1
  `0.5932591901893336`, loss `1.1537981840361535`
- future Gate C S5 / Step-8 D3 reference: accuracy `0.22596823627751464`,
  macro-F1 `0.1958426679087715`, loss `1.883221954371022`
- reference tolerances: accuracy `0.001`, macro-F1 `0.001`, loss `0.005`
- individual slot labels: `>=10 pp HIGH_SLOT_SENSITIVITY`, `5-10 pp
  MODERATE_SLOT_SENSITIVITY`, `<5 pp LOW_SLOT_SENSITIVITY`
- decisions: `SINGLE_HIGH_LOCAL_SLOT`, `MULTIPLE_HIGH_LOCAL_SLOTS`, or
  `NO_SINGLE_HIGH_LOCAL_SLOT_WITH_JOINT_DEPENDENCY`

Bounded `--limit-val-batches` execution is explicitly marked implementation
smoke only and produces no scientific interpretation. It is forbidden for a
future registered full run.

## Verification

- focused Step-9 harness suite: `12 passed`
- reviewed Step-7 plus Step-6 suites: `43 passed`
- parent-import and TensorFlow package PyTorch-isolation tests: `2 passed`
- fresh harness import: `0` PyTorch modules loaded
- package checksums: `PASS checked=267 failures=0`
- frozen scientific payload reproduced exactly
- frozen execution contract reproduced exactly
- `git diff --check`: required before PR publication

No Kaggle execution was performed. The real Issue #7 checkpoint was not run on
full validation, and this implementation evidence contains no S1-S4 scientific
outcome.
