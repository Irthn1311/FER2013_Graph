# NPF-anchored conditional pixel residual

This additive experiment implements only the registered variants:

- `R0`: frozen `NO_PIXEL_FUSION` baseline;
- `R1`: `z_base + PixelDelta(rP)`;
- `R2`: `z_base + gate(rP,rM) * PixelDelta(rP)` with a seven-output gate.

The last `PixelDelta` linear layer is zero initialized, so R1 and R2 are
exactly equal to frozen NPF before optimization. Stage 1 freezes the complete
NPF backbone and gives R1/R2 independent AdamW optimizers. The runner shares a
single no-gradient backbone feature pass per training batch; it does not share
or combine residual parameters, losses, selectors, or checkpoints.

Phase 0 and checkpoint selection use PublicTest. PrivateTest is not opened
until both independently selected residual checkpoints are frozen. Stage 2 is
not entered unless the registered GO rule is satisfied.

Kaggle Stage 1 completed and passed the independent artifact validator. R1 and
R2 both selected epoch 2; R2 won the PublicTest selector but had no accuracy
gain over frozen NPF. Its final Test horizontal-flip TTA accuracy was 70.4096%,
below both frozen NPF (70.6604%) and the 71.0504% motif-head reference. The
registered decision is therefore `NO_GO_STOP_STAGE_1`; Stage 2 was not run.

The complete runtime bundle, including the two selected checkpoints and the
Phase-0 logits/readouts archive, is stored outside Git at:

`D:\KaggleStaging\mpg-fer-npf-residual-20261003\downloaded_v1`

`KAGGLE_RUNTIME_CHECKSUMS.sha256` records the complete Kaggle output identity.
`checksums.sha256` covers the lightweight files committed with this analysis.
