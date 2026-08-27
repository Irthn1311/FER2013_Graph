# TF Step 9: execution-contract return-shape hotfix evidence

Status: `PRE_INTERVENTION_TECHNICAL_HARNESS_FIX`

Issue: #25

Exact implementation base: `73b32b1a87505d1f5732e86f85a49cb8e66b2c5b`

## Failed-run preservation

- reviewed archive: `tf_step9_local_residual_slot_decomposition_kaggle_t4.zip`
- archive SHA-256: `ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17`
- classification: `PRE-INTERVENTION TECHNICAL HARNESS FAILURE`
- wrapper status: `TECHNICAL_OR_GATE_FAILURE`
- failure: `LocalResidualSlotProbeError: Frozen execution contract drift`
- scientific result valid: `false`
- scientific interpretation: `null`
- S0-S5 scientific outcome: none

The failed run remains technical evidence only. Kaggle was not rerun for this
hotfix.

## Root cause and functional change

Reviewed Step-6 `validate_frozen_contract()` returns the already validated
execution-contract identity inside its `locked` mapping. Step-9 incorrectly
read the nonexistent top-level key:

```python
contract.get("execution_contract_sha256")
```

The Step-9 harness now consumes the real Step-6 return shape and requires:

```python
locked = contract.get("locked")
isinstance(locked, Mapping)
locked.get("execution_contract_sha256") == EXPECTED_EXECUTION_CONTRACT_SHA256
```

A wrong SHA, missing nested field, or missing/non-mapping `locked` value raises
the unchanged fail-closed error `Frozen execution contract drift`. Step-6
validation is neither duplicated nor weakened; Step-9 only checks the identity
in the structure Step-6 actually returns.

Old Step-9 harness SHA-256:
`a35893cc90c4179d31c101f7db026c4c41eaf2509e9c3b0e19a0c53bc8887645`

New Step-9 harness SHA-256:
`50a310f622cdf9dccf13eff4edf6394f1d39b8ccf315dce5ede07d0a45bdd77a`

## Regression evidence

- focused Step-9 suite: `16 passed`
- main-path mock now matches the real Step-6 structure, including widths,
  scientific payload, and nested validated `locked` identities
- valid nested execution-contract SHA: PASS
- wrong nested execution-contract SHA: fail closed PASS
- missing nested field: fail closed PASS
- missing `locked` mapping: fail closed PASS
- S0 remains exactly equivalent to reviewed Step-7 D0 on the golden fixture
- S5 remains exactly equivalent to reviewed Step-7 D3 on the golden fixture
- S1-S4 isolation semantics remain unchanged
- reviewed Step-7 and Step-6 suites: `43 passed`
- parent-import/PyTorch-runtime isolation: `2 passed`
- fresh Step-9 import: `0` PyTorch modules loaded
- package checksums: `PASS checked=267 failures=0`

## Frozen boundary

- scientific payload unchanged:
  `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- execution contract unchanged:
  `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`
- Step-7 unchanged:
  `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`
- Step-6 unchanged:
  `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`

No file under `src/lap_gnn_tf`, `contracts`, or `validation_assets` changed.
No S0-S5 semantics, Gate A/B/C values, thresholds, decisions, checkpoint/config
identity, graph, features, topology, model, or runtime policy changed.

Stop for research-lead review before any Issue #23 adapter relock or Kaggle
rerun.
