# Zero-Head Recovery Feature Transfer Implementation Plan

## Task 1: Pure feature-transplant contract

- Add focused RED tests for exact expert hash/tree/shape validation, Dense_0
  transfer, exact-zero Dense_1, scalar provenance, and invalid resume settings.
- Implement the smallest pure loader/transplant/resume helpers in
  `residual_preview_adapter.py`.
- Run the focused residual tests and static checks.

## Task 2: SHAC migration integration

- Add explicit path/SHA arguments to `train` and persist them in hparams.
- Apply the transplant only during the authorized E023-to-residual migration,
  before fresh adapter optimizer initialization.
- Extend the migration report with expert SHA, Dense_0 identity, zero-head
  identity, and initial action equivalence.
- Add compiled integration tests for forwarding, resume fail-closed behavior,
  exact frozen parent, and nonzero adapter gradients after one update.

## Task 3: E041 runner and evaluation

- Add one immutable runner derived from E027 with only the feature-transfer
  initialization delta and checkpoints at updates 8/16/32/64.
- Add paired full-bank evaluation and no-regression selection using the existing
  E040 evidence contract where possible.
- Register E041, validate and dry-run it, then launch the single seed.

## Task 4: Curate and decide

- Independently validate training/evaluation artifacts and the selected result.
- Update the experiment record, current state, finding, generated vault page,
  and Dashboard.
- Continue from the selected checkpoint only if the preregistered eligibility
  and advancement gates pass.
