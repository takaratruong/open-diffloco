# Zero-Head Recovery Feature Transfer Design

## Objective

Test whether E038 learned a useful recovery representation even though E040
proves that its final correction policy is globally destructive.

## Alternatives considered

1. **Copy the whole E038 adapter.** Rejected: E040 shows 73/120 regressions.
2. **Hard state/phase gate.** Rejected for this test: E029 could not certify the
   existing nearest-support gate, and a hand-written phase gate would only
   memorize the observed failure band.
3. **Copy hidden features and zero the output head (selected).** Copy exactly
   E038 `Dense_0`; retain a bit-exact zero `Dense_1`. The initialized actor is
   therefore exactly E023 everywhere while SHAC begins from recovery-shaped
   features rather than a random hidden layer.
4. **Teacher loss plus zero initialization.** Plausible later, but it adds a new
   loss and weight before testing the smaller representation-transfer claim.

## Fixed treatment

- Resume the exact E023 TrainState and LAFAN reference migration used by E027.
- Create the same 328-to-256-to-29 ELU/tanh residual adapter.
- Copy only E038's hash-bound 328-to-256 kernel and 256 bias.
- Require the 256-to-29 kernel and 29 bias to remain bit-exact zero, and prove
  parent-plus-adapter actions equal E023 before the first update.
- Initialize fresh adapter optimizer state; freeze the E023 parent, its prior
  optimizer state, and actor normalizer exactly.
- Preserve E027's H24, effective population 512, five-bin CAGrad, per-environment
  clipping, 25-percent history-faithful carried resets, RMR action noise,
  zero bootstrap, parent-action proximal anchor 1.0, nominal physics, and seed.
- Run 64 updates with immutable checkpoints after 8/16/32/64 updates.

## Selection

Every checkpoint is evaluated replay-free on the ordinary five-phase LAFAN grid
and on all 120 E027 carried states at H32. A checkpoint is eligible only if it
has no carried-state survival regression versus E023. Select lexicographically
by phase-zero H32 count, total H32 count, minimum/median/mean carried survival,
ordinary five-phase minimum/median/mean, then earliest update.

## Outcome map

- `zero-head-features-advance`: an eligible checkpoint improves at least one
  carried state or ordinary phase without regressing any carried state.
- `zero-head-features-solve`: an eligible checkpoint completes all ordinary
  suffixes and all registered H32 carried recoveries.
- `zero-head-features-insufficient`: training is valid but no checkpoint is
  eligible or no eligible checkpoint improves E023.
- `invalid-execution`: any provenance, zero-effect migration, frozen-state,
  finite-gradient, CAGrad, action-boundary, checkpoint, evaluation, or artifact
  contract fails.

## Limits

This tests safe representation transfer for one fixed motion and bank. It is not
an architecture sweep, a generic recovery gate, or sim-to-real evidence.
