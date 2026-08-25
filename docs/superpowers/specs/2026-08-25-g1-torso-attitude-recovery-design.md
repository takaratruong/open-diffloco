# G1 Torso-Attitude Recovery Design

## Goal

Turn E026's completed short walk into a stable short walk by reducing its
late backward torso lean without sacrificing any of its matched five-phase
survival.

## Evidence boundary

E-20260825-000 proves that the frozen E026 update-16 residual improves the
same-process E023 parent from `114/99/69/49/24` to `124/99/74/49/24` with no
survival regression.  It also shows that a body-position-error gate activates
only after the necessary recovery action.  The E026 phase-zero torso has about
13 degrees mean and 18 degrees maximum reference-relative pitch error over its
last ten transitions.  This experiment therefore changes the training
objective, not the selector, observations, reset distribution, optimizer, or
physics.

## Single causal change

Resume the exact E026 update-16 checkpoint and keep its frozen E023 parent,
normalizer, residual architecture, optimizer state, carried reset bank,
effective-512 H24 batching, five-bin clipped CAGrad, action-noise schedule,
solver, controller, and reference unchanged.  Add a weight-one torso
orientation reward to the residual continuation:

`2 - sqrt(1 + 2 * (theta / 0.4)^2)`

where `theta` is the shortest reference-relative quaternion angle for
`torso_link` after the same yaw alignment used by the existing rigid-body
tracking reward.  The legacy weight is exactly zero and must preserve the old
reward path.  A resumed nonzero weight requires explicit authority.

Run 32 additional updates and archive updates 8, 16, 24, and 32.  This is one
bounded continuation, not a weight, learning-rate, horizon, or reset sweep.

## Selection

Evaluate the source and every checkpoint from exact phases 0/25/50/75/100 in
one replay-free evaluator.  A candidate is eligible only if it:

- preserves source survival componentwise at `124/99/74/49/24`;
- has finite, occupied, valid clipped-CAGrad telemetry and exact frozen-parent
  state;
- preserves mean body-position and body-orientation errors within five percent
  on every source-completed suffix; and
- reduces phase-zero last-ten mean absolute torso pitch error by at least 25
  percent while not increasing its maximum absolute error.

Among eligible candidates select minimum last-ten mean absolute torso pitch,
then maximum, then earliest update.  Training reward never selects a policy.

## Outcomes

- `torso-objective-stabilizes-short-walk`: an eligible checkpoint is selected.
- `torso-objective-redistributes`: at least one checkpoint reduces the tail
  lean but violates componentwise survival or tracking preservation.
- `torso-objective-insufficient`: training is valid but no checkpoint reduces
  the tail lean by 25 percent.
- `invalid-execution`: any provenance, resume, objective, telemetry,
  evaluation, or artifact gate fails.

## Interpretation limit

Success is a stable 2.48-second short-walk result only.  It authorizes an
unchanged longer-walk continuation; it is not yet a long-motion or sim-to-real
claim.
