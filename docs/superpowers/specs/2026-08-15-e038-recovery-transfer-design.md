# E038 Recovery Expert Transfer Design

## Objective

Measure whether the E038 state-conditioned recovery expert transfers beyond
the 416 state/action pairs used for distillation before differentiable SHAC
changes its parameters.

## Alternatives considered

1. **Matched all-bank evaluation, then SHAC (selected).** Evaluate the frozen
   E023 parent and frozen E038 expert from all 120 immutable E027 carried states.
   This separates offline generalization from later differentiable learning.
2. **Immediate SHAC continuation.** This is faster but confounds expert
   initialization with learning and leaves no clean transfer baseline.
3. **Phase/state-gate the expert.** This protects unrelated states but avoids
   rather than answers whether the learned feedback representation transfers.

## Fixed experiment

- Exact E023 actor, normalizer, controller, LAFAN reference, E027 carried bank,
  E038 expert checkpoint, seed zero, and `g1-4x5` solver.
- Reconstruct all 120 carried states: 24 each from source phases
  0/100/200/300/400.
- Run matched H32 closed-loop parent and expert arms. The expert action is the
  frozen parent action plus the frozen E038 328-to-256-to-29 ELU/tanh residual;
  the environment retains the exact final action boundary.
- Persist per-step qpos, phase, parent/correction/raw/effective actions, alive,
  terminal, reward, and normalized termination errors for both arms. Publish a
  hash-bound manifest last.

## Outcome map

- `recovery-expert-generalizes`: phase-zero reproduces at least 12 E038 H32
  successes; no one of 120 states has lower survival than E023; at least 10 of
  96 untouched states newly reach H32.
- `recovery-expert-local-only`: phase-zero reproduces at least 12 E038 H32
  successes, no state regresses, but fewer than 10 untouched states reach H32.
- `recovery-expert-mixed-transfer`: finite execution produces both survival
  improvements and regressions, or phase-zero reproduces only 10-11 successes.
- `recovery-expert-destructive`: phase-zero reproduces fewer than 10 successes,
  untouched-state median survival falls below E023, or any survival regression
  occurs without an improvement.
- `invalid-execution`: any provenance, shape, state reconstruction, finite,
  action-boundary, paired-evidence, or artifact contract fails.

## Decision

Only after this result is curated will one progressive SHAC treatment be
registered. A generalizing/local-only result may use E038 directly as the
residual initialization. A mixed/destructive result must preserve E038 through
an explicit state gate or supervised anchor rather than exposing its global
correction unchanged.

## Limits

This is an offline-policy evaluation on registered carried states, not
differentiable learning, a complete locomotion claim, or sim-to-real evidence.
