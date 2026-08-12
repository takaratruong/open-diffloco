# G1 Terminal-Value Calibration Design

## Purpose

Determine whether the remaining differentiable-physics tracking failure is a
terminal-value problem before changing the actor again. E017 showed that more
H12 updates do not raise the robust survival floor, and E018 showed that a
broader root-only reset distribution changes the phase tradeoff without raising
that floor. The actor therefore needs information about failure occurring well
beyond its 12-step differentiable window, not another reset or duration sweep.

## Selected Diagnostic

Evaluate the exact E012 selected zero-wrench checkpoint from phases
`0/100/200/300/400`. During each ordinary replay-free rollout, record the
checkpoint target critic's value at every pre-step state and the realized
discounted return from that state through the actual terminal transition. Use
the training-identical critic observation, normalizer, gamma, actor, residual
adapter, reference, controller, and fixed `g1-4x5` solver.

The primary slice is the H12 boundary: for every start phase, compare the value
predicted after 12 carried transitions with the subsequently realized return.
Also report all-state Pearson correlation, rank correlation, normalized RMSE,
bias, and per-phase metrics. Low training critic loss is not accepted as
calibration evidence because its targets are themselves bootstrapped.

## Alternatives Rejected

- More root perturbation tuning: E018 directly tested it and only redistributed
  competence.
- More identical training: E017 directly tested it and did not improve the
  robust floor.
- Doubling the actor horizon immediately: E009 was finite but regressed every
  phase. A longer-horizon critic target is separable from a longer actor adjoint.
- Actor training before diagnosis: this would mix critic repair with another
  policy update and leave the causal failure ambiguous.

## Frozen Boundary

The diagnostic changes no checkpoint and performs no optimizer update. It
requires exact hashes for code, checkpoint, reference, model, and controller;
JAX x64; one enumerated GPU; the delta-preview 256-wide residual actor; ten-frame
history; lookahead `(4,8,12)`; gamma `0.99`; and exact-zero external wrench.
Every rollout must terminate naturally, remain finite, and reproduce the
survival vector inside the same manifest as its value/return trace. Exact
equality to an archived process's survival count is deliberately not an
identity gate: prior same-checkpoint MJX evaluations established small
process-level contact divergence. Code, checkpoint, assets, solver, and seed
remain strict identity gates; calibration is computed only between predictions
and returns from the same realized trajectory.

## Outcomes

- `terminal-value-calibration-adequate`: all-state rank correlation is at least
  0.8, normalized RMSE is at most 0.25, and every H12-boundary relative error is
  at most 0.25. Do not retune the critic; design a different long-credit
  mechanism.
- `terminal-value-miscalibrated`: execution is valid but any calibration gate
  fails. Freeze the actor and train only the critic on longer carried Monte
  Carlo returns before a separately registered actor continuation.
- `invalid-execution`: any provenance, replay-free rollout, zero-wrench,
  finiteness, phase, terminal, or artifact gate fails. Repair only execution.

Passing scalar calibration would not prove that the critic's state derivative
is correct. Failing it is sufficient to reject the current terminal bootstrap
as a reliable long-credit target; passing it would require a subsequent local
directional audit before actor training.

## Artifacts

Write one manifest-last JSON containing provenance, raw per-state values and
returns, H12 records, phase summaries, aggregate calibration metrics, and the
selected outcome. No rendering or policy selection is part of this diagnostic.
