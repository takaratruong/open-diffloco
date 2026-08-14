# Fresh Full-Action H24 Walking SHAC Design

## Scientific question

Does increasing the differentiable physical credit horizon from 12 to 24
steps teach fresh SHAC the anticipatory right-leg catch command that is missing
from E021?

E021 establishes the boundary. Full unbounded PPO-style action authority
improves fresh SHAC from E020's diagnostic-best `49/45/52/49/24` to
`62/63/55/49/24`, but the selected actor still terminates at phase 62. During
phases 40--61, the competent PPO's right-hip action averages `-1.171` while
E021 averages `+0.421`; their signs disagree on every step. Future-reference
observations expose the desired kinematics, but E021's zero-bootstrap H12 loss
cannot assign the phase-62 failure to the beginning of this catch window.

## Alternatives considered

1. **Preserve 128 updates and effective population 512 (selected).** Change
   only `unroll_length` from 12 to 24 and scale the transition/checkpoint
   budgets accordingly. This preserves the number and statistical quality of
   optimizer decisions. E020 is the negative control showing that additional
   H12 transitions alone do not solve the walk.
2. **Preserve total transitions with 64 updates.** This would halve optimizer
   updates and make a negative result ambiguous between horizon and
   under-optimization.
3. **Preserve transitions and updates by halving the effective population.**
   This would change gradient averaging at the same time, reintroducing a
   known variance sensitivity.

## Treatment

Derive the complete E021 treatment and change only:

- `unroll_length: 12 -> 24`;
- `total_steps: 786432 -> 1572864` so the run still performs exactly 128
  optimizer updates;
- `checkpoint_interval: 98304 -> 196608` so checkpoints remain every 16
  updates.

Preserve the fresh zero output head, no PPO/BC/resume state, actor architecture,
full residual scale 1.0, linear actor mean, no sampled-action clip, fixed 0.2
reparameterized action noise, LR 1e-3, zero terminal bootstrap, clean
observations, exact uniform reference-state initialization, nominal physics,
no pushes/randomization/assistance/terrain, physical population 256 with two
gradient-accumulation shards, five-bin CAGrad, per-environment clip 1.0, future
reference deltas at 4/8/12, solver profile g1-4x5, seed zero, and reference.

## Evidence and selection

Require eight exact checkpoint archives at updates
16/32/48/64/80/96/112/128. Evaluate updates 16/32/64/128 at exact phases
0/25/50/75/100. The selected checkpoint is the componentwise-safe candidate
with the greatest phase-zero survival, then normalized minimum, median, mean,
and earlier update. Componentwise-safe means no phase falls below E021's
`62/63/55/49/24` vector.

For the selected checkpoint, produce a clean rollout, a fixed-0.2 noisy
training rollout, learning/action diagnostics, and a PPO-versus-SHAC right-leg
trace over phases 40--61. The command diagnostic is secondary to replay-free
survival and cannot select a behaviorally regressing checkpoint.

## Outcome map

- **H24 solves:** one checkpoint completes `124/99/74/49/24`.
- **H24 advances catch credit:** a componentwise-safe checkpoint reaches at
  least 80 phase-zero transitions.
- **H24 shifts commands only:** no checkpoint reaches 80 safely, but a
  componentwise-safe checkpoint reduces the fixed PPO right-hip action RMSE
  from E021's `1.667477` by at least 20 percent and reduces opposite-sign
  fraction below 0.5.
- **H24 insufficient:** valid training produces neither behavioral advancement
  nor the registered command shift.
- **H24 unstable:** states, losses, actions, gradients, or updates become
  nonfinite or violate the per-environment gradient bound.
- **Invalid execution:** provenance, treatment isolation, checkpoint cadence,
  telemetry, evaluation, or required artifacts fail.

This is one experiment, not a horizon sweep. A negative result retires fixed
H24 as the next recipe ingredient and moves the program to a different carried
objective or state-distribution intervention.
