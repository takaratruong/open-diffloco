# G1 Raw-Reference AHAC Walking Design

## Purpose

Determine whether contact-stiffness-induced first-order gradient error is the
reason fresh differentiable-physics training improves the beginning of the G1
walk while failing later catches. The treatment retains the raw kinematic
reference. It performs no collocation, dynamic retargeting, reward addition,
observation addition, or motion-specific phase decomposition.

## Scientific question

Can Adaptive Horizon Actor-Critic (AHAC) replace the optimization machinery of
the E023 H24 SHAC recipe and produce a componentwise safer walking policy by
avoiding backpropagation through stiff contact while fitting a less brittle
value estimate?

## Fixed baseline

The scientific baseline is E023. Preserve its G1 model, 50 Hz raw walking
reference, g1-4x5 fixed solver, 512 effective environments, RMR action-noise
schedule, five-bin CAGrad, actor network, observations, rewards, reset
distribution, action contract, seed zero, and 128 actor-update budget.

The treatment changes only the optimizer family from SHAC to AHAC. AHAC
contains three coupled changes from the ICML 2024 implementation and they are
treated as one algorithmic intervention:

1. a global differentiable rollout horizon bounded to 8--24 transitions;
2. a nonnegative dual variable driven by contact stiffness; and
3. a double value critic trained until convergence, capped at 64 iterations.

## Contact-stiffness proxy and calibration

MuJoCo MJX does not expose the dFlex spatial contact tensor used by the AHAC
reference code. For G1, use the transition's six floating-base constraint
forces and accelerations:

`c_t = ||qfrc_constraint[:6] / max(abs(qacc[:6]), 1)||_2`.

All operations are finite, float64, and stop-gradient diagnostics. This is the
same force-over-modified-acceleration principle used by AHAC, with matching
six-dimensional MuJoCo quantities.

Do not tune the threshold against policy performance. Before training, replay
the immutable E023 selected checkpoint over the exact five registered start
phases and set `C` to the deterministic 90th percentile of all finite active
transition signals. Publish the complete raw signal vector, quantile method,
checkpoint/reference/model/code hashes, and resulting scalar. A missing,
nonfinite, nonpositive, or provenance-mismatched calibration invalidates the
experiment.

## Adaptive horizon

Initialize `H=8` and a length-24 zero dual vector. Execute a static length-24
JAX scan, but after `round(H)` transitions freeze the environment carry and
mask reward/contact/critic records. Thus later scan slots contribute neither
state motion nor gradient and the returned carried state is exactly the state
at the active horizon.

For every active time index, average contact stiffness across the effective
population and apply projected dual ascent:

`lambda_t <- max(0, lambda_t + 5e-4 * (mean(c_t) - C))`.

Then update and clip the global horizon:

`H <- clip(H + 5e-4 * sum(lambda), 8, 24)`.

Inactive dual entries remain zero. Persist `H`, the dual vector, active
transition count, stiffness mean/max/quantiles, and truncation fraction in the
checkpoint telemetry. Resumption requires exact AHAC settings unless explicit
change authority is supplied.

## Double critic

Use two independent value heads with identical architecture. Actor bootstrap
and TD(lambda) targets use the pointwise minimum. Train both heads against the
same stopped-gradient target. Run up to 64 critic updates and stop applying
updates once the absolute mean change across the latest five finite critic
losses is below 0.2. Record the executed iteration count, both head losses,
head disagreement, convergence flag, and finiteness.

Fresh AHAC training initializes both heads independently. No E023 critic,
optimizer, normalizer, or actor checkpoint is resumed. E023 is used only for
the mechanical threshold calibration and behavioral comparison.

## Evaluation and truthful success

Evaluate updates 16, 32, 64, and 128 at phases 0/25/50/75/100 under the exact
E023 replay-free evaluator. Report survival and tracking errors separately;
averages cannot compensate for a regressed phase.

The short reference ending is not evidence of stable completion. Every video
and result must explicitly label whether it reached the reference boundary
while already violating the registered termination margins. A checkpoint is
"short-clip competent" only if it reaches all suffix boundaries and every
final-frame termination margin remains within its threshold with non-worsening
trend over the last ten frames. This gate prevents E026-style visible falling
from being called success.

## Outcomes

- `ahac-solves-short-walk`: a finite update completes all five suffixes and
  clears the truthful final-margin gate.
- `ahac-advances`: a finite update componentwise preserves E023
  `116/99/67/49/24`, improves at least one phase, and improves the final-margin
  evidence without a compensating regression.
- `ahac-controls-contact-but-not-behavior`: the learned horizon responds to
  contact and all telemetry is valid, but no checkpoint clears behavioral
  preservation.
- `ahac-does-not-adapt`: horizon/dual telemetry shows no meaningful response to
  registered stiffness despite valid execution.
- `ahac-unstable`: actor, critic, state, dual, stiffness, or checkpoint evidence
  becomes nonfinite or violates its bound.
- `invalid-execution`: any provenance, calibration, treatment-delta, budget,
  resume, telemetry, checkpoint, or evaluator gate fails.

## Scope boundary

This experiment does not authorize threshold sweeps, reward tuning, reference
repair, collocation, phase-specific policies, new observations, domain
randomization, pushes, or multiple seeds. A valid negative result changes the
gradient estimator or state-matching mechanism rather than tuning this run.
