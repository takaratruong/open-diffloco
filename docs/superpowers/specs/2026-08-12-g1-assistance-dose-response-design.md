# G1 Assistance Dose-Response Design

## Question

Does the minimum analytic torso-wrench scale required to complete each exact
reference suffix decrease over the already completed E012 curriculum?

This is an evaluation-only discriminator. It does not train, tune, select a
policy, or render evidence. Its stop condition is one complete registered
four-checkpoint by five-phase by five-scale matrix.

## Frozen Inputs

- Checkpoints, in causal training order:
  - E008 parent at transition 1,327,104.
  - E012 midpoint at transition 1,474,560.
  - E012 assistance-end at transition 1,622,016.
  - E012 final zero-tail checkpoint at transition 1,720,320.
- Exact phases: `0, 100, 200, 300, 400`.
- Fixed assistance scales: `0, 0.1, 0.25, 0.5, 1.0`.
- Seed zero, exact reference resets, nominal replay-free physics, solver profile
  `g1-4x5`, unchanged 29-dimensional policy action, and the E011 analytic
  torso PD wrench.
- Completion means executing every remaining reference transition without a
  true terminal. Survival short of the suffix is not completion.

The four checkpoints are fixed by training time, not post-hoc zero-wrench
selection. This prevents the E012 selected checkpoint from biasing the trend.

## Execution

One worker process owns one checkpoint and one physical GPU. Within that same
process it loads the policy once, creates the environment once, and evaluates
all 25 phase/scale conditions. It writes a manifest-last JSON artifact with
checkpoint, reference, model, controller, solver, code, device, completion,
survival, and finite/cap-compliant wrench telemetry.

The parent runner launches the four checkpoint workers concurrently, fails
fast if any worker exits, validates all worker artifacts, and writes one atomic
aggregate. It does not import JAX before assigning devices.

No separate render replay is part of this experiment. This directly avoids the
cross-process terminal-count mismatch that invalidated E015.

## Analysis

For each checkpoint and phase, the required scale is the smallest registered
scale that completes the suffix; it is null when even scale 1.0 fails. Null is
treated as infinity only for trend classification.

- `assistance-requirement-decreases`: every phase's required scale is
  non-increasing across all four checkpoints and at least one phase strictly
  improves.
- `mixed-threshold-transfer`: at least one later checkpoint lowers a phase's
  requirement, but the four-checkpoint trend is not componentwise
  non-increasing.
- `assistance-dependent-no-transfer`: no phase's required scale falls below
  its parent requirement.
- `invalid-execution`: any provenance, device, rollout, finiteness, cap,
  artifact, or completeness gate fails.

Only the first outcome justifies another assistance curriculum, using a
competence-gated or slower continuous anneal. Mixed or no transfer stops the
assistance branch and returns the research program to the unassisted
objective/optimization bottleneck. The matrix is descriptive of fixed-scale
support and is not itself evidence that the final zero-wrench policy tracks.

## Safety and Scope

- No hardware, external services, environment mutation, or training.
- Force and torque are bounded by the already validated E011 caps.
- Scale zero must write bit-exact zero wrench.
- All output is finite JSON written atomically beneath the registered run.
- No changes to the training path, checkpoint format, actor, environment, or
  solver.

