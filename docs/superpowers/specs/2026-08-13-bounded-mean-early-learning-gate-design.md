# Bounded-Mean Early Learning Gate Design

## Goal

Determine whether the action boundary validated by E-20260813-003 can learn
for 98,304 transitions without reproducing E-20260813-002's actor-mean
divergence. This is an early discriminator, not a locomotion or sim-to-real
claim.

## Alternatives considered

1. Extend the existing action-parity runner with one explicit early-learning
   mode. This reuses the pinned preflight, trainer, evaluator, and evidence
   validation and is the selected approach.
2. Start the existing 786,432-transition runner and interrupt it manually at
   98,304. This is rejected because the executed budget would differ from the
   registered contract and a missed interrupt would waste compute.
3. Add a separate early-learning runner. This is rejected because it would
   duplicate the action contract and increase the chance that training and
   evaluation drift apart.

## Training contract

Add `--early-learning-gate` to the existing
`tools/run_g1_rmr_action_space_parity.py` runner. It is mutually exclusive with
`--gate-only` and requires `--decoupled-exploration`.

The mode performs exactly 98,304 transitions: 16 effective-512 H12 updates,
with one checkpoint at step 98,304. It inherits E-20260813-003 unchanged:

- tanh-bounded learned actor mean;
- unclipped reparameterized Gaussian action sample;
- scalar-1.0 to learned 29-joint RMR noise schedule over 800,000 transitions;
- full reference-residual scale 1.0;
- clean observations and exact uniformly sampled reference-state resets;
- nominal model, friction, mass, gains, COM, and effort limits;
- no pushes, domain randomization, terrain force, or torso assistance;
- future references at 4/8/12 steps;
- two 256-environment shards, H12, and five-bin CAGrad;
- seed zero and the g1-4x5 solver profile.

## Evaluation and evidence flow

After training, the runner evaluates the exact step-98,304 checkpoint before
publishing validation:

1. An exact-phase-zero training-distribution rollout uses clean observations
   and the checkpoint's scheduled action noise for 120 transitions. It records
   action mean, epsilon, noise scale, sampled/effective action, trajectory,
   video, contact sheet, and summary.
2. A deterministic clean phase-zero rollout runs until its first terminal or
   the end of the 499-transition reference suffix.
3. The parent process independently binds both summaries to the checkpoint
   SHA, reference, solver, phase, and expected training/evaluation mode.
4. It computes actor-mean RMS, maximum, and saturation fraction
   `mean(abs(action_mean) >= 0.95)` from the noisy action tape. It extracts the
   first noisy episode survival before any auto-reset and the clean survival
   from the clean trajectory.
5. Gate validation is written atomically only after all provenance, shape,
   finiteness, CAGrad, and behavior checks pass.

The Dashboard receives a robot-only noisy training episode and a clean control
only after the canonical run artifacts are valid. Canonical evidence remains
under `runs/`.

## Quantitative outcomes

The run advances only if all of the following hold:

- exactly 98,304 transitions and 16 updates complete;
- all five CAGrad bins are occupied and finite, with valid positive actor
  gradient and update norms;
- the `(120, 29)` noisy tape is finite and hash-bound;
- actor-mean maximum is at most 1.000001;
- actor-mean saturation fraction is below 0.20;
- clean phase-zero survival is at least 40 transitions;
- no unexpected observation noise, reset perturbation, physics randomization,
  push, assistance, clipping, or provenance drift occurs.

Passing these gates authorizes a separately registered longer continuation; it
does not itself establish robust tracking. Failure due to saturation or clean
survival stops this recipe and pivots to either a lower actor learning rate or
a trust-region-style actor objective. Invalid artifacts or provenance cause an
execution repair only.

## Error handling and tests

- Parser tests enforce mode exclusivity and the decoupled requirement.
- Builder tests prove the early mode changes only total steps, checkpoint
  interval, and the already-established short-budget curriculum endpoints.
- Validator tests reject wrong checkpoint hashes, rollout modes, step counts,
  shapes, nonfinite values, escaped means, saturation at or above 0.20, clean
  survival below 40, and missing CAGrad evidence.
- Runner tests prove both evaluations complete before validation publication.
- Existing legacy, E-20260813-002, and E-20260813-003 behavior remains
  unchanged when the new flag is absent.
