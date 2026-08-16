# G1 Objective-Direction Audit Design

## Decision

Run one frozen-policy diagnostic at the retained E023 LAFAN boundary. On one
immutable training-matched state batch, compute and compare four adapter
directions:

1. the current zero-bootstrap H24 differentiable-physics direction under fixed
   action-noise tape A;
2. the same H24 direction under independent fixed action-noise tape B;
3. a zero-bootstrap H48 differentiable-physics direction under tape A, whose
   first 24 samples are byte-identical to the H24 tape; and
4. the H24 terminal-value bootstrap component from E023's unchanged target
   critic under tape A.

Apply no optimizer update during gradient capture. Then form equal-norm
gradient-descent proposals from the H24, H48, and bootstrap directions and run
a preregistered, bounded line search against the complete 120-state carried
bank. Only carried-safe candidates proceed to the five ordinary replay-free
LAFAN suffixes. This is a diagnostic; no candidate becomes a retained policy.

## Question

Does current G1 training fail because stochastic H24 gradients are
inconsistent, because H24 points away from longer carried credit, because the
terminal critic supplies a conflicting direction, or because even aligned
directions are behaviorally unsafe under local parameter steps?

The outcome changes the next branch:

- low H24 tape-to-tape agreement motivates variance reduction;
- H48 behavioral superiority motivates a multihorizon actor objective;
- bootstrap behavioral superiority motivates a phase-conditioned calibrated
  bootstrap;
- aligned but unsafe directions motivate a carried-state constrained optimizer
  rather than another objective or architecture change.

## Alternatives Considered

### Infer direction from checkpoint deltas

E042 checkpoint deltas are already positively aligned, but Adam-integrated
parameter changes cannot separate H24 physics, longer credit, and critic
bootstrap. This is retained as supporting context, not the discriminator.

### Train separate H24, H48, and bootstrap arms

This directly measures final learning but introduces optimizer history,
state-distribution drift, and substantial compute. It cannot explain *why* the
arms differ. Defer training until the frozen diagnostic identifies a direction
worth testing.

### Exact component-gradient capture and bounded line search — selected

This isolates objective components at one policy and one state distribution.
Common random numbers make H24/H48 differences causal, while a second H24 tape
measures stochastic consistency. Replay-free proposals determine whether a
geometric difference is behaviorally meaningful.

## Immutable Boundary

Use these exact inputs:

- E023 checkpoint:
  `/home/ubuntu/projects/diffsim2real-lab/runs/E-20260814-023/20260815T023359Z/seed-0/training_runs/training_runs/shac_20260814_193402/checkpoint_step_1572864.pkl`,
  SHA-256 `2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f`;
- sibling hparams SHA-256
  `a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650`;
- LAFAN reference SHA-256
  `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`;
- 120-state carried bank SHA-256
  `d91dfb1b5190f14a5204cb16abbf527ede4f08e0a9b46cec9dfa602500d708a5`;
- G1 XML SHA-256
  `5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1`;
- controller artifact SHA-256
  `f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee`;
- `g1-4x5` fixed solver, JAX x64, seed zero, nominal physics, no observation
  noise, root perturbation, domain randomization, pushes, terrain, or torso
  assistance.

Reconstruct E041's exact zero-effect representation boundary: freeze the E023
parent and actor normalizer, copy the verified E038 hidden adapter layer, retain
an exact-zero output head, and initialize no optimizer. Prove composite actions
are bit-identical to E023 before gradient capture.

## Fixed State And Noise Batch

Build one effective population of 512 starts matching the carried-recovery
training mixture:

- 128 carried rows sampled deterministically and uniformly from the complete
  120-row bank using seed zero; and
- 384 exact reference-state starts sampled deterministically and uniformly
  over the 499 reference transitions.

Restore qpos, qvel, phase, previous action, full ten-frame actor history, and
critic context exactly. Persist row source, source index, phase bin, and all
identity checks. Reject any non-finite value, inconsistent newest history
frame, reference mismatch, or physical/context drift.

Generate two independent fixed float32 RMR action-noise tapes, A and B, with
shape `(512, 48, 29)`. H24-A and H48-A share the first 24 samples exactly.
H24-B uses the first 24 samples of tape B. The policy observation, action
boundary, reward, termination, and auto-reset semantics match E042 training.

## Gradient Capture

Compute per-environment adapter gradients before aggregation. For every
direction, preserve both raw and training-effective diagnostics:

- finite fraction and norm distribution before clipping;
- exact per-environment norm clipping at 1.0;
- five starting-phase bins and their counts;
- per-bin aggregate gradients, Gram matrix, cosine matrix, and valid CAGrad
  combination;
- flattened aggregate direction and norm;
- pairwise cosine and dot product among H24-A, H24-B, H48-A, and bootstrap-A;
- per-environment and per-phase cosine summaries where both directions are
  nonzero.

The H48 loss is the same discounted differentiable reward accumulation as H24,
with zero terminal bootstrap and no change except horizon. The bootstrap
component is the gradient of the discounted H24 target-critic terminal value,
with immediate H24 rewards excluded. Stop gradients through critic parameters;
differentiate only the actor adapter through the H24 trajectory and terminal
critic input.

Persist aggregate gradient vectors, phase aggregates, norms, cosines, state
identities, noise hashes, and complete provenance in atomic NPZ and JSON
artifacts. Full per-environment gradient tensors need not be persisted; compute
their registered summaries before releasing device memory.

## Equal-Norm Proposal Test

Use descent directions `-g / ||g||` for H24-A, H48-A, and bootstrap-A. Give
each the same base adapter displacement norm: the exact first-update adapter
parameter displacement measured in E042, `0.09495018422603607`. Evaluate preregistered multipliers
`0.125`, `0.25`, `0.5`, and `1.0`, plus the untouched baseline.

First evaluate all twelve proposals on the exact 120 carried states for H32
with no noise, no replay, and the unchanged action boundary. A proposal is
carried-safe only if every row preserves E023's matched survival count and at
least one row improves. For each direction, choose at most one carried-safe
candidate by maximum improvement count, then unchanged count, then total
survival gain, then smaller multiplier.

Evaluate only those at-most-three candidates on ordinary LAFAN starts
`0/100/200/300/400`. Require componentwise preservation of the baseline
`116/63/49/39/47` vector. The line search is diagnostic and may not publish a
new retained policy.

## Outcome Map

- `stochastic-gradient-inconsistent`: H24-A versus H24-B aggregate cosine is
  below `0.5`, or at least three phase-bin cosines are below `0.25`, and no
  direction clears the full behavioral gate. Next test matched-noise gradient
  accumulation or larger independent-tape averaging.
- `current-h24-direction-useful`: an H24 proposal clears carried and ordinary
  componentwise gates. The local physics direction is behaviorally useful;
  next test an accept/reject constrained H24 optimizer rather than changing
  horizon, critic, or activation.
- `short-horizon-credit-misaligned`: an H48 proposal clears carried and
  ordinary componentwise gates while no H24 proposal does, or H24/H48
  aggregate cosine is nonpositive with H48 strictly better on the registered
  carried ordering. Next test one H24+H48 multihorizon training objective.
- `terminal-bootstrap-useful`: a bootstrap proposal clears both behavioral
  gates while H24 and H48 do not. Next test a phase-conditioned bootstrap with
  this exact critic boundary; do not enable it globally.
- `cross-state-objective-conflict`: H24 tape agreement is at least `0.5`, no
  proposal clears both gates, and at least two phase-bin H24/H48 cosines are
  negative or every direction improves some carried rows while regressing
  others. Next design a carried-state constrained or accept/reject optimizer.
- `aligned-local-step-insufficient`: all aggregate pairwise cosines among H24-A,
  H48-A, and bootstrap-A are at least `0.8`, H24 tape agreement is at least
  `0.5`, and no proposal clears both gates. Stop horizon/bootstrap tuning and
  target curvature or behavioral constraints.
- `direction-audit-inconclusive`: execution is valid but none of the preceding
  scientific predicates holds. Preserve every artifact and design one narrower
  discriminator from the measured geometry; do not tune this audit.
- `invalid-execution`: any provenance, exact-state, common-noise, gradient,
  clipping, CAGrad, proposal, replay-free evaluation, finiteness, or atomic
  artifact gate fails.

`invalid-execution` always takes precedence; the remaining labels are ordered
as written. If multiple scientific predicates remain true, publish the earliest
label and include all secondary predicates descriptively.

## Artifacts And Presentation

Publish under the registered run directory:

- `preflight.json`;
- `gradient_directions.npz`;
- `gradient_direction_summary.json`;
- `carried_line_search.json` plus per-candidate NPZ evidence;
- ordinary phase-grid summaries for the at-most-three carried-safe candidates;
- `selection.json` containing the preregistered outcome only;
- a cosine heatmap and a compact carried/ordinary survival plot for the
  Obsidian Dashboard.

The JSON manifest is written last and hash-binds every NPZ/plot and every
consumed input. No media or diagnostic proposal is canonical evidence unless
the manifest and independent evaluator agree.

## Tests

Test pure direction normalization, zero/non-finite rejection, common-prefix
noise identity, per-environment clip-before-bin aggregation, CAGrad direction
assembly, cosine behavior for zero and opposing vectors, fixed-batch identity,
bootstrap-only loss separation, equal-norm proposals, carried-safe ordering,
ordinary componentwise gates, ordered outcome classification, recursive
pytree hashing, atomic artifact publication, and tamper rejection. A compiled
one-environment H24/H48/bootstrap smoke is required before the guarded GPU run.

## Stop Rule

This experiment ends after one frozen capture and its bounded proposal
evaluations. Do not train, change activation, tune horizons, add noise tapes,
alter thresholds, or launch a successor from the same result. The next action
must follow the selected outcome map.
