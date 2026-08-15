# Progressive State-Gated Recovery Expert

## Objective

Test whether the retained E023 LAFAN policy can learn a corrective maneuver at
its earliest failure without paying for that improvement by changing behavior
throughout the motion.

The experiment changes the representation of the correction, not another
scalar in the existing SHAC recipe. E023 remains the immutable parent. A new
zero-output residual expert is allowed to act only inside a fixed support
around the pre-failure state corridor. Outside that support, its contribution
is exactly zero.

## Evidence motivating the design

- E023's current-code LAFAN control survives `116/63/49/39/47` transitions
  from phases `0/100/200/300/400`.
- E027's globally active residual falls to `86/63/49/37/45` after only eight
  updates, then reaches only `106/65/50/42/46` after 64 updates.
- E028 continues the same treatment to update 128. Its best vector is
  `91/66/51/42/47`, so more optimization improves later suffixes while losing
  25 phase-zero transitions.
- E015 and E016 show that smaller interpolation or optimizer steps do not
  remove the cross-phase tradeoff. Even one eighth of a proposal regresses a
  protected suffix.
- The existing action anchor is evaluated on sampled training observations,
  while CAGrad combines local H24 losses. Neither represents replay-free
  suffix survival.

These results reject update count, scalar step size, horizon, noise width, and
ordinary shared-parameter phase balancing as the next standalone axis.

## Architecture

The deployed action is

`a = frozen_parent(obs) + gate(obs, phase) * expert(obs)`.

The parent is the exact E023 actor, normalizer, and action contract. The expert
is the existing one-hidden-layer `328 -> 256 -> 29` ELU residual adapter with a
zero-initialized output head. Only the expert parameters and their optimizer
state are trainable.

The gate is fixed before training. It is not a learned policy component and
receives no gradient. It combines:

1. a compact absolute-reference-phase window covering the 24 recorded E023
   phase-zero states that occur 6--29 transitions before terminal; and
2. compact support in the normalized current-frame observation around those
   carried states.

Let `p_min` and `p_max` be the minimum and maximum carried-anchor phases. The
phase gate is one on `[p_min, p_max]`, tapers with a cubic smoothstep over four
reference frames on each side, and is exactly zero outside
`[p_min - 4, p_max + 4]`. Training and evaluation pass the pre-reset
`state.info["phase"]` directly to the composite policy application; phase is
not appended to the parent observation or used to modify E023's normalizer.

The observation support uses frozen E023 normalization and Euclidean
nearest-anchor distance over the complete 328-value current frame. Let
`d_neg` be the minimum distance from any protected negative to the 24 positive
anchors. The registered radius is exactly `0.5 * d_neg`. A smooth
compact-support kernel is positive for distance below that radius and exactly
zero at or above it. Preflight additionally requires at least 20 of 24
positive anchors to have a leave-one-out positive-neighbor distance below the
radius. This fixes a non-post-hoc safety margin while rejecting an isolated
anchor set that would not provide a usable neighborhood.

Protected negatives are E023 states before the marked phase-zero failure
corridor and states from the four other exact-start rollouts wherever their
reference phase overlaps the expert's phase window. The gate preflight must
show:

- every protected negative has gate value exactly zero;
- at least 20 of the 24 carried anchors have positive gate value;
- gate values, distances, radius, anchors, and protected-state hashes are
  finite and persisted; and
- zero-expert and gate-zero actions are bit-identical to E023.

If no radius satisfies these conditions, the experiment stops as
`support-not-separable`; training must not start. The next branch is direct
action-sequence recovery, not gate-threshold tuning.

## Training protocol

Create a phase-zero carried bank from the already validated 24 E023 states in
the immutable E027 bank. Preserve their exact qpos, qvel, phase, last action,
and ten-frame actor history.

Train the expert from its zero head with:

- the exact E023 parent and optimizer/normalizer snapshots frozen;
- the pinned 500-state LAFAN reference;
- the `g1-4x5` fixed solver;
- H24 differentiable rollouts;
- effective population 512;
- 25-percent targeted carried resets and 75-percent exact reference resets;
- the existing per-environment gradient clip of 1.0;
- ordinary mean aggregation of the clipped expert gradients, with CAGrad
  disabled because this experiment has one trainable task-specific module;
- Adam at actor learning rate `1e-3`, matching E027's adapter optimizer;
- zero terminal bootstrap and zero assistance;
- E023's fixed RMR-calibrated action-noise endpoint;
- exact carried resets from the targeted bank plus unmodified exact reference
  resets; and
- no observation noise, random reset noise, pushes, friction randomization,
  reward changes, optimizer sweep, or horizon sweep.

The budget is exactly 32 updates with immutable checkpoints at updates 8, 16,
and 32. This is a discriminator, not an open-ended training run. A checkpoint
that does not clear the behavioral gate cannot be continued or retuned.

Training telemetry must include support activation fraction, expert output
RMS/max, action-boundary statistics, per-environment gradient norms, expert
update norms, carried-versus-reference loss, and exact frozen-state drift.

## Evaluation and selection

Evaluate the E023 control and every candidate checkpoint through the same
compiled, replay-free phase grid at phases `0/100/200/300/400`. Run two paired
replicates to bound the already observed one-to-two-transition contact-boundary
variation. Persist trajectories, metrics, videos, montages, action diagnostics,
and the complete gate-activation trace.

A candidate is eligible only when:

- all provenance, gate, frozen-state, finite-gradient, artifact, and evaluator
  contracts pass;
- phase-zero survival exceeds its paired E023 control by at least three
  transitions in both replicates;
- no other phase loses more than two transitions relative to its paired E023
  control in either replicate; and
- the expert is exactly inactive outside its registered support.

Rank eligible candidates by worst paired phase-zero improvement, then minimum,
median, and mean five-phase survival, then earlier update.

## Outcome map

### `gated-recovery-advances`

At least one eligible checkpoint improves phase zero without a compensating
loss. Freeze the accepted expert, compose it with E023, roll out the composed
policy, and register the next expert around the new earliest failure.

### `gated-recovery-solves`

An eligible checkpoint completes all five suffixes terminal-free. Retain it
and proceed to perturbation and sim-to-sim robustness gates.

### `gated-recovery-insufficient`

The gate is valid and training is finite, but no checkpoint clears the paired
behavioral gate. Retain E023 and run a bounded direct action-sequence recovery
oracle from the same carried states. If the oracle succeeds, distill its
correction into the expert; if it fails, change the differentiable objective
or horizon formulation rather than policy architecture.

### `support-not-separable`

No preregistered compact support activates on the failure corridor while
remaining exactly zero on protected trajectories. Do not tune the radius.
Proceed directly to the action-sequence recovery oracle, then reconsider the
state features used to identify recoverable drift.

### `invalid-execution`

Any provenance, migration, gate, frozen-parent, optimizer, gradient,
checkpoint, replay-free evaluation, or artifact contract fails. Repair only
the execution defect before interpretation.

## Scope and stopping rules

This experiment tests one recovery expert at the first LAFAN failure. It does
not claim general motion tracking, automatic expert discovery, sim-to-real
robustness, or that an unbounded collection of experts is acceptable. Stop
after exactly 32 updates. Do not add another expert until this experiment
establishes that structural isolation can make one componentwise-safe advance.

If successful, cap the initial progressive sequence at four experts. The
composed policy remains the scientific artifact; distillation into a compact
actor is optional follow-up work and must reproduce the complete replay-free
vector before replacing it.
