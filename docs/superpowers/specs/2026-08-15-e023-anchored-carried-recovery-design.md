# E023 Anchored Carried-Recovery Design

## Decision

Continue from E023's selected update-128 actor, but make the parent actor
structurally immutable and train only a new zero-output residual adapter. Train
that adapter for at most 64 H24 SHAC updates on a fixed mixture of 75 percent
E023's original exact reference resets and 25 percent history-faithful states
collected 6 through 29 transitions before E023's observed failures. Apply the
existing action-space proximal penalty with weight 1.0 between the composite
action and the frozen E023 parent action on every sampled state.

This is one preregistered treatment, not a reset-probability, penalty, learning-
rate, or horizon sweep. E023 remains the retained policy unless a treatment
checkpoint preserves its complete replay-free phase vector componentwise.

## Why This Experiment

E023's selected exact-state H24 policy reaches `116/99/67/49/24`. E024 shows
that 128 additional unchanged updates redistribute competence instead of
solving the walk. E025 shows that fresh full-envelope noisy reference-state
initialization improves some early checkpoints but ends at `63/99/62/49/24`,
well below E023 at the two incomplete suffixes. The remaining mismatch is more
specific than generic reset noise: training rarely begins in the actual
closed-loop states immediately before E023's failed stance transitions.

The older E010 carried-state result does not settle this experiment. E010 used
a long dance, H12, domain randomization, noisy-reference fallback resets, a
fixed 50/50 carried mixture, and continued an already-trained residual without
an action anchor. This treatment uses the short walk, H24, nominal physics,
E023's exact fallback resets, a conservative 25-percent carried cohort, a new
zero-effect adapter, a frozen parent, and an explicit proximal preservation
term.

## Alternatives Considered

### 1. Resume the full E023 actor on carried resets

This is the smallest code change, but E024 already demonstrates destructive
competence tradeoffs when all actor parameters remain trainable. It lacks a
structural preservation boundary and is rejected.

### 2. Frozen parent plus anchored recovery adapter — selected

At initialization the composite policy is exactly E023. Only the adapter and
its optimizer moments may change. The carried cohort supplies gradients near
the observed failures, while the exact cohort and proximal penalty constrain
the correction away from those failures.

### 3. Increase the differentiable horizon again

H24 materially improved the walk, but the observed failures are 67 to 116
closed-loop transitions from their starts. Another horizon increase is more
expensive and less targeted than putting those reached states inside the H24
training distribution. It is deferred.

## Immutable Source And Carried Bank

The source is E023's selected `checkpoint_step_1572864.pkl`, its sibling
`hparams.json`, the pinned walk reference, G1 model/controller assets, and the
`g1-4x5` solver profile. The collector runs the exact clean E023 actor without
exploration noise from source phases 0 and 50. Each source rollout must end in
an ordinary terminal transition before the reference suffix ends. From each,
retain the 24 nonterminal pre-step states whose terminal transition is 6
through 29 transitions away.

Every bank row stores exact `qpos`, `qvel`, phase, previous prepared action,
complete `(10, actor_frame_obs_dim)` observation history, selected clean
action, source phase and source step, transitions to terminal, and termination
errors. The collector rejects non-finite rows, hard-limit violations,
unnormalized root quaternions, phase discontinuities, or mismatched last
history frames. The atomic bank artifact and summary bind checkpoint,
reference, model, controller, code, solver, and bank SHA-256.

## Training Contract

The dedicated runner explicitly authorizes one migration from E023's plain
Flax actor to `FrozenPreviewResidualParams(parent=E023, adapter=zeros)`. The
migration must prove before training that:

- parent actions and composite actions are equal on the live resumed batch;
- all parent parameters, parent optimizer state, and actor normalizer match
  E023 exactly;
- the adapter output and adapter optimizer moments are exactly zero; and
- the complete resumed environment, critic, target critic, RNG, and global
  step are otherwise unchanged.

Training retains E023's H24 horizon, effective population 512, five phase-bin
CAGrad, per-environment gradient clipping, zero terminal bootstrap, RMR
endpoint action-noise vector, actor and critic learning rates, observations,
reward, termination, plant, solver, and seed. Reference reset noise,
observation noise, domain randomization, pushes, torso assistance, and adaptive
phase sampling remain disabled. On reset, a fixed Bernoulli probability 0.25
selects one carried row; otherwise the environment performs E023's unchanged
uniform exact reference reset.

The action anchor is the mean squared difference between the composite clean
action and stop-gradient E023 parent action, weighted by 1.0 and added to the
actor objective. It is evaluated on all sampled states. This does not guarantee
behavioral preservation, so replay-free selection remains the authority.

## Execution And Stop Rule

Run at most 64 additional updates and archive updates 8, 16, 32, and 64. The
first completed update is an operational gate: parent and normalizer drift must
be zero, adapter gradient and update norms must be finite and positive, all
five CAGrad bins must be occupied and valid, and the carried-reset audit must
restore a bank row's complete physical and actor context exactly. Any failure
stops execution as invalid.

Evaluate every archived checkpoint without noise from phases
`0/25/50/75/100`. A checkpoint is eligible only if its survival vector is at
least E023's `116/99/67/49/24` componentwise. Select eligible checkpoints
lexicographically by minimum, median, mean, then earliest update. Exact
`124/99/74/49/24` completes the short walk. If update 32 has no eligible
checkpoint and neither phase 0 nor phase 50 exceeds E023, the runner may stop
without spending the final 32 updates; otherwise it completes the fixed budget.

## Outcomes

- `anchored-carried-solves-walk`: an eligible checkpoint completes
  `124/99/74/49/24`.
- `anchored-carried-advances`: an eligible checkpoint preserves E023 and
  improves phase 0 or 50 without compensation elsewhere.
- `anchored-carried-insufficient`: execution is valid but no checkpoint
  preserves E023 componentwise.
- `invalid-execution`: provenance, migration, context restoration, frozen-state
  identity, finite-gradient, CAGrad, checkpoint, evaluator, or artifact gates
  fail.

No outcome authorizes tuning the carried probability or anchor weight from the
same evidence. A valid negative result retires this carried-reset recipe and
returns the next decision to objective/credit assignment rather than another
reset-distribution sweep.

## Evidence And Dashboard

Publish the bank summary, checkpoint-aligned training/action diagnostics,
five-phase vectors, selection record, selected clean rollout, selected
training-distribution rollout, contact sheet, and learning curves through the
experiment registry and generated Obsidian dashboard. Media are presentation
copies; canonical evidence remains under the registered run directory.

## Tests

Tests must cover the explicit plain-actor-to-frozen-residual migration authority,
zero-effect migration, exact frozen parent/optimizer/normalizer identity,
unauthorized resume rejection, scalar anchor persistence, E023 bank collection
and schema validation, probability-zero legacy equivalence, probability-one
complete-context restoration, runner single-delta validation, complete finite
checkpoint telemetry, and componentwise replay-free selection. A compiled
one-update smoke is the launch validation for the enabled carried-reset and
adapter-gradient path.
