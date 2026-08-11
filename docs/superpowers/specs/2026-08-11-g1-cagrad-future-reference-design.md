# G1 CAGrad Future-Reference Continuation Design

## Objective

Test whether explicit anticipatory reference information converts E008's
five-phase CAGrad near-miss into a robust G1 tracking improvement. The
treatment must preserve E008's learned behavior before the first update and
change only the actor's reference representation during the continuation.

E008 is the immutable parent. Its selected final checkpoint survives
`28/55/77/21/45` frames from phases `0/100/200/300/400`, raises every phase
relative to E006, and misses the median-survival gate by three frames. E009
shows that doubling the differentiable horizon to 24 is finite but regresses
all five phases. Return to horizon 12 and retain E008's five-bin CAGrad.

This experiment does not assume that the original actor observation is
insufficient: PPO can solve related tracking tasks from the current reference.
It asks the narrower empirical question of whether trajectory preview makes
the differentiable optimization problem easier.

## Alternatives Considered

### Recommended: multiscale preview with behavior-preserving E008 migration

Append reference joint positions and velocities at 4, 8, and 12 future control
steps. These points expose the shape of the upcoming 0.08--0.24 second
trajectory through contact transitions. Expand E008's actor and optimizer
state so the new inputs initially have zero influence. This preserves the best
policy and provides the cleanest causal comparison.

### One endpoint at 12 future control steps

This adds only 58 inputs, but an endpoint can alias different intervening
trajectories. It is cheaper yet less informative near a stance swap.

### Fresh training with the expanded observation

Fresh training avoids checkpoint migration, but discards E008's established
CAGrad improvement and pays again to reach the same failure boundary. It also
confounds representation value with a new optimization trajectory. A passing
continuation may justify a separately registered fresh-seed confirmation.

## Observation Contract

Add a default-off `actor_reference_lookahead_steps` environment option. It is
an immutable, strictly increasing tuple of positive control-step offsets. The
registered treatment uses `(4, 8, 12)`.

For each offset `k`, compute the reference index as:

```text
min(current_phase + k * reference_stride, reference_length - 1)
```

At that index, gather the 29 reference joint positions and 29 reference joint
velocities in actor joint order. Append each 58-value command, in offset order,
after the legacy 154-value actor frame. Appending preserves every legacy input
index. The treatment frame therefore has `154 + 3 * 58 = 328` values. The
critic observation remains unchanged at 286 values.

Future commands receive no observation noise. Existing current-state noise is
unchanged. Generalize the existing 154-value training-noise boundary to accept
the append-only future-command suffix while retaining zeros in the suffix's
noise mask. Exact phase resets, terminal bootstrap observations, history
updates, and evaluation use the same observation constructor, so no code path
can silently omit the preview. End-of-reference lookahead clamps to the final
frame and never wraps or teleports.

## Behavior-Preserving Checkpoint Migration

Resume the exact E008 final TrainState at step `1,179,648`, checkpoint SHA-256
`da661e4bec6fbbd578face31ea2129e00c01d985cc6308e20473db4d9055fe3c`.
The migration is allowed only when the saved actor uses the legacy 154-value
frame, history length is one, and the requested suffix is exactly the declared
future-reference layout. Any other observation mismatch is an error.

Perform one explicit append-only migration:

- pad the actor's first-layer input kernel with zero rows for all 174 new
  inputs, leaving every existing parameter byte unchanged;
- pad the corresponding Adam first- and second-moment leaves with zero rows;
- retain the normalizer's legacy mean and variance exactly, and initialize the
  new dimensions from deterministic mean and variance over the complete
  clamped reference-preview table, using the saved scalar count;
- rebuild the saved environment actor observations, bootstrap observations,
  and history from their unchanged physical data and phase information;
- leave critic parameters, target critic, critic optimizer, critic normalizer,
  PRNG state, simulator state, and every non-observation training leaf
  unchanged.

Before training, write a migration-equivalence artifact. On a fixed batch
reconstructed from the saved environment state, the legacy 154 inputs and
normalization must be exact, all new actor-kernel and optimizer rows must be
zero, all new normalizer values must be finite, and migrated actor outputs must
match E008 within `1e-7` absolute and relative tolerance. Failure is invalid
execution, not a fresh initialization fallback.

## Training And Experiment Contract

Continue for exactly 64 actor updates at physical `num_envs=256`, accumulation
factor two, effective batch 512, and differentiable horizon 12. This adds
exactly `64 * 512 * 12 = 393,216` transitions and ends at step `1,572,864`.
Archive checkpoints after 32 and 64 updates, at steps `1,376,256` and
`1,572,864`.

Retain E008's actor CAGrad with five phase bins, `alpha=0.5`, and 32 fixed
Frank-Wolfe iterations. Preserve its actor, critic, learning rates, optimizer
state, action-noise schedule, terminal critic, reset distribution, domain
randomization, rewards, terminations, controller, fixed 4x5 solver, model,
reference, and seed. The future-reference suffix and its required migration are
the sole scientific treatment.

Add one immutable runner derived from the E008 continuation contract. It may
accept only solver profile, reference path, seed, output root, and resume path.
It must not expose lookahead offsets or other scientific settings as command
line overrides. The runner's contract test must show that its only scientific
difference from E008 is the declared future-reference representation.

## Code Boundaries

- Keep preview construction inside the G1 environment through one pure helper
  used by actor observation assembly.
- Keep append-only TrainState migration in a focused SHAC checkpoint-migration
  module rather than embedding shape surgery throughout the training loop.
- Integrate migration at the existing resume boundary before JIT compilation.
- Record the lookahead tuple and migration provenance in `hparams.json` so
  treatment checkpoints are self-describing and exact treatment resume works.
- Do not change the critic observation, create a second trainer, or generalize
  this into an arbitrary checkpoint-conversion framework.

All new behavior remains default-off. Existing 154-value policies and runners
must continue to reproduce their current contracts.

## Testing And Verification

Use test-driven development at these boundaries:

1. Preview indices apply `reference_stride`, preserve offset order, and clamp
   rather than wrap at the final reference frame.
2. Preview joint positions and velocities use actor joint order and are
   appended after the exact legacy 154 values.
3. Noise affects precisely the legacy noisy fields and never the future
   command suffix.
4. Reset, step, bootstrap, termination-reset, and history paths all produce
   the declared 328-value treatment observation.
5. Migration preserves all legacy actor parameters, normalizer entries, and
   non-actor TrainState leaves; only declared first-layer and optimizer leaves
   are zero-padded.
6. Migrated actor outputs match the original E008 actor before any update.
7. Unsupported observation mismatches fail closed, while default-off exact
   resume remains unchanged.
8. The immutable runner differs from the E008 H12 CAGrad contract only by the
   preview setting, endpoint, and mechanically required checkpoint schedule.

Run focused environment, noise, migration, resume, CAGrad, runner, and phase
grid tests; Ruff; Python compilation; the canonical SHAC smoke; registry
validation; and a guarded experiment dry run before GPU execution.

## Evaluation And Decision Gate

Evaluate both fixed checkpoints from exact phases `0`, `100`, `200`, `300`,
and `400` under nominal replay-free physics. Select lexicographically by
maximum minimum survival, then median, then mean, then earliest checkpoint.

The treatment advances only when:

- migration equivalence, training completion, telemetry, artifacts, and every
  numeric final TrainState leaf are valid and finite;
- the selected minimum survival is at least 19 and median at least 48; and
- per-phase survival is at least `23/50/72/16/40`, E008 minus five frames at
  each registered start.

If it advances, retain multiscale future reference plus CAGrad and register a
fresh-seed confirmation. If it is finite but misses a behavioral gate, stop
this preview treatment and return to gradient-estimator or optimizer changes;
do not tune offsets on the same evidence. Invalid execution authorizes only a
repair of the demonstrated implementation or provenance fault.

A pass is an early closed-loop advancement gate. It does not establish full
suffix completion, seed robustness, sim-to-sim transfer, or hardware readiness.
