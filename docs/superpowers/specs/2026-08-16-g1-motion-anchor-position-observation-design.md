# G1 Motion-Anchor Position Observation Design

## Decision

Run one bounded fresh short-walk SHAC experiment derived from E023. Preserve
the successful ten-frame E023 actor, H24 optimizer recipe, reset distribution,
action noise, reward, plant, and solver. Add only RMR's three-value
`motion_anchor_pos_b` signal to each actor observation frame.

This is a single observability intervention, not another hyperparameter sweep.

## Scientific Question

Does E023 fail to continue the walking sequence because the actor cannot
directly observe the translational error between the current root and the
reference root, even though it observes orientation error, joint tracking
error, angular velocity, previous action, and future joint targets?

## Matched Control

The matched control is evaluated E023 (`E-20260814-023`). Its actor receives
ten 328-value frames, for 3,280 input values. The replay-free phase-grid
survival vectors at updates 16 and 32 are:

- update 16: `42/36/48/49/24`;
- update 32: `45/50/53/49/24`.

The phase starts are `0/25/50/75/100`. Phase 100 has only 24 transitions left
and is retained as a completion check but excluded from checkpoint selection.

## Treatment Signal

For each actor frame, compute

```
motion_anchor_pos_b = R(q_actual)^T * (p_reference - p_actual)
```

where position and orientation belong to the existing G1 motion anchor. This
is the position component already returned by `_anchor_relative_reference`
and matches the RMR observation named `motion_anchor_pos_b`.

Insert these three values immediately before the existing six-value anchor
orientation error. The treatment frame is therefore 331 values and the
ten-frame actor input is exactly 3,310 values.

Do not add absolute world position, root linear velocity, observation noise,
root perturbations, pushes, or any other feature. The new position values use
zero observation-noise amplitude so the feature addition remains the only
causal change even if actor observation noise is enabled in a future run.

## Compatibility and Resume Contract

Expose one explicit boolean setting,
`actor_observe_motion_anchor_position`, defaulting to false. When false, the
legacy observation values, ordering, dimensions, and scalar training path must
remain unchanged. Persist the flag in training hparams and evaluator evidence.

On resume, the persisted value is authoritative. A checkpoint lacking this
field is a legacy false checkpoint. Resuming with a requested value that
differs from the saved value must fail closed; this experiment is fresh and
does not use a migration.

## Training Budget

Use the complete E023 recipe except for the observation flag and bounded
budget:

- 256 physical environments with two gradient-accumulation shards;
- horizon 24, effective population 512;
- five-bin CAGrad and per-environment gradient clipping;
- actor hidden widths `512/256/128`, LayerNorm, zero output head;
- exact reference-state resets, nominal physics, no observation noise;
- RMR action noise with E023's unchanged schedule endpoint at transition
  `1,572,864`;
- 32 updates / `393,216` transitions;
- archives at updates 16 and 32 (`196,608` and `393,216` transitions).

Changing `total_steps`, checkpoint locations, and the expected input-width
guard are execution metadata. The only changed learning semantic is the new
observation flag.

## Evaluation and Selection

Evaluate both checkpoints through the same deterministic, clipped, compiled,
replay-free five-phase evaluator used for E023. Select the checkpoint with the
lexicographically best `(minimum, median, mean)` survival over phases
`0/25/50/75`; ties choose the earlier update.

Persist exact provenance, training hparams, checkpoint hashes, complete finite
CAGrad telemetry, both phase grids, the paired comparison, a clean selected
trajectory, summary, MP4, contact sheet, learning/action diagnostics, and a
manifest written only after all dependencies validate.

## Outcomes

- `root-position-early-advances`: update 16 or 32 preserves every component of
  its matched E023 vector and strictly improves at least one of phases
  `0/25/50/75`.
- `root-position-early-parity`: no checkpoint advances, but one treatment
  checkpoint is within two transitions of its matched control at all phases.
- `root-position-early-mixed`: no checkpoint advances or reaches parity, and
  some checkpoint improves a matched phase by more than two while regressing
  another by more than two.
- `root-position-early-underperforms`: valid execution satisfies none of the
  preceding outcomes.
- `invalid-execution`: any provenance, single-delta, observation-shape,
  finite-gradient, CAGrad, checkpoint, evaluator, media, or artifact gate
  fails.

An `advances` result is evidence that translational observability was a missing
ingredient. `parity` means the signal is harmless but not explanatory at this
budget. `mixed` or `underperforms` rejects this intervention rather than
authorizing tuning within the same experiment.

## Test and Implementation Boundary

Implement the optional observation at the existing environment seam, thread
the flag through SHAC hparams and generic evaluators, and add one bounded
experiment runner. Test first:

- legacy false mode is value- and shape-identical;
- true mode contains the exact body-frame anchor-position error at the stated
  location and has 331-value frames / 3,310-value histories;
- the noise mask and normalizer widths match both modes;
- resume provenance fails closed;
- the runner has exactly the intended scientific delta and budget;
- selection and every outcome class are covered;
- evaluator evidence records and enforces the observation contract.

## Stop Rule

Stop after update 32. Do not tune learning rate, history, preview offsets,
reset noise, horizon, action noise, root velocity, or reward inside this run.
Curate the registered result before selecting the next intervention.
