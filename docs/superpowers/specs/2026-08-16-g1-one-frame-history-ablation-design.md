# G1 One-Frame History Ablation Design

## Decision

Run one bounded fresh short-walk SHAC experiment derived from E023. Change the
actor history length from ten frames to one frame and change no observation
field, network hidden width, action boundary, optimizer, objective, reset,
noise, plant, or solver setting. Stop after 32 actor updates so the first result
is available quickly. A separately registered continuation may spend the
remaining E023 budget only if this early discriminator is viable.

Root-position features are explicitly excluded from this experiment.

## Scientific Question

Does the successful E023 H24 recipe require a ten-frame actor history for early
learning, or can a one-frame 328-value current/reference-preview observation
match its early replay-free competence with a much smaller actor input?

This retires one narrow uncertainty. It does not claim that one frame solves
the long LAFAN motion.

## Matched Control

The control is evaluated E023 (`E-20260814-023`). Its fresh H24 actor used:

- ten 328-value frames, for a 3,280-value input;
- current reference joint position and velocity;
- root-orientation error, root-frame angular velocity, actual joint state, and
  previous action;
- reference joint position/velocity deltas at `+4`, `+8`, and `+12` frames;
- hidden widths `512/256/128`, LayerNorm, and a zero output head;
- 256 physical environments, two gradient-accumulation shards, H24, five-bin
  CAGrad, actor learning rate `1e-3`, zero terminal bootstrap, exact reference
  resets, nominal physics, and the RMR action-noise schedule.

E023's replay-free vectors at updates 16 and 32 were respectively
`42/36/48/49/24` and `45/50/53/49/24` from phases `0/25/50/75/100`.

## Treatment

Build the same fresh actor and environment with `actor_history_len=1`. The
actor input is therefore exactly one 328-value frame. Preserve the RMR noise
schedule endpoint at E023's original transition `1,572,864`; ending this
discriminator at transition `393,216` must not accelerate the noise anneal.

Training writes checkpoints at transitions `196,608` and `393,216`, exactly
updates 16 and 32. All scientific keyword differences from E023 must reduce to
`actor_history_len`; `total_steps` and the exact archive set are execution
budget metadata, not changed learning semantics before the stop.

## Evaluation

Evaluate both checkpoints through the same clipped/compiled, replay-free,
noise-free five-phase boundary used for E023. Persist:

- exact training preflight and hparams;
- both checkpoint archives and complete finite five-bin CAGrad telemetry;
- update-16 and update-32 five-phase summaries;
- a comparison JSON containing the two E023 control vectors and the two
  treatment vectors;
- one selected clean phase-zero trajectory, summary, MP4, and contact sheet;
- an early learning/action diagnostic plot;
- a manifest-last validation record binding every artifact.

The selected treatment checkpoint is the one with the lexicographically best
`(minimum, median, mean)` survival key over the first four nontrivial phase
starts; ties select the earlier update. Phase 100 is reported but excluded from
that key because only 24 transitions remain and completion fixes its ceiling.

## Outcomes

- `one-frame-early-advances`: update 16 or 32 preserves its matched E023 vector
  componentwise and improves at least one of phases 0/25/50/75.
- `one-frame-early-parity`: no checkpoint advances, but update 16 or 32 remains
  within two transitions of its matched E023 vector at every phase.
- `one-frame-early-mixed`: no checkpoint advances or reaches parity, and at
  least one treatment checkpoint improves one matched phase by more than two
  transitions while regressing another by more than two.
- `one-frame-early-underperforms`: valid execution satisfies none of the three
  preceding outcomes.
- `invalid-execution`: provenance, single-delta, shape, finite-gradient,
  CAGrad, checkpoint, paired evaluator, media, or artifact validation fails.

`advances` or `parity` authorizes one separately registered continuation to
update 128. `mixed` or `underperforms` stops the one-frame branch and restores
the ten-frame contract before testing root-position conditioning.

## Implementation Boundary

Add one experiment-specific runner and focused tests. Reuse the existing E023
builder, trainer, artifact validator, phase-grid evaluator, and media tools.
Do not alter the environment observation implementation, generic SHAC trainer,
existing runners, or E023 artifacts. Test-first coverage must prove the exact
one-frame shape, the single scientific delta, the unchanged noise endpoint,
the two-checkpoint budget, fail-closed provenance, and exhaustive outcome
classification before GPU launch.

## Stop Rule

This experiment stops after update 32 regardless of result. Do not tune
learning rate, history length, preview offsets, root features, reset noise,
horizon, or action noise inside this run. Curate the registered outcome before
authorizing any continuation.
