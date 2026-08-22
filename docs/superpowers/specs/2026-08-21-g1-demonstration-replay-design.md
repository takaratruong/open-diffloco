# G1 DiffMimic Demonstration Replay Design

## Decision

Run one bounded fresh G1 SHAC discriminator derived from E023. Preserve its
plant, observation, action, network, optimizer, CAGrad, H24 rollout, action
noise, exact RSI, solver, and seed. Add only DiffMimic-style intra-rollout
demonstration replay when mean aligned body-position error exceeds `0.20 m`.

This is not another reset-distribution experiment. Replay is evaluated after a
real transition and can restore the current reference frame before the next
policy action. All policy evaluation is replay-free.

## Replay Contract

The trigger uses the already-persisted G1 `body_position_error`: mean Euclidean
link-position error in metres. This matches DiffMimic's reduction and fixed
walking threshold while retaining the G1 task's yaw/root alignment convention.

For a finite, nonterminal transition with error above `0.20`:

- restore exact reference `qpos` and `qvel` at the returned phase;
- use the environment's currently carried physical randomization;
- preserve RNG, difficulty, phase, and elapsed episode step;
- set prior action to zero and rebuild all ten actor-history frames from the
  restored state;
- rebuild actor and critic bootstrap observations;
- stop gradients across the discrete teleport boundary; and
- report the replay event separately from ordinary termination/reset.

Terminal or clip-end transitions retain the environment's normal reset and are
never overwritten by demonstration replay.

## Bounded Experiment

Use E023's update-16 and update-32 budget: checkpoints at transitions 196,608
and 393,216. The action-noise schedule retains its original 1,572,864-transition
endpoint. Compare replay-free five-phase survival at phases 0/25/50/75/100 to
E023's matched vectors:

- update 16: `42/36/48/49/24`;
- update 32: `45/50/53/49/24`.

Persist replay count/fraction at each checkpoint. A valid execution must have a
strictly positive replay fraction at most 1.0, finite gradients, complete valid
five-bin CAGrad, and exact checkpoint provenance. A fraction at least 0.95 is a
valid but scientifically overassisted outcome that stops the branch.

## Outcomes

- `demo-replay-early-advances`: a matched checkpoint preserves E023 at every
  phase and improves at least one nontrivial phase.
- `demo-replay-early-parity`: no advancement, but a matched checkpoint remains
  within two transitions at every phase.
- `demo-replay-overassisted`: replay fraction is at least 0.95 at either
  checkpoint; no threshold tuning is authorized.
- `demo-replay-early-mixed`: at least one phase improves by more than two and
  another regresses by more than two.
- `demo-replay-early-underperforms`: valid execution reaches none of the above.
- `invalid-execution`: provenance, single-delta, history restoration, finite
  gradient, telemetry, checkpoint, replay-free evaluation, or artifact gates
  fail.

Only advancement authorizes a separately registered longer continuation. This
run does not tune replay threshold, learning rate, horizon, reset noise, reward,
history length, or action noise.
