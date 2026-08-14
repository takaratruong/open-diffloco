# Fresh PPO-Action-Contract Walking SHAC Design

## Scientific question

Can fresh native SHAC surpass the E019/E020 48-step walking plateau when its
action interface can express the physical commands used by the competent
full-scale PPO controller?

E020 shows that 384 additional unchanged updates do not improve phase-zero
survival beyond 48--49 transitions. Over the shared first 48 transitions, the
competent PPO uses hip-pitch reference offsets outside E019's half-scale tanh
envelope on 77.1% of left-hip and 81.2% of right-hip commands. This experiment
tests that diagnosed representational boundary directly.

## Alternatives considered

1. **Exact PPO-style action contract (selected):** full residual scale 1.0,
   linear/unbounded actor mean, and no sampled-action clipping. This matches the
   known competent controller's action semantics and tests the full authority
   hypothesis.
2. **Full-scale tanh mean:** safer numerically, but still cannot represent the
   roughly half of critical PPO hip commands with `|raw action| > 1`; a negative
   result would remain ambiguous.
3. **Half-scale unbounded mean:** has unbounded theoretical support, but changes
   the physical Jacobian and does not match the positive-control interface.

## Treatment

Start a new actor from the same exact zero output head as E019. Preserve the
E019 walking reference, 512-256-128 ELU LayerNorm actor, actor LR 1e-3, fixed
0.2 reparameterized action noise, clean observations, exact uniform
reference-state initialization, nominal physics, no pushes, no torso
assistance, no terrain, zero terminal bootstrap, H12 effective population 512,
five-bin CAGrad, and per-environment gradient clip 1.0.

Change the action contract as one coupled treatment:

- `reference_residual_scale=1.0`;
- `squash_actor_mean=False` (linear/unbounded mean);
- `clip_sampled_actor_actions=False`.

No PPO parameters, behavior-cloning parameters, critic state, optimizer state,
or normalizer state are loaded. The run is fresh SHAC policy learning.

## Execution and evidence

Train one seed-zero run for 128 effective updates / 786,432 transitions, with
checkpoints at 16, 32, 64, and 128 updates. The runner must fail closed on code,
model, controller, reference, solver, hparams, checkpoint, finite-state, and
CAGrad provenance.

For every checkpoint, record clean and fixed-0.2 training-distribution phase-0
rollouts with actor mean, sampled action, effective action, action RMS/max,
outside-`[-1,1]` fraction, and physical joint-target offsets. Evaluate the
replay-free five-phase grid at phases 0/25/50/75/100. Publish the selected
video, montage, and learning/action diagnostic plot to the Obsidian dashboard.

## Decision gate

- **Authority advances:** a checkpoint exceeds E020's phase-zero plateau by at
  least 12 transitions (survival >= 60), improves five-phase minimum and mean,
  and remains finite with valid clipped CAGrad. Continue this contract.
- **Authority solves:** a checkpoint completes all suffixes
  124/99/74/49/24. Treat fresh SHAC on the short walk as solved.
- **Authority insufficient:** the run remains below 60 phase-zero transitions
  or becomes unstable despite valid execution. Reject action support as the
  primary missing ingredient and return to objective/state-distribution work.
- **Invalid:** any treatment, provenance, telemetry, or evidence contract is
  violated. Repair execution only; do not interpret learning.

This is one experiment, not a scalar sweep. The result selects the next branch
regardless of outcome.
