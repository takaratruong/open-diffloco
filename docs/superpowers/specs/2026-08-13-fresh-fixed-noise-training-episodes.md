# Fresh fixed-noise training episode design

## Goal

Retrain G1 tracking from a fresh zero-output actor and preserve visual samples
from the policy distribution at regular training checkpoints.  The experiment
is diagnostic: it must reveal whether useful reference corrections emerge or
whether optimization drives the actor into clipped/saturated commands.

## Training contract

- Start from a fresh `(512, 256, 128)` ELU actor with LayerNorm and an exact
  zero output head.  At initialization the commanded target is the reference
  pose.
- Use reference-residual control with the established bounded correction scale
  `0.5` and the trainer's normalized action clamp immediately before `env.step`.
- Use fixed scalar reparameterized Gaussian action noise `std=0.2` for every
  update.
- Disable actor observation noise, reference-state perturbation, root
  perturbation, carried resets, domain randomization, pushes, terrain forces,
  and torso assistance.
- Keep the established 50 Hz source-step dynamics, g1-4x5 solver, exact phase
  resets, H12 rollouts, effective batch 512, five-bin CAGrad, delta future
  reference preview, reward, and termination.
- Train 128 updates / 786,432 transitions.  Archive every 16 updates so the
  learning trajectory has eight checkpoints.

## Episode log

For every checkpoint, replay a seeded phase-zero episode through the exact
training action boundary: actor mean, the fixed `0.2` epsilon perturbation,
normalized clamp, reference-residual PD target, and nominal physics.  Save the
rollout video, H12 slice, action/noise tape, contact sheet, and summary.  Also
save a clean rollout from the same checkpoint for attribution.  The episode
log is post-checkpoint replay of the exact configured distribution, not a claim
that it is one of the population trajectories consumed by the optimizer.

The manifest must expose mean-action RMS, injected-noise RMS, effective-action
RMS, and clamp/saturation fraction.  A progression video presents the noisy
episodes in checkpoint order and is the experiment's dashboard preview.

## Decision

The run is useful even if it fails to track.  We specifically distinguish:

1. tracking improves while saturation remains low;
2. tracking improves only by saturating the actor;
3. the actor quickly saturates and tracking collapses; or
4. the actor remains near zero and does not learn.

Only the first case justifies a longer continuation.  Training reward alone
does not select a policy.
