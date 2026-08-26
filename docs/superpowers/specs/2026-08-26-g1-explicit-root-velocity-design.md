# Explicit G1 Root-Velocity Reward Design

## Goal

Test whether an explicit, non-saturating pelvis-velocity objective supplies the
missing balance credit that the existing fourteen-body averaged velocity reward
dilutes during the retained E026 short walk.

## Evidence boundary

E026 remains the retained controller at `131/114/74/71/74` on the continuous,
prefix-exact five-phase evaluation. The default reward contains pelvis velocity
only inside a fourteen-body mean before an exponential kernel. E017 established
that replacing both complete body-velocity kernels with pseudo-Huber changes
useful gradient directions but redistributes competence. This experiment does
not replace those body terms. It adds one pelvis-specific signal while freezing
the complete E026 controller and normalizer.

## Single causal change

Add one optional root-velocity reward with one scalar weight:

`0.5 * (2 - sqrt(1 + 2 * ||v_root-v_ref||^2 / 1.0^2))`

`+ 0.5 * (2 - sqrt(1 + 2 * ||w_root-w_ref||^2 / pi^2))`

The environment multiplies the combined term by
`tracking_root_velocity_weight`; the experiment fixes that weight to `1.0`.
The existing global scale `0.02` remains unchanged, so the new term contributes
at most `0.02` per step. The legacy default is exactly zero. A resumed nonzero
weight requires explicit authority and complete sibling hparams.

Root is rigid-body slot zero (`pelvis`). Linear and angular velocities use the
same stored world-frame reference and simulated body-state convention already
used by the fourteen-body velocity terms. No acceleration, contact, foot,
phase, reset, assistance, replay, or physics treatment changes.

## Training and selection

Resume exact E026 on the prefix-exact continuous walk. Freeze its complete
actor, moments, and normalizer, and train only one new zero-effect 328-256-29
ELU residual for 32 effective-512 H24 updates. Preserve the existing
five-phase clipped CAGrad, action-noise schedule, zero bootstrap, solver, seed,
and action boundary. Archive updates 8, 16, 24, and 32.

Evaluate the source and all candidates deterministically on CPU at phases
`0/25/50/75/100`. A policy can be retained only if it componentwise preserves
E026 `131/114/74/71/74` and strictly improves at least one phase. Eligible
candidates rank by minimum, median, mean survival, then earlier update.
Training reward never selects a policy.

## Outcomes

- `root-velocity-advances`: an eligible checkpoint is retained.
- `root-velocity-redistributes`: a checkpoint improves aggregate or one phase
  but violates at least one E026 componentwise floor.
- `root-velocity-insufficient`: valid training produces no survival gain.
- `invalid-execution`: provenance, legacy identity, resume authority, finite
  telemetry, frozen state, checkpoint, evaluation, or artifact evidence fails.

## Interpretation limit

Success establishes only that explicit pelvis velocity improves the
prefix-exact short-walk controller. It does not establish long-motion or
sim-to-real robustness and does not justify an acceleration reward.
