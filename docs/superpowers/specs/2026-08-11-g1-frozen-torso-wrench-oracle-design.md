# Frozen E008 Torso-Wrench Oracle Design

**Date:** 2026-08-11
**Status:** Approved diagnostic design

## Question

Can bounded external root authority rescue every strict suffix of the frozen E008
humanoid tracker? This is a physical-sufficiency discriminator, not a new policy
training method. It runs independently from the E010 carried-reset continuation
and cannot affect E010's code, checkpoint, reset distribution, or selection.

## Alternatives

1. **Analytic torso PD oracle (selected).** Freeze E008 and add a deterministic
   bounded six-dimensional wrench during evaluation. This is the smallest test
   of whether missing whole-body corrective authority explains the failures.
2. **Learned assistance head.** Jointly train a privileged wrench controller and
   anneal it away. This is useful only after the analytic oracle establishes
   sufficiency; otherwise it adds training and leakage failure modes without
   answering the physical question.
3. **Phase-indexed wrench tape.** Optimize a wrench per reference frame. This is
   cheap but can memorize the exact clip and has the weakest scientific meaning,
   so it is rejected for the first discriminator.

## Fixed Controller

Use the immutable E008 selected checkpoint, reference, solver profile, policy
normalizer, actor architecture, seed, and phase grid `0/100/200/300/400`. The
deployed actor still outputs exactly 29 joint actions and receives no privileged
input.

At each policy step, compute the current and aligned-reference pose and velocity
of `torso_link` (reference body slot 7; pinned MuJoCo body ID 16). Position,
linear-velocity, orientation, and angular-velocity errors are expressed in the
current torso-yaw frame. Orientation error is the shortest quaternion rotation
vector. Use a critically damped 2 Hz PD oracle:

- translational effective mass: total nominal robot mass;
- rotational effective inertia: `mass * (0.3 m)^2`;
- `Kp = effective_mass * omega^2`, `Kd = 2 * effective_mass * omega`, with
  `omega = 2*pi*2 Hz`;
- rotate the resulting force and torque back into the world frame;
- norm-limit force to one body weight and torque to
  `body_weight * 0.3 m * 0.3` (about 29 Nm for the pinned G1).

Write the world-frame force to `xfrc_applied[16, :3]` and world-frame torque to
`xfrc_applied[16, 3:]` at the body's instantaneous inertial COM. The wrench is
held for the four 5 ms simulation substeps. It is overwritten on every policy
step and the disabled path writes exact zeros, because MuJoCo carries applied
wrenches rather than clearing them automatically.

The implementation is evaluation-only: a pure wrench computation plus a thin
rollout wrapper that replaces `state.data.xfrc_applied` before calling the
unchanged environment step. The training environment and SHAC implementation
remain untouched.

## Evidence and Gate

Run paired unassisted and assisted evaluations from identical resets. Preserve
the existing replay-free five-phase tracking metrics and additionally record per
phase maximum/RMS force, maximum/RMS torque, accumulated absolute wrench power,
finiteness, and cap compliance.

The oracle passes only if the assisted frozen E008 actor completes exactly
`499/399/299/199/99` transitions from phases `0/100/200/300/400`, with no
terminal failure and finite, bounded wrench traces. The paired unassisted result
must reproduce E008's registered survival within deterministic evaluator
semantics. A pass means a later learned-and-annealed assistance experiment is
plausible; it does not itself constitute a deployable method. A failure means
torso-only assistance is insufficient and no learned assistance head is built.

## Required Tests

- disabled scale produces an exact zero wrench and rollout-equivalent next state;
- force and torque occupy the correct MuJoCo/MJX `xfrc_applied` channels;
- every call overwrites stale wrench values;
- yaw-frame/world-frame rotation and quaternion shortest-path error are correct;
- force and torque norm caps hold and all outputs are finite;
- payload construction rejects incomplete/nonfinite evidence and reports the
  preregistered all-suffix gate.

## Non-goals

No actor or critic training, no checkpoint mutation, no phase-tape optimization,
no parameter sweep, no privileged observation in the actor, and no claim of
sim-to-real readiness. A learned assistance curriculum is a separate follow-up
requiring a new design and experiment registration.
