# G1 Foot-Propulsion Diagnostic Design

## Goal

Determine whether the unassisted walking failure is caused by insufficient or
mistimed foot-generated forward propulsion before changing the training
objective.

## Scope

This change is evaluation-only. It does not alter the environment reward,
policy observation, training update, contact parameters, or the active E005
run. It augments the existing replay-free G1 evaluator with raw arrays and a
time-aligned diagnostic plot.

## Measurements

For every completed control interval, record:

- the world-frame translational constraint force on the floating base from
  `data.qfrc_constraint[:3]`;
- that force rotated into the current pelvis-yaw frame, where component zero
  is forward;
- the left/right foot support bits from the existing grouped contact topology;
- the reference-required forward force, defined as total model mass times the
  finite-difference acceleration of the reference pelvis velocity over the
  same control interval and expressed in the current pelvis-yaw frame;
- torso pitch and the actually applied torso force.

The constraint force is the primary differentiable signal. Per-foot force is
not inferred in this first diagnostic because decomposing MuJoCo's friction
constraints introduces a separate sign/frame validation problem. Left/right
support still establishes which stance generated the measured net propulsion.

## Outputs

`evaluation.npz` gains finite, row-aligned arrays for net constraint force,
yaw-frame constraint force, support bits, required reference force, torso
pitch, and applied torso force. The evaluator also
writes `foot_propulsion_diagnostics.png`, containing shared-time plots for
forward actual versus required force, left/right support, torso pitch, and
assistance force.

The JSON summary records RMS forward-force error and peak absolute forward
force. Publication is descriptive: it does not make a learning claim.

## Correctness Boundaries

- The diagnostic requires `xfrc_applied` to be exactly zero when used to judge
  unassisted propulsion; assisted rollouts remain plottable but are labeled.
- World-to-yaw rotation must preserve vertical force and vector norm.
- All arrays must have exactly one row per recorded transition and contain only
  finite values.
- The reference endpoint uses a one-sided finite difference rather than
  indexing past the final frame.
- A zero constraint force must produce exactly zero propulsion.

## Validation

Pure unit tests cover yaw rotation, force-to-impulse conversion, reference
finite differences, endpoint handling, shape/finiteness rejection, and summary
statistics. Evaluator integration tests verify NPZ/PNG/JSON publication and
row alignment. A replay-free E005 update-128 rollout then provides the first
descriptive plot. Only after inspecting that plot should a separately
registered experiment add a pseudo-Huber propulsion reward.
