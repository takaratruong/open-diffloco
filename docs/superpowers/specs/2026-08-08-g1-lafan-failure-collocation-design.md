# G1 LAFAN Failure-Centered Collocation Design

## Goal and evidence boundary

Build the smallest inspectable trajectory-optimization path that can alter the
specific delayed pelvis-height collapse on the pinned 500-state LAFAN1 slice.
The optimizer produces a corrected physical state and normalized PD-target
action sequence. Policy fitting is explicitly out of scope for this slice.

The immutable scientific inputs are:

- reference
  `dance1_subject2_f122_422_50hz.npz`, SHA-256
  `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`;
- the validated G1 task at 50 Hz, MJX solver iterations 4 and line-search
  iterations 5, float64;
- normalized 29-dimensional policy actions mapped to RMR PD position targets;
- the selected targeted-recovery actor at step 688128, whose strict phase-zero
  rollout survives 136 transitions and terminates only on anchor-height error.

The unchanged strict uninterrupted evaluator is the only behavioral decision
gate. A lower transcription merit is not evidence of improved tracking.

## Why the previous collocation stack failed

The earlier work established two boundaries that this design must respect.

First, reconstructing an MJX knot from physical state does not strictly close
the ordinary carried transition. Adding exact qpos, qvel, warm-start
acceleration, and carried control reduced but did not remove the mismatch. The
best fully fixed reconstruction still reached `1.15e-4` qpos and `0.00339`
qvel error. Therefore the transcription plant is an explicitly stateless
qpos/qvel segment map, while the uninterrupted evaluator remains authoritative.

Second, the prior injected 16-knot policy/iLQR calibration was too large and
fragile for the evidence question. It spent about 33 minutes assembling
derivatives, failed action finite-difference gates at seven contact or
support-boundary stages, produced a non-finite state search direction, and
accepted zero steps. This design does not optimize policy parameters, inject a
full-trajectory direction, or require a monolithic long-horizon adjoint.

## Formulation

Use a failure window beginning at phase 111 and ending at phase 135. Divide its
24 control transitions into 12 independent two-step shooting segments.
Assemble those independent segment maps with a single vectorized lowering so
the full residual/JVP probe does not duplicate twelve large MJX graphs.

For segment `s`, the state knot is physical `x_s = (qpos_s, qvel_s)` with
dimensions 36 and 35. The initial knot is fixed to the selected actor rollout.
The remaining 12 state knots and all 24 normalized actions are decision
variables. This gives `12 * 71 + 24 * 29 = 1548` scalar decision variables.

Each segment reconstructs fresh MJX data from its qpos/qvel knot with
`mjx.forward`, then advances exactly two 50 Hz validated task transitions using
the unchanged RMR PD target conversion, torque limits, five-millisecond source
physics step, four physics substeps per action, and solver budget 4/5.

The segment defect has 70 coordinates:

- root translation difference: 3;
- shortest-sign relative root quaternion rotation vector: 3;
- hinge-joint position difference: 29;
- generalized velocity difference: 35.

Each free knot also has one unit-quaternion equality. Thus the 12 segments have
`12 * (70 + 1) = 852` equality residuals. The qpos/qvel-only reconstruction
mismatch is an admitted property of this stateless transcription, not silently
claimed to reproduce uninterrupted carried solver caches.

The objective contains only terms needed for recovery:

- squared signed pelvis-height error to the pinned reference at every knot;
- a small joint-position tracking term;
- squared deviation from the actor's nominal actions;
- action-rate regularization.

The feasibility merit adds squared equality defects and squared violations of:

- the four unchanged RMR terminal slacks: anchor z, anchor xy, gravity z, and
  distal-body z;
- normalized action support `[-1, 1]`;
- raw PD torque authority before clipping;
- a declared maximum contact penetration diagnostic.

Contact dynamics themselves remain inside the validated MJX segment map. The
penetration diagnostic is not a substitute contact model. The first physical
probe found non-finite derivatives in the combined path vector; removing only
penetration still left a non-finite derivative because terminal body poses had
been obtained through a full contact solve. The smoke now evaluates those
qpos-only terminal poses with MJX kinematics and returns terminal, action,
torque, and contact-penetration slacks as separate groups. Contact values and
active indices remain mandatory evidence; non-finite contact derivatives are
reported and excluded from the candidate solve without smoothing. A later
differentiable contact surrogate requires a separate preregistered formulation
choice.

## Initialization and first vertical slice

Capture the selected actor's qpos, qvel, actions, phases, and hard-limit errors
through pre-action phase 135 using the existing rollout-capture path bound to
the pinned reference. This is the last physical state exposed before the
strict evaluator's failing transition resets its returned data. Subsample every
second state from phase 111 as knots and retain every action; do not invent a
post-reset phase-136 terminal knot.

Before any solve, the vertical slice must prove:

1. the fixed window has exactly 1548 decision variables and 852 equality
   residuals;
2. an exact rollout generated by the same stateless segment map has zero or
   floating-point-near-zero defects and unit-quaternion residuals;
3. the scalar objective, equality directional derivative, and constraint
   directional derivative are finite on a one-segment physical smoke case;
4. the complete 12-segment/852-residual map and its directional derivative are
   finite when one fixed-shape two-step kernel is compiled once and evaluated
   across every physical segment, with at least one probed segment explicitly
   verified to begin in active contact;
5. direct plant calls are exactly deterministic, while the separately compiled
   checkpointed task step remains within declared `3e-6` qpos and `3e-4` qvel
   compiler-path bounds;
6. all phase, state, action, reference, model, solver, and precision identities
   are reported.

No multi-minute or multi-iteration optimizer may run until these gates pass.

## Matched before/after behavioral gate

The later bounded solve will save a corrected action tape and optimized knots,
but comparison uses actions only in the ordinary strict evaluator:

- **before:** the selected actor, exact phase zero, unchanged 499-transition
  reference and true-terminal rules;
- **after:** the same actor through the fixed splice start, the corrected action
  tape over the fixed recovery window, then a preregistered continuation rule;
- both use the same plant, reference, float64 precision, solver 4/5, phase
  convention, reward, and hard termination thresholds.

The first material gate is strict survival of at least 161 transitions: 25
beyond the selected actor's immutable 136-transition baseline. Completion of
all 499 transitions is a stronger outcome. Anything shorter than 161 is
nonmaterial even if transcription defects or local pelvis height improve.

## Framework-neutral corrected episode adapter

The transcription output must not depend on JAX, Flax, or Torch checkpoints.
Expose one thin local mapping containing exact 0.02-second timing, reference
chronology, root pose, tracked-body pose and velocity, normalized
29-dimensional PD-target actions, canonical joint/body orders, hashes, and
source/correction provenance.

The offline-diffusion lane owns canonical schema
`sonic_grail_rollout_npz_v1`. The local adapter returns its exact raw fields:
root position/rotation and world-frame angular velocity; all 30 non-world body
positions, WXYZ rotations, and world-frame linear velocities; 29 canonical
joint positions, velocities, normalized PD-target actions, joint names,
defaults, and action scales; and canonical body names. It preserves pre-action
state/action chronology without the TML P2/+1 interpretation or a storage
shift. Source actions are reordered by joint name and converted by preserving
the physical PD target before applying TML's frozen defaults and scales.

The adapter requires at least 13 state/action rows, exact 0.005-second
simulation and 0.02-second control timing with decimation four, complete qdot
and actions, and the frozen raw-action semantics. DiffSim outputs additionally
bind correction method, run ID, source hash, 40-hex code commit, dynamics model
hash/backend, `trajectory_source: diffsim_corrected`, and positive episode
weight, plus the standard clip/origin/checkpoint/config/motion/terrain/GRAIL
provenance. No direct Torch-through-JAX bridge is required.

## Failure handling and scope

Non-finite values, incorrect dimensions, wrong provenance, non-negligible
identity defects, or non-finite directional derivatives stop before a solve.
Contact active-set derivative failure causes a bounded formulation review, not
automatic smoothing or policy optimization. This slice adds no framework,
generic optimizer family, critic, policy fitting, reset distribution, or new
reward term to the strict evaluator.
