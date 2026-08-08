# G1 LAFAN1 Long-Reference DiffSim Design

## Objective

Test whether the existing direct-from-random-weights MJX/SHAC method enters a
stable tracking basin on a continuous ten-second G1 dance reference rather than
the current 1.2-second clip. This is a one-seed mechanism test, not a
repeatability or sim-to-real claim.

The experiment changes only the reference motion. It preserves the selected
E-20260807-006 training contract: a random 512-by-512 actor, ordinary nonzero
output head, no actor LayerNorm, 256 physical environments, a 12-step
differentiable horizon, 128 actor updates, actor learning rate 1e-3, zero actor
value bootstrap, and action noise annealed from 1.0 to 0.1. It also preserves
exact reference-state initialization and excludes reset perturbations,
observation noise, pushes, terrain randomization, and adaptive phase sampling.

## Pinned Motion Input

- Dataset: `lvhaidong/LAFAN1_Retargeting_Dataset`
- Dataset revision: `ce1572906efe6157840e8474d5a0d7aa87481e74`
- Dataset license: CC BY-NC-ND 4.0
- Robot and source: `g1/dance1_subject2.csv`
- Source format: 30 FPS, root `xyz+xyzw`, followed by the 29 documented G1
  joints
- Selected inclusive source frames: 122 through 422
- Selected duration: 10.0 seconds from 301 source samples
- Expected RMR output: 500 samples at 50 Hz
- Dataset source:
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>

The source file is 1,352,694 bytes at the pinned revision and has Hugging Face
ETag `c3be5fdfc1b07040a724897941a4f2637f01e8c6`. Acquisition must additionally
record its local SHA-256 before experiment approval.

## Reference Preprocessing

Use the clean RMR converter at
`/home/ubuntu/projects/rmr_tracking/scripts/csv_to_npz.py`. The tracked file is
clean at the pinned repository commit
`8a886f3a9df561df3454e6b6233ab8c54d66f097`; its last content change was
`fcc0cb2bf68c7e727f8ecd2168a4690899a95276`, and it has SHA-256
`1724fe91ee4da6d1136510db83598c4875d20297cf0f56dd55c13326122ea4f5`.
Run it at input FPS 30, output FPS 50, and frame range 122–422. Network logging
must be disabled or offline; the authoritative artifact is the locally saved
NPZ, not a mutable W&B alias.

This path is required because RMR:

- linearly interpolates root position and joint position;
- spherically interpolates root quaternion;
- computes root and joint linear velocity with the output-timebase gradient;
- computes root angular velocity with the centered SO(3) derivative; and
- evaluates forward kinematics before publishing rigid-body targets.

Do not load the CSV directly in the training environment and do not recompute
velocities with an unrelated finite-difference convention.

## RMR-NPZ Compatibility

Extend `load_mujoco_reference` without changing its existing `X`/`V` behavior.
For an RMR-format archive, require all of:

- scalar or one-element `fps` equal to 50;
- `joint_pos` and `joint_vel` shaped `(T, 29)`;
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w` with the
  same `T` and finite numeric values;
- the first logged rigid body identified as the pelvis/root link; and
- a conversion sidecar recording that the pinned RMR articulation joint order
  matches the controller's recorded source joint order.

The loader remains length-generic. The experiment preflight—not the reusable
loader—requires exactly 500 frames for this registered input.

Construct MJX generalized positions from the logged root-link pose plus joint
positions permuted from RMR source order into MuJoCo model order. Construct
joint velocities with the same permutation. Recover the six root generalized
velocities through the exact MuJoCo root-body Jacobian so that forward
kinematics reproduces RMR's logged world-frame root linear and angular
velocities; do not assume a free-joint angular-velocity convention.

After conversion, reuse the existing MuJoCo forward-kinematics path to compute
the 14 RMR tracking-body targets in the exact model used by MJX. Reject rather
than clip nonfinite values, invalid quaternions, shape mismatches, joint-order
mismatches, or reference/model velocity disagreement.

## Training and Evaluation Plumbing

Add an explicit `reference_path` to the SHAC training and evaluation entry
points and record its absolute path and SHA-256 in `hparams.json`. The new NPZ
is already sampled at the 50 Hz control rate, so this reference uses stride one;
the existing 100 Hz `X`/`V` reference remains stride two. Validate that
`reference_stride / reference_fps` equals the 20 ms control interval.

Do not hard-code a 60-step episode for the new reference. Clip-end completion
is 500 carried control steps. Training environments continue across 12-step
gradient windows and reset only at a true terminal or the actual reference end.

Use seed zero only. Evaluate fixed actor updates 10, 20, 30, 40, 50, and 128
without simulator-state replay. Select first by full-reference carried
survival, then reward and reference-relative tracking errors. Publish the
selected full 500-step trajectory, summary, video, and contact sheet.

## Evidence Gates

### Preprocessing gate

- Source revision, byte count, ETag, local SHA-256, converter commit, converter
  file hash, frame range, input FPS, and output FPS are recorded.
- Output contains 500 finite frames with unit root quaternions and valid G1
  shapes.
- Reconstructed MuJoCo root-link pose and velocity agree with the RMR NPZ at
  every frame within float32 conversion tolerance.
- No joint value is silently reordered, clipped, or replaced.

### Execution gate

- Exactly 128 actor updates and 393,216 transitions execute at 256 environments
  and horizon 12.
- Sampled actor and critic gradients and every selected checkpoint leaf remain
  finite.
- All hyperparameters other than reference path, reference FPS/stride, and
  reference-derived episode length match E-20260807-006.

### Behavioral gate

The selected actor must carry simulator state for all 500 control steps with:

- zero true terminals and zero intermediate resets;
- mean anchor-position error no greater than 0.15 m;
- mean anchor-orientation error no greater than 0.20 rad;
- mean tracked-body position error no greater than 0.10 m;
- mean tracked-body orientation error no greater than 0.25 rad;
- reference-relative anchor-height error below 0.15 m at every frame; and
- reference-relative projected-gravity error below 0.4 at every frame.

The video must show continued reference-following without progressive lateral
collapse. Numerical passage without that visible condition is a tracking-quality
failure, not stable walking.

## Outcome Map

1. **Long-reference stable tracking:** all preprocessing, execution, numerical,
   and visible gates pass. Retain the longer motion as the new baseline before
   adding recovery perturbations.
2. **Long-reference stepping with delayed instability:** learning and stepping
   are visible, but a carried rollout violates a stability gate. Preserve the
   best checkpoint and next isolate structured reset perturbations from a longer
   temporal-credit horizon.
3. **Reference complexity prevents basin entry:** execution is finite, but no
   fixed checkpoint produces meaningful carried tracking. Use a continuous
   five-second subsection of the same source and keep every other setting fixed.
4. **Invalid reference conversion:** any provenance, joint-order, timebase,
   kinematic, or velocity gate fails. Repair preprocessing without interpreting
   policy learning.

## Tests

- Existing `X`/`V` reference loading remains byte-behavior compatible.
- A synthetic RMR-format fixture reconstructs exact model-order qpos and qvel.
- Root Jacobian conversion reproduces prescribed world-frame pelvis linear and
  angular velocity at multiple orientations.
- Wrong FPS, missing keys, inconsistent frame counts, malformed joint order,
  nonunit/zero quaternions, and nonfinite values fail explicitly.
- Runner tests prove `reference_path`, FPS, stride, derived episode length, and
  artifact hash reach both training and evaluation.
- A short evaluation fixture proves clip end is not reported as a true terminal
  and that no hidden 60-step cap truncates a longer reference.

## Non-Goals

- No second seed.
- No full two-minute dance yet.
- No perturbed RSI, pushes, observation noise, environment-count increase, or
  horizon change in the same experiment.
- No PPO initialization, teacher actions, state replay, residual controller,
  collocation, sim-to-sim, or hardware claim.
