# Capture-point tracking objective

## Decision

Do not train the rejected four-step reference-momentum objective. Retain its
verified centroidal measurement and replace the scientific target with the
planar divergent component of motion (capture point):

`xi = com_xy + com_velocity_xy / sqrt(g / com_z)`.

The auxiliary error is the robot/reference capture-point difference divided
by the pinned standing COM height. It is evaluated at every active pre-reset
state with a pseudo-Huber kernel (`delta=0.1`). It does not name a foot, gait
phase, or joint and remains differentiable through contact and the complete
joint-to-root dynamics.

## Evidence before implementation

On the exact E006 common 106-transition prefix, normalized capture-point
error separates the successful assisted controller from both failures:

| Controller | RMS | p99 | final |
| --- | ---: | ---: | ---: |
| retained unassisted E026 | 0.3651 | 0.9479 | 0.9877 |
| successful assisted E004 | 0.1908 | 0.4426 | 0.4616 |
| exact-zero E005 | 0.4842 | 1.4369 | 1.4849 |

This succeeds where raw four-step momentum failed because assistance need not
imitate the reference force or momentum increment. It must keep the combined
COM position/velocity state recoverable. The signal also gives non-saturating
credit for the observed backward lean and insufficient recovery step.

## Fail-closed gates

1. Recompute the E006 values from hash-bound qpos/qvel/momentum traces. E004
   p99 and final error must each be at most 80 percent of both comparators.
2. Verify CPU and MJX capture-point measurements agree and gradients are
   finite and nonzero.
3. Audit one frozen-E026 training batch before treatment. Register one static
   loss weight from the measured auxiliary/base gradient-norm ratio; do not
   sweep weights.
4. Only after all gates pass, train a matched pair of new zero-head joint
   residuals over the exact frozen E026 composite. Both arms use the same
   prefix-exact continuous reference, uniform exact reference-state resets,
   no carried bank, no policy-anchor penalty, and exact-zero assistance. The
   treatment adds only the capture-point loss; the control weight is zero.
5. Select only replay-free five-phase survival. The capture loss is a training
   diagnostic and never a checkpoint-selection substitute.

The continuous reference and reset distribution are intentional shared arm
conditions, not inherited learned-wrench behavior: the short E026 reference
ends before the failure being solved. Causal interpretation comes from the
matched zero-capture residual control, while the original E026 policy remains
bit-exact frozen in both arms.

## Rejected alternatives

- Do not optimize raw ground force, a named knee torque, or a particular foot.
  Those are morphology- and phase-specific proxies.
- Do not optimize distance to the instantaneous support center. On the E006
  prefix, successful E004 intentionally carries a larger capture-to-support
  distance than E026 while setting up the next step.
- Do not add another reward/horizon/noise sweep. The next run changes only the
  physically motivated capture objective behind a frozen parent.
