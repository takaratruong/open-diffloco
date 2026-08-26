# Counterfactual Wrench-to-Leg Distillation

## Decision

Test whether the successful E004 torso-wrench teacher can provide direct
credit to a strictly leg-only residual without applying any external wrench to
the trained policy. Freeze the complete retained E026 controller, its
normalizer, and the E004 wrench head. Train one zero-effect residual over the
twelve hip, knee, and ankle actions. The residual must reproduce the local
closed-loop effect of the teacher's wrench through ordinary foot contact.

This is not another wrench annealing experiment. The trained branch receives
an exact-zero `xfrc_applied` at every step.

## Alternatives Considered

1. **Counterfactual transition matching (selected).** From every student state,
   evaluate a stop-gradient assisted teacher transition and an unassisted
   student transition. Match their base twist and centroidal-momentum changes.
   This supplies the legs with the effect the wrench produced while retaining
   the real differentiable contact path.
2. **Direct contact-wrench matching.** Match student foot-contact wrench to the
   teacher's contact wrench plus torso wrench. This is physically explicit but
   depends directly on noisier solver contact-force outputs and is a poorer
   first experiment.
3. **Constrained inverse-dynamics labels.** Project the teacher wrench through
   a contact QP and imitate the resulting torques. This exposes feasibility
   clearly, but adds a second optimizer and modeling assumptions before testing
   whether the existing differentiable simulator already supplies the needed
   credit.

## Frozen Teacher And Student

The teacher is the exact E004 update-16 composite that first completes all 271
phase-zero transitions:

- frozen E026 joint controller and normalizer;
- frozen learned 328-256-6 torso-wrench head;
- the registered force and torque caps and exact continuous reference;
- no parameter or optimizer update anywhere in the teacher.

The student starts from exact E026 and adds a new 328-256-12 ELU residual with
a zero output head. Its twelve outputs are scattered into the canonical
29-action order at:

- left/right hip pitch;
- left/right hip roll;
- left/right hip yaw;
- left/right knee;
- left/right ankle pitch;
- left/right ankle roll.

All waist and upper-body correction coordinates remain bit-exact zero. The
student action is `E026_action + scattered_leg_residual`, passed through the
same action boundary as E026. The student always applies an exact-zero torso
wrench.

## Counterfactual Data Flow

At each carried student state `x_t`:

1. Compute the frozen E026 action and frozen E004 wrench from the same
   normalized observation.
2. Step one teacher counterfactual from `x_t` using the E026 action plus the
   E004 torso wrench. Stop gradients through the complete teacher result.
3. Step the real student from the same `x_t` using E026 plus the leg residual
   and exact-zero torso wrench.
4. Compare teacher and student changes in root-frame base linear velocity,
   base angular velocity, and centroidal linear/angular momentum.
5. Carry only the student state into the next rollout step.

The teacher branch is therefore a local vector-field target, not a parallel
trajectory whose state can drift away from the state the student actually
visits.

## Objective

Use normalized pseudo-Huber losses on:

- base linear-velocity change (3);
- base angular-velocity change (3);
- centroidal linear-momentum change (3);
- centroidal angular-momentum change (3).

Normalize each three-vector by the teacher target RMS measured by the frozen
feasibility artifact, with a floor of `1e-3`. Use pseudo-Huber delta `0.1`,
average the four blocks with equal weight `0.25`, add residual-action magnitude
weight `0.01`, and add residual temporal-change weight `0.001`. These constants
are fixed before observing training and are not command-line sweep parameters.

Retain the established reference-tracking objective and CAGrad phase bins. Add
the counterfactual loss as one task rather than scalarizing it into the
tracking reward. Regularize only the 12-D correction magnitude and temporal
change. Do not add a phase, named-foot reward, capture reward, assistance
schedule, replay, pushes, observation noise, or physics randomization.

The four-block counterfactual loss is one CAGrad task beside the unchanged
tracking tasks. There is no weight sweep.

## Feasibility Discriminator

Before a training budget is authorized, evaluate the frozen teacher on the
same five exact reset phases used for E007. At each valid stance state, compute
the Jacobian of the 12-D leg residual to the one-step 12-D target change and a
bounded damped-least-squares correction under the policy action boundary.

The discriminator reports:

- teacher target-change norm;
- projection residual norm and normalized residual;
- Jacobian rank and singular values;
- required action RMS/max and bound activity;
- foot contacts and teacher wrench;
- per-phase and aggregate finite coverage.

Proceed to training only if all five phases have finite nonzero target and
gradient coverage, and the median normalized projection residual over stance
states is at most 0.50. This is a permissive necessary-value gate: it asks only
whether the leg action path can explain at least half of the teacher effect
locally. Failure closes strict leg-only distillation and redirects the next
design to a contact-feasible projection or inclusion of the waist; it does not
authorize threshold tuning.

## Training And Selection

If the discriminator passes, run one bounded 32-update effective-512 H24
continuation with checkpoints every eight updates. Selection uses deterministic
CPU replay at phases `0/25/50/75/100` because E007 established terminal-sensitive
GPU nondeterminism.

A candidate advances only if it:

- preserves the complete frozen E026 and teacher trees bit-exactly;
- has finite positive leg-residual gradients and updates;
- applies an exact-zero external wrench throughout training and evaluation;
- keeps every non-leg residual coordinate bit-exact zero;
- componentwise preserves retained E026 survival `131/114/74/71/74`;
- strictly improves at least one phase; and
- improves the minimum/median/mean survival key without a compensating phase
  collapse.

If training is finite but no checkpoint meets every preservation gate, retain
E026 and classify counterfactual wrench distillation as insufficient. Do not
continue longer or tune the task weights from the same evidence.

## Evidence And Failure Handling

Preflight binds code, E004 checkpoint and hparams, E026 subtree, reference,
model, controller, solver, joint order, seed, and dirty-patch state. Artifacts
must include the feasibility tensors/report, training telemetry, frozen-tree
hashes, checkpoint hashes, deterministic phase grid, selection JSON, learning
curves, phase-zero video, and contact sheet.

Any nonfinite teacher/student state, mismatched reset, stale `xfrc_applied`,
nonzero frozen or non-leg update, missing artifact, or provenance mismatch is
invalid execution rather than negative scientific evidence.
