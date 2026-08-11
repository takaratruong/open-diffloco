# G1 Fixed Torso-Assistance Curriculum Design

**Date:** 2026-08-11
**Status:** Approved for implementation

## Goal

Test whether temporary analytic torso assistance can improve the exact E008
residual policy while producing a final policy that runs with no external
wrench. This is a fixed curriculum, not a learned assistance controller: the
actor remains a 29-action joint controller and receives no new observation or
output.

## Starting Point

Resume the exact selected E008 checkpoint at transition `1,327,104`. Preserve
its frozen parent actor, trainable `328 -> 256 -> 29` residual adapter, Adam
state, actor normalizer, five-bin CAGrad, 256 physical environments with two
gradient-accumulation shards, H12 rollout, fixed `g1-4x5` solver, reference,
noise, domain randomization, rewards, and critic settings.

The existing analytic 2 Hz critically damped torso oracle remains unchanged:
one-body-weight force cap, `0.3 * bodyweight * 0.3 m` torque cap, world-frame
application at `torso_link`'s inertial COM, and exact zero overwrite when its
scale is zero.

## Fixed Curriculum

Train for exactly 64 actor updates (`393,216` transitions), ending at
`1,720,320`, with checkpoints every eight updates (`49,152` transitions).

- At transition `1,327,104`, the scheduled assistance scale is `1.0`.
- Linearly decay it to exact `0.0` at transition `1,622,016` (48 updates).
- Keep it at exact `0.0` for the final 16 updates (`98,304` transitions).
- During every nonzero-assistance update, independently force 25 percent of
  environments to exact zero assistance for the whole H12 unroll.
- Preserve the existing PRNG streams by using the train step's previously
  unused sixth split key for the assistance mask.
- Differentiate through the analytic feedback wrench and MJX dynamics. There
  is no additional assistance penalty because the externally fixed cap reaches
  exact zero; reward and policy gradients cannot choose to retain assistance.

This schedule is deliberately fixed and has no competence-triggered stage
logic. It is the smallest test of the user's curriculum proposal.

## Architecture

Add pure schedule and sampling helpers in a small SHAC curriculum module. Move
the evaluator's aligned-reference torso-target extraction into the existing
torso-wrench module so evaluation and training use one implementation.

The SHAC integration is opt-in and defaults off. When enabled, each per-env
actor rollout computes the analytic wrench from the current state and aligned
reference, scales it by that environment's curriculum value, overwrites only
the torso `xfrc_applied` row, and then calls the unchanged environment step.
The disabled path does not execute or write the wrench code.

Record scheduled scale, active-environment fraction, and force/torque magnitude
telemetry in training metrics and checkpoint metadata. Resume activation is
fail-closed unless the caller explicitly authorizes changing the assistance
treatment; an interrupted treatment must restore the same schedule exactly.

## Scientific Evaluation

Every archived checkpoint is evaluated with the existing replay-free five-phase
evaluator and exactly zero external wrench. Assisted training reward is never a
checkpoint-selection metric. The fixed E008 vector `70/63/95/70/44` and robust
key `(minimum=44, median=70, mean=68.4)` remain the comparison.

The pilot advances only if a zero-wrench checkpoint is finite, preserves the
frozen parent/parent optimizer/normalizer contract, and strictly improves the
lexicographic `(minimum, median, mean)` key. Completing the phase-zero
499-transition suffix with zero wrench is a major success but is not required
to identify a useful improvement. If no checkpoint improves the key, reject
this fixed curriculum rather than retaining an assisted policy.

## Required Tests

- schedule is exactly `1.0` at the start, `0.5` halfway through decay, and
  bit-exact `0.0` at and after the end;
- invalid schedule bounds and zero fractions fail before environment creation;
- zero-assistance sampling is deterministic, bounded, and produces exact zeros;
- aligned torso targets and wrench computation are shared by evaluator and
  training without changing evaluator results;
- disabled training configuration preserves the old rollout path;
- enabled rollouts overwrite stale torso force, retain other body forces, emit
  finite bounded wrenches, and have finite policy gradients;
- resume activation requires explicit authorization and an interrupted active
  curriculum restores its exact parameters;
- the immutable runner changes only endpoint and assistance parameters relative
  to the E008 continuation and exposes no scientific CLI overrides.

## Non-goals

No scalar gate, learned six-dimensional wrench head, force tape, adaptive stage
advancement, reward reshaping, architecture change, optimizer change, carried
reset bank, new motion, sim-to-real claim, or hardware deployment is included.
