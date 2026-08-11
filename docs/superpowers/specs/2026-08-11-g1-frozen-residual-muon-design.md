# G1 Frozen Residual Muon Design

## Scientific question

Does Muon improve the robust replay-free survival of the frozen nonlinear
future-residual adapter when its parent policy, adapter initialization,
architecture, auxiliary Adam updates, differentiable objective, simulator,
batch, horizon, exposure, checkpoint grid, and evaluator remain fixed?

E-20260811-008 establishes the Adam control: the selected checkpoint survives
70/63/95/70/44 transitions from phases 0/100/200/300/400, with a
minimum/median/mean key of 44/70/68.4. This is a large improvement over the
linear preview control, but phase-zero survival remains 70 of 499 transitions.
The optimizer comparison is therefore a bounded discriminator, not a claim
that optimizer choice alone will solve full-sequence tracking.

## Alternatives considered

1. Apply the stock Optax Muon wrapper to every two-dimensional adapter
   parameter. This would also apply Muon to the zero-output action head and use
   Nesterov Adam for auxiliary parameters, confounding the matrix treatment
   with the head optimizer.
2. Vendor or rewrite Muon. This duplicates maintained Optax machinery and adds
   numerical and maintenance risk without changing the scientific question.
3. Use Optax's installed official Muon transform only for the adapter's
   328-by-256 input-to-hidden kernel, while preserving ordinary Adam for its
   input bias, 256-by-29 zero-output kernel, and output bias. This is the
   selected design because it changes one optimizer assignment and preserves
   the E008 head behavior exactly.

## Optimizer architecture

The treatment starts from the same exact E008 no-preview checkpoint and repeats
the same append-only delta-preview migration plus zero-effect residual adapter
initialization as E008. It does not continue from E008's trained adapter. The
initial actor action, residual parameters, environment state, critic state,
normalizer, PRNG state, and training step are therefore identical to E008 at
the treatment boundary.

The existing global actor-gradient clip at norm 1.0 is applied once to the
complete adapter gradient tree before optimizer partitioning. The
input-to-hidden kernel is then updated by `optax.contrib.muon` with five
Newton-Schulz steps, beta 0.95, Nesterov momentum, Frobenius preconditioning,
zero weight decay, and `consistent_rms=0.2`. The remaining adapter parameters
are updated by the unchanged `optax.adam` transform with beta1 0.9, beta2
0.999, and the same actor learning-rate schedule.

The parent checkpoint's Adam state is retained as an immutable snapshot rather
than passed through either optimizer. Both new adapter optimizer states inherit
the parent's optimizer and schedule counts while beginning with zero adapter
momenta. This matches E008's continuation schedule and zero-moment boundary
without allowing zero gradients to decay frozen parent moments. A dedicated
state type holds the immutable parent snapshot, Muon kernel state, and
auxiliary Adam state.

## Training integration and telemetry

Add one default-`adam` string option,
`actor_residual_preview_optimizer`, with allowed values `adam` and `muon`.
`muon` is valid only when `actor_residual_preview_adapter` is true. Treated
resumes must match the saved optimizer value; legacy and non-residual paths are
unchanged.

The Muon update helper returns the same composite actor-update tree and existing
diagnostic keys used by the Adam residual path. Parent updates are exact zeros,
the parent optimizer snapshot is byte-exact, and only the adapter states
advance. Additional metadata records the optimizer assignment and fixed Muon
hyperparameters. Checkpoint audits require finite numeric state, nonzero Muon
kernel updates after the zero head becomes active, nonzero auxiliary Adam
updates, exact parent parameter/snapshot/normalizer retention, and the same
five occupied CAGrad bins.

## Experiment boundary

The immutable runner differs from E008 only by setting
`actor_residual_preview_optimizer="muon"`. It preserves seed 0, the exact E008
parent, delta lookahead offsets +4/+8/+12, the 328-to-256-to-29 ELU residual,
zero output head, 64 H12 updates, 256 physical environments, accumulation two,
effective population 512, five-bin CAGrad, fixed 4x5 MJX solver, zero
termination margin, domain randomization, action noise, 393,216 transitions,
and eight-checkpoint cadence.

All eight checkpoints use the existing replay-free evaluator at phases
0/100/200/300/400. Selection remains maximum minimum survival, then median,
mean, and earliest step. Muon advances only if it clears E008's per-phase floors
70/63/95/70/44 and obtains a key strictly greater than 44/70/68.4. Training
reward cannot select a checkpoint.

## Branch decision and stop condition

If Muon advances, retain it for the subsequent carried-survival intervention.
If it is finite but does not advance, retain E008's Adam optimizer. Invalid
provenance, zero-effect migration, frozen-state, optimizer-state, telemetry,
evaluation, crash, or timeout evidence is repaired without scientific
interpretation.

Do not sweep Muon learning rate, beta, Newton-Schulz steps, RMS scaling,
preconditioning, activation, width, or output-head optimizer. After this single
comparison, the next scientific intervention targets carried survival or
terminal value because neither optimizer changes the fact that the best current
phase-zero policy survives only 70 of 499 transitions.

## Verification boundary

Tests must prove one global clip before partitioning; Muon selection only for
the 328-by-256 kernel; ordinary Adam equivalence for the remaining parameters;
inherited optimizer/schedule counts; zero initial adapter moments; exact parent
optimizer snapshot preservation; finite adapter-only updates; strict
configuration and resume validation; unchanged default behavior; exact runner
delta; and complete optimizer metadata.
