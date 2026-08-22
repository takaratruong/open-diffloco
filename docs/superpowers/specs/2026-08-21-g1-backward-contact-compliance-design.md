# G1 Backward-Only Contact Compliance Gradient Design

## Decision

Test one fixed compliant-contact straight-through estimator before any further
G1 policy training. The ordinary hard-contact MJX transition remains the exact
forward trajectory and evaluation target. A second transition from the same
state and action uses a fixed softer MuJoCo contact time constant only to supply
the backward derivative.

This is the smallest direct successor to E005 and E006. E005 showed that deleting
cross-contact credit is destructive. E006 showed that phase-local score/pathwise
mixing preserves the aggregate direction but damages fresh late-contact credit.
The remaining causal question is whether the contact derivative itself can be
made smoother without changing the behavior being optimized.

## Fixed Treatment

- Hard forward model: exact E023 G1 XML and runtime, `geom_solref[:, 0] = 0.02`,
  `geom_solref[:, 1] = 1.0`.
- Backward model: identical model except `geom_solref[:, 0] = 0.05` for every
  contact geom; dampratio, `solimp`, friction, timestep, solver, controller,
  reference, observations, rewards, resets, and action boundary are unchanged.
- At every transition, compute hard and compliant states from the same primal
  state/action. Publish the hard state exactly, but route floating-leaf cotangents
  only through the compliant state. Integer and boolean control leaves remain the
  hard values and are nondifferentiable.
- The 2.5x time-constant change is fixed before execution. Under MuJoCo's positive
  `solref` convention it gives 0.16x nominal stiffness and 0.4x damping. It is a
  discriminator, not a parameter sweep.

This is a narrow local analogue of backward-only soft-contact gradients. It is
not full DiffMJX Contacts From Distance: it adds neither positive-distance
contacts nor collision smoothing, and it does not use adaptive integration.

## Frozen Audit

Use only the fresh E023 actor initialization, because from-scratch viability is
the failed boundary. Reuse E006's exact H24 negative-return objective, 120 starts
(24 each at phases 0/25/50/75/100), fixed action-noise tapes 913024/913025,
`g1-4x5` and `diagnostic-10x20` solvers, per-environment norm-1 filtering, and
five-bin CAGrad. Capture ordinary hard-contact and backward-compliant gradients.
No actor update or training run is permitted in this experiment.

The audit must prove:

1. hard and surrogate forward qpos, qvel, reward, done, terminal, and actor
   histories are bit-identical for every captured transition;
2. direct actor gradients equal the action-node gradient pushed through the
   policy (cosine at least 0.999 and norm ratio in `[0.999, 1.001]`);
3. all five phase bins retain at least 16 finite nonzero contributors;
4. the hard and compliant models differ only in the registered contact
   time-constant column;
5. evidence, summary, plot, and completion manifest are written atomically,
   hash-bound, and reopened before interpretation.

## Registered Outcomes

- `backward-compliance-robust`: compliant gradients improve mean solver cosine
  and independent-tape cosine by at least 0.05, regress no phase by more than
  0.05, retain cosine at least 0.5 with the hard gradient, and retain norm ratio
  in `[0.25, 4]`. Authorize one separately registered 16-update fresh training
  gate with hard-contact forward physics and this backward estimator.
- `backward-compliance-neutral`: all validity and retained-signal gates pass, but
  either mean reliability improvement is below 0.05. Reject this estimator
  without tuning `solref`; move to an objective/demonstration curriculum.
- `backward-compliance-destructive`: retained direction/norm fails or any phase
  solver/tape cosine regresses by more than 0.05. Reject the estimator without a
  compliance sweep.
- `invalid-execution`: any provenance, exact-forward, model-delta, decomposition,
  finite-signal, or artifact gate fails. Repair execution before interpretation.

## Safety And Scientific Limits

The compliant model must never be used for published rollout behavior, reward
selection, or sim-to-real evidence. A positive frozen-gradient audit is only
authorization for a short training gate, not a locomotion result. A negative
result closes this fixed backward-compliance branch without tuning.

