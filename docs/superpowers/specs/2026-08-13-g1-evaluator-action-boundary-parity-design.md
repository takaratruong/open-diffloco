# G1 Evaluator Action-Boundary Parity Design

## Problem

SHAC training clips the composite frozen-parent-plus-residual action to
`[-1, 1]` whenever `squash_actor_actions` is enabled. The G1 source-step
environment itself is deliberately unclipped, while `evaluate_g1_tracking.py`
currently passes the composite policy action directly to `env.step`. E015's
rendered "training" rollout therefore used actions outside the training
boundary and its phase-grid evaluation did not reproduce training inference.

## Design

Add one pure helper in `tools/evaluate_g1_tracking.py` that applies the same
post-noise action boundary as SHAC: clip to `[-1, 1]` exactly when the
environment advertises `squash_actor_actions`, otherwise preserve the action
bit-for-bit. Use it after optional training-noise injection and immediately
before `env.step`. Record both the raw policy/noise tensors and the effective
post-boundary action in training-rollout evidence so the distinction remains
auditable.

Do not change training, actor parameters, action scales, PD gains, reset
distribution, noise magnitude, or environment clipping. Existing unbounded
RMR action-parity environments retain unbounded evaluation because they set
`squash_actor_actions` false.

## Verification and Evidence

Use test-driven development to prove bounded and unbounded behavior before the
implementation. Run the focused evaluator tests and static checks. Then rerun
the selected E015 checkpoint at phase zero with and without its fixed 0.2
training noise, followed by the exact five-phase deterministic grid. Treat the
old E015 videos and phase-grid result as superseded diagnostic artifacts; do
not retrain.

