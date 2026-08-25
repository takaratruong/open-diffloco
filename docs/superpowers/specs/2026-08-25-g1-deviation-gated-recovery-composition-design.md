# G1 Deviation-Gated Recovery Composition

## Purpose

Determine whether E026's proven short-walk recovery correction can be made
behaviorally safe by activating it only after the retained E023 actor has
measurably drifted from the reference. This is a frozen-policy discriminator,
not another training sweep.

The experiment retires one uncertainty: whether the observed cross-phase
regressions come from applying useful recovery corrections on nominal states,
or whether the correction itself is incompatible with the retained policy.

## Evidence boundary

- E023 is the retained fresh differentiable-physics actor. Its registered
  short-walk phase vector is `116/99/67/49/24`.
- E026 freezes E023 and trains a residual adapter from actual pre-failure
  carried states. It completes the finite short-reference suffixes, but its
  phase-zero video is already toppling at the clip boundary.
- E027/E028 show that applying a recovery adapter globally and training it
  longer redistributes competence rather than preserving all starts.
- E029 rejects a high-dimensional nearest-neighbor compact-support gate because
  the positive and protected observation sets are not separable.
- E016 shows that H24 and H48 gradients are already strongly aligned. Horizon,
  learning-rate, activation, and scalar KL tuning are not the next axes.

## Treatment

Load the exact E026 composite checkpoint, whose parameter tree contains the
immutable E023 parent and the trained E026 residual adapter. Replace only its
composition rule:

`action = parent_action + gate(error) * residual_action`.

`error` is the previous transition's aligned mean body-position error in
metres, the same physical scalar already computed by the G1 environment and
used by the registered DiffMimic replay treatment. At reset, the error is zero.
The gate is fixed before execution:

- exactly zero for `error <= 0.10 m`;
- cubic smoothstep from zero to one on `(0.10 m, 0.20 m)`;
- exactly one for `error >= 0.20 m`.

The two endpoints are not tuned from this experiment. `0.20 m` is the
independently registered E008 DiffMimic replay threshold and `0.10 m` is its
fixed half-threshold. The gate is phase-free, reference-length independent,
and deterministic. It uses no future outcome, termination flag, assistance,
replay, noise, randomization, or privileged force signal.

The parent and residual parameters, normalizer, plant, controller, reference,
solver, action boundary, and seed remain bit-identical. No gradients or
optimizer steps execute.

## Evaluation

In one process, evaluate three frozen arms from identical exact resets and RNG:

1. E023 parent only;
2. E026 globally active residual;
3. E026 deviation-gated residual.

Evaluate phases `0/25/50/75/100` on the pinned 125-state walk. Persist, for
every transition, phase, reward, done/terminal, all registered tracking and
termination errors, parent action, raw residual, gate, gated residual, sampled
action, effective action, qpos, and qvel.

The evaluator must independently recompute the gate and composed actions from
the raw trajectory. It must prove exact parent identity whenever the gate is
zero and exact agreement with the E026 global arm whenever the gate is one and
both arms occupy the same state. All assets and outputs are SHA-bound and
published manifest-last.

Render the gated phase-zero rollout side by side with the reference and create
a contact sheet. The visible scene contains only the policy robot, reference
robot, flat ground, and labels. The video is evidence only after numerical
validation passes.

## Outcomes

### `deviation-gating-advances`

The gated arm componentwise preserves the same-process E023 survival vector,
strictly improves at least one incomplete suffix, and does not worsen mean body
position/orientation error by more than five percent on any completed suffix.
Use this composition as the initializer for one separately registered
deviation-gated SHAC continuation on a longer continuous walk.

### `deviation-gating-solves-short-clip`

The gated arm completes all five finite suffixes, clears the tracking-metric
gate, and its phase-zero final ten frames remain below all termination
thresholds with non-increasing body-position error. Treat this only as a
short-clip solution; next require a longer continuous walk before robustness or
sim-to-real claims.

### `useful-correction-not-localizable`

The global E026 arm improves a suffix, but the gated arm either fails to improve
or regresses a protected suffix. Reject this scalar gate and move to a learned
correction/parent selector trained on ordinary and failure-state corpora.

### `correction-intrinsically-insufficient`

Neither global nor gated E026 improves the same-process E023 control under the
current evaluator. Stop composing E026 and train a new recovery module on
current-policy failure states instead of tuning this gate.

### `invalid-execution`

Any checkpoint, asset, identity, deterministic-reset, action, gate, telemetry,
artifact, or render contract fails. Repair execution only before scientific
interpretation.

## Testing and stopping boundary

Unit tests cover endpoint and interior gate values, nonfinite rejection,
broadcasting, exact-zero parent identity, exact-one global identity, action
recomposition, outcome precedence, componentwise preservation, terminal-tail
trend, provenance, and manifest-last validation. A CPU one-step smoke exercises
the real composite checkpoint before GPU execution.

Run exactly this frozen three-arm evaluation. Do not sweep gate thresholds,
train a policy, extend a budget, or reinterpret clip-end completion as stable
walking. The next branch is determined exclusively by the registered outcome.
