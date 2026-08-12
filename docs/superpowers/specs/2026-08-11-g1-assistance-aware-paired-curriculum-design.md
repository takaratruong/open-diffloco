# G1 Assistance-Aware Paired Curriculum Design

## Objective

Determine whether telling the trainable residual policy the scalar magnitude of
training-only torso assistance improves the final, exact-zero-wrench policy.
The experiment is a paired comparison, not a deployment-assistance proposal.

## Parent and arms

Both arms resume the exact E012 selected checkpoint at step 1,671,168 (SHA-256
`f375cadc9bf8b5cef26fc7414133071910fed393344c99bbacffea963aa9f4f7`). Both
receive identical random streams, continuous per-environment assistance
magnitudes, zero-assistance cohort, optimizer, update budget, and checkpoint
cadence.

- **Aware:** the residual adapter receives the true scalar assistance magnitude.
- **Blind:** the same added scalar input is always exactly zero, although the
  physical environment receives the same assistance as the aware arm.

The arm-to-arm causal change is therefore assistance observability alone.

## Policy boundary and migration

Only the trainable residual adapter is conditioned. The frozen parent actor,
its input, the actor observation history, and the actor normalizer remain bit
exact. The scalar is concatenated to the newest normalized 328-dimensional
frame immediately before the residual adapter, producing a 329-dimensional
adapter input. It is already bounded to `[0, 1]` and is not normalized.

Migration appends one exact-zero row to the residual adapter's first kernel and
to the corresponding Adam first- and second-moment arrays. All existing
parameters, optimizer counts, frozen-parent values, and normalizer values must
remain exact. Consequently both migrated arms reproduce the E012 action exactly
when the scalar is zero. Evaluation and export default the new input to exact
zero.

No wrench vector, torso error, oracle target, or other privileged quantity is
exposed to the policy.

## Assistance distribution and schedule

At every assisted update, 25% of environments receive exact zero assistance.
The remaining environments draw one scale per unroll uniformly from
`[0, scheduled_cap]`. The cap decays linearly and continuously from one to zero.
Both paired arms consume the same sampled scales. A final exact-zero assistance
tail permits consolidation and is the only region eligible for model selection.

The immutable paired runners define the exact start, fade endpoint, final
endpoint, random seed domains, and output roots. Neither runner exposes
scientific command-line overrides.

## Evidence and selection

Every periodic checkpoint in the final zero-assistance tail is evaluated on the
same replay-free five-phase grid. Assisted training reward never selects a
checkpoint. Primary comparison is lexicographic:

1. minimum five-phase survival;
2. median five-phase survival;
3. mean five-phase survival.

The aware condition is useful only if it exceeds the blind arm and the E012 key
`51/66/68.6` without a compensating collapse in another phase. Full success is
`499/399/299/199/99` exact-zero-wrench survival. All conclusions remain limited
to simulation and this reference motion.

## Failure handling

Training fails closed on nonfinite/out-of-range scalar inputs, conditioning
migration drift, assistance telemetry failure, empty zero-assistance selection
windows, checkpoint provenance mismatch, or unequal paired contracts outside
the declared observability field. No result is promoted without trajectory,
metrics, video, and montage artifacts from deterministic evaluation.
