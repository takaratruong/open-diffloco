# G1 Zero-Bootstrap Continuation Design

## Question

Does the demonstrably miscalibrated terminal critic help or harm the E012
actor update? Resume the exact E012 selected checkpoint at step `1,671,168`
and run 32 additional effective-512 H12 CAGrad
updates while changing only `actor_bootstrap_scale` from `1.0` to `0.0`.

## Resume boundary

Add one explicit `allow_resume_actor_bootstrap_scale_change` authority to SHAC.
Without it, a resumed run must restore the checkpoint's scale and reject a
mismatch. With it, a finite nonnegative requested scale is retained and written
to hparams. The flag changes no parameters or optimizer state at migration.

The immutable runner pins the exact E012 selected checkpoint/hparams, reference,
model, controller, clean code commit, solver, seed, and all inherited settings.
It ends at step `1,867,776`, exactly 196,608 transitions or 32 updates after
the selected checkpoint, with checkpoints every 49,152 transitions at
`1,720,320/1,769,472/1,818,624/1,867,776`. Assistance remains exactly zero.

## Decision

Evaluate all four checkpoints at phases `0/100/200/300/400`, select by maximum
minimum survival, then median, mean, and earliest step. Compare against E012's
selected `66/61/89/51/76` vector. The first treatment checkpoint at step
`1,720,320` is also an exact same-start/eight-update comparison with E012's
scale-one final checkpoint at that step. Select `zero-bootstrap-advances` only
if the treatment strictly improves
E012's minimum-first key and does not regress any phase by more than five
transitions. A finite miss rejects bootstrap removal. Invalid provenance,
nonzero assistance, wrong scale/update count, nonfinite telemetry, or incomplete
evaluation is not scientific evidence.

This is a bounded actor-objective discriminator. It does not claim that a
bootstrap-free H12 objective is the final method; a positive result motivates a
separate multihorizon recipe, while a negative result motivates direct
value-gradient alignment measurement.
