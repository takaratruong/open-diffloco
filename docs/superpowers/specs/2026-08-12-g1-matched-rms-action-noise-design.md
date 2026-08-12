# G1 Matched-RMS Action-Noise Control Design

## Question

E-20260812-010 improved the deterministic five-phase robust key after replacing
the inherited scalar action-noise standard deviation `0.32` with the learned
29-dimensional RMR vector. The vector has RMS `0.25027265203867416`, so that
treatment changed both noise magnitude and per-joint shape. This control asks
whether the gain survives when every joint receives the same fixed standard
deviation with exactly that RMS.

## Immutable comparison

Resume the exact selected E008 TrainState at step `1,867,776`. Preserve the
E010 32-update, effective-512, H12, five-bin CAGrad, zero-bootstrap,
exact-zero-assistance, reset, domain-randomization, solver, model, reference,
seed, optimizer, actor, critic, checkpoint-cadence, and deterministic selection
contracts. Change only both action-noise endpoints to scalar
`0.25027265203867416`, the schedule endpoint to `2,064,384`, and the explicit
resume authority already required for that treatment change.

The scalar must be derived once from the pinned float32 RMR vector as
`sqrt(mean(float64(vector) ** 2))` and then frozen as a named constant. It is
not a tunable argument.

## Outcomes

- **anisotropy-advances:** the RMR-vector selected key exceeds the matched-RMS
  scalar key without a per-phase compensating collapse. Retain the per-joint
  shape as the active recipe ingredient.
- **magnitude-explains-gain:** the matched scalar equals or exceeds the RMR
  vector under the minimum/median/mean selector and registered phase floors.
  Retain lower noise magnitude; do not claim anisotropy.
- **both-positive-mixed:** both clear E008 but trade phases without one
  dominating. Treat lower magnitude as supported and anisotropy as unresolved.
- **invalid-execution:** any immutable execution, finite-training, or evidence
  gate fails. Repair only the execution defect.

Every suffix may still terminate; this discriminator identifies the useful
noise mechanism and does not claim complete tracking or transfer.
