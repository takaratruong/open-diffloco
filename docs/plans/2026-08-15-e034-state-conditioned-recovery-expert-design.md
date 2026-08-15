# E034 State-Conditioned Recovery Expert Design

## Question

Can the successful E034 action-sequence corrections be represented by the
existing small state-conditioned residual adapter and reproduce recovery in
closed loop from the same 24 registered E023 failure states?

## Fixed inputs

- Freeze the E023 parent actor and normalizer exactly.
- Consume only the hash-bound E036 dataset.
- Train only on the 13 explicit successful rows (416 transitions).
- Use seed 0, the `g1-4x5` solver, and the registered reference and carried
  reset bank.

## Expert

Use the existing `PreviewResidualAdapter`: newest normalized 328-D frame,
one 256-unit ELU hidden layer, and a tanh-bounded 29-D correction. The parent
actor remains immutable. This deliberately tests the same compact residual
interface already supported by SHAC rather than adding another architecture.

Train full-batch Adam for 2,000 updates at learning rate 1e-3. Minimize mean
squared correction error plus an effective-action auxiliary loss between
`clip(parent + predicted_correction, -1, 1)` and the teacher effective action.
Select the minimum finite training loss because this is a representability
test, not a generalization claim.

## Closed-loop evaluation

Reconstruct the exact 24 initial states. At every step, normalize the current
history with the frozen E023 normalizer, evaluate the frozen parent and expert,
add their actions, and pass the raw sum to the environment (which applies the
registered action boundary). Persist survival, actions, rewards, termination
errors, fitted parameters, and the learning curve.

## Outcomes

- `state-conditioned-recovery-reproduced`: at least 10 of the 13 E034 teacher
  successes reach H32 and every start is no worse than the E023 baseline.
- `state-conditioned-recovery-partial`: more H32 successes than the E023
  baseline but the reproduction gate is not met.
- `state-conditioned-recovery-insufficient`: no increase in H32 successes.
- `invalid-execution`: any provenance, finite-value, shape, boundary, or
  artifact gate fails.

No SHAC integration, hyperparameter sweep, or claim beyond these 24 registered
states is permitted by this experiment.
