# E033 Recovery Action-Authority Design

## Question

Are the fourteen remaining E033 recovery failures caused by the oracle's
artificial 0.5 correction-tape bound?

## Evidence

E033 improves twelve starts when trained longer, but 38.70 percent of all tape
coordinates exceed magnitude 0.45, 28.65 percent exceed 0.475, and 16.49 percent
exceed 0.49. The maximum is 0.49916094. Every remaining failure crosses the
normalized anchor-height termination criterion. This is direct evidence that
the tape parameterization is pressing against its correction limit.

## Design

Expose the oracle correction bound as a positive finite CLI argument with the
existing 0.5 default. Register one successor that changes only the bound from
0.5 to 1.0 and preserves E033's 256 updates, independent H32 tapes,
worst-margin objective, zero initialization, Adam learning rate, gradient clip,
states, policy, solver, seed, and environment.

The environment continues to apply its existing normalized action clip to
`[-1, 1]` immediately before PD target construction. Therefore this treatment
does not grant invalid deployed action authority; it only lets the correction
move the frozen parent's action across the full existing action domain.

## Gates

- Require exactly 256 finite updates and a finite correction tape at or below
  the requested 1.0 bound.
- Require the existing final environment action clip to remain enabled and
  unchanged.
- Evaluate the exact same 24 carried starts for H32.
- If all 24 recover, distill the action targets into a state-conditioned expert.
- If recovery improves but remains incomplete, inspect effective-action
  clipping and convergence before choosing one final bounded oracle test.
- If recovery is no better than E033, reject correction authority as the main
  bottleneck and implement a no-reset post-threshold objective.

No action-bound sweep, policy training, reset change, reward change, or horizon
change is in scope.
