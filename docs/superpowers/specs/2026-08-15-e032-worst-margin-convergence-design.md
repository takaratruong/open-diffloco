# E032 Worst-Margin Convergence Design

## Question

Did E032 stop because the worst-margin recovery objective is structurally
insufficient, or because its fixed 64-update budget ended while the oracle was
still learning?

## Evidence and alternatives

E032 improved the number of terminal-free H32 starts from six to eight, reduced
its objective from 1.133580 to 0.993909, and retained a finite nonzero aggregate
gradient norm of 0.139876 at update 64. Its correction tape is not globally at
the registered 0.5 bound, although 24.49 percent of coordinates exceed 0.4.

Three plausible next tests are:

1. **Recommended: extend only the optimizer budget to 256 updates.** This is the
   smallest test of nonconvergence and preserves the exact scientific problem.
2. Increase H32 to H64. This is rejected for now because the current oracle
   masks objective terms after first terminal; starts failing before H32 receive
   no additional causal gradient from a nominally longer horizon.
3. Add a no-reset post-threshold dynamics path or widen correction authority.
   Either may ultimately be needed, but each changes a physical boundary and is
   justified only if the unchanged objective first plateaus.

## Design

Parameterize the existing oracle's update count with a positive integer CLI
argument whose default remains exactly 64. Thread that value through tape
optimization, summary telemetry, and protocol validation without changing the
environment, H32 rollout, independent tapes, worst-margin objective, Adam
learning rate, correction bound, per-start gradient clipping, carried states,
parent policy, or solver.

Register one successor experiment that runs from zero initialization for 256
updates. Starting from zero avoids an unrecorded Adam-state reset and makes the
only causal delta the number of optimizer updates. Record curve samples and the
same terminal-free survival vector as E032.

## Gates and decision

- Execution is valid only if all 256 updates, gradients, actions, termination
  errors, and artifacts are finite and corrections remain at or below 0.5.
- If all 24 starts complete H32, distill the resulting state-conditioned action
  targets into the recovery expert.
- If fewer than 24 complete but the final 64-update interval still materially
  improves survival or worst-margin loss, authorize one explicitly bounded
  continuation decision from the evidence rather than silently tuning.
- If the final 64-update interval is flat and starts still fail, stop optimizing
  this formulation. Inspect whether correction authority is binding, then test
  either a wider physically valid action correction or a no-reset post-threshold
  objective as a separately registered causal change.

## Tests

Add parser and execution tests proving the default remains 64, positive custom
budgets are accepted, nonpositive budgets fail closed, and the requested count
is used in both the loop and summary. Re-run the focused oracle tests, Ruff,
`py_compile`, and `git diff --check` before registering compute.
