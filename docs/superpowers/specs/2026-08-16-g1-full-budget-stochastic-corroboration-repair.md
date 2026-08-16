# G1 Full-Budget Stochastic-Corroboration Repair

## Evidence

E005 used the reviewed full-budget E023-matched treatment, but its update-16
replay-free vector was `42/36/49/47/24` instead of E004's
`43/38/48/49/24`. The runner kwargs differ only in execution length, and all
learning schedules that affect the first 16 updates are otherwise identical.
The small behavioral drift therefore demonstrates that a fresh GPU SHAC run is
not bitwise deterministic even at fixed seed; it does not demonstrate a
scientific treatment or provenance change.

## Decision

Close E005 as invalid because its preregistered exact behavioral-corroboration
gate cannot pass reliably. Register one fresh execution-only successor with the
same treatment and full budget. Keep updates 16 and 32 as descriptive learning
diagnostics, but never use their equality to another stochastic training run as
an execution gate.

The corrected classifier consumes exactly updates 64 and 128. Update 64 remains
diagnostic. Update 128 alone is compared to E023 `116/99/67/49/24` using the
unchanged advance/parity/mixed/underperform ordering. Provenance, exact kwargs,
observation shape, eight archives, finite gradients, CAGrad, action schedules,
and evaluator artifacts remain fail-closed execution gates.

## Stop Rule

Run one corrected fresh seed-zero treatment through update 128, evaluate and
curate it, and do not tune or add another scientific change in this successor.
