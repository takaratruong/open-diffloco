# Frozen-Preview Termination-Margin Design

## Problem

The selected frozen-preview actor improves replay-free phase survival, but its
differentiable actor loss is nearly uncorrelated with that survival. CAGrad
balances phase-local gradients; it does not make a dense tracking objective
care about approaching the actual termination boundary.

## Treatment

Resume the selected E011 midpoint and preserve its frozen legacy actor,
+4/+8/+12 preview columns, five-bin CAGrad, H12, effective batch 512, simulator,
randomization, optimizer state, and source normalization. Change only the
environment reward by enabling the already implemented margin penalty at weight
0.5. The penalty is zero while all four termination errors are below half their
threshold, rises quadratically between half-threshold and threshold, and is
capped beyond the threshold.

Continue for 196,608 transitions and archive every 49,152 transitions. This
reaches the exact same global step as E011's zero-margin final checkpoint, so
the primary comparison has matched training exposure. Select only by the same
dense replay-free five-phase survival rule; training reward and margin loss are
diagnostic, not checkpoint authority.

## Why this treatment

- It directly addresses objective mismatch without changing the policy class.
- It preserves gradients before the hard terminal instead of differentiating a
  boolean termination event.
- Weight 0.5 is bounded: one threshold at its limit contributes -0.5 before the
  existing reward scale, while the six positive tracking terms retain their
  original relative weights.
- Starting from the selected midpoint separates objective alignment from the
  full-network interference already removed by the frozen adapter.

## Stop conditions

Stop on resume mismatch, frozen parameter or optimizer-moment drift, nonfinite
state or telemetry, incomplete checkpoint set, or missing replay-free phase
evaluation. A finite run with no behavioral gain rejects this fixed margin
treatment; do not tune its weight post hoc.
