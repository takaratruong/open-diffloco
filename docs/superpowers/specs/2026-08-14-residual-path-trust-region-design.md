# Residual-Path Behavioral Trust-Region Design

## Objective

Test whether a non-regressive policy exists on either straight parameter path
from the retained E013 H24 residual actor to E014's best scalar-aware or blind
actor. This is an evaluation-only discriminator; it does not train, change
physics, or claim that interpolation is itself a final learning algorithm.

## Why This Test

E013's five CAGrad task gradients all have positive inner product with the
combined update at both selected checkpoints, yet later replay-free behavior
still trades phase 400 for phases 0/200/300. A stricter local common-descent
solver therefore does not target the observed failure. A behavioral path test
directly asks whether smaller movement along already useful learned directions
can preserve every retained phase.

## Policy Construction

The source and both targets have bit-identical frozen parent actors and actor
normalizers. The source adapter has a 328-row input kernel. E014 targets have a
329th assistance-scalar row; exact-zero deployment never uses that row. Remove
only that row, verify exact-zero target actions remain bit-identical, then form
each adapter leaf as `source + alpha * (target - source)` for alpha
`0.125, 0.25, 0.5, 0.75`. Preserve the source parent, normalizer, critic,
environment state, and optimizer state in diagnostic checkpoint artifacts.

The builder fails closed on non-identical parent/normalizer trees, unexpected
adapter structure or shape, alpha outside `[0,1]`, nonfinite values, or failure
of endpoint action parity.

## Evaluation And Decision

Evaluate all eight diagnostic checkpoints with the existing corrected Flax
phase-grid evaluator: phases `0/100/200/300/400`, seed zero, exact assistance
scalar zero, post-policy `[-1,1]` clip, compiled `env.step`, delta preview, and
`g1-4x5`.

The retained E013 vector is `44/65/67/80/53`. A candidate is safe only if it is
componentwise no worse than that exact vector and its minimum/median/mean key
is strictly greater. Select by key, then lower alpha, then aware before blind.
If no candidate is safe, parameter interpolation cannot cheaply turn the
existing phase-specialized actors into a robust parent, and the next treatment
must enforce replay-free behavioral acceptance during optimization.

## Artifacts And Tests

The builder writes hash-bound diagnostic checkpoints plus one manifest with
source/target hashes, arm, alpha, parent/normalizer equality, zero-scale
endpoint parity, and output hashes. Unit tests cover exact endpoints, the
329-to-328 zero-scale projection, interpolation arithmetic, invalid inputs,
and deterministic manifest ordering. Existing evaluator tests cover the
compiled/clipped rollout boundary.
