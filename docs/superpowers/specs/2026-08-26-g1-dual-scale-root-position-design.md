# G1 Dual-Scale Root-Position Reward Design

## Question

Does restoring a useful root-position gradient at large tracking error improve
the retained E002 walking policy without sacrificing any of its five replay-free
phase suffixes?

## Treatment

Keep the complete E002 update-8 training contract unchanged except for the
anchor-position reward kernel. The legacy component is

`exp(-(e / 0.3)^2)`.

The treatment is the normalized mixture

`0.75 * exp(-(e / 0.3)^2) + 0.25 * exp(-(e / 0.8)^2)`.

The existing outer anchor-position weight remains 0.5, so the maximum reward
and all other reward weights are unchanged. Root-velocity weight remains 1.0.

## Causal comparison

Resume two matched arms from the exact retained E002 update-8 checkpoint. The
control retains the legacy kernel and the treatment changes only the kernel.
Both arms continue the existing residual adapter for 16 H24 effective-512
updates, with checkpoints after updates 4, 8, 12, and 16. This avoids adding a
new model-capacity treatment while directly measuring the reward change.

Evaluate the source and every checkpoint at phases 0, 25, 50, 75, and 100 using
the same replay-free CPU evaluator. A treatment checkpoint is eligible only if
it componentwise preserves E002 survival `[136, 144, 84, 90, 79]` and strictly
improves at least one phase. The paired result must also beat the best eligible
control checkpoint under the same lexicographic minimum/median/mean ordering.

## Fail-closed boundaries

- The legacy kernel remains the exact default code path.
- A resumed kernel change requires explicit authority and persisted metadata.
- Training, evaluation, and rendering restore the kernel from checkpoint
  hparams.
- Checkpoint, reference, code, solver, and phase-grid provenance remain bound.
- No sweep or post-observation threshold change is permitted.
