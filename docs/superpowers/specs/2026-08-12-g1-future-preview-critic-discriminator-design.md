# G1 Future-Preview Critic Discriminator Design

## Question

E021 and E022 improved carried-return calibration but both stopped near NRMSE
`0.31` on unseen trajectories. The actor observes three delta-reference
lookaheads at 4, 8, and 12 control steps, while the 286-dimensional critic sees
only the current reference and state. This experiment asks whether that missing
future context causes the remaining critic generalization floor.

## Paired treatment

Use the immutable 1,862-row E021 dataset and exact E012 checkpoint. Fit two
critics for exactly 1,640 continued-Adam full-dataset updates at `5e-4`:

- control: unchanged normalized 286-dimensional critic observations;
- treatment: the same observations plus the actor's normalized 174-dimensional
  delta-reference preview, producing a 460-dimensional input.

The treatment first-layer kernel is the control kernel followed by 174 exact
zero rows. Its Adam `mu` and `nu` receive the same zero-row migration, while the
count and every other leaf remain exact. Preview normalization reuses the exact
174-element suffix of E012's actor-frame normalizer. Before fitting, treatment
and control predictions must agree within `1e-6` on the complete dataset.

Stored dataset row phases are reconstructed as the clamped sequence
`min(start_phase + row_index, reference_length - 1)`, matching the source-step
environment's unit reference stride. Future commands come only from the pinned
reference through the environment's existing `_future_reference_command`.

## Evaluation and decision

Neither arm observes confirmation data during fitting. After both fits finish,
capture one shared, previously unused trajectory grid at phases
`15/115/215/315/415`. Evaluate the original target, current-only control, and
future-preview treatment on identical states and realized returns.

Select `future-preview-critic-advances` only if the treatment has rank
correlation at least `0.8`, NRMSE at most `0.25`, every H12 relative error at
most `0.25`, strictly better rank and NRMSE than the paired control and original
target, and no worse H12 error than the paired control at any phase. All inputs,
runtime assets, migrated shapes, initial equivalence, trajectories, metrics,
and optional output artifact are hash-bound in a manifest-last report.

If the treatment passes, publish only a standalone critic artifact and next
integrate the 460-dimensional critic representation into SHAC under a separate
experiment. If it fails, stop critic-representation fitting and return to the
actor objective/horizon branch. This experiment cannot itself improve the
actor or establish sim-to-real readiness.
