# RMR Frozen-Preview Walk Design

## Goal

Test the verified frozen current-preview mechanism on the independent default
walking motion while preserving the already competent RMR/PPO parent exactly.
Differentiable physics may train only newly appended future-reference input
weights; it may not change the legacy actor, source normalizer, or any legacy
optimizer moment.

## Scientific comparison

1. Evaluate the untouched RMR/PPO actor replay-free from five evenly spaced
   exact walking phases.
2. Append the existing +4/+8/+12 reference command to the one-frame 154-value
   RMR observation, giving 328 inputs.
3. Extend only the first RMR layer from 154 to 328 inputs. Copy all 154 legacy
   columns exactly and initialize all 174 preview columns to zero.
4. Freeze every parameter except those 174 preview columns, and freeze both
   legacy and preview normalization statistics after deriving finite preview
   statistics from the reference.
5. Train those columns with the existing H12, effective-512, five-bin CAGrad,
   fixed 4x5 MJX contract. Archive checkpoints every eight updates.
6. Select the checkpoint only by replay-free five-phase behavior, with the
   untouched parent as the non-regression authority.

## Code boundary

- Extend `preview_adapter.py` with RMR-policy migration and a policy-layout mask.
- Permit `actor_preview_adapter` with a one-frame `initial_full_actor_policy`.
  Keep the existing ten-frame Flax path unchanged.
- In the full-RMR action branch, compute candidate action from the complete
  observation and parent action from the same observation with only the new
  preview suffix zeroed.
- Add a fixed runner for the walking treatment and a generic, no-render,
  replay-free RMR phase-grid evaluator.

## Invariants

- Initial candidate actions equal parent actions to numerical tolerance.
- The legacy first-layer columns and every non-first-layer value are exact.
- Only first-layer preview columns and their Adam moments may change.
- Source mean/std values are exact; appended statistics are finite with positive
  standard deviation.
- The original Flax frozen-preview tests and behavior remain unchanged.
- Training loss never selects a checkpoint.

## Stop conditions

- Stop before training on migration/action inequivalence, non-finite state, or
  a parent that is not competent on the registered phase grid.
- Stop the branch if no archived candidate preserves the parent-relative phase
  floors. Do not compensate by unfreezing the parent.

