# G1 Full-Policy Duration Comparison Design

## Purpose

Select the earliest useful differentiable fine-tuning duration without using
training reward and without allowing separate-process MJX contact variation to
confound small controller differences.

## Architecture

Add a dedicated
`tools/compare_g1_tracking_full_policy_checkpoints.py` command. It constructs
the validated G1 evaluation environment and source RMR policy once. For each
registered reference phase it rolls out the source once, then rolls out every
ordered full-actor checkpoint in the same process and environment. The existing
single-checkpoint comparator remains unchanged.

The output records source summaries, every checkpoint's phase and aggregate
summaries, full-minus-source deltas, and one deterministic selection. Selection
uses input order as training-duration order and chooses the first candidate
with no more terminals than source, positive aggregate reward delta, and at
least four negative deltas among anchor position/orientation, body
position/orientation, and body linear/angular velocity errors. If no candidate
passes, the selected checkpoint is null and all candidate evidence remains
available.

## Interfaces And Evidence

The CLI accepts repeated ordered paths through `--checkpoints`, the source
checkpoint, output path, phases, seed, maximum steps, and solver budgets.
Checkpoint labels are their resolved paths and serialized training steps.
Malformed phase indices, duplicate checkpoint paths, absent checkpoints, and
non-finite output fail before publication. JSON is written only after every
rollout and selection completes.

## Testing

Focused unit tests cover ordered multi-checkpoint parsing and the pure earliest
passing selection rule, including rejection for terminals, nonpositive reward,
and fewer than four improved error groups. Existing full-policy comparison and
evaluator tests must remain green. The scientific execution is separately
preregistered and must use exact checkpoint hashes.
