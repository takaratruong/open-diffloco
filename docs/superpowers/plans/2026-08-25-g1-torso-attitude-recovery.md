# G1 Torso-Attitude Recovery Implementation Plan

## Task 1: Reward and environment contract

Add failing tests for a direct torso pseudo-Huber orientation term, exact
zero-weight legacy identity, invalid weights, aligned torso indexing, metrics,
and environment forwarding.  Implement the smallest reward/environment
plumbing that makes them pass.

## Task 2: SHAC resume contract

Add failing tests for fresh/default behavior, exact resume, unauthorized
change rejection, missing resume metadata rejection, explicit change
authority, hparams persistence, and environment construction.  Implement the
new `tracking_torso_orientation_weight` and
`allow_resume_tracking_torso_orientation_change` boundary.

## Task 3: Bounded E026 continuation

Add a runner that pins the exact E026 update-16 checkpoint and sibling hparams,
the carried bank, model, controller, reference, solver, seed, 32-update budget,
four checkpoint steps, and the sole objective delta.  Reuse existing phase-grid
evaluation seams and add pure selection tests for survival preservation and
tail-pitch reduction.

## Task 4: Verification and execution

Run focused reward/environment/resume/runner tests, Ruff, py_compile, and diff
checks.  Review the exact diff.  Commit clean code, register one experiment,
run it on one available GPU, validate all artifacts, render the selected arm
only, publish its video and learning diagnostics, and update the research
state.
