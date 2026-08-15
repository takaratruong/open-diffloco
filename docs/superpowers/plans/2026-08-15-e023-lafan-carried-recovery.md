# E023 LAFAN Carried-Recovery Implementation Plan

**Goal:** Transfer the validated E026 recovery architecture to the pinned
10-second LAFAN reference without transferring E026's non-generalizing
residual parameters.

## Task 1: Explicit reference-path resume authority

- Add focused RED tests to `tests/test_shac_exact_resume.py`.
- Add a fail-closed resolver and `allow_resume_reference_path_change` to
  `src/algorithms/shac/algorithm.py`.
- Persist parent/requested path hashes and a migration artifact.
- Verify legacy resume equivalence and the focused resume/CAGrad/anchor suite.

## Task 2: LAFAN pre-failure bank

- Add `tools/build_g1_e023_lafan_carried_reset_bank.py` and focused tests.
- Reuse the existing context-faithful collector at starts 0/100/200/300/400.
- Require 120 rows, exact E023 parent, pinned LAFAN reference, clean code, and
  atomic SHA-bound output.
- Build the real bank and validate its observed survivals against the paired
  zero-shot grid.

## Task 3: Registered continuation runner

- Add `tools/run_g1_e023_lafan_anchored_carried_recovery.py` and focused tests.
- Pin 64 H24 adapter-only updates, checkpoints every 8, carried probability
  0.25, anchor 1.0, and explicit reference migration authority.
- Validate all archives, reference migration, zero-effect residual migration,
  frozen parent/moments/normalizer, positive finite adapter updates, exact
  action noise, zero bootstrap, and five occupied valid CAGrad bins.

## Task 4: Execute and curate

- Register one seed-zero E027 experiment and run the full validated job.
- Evaluate updates 8/16/32/64 at phases 0/100/200/300/400.
- Apply componentwise floors 118/63/49/39/46, render selected clean/noisy
  episodes, generate learning/action diagnostics, and update the vault.
- Retain E023 if no checkpoint is eligible.

