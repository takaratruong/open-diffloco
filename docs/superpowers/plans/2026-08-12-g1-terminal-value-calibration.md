# G1 Terminal-Value Calibration Implementation Plan

**Goal:** Measure whether the exact E012 target critic supplies an accurate
longer-carried-return terminal value at the H12 actor boundary.

**Architecture:** Add a small evaluator that reuses the production environment,
actor/residual loading, critic network, normalizers, and solver profile. Keep
metric computation pure and separately tested; keep the executable evaluation
read-only and manifest-last.

## Task 1: Pure calibration metrics

- Create `tests/test_g1_terminal_value_calibration.py` with failing tests for
  discounted realized returns, Pearson/rank correlation, NRMSE, H12 extraction,
  invalid finite/length inputs, and outcome classification.
- Create `tools/evaluate_g1_terminal_value_calibration.py` and implement only
  the pure functions needed to make those tests pass.
- Run the focused test and Ruff/compile checks.

## Task 2: Production checkpoint evaluation

- Load the frozen residual checkpoint and require target critic parameters plus
  actor and critic normalizers.
- At each registered phase, record pre-step target-critic value, reward, done,
  terminal, and exact-zero applied wrench during an ordinary policy rollout.
- Compute realized terminal returns backward with gamma `0.99`; require natural
  terminal completion and finite arrays.
- Bind code/checkpoint/reference/model/controller/solver provenance and write
  one atomic JSON after every gate passes.
- Add parser/payload/provenance tests and run the affected focused suite.

## Task 3: Review, register, and execute

- Self-review the complete diff for evaluator/trainer mismatch and fail-open
  provenance.
- Commit a clean code SHA, register one seed-zero GPU diagnostic with the exact
  E012 selected checkpoint, validate the registry, and launch it once.
- Curate the result before authorizing any critic or actor training.
