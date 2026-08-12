# G1 Carried-Return Critic Refit Implementation Plan

**Goal:** Produce a provenance-bound E012 checkpoint whose actor is unchanged
and whose critic is calibrated on held-out longer carried returns.

## Task 1: Pure dataset and selection contracts

- Add tests for immutable phase splits, backward terminal returns, padded
  trajectory extraction, calibration selection, and exact non-critic tree
  preservation.
- Implement the pure functions in a focused refit tool and make the tests pass.

## Task 2: Vectorized replay-free capture and critic-only fit

- Load the exact E012 checkpoint and production actor/critic/environment.
- Capture all 30 trajectories in one vmapped fixed-solver rollout, retaining
  only each trajectory's first natural-terminal prefix.
- Evaluate original target/current critics, continue only the critic Adam state
  for 2,000 fixed full-batch steps, select on validation, and evaluate test once.
- Save the selected TrainState, copied resume hparams with an explicit refit
  sidecar, raw dataset NPZ, metrics JSON, and manifest atomically.

## Task 3: Verify, review, register, execute

- Run focused tests, Ruff, compilation, and diff checks; independently review
  checkpoint mutation, data leakage, rollout semantics, and provenance.
- Register one guarded GPU refit only from a clean reviewed commit.
- Curate the result before registering any actor continuation.
