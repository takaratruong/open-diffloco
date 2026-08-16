# G1 Motion-Anchor Position Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with test-first checkpoints.

**Goal:** Run a bounded E023-derived SHAC experiment that adds only RMR's body-frame `motion_anchor_pos_b` actor feature and reports matched update-16/32 replay-free behavior.

**Architecture:** Add an optional, default-off three-value field at the G1 actor-observation seam. Thread its immutable contract through SHAC hparams and the two generic evaluators. Derive one experiment runner from E023, changing only the observation flag plus execution metadata. Reuse the existing trainer, five-phase evaluator, diagnostics, renderers, and registry pipeline.

**Tech Stack:** Python 3.11, JAX/MJX, Flax, NumPy, pytest, Ruff, DiffSim2Real registry/runexp tooling.

## Global Constraints

- The signal is exactly `R(q_actual)^T * (p_reference - p_actual)` from the existing motion anchor.
- False mode remains exactly 328 values/frame; true mode is 331 values/frame and 3,310 values for history ten.
- Add no root velocity, absolute position, noise, reset perturbation, push, reward, optimizer, action, or solver change.
- Preserve E023's H24/effective-512/five-bin-CAGrad recipe and action-noise endpoint `1,572,864`.
- Stop at update 32 with archives at updates 16 and 32.
- Never mutate E023 artifacts or unrelated dirty files.

---

### Task 1: Add the optional actor observation test-first

**Files:**
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_environment.py`

- [ ] Write RED tests proving false-mode legacy shape/value identity, true-mode exact feature ordering/value, 331-frame/3310-history dimensions, zero-noise mask alignment, and boolean validation.
- [ ] Run the focused environment tests and record the expected missing-interface failures.
- [ ] Add `actor_observe_motion_anchor_position: bool = False`; reuse `_anchor_relative_reference`; concatenate position immediately before orientation only when enabled; make dimensions and noise mask conditional.
- [ ] Run the focused environment and reset/history neighbors; require GREEN.
- [ ] Commit only the environment and its tests.

### Task 2: Bind training and evaluation provenance

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify focused tests for these modules.

- [ ] Write RED tests for trainer-to-environment forwarding, hparam persistence, expected 3,310 input guard, fail-closed resume mismatch, evaluator construction, checkpoint-contract loading, and payload reporting.
- [ ] Run focused tests and record RED.
- [ ] Thread the flag through `train`, G1 environment kwargs, hparams, evaluator CLI/constructor, and phase-grid contract/payload. Treat a missing resume hparam as legacy false and reject any requested mismatch.
- [ ] Run trainer/evaluator/environment focused tests and static checks; require GREEN.
- [ ] Commit only the provenance plumbing and tests.

### Task 3: Add the bounded E023-derived runner

**Files:**
- Create: `tools/run_g1_motion_anchor_position_h24_walk.py`
- Create: `tests/test_g1_motion_anchor_position_h24_walk.py`

- [ ] Write RED tests for the exact scientific delta, 32-update budget, unchanged noise endpoint, expected input dimension, two archive steps, fail-closed asset/code preflight, exhaustive outcome classification, and checkpoint selection.
- [ ] Implement the smallest runner by reusing E023's builder and existing training-artifact validator. The treatment builder sets `actor_observe_motion_anchor_position=True`, `expected_actor_obs_dim=3310`, and `total_steps=393216` only.
- [ ] Use registered outcomes `root-position-early-{advances,parity,mixed,underperforms}` and first-four-phase lexicographic selection.
- [ ] Run runner, parent-runner, trainer, evaluator, and environment focused tests plus Ruff/pycompile/diff-check.
- [ ] Commit only the runner and tests.

### Task 4: Independent review and launch gate

- [ ] Invoke `requesting-code-review` on the complete bounded diff. Repair every Critical/Important issue test-first and re-review until launch-ready.
- [ ] Run fresh verification from a clean code worktree and record exact outputs.
- [ ] Register `E-20260816-004` with parent `E-20260814-023`, pinned clean commit/assets/seed, required artifacts, registered outcomes, one GPU, and a bounded time budget.
- [ ] Run registry validation and `uv run python -m tools.runexp E-20260816-004 --dry-run` before any GPU process.

### Task 5: Execute, evaluate, visualize, and curate

- [ ] Inspect device availability, then launch exactly one registered run with `uv run python -m tools.runexp E-20260816-004`.
- [ ] Validate the two immutable archives and finite five-bin CAGrad telemetry.
- [ ] Evaluate phases `0 25 50 75 100` at updates 16 and 32, classify against the matched E023 vectors, and select by the registered key.
- [ ] Render the selected clean phase-zero rollout plus contact sheet and generate learning/action diagnostics.
- [ ] Publish manifest-last evidence, update the experiment record and current research state, rebuild the Obsidian vault/dashboard with newest-first ordering, and ensure MP4/PNG embeds resolve.
- [ ] Run registry tests, static checks, and artifact reopen validation before claiming a result.

## Stop Rule

Stop after update 32 and curate the result. Do not tune this treatment inside E004. The next experiment must follow from the registered outcome rather than an untracked sweep.
