# Isolated RMR Action-Noise Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and visualize one finite SHAC update under full-scale RMR action exploration with every other stochastic input disabled.

**Architecture:** Narrow the existing RMR action-space parity runner rather than changing the trainer or environment. Reuse the existing training-distribution episode logger to replay the exact checkpoint/noise contract and publish only the robot panel for visual inspection.

**Tech Stack:** Python, JAX/MJX, Flax, pytest, existing guarded experiment registry and Obsidian projection.

## Global Constraints

- Exact reference resets and clean actor observations.
- Nominal physics and no pushes or external assistance.
- Full-scale linear/unclipped RMR action boundary.
- Action noise is the sole stochastic treatment.
- One 6,144-transition gate precedes longer training.

---

### Task 1: Isolate the runner contract

**Files:**
- Modify: `tests/test_g1_rmr_action_space_parity_runner.py`
- Modify: `tools/run_g1_rmr_action_space_parity.py`

**Interfaces:**
- Consumes: `build_canonical_kwargs(...) -> dict`
- Produces: `build_rmr_action_space_parity_kwargs(...) -> dict` with `actor_observation_noise=False` and `reference_reset_noise_scale=0.0`.

- [ ] **Step 1: Write failing assertions** requiring clean observations, exact resets, and matching preflight and artifact-validation metadata.
- [ ] **Step 2: Run** `/home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q tests/test_g1_rmr_action_space_parity_runner.py` and confirm the new assertions fail on inherited noisy settings.
- [ ] **Step 3: Implement the minimal runner overrides and fail-closed validation fields.**
- [ ] **Step 4: Re-run the focused test and require all tests pass.**
- [ ] **Step 5: Run Ruff, py_compile, and `git diff --check`, then commit the scoped change.**

### Task 2: Reuse the episode evidence logger

**Files:**
- Add from reviewed commit: `tools/log_g1_training_episodes.py`
- Test with its existing focused logger tests.

**Interfaces:**
- Consumes: gate checkpoint, sibling `hparams.json`, seeded action-noise override.
- Produces: `training_rollout.mp4`, `training_action_noise.npz`, `contact_sheet.png`, and manifest.

- [ ] **Step 1: Integrate the existing logger commit without changing its behavior.**
- [ ] **Step 2: Run its focused tests and static checks.**
- [ ] **Step 3: Commit only the reusable logger integration.**

### Task 3: Register, execute, and inspect the one-update gate

**Files:**
- Create in lab repository: `research/experiments/E-20260813-001.yaml`
- Update after deterministic evaluation: `research/state/current.yaml`
- Generate: `docs/vault/Dashboard.md`, `docs/vault/experiments/E-20260813-001.md`

**Interfaces:**
- Consumes: clean committed code SHA, pinned model/controller/reference/solver.
- Produces: immutable run artifacts and one robot-only noisy training preview.

- [ ] **Step 1: Register one causal experiment with pass/fail/invalid outcomes and exact provenance.**
- [ ] **Step 2: Run registry validation and dry-run.**
- [ ] **Step 3: Execute the guarded one-update experiment on one free GPU.**
- [ ] **Step 4: Validate finite CAGrad/update evidence and render the seeded training-distribution episode.**
- [ ] **Step 5: Copy the robot-only preview into the untracked Obsidian preview directory and open it.**
- [ ] **Step 6: Curate the registered result, validate, rebuild the vault, and commit owned registry files.**

