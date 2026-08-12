# G1 Future-Preview Critic Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine causally whether the actor's existing future-reference preview closes the held-out carried-return critic calibration floor.

**Architecture:** Add one experiment-specific paired evaluator. It reconstructs future commands for the immutable E021 rows, performs a zero-effect critic/Adam input migration, fits unchanged and augmented critics under identical budgets, then evaluates both on one shared unseen phase grid. It does not modify the environment or SHAC training path.

**Tech Stack:** Python, JAX, Flax, Optax, NumPy, pytest, Ruff.

## Global Constraints

- Exact E012 checkpoint, E021 dataset, reference, model, controller, solver, seed, and clean code commit are mandatory.
- The control and treatment each execute exactly 1,640 full-dataset updates at learning rate `5e-4`.
- Confirmation phases are fixed at `15/115/215/315/415` and cannot affect fitting.
- No actor, environment, or SHAC training code changes are authorized.
- The report is written atomically after every required artifact is validated.

---

### Task 1: Pure future-preview augmentation and migration

**Files:**
- Create: `tools/compare_g1_future_preview_critic.py`
- Test: `tests/test_g1_future_preview_critic.py`

**Interfaces:**
- Consumes: E021 phase-keyed dataset, `env._future_reference_command`, E012 critic parameters/optimizer and normalizers.
- Produces: `row_phases(start_phase, count, reference_length)`, `augment_critic_observations(...)`, and `migrate_critic_input(...)`.

- [ ] **Step 1: Write failing tests** for clamped row phases, exact command ordering, 286-to-460 first-kernel migration, exactly two Adam moment expansions, unchanged optimizer count/leaves, and initial prediction equivalence.
- [ ] **Step 2: Run RED** with `conda run -n diffsim pytest -q tests/test_g1_future_preview_critic.py` and confirm failures are missing interfaces.
- [ ] **Step 3: Implement minimal pure helpers** using zero-row append-only migration and the actor normalizer's final 174 entries.
- [ ] **Step 4: Run GREEN** with the same focused command and require zero failures.

### Task 2: Paired fitting, evaluation, and fail-closed report

**Files:**
- Modify: `tools/compare_g1_future_preview_critic.py`
- Modify: `tests/test_g1_future_preview_critic.py`

**Interfaces:**
- Consumes: Task 1 augmented arrays and exact existing calibration helpers.
- Produces: `fit_paired_critics(...)`, `future_preview_advances(...)`, CLI output `future_preview_critic_report.json`, and optional `future_preview_critic.pkl`.

- [ ] **Step 1: Write failing tests** proving equal update counts, absolute gates, paired no-regression behavior, malformed provenance rejection, and manifest-last optional artifact binding.
- [ ] **Step 2: Run RED** and confirm each new test fails for the intended missing behavior.
- [ ] **Step 3: Implement the paired evaluator** with one shared confirmation capture after fitting and no confirmation-dependent selection.
- [ ] **Step 4: Run GREEN**, Ruff, `py_compile`, and `git diff --check`.

### Task 3: Review, register, and execute

**Files:**
- Create in lab repository: `research/experiments/E-20260812-007.yaml`
- Modify after execution: experiment result, hypothesis evidence, current state, generated vault projections.

**Interfaces:**
- Consumes: clean reviewed code commit and pinned E012/E021 artifacts.
- Produces: one guarded seed-zero report and a curated branch decision.

- [ ] **Step 1: Self-review** the diff against the design, then request one bounded independent code/scientific review.
- [ ] **Step 2: Fix verified blockers**, rerun focused checks, and commit a clean code revision.
- [ ] **Step 3: Register E-20260812-007**, validate the registry, dry-run the exact command, and approve it under Codex's standing simulation authority.
- [ ] **Step 4: Execute** `uv run python -m tools.runexp E-20260812-007`, validate all artifacts, and select only a preregistered outcome.
- [ ] **Step 5: Curate and commit** the experiment result, state, hypothesis edge if warranted, and generated Obsidian projections while preserving user-owned settings/canvas files.
