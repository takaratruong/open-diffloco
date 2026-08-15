# E023 Anchored Carried-Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one preregistered E023 continuation that freezes the selected actor, trains a zero-effect recovery adapter from exact and history-faithful pre-failure states, and retains only componentwise replay-free improvement.

**Architecture:** Extend the existing residual-preview resume migration with one explicit plain-actor start authority, reuse the existing frozen adapter and policy-anchor machinery, and add an E023-specific carried-bank collector and guarded continuation runner. Keep canonical evidence in `diffsim2real-lab/runs/`; project media through the existing registry-driven Obsidian dashboard.

**Tech Stack:** Python 3.11, JAX/MJX, Flax, Optax, NumPy, pytest, Ruff, YAML research registry.

## Global Constraints

- E023 `checkpoint_step_1572864.pkl` is the immutable parent and retained control.
- Only the residual adapter may update; parent parameters/optimizer and actor normalizer must have exact zero drift.
- Use H24, effective population 512, CAGrad, action-noise endpoint vector, zero bootstrap, nominal physics, exact fallback resets, and fixed `g1-4x5` solver.
- Use carried probability 0.25, anchor weight 1.0, at most 64 updates, and checkpoints 8/16/32/64.
- No observation noise, domain randomization, pushes, assistance, adaptive phase sampling, or reset-noise sweep.
- Retain only a checkpoint at least `116/99/67/49/24` componentwise.

---

### Task 1: Explicit frozen-recovery migration

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Test: `tests/test_shac_exact_resume.py`
- Test: `tests/test_shac_cagrad_integration.py`

**Interfaces:**
- Consumes: existing `resolve_residual_preview_adapter_resume_setting`, `FrozenPreviewResidualParams`, `initialize_residual_adapter_optimizer`, and `policy_anchor_penalty`.
- Produces: `train(..., allow_resume_actor_residual_preview_adapter_start: bool = False)` and an explicit migration path from a saved plain actor to a zero-effect frozen residual actor.

- [ ] **Step 1: Write failing resolver and integration tests**

Add tests proving saved `actor_residual_preview_adapter=False` rejects a requested start by default, accepts it only with explicit authority, and causes `train` to route through the migration condition while allowing anchor weight 1.0.

- [ ] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py`

Expected: failures for the missing authority argument and migration route.

- [ ] **Step 3: Implement the minimal migration authority**

Add the boolean train parameter and validation, permit only false-to-true residual starts when it is set, treat that transition as an adapter migration even when future-reference inputs already exist, and allow the existing policy anchor only for that frozen-parent resume treatment. Persist the authority and migration report in hparams/artifacts.

- [ ] **Step 4: Run GREEN and static checks**

Run the focused pytest command above, then:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check src/algorithms/shac/algorithm.py tests/test_shac_exact_resume.py tests/test_shac_cagrad_integration.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile src/algorithms/shac/algorithm.py
git diff --check
```

- [ ] **Step 5: Commit**

Commit only Task 1 files with message `feat: allow anchored residual recovery resumes`.

### Task 2: E023 history-faithful failure bank

**Files:**
- Create: `tools/build_g1_e023_carried_reset_bank.py`
- Create: `tests/test_g1_e023_carried_reset_bank.py`
- Reuse: `tools/build_g1_history_carried_reset_bank.py`

**Interfaces:**
- Consumes: shared `select_preterminal_indices`, `validate_history_bank`, `_collect_source`, atomic writers, E023 plain Flax checkpoint, and the evaluation environment/action boundary.
- Produces: `collect_e023_bank(checkpoint_path, reference_path, seed) -> (arrays, survival)` plus an atomic NPZ/JSON CLI bound to checkpoint/reference/code/model/controller/solver hashes.

- [ ] **Step 1: Write failing collector-contract tests**

Cover source phases `(0, 50)`, 24 rows per terminal rollout, exact `(10, frame_dim)` history, plain actor reconstruction, action-parity boundary, SHA mismatch rejection, and nonterminal/hard-limit fail-closed behavior.

- [ ] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_g1_e023_carried_reset_bank.py`

Expected: import failure because the E023 collector does not exist.

- [ ] **Step 3: Implement the minimal collector**

Construct the exact E023 evaluation environment from pinned hparams, load the plain `TrainState`, reconstruct the `512-256-128` LayerNorm actor, normalize with its saved normalizer, apply the training-identical unclipped action boundary, collect sources 0 and 50, validate, and atomically bind the resulting bank.

- [ ] **Step 4: Run GREEN and static checks**

Run the focused pytest, Ruff, py_compile, and `git diff --check` on Task 2 files.

- [ ] **Step 5: Commit**

Commit only Task 2 files with message `feat: collect E023 pre-failure recovery states`.

### Task 3: Guarded E026 continuation and selection

**Files:**
- Create: `tools/run_g1_e023_anchored_carried_recovery.py`
- Create: `tests/test_g1_e023_anchored_carried_recovery.py`
- Create: `docs/superpowers/results/2026-08-15-e023-anchored-carried-recovery.md`

**Interfaces:**
- Consumes: Task 1 resume authority, Task 2 bank, E023 checkpoint/hparams, existing training validator, phase-grid evaluator, and frozen-preview telemetry.
- Produces: fixed kwargs builder, provenance preflight, training-artifact validator, componentwise selector, and CLI writing preflight/training validation under the run root.

- [ ] **Step 1: Write failing runner tests**

Assert exact parent equality for all inherited scientific kwargs and exact deltas for residual start, carried bank/probability, anchor weight, total endpoint, checkpoint cadence, and explicit resume authorities. Test rejection of drift, incomplete or nonfinite archives, frozen parent/normalizer movement, invalid CAGrad, invalid bank hashes, and compensated phase regressions.

- [ ] **Step 2: Run RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_g1_e023_anchored_carried_recovery.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement the runner and selector**

Build 64 additional updates from global step 1,572,864 with checkpoint interval 98,304 transitions (8 H24 updates), carried probability 0.25, adapter hidden 256, Adam optimizer, and anchor weight 1.0. Validate exact archive steps, finite TrainState leaves, migration report, zero frozen drift, positive adapter gradient/update, complete finite clipped CAGrad, exact action-noise endpoint, zero bootstrap, and componentwise selection.

- [ ] **Step 4: Run GREEN and focused neighboring tests**

Run Task 3 tests plus residual adapter, carried environment, exact resume, and training validator tests; then Ruff, py_compile, and `git diff --check`.

- [ ] **Step 5: Commit**

Commit Task 3 files and approved spec/plan with message `feat: add anchored carried recovery experiment`.

### Task 4: Curate E025 and register/execute E026

**Files:**
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-025.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260815-026.yaml`
- Generate: canonical run artifacts and vault projection.

**Interfaces:**
- Consumes: verified E025 phase grids, E026 clean code commit, bank SHA, registry taxonomy, and `tools.runexp`.
- Produces: truthful E025 `noisy-rsi-insufficient` curation, approved E026 registry entry, dry-run proof, executed/evaluated E026, and dashboard media/diagnostics.

- [ ] **Step 1: Curate E025**

Write its final vectors `43/49/49/49/24`, `45/54/54/49/24`, `50/61/55/49/24`, and `63/99/62/49/24`; classify `noisy-rsi-insufficient`; retain E023; create deterministic selection/learning diagnostics and presentation media from canonical artifacts.

- [ ] **Step 2: Build and freeze the E023 bank**

Run the Task 2 CLI against the exact E023 checkpoint and verify its atomic JSON/NPZ hashes and 48 admitted rows.

- [ ] **Step 3: Register and validate E026**

Create one schema-v1 experiment with the approved causal change and outcomes from the design. Run:

```bash
uv run python tools/researchctl.py validate
uv run python -m tools.runexp E-20260815-026 --dry-run
```

Expected: registry validation and dry run exit 0 with the pinned clean commit and bank hash.

- [ ] **Step 4: Execute and monitor E026**

Run: `uv run python -m tools.runexp E-20260815-026`

Inspect the first completed-update gate, then continue the fixed run only while its registered stop rule permits.

- [ ] **Step 5: Evaluate and curate**

Evaluate updates 8/16/32/64 at phases `0/25/50/75/100`, apply the componentwise selector, render the selected clean and training-distribution episodes, generate diagnostics, update experiment/current state, rebuild the vault, and validate focused registry tests.

- [ ] **Step 6: Commit owned registry sources**

Commit only E025/E026 registry sources, current state, qualifying findings, and generator-owned tracked files; preserve user `.obsidian` and unrelated dirty files.
