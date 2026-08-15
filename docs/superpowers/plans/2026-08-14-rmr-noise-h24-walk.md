# RMR-Calibrated H24 Walking Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one provenance-bound fresh H24 SHAC treatment whose only scientific change from E022 is RMR-calibrated normalized action noise.

**Architecture:** Store the pinned RMR walking model-999 vector beside the existing action-noise provenance constants. Add one thin E022-derived runner that changes only the noise endpoints and validates exact training artifacts through the existing generalized validator.

**Tech Stack:** Python, JAX, MJX, pytest, Ruff, diffsim2real experiment registry.

## Global Constraints

- Preserve every E022 scientific setting except the three action-noise schedule fields.
- Start at scalar `1.0`; end at the exact 29-joint model-999 vector in actor order; schedule through step `1,572,864`.
- Use seed zero, GPU 4 or another idle single GPU, and no observation noise, random reset noise, pushes, assistance, terrain, or domain randomization.
- Evaluate only the preregistered updates 16, 32, 64, and 128 before selecting.

---

### Task 1: Pin the RMR walking noise vector and runner

**Files:**
- Modify: `src/core/rmr_action_noise.py`
- Create: `tools/run_g1_rmr_noise_h24_walk.py`
- Create: `tests/test_g1_rmr_noise_h24_walk.py`

**Interfaces:**
- Consumes: `build_fresh_full_action_h24_kwargs`, `validate_preflight`, `validate_training_artifacts`, `expected_checkpoint_steps`.
- Produces: `RMR_WALK_MODEL_999_ACTION_STD` and `build_rmr_noise_h24_kwargs(profile_name, reference_path, seed)`.

- [ ] **Step 1: Write failing tests** asserting the exact float32 vector, source hash metadata, exact three-field delta from E022, preserved H24/update/population/action/reset contracts, parser behavior, preflight behavior, and artifact-validator delegation.
- [ ] **Step 2: Run RED** with `PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/pytest -q tests/test_g1_rmr_noise_h24_walk.py`; expect import failure for the missing runner/constant.
- [ ] **Step 3: Implement minimally** by adding the pinned constant and a thin runner that derives E022 kwargs, changes only `action_noise_std_start`, `action_noise_std_end`, and the already-equal full-run `action_noise_schedule_steps`, then calls the existing training validator.
- [ ] **Step 4: Run GREEN and neighbors** with the focused test plus `tests/test_g1_fresh_full_action_h24_walk.py`, `tests/test_shac_action_noise.py`, and `tests/test_g1_fresh_ppo_action_contract_walk.py`.
- [ ] **Step 5: Run static checks** using Ruff, `py_compile`, and `git diff --check`, then commit only the owned files.

### Task 2: Register, execute, evaluate, and curate

**Files:**
- Create: `research/experiments/E-20260814-023.yaml` in the lab repository.
- Modify through generators: experiment result, vault dashboard, and local preview manifest.

**Interfaces:**
- Consumes: the committed runner SHA and E022 evidence.
- Produces: one registered run, four clean phase grids, selected clean/noisy videos, action diagnostics, and a final registry verdict.

- [ ] **Step 1: Register E023** with exact code/asset hashes, one GPU, seed zero, the three-field noise delta, required artifacts, and fail-closed outcome map.
- [ ] **Step 2: Validate and dry-run** with `uv run python tools/researchctl.py validate` and `uv run python -m tools.runexp E-20260814-023 --dry-run`.
- [ ] **Step 3: Launch once** and monitor the first checkpoint for finite occupied CAGrad and the expected current noise value before allowing the run to continue.
- [ ] **Step 4: Evaluate updates 16/32/64/128** on the exact five-phase grid and select only a checkpoint preserving E021 floors.
- [ ] **Step 5: Render and curate** the selected clean and exact training-noise episodes, learning/action diagnostics, registry result, and Obsidian dashboard media.

## Self-review

The plan covers every design requirement, introduces no new training abstraction, has no placeholders, and keeps vector provenance separate from experiment orchestration. The three scientific noise fields are explicit and all later evaluation uses existing tools.
