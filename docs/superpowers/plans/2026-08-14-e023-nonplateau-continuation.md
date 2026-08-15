# E023 Non-Plateau Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume E023 exactly for 128 additional H24 SHAC updates and emit fail-closed checkpoints for learning-curve evaluation.

**Architecture:** Add one experiment-specific thin runner derived from the tested E023 builder and shared training validator. Pin the immutable parent checkpoint/hparams and change only resume source, endpoint, and checkpoint cadence. Register and execute one experiment through `tools.runexp`.

**Tech Stack:** Python, JAX/MJX, pytest, Ruff, YAML experiment registry.

## Global Constraints

- Resume checkpoint step is exactly 1,572,864 with SHA-256 `2bbad61f735103c09dad11bcc701ac48fe1d41e4719b63437ea3b7a229645b9f`.
- Parent hparams SHA-256 is `a4435aebb4be1d3f539fb82634b47134424a57726fc11c4f0011821bc15ff650`.
- Add 128 updates at effective population 512 and H24: final step 3,145,728.
- Archive every 16 additional updates: 196,608 transitions.
- Preserve exact uniform RSI and the endpoint model-999 action-noise vector.

---

### Task 1: Exact continuation runner

**Files:**
- Create: `tools/run_g1_rmr_noise_h24_continuation.py`
- Test: `tests/test_g1_rmr_noise_h24_continuation.py`

**Interfaces:**
- Consumes: `build_rmr_noise_h24_kwargs`, E023 final checkpoint and hparams.
- Produces: `build_rmr_noise_h24_continuation_kwargs`, `expected_checkpoint_steps`, guarded CLI and validation report.

- [ ] **Step 1: Write failing tests** for exact step math, three-field continuation delta, pinned preflight hashes, and required parser arguments.
- [ ] **Step 2: Verify RED** with `PYTHONPATH=. /home/ubuntu/miniconda3/envs/rl/bin/python -m pytest -q tests/test_g1_rmr_noise_h24_continuation.py`.
- [ ] **Step 3: Implement the minimal runner** by deriving E023 kwargs, adding `resume_from`, `total_steps=3_145_728`, and `checkpoint_interval=196_608`, then reusing the shared artifact validator.
- [ ] **Step 4: Verify GREEN and neighbors** with the new test plus `tests/test_g1_rmr_noise_h24_walk.py`, `tests/test_g1_fresh_full_action_h24_walk.py`, and `tests/test_shac_action_noise.py` in the pinned `rl` environment; run Ruff, `py_compile`, and `git diff --check`.
- [ ] **Step 5: Commit** only the runner, tests, design, and plan.

### Task 2: Register and launch

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260814-024.yaml`

**Interfaces:**
- Consumes: the clean Task 1 commit and E023 evidence.
- Produces: one guarded continuation run with four decisive phase-grid evaluations and updated learning curves.

- [ ] **Step 1: Register** the exact command, hashes, GPU, 90-minute budget, outcome map, and required artifacts.
- [ ] **Step 2: Validate and dry-run** with `uv run python tools/researchctl.py validate` and `uv run python -m tools.runexp E-20260814-024 --dry-run`.
- [ ] **Step 3: Launch** with `uv run python -m tools.runexp E-20260814-024` and monitor immutable checkpoints.
- [ ] **Step 4: Evaluate** continuation updates 16/32/64/128 on the exact five-phase grid, rebuild the extended learning curves, render the selected rollout, curate the result, and rebuild the Obsidian vault.
