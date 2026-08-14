# Full-Scale Walking DiffSim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the competent E020 full-scale walking actor under the exact MJX contract and, if competent, apply a bounded zero-bootstrap CAGrad DiffSim continuation without the old half-scale action mismatch.

**Architecture:** Extend the existing replay-free RMR phase-grid evaluator and full-actor recovery builder with one explicit `reference_residual_scale` boundary. Preserve legacy half-scale defaults while requiring the new walking experiment to request full scale (`1.0`) explicitly. Register and execute the parent gate first; only a passing exact MJX parent authorizes the separate DiffSim continuation.

**Tech Stack:** Python, JAX/MJX, MuJoCo, pytest, YAML experiment registry.

## Global Constraints

- Work only in the existing linked research worktree.
- Preserve legacy evaluator and recovery defaults at `0.5`.
- E020 walking uses source joint order, unclipped actions, `q_target = q_reference + scale * action`, and `reference_residual_scale = 1.0`.
- Parent evaluation is replay-free at phases `0,25,50,75,100` on the 125-state named reference.
- DiffSim continuation uses the E020 actor, nominal physics, exact RSI, clean observations, fixed noise `0.05`, zero terminal bootstrap, two-shard five-bin CAGrad, H12, and no assistance/randomization/replay.
- Do not launch the continuation unless the exact MJX parent gate validates.

---

### Task 1: Explicit Full-Scale Phase-Grid Evaluation

**Files:**
- Modify: `tools/evaluate_g1_rmr_phase_grid.py`
- Modify: `tests/test_g1_rmr_full_actor_recovery_runner.py`

**Interfaces:**
- Consumes: `make_evaluation_env(...)` and the existing RMR checkpoint loader.
- Produces: CLI option `--reference-residual-scale {0.5,1.0}` and phase-grid JSON that records the selected scale.

- [x] **Step 1: Write failing tests**

Add tests asserting that parser/default behavior remains `0.5`, explicit `1.0` reaches `make_evaluation_env`, and the JSON evidence records the exact scale.

- [x] **Step 2: Verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_rmr_full_actor_recovery_runner.py -k full_scale_phase_grid`

Expected: FAIL because the evaluator has no full-scale argument/evidence.

- [x] **Step 3: Implement the minimal evaluator boundary**

Add a choices-constrained parser argument, pass it to `make_evaluation_env`, and persist it in the atomic result JSON. Do not change the default.

- [x] **Step 4: Verify GREEN and neighbors**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_rmr_full_actor_recovery_runner.py tests/test_g1_tracking_evaluator.py`

Expected: PASS.

### Task 2: Explicit Full-Scale DiffSim Recovery Builder

**Files:**
- Modify: `tools/run_g1_rmr_full_actor_recovery.py`
- Modify: `tests/test_g1_rmr_full_actor_recovery_runner.py`

**Interfaces:**
- Consumes: `build_rmr_full_actor_recovery_kwargs(...)`.
- Produces: optional keyword `reference_residual_scale: float = 0.5` and CLI `--reference-residual-scale {0.5,1.0}` forwarded to training.

- [x] **Step 1: Write failing tests**

Add tests proving the legacy builder remains half-scale and an explicit full-scale request returns `reference_residual_scale == 1.0` with the unchanged action-parity environment.

- [x] **Step 2: Verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_rmr_full_actor_recovery_runner.py -k full_scale_recovery`

Expected: FAIL because the builder/CLI has no explicit scale input.

- [x] **Step 3: Implement minimal forwarding**

Validate the scale is exactly `0.5` or `1.0`, override the canonical half-scale value, expose the CLI argument, and forward it from `main()`.

- [x] **Step 4: Verify GREEN and static checks**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_rmr_full_actor_recovery_runner.py tests/test_g1_rmr_full_actor_h24_recovery_runner.py tests/test_g1_tracking_evaluator.py`

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tools/evaluate_g1_rmr_phase_grid.py tools/run_g1_rmr_full_actor_recovery.py tests/test_g1_rmr_full_actor_recovery_runner.py`

Expected: PASS.

### Task 3: Register, Gate, and Execute the Walking Experiment

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260813-023.yaml`
- Modify after evidence: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260813-023.yaml`
- Create after parent pass: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260813-024.yaml`

**Interfaces:**
- Consumes: E020 `model_999.pt`, named walking reference, exact full-scale evaluator and recovery runner.
- Produces: an immutable exact-MJX parent verdict, followed only on pass by a bounded full-scale DiffSim recovery verdict.

- [ ] **Step 1: Commit the code boundary**

Commit only the evaluator, runner, tests, and this plan. Record the clean commit in experiment provenance.

- [ ] **Step 2: Close stale E022 truthfully**

Record the stopped two-iteration PPO result as `evaluated/fail`: online normalization reduced but did not repair the trust-region violation, and no checkpoint was produced.

- [ ] **Step 3: Register and validate E023**

Pin E020 checkpoint SHA `5db9d8371754a635d162c416e192b49ec2064d3133d20eea0df63463d1c8ae03`, reference SHA `b1197c389887055244f05000a2ebb9cb2748dea26de05bdc6850ed4089dcfdca`, full-scale action contract, solver, code commit, and phases `0,25,50,75,100`.

- [ ] **Step 4: Execute and curate E023**

Run `uv run python -m tools.runexp E-20260813-023`. Require all five suffixes to complete. Render selected rollout evidence only after the metrics validate.

- [ ] **Step 5: Register and execute E024 only if E023 passes**

Use the exact E020 actor and full-scale recovery runner for 16 H12 DiffSim updates. Select only a checkpoint that preserves every completed parent suffix and improves the minimum-first survival key; otherwise record a failure and move to the separately controlled assistance treatment.
