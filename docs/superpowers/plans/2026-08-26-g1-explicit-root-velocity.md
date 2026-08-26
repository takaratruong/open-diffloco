# Explicit G1 Root-Velocity Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit pseudo-Huber pelvis linear/angular-velocity reward and test it in one frozen-E026 residual continuation.

**Architecture:** Extend the existing G1 tracking reward with one zero-default combined root-velocity component, thread one weight through environment, SHAC resume metadata, and evaluators, then reuse the frozen-E026 H24 continuation and CPU five-phase selector. No new optimizer or training architecture is introduced.

**Tech Stack:** Python 3.11, JAX, MJX/MuJoCo, Flax, pytest, Ruff, existing SHAC and research registry.

## Global Constraints

- Legacy `tracking_root_velocity_weight=0.0` must preserve reward values and behavior exactly.
- Treatment weight is exactly `1.0`; linear and angular halves use scales `1.0 m/s` and `pi rad/s`.
- Root is body slot zero and velocities remain in the existing world-frame convention.
- Freeze complete E026 actor, optimizer parent state, and normalizer; train only the new zero-effect residual.
- Use one seed, effective-512 H24, 32 updates, checkpoints at 8/16/24/32, solver `g1-4x5-fixed-scan`.
- No parameter sweep, acceleration reward, assistance, replay, reset, contact, action, observation, reference, or physics change.
- Retention requires componentwise preservation of `131/114/74/71/74` and a strict gain.

---

### Task 1: Reward and Environment Contract

**Files:**
- Modify: `src/envs/g1_tracking/reward.py`
- Modify: `src/envs/g1_tracking/environment.py`
- Modify: `tests/test_g1_tracking_environment.py`

**Interfaces:**
- Produces `root_velocity_tracking_reward(target_body_lin_vel, actual_body_lin_vel, target_body_ang_vel, actual_body_ang_vel)` and environment option `tracking_root_velocity_weight: float = 0.0`.

- [ ] Write failing tests proving the exact formula, zero-weight reward identity, slot-zero selection, finite metrics, and rejection of negative/nonfinite/bool weights.
- [ ] Run `JAX_ENABLE_X64=1 /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_tracking_environment.py -k root_velocity` and observe failure because the option/helper is absent.
- [ ] Implement the helper with `0.5` pseudo-Huber linear and angular terms, add it only for positive weight, and publish `rew_root_linear_velocity` and `rew_root_angular_velocity` metrics.
- [ ] Re-run the focused test and neighboring tracking-kernel tests; require green.
- [ ] Commit reward/environment/test files only.

### Task 2: SHAC Resume and Evaluation Plumbing

**Files:**
- Modify: `src/algorithms/shac/algorithm.py`
- Modify: `tools/evaluate_g1_tracking.py`
- Modify: `tools/evaluate_g1_flax_phase_grid.py`
- Modify: `tests/test_shac_exact_resume.py`
- Modify: `tests/test_g1_tracking_evaluator.py`
- Modify: `tests/test_evaluate_g1_flax_phase_grid.py`

**Interfaces:**
- Produces `resolve_tracking_root_velocity_resume_weight(..., requested, allow_change, is_resume)` and train flags `tracking_root_velocity_weight`, `allow_resume_tracking_root_velocity_change`.

- [ ] Write failing tests for fresh default, exact resume, missing metadata, unauthorized change, authorized start, hparams persistence, evaluator restoration, and invalid weights.
- [ ] Run the focused resume/evaluator tests and observe missing-interface failures.
- [ ] Implement minimal train/environment/hparams/evaluator threading. A real resume with missing hparams must fail closed.
- [ ] Re-run the focused tests and require green with legacy evaluator contracts unchanged.
- [ ] Commit only the SHAC/evaluator/test files.

### Task 3: Guarded Frozen-E026 Continuation

**Files:**
- Create: `tools/run_g1_root_velocity_continuation.py`
- Create: `tests/test_g1_root_velocity_continuation.py`

**Interfaces:**
- Produces exact checkpoint grid, one-delta kwargs builder, training validator, CPU selection classifier, and guarded CLI runner.

- [ ] Write failing tests for the exact 32-update grid, weight-one treatment, forbidden-delta equality, source hashes, frozen-tree checks, complete telemetry, and componentwise selector.
- [ ] Run the focused runner tests and observe import failure.
- [ ] Reuse `build_capture_point_kwargs` with capture disabled, then enable only the root-velocity weight and explicit resume authority. Validate exact E026/reference/model/controller/code provenance and all four finite checkpoints.
- [ ] Implement pure selection: retain only componentwise-safe strict improvements; otherwise distinguish redistribution from insufficiency.
- [ ] Run focused tests, Ruff, py_compile, and `git diff --check`; require green.
- [ ] Commit runner/tests only.

### Task 4: Register, Execute, Evaluate, and Curate

**Files:**
- Create in lab repo: `research/experiments/E-20260826-002.yaml`
- Modify after validated evidence: `research/state/current.yaml`
- Generated after curation: `docs/vault/Dashboard.md`, `docs/vault/experiments/E-20260826-002.md`

- [ ] Register one approved experiment with the design's causal delta, pinned clean code SHA, hashes, one GPU, seed zero, 90-minute budget, four checkpoints, CPU phase grid, plots, video, and montage.
- [ ] Run registry validation and `runexp --dry-run`, then execute the approved command on one available GPU.
- [ ] Validate training artifacts and evaluate every checkpoint on CPU at phases `0/25/50/75/100`.
- [ ] Apply only the preregistered outcome, update current state, rebuild/validate the vault, and publish MP4/PNG diagnostics to the dashboard.
- [ ] Run final focused tests and static checks, review the exact diff, and commit/push owned code and research records.
